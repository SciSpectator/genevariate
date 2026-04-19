"""
ΔVariance enrichment — novel addition to the GeneVariate analysis suite.

Standard GSEA / ORA ranks genes by a *mean-shift* statistic (t, logFC).
This module ranks genes by how their *distribution* changes between
conditions, then feeds that ranking into the same GSEA prerank machinery.

Primary statistic (recommended):

    logvar_z        Formally directional log-variance z-test.
                    For each gene, the log of the sample variance is
                    asymptotically normal: log(s²) ~ N(log(σ²), 2/(n−1)).
                    (Bartlett 1937; Cochran 1941; Box & Hill 1974.)
                    Therefore
                        z = (log s²_case − log s²_ctrl) /
                            sqrt( 2/(n_c−1) + 2/(n_k−1) )
                    is a valid directional (signed) test for scale
                    differences. Unlike Levene/Brown-Forsythe this is
                    natively two-sided AND directional, making it a
                    legitimate GSEA prerank statistic.

Auxiliary statistics (non-directional by construction, signed post-hoc;
retained for comparison / sensitivity analysis):

    levene          Levene's test for equality of variances (robust; W ≥ 0).
                    Signed by sign(var_case − var_ctrl). Interpretation:
                    "how much evidence for any scale difference, oriented
                    by which group is more variable."
    bf              Brown-Forsythe variant of Levene (center=median), signed.
    ks              Kolmogorov-Smirnov two-sample D (0 ≤ D ≤ 1), signed.
                    Conflates mean- and scale-shift.
    wasserstein     Wasserstein-1 distance (≥ 0), signed.
    logvar_ratio    log2(s²_case / s²_ctrl) — directional, no standard error.

Default: "logvar_z". The earlier signed-Levene/KS variants are kept
behind opt-in method= flags so existing pipelines still run, but new
analyses should use the directional log-variance z.

References:
  * Bartlett MS (1937), Proc R Soc Lond A 160:268–282.
  * Cochran WG (1941), Ann Math Stat 12:335–345.
  * Levene H (1960), in *Contributions to Probability and Statistics*.
  * Brown MB, Forsythe AB (1974), JASA 69:364–367.
  * Korthauer K et al. (2016), Genome Biol 17:222 (differential distribution
    in scRNA-seq).

Typical comparisons where ΔVariance matters:
  * tumour heterogeneity vs. normal tissue (variance explodes in tumours)
  * aging (expression variance increases with age even when means are stable)
  * drug-resistant populations (bimodal / bet-hedging phenotypes)
  * single-cell / bulk hybrids where bimodal genes escape mean-based DE
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

try:
    import gseapy
    _HAS_GSEAPY = True
except Exception:
    gseapy = None
    _HAS_GSEAPY = False

from .enrichment import (
    DEFAULT_LIBRARIES, _df_to_md, _expr_from_canonical,
)


SUPPORTED_METHODS: Tuple[str, ...] = (
    "logvar_z",         # recommended: directional log-variance z-test
    "bf",               # Brown-Forsythe (center=median), signed
    "levene",           # Levene (center=mean),          signed (legacy)
    "ks",               # Kolmogorov-Smirnov D,          signed (legacy)
    "wasserstein",      # Wasserstein-1 distance,        signed (legacy)
    "logvar_ratio",     # log2(var_case / var_ctrl), natively directional
)

RECOMMENDED_METHOD = "logvar_z"


# -----------------------------------------------------------------
# Per-gene variability statistics
# -----------------------------------------------------------------
def _logvar_z_stat(case: np.ndarray, ctrl: np.ndarray) -> Tuple[float, float]:
    """
    Directional log-variance z-test.

    Under the null σ²_case = σ²_ctrl and asymptotic normality of log s²,
        log(s²) ~ N(log σ², 2 / (n-1))
    (Bartlett 1937; Cochran 1941). The two-sample contrast

        z = (log s²_case − log s²_ctrl) / sqrt(2/(n_c − 1) + 2/(n_k − 1))

    is a valid signed test statistic for scale differences.
    Returns (z, two-sided p-value).
    """
    nc, nk = len(case), len(ctrl)
    if nc < 3 or nk < 3:
        return np.nan, np.nan
    vc = float(np.nanvar(case, ddof=1))
    vk = float(np.nanvar(ctrl, ddof=1))
    if vc <= 0 or vk <= 0 or not np.isfinite(vc) or not np.isfinite(vk):
        return np.nan, np.nan
    se = np.sqrt(2.0 / (nc - 1) + 2.0 / (nk - 1))
    z = (np.log(vc) - np.log(vk)) / se
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return float(z), float(p)


def _levene_stat(case: np.ndarray, ctrl: np.ndarray,
                 center: str = "mean") -> Tuple[float, float]:
    if len(case) < 2 or len(ctrl) < 2:
        return np.nan, np.nan
    try:
        W, p = stats.levene(case, ctrl, center=center)
        return float(W), float(p)
    except Exception:
        return np.nan, np.nan


def _ks_stat(case: np.ndarray, ctrl: np.ndarray) -> Tuple[float, float]:
    if len(case) < 2 or len(ctrl) < 2:
        return np.nan, np.nan
    try:
        D, p = stats.ks_2samp(case, ctrl, alternative="two-sided", mode="asymp")
        return float(D), float(p)
    except Exception:
        return np.nan, np.nan


def _wasserstein(case: np.ndarray, ctrl: np.ndarray) -> float:
    if len(case) < 2 or len(ctrl) < 2:
        return np.nan
    try:
        return float(stats.wasserstein_distance(case, ctrl))
    except Exception:
        return np.nan


def _logvar_ratio(case: np.ndarray, ctrl: np.ndarray) -> float:
    vc = np.nanvar(case, ddof=1)
    vk = np.nanvar(ctrl, ddof=1)
    if vc <= 0 or vk <= 0 or not np.isfinite(vc) or not np.isfinite(vk):
        return np.nan
    return float(np.log2(vc / vk))


# -----------------------------------------------------------------
# rank_genes_by_variability
# -----------------------------------------------------------------
def rank_genes_by_variability(df: pd.DataFrame,
                              labels: Dict[str, str],
                              case_label: str,
                              control_label: str,
                              method: str = RECOMMENDED_METHOD) -> pd.DataFrame:
    """
    Rank genes by differential variability between two condition groups.

    method (recommended):
      'logvar_z'       Directional log-variance z-test (Bartlett/Cochran
                       asymptotic normality). The statistic is natively
                       signed; p-value is two-sided normal. This is the
                       recommended default for GSEA prerank.

    method (auxiliary — signed post-hoc, kept for sensitivity analysis):
      'bf'             Brown-Forsythe W (Levene, center=median), signed by
                       sign(var_case − var_ctrl).
      'levene'         Levene W (center=mean), signed. Legacy.
      'ks'             KS two-sample D, signed. Legacy.
      'wasserstein'    Wasserstein-1 distance, signed. Legacy.
      'logvar_ratio'   log2(var_case / var_ctrl). Directional but no SE.

    Returns DataFrame indexed by gene with columns:
        var_case, var_control, mean_case, mean_control, delta_var,
        stat, p_value (may be NaN), rank
    Ranking is descending in 'rank'; suitable for gseapy.prerank.
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"method must be one of {SUPPORTED_METHODS}, got {method!r}")

    expr, gsm = _expr_from_canonical(df)
    labels = {str(k).upper(): v for k, v in labels.items()}

    case_samples = [g for g in gsm if labels.get(g) == case_label]
    ctrl_samples = [g for g in gsm if labels.get(g) == control_label]
    if len(case_samples) < 3 or len(ctrl_samples) < 3:
        raise ValueError(
            f"Variability tests need ≥3 samples per group — got "
            f"case={len(case_samples)}, control={len(ctrl_samples)}"
        )

    case = expr[case_samples].astype(float)
    ctrl = expr[ctrl_samples].astype(float)

    mean_c = case.mean(axis=1)
    mean_k = ctrl.mean(axis=1)
    var_c  = case.var(axis=1, ddof=1)
    var_k  = ctrl.var(axis=1, ddof=1)
    delta  = var_c - var_k
    sign   = np.sign(delta).replace(0, 1.0)

    stat_vals: List[float] = []
    p_vals: List[float] = []

    for gene in expr.index:
        c = case.loc[gene].dropna().to_numpy()
        k = ctrl.loc[gene].dropna().to_numpy()

        if method == "logvar_z":
            s, p = _logvar_z_stat(c, k)
        elif method == "bf":
            s, p = _levene_stat(c, k, center="median")
        elif method == "levene":
            s, p = _levene_stat(c, k, center="mean")
        elif method == "ks":
            s, p = _ks_stat(c, k)
        elif method == "wasserstein":
            s = _wasserstein(c, k); p = np.nan
        elif method == "logvar_ratio":
            s = _logvar_ratio(c, k); p = np.nan
        else:  # pragma: no cover — guarded above
            s, p = np.nan, np.nan
        stat_vals.append(s); p_vals.append(p)

    stat_series = pd.Series(stat_vals, index=expr.index, dtype=float)
    p_series    = pd.Series(p_vals,    index=expr.index, dtype=float)

    out = pd.DataFrame({
        "var_case":    var_c,
        "var_control": var_k,
        "mean_case":   mean_c,
        "mean_control": mean_k,
        "delta_var":   delta,
        "stat":        stat_series,
        "p_value":     p_series,
    })

    # Methods that produce an inherently signed statistic:
    natively_signed = {"logvar_z", "logvar_ratio"}
    if method in natively_signed:
        out["rank"] = out["stat"].fillna(0.0)
    else:
        # non-negative statistic — sign post-hoc by variance delta
        out["rank"] = (out["stat"].fillna(0.0) * sign).astype(float)

    out = out.sort_values("rank", ascending=False)
    return out


# -----------------------------------------------------------------
# Variability prerank GSEA
# -----------------------------------------------------------------
def run_variability_gsea(ranked: pd.DataFrame,
                         gene_sets: Sequence[str] = DEFAULT_LIBRARIES,
                         outdir: Optional[str] = None,
                         permutation_num: int = 1000,
                         seed: int = 42) -> pd.DataFrame:
    """
    GSEA prerank driven by the variability rank. Same signature as
    run_prerank_gsea so the GUI can swap one for the other.
    """
    if not _HAS_GSEAPY:
        raise RuntimeError("gseapy is not installed.")
    if "rank" not in ranked.columns:
        raise ValueError("ranked DataFrame must contain a 'rank' column")

    rnk = ranked["rank"].dropna().sort_values(ascending=False)
    rnk.index = rnk.index.astype(str).str.upper()
    rnk = rnk[~rnk.index.duplicated(keep="first")]

    frames: List[pd.DataFrame] = []
    for lib in gene_sets:
        try:
            res = gseapy.prerank(
                rnk=rnk.reset_index().rename(columns={"index": "gene", "rank": "score"}),
                gene_sets=lib,
                outdir=outdir,
                permutation_num=permutation_num,
                seed=seed,
                no_plot=True,
            )
            if res is not None and res.res2d is not None:
                d = res.res2d.copy()
                d["library"] = lib
                frames.append(d)
        except Exception as e:
            frames.append(pd.DataFrame([{"library": lib, "error": str(e)}]))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


# -----------------------------------------------------------------
# Markdown report
# -----------------------------------------------------------------
def variability_report_markdown(ranked: pd.DataFrame,
                                gsea: pd.DataFrame,
                                comparison: str,
                                method: str,
                                top_n: int = 15,
                                out_path: Optional[str] = None) -> str:
    lines: List[str] = []
    lines.append(f"# ΔVariance Enrichment report — {comparison}\n")
    lines.append(f"**Method**: `{method}`  (signed variability statistic)\n")

    lines.append("## Top genes by variability (rank)")
    if ranked is None or ranked.empty:
        lines.append("_No genes ranked._\n")
    else:
        cols = [c for c in ("var_case", "var_control", "delta_var",
                            "stat", "p_value", "rank") if c in ranked.columns]
        top = ranked.head(top_n).copy()
        top.insert(0, "gene", top.index)
        lines.append(_df_to_md(top[["gene"] + cols]))
        lines.append("")

    lines.append("## Pathways by ΔVariance GSEA")
    if gsea is None or gsea.empty:
        lines.append("_No results._\n")
    else:
        cols = [c for c in ("library", "Term", "NES", "FDR q-val", "NOM p-val",
                            "Lead_genes") if c in gsea.columns]
        sort_col = "NES" if "NES" in gsea.columns else cols[0]
        top = gsea[cols].sort_values(sort_col, ascending=False).head(top_n)
        lines.append(_df_to_md(top))
        lines.append("")

    md = "\n".join(lines)
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(md)
        return out_path
    return md
