"""Enrichment analysis driven by GeneVariate sample-level Condition labels."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    import gseapy
    _HAS_GSEAPY = True
except Exception:
    gseapy = None
    _HAS_GSEAPY = False


DEFAULT_LIBRARIES: Tuple[str, ...] = (
    "GO_Biological_Process_2023",
    "KEGG_2021_Human",
    "Reactome_2022",
    "MSigDB_Hallmark_2020",
)


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------
def benjamini_hochberg(pvalues: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values (q-values)."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    finite = np.isfinite(p)
    m = int(finite.sum())
    if m == 0:
        return out
    idx = np.where(finite)[0]
    pv = p[idx]
    order = np.argsort(pv)
    ranked = pv[order]
    n = m
    adj = ranked * n / (np.arange(1, n + 1))
    # enforce monotonicity from the largest p-value downward
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    q = np.empty(n, dtype=float)
    q[order] = adj
    out[idx] = q
    return out


_NON_GENE_COLS = frozenset({"GSM", "series_id", "gse", "gpl", "n_cells"})


def _expr_from_canonical(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split canonical (GSM|series_id|gene..) into (expr genes x samples, GSM index)."""
    if "GSM" not in df.columns:
        raise ValueError("Input must be in GeneVariate canonical format (GSM column required).")
    gsm = df["GSM"].astype(str).str.upper()
    gene_cols = [c for c in df.columns
                 if c not in _NON_GENE_COLS
                 and pd.api.types.is_numeric_dtype(df[c])]
    expr = df[gene_cols].T.copy()
    expr.columns = gsm.values
    return expr, gsm


def _theo_logf_winsor_moments(dfree: float, d0: float,
                              p: float) -> Tuple[float, float]:
    """Returns (E[W], Var[W]) for z = log F(dfree, d0) winsorized at tail proportion ``p``."""
    from scipy.stats import f as _f
    from scipy.integrate import quad
    fr = _f(dfree, d0)
    zlo = float(np.log(fr.ppf(p)))
    zhi = float(np.log(fr.ppf(1.0 - p)))
    i1 = quad(lambda u: np.log(fr.ppf(u)), p, 1.0 - p, limit=100)[0]
    i2 = quad(lambda u: np.log(fr.ppf(u)) ** 2, p, 1.0 - p, limit=100)[0]
    ew = p * zlo + p * zhi + i1
    ew2 = p * zlo ** 2 + p * zhi ** 2 + i2
    return ew, ew2 - ew ** 2


def _empirical_bayes_variance(s2: np.ndarray, dfree: float,
                               winsor: float = 0.05,
                               d0_cap: float = 1e6
                               ) -> Tuple[np.ndarray, float, float]:
    """limma-style empirical Bayes variance shrinkage; returns (shrunken_s2, d0, s0_squared)."""
    s2 = np.asarray(s2, dtype=float)
    valid = np.isfinite(s2) & (s2 > 0)
    if valid.sum() < 30:
        s0_2 = float(np.nanmedian(s2[valid])) if valid.any() else 1.0
        d0 = 1.0
        with np.errstate(invalid="ignore"):
            out = (d0 * s0_2 + dfree * s2) / (d0 + dfree)
        out = np.where(np.isfinite(s2), out, np.nan)
        return out, d0, s0_2

    z = np.log(s2[valid])
    zlo, zhi = np.quantile(z, [winsor, 1.0 - winsor])
    zw = np.clip(z, zlo, zhi)
    wv_obs = float(np.var(zw, ddof=1))
    wm_obs = float(np.mean(zw))

    from scipy.optimize import brentq
    _, v_floor = _theo_logf_winsor_moments(dfree, d0_cap, winsor)
    if wv_obs <= v_floor:
        d0 = d0_cap
    else:
        def _gap(log_d0: float) -> float:
            _, v = _theo_logf_winsor_moments(dfree, float(np.exp(log_d0)),
                                             winsor)
            return v - wv_obs
        lo, hi = np.log(0.05), np.log(d0_cap)
        d0 = 0.05 if _gap(lo) < 0 else float(np.exp(brentq(_gap, lo, hi,
                                                           xtol=1e-6)))

    ew, _ = _theo_logf_winsor_moments(dfree, d0, winsor)
    s0_2 = max(float(np.exp(wm_obs - ew)), 1e-12)

    out = np.array(s2, dtype=float)
    valid_mask = np.isfinite(out) & (out > 0)
    out[valid_mask] = (d0 * s0_2 + dfree * out[valid_mask]) / (d0 + dfree)
    # fall back to s0² for invalid positions
    out[~valid_mask] = s0_2
    return out, float(d0), s0_2


def _study_map(df: pd.DataFrame) -> Optional[pd.Series]:
    """GSM -> study for a canonical frame, or None if it carries no study column."""
    for col in ("series_id", "gse"):
        if col not in df.columns:
            continue
        s = pd.Series(df[col].astype(str).str.strip().values,
                      index=df["GSM"].astype(str).str.upper().values)
        s = s[~s.isin(("", "nan", "None"))]
        s = s[~s.index.duplicated(keep="first")]
        if len(s):
            return s
    return None


def _collapse_by_study(expr: pd.DataFrame, samples: Sequence[str],
                       study_of: pd.Series) -> pd.DataFrame:
    """One column per study: the mean of that study's samples in this group."""
    keep = [s for s in samples if s in study_of.index]
    if not keep:
        return pd.DataFrame(index=expr.index)
    sub = expr[keep].astype(float)
    return sub.T.groupby(study_of.reindex(keep).values).mean().T


def rank_genes_by_condition(df: pd.DataFrame,
                            labels: Dict[str, str],
                            case_label: str,
                            control_label: str,
                            moderated: bool = False) -> pd.DataFrame:
    """Indexed by gene, columns mean_case, mean_control, logFC, t_stat, p_value, padj, rank."""
    expr, gsm = _expr_from_canonical(df)
    labels = {str(k).upper(): v for k, v in labels.items()}

    case_samples = [g for g in gsm if labels.get(g) == case_label]
    ctrl_samples = [g for g in gsm if labels.get(g) == control_label]
    if len(case_samples) < 2 or len(ctrl_samples) < 2:
        raise ValueError(
            f"Need at least 2 samples per group - got "
            f"case={len(case_samples)}, control={len(ctrl_samples)}"
        )

    study_of = _study_map(df)
    unit, note = "samples", ""
    case = ctrl = None
    if study_of is not None:
        c = _collapse_by_study(expr, case_samples, study_of)
        k = _collapse_by_study(expr, ctrl_samples, study_of)
        if c.shape[1] >= 2 and k.shape[1] >= 2:
            case, ctrl, unit = c, k, "studies"
            shared = sorted(set(c.columns) & set(k.columns))
            note = (
                f"Tested across studies: {c.shape[1]} case studies collapsed "
                f"from {len(case_samples)} samples, {k.shape[1]} control "
                f"studies from {len(ctrl_samples)}. "
                + (f"{len(shared)} studies contribute to both groups. "
                   if shared else
                   "No study contributes to both groups, so every difference "
                   "here is also a between-study difference. ")
                + "Samples within a series share protocol and batch, so testing "
                  "them as independent would inflate the degrees of freedom.")
        else:
            note = (
                f"Only {c.shape[1]} case / {k.shape[1]} control studies, too "
                f"few to test across studies, so samples were pooled. These "
                f"p-values assume an independence the design does not provide.")
    if case is None:
        case = expr[case_samples].astype(float)
        ctrl = expr[ctrl_samples].astype(float)
        if not note:
            note = ("No series_id/gse column on this frame, so samples were "
                    "pooled and assumed independent.")

    nc, nk = case.shape[1], ctrl.shape[1]
    mc = case.mean(axis=1)
    mk = ctrl.mean(axis=1)
    logfc = mc - mk

    # Pooled variance with (nc+nk-2) df - the natural input for limma EB
    var_c = case.var(axis=1, ddof=1)
    var_k = ctrl.var(axis=1, ddof=1)
    dfree = nc + nk - 2
    pooled = ((nc - 1) * var_c + (nk - 1) * var_k) / dfree

    if moderated:
        shrunken, d0, s0_2 = _empirical_bayes_variance(pooled.values, dfree)
        sigma2 = pd.Series(shrunken, index=pooled.index)
        se = np.sqrt(sigma2 * (1.0 / nc + 1.0 / nk))
        df_total = dfree + d0  # posterior df
    else:
        # Welch-style separate-variance
        vc = var_c.replace(0, np.nan)
        vk = var_k.replace(0, np.nan)
        se = np.sqrt(vc / nc + vk / nk)
        # Welch-Satterthwaite df (per-gene)
        df_total = ((vc / nc + vk / nk) ** 2) / (
            (vc / nc) ** 2 / max(nc - 1, 1) + (vk / nk) ** 2 / max(nk - 1, 1)
        )

    t = (logfc / se)
    # p-value (two-sided). For moderated t, use scalar df; for Welch, use per-gene.
    try:
        from scipy.stats import t as t_dist
        tv = np.abs(t.values.astype(float))
        if np.isscalar(df_total) or isinstance(df_total, (int, float)):
            p = 2.0 * t_dist.sf(tv, df_total)
        else:
            p = 2.0 * t_dist.sf(tv, np.asarray(df_total).clip(min=1.0))
    except Exception:
        p = np.full_like(t.values, np.nan, dtype=float)

    out = pd.DataFrame({
        "mean_case": mc,
        "mean_control": mk,
        "logFC": logfc,
        "t_stat": t,
        "p_value": pd.Series(p, index=t.index),
    })
    out["padj"] = benjamini_hochberg(out["p_value"].values)
    out["rank"] = out["t_stat"].astype(float)
    out = out.sort_values("rank", ascending=False, na_position="last")
    out.attrs["unit"] = unit
    out.attrs["n_case"] = int(nc)
    out.attrs["n_control"] = int(nk)
    out.attrs["note"] = note
    return out


# -----------------------------------------------------------------
# Over-representation (Enrichr)
# -----------------------------------------------------------------
def run_enrichr(gene_list: Sequence[str],
                gene_sets: Sequence[str] = DEFAULT_LIBRARIES,
                organism: str = "human",
                outdir: Optional[str] = None,
                cutoff: float = 0.05) -> pd.DataFrame:
    """Significant terms (Adjusted P-value < cutoff) across all libraries."""
    if not _HAS_GSEAPY:
        raise RuntimeError("gseapy is not installed. `pip install gseapy`")
    gene_list = [str(g).strip() for g in gene_list if str(g).strip()]
    if not gene_list:
        raise ValueError("gene_list is empty")

    enr = gseapy.enrichr(
        gene_list=list(gene_list),
        gene_sets=list(gene_sets),
        organism=organism,
        outdir=outdir,
        cutoff=cutoff,
        no_plot=True,
    )
    res = enr.results if enr is not None else pd.DataFrame()
    if res is None or res.empty:
        return pd.DataFrame()
    sig = res[res["Adjusted P-value"] < cutoff].copy()
    sig = sig.sort_values("Adjusted P-value").reset_index(drop=True)
    return sig


# -----------------------------------------------------------------
# GSEA prerank
# -----------------------------------------------------------------
def run_prerank_gsea(ranked: pd.DataFrame,
                     gene_sets: Sequence[str] = DEFAULT_LIBRARIES,
                     outdir: Optional[str] = None,
                     permutation_num: int = 1000,
                     seed: int = 42) -> pd.DataFrame:
    """GSEA prerank on the output of `rank_genes_by_condition`, merged across libraries."""
    if not _HAS_GSEAPY:
        raise RuntimeError("gseapy is not installed. `pip install gseapy`")
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
def _df_to_md(frame: pd.DataFrame) -> str:
    """Render a small DataFrame as a GitHub-flavored markdown table."""
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        pass
    cols = [str(c) for c in frame.columns]
    def _fmt(x):
        if isinstance(x, float):
            return f"{x:.4g}"
        return str(x)
    rows = [[_fmt(v) for v in row] for row in frame.itertuples(index=False, name=None)]
    head = "| " + " | ".join(cols) + " |"
    sep  = "|" + "|".join("---" for _ in cols) + "|"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join([head, sep, body])


def enrichment_report_markdown(ora: pd.DataFrame,
                               gsea: pd.DataFrame,
                               comparison: str,
                               top_n: int = 15,
                               out_path: Optional[str] = None,
                               ranked: Optional[pd.DataFrame] = None) -> str:
    """ORA + GSEA markdown report; returns ``out_path`` when given, else the markdown."""
    lines: List[str] = []
    lines.append(f"# Enrichment report - {comparison}\n")
    if ranked is not None and ranked.attrs.get("note"):
        lines.append("## What was compared")
        lines.append(f"- **Unit of replication**: {ranked.attrs.get('unit')} "
                     f"(n_case={ranked.attrs.get('n_case')}, "
                     f"n_control={ranked.attrs.get('n_control')})")
        lines.append(f"- {ranked.attrs['note']}\n")
    lines.append("## Over-representation (Enrichr)")
    if ora is None or ora.empty:
        lines.append("_No significant terms._\n")
    else:
        cols = [c for c in ("Gene_set", "Term", "Adjusted P-value",
                            "Combined Score", "Genes") if c in ora.columns]
        lines.append(_df_to_md(ora[cols].head(top_n)))
        lines.append("")

    lines.append("## GSEA prerank")
    if gsea is None or gsea.empty:
        lines.append("_No results._\n")
    else:
        cols = [c for c in ("library", "Term", "NES", "FDR q-val", "NOM p-val",
                            "Lead_genes") if c in gsea.columns]
        available = [c for c in cols if c in gsea.columns]
        sort_col = "FDR q-val" if "FDR q-val" in gsea.columns else available[0]
        lines.append(_df_to_md(gsea[available].sort_values(sort_col).head(top_n)))
        lines.append("")

    md = "\n".join(lines)
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(md)
        return out_path
    return md
