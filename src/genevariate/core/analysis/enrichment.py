"""
Enrichment analysis driven by GeneVariate sample-level Condition labels.

Two entry points:
  - run_enrichr(gene_list, gene_sets=...)       over-representation (ORA)
  - run_prerank_gsea(ranked_df, gene_sets=...)  GSEA on a ranked gene list

rank_genes_by_condition(expr_df, labels) produces a ranked list suitable for
feeding into run_prerank_gsea, using a simple mean-difference + t-statistic
(no external DE dependency required).

The canonical input DataFrame is GeneVariate's standard:
    GSM | series_id | GENE1 | GENE2 | ...  (rows = samples)
Sample labels come from the LLM condition curator (case vs control / cluster
names), passed as a dict {GSM -> label}.
"""

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
    """
    Benjamini-Hochberg FDR-adjusted p-values (q-values).

    NaN p-values are ignored in the ranking and returned as NaN. The result
    is the standard step-up BH adjustment with monotonicity enforced.
    """
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


def _expr_from_canonical(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split canonical (GSM|series_id|gene..) into (expr genes x samples, GSM index)."""
    meta_cols = [c for c in ("GSM", "series_id") if c in df.columns]
    if "GSM" not in df.columns:
        raise ValueError("Input must be in GeneVariate canonical format (GSM column required).")
    gsm = df["GSM"].astype(str).str.upper()
    gene_cols = [c for c in df.columns if c not in meta_cols]
    expr = df[gene_cols].T.copy()
    expr.columns = gsm.values
    return expr, gsm


def _empirical_bayes_variance(s2: np.ndarray, dfree: float,
                               trim: float = 0.05) -> Tuple[np.ndarray, float, float]:
    """
    limma-style empirical Bayes variance shrinkage (Smyth 2004).

    Given per-gene sample variances s2 with dfree degrees of freedom each,
    fit a scaled inverse chi-squared prior (s0², d0) via the method of
    moments on log s² (trimmed to handle outliers), then return posterior
    shrunken variances

        tilde_s² = (d0 * s0² + dfree * s²) / (d0 + dfree).

    Returns (shrunken_s2, d0, s0_squared).
    References: Smyth GK (2004), "Linear models and empirical Bayes
    methods for assessing differential expression in microarray
    experiments," Stat Appl Genet Mol Biol 3, Article 3.
    """
    s2 = np.asarray(s2, dtype=float)
    valid = np.isfinite(s2) & (s2 > 0)
    if valid.sum() < 30:
        return s2, 0.0, float(np.nanmedian(s2[valid]) if valid.any() else 1.0)
    z = np.log(s2[valid])
    # Trim tails to stabilise moments
    lo, hi = np.quantile(z, [trim, 1 - trim])
    core = z[(z >= lo) & (z <= hi)]
    if core.size < 10:
        core = z
    mean_z = float(np.mean(core))
    var_z  = float(np.var(core, ddof=1))

    # Inverse of trigamma, Newton iterations (Smyth 2004, Appx.)
    from scipy.special import digamma, polygamma
    target = var_z - polygamma(1, dfree / 2.0)
    if target <= 0 or not np.isfinite(target):
        d0 = 1e6  # strong shrinkage to global variance
    else:
        # Solve polygamma(1, d0/2) = target
        x = 0.5 + 1.0 / target
        for _ in range(30):
            psi1 = polygamma(1, x)
            psi2 = polygamma(2, x)
            delta = psi1 * (1 - psi1 / target) / psi2
            x = x + delta
            if abs(delta) < 1e-8 * x:
                break
        d0 = 2.0 * x
    s0_2 = float(np.exp(mean_z + digamma(dfree / 2.0) - np.log(dfree / 2.0)
                         - digamma(d0 / 2.0) + np.log(d0 / 2.0)))
    s0_2 = max(s0_2, 1e-12)

    out = np.array(s2, dtype=float)
    valid_mask = np.isfinite(out) & (out > 0)
    out[valid_mask] = (d0 * s0_2 + dfree * out[valid_mask]) / (d0 + dfree)
    # fall back to s0² for invalid positions
    out[~valid_mask] = s0_2
    return out, float(d0), s0_2


def rank_genes_by_condition(df: pd.DataFrame,
                            labels: Dict[str, str],
                            case_label: str,
                            control_label: str,
                            moderated: bool = False) -> pd.DataFrame:
    """
    Rank genes by case-vs-control mean difference, with optional limma-style
    empirical Bayes variance shrinkage.

    Parameters
    ----------
    moderated : bool
        If True, replace per-gene sample variances with empirical Bayes
        shrunken variances (Smyth 2004). This is the recommended option
        for microarray / log-transformed RNA-seq data with few replicates.
        For raw RNA-seq counts, prefer `pydeseq2` (see README).

    Returns a DataFrame indexed by gene with columns:
        mean_case, mean_control, logFC, t_stat, p_value, padj, rank
    `padj` is the Benjamini-Hochberg FDR-adjusted p-value. Ranking uses the
    (moderated) t-stat, which is monotonic in both magnitude and direction —
    suitable for gseapy prerank.
    """
    expr, gsm = _expr_from_canonical(df)
    labels = {str(k).upper(): v for k, v in labels.items()}

    case_samples = [g for g in gsm if labels.get(g) == case_label]
    ctrl_samples = [g for g in gsm if labels.get(g) == control_label]
    if len(case_samples) < 2 or len(ctrl_samples) < 2:
        raise ValueError(
            f"Need at least 2 samples per group — got "
            f"case={len(case_samples)}, control={len(ctrl_samples)}"
        )

    case = expr[case_samples].astype(float)
    ctrl = expr[ctrl_samples].astype(float)

    nc, nk = case.shape[1], ctrl.shape[1]
    mc = case.mean(axis=1)
    mk = ctrl.mean(axis=1)
    logfc = mc - mk

    # Pooled variance with (nc+nk-2) df — the natural input for limma EB
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
        if np.isscalar(df_total) or isinstance(df_total, (int, float)):
            p = 2.0 * (1.0 - t_dist.cdf(np.abs(t.fillna(0.0).values), df_total))
        else:
            p = 2.0 * (1.0 - t_dist.cdf(np.abs(t.fillna(0.0).values),
                                         np.asarray(df_total).clip(min=1.0)))
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
    out["rank"] = out["t_stat"].fillna(0.0)
    out = out.sort_values("rank", ascending=False)
    return out


# -----------------------------------------------------------------
# Over-representation (Enrichr)
# -----------------------------------------------------------------
def run_enrichr(gene_list: Sequence[str],
                gene_sets: Sequence[str] = DEFAULT_LIBRARIES,
                organism: str = "human",
                outdir: Optional[str] = None,
                cutoff: float = 0.05) -> pd.DataFrame:
    """
    Run Enrichr over-representation on `gene_list`. Returns a DataFrame
    of significant terms (Adjusted P-value < cutoff) across all libraries.
    """
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
    """
    GSEA prerank on the output of `rank_genes_by_condition`.
    `ranked` must have a 'rank' column and a gene index.
    Returns merged gseapy results for each library.
    """
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
    """Render a small DataFrame as a GitHub-flavored markdown table.
    Falls back to a hand-rolled implementation when `tabulate` isn't installed."""
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
                               out_path: Optional[str] = None) -> str:
    """
    Build a short markdown report summarising ORA + GSEA results.
    If out_path is given, the report is written to disk and the path returned;
    otherwise the markdown string itself is returned.
    """
    lines: List[str] = []
    lines.append(f"# Enrichment report — {comparison}\n")
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
