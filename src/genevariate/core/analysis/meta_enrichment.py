"""
Cross-platform meta-enrichment — combine gene ranks from multiple GPL
platforms BEFORE running enrichment. Pathway calls then survive platform
batch effects because the combined rank is driven by consistent signal
across platforms, not a single noisy GPL.

Two combination statistics are implemented:

    rank_product      geometric mean of per-platform ranks (Breitling et al.
                      2004); robust, non-parametric.
    stouffer          Stouffer's weighted z combination of per-platform
                      signed t-statistics; assumes approximate normality
                      but preserves direction.

Both accept a dict of {platform_name -> ranked_genes_dataframe} where each
dataframe is the output of rank_genes_by_condition or
rank_genes_by_variability (has a 'rank' column, gene-indexed).

Typical use:
    per_plat = {"GPL570": rank_genes_by_condition(df570, ...),
                "GPL96":  rank_genes_by_condition(df96,  ...),
                "GPL13534": rank_genes_by_variability(dfm, ..., method="ks")}
    combined = combine_ranks(per_plat, method="stouffer")
    gsea     = run_meta_enrichment_gsea(combined, gene_sets=[...])
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

try:
    import gseapy
    _HAS_GSEAPY = True
except Exception:
    gseapy = None
    _HAS_GSEAPY = False

from .enrichment import DEFAULT_LIBRARIES, _df_to_md


# -----------------------------------------------------------------
# Per-platform rank normalisation
# -----------------------------------------------------------------
def _to_signed_fractional_rank(ranked: pd.DataFrame,
                                rank_col: str = "rank") -> pd.Series:
    """
    Convert the 'rank' column into a signed fractional rank in [-1, +1]:
        +1 = most upregulated gene
        -1 = most downregulated gene
    Tie handling: average. Genes with NaN rank are dropped.
    """
    if rank_col not in ranked.columns:
        raise ValueError(f"rank dataframe missing '{rank_col}' column")
    r = ranked[rank_col].dropna()
    n = len(r)
    if n == 0:
        return pd.Series(dtype=float)
    # rank descending so highest value → 1
    frac = r.rank(method="average", ascending=True) / n       # in (0, 1]
    frac = 2.0 * frac - 1.0                                     # in (-1, 1]
    frac.index = frac.index.astype(str).str.upper()
    return frac[~frac.index.duplicated(keep="first")]


# -----------------------------------------------------------------
# Rank-product combination
# -----------------------------------------------------------------
def _rank_product(per_platform: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Breitling rank-product: geometric mean of the *absolute* signed rank
    across platforms (small = consistently extreme). Direction is inferred
    from the sign of the mean signed-rank.
    """
    series_list = {name: _to_signed_fractional_rank(r)
                   for name, r in per_platform.items()}
    all_genes = sorted(set().union(*[s.index for s in series_list.values()]))
    if not all_genes:
        return pd.DataFrame()

    frame = pd.DataFrame({name: s.reindex(all_genes)
                          for name, s in series_list.items()})
    signed_mean = frame.mean(axis=1)
    # abs rank in [0, 1]; tiny eps to avoid log(0)
    absrank = frame.abs()
    geom = np.exp(np.nanmean(np.log(absrank.replace(0, np.nan) + 1e-6), axis=1))
    geom = pd.Series(geom, index=frame.index)
    direction = np.sign(signed_mean).replace(0, 1.0)
    rank = geom * direction                           # keep sign so GSEA prerank works

    out = pd.DataFrame({
        "rank_product":  geom,
        "signed_mean":   signed_mean,
        "n_platforms":   frame.notna().sum(axis=1),
        "rank":          rank,
    }).sort_values("rank", ascending=False)
    return out


# -----------------------------------------------------------------
# Stouffer z combination (per-platform t-stat required)
# -----------------------------------------------------------------
def _stouffer_z(per_platform: Dict[str, pd.DataFrame],
                stat_col: str = "t_stat",
                weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    Weighted Stouffer z-score combination of per-platform test statistics.
    Default weight per platform = sqrt(n_samples_in_rank) if available,
    else 1. Stat column can be any per-gene z-score-like quantity (t, KS D*sign,
    logFC, etc.). NaN entries are excluded.
    """
    series_list: Dict[str, pd.Series] = {}
    w_list: Dict[str, float] = {}
    for name, r in per_platform.items():
        if stat_col not in r.columns:
            raise ValueError(f"'{stat_col}' missing from {name} ranking")
        s = r[stat_col].dropna()
        s.index = s.index.astype(str).str.upper()
        s = s[~s.index.duplicated(keep="first")]
        series_list[name] = s
        w_list[name] = float((weights or {}).get(name, 1.0))

    if not series_list:
        return pd.DataFrame()
    all_genes = sorted(set().union(*[s.index for s in series_list.values()]))
    frame   = pd.DataFrame({n: s.reindex(all_genes) for n, s in series_list.items()})
    weights_arr = np.array([w_list[c] for c in frame.columns], dtype=float)

    vals = frame.values
    mask = ~np.isnan(vals)
    num  = np.nansum(vals * weights_arr, axis=1)
    den  = np.sqrt(np.nansum((weights_arr ** 2) * mask, axis=1))
    z = np.where(den > 0, num / den, np.nan)

    out = pd.DataFrame({
        "stouffer_z": z,
        "n_platforms": mask.sum(axis=1),
        "rank": z,
    }, index=frame.index).sort_values("rank", ascending=False)
    return out


# -----------------------------------------------------------------
# Public entry points
# -----------------------------------------------------------------
def combine_ranks(per_platform: Dict[str, pd.DataFrame],
                  method: str = "rank_product",
                  stat_col: str = "t_stat",
                  weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    Combine per-platform gene rankings into a single consensus ranking.

    per_platform: {platform_name -> ranking_df} where each df has either
        a 'rank' column (for rank_product) or a stat column (for stouffer).
    method: 'rank_product' or 'stouffer'.
    """
    if method == "rank_product":
        return _rank_product(per_platform)
    if method == "stouffer":
        return _stouffer_z(per_platform, stat_col=stat_col, weights=weights)
    raise ValueError(f"unknown method {method!r}")


def run_meta_enrichment_gsea(combined: pd.DataFrame,
                             gene_sets: Sequence[str] = DEFAULT_LIBRARIES,
                             outdir: Optional[str] = None,
                             permutation_num: int = 1000,
                             seed: int = 42) -> pd.DataFrame:
    """GSEA prerank on the consensus rank column from combine_ranks()."""
    if not _HAS_GSEAPY:
        raise RuntimeError("gseapy is not installed.")
    if "rank" not in combined.columns:
        raise ValueError("combined dataframe must have a 'rank' column")
    rnk = combined["rank"].dropna().sort_values(ascending=False)

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


def meta_enrichment_report_markdown(combined: pd.DataFrame,
                                     gsea: pd.DataFrame,
                                     platforms: Sequence[str],
                                     comparison: str,
                                     method: str,
                                     top_n: int = 15,
                                     out_path: Optional[str] = None) -> str:
    lines: List[str] = []
    lines.append(f"# Cross-platform Meta-Enrichment — {comparison}\n")
    lines.append(f"**Combination method**: `{method}`")
    lines.append(f"**Platforms combined**: {', '.join(platforms)} ({len(platforms)} total)\n")

    lines.append("## Top consensus genes")
    if combined is None or combined.empty:
        lines.append("_No consensus genes._\n")
    else:
        show_cols = [c for c in ("rank_product", "stouffer_z", "signed_mean",
                                 "n_platforms", "rank") if c in combined.columns]
        top = combined.head(top_n).copy()
        top.insert(0, "gene", top.index)
        lines.append(_df_to_md(top[["gene"] + show_cols]))
        lines.append("")

    lines.append("## Pathways (meta-GSEA)")
    if gsea is None or gsea.empty or "NES" not in gsea.columns:
        lines.append("_No results._\n")
    else:
        cols = [c for c in ("library", "Term", "NES", "FDR q-val", "NOM p-val",
                            "Lead_genes") if c in gsea.columns]
        top = gsea[cols].sort_values("NES", ascending=False).head(top_n)
        lines.append(_df_to_md(top))
        lines.append("")

    md = "\n".join(lines)
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(md)
        return out_path
    return md
