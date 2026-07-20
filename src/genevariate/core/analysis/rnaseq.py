"""
NGS (raw-count RNA-seq) differential-expression pipeline.

This module completes the RNA-seq path that was previously only declared
(`pydeseq2` in the ``rnaseq`` optional-dependency group but never wired in).
It provides, headless and Tk-free:

  - QC metrics on a raw count matrix (library size, genes detected, %mito)
  - CPM / log-CPM normalisation (pure numpy — no optional deps)
  - DESeq2 median-of-ratios size factors (pure numpy — works without pydeseq2)
  - DESeq2 negative-binomial differential expression via pydeseq2
  - a bridge that turns DESeq2 results into a ranked list for the existing
    GSEA (`genevariate.core.analysis.enrichment.run_prerank_gsea`)
  - a bridge that turns normalised counts into GeneVariate's canonical
    "platform DataFrame" (GSM | genes...) so DESeq2 output flows into every
    existing analysis window unchanged.

Count-matrix convention throughout this module: **genes x samples**
(rows = genes, columns = samples), matching how counts are usually
distributed (e.g. featureCounts / STAR gene-count tables). pydeseq2 wants
samples x genes, so `run_deseq2` transposes internally.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:  # pydeseq2 is optional (rnaseq extra). Import-guard like enrichment.gseapy.
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    _HAS_PYDESEQ2 = True
except Exception:  # pragma: no cover - exercised only when dep absent
    DeseqDataSet = None
    DeseqStats = None
    _HAS_PYDESEQ2 = False


DEFAULT_MITO_PREFIXES: Tuple[str, ...] = ("MT-", "mt-", "Mt-")


# -----------------------------------------------------------------
# QC
# -----------------------------------------------------------------
def compute_qc_metrics(counts: pd.DataFrame,
                       mito_prefixes: Sequence[str] = DEFAULT_MITO_PREFIXES
                       ) -> pd.DataFrame:
    """Per-sample QC on a raw count matrix (genes x samples).

    Returns a DataFrame indexed by sample with columns:
        library_size     total counts in the sample
        n_genes_detected number of genes with count > 0
        pct_mito         percentage of counts from mitochondrial genes
    Pure numpy/pandas — no optional dependencies.
    """
    if counts.empty:
        raise ValueError("counts is empty")
    c = counts.astype(float)
    library_size = c.sum(axis=0)
    n_genes = (c > 0).sum(axis=0)

    gene_index = c.index.astype(str)
    is_mito = np.zeros(len(gene_index), dtype=bool)
    for pref in mito_prefixes:
        is_mito |= gene_index.str.startswith(pref)
    mito_counts = c.loc[is_mito].sum(axis=0) if is_mito.any() else pd.Series(
        0.0, index=c.columns)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_mito = np.where(library_size > 0,
                            100.0 * mito_counts / library_size, 0.0)

    return pd.DataFrame({
        "library_size": library_size.astype(float),
        "n_genes_detected": n_genes.astype(int),
        "pct_mito": pd.Series(pct_mito, index=c.columns),
    })


# -----------------------------------------------------------------
# Normalisation
# -----------------------------------------------------------------
def cpm_normalize(counts: pd.DataFrame, log: bool = True,
                  prior_count: float = 1.0) -> pd.DataFrame:
    """Counts-per-million normalisation (genes x samples).

    CPM_ij = counts_ij / library_size_j * 1e6. If ``log`` is True, returns
    ``log2(CPM + prior_count)`` (the form expected by the existing t-test /
    ΔVariance analyses). Pure numpy — no optional deps.
    """
    c = counts.astype(float)
    lib = c.sum(axis=0)
    lib = lib.replace(0, np.nan)
    cpm = c.div(lib, axis=1) * 1e6
    cpm = cpm.fillna(0.0)
    if log:
        return np.log2(cpm + prior_count)
    return cpm


def deseq2_size_factors(counts: pd.DataFrame) -> pd.Series:
    """DESeq2 median-of-ratios size factors (genes x samples), in numpy.

    Implemented directly (no pydeseq2 required) so QC/normalisation work even
    when pydeseq2 is absent, and so `run_deseq2` has a deterministic fallback.
    Genes with a zero across any sample (non-positive geometric mean) are
    excluded from the ratio, per Anders & Huber (2010).
    """
    c = counts.astype(float)
    with np.errstate(divide="ignore"):
        log_counts = np.log(c.values)
    # Geometric mean per gene across samples; genes with any zero -> -inf.
    log_gmean = log_counts.mean(axis=1)
    finite = np.isfinite(log_gmean)
    if not finite.any():
        # Degenerate matrix (every gene has a zero) — fall back to library-size.
        lib = c.sum(axis=0)
        sf = lib / lib[lib > 0].mean() if (lib > 0).any() else lib.replace(0, 1.0)
        return pd.Series(np.where(sf > 0, sf, 1.0), index=c.columns)

    ratios = log_counts[finite, :] - log_gmean[finite][:, None]
    log_sf = np.median(ratios, axis=0)
    sf = np.exp(log_sf)
    sf = np.where(np.isfinite(sf) & (sf > 0), sf, 1.0)
    return pd.Series(sf, index=c.columns, name="size_factor")


# -----------------------------------------------------------------
# Differential expression (DESeq2 via pydeseq2)
# -----------------------------------------------------------------
def run_deseq2(counts: pd.DataFrame,
               design: pd.DataFrame,
               contrast: Tuple[str, str, str],
               min_count: int = 10,
               shrink: bool = True) -> pd.DataFrame:
    """Negative-binomial differential expression via pydeseq2.

    Parameters
    ----------
    counts : genes x samples raw integer counts.
    design : DataFrame indexed by sample, with at least the design factor
             column named ``contrast[0]``.
    contrast : (factor, test_level, reference_level), e.g.
               ("condition", "treated", "control").
    min_count : drop genes whose total count across samples is below this.
    shrink : if True (default), apply pydeseq2's native **apeglm** log-fold-change
             shrinkage. Shrunken LFCs stabilise the ranking of low-count genes
             and are the current recommended default for reporting effect sizes.
             Falls back silently to unshrunken LFCs if shrinkage is unavailable.

    Returns a DataFrame indexed by gene with columns
        log2FoldChange, pvalue, padj, baseMean, stat
    When ``shrink`` is True the returned ``log2FoldChange`` is the shrunken
    estimate; ``stat`` (the Wald statistic) is preserved for ranking.
    """
    if not _HAS_PYDESEQ2:
        raise RuntimeError(
            "pydeseq2 is not installed. `pip install genevariate[rnaseq]` "
            "(or `pip install pydeseq2`) to run DESeq2 differential expression."
        )
    factor, test_level, ref_level = contrast

    # Align samples between counts and design.
    samples = [s for s in counts.columns if s in design.index]
    if len(samples) < 2:
        raise ValueError("counts and design share fewer than 2 samples")
    counts = counts[samples]
    design = design.loc[samples]
    if factor not in design.columns:
        raise ValueError(f"design has no column {factor!r}")

    # Filter low-count genes, then transpose to samples x genes for pydeseq2.
    keep = counts.sum(axis=1) >= min_count
    counts = counts.loc[keep]
    if counts.empty:
        raise ValueError(f"No genes pass min_count={min_count}")
    counts_sxg = counts.T.round().astype(int)

    metadata = design[[factor]].copy()
    metadata[factor] = metadata[factor].astype(str)

    try:
        dds = DeseqDataSet(
            counts=counts_sxg,
            metadata=metadata,
            design_factors=factor,
            ref_level=[factor, ref_level],
            quiet=True,
        )
    except TypeError:
        # pydeseq2 API drift across 0.4.x: older/newer signatures differ.
        dds = DeseqDataSet(
            counts=counts_sxg,
            metadata=metadata,
            design=f"~{factor}",
        )
    dds.deseq2()

    try:
        stats = DeseqStats(dds, contrast=[factor, test_level, ref_level],
                           quiet=True)
    except TypeError:
        stats = DeseqStats(dds, contrast=[factor, test_level, ref_level])
    stats.summary()

    if shrink:
        # apeglm shrinkage of the tested coefficient. pydeseq2 names the
        # coefficient "<factor>_<test>_vs_<ref>"; API drifts across versions,
        # so isolate the whole thing behind try/except and keep unshrunken
        # LFCs on any failure.
        coeff = f"{factor}_{test_level}_vs_{ref_level}"
        try:
            stats.lfc_shrink(coeff=coeff)
        except Exception:
            try:
                stats.lfc_shrink()
            except Exception:
                pass
    res = stats.results_df.copy()

    # Normalise column names across pydeseq2 versions.
    rename = {}
    if "pvalue" not in res.columns and "pval" in res.columns:
        rename["pval"] = "pvalue"
    res = res.rename(columns=rename)
    for col in ("log2FoldChange", "pvalue", "padj", "baseMean", "stat"):
        if col not in res.columns:
            res[col] = np.nan
    return res[["log2FoldChange", "pvalue", "padj", "baseMean", "stat"]]


def deseq_results_to_ranked(res: pd.DataFrame,
                            rank_by: str = "stat") -> pd.DataFrame:
    """Turn DESeq2 results into a ranked list for `run_prerank_gsea`.

    Produces a gene-indexed DataFrame with a ``rank`` column (descending =
    up in the test level). Default rank = DESeq2 Wald ``stat`` (signed and
    monotonic). Fallback when ``stat`` is unavailable:
    ``sign(log2FoldChange) * -log10(pvalue)``.
    """
    out = res.copy()
    if rank_by == "stat" and "stat" in out.columns and out["stat"].notna().any():
        rank = out["stat"].astype(float)
    else:
        lfc = out.get("log2FoldChange", pd.Series(0.0, index=out.index)).astype(float)
        p = out.get("pvalue", pd.Series(np.nan, index=out.index)).astype(float)
        p = p.clip(lower=1e-300).fillna(1.0)
        rank = np.sign(lfc) * -np.log10(p)
    out["rank"] = rank.fillna(0.0)
    return out.sort_values("rank", ascending=False)


# -----------------------------------------------------------------
# Bridge to the canonical platform DataFrame
# -----------------------------------------------------------------
def counts_to_platform_df(norm_counts: pd.DataFrame,
                          sample_meta: Optional[pd.DataFrame] = None
                          ) -> pd.DataFrame:
    """Turn normalised counts (genes x samples) into GeneVariate canonical
    format (rows = samples): ``GSM | [Classified_*...] | GENE1 | GENE2 | ...``.

    The ``GSM`` column is required by
    `genevariate.core.analysis.enrichment._expr_from_canonical`, so DESeq2
    output can be registered as a platform and reused by every analysis window.
    ``sample_meta`` (indexed by sample) contributes any ``Classified_*`` /
    ``series_id`` columns.
    """
    expr = norm_counts.astype(float).T  # samples x genes
    df = expr.reset_index().rename(columns={"index": "GSM"})
    df["GSM"] = df["GSM"].astype(str)

    if sample_meta is not None and not sample_meta.empty:
        meta = sample_meta.copy()
        meta.index = meta.index.astype(str)
        keep_cols = [c for c in meta.columns
                     if c == "series_id" or str(c).startswith("Classified_")]
        if keep_cols:
            df = df.merge(meta[keep_cols], left_on="GSM", right_index=True,
                          how="left")
            # Re-order so metadata sits right after GSM.
            front = ["GSM"] + [c for c in ("series_id", *keep_cols)
                               if c in df.columns and c != "GSM"]
            front = list(dict.fromkeys(front))
            rest = [c for c in df.columns if c not in front]
            df = df[front + rest]
    return df
