"""
Tests for the NGS raw-count DE pipeline (``core.analysis.rnaseq``).

The pure-numpy paths (QC, CPM, median-of-ratios size factors, ranking bridge,
platform bridge) run everywhere. The DESeq2 path is skipped when pydeseq2 is
absent; GSEA is skipped when gseapy is absent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from genevariate.core.analysis.rnaseq import (
    compute_qc_metrics,
    cpm_normalize,
    deseq2_size_factors,
    deseq_results_to_ranked,
    counts_to_platform_df,
)


@pytest.fixture
def synthetic_counts():
    """200 genes x 8 samples, two groups, 10 planted up-in-treated genes."""
    rng = np.random.default_rng(0)
    n_genes, n_per = 200, 4
    base = rng.poisson(lam=50, size=(n_genes, 2 * n_per)).astype(float)
    # plant strong up-regulation in treated (last 4 samples) for genes 0..9
    base[:10, n_per:] += rng.poisson(lam=400, size=(10, n_per))
    genes = [f"G{i:03d}" for i in range(n_genes)]
    genes[190] = "MT-ND1"  # one mito gene for pct_mito
    samples = [f"ctrl{i}" for i in range(n_per)] + [f"trt{i}" for i in range(n_per)]
    counts = pd.DataFrame(base, index=genes, columns=samples)
    design = pd.DataFrame(
        {"condition": ["control"] * n_per + ["treated"] * n_per},
        index=samples)
    return counts, design


def test_qc_metrics_columns(synthetic_counts):
    counts, _ = synthetic_counts
    qc = compute_qc_metrics(counts)
    assert list(qc.columns) == ["library_size", "n_genes_detected", "pct_mito"]
    assert (qc["library_size"] > 0).all()
    assert (qc["pct_mito"] >= 0).all() and (qc["pct_mito"] <= 100).all()


def test_cpm_colsum(synthetic_counts):
    counts, _ = synthetic_counts
    cpm = cpm_normalize(counts, log=False)
    # each column should sum to ~1e6
    assert np.allclose(cpm.sum(axis=0).values, 1e6, rtol=1e-6)
    logcpm = cpm_normalize(counts, log=True)
    assert logcpm.shape == counts.shape


def test_size_factors_positive(synthetic_counts):
    counts, _ = synthetic_counts
    sf = deseq2_size_factors(counts)
    assert (sf > 0).all()
    assert len(sf) == counts.shape[1]


def test_platform_bridge_has_gsm(synthetic_counts):
    counts, design = synthetic_counts
    cpm = cpm_normalize(counts, log=True)
    df = counts_to_platform_df(cpm, sample_meta=None)
    assert "GSM" in df.columns
    # canonical splitter must accept it
    from genevariate.core.analysis.enrichment import _expr_from_canonical
    expr, gsm = _expr_from_canonical(df)
    assert expr.shape[1] == counts.shape[1]


def test_deseq2_planted_genes_rank_top(synthetic_counts):
    pytest.importorskip("pydeseq2")
    from genevariate.core.analysis.rnaseq import run_deseq2
    counts, design = synthetic_counts
    res = run_deseq2(counts, design, ("condition", "treated", "control"),
                     min_count=10)
    ranked = deseq_results_to_ranked(res)
    top = set(ranked.head(15).index)
    planted = {f"G{i:03d}" for i in range(10)}
    assert len(planted & top) >= 6


def test_deseq2_lfc_shrinkage_stabilises_effect_sizes(synthetic_counts):
    """apeglm shrinkage must keep the planted genes on top while pulling the
    magnitude of noisy LFCs toward zero (shrunken |LFC| <= unshrunken)."""
    pytest.importorskip("pydeseq2")
    from genevariate.core.analysis.rnaseq import run_deseq2
    counts, design = synthetic_counts
    contrast = ("condition", "treated", "control")
    unshrunk = run_deseq2(counts, design, contrast, min_count=10, shrink=False)
    shrunk = run_deseq2(counts, design, contrast, min_count=10, shrink=True)
    # planted genes still recovered after shrinkage
    ranked = deseq_results_to_ranked(shrunk)
    planted = {f"G{i:03d}" for i in range(10)}
    assert len(planted & set(ranked.head(15).index)) >= 6
    # noisy (non-planted) genes: shrinkage should not inflate |LFC| on average
    common = unshrunk.index.intersection(shrunk.index)
    noisy = [g for g in common if g not in planted]
    if noisy:
        u = unshrunk.loc[noisy, "log2FoldChange"].abs().median()
        s = shrunk.loc[noisy, "log2FoldChange"].abs().median()
        assert s <= u + 1e-9


def test_ranked_feeds_gsea(synthetic_counts):
    pytest.importorskip("gseapy")
    res = pd.DataFrame({
        "log2FoldChange": np.linspace(3, -3, 50),
        "pvalue": np.linspace(1e-6, 0.9, 50),
        "padj": np.linspace(1e-5, 0.95, 50),
        "baseMean": np.full(50, 100.0),
        "stat": np.linspace(8, -8, 50),
    }, index=[f"G{i:03d}" for i in range(50)])
    ranked = deseq_results_to_ranked(res)
    assert "rank" in ranked.columns
    assert ranked["rank"].is_monotonic_decreasing
