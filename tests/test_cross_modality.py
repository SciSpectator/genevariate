"""
Tests for cross-modality gene analysis (``core.analysis.cross_modality``):
same-gene comparison on a harmonised scale, single-source co-expression, and
cross-modality co-expression consensus. All offline — pure numpy/pandas/scipy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from genevariate.core.analysis import (
    infer_modality,
    harmonize_vectors,
    compare_gene_across_modalities,
    gene_coexpression,
    coexpression_consensus,
)


def _source(seed, n, scale, offset):
    """A platform DataFrame where MDM2 tracks TP53 and ANTI opposes it."""
    r = np.random.default_rng(seed)
    tp53 = r.normal(offset, scale, n)
    return pd.DataFrame({
        "GSM": [f"S{seed}_{i}" for i in range(n)],
        "Classified_condition": ["a"] * n,
        "TP53": tp53,
        "MDM2": tp53 * 0.9 + r.normal(0, scale * 0.2, n),
        "ANTI": -tp53 * 0.8 + r.normal(0, scale * 0.2, n),
        "NOISE": r.normal(0, 1, n),
    })


def test_infer_modality():
    assert infer_modality("GPL570") == "microarray"
    assert infer_modality("scRNA_lung") == "single-cell"
    assert infer_modality("RNAseq_counts") == "rna-seq"
    assert infer_modality("mystery") == "expression"


def test_harmonize_zscore_and_rank():
    vecs = {"a": np.array([10.0, 12.0, 14.0]),
            "b": np.array([100.0, 120.0, 140.0])}  # same shape, different scale
    z = harmonize_vectors(vecs, method="zscore")
    # z-scored vectors of an affine-scaled series are identical
    assert np.allclose(z["a"], z["b"])
    rk = harmonize_vectors(vecs, method="rank")
    assert np.allclose(rk["a"], [0.0, 0.5, 1.0])


def test_compare_across_modalities_harmonises_scale():
    # three different scales/offsets, same underlying normal shape
    sources = {"GPL570": _source(1, 60, 1.0, 7.0),
               "RNAseq_x": _source(2, 40, 3.0, 2.0),
               "scRNA_lung": _source(3, 80, 0.5, 9.0)}
    res = compare_gene_across_modalities(sources, "TP53", method="zscore")
    tbl = res["table"]
    assert set(tbl["modality"]) == {"microarray", "rna-seq", "single-cell"}
    # after harmonisation the shapes should agree (same normal)
    assert not res["pairwise"].empty
    assert (res["pairwise"]["verdict"] == "same shape").all()
    assert res["concordant"] is True
    assert "TP53" in res["summary"]


def test_compare_missing_gene_reports_not_found():
    sources = {"GPL570": _source(1, 30, 1.0, 7.0),
               "RNAseq_x": _source(2, 30, 3.0, 2.0)}
    res = compare_gene_across_modalities(sources, "NOTAGENE")
    assert "not found" in res["summary"].lower()
    assert res["pairwise"].empty


def test_gene_coexpression_finds_partners():
    df = _source(1, 100, 1.0, 7.0)
    out = gene_coexpression(df, "TP53", method="pearson", top_n=25)
    # MDM2 positive, ANTI negative, both strong; NOISE weak
    assert out.loc["MDM2", "r"] > 0.8
    assert out.loc["ANTI", "r"] < -0.7
    assert abs(out.loc["NOISE", "r"]) < 0.3


def test_gene_coexpression_rho_proportionality():
    """Lovell's rho ranks the proportional partner high and noise low."""
    df = _source(1, 100, 1.0, 7.0)
    out = gene_coexpression(df, "TP53", method="rho", top_n=25)
    assert out.loc["MDM2", "r"] > 0.6      # proportional log profile
    assert abs(out.loc["NOISE", "r"]) < 0.5
    # MDM2 (proportional) should outrank NOISE by |rho|
    assert out.loc["MDM2", "abs_r"] > out.loc["NOISE", "abs_r"]


def test_gene_coexpression_missing_gene_raises():
    df = _source(1, 20, 1.0, 7.0)
    try:
        gene_coexpression(df, "NOPE")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_coexpression_consensus_keeps_reproducible_links():
    sources = {"GPL570": _source(1, 80, 1.0, 7.0),
               "RNAseq_x": _source(2, 60, 3.0, 2.0),
               "scRNA_lung": _source(3, 100, 0.5, 9.0)}
    res = coexpression_consensus(sources, "TP53", method="pearson",
                                 top_n=25, min_abs=0.3)
    tbl = res["table"]
    # MDM2 (positive) and ANTI (inverse) reproduce across all three modalities
    assert "MDM2" in tbl.index and "ANTI" in tbl.index
    assert (tbl.loc["MDM2", "n_sources"]) == 3
    assert tbl.loc["MDM2", "mean_r"] > 0
    assert tbl.loc["ANTI", "mean_r"] < 0
    # NOISE is not a reproducible connection
    assert "NOISE" not in tbl.index


def test_consensus_needs_two_sources():
    sources = {"only": _source(1, 30, 1.0, 7.0)}
    res = coexpression_consensus(sources, "TP53")
    assert res["table"].empty
    assert "at least two" in res["summary"].lower()
