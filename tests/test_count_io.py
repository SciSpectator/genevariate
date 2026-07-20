"""
Tests for the raw-count readers (``core.count_io``): CSV round-trip and a tiny
in-memory 10x MTX directory built in ``tmp_path``.
"""
from __future__ import annotations

import gzip

import numpy as np
import pandas as pd
import pytest

from genevariate.core.count_io import (
    read_counts_csv,
    read_10x_mtx,
    load_counts,
    read_sidecar_meta,
)


def test_csv_roundtrip(tmp_path):
    counts = pd.DataFrame(
        {"s1": [5, 0, 12], "s2": [3, 7, 0]},
        index=["GENE1", "GENE2", "GENE3"])
    p = tmp_path / "counts.csv"
    counts.to_csv(p)
    loaded = read_counts_csv(p)
    assert list(loaded.columns) == ["s1", "s2"]
    assert list(loaded.index) == ["GENE1", "GENE2", "GENE3"]
    assert loaded.loc["GENE1", "s1"] == 5

    # via dispatch
    got, meta = load_counts(p)
    assert meta is None
    assert got.shape == (3, 2)


def test_platform_bridge_from_csv(tmp_path):
    from genevariate.core.analysis import counts_to_platform_df
    from genevariate.core.analysis.enrichment import _expr_from_canonical
    counts = pd.DataFrame(
        {"s1": [5, 1, 12], "s2": [3, 7, 2]},
        index=["GENE1", "GENE2", "GENE3"])
    p = tmp_path / "c.csv"
    counts.to_csv(p)
    got, _ = load_counts(p)
    df = counts_to_platform_df(got.astype(float))
    assert "GSM" in df.columns
    expr, gsm = _expr_from_canonical(df)
    assert expr.shape[1] == 2


def test_sidecar_meta_supplies_design_factor(tmp_path):
    """A sibling ``<stem>.meta.csv`` becomes the DESeq2 design factor."""
    counts = pd.DataFrame(
        {"S0": [5, 1], "S1": [6, 2], "S2": [1, 9], "S3": [0, 8]},
        index=["G1", "G2"])
    p = tmp_path / "counts.csv"
    counts.to_csv(p)
    meta = pd.DataFrame({"sample": ["S0", "S1", "S2", "S3"],
                         "condition": ["treated", "treated", "control", "control"]})
    meta.to_csv(tmp_path / "counts.meta.csv", index=False)

    # direct: reindexed to the count columns, id column dropped
    got = read_sidecar_meta(p, counts.columns)
    assert list(got.index) == ["S0", "S1", "S2", "S3"]
    assert list(got["condition"]) == ["treated", "treated", "control", "control"]
    assert "sample" not in got.columns

    # via dispatch — load_counts now returns the sidecar as sample_meta
    _, m = load_counts(p)
    assert m is not None and m.loc["S2", "condition"] == "control"


def test_no_sidecar_meta_returns_none(tmp_path):
    counts = pd.DataFrame({"s1": [1, 2], "s2": [3, 4]}, index=["G1", "G2"])
    p = tmp_path / "bare.csv"
    counts.to_csv(p)
    assert read_sidecar_meta(p, counts.columns) is None
    _, m = load_counts(p)
    assert m is None


def test_10x_mtx_roundtrip(tmp_path):
    pytest.importorskip("scipy")
    from scipy.io import mmwrite
    from scipy.sparse import csr_matrix

    d = tmp_path / "mtx"
    d.mkdir()
    # 3 genes x 2 cells
    mat = csr_matrix(np.array([[4, 0], [0, 9], [2, 1]], dtype=int))
    mmwrite(str(d / "matrix.mtx"), mat)
    with gzip.open(d / "features.tsv.gz", "wt") as fh:
        fh.write("ENSG1\tGENE1\tGene Expression\n")
        fh.write("ENSG2\tGENE2\tGene Expression\n")
        fh.write("ENSG3\tGENE3\tGene Expression\n")
    with gzip.open(d / "barcodes.tsv.gz", "wt") as fh:
        fh.write("CELL1\n")
        fh.write("CELL2\n")

    counts = read_10x_mtx(d)
    assert counts.shape == (3, 2)
    assert list(counts.columns) == ["CELL1", "CELL2"]
    assert counts.loc["ENSG2", "CELL2"] == 9

    # dispatch on a directory
    got, meta = load_counts(d)
    assert got.shape == (3, 2) and meta is None
