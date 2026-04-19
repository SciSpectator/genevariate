"""
Tests for the base data-source contract.

  - to_canonical() must emit GSM | series_id | <gene_cols>.
  - save_csv() writes .csv.gz with provenance sidecar when requested.
  - sidecar metadata contains the required reproducibility fields.
"""

from __future__ import annotations

import gzip
import json
import os

import numpy as np
import pandas as pd
import pytest

from genevariate.core.sources.base import (
    BaseSource,
    CANONICAL_META_COLS,
    SourceInfo,
)


def test_to_canonical_orientation_and_cols():
    expr = pd.DataFrame(
        np.arange(12).reshape(3, 4).astype(float),
        index=["GeneA", "GeneB", "GeneC"],
        columns=["gsm1", "gsm2", "gsm3", "gsm4"],
    )
    out = BaseSource.to_canonical(expr, gsm_to_series={"GSM1": "GSE1"})
    assert list(out.columns[:2]) == list(CANONICAL_META_COLS)
    assert out.shape == (4, 2 + 3)
    assert set(out["GSM"]) == {"GSM1", "GSM2", "GSM3", "GSM4"}
    assert out.loc[out["GSM"] == "GSM1", "series_id"].iloc[0] == "GSE1"


def test_to_canonical_empty_returns_empty_with_meta_cols():
    out = BaseSource.to_canonical(pd.DataFrame())
    assert list(out.columns) == list(CANONICAL_META_COLS)
    assert out.empty


def test_save_csv_writes_provenance_sidecar(tmp_path):
    df = pd.DataFrame({
        "GSM":       ["GSM1", "GSM2"],
        "series_id": ["GSE1", "GSE1"],
        "TP53":      [1.0, 2.0],
        "BRCA1":     [3.0, 4.0],
    })
    provenance = {
        "source":        "ARCHS4",
        "h5_version":    "2.5",
        "h5_filename":   "human_gene_v2.5.h5",
        "species":       "human",
        "query":         "GSE123",
    }
    path = BaseSource.save_csv(df, str(tmp_path), "unit.csv.gz",
                                provenance=provenance)
    assert os.path.exists(path)
    sidecar = path.replace(".csv.gz", ".meta.json")
    assert os.path.exists(sidecar), "save_csv should write a .meta.json sidecar"

    with open(sidecar) as fh:
        meta = json.load(fh)
    # Provenance fields surface
    for k, v in provenance.items():
        assert meta.get(k) == v
    # Bookkeeping fields
    assert "generated_at_utc" in meta
    assert "shape" in meta and meta["shape"] == [2, 4]
    assert "genevariate_version" in meta


def test_save_csv_skips_sidecar_without_provenance(tmp_path):
    df = pd.DataFrame({"GSM": ["GSM1"], "series_id": ["GSE1"], "TP53": [1.0]})
    path = BaseSource.save_csv(df, str(tmp_path), "noprov.csv.gz")
    assert os.path.exists(path)
    assert not os.path.exists(path.replace(".csv.gz", ".meta.json"))


def test_save_csv_round_trip(tmp_path):
    df = pd.DataFrame({
        "GSM":       ["GSM1", "GSM2"],
        "series_id": ["GSE1", "GSE1"],
        "TP53":      [1.5, 2.5],
    })
    path = BaseSource.save_csv(df, str(tmp_path), "rt.csv.gz",
                                provenance={"source": "unit-test"})
    back = pd.read_csv(path, compression="gzip")
    pd.testing.assert_frame_equal(back, df)


def test_source_info_defaults():
    info = SourceInfo(name="X", technology="bulk-rna-seq")
    assert info.species == "human"
    assert info.description == ""
