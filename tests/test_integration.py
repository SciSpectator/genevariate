"""
Tests for batch integration (``core.analysis.integration``).

``common_gene_matrix`` is pure numpy/pandas and always runs; ComBat / Harmony
paths are skipped when no backend is installed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from genevariate.core.analysis.integration import (
    common_gene_matrix, combat_correct, harmony_embed,
)


def _source(seed, n, offset):
    r = np.random.default_rng(seed)
    return pd.DataFrame({
        "GSM": [f"S{seed}_{i}" for i in range(n)],
        "Classified_condition": ["a"] * n,
        "TP53": r.normal(offset, 1.0, n),
        "MDM2": r.normal(offset, 1.0, n),
        "EXTRA": r.normal(offset, 1.0, n),  # only in this source when seed==1
    }) if seed == 1 else pd.DataFrame({
        "GSM": [f"S{seed}_{i}" for i in range(n)],
        "Classified_condition": ["a"] * n,
        "TP53": r.normal(offset, 1.0, n),
        "MDM2": r.normal(offset, 1.0, n),
    })


def test_common_gene_matrix_shared_genes_and_batch():
    sources = {"GPL570": _source(1, 20, 7.0), "RNAseq": _source(2, 15, 2.0)}
    matrix, batch = common_gene_matrix(sources)
    # only the shared genes (TP53, MDM2) — EXTRA is dropped
    assert set(matrix.index) == {"TP53", "MDM2"}
    assert matrix.shape[1] == 35                      # 20 + 15 samples
    assert set(batch.unique()) == {"GPL570", "RNAseq"}
    assert (batch == "GPL570").sum() == 20


def test_common_gene_matrix_needs_two_sources():
    with pytest.raises(ValueError):
        common_gene_matrix({"only": _source(2, 10, 2.0)})


def _has_combat():
    for mod in ("inmoose.pycombat", "pycombat", "combat.pycombat"):
        try:
            __import__(mod)
            return True
        except Exception:
            continue
    return False


@pytest.mark.skipif(not _has_combat(), reason="no ComBat backend installed")
def test_combat_correct_reduces_batch_offset():
    # two sources with a large mean offset (a pure batch effect)
    sources = {"GPL570": _source(1, 40, 10.0), "RNAseq": _source(2, 40, 0.0)}
    before = abs(sources["GPL570"]["TP53"].mean()
                 - sources["RNAseq"]["TP53"].mean())
    corrected = combat_correct(sources)
    assert set(corrected) == {"GPL570", "RNAseq"}
    after = abs(corrected["GPL570"]["TP53"].mean()
                - corrected["RNAseq"]["TP53"].mean())
    assert after < before          # batch offset shrunk
    assert "GSM" in corrected["GPL570"].columns


@pytest.mark.skipif(not _has_combat(), reason="no ComBat backend installed")
def test_combat_absent_raises_clearly(monkeypatch):
    # even with a backend present, ensure the guarded path reports missing genes
    with pytest.raises(ValueError):
        combat_correct({"a": _source(1, 5, 1.0)})


def test_harmony_missing_dep_raises_or_runs():
    sources = {"GPL570": _source(1, 20, 7.0), "RNAseq": _source(2, 20, 2.0)}
    try:
        import harmonypy  # noqa: F401
    except Exception:
        with pytest.raises(RuntimeError):
            harmony_embed(sources)
        return
    out = harmony_embed(sources, n_pcs=2)
    assert "embedding" in out and out["embedding"].shape[0] == 40
