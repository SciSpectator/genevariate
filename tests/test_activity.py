"""
Tests for activity inference (``core.analysis.activity``).

The decoupleR-dependent paths are skipped when decoupler is absent; the
error-message contract is checked unconditionally.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from genevariate.core.analysis import activity as act


def _platform(n=30, seed=0):
    rng = np.random.default_rng(seed)
    genes = ["TP53", "MDM2", "MYC", "EGFR", "STAT1", "STAT3", "JUN", "FOS"]
    data = {"GSM": [f"S{i}" for i in range(n)],
            "Classified_condition": ["a"] * n}
    for g in genes:
        data[g] = rng.normal(5, 1, n)
    return pd.DataFrame(data)


def test_expr_matrix_drops_metadata_and_uppercases():
    mat = act._expr_matrix(_platform())
    assert "GSM" not in mat.columns
    assert "CLASSIFIED_CONDITION" not in mat.columns
    assert "TP53" in mat.columns
    assert mat.shape[0] == 30


def test_activity_requires_decoupler_message():
    if act._HAS_DECOUPLER:
        pytest.skip("decoupler installed — error-path not exercised")
    with pytest.raises(RuntimeError) as e:
        act.tf_activity(_platform())
    assert "decoupler" in str(e.value).lower()


@pytest.mark.skipif(not act._HAS_DECOUPLER, reason="decoupler not installed")
def test_tf_activity_runs_when_available():
    res = act.tf_activity(_platform(), organism="human")
    assert "activities" in res and "ranked" in res
    assert res["activities"].shape[0] == 30
    assert res["report"]
