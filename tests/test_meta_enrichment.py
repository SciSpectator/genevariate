"""
Unit tests for cross-platform meta-enrichment combination.

Two core mathematical guarantees:
- Identity:      a single-platform input returns the same ranking (up to
                 sorting) as the per-platform input.
- Stouffer z:    for identical z-scores across k platforms with equal weights
                 the combined z equals the per-platform z × sqrt(k).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from genevariate.core.analysis.meta_enrichment import combine_ranks


def _mock_platform(genes, rank_values, t_values=None):
    df = pd.DataFrame({"rank": rank_values}, index=genes)
    if t_values is not None:
        df["t_stat"] = t_values
    return df


def test_rank_product_identity_single_platform():
    """Single-platform rank_product output should preserve gene ordering."""
    genes = [f"G{i}" for i in range(8)]
    rv = [5.0, -3.0, 0.1, 2.0, -1.5, 4.4, 0.0, -0.2]
    per = {"GPL1": _mock_platform(genes, rv)}
    out = combine_ranks(per, method="rank_product")
    # Largest rank value => first in sorted output
    top = out.index.tolist()[:3]
    # With descending sort the top genes should be those with most positive input
    positive_top = pd.Series(rv, index=genes).sort_values(ascending=False).head(3).index.tolist()
    assert top == positive_top


def test_stouffer_z_scales_with_sqrt_k():
    """Stouffer: identical z across k platforms → combined = z * sqrt(k)."""
    genes = [f"G{i}" for i in range(6)]
    t_vals = np.array([2.0, -1.0, 0.5, 0.0, 3.5, -2.2])
    platforms = {name: _mock_platform(genes, rank_values=t_vals, t_values=t_vals)
                 for name, _ in [("P1", None), ("P2", None), ("P3", None), ("P4", None)]}
    out = combine_ranks(platforms, method="stouffer", stat_col="t_stat")
    # Combined z should equal t * sqrt(4) = t * 2
    combined_z = out["stouffer_z"].reindex(genes).values
    expected = t_vals * np.sqrt(4)
    np.testing.assert_allclose(combined_z, expected, rtol=1e-6)
    assert (out["n_platforms"] == 4).all()


def test_stouffer_handles_missing_platforms_per_gene():
    """Genes measured on only some platforms use the valid subset."""
    per = {
        "P1": _mock_platform(["A", "B", "C"], [1, 2, 3], t_values=[1.0, 2.0, 3.0]),
        "P2": _mock_platform(["B", "C", "D"], [1, 2, 3], t_values=[2.0, 3.0, 4.0]),
    }
    out = combine_ranks(per, method="stouffer", stat_col="t_stat")
    assert out.loc["A", "n_platforms"] == 1
    assert out.loc["B", "n_platforms"] == 2
    assert out.loc["D", "n_platforms"] == 1
    # Gene B appears in both with weight 1 → z = (2+2)/sqrt(2) = 2*sqrt(2)
    assert np.isclose(out.loc["B", "stouffer_z"], 4.0 / np.sqrt(2.0))


def test_combine_ranks_unknown_method_raises():
    with pytest.raises(ValueError):
        combine_ranks({}, method="bogus")
