"""
Tests for the multi-gene conjunction box and its multiplicative-null synergy.

The score has to answer one question honestly: does the *combination* of genes
say more than the genes do separately? Two genes that mark the same samples are
redundant (synergy ~ 1 or below); two genes that only agree on a rare corner of
the platform are synergistic (synergy > 1). A box that collapses onto a couple
of studies must not come back with a confident interval.
"""

from __future__ import annotations

import numpy as np
import pytest

from genevariate.core.analysis.synergy import (
    conjunction_mask,
    multiplicative_null,
    synergy_diagnostics,
)


N_STUDIES = 200
STUDY_SIZE = 30
N = N_STUDIES * STUDY_SIZE


def _groups():
    return np.repeat(np.arange(N_STUDIES), STUDY_SIZE)


def test_conjunction_mask_is_the_intersection():
    m = conjunction_mask({
        "A": [True, True, False, False],
        "B": [True, False, True, False],
    })
    assert list(m) == [True, False, False, False]


def test_conjunction_mask_rejects_ragged_input():
    with pytest.raises(ValueError):
        conjunction_mask({"A": [True, False], "B": [True]})
    with pytest.raises(ValueError):
        conjunction_mask({})


def test_multiplicative_null_is_the_product_of_lifts():
    assert multiplicative_null([2.0, 3.0]) == pytest.approx(6.0)
    assert np.isnan(multiplicative_null([2.0, float("nan")]))


def test_redundant_genes_have_no_identifiable_interaction():
    """Two genes marking the SAME samples leave the off-diagonal cells empty."""
    rng = np.random.default_rng(0)
    slab = rng.random(N) < 0.2
    labels = np.where(rng.random(N) < np.where(slab, 0.6, 0.1), "Liver", "Other")

    d = synergy_diagnostics({"G1": slab, "G2": slab.copy()}, labels,
                            _groups(), values=["Liver"], n_boot=300)["Liver"]
    assert d["lift_box"] > 2.0          # the box IS enriched...
    assert d["empty_cells"] == 2        # ...but G1-only and G2-only never happen
    assert np.isnan(d["synergy"])       # so "does the pair add anything" is unanswerable


def test_genes_that_only_agree_on_the_signal_are_synergistic():
    """Label lives ONLY where both slabs overlap -> a real AND interaction."""
    rng = np.random.default_rng(1)
    g1 = rng.random(N) < 0.3
    g2 = rng.random(N) < 0.3
    both = g1 & g2
    labels = np.where(rng.random(N) < np.where(both, 0.9, 0.01), "Liver", "Other")

    d = synergy_diagnostics({"G1": g1, "G2": g2}, labels, _groups(),
                            values=["Liver"], n_boot=300)["Liver"]
    assert d["synergy"] > 10.0
    assert d["ci_low"] > 1.0
    assert d["exp_a"] < d["a"]


def test_log_additive_genes_land_on_the_null():
    """Two genes acting multiplicatively on the odds -> no interaction left."""
    rng = np.random.default_rng(2)
    g1 = rng.random(N) < 0.3
    g2 = rng.random(N) < 0.3
    # log-additive risk: exactly the model the interaction contrast cancels
    p = 0.02 * np.where(g1, 3.0, 1.0) * np.where(g2, 3.0, 1.0)
    labels = np.where(rng.random(N) < p, "Liver", "Other")

    d = synergy_diagnostics({"G1": g1, "G2": g2}, labels, _groups(),
                            values=["Liver"], n_boot=400)["Liver"]
    assert d["synergy"] == pytest.approx(1.0, abs=0.35)
    assert d["ci_low"] <= 1.0 <= d["ci_high"]


def test_antagonistic_genes_score_below_one():
    """Either gene alone marks the label; needing both adds nothing."""
    rng = np.random.default_rng(12)
    g1 = rng.random(N) < 0.3
    g2 = rng.random(N) < 0.3
    either = g1 | g2
    labels = np.where(rng.random(N) < np.where(either, 0.5, 0.02), "Liver", "Other")

    d = synergy_diagnostics({"G1": g1, "G2": g2}, labels, _groups(),
                            values=["Liver"], n_boot=300)["Liver"]
    assert d["synergy"] < 0.5
    assert d["ci_high"] < 1.0


def test_counts_and_expectation_match_the_box():
    rng = np.random.default_rng(3)
    g1 = rng.random(N) < 0.4
    g2 = rng.random(N) < 0.4
    labels = np.where(rng.random(N) < 0.25, "Liver", "Other")

    d = synergy_diagnostics({"G1": g1, "G2": g2}, labels, _groups(),
                            values=["Liver"], n_boot=0)["Liver"]
    box = g1 & g2
    assert d["n_box"] == int(box.sum())
    assert d["a"] == int(((labels == "Liver") & box).sum())
    assert d["n_genes"] == 2
    assert set(d["marginal_lifts"]) == {"G1", "G2"}


def test_box_confined_to_one_study_is_not_defended_by_the_ci():
    rng = np.random.default_rng(4)
    groups = _groups()
    g1 = rng.random(N) < 0.3
    g2 = rng.random(N) < 0.3
    labels = np.full(N, "Other", dtype=object)
    labels[(groups == 0) & g1 & g2] = "Liver"

    d = synergy_diagnostics({"G1": g1, "G2": g2}, labels, groups,
                            values=["Liver"], n_boot=400)["Liver"]
    assert d["n_gse"] == 1
    assert not (d["ci_low"] > 1.0)
    assert d["n_eff_box"] < d["n_box"]


def test_without_groups_clumping_fields_are_unknown_not_faked():
    rng = np.random.default_rng(5)
    g1 = rng.random(N) < 0.3
    g2 = rng.random(N) < 0.3
    labels = np.where(rng.random(N) < 0.2, "Liver", "Other")

    d = synergy_diagnostics({"G1": g1, "G2": g2}, labels, None,
                            values=["Liver"])["Liver"]
    assert d["n_gse"] is None
    assert np.isnan(d["rho"])
    assert np.isnan(d["ci_low"]) and np.isnan(d["ci_high"])
    assert d["n_eff_box"] == pytest.approx(d["n_box"])


def test_three_genes_compose():
    rng = np.random.default_rng(6)
    masks = {f"G{i}": rng.random(N) < 0.5 for i in range(3)}
    labels = np.where(rng.random(N) < 0.2, "Liver", "Other")
    d = synergy_diagnostics(masks, labels, _groups(), values=["Liver"],
                            n_boot=200)["Liver"]
    assert d["n_genes"] == 3
    assert d["expected_lift"] == pytest.approx(
        np.prod(list(d["marginal_lifts"].values())))


def test_rejects_misaligned_labels_and_groups():
    with pytest.raises(ValueError):
        synergy_diagnostics({"G": [True, False]}, ["a"], None)
    with pytest.raises(ValueError):
        synergy_diagnostics({"G": [True, False]}, ["a", "b"], ["S1"])


def test_gui_helper_flags_a_thin_box():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from genevariate.gui.region_analysis import _is_thin_box

    assert not _is_thin_box({"n_gse": 14, "n_box": 900, "ci_low": 1.8})
    assert _is_thin_box({"n_gse": 2, "n_box": 900, "ci_low": 1.8})     # 2 studies
    assert _is_thin_box({"n_gse": 14, "n_box": 6, "ci_low": 1.8})      # box empty
    assert _is_thin_box({"n_gse": 14, "n_box": 900, "ci_low": 0.7})    # CI covers 1
    assert not _is_thin_box({"n_gse": None, "n_box": 900,
                             "ci_low": float("nan")})
