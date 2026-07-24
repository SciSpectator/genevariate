"""
Tests for the calibrated P(label | genes) box model.

The model exists to answer a question counting cannot: what a box holds when
the box holds nothing. That is only worth doing if the probabilities are real
probabilities, if the folds respect study structure, and if the extrapolation
is labelled as extrapolation.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from genevariate.core.analysis.box_model import (  # noqa: E402
    fit_label_model,
    integrate_box,
    relaxation_attribution,
    reliability_curve,
)


N_STUDIES = 60
PER = 40
N = N_STUDIES * PER


def _panel(seed=0):
    """Two genes; the label lives where BOTH are high."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(N_STUDIES), PER).astype(str)
    X = rng.normal(6.0, 2.0, size=(N, 2))
    both = (X[:, 0] > 7.5) & (X[:, 1] > 7.5)
    y = rng.random(N) < np.where(both, 0.85, 0.03)
    return X, y, groups


def test_model_learns_the_conjunction():
    X, y, groups = _panel()
    m = fit_label_model(X, y, groups, feature_names=["A", "B"])
    # ~39% of the positives are planted as background noise and are genuinely
    # unlearnable, so the Bayes ceiling here is around 0.83
    assert m.auc > 0.78
    assert m.grouped is True
    assert m.n_splits >= 2
    assert m.features == ["A", "B"]


def test_isotonic_layer_does_not_make_calibration_worse():
    X, y, groups = _panel(1)
    m = fit_label_model(X, y, groups, feature_names=["A", "B"])
    assert m.ece_cal <= m.ece_raw + 1e-6
    assert m.brier_cal <= m.brier_raw + 1e-6
    assert 0.0 <= m.p_oof[np.isfinite(m.p_oof)].min()
    assert m.p_oof[np.isfinite(m.p_oof)].max() <= 1.0


def test_cross_fitted_predictions_cover_every_sample():
    X, y, groups = _panel(2)
    m = fit_label_model(X, y, groups, feature_names=["A", "B"])
    assert np.isfinite(m.p_oof).all()
    # and they are out-of-fold, so they cannot be perfect
    assert m.p_oof[m.y == 1].mean() > m.p_oof[m.y == 0].mean()


def test_integrating_the_signal_box_recovers_the_planted_rate():
    X, y, groups = _panel(3)
    m = fit_label_model(X, y, groups, feature_names=["A", "B"])
    hot = integrate_box(m, [(7.5, X[:, 0].max()), (7.5, X[:, 1].max())], seed=0)
    cold = integrate_box(m, [(X[:, 0].min(), 5.0), (X[:, 1].min(), 5.0)], seed=0)
    assert hot["p_uniform"] > 0.6          # planted rate is 0.85
    assert cold["p_uniform"] < 0.15        # planted background is 0.03
    assert hot["fold_low"] <= hot["p_uniform"] <= hot["fold_high"]


def test_relaxing_either_gene_of_an_and_gate_costs_a_lot():
    X, y, groups = _panel(4)
    m = fit_label_model(X, y, groups, feature_names=["A", "B"])
    bounds = [(7.5, X[:, 0].max()), (7.5, X[:, 1].max())]
    attr = relaxation_attribution(m, bounds, seed=0)
    assert set(attr) == {"A", "B"}
    # an AND gate cannot spare either constraint
    assert attr["A"]["drop"] > 0.2
    assert attr["B"]["drop"] > 0.2


def test_an_irrelevant_gene_contributes_nothing():
    rng = np.random.default_rng(5)
    groups = np.repeat(np.arange(N_STUDIES), PER).astype(str)
    X = rng.normal(6.0, 2.0, size=(N, 2))
    y = rng.random(N) < np.where(X[:, 0] > 7.5, 0.8, 0.03)   # gene B is noise
    m = fit_label_model(X, y, groups, feature_names=["Real", "Noise"])
    attr = relaxation_attribution(
        m, [(7.5, X[:, 0].max()), (7.5, X[:, 1].max())], seed=0)
    assert attr["Real"]["drop"] > 0.2
    assert abs(attr["Noise"]["drop"]) < 0.1


def test_reliability_curve_tracks_the_diagonal_after_calibration():
    X, y, groups = _panel(6)
    m = fit_label_model(X, y, groups, feature_names=["A", "B"])
    pred, obs, cnt = reliability_curve(m)
    assert cnt.sum() == np.isfinite(m.p_oof).sum()
    # weighted by bin occupancy - an unweighted mean is dominated by the
    # near-empty bins at the top of the range, where 3 samples is 100% or 0%
    assert np.average(np.abs(pred - obs), weights=cnt) < 0.1


def test_ungrouped_fit_reports_that_it_is_ungrouped():
    X, y, _ = _panel(7)
    m = fit_label_model(X, y, None, feature_names=["A", "B"])
    assert m.grouped is False


def test_refuses_a_label_too_rare_to_model():
    X, y, groups = _panel(8)
    rare = np.zeros(N, dtype=bool)
    rare[:5] = True
    with pytest.raises(ValueError):
        fit_label_model(X, rare, groups, feature_names=["A", "B"])


def test_rejects_misshapen_input():
    X, y, groups = _panel(9)
    with pytest.raises(ValueError):
        fit_label_model(X[:10], y, groups)
    with pytest.raises(ValueError):
        fit_label_model(X[:, 0], y, groups)


def test_empty_box_still_yields_an_estimate():
    """The whole point: a box with no samples in it is still answerable."""
    X, y, groups = _panel(10)
    m = fit_label_model(X, y, groups, feature_names=["A", "B"])
    hi_a, hi_b = X[:, 0].max(), X[:, 1].max()
    corner = [(hi_a - 0.01, hi_a), (hi_b - 0.01, hi_b)]
    n_support = int(((X[:, 0] >= corner[0][0]) & (X[:, 1] >= corner[1][0])).sum())
    assert n_support == 0
    res = integrate_box(m, corner, seed=0)
    assert np.isfinite(res["p_uniform"])
