"""
Tests for the study-clumping (overdispersion) corrections.

The property that matters is directional: labels scattered independently across
studies must look like the raw count, and labels that arrive in study-sized
clumps must be discounted. Getting this backwards is what makes a region
enrichment confidently wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from genevariate.core.analysis.overdispersion import (
    design_effect,
    effective_sample_size,
    enrichment_diagnostics,
    estimate_rho,
    group_counts,
)


N_STUDIES = 300
STUDY_SIZE = 20


def _panel(seed=0):
    """(groups, in_region) for a synthetic platform of clumped studies."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(N_STUDIES), STUDY_SIZE)
    in_region = rng.random(N_STUDIES * STUDY_SIZE) < 0.2
    return groups, in_region


def test_group_counts_collapses_to_per_study_totals():
    groups = np.array(["A", "A", "B", "B", "B"])
    hits = np.array([True, False, True, True, False])
    succ, sizes = group_counts(hits, groups)
    assert list(sizes) == [2.0, 3.0]
    assert list(succ) == [1.0, 2.0]


def test_group_counts_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        group_counts([True, False], ["A"])


def test_rho_is_zero_when_label_is_independent_of_study():
    """Per-sample assignment carries no study structure -> rho ~ 0."""
    rng = np.random.default_rng(1)
    sizes = np.full(N_STUDIES, float(STUDY_SIZE))
    succ = rng.binomial(STUDY_SIZE, 0.25, size=N_STUDIES).astype(float)
    assert estimate_rho(succ, sizes) < 0.05


def test_rho_is_high_when_whole_studies_share_the_label():
    """All-or-nothing studies are maximal clumping -> rho near 1."""
    rng = np.random.default_rng(2)
    sizes = np.full(N_STUDIES, float(STUDY_SIZE))
    succ = np.where(rng.random(N_STUDIES) < 0.25, float(STUDY_SIZE), 0.0)
    assert estimate_rho(succ, sizes) > 0.9


def test_rho_degenerate_cases_are_zero_not_nan():
    sizes = np.full(10, 5.0)
    assert estimate_rho(np.zeros(10), sizes) == 0.0        # label absent
    assert estimate_rho(np.full(10, 5.0), sizes) == 0.0    # label universal
    assert estimate_rho([3.0], [5.0]) == 0.0               # single study


def test_effective_sample_size_shrinks_with_clumping():
    assert effective_sample_size(1000, 20, 0.0) == pytest.approx(1000)
    # rho = 1 with 20-sample studies: every study is worth one observation
    assert effective_sample_size(1000, 20, 1.0) == pytest.approx(50)
    assert design_effect(20, 0.5) == pytest.approx(10.5)
    # no study information -> no correction rather than a fabricated one
    assert design_effect(float("nan"), 0.5) == 1.0


def test_diagnostics_independent_label_keeps_its_sample_size():
    groups, in_region = _panel(seed=3)
    rng = np.random.default_rng(4)
    labels = np.where(rng.random(groups.size) < 0.25, "Liver", "Other")

    d = enrichment_diagnostics(in_region, labels, groups,
                               values=["Liver"], n_boot=300)["Liver"]
    n_sel = int(in_region.sum())
    assert d["rho"] < 0.05
    assert d["n_eff_sel"] == pytest.approx(n_sel, rel=0.1)
    assert d["n_gse"] > 100


def test_diagnostics_clumped_label_is_heavily_discounted():
    groups, in_region = _panel(seed=3)
    rng = np.random.default_rng(5)
    study_is_liver = rng.random(N_STUDIES) < 0.25
    labels = np.where(np.repeat(study_is_liver, STUDY_SIZE), "Liver", "Other")

    d = enrichment_diagnostics(in_region, labels, groups,
                               values=["Liver"], n_boot=300)["Liver"]
    n_sel = int(in_region.sum())
    assert d["rho"] > 0.9
    # 20-sample studies fully clumped -> ~20x less independent evidence
    assert d["n_eff_sel"] < n_sel / 10
    assert d["n_gse"] < N_STUDIES


def test_study_bootstrap_ci_covers_one_when_there_is_no_enrichment():
    groups, in_region = _panel(seed=6)
    rng = np.random.default_rng(7)
    labels = np.where(rng.random(groups.size) < 0.25, "Liver", "Other")

    d = enrichment_diagnostics(in_region, labels, groups,
                               values=["Liver"], n_boot=400)["Liver"]
    assert d["ci_low"] <= 1.0 <= d["ci_high"]


def test_study_bootstrap_ci_excludes_one_for_a_replicated_signal():
    """Enrichment planted in MANY studies survives resampling of studies."""
    rng = np.random.default_rng(8)
    groups = np.repeat(np.arange(N_STUDIES), STUDY_SIZE)
    in_region = rng.random(N_STUDIES * STUDY_SIZE) < 0.2
    # every study contributes: label is 4x more likely inside the region
    p = np.where(in_region, 0.8, 0.2)
    labels = np.where(rng.random(groups.size) < p, "Liver", "Other")

    d = enrichment_diagnostics(in_region, labels, groups,
                               values=["Liver"], n_boot=400)["Liver"]
    assert d["ci_low"] > 1.0
    assert d["n_gse"] > 100


def test_single_study_signal_is_not_defended_by_the_ci():
    """A whole-signal-in-one-study hit must NOT come back with a tight CI."""
    rng = np.random.default_rng(9)
    groups = np.repeat(np.arange(N_STUDIES), STUDY_SIZE)
    in_region = rng.random(N_STUDIES * STUDY_SIZE) < 0.2
    labels = np.full(groups.size, "Other", dtype=object)
    labels[(groups == 0) & in_region] = "Liver"   # one study, inside region only

    d = enrichment_diagnostics(in_region, labels, groups,
                               values=["Liver"], n_boot=400)["Liver"]
    assert d["n_gse"] == 1
    # the study either is or is not resampled, so the interval must reach 0
    assert not (d["ci_low"] > 1.0)


def test_diagnostics_without_groups_reports_unknown_not_independent():
    groups, in_region = _panel(seed=10)
    rng = np.random.default_rng(11)
    labels = np.where(rng.random(groups.size) < 0.25, "Liver", "Other")

    d = enrichment_diagnostics(in_region, labels, None, values=["Liver"])["Liver"]
    assert d["n_gse"] is None
    assert np.isnan(d["rho"])
    assert np.isnan(d["ci_low"]) and np.isnan(d["ci_high"])
    assert d["n_eff_sel"] == pytest.approx(int(in_region.sum()))


def test_diagnostics_counts_match_a_plain_contingency_table():
    groups, in_region = _panel(seed=12)
    rng = np.random.default_rng(13)
    labels = np.where(rng.random(groups.size) < 0.3, "Liver", "Other")

    d = enrichment_diagnostics(in_region, labels, groups,
                               values=["Liver"], n_boot=0)["Liver"]
    hit = labels == "Liver"
    assert d["a"] == int((hit & in_region).sum())
    assert d["c"] == int((hit & ~in_region).sum())


def test_diagnostics_rejects_misaligned_inputs():
    with pytest.raises(ValueError):
        enrichment_diagnostics([True, False], ["a"], None)
    with pytest.raises(ValueError):
        enrichment_diagnostics([True, False], ["a", "b"], ["G1"])


def test_region_tab_flags_unreplicated_hits():
    """The enrichment tab must grey out hits no q-value can justify."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from genevariate.gui.region_analysis import _is_thin

    # replicated across many studies, CI clear of 1.0 -> a real finding
    assert not _is_thin({"n_gse": 25, "ci_low": 8.4, "ci_high": 104.4})
    # one study carries the whole signal
    assert _is_thin({"n_gse": 1, "ci_low": 5.0, "ci_high": 90.0})
    # many studies but resampling them still admits "no enrichment"
    assert _is_thin({"n_gse": 11, "ci_low": 0.6, "ci_high": 11.4})
    # no study information at all -> cannot claim it is thin
    assert not _is_thin({"n_gse": None, "ci_low": float("nan"),
                         "ci_high": float("nan")})
