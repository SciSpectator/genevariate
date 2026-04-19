"""
Tests for ΔVariance ranking (log-variance z, Brown-Forsythe, Levene, KS,
Wasserstein, log-variance ratio).

Two critical guarantees:

1. Label-shuffling null: under random labels the distribution of p-values
   must be near-uniform (mean close to 0.5) — i.e. no spurious signal.
2. Signal recovery: genes with true variance inflation rank above pure-
   noise genes for every supported method.
"""

from __future__ import annotations

import numpy as np
import pytest

from genevariate.core.analysis.variability import (
    rank_genes_by_variability,
    SUPPORTED_METHODS,
    RECOMMENDED_METHOD,
)


# Methods that expose an analytical p-value column.
P_VALUE_METHODS = ["logvar_z", "bf", "levene", "ks"]


# -------------------------------------------------------------------
# Null behaviour under label shuffling
# -------------------------------------------------------------------
@pytest.mark.parametrize("method", P_VALUE_METHODS)
def test_label_shuffle_null_pvalues_uniform(tiny_expression_df, method):
    """Shuffled labels must yield approximately uniform p-values (mean ≈ 0.5)."""
    df, labels = tiny_expression_df
    rng = np.random.default_rng(123)

    shuffled = labels.sample(frac=1.0, random_state=int(rng.integers(1, 1e9))).values
    shuffled_series = labels.copy()
    shuffled_series[:] = shuffled

    out = rank_genes_by_variability(
        df=df,
        labels=shuffled_series,
        case_label="case",
        control_label="control",
        method=method,
    )
    assert "p_value" in out.columns
    p = out["p_value"].dropna().values
    assert len(p) >= 30
    assert 0.3 < float(np.mean(p)) < 0.7, (
        f"{method}: shuffled-label p-value mean {np.mean(p):.3f} not near 0.5 — "
        "possible statistical bias"
    )


# KS is a shape test, not a variance test — it also picks up mean shifts.
# So for KS we compare against the *full* signal set (variance + mean genes).
VARIANCE_SPECIFIC = {"logvar_z", "bf", "levene", "logvar_ratio"}


@pytest.mark.parametrize("method", SUPPORTED_METHODS)
def test_signal_recovery_variance_genes_rank_first(tiny_expression_df, method):
    """Genes 5..9 (true variance signal) should top the ranking."""
    df, labels = tiny_expression_df
    out = rank_genes_by_variability(
        df=df,
        labels=labels,
        case_label="case",
        control_label="control",
        method=method,
    )
    out = out.copy()
    out["abs_stat"] = out["stat"].abs()
    top10 = set(out.sort_values("abs_stat", ascending=False).head(10).index.tolist())

    if method in VARIANCE_SPECIFIC or method == "wasserstein":
        true_signal = {f"gene{i:02d}" for i in range(5, 10)}
    else:
        # KS is distribution-shape — picks up mean AND variance differences.
        true_signal = {f"gene{i:02d}" for i in range(10)}

    recovered = len(true_signal & top10)
    assert recovered >= 4, (
        f"{method}: only {recovered} true signal genes in top-10 — {sorted(top10)}"
    )


# -------------------------------------------------------------------
# log-variance z specific: normal null + signedness
# -------------------------------------------------------------------
def test_logvar_z_is_directionally_signed(tiny_expression_df):
    """Case-inflated genes should have positive stat under logvar_z."""
    df, labels = tiny_expression_df
    out = rank_genes_by_variability(
        df=df,
        labels=labels,
        case_label="case",
        control_label="control",
        method="logvar_z",
    )
    signal_genes = [f"gene{i:02d}" for i in range(5, 10)]
    stats_signal = out.loc[signal_genes, "stat"]
    # Majority should be positive — sampling noise can flip a gene or two at n=30
    # even when the true variance ratio is 6:1.
    assert (stats_signal > 0).sum() >= 4, (
        f"logvar_z: expected ≥4/5 signal genes positive; got: {stats_signal.to_dict()}"
    )


def test_recommended_method_is_logvar_z():
    """If the recommended method changes we want tests to fail loudly."""
    assert RECOMMENDED_METHOD == "logvar_z"
