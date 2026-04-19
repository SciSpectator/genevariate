"""
Tests for the distribution classifier.

Synthetic distributions should be classified correctly at high fidelity:
  - A clean two-component Gaussian mixture → 'Bimodal' (or 'Multimodal')
  - A single-component Gaussian           → 'Normal'
  - A positive lognormal                  → 'Lognormal' (or other heavy-tailed tag)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from genevariate.core.analysis.bimodality import (
    classify_gene_distribution,
    classify_distributions,
    filter_ranked_by_distribution,
    distribution_summary,
    BIMODAL_TAGS,
)


def test_classify_bimodal_sample():
    rng = np.random.default_rng(10)
    # Two well-separated modes, n=200 (classifier needs enough density per mode)
    vals = np.concatenate([rng.normal(-5, 0.5, 200), rng.normal(5, 0.5, 200)])
    tag = classify_gene_distribution(vals)
    assert tag in BIMODAL_TAGS, f"expected Bimodal/Multimodal, got {tag!r}"


def test_classify_normal_sample():
    rng = np.random.default_rng(11)
    vals = rng.normal(0, 1, 200)
    tag = classify_gene_distribution(vals)
    assert tag == "Normal"


def test_classify_lognormal_sample():
    rng = np.random.default_rng(12)
    vals = rng.lognormal(0.0, 0.8, 200)
    tag = classify_gene_distribution(vals)
    # We don't pin to exactly 'Lognormal' — as long as the classifier doesn't
    # call it Normal (heavy tail is the key thing to recognise).
    assert tag != "Normal"


def test_classify_not_enough_data():
    assert classify_gene_distribution(np.array([1.0, 2.0, 3.0])) == "Not Enough Data"


def test_classify_effectively_constant():
    assert classify_gene_distribution(np.ones(100)) == "Effectively Constant"


def test_classify_distributions_on_df(bimodal_expression_df):
    tags = classify_distributions(bimodal_expression_df)
    assert "gNormal" in tags.index
    assert tags["gNormal"] == "Normal"
    assert tags["gBimodal"] in BIMODAL_TAGS


def test_filter_ranked_by_distribution(bimodal_expression_df):
    tags = classify_distributions(bimodal_expression_df)
    ranked = pd.DataFrame({"rank": [1.0, 2.0, 3.0]},
                          index=["gNormal", "gBimodal", "gLognormal"])
    gated = filter_ranked_by_distribution(ranked, tags, keep=BIMODAL_TAGS)
    # Only bimodal rows should remain after bimodal filtering
    assert "GBIMODAL" in [i.upper() for i in gated.index]
    assert "GNORMAL" not in [i.upper() for i in gated.index]


def test_distribution_summary_sums_to_one(bimodal_expression_df):
    tags = classify_distributions(bimodal_expression_df)
    summary = distribution_summary(tags)
    assert np.isclose(summary["fraction"].sum(), 1.0)
