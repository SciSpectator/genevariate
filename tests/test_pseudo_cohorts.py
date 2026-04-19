"""
Tests for embedding-clustered pseudo-cohort discovery.

These run against the TF-IDF fallback backend so they don't require a running
Ollama server. The tests verify:
  - deterministic output for a fixed seed + backend,
  - correct cluster count recovery on clean synthetic labels,
  - bootstrap Jaccard stability yields high scores on real clusters and
    low scores on random-noise labels,
  - cohort_pairs enforces the stability floor.
"""

from __future__ import annotations

import random
import string

import numpy as np
import pytest

from genevariate.core.analysis.pseudo_cohorts import (
    discover_pseudo_cohorts,
    cohort_pairs,
    cohort_summary,
)


def _three_cluster_labels():
    labels = {}
    for i in range(8):
        labels[f"GSM{i:03d}"] = f"MCF7 tamoxifen 24h rep{i}"
    for i in range(8, 16):
        labels[f"GSM{i:03d}"] = f"MCF7 DMSO control 24h rep{i}"
    for i in range(16, 24):
        labels[f"GSM{i:03d}"] = f"A549 cisplatin 48h rep{i}"
    return labels


def test_discover_three_clusters_tfidf_deterministic():
    labels = _three_cluster_labels()
    res_a = discover_pseudo_cohorts(labels, k_range=(2, 5),
                                    random_state=0, prefer_backend="tfidf",
                                    n_bootstrap=0)
    res_b = discover_pseudo_cohorts(labels, k_range=(2, 5),
                                    random_state=0, prefer_backend="tfidf",
                                    n_bootstrap=0)
    assert res_a.cluster_of == res_b.cluster_of
    assert res_a.k_selected == res_b.k_selected
    assert res_a.k_selected == 3


def test_bootstrap_jaccard_high_for_true_clusters():
    labels = _three_cluster_labels()
    res = discover_pseudo_cohorts(labels, k_range=(2, 5),
                                  random_state=0, prefer_backend="tfidf",
                                  n_bootstrap=25)
    assert res.cluster_stability is not None
    assert all(v >= 0.8 for v in res.cluster_stability.values()), (
        f"Well-separated clusters should have Jaccard >= 0.8: {res.cluster_stability}"
    )


def test_bootstrap_jaccard_low_for_random_labels():
    random.seed(42)
    labels = {f"GSM{i:03d}": "".join(random.choices(string.ascii_lowercase, k=20))
              for i in range(20)}
    res = discover_pseudo_cohorts(labels, k_range=(2, 5),
                                  random_state=0, prefer_backend="tfidf",
                                  n_bootstrap=25)
    assert res.cluster_stability is not None
    # At least one cluster should be below 0.75 on pure random labels.
    assert min(res.cluster_stability.values()) < 0.75, (
        f"Random-noise labels should produce at least one unstable cluster: "
        f"{res.cluster_stability}"
    )


def test_cohort_pairs_respects_stability_floor():
    """Force an unstable cluster and ensure it's dropped from returned pairs."""
    labels = _three_cluster_labels()
    res = discover_pseudo_cohorts(labels, k_range=(3, 3),
                                  random_state=0, prefer_backend="tfidf",
                                  n_bootstrap=10)
    # Manually demote cluster 0's stability
    res.cluster_stability = {**(res.cluster_stability or {}), 0: 0.10}
    pairs_with_floor = cohort_pairs(res, min_size=3, enforce_stability=True)
    pairs_no_floor = cohort_pairs(res, min_size=3, enforce_stability=False)
    assert all(0 not in p for p in pairs_with_floor)
    assert any(0 in p for p in pairs_no_floor)


def test_cohort_summary_has_stability_columns():
    labels = _three_cluster_labels()
    res = discover_pseudo_cohorts(labels, k_range=(2, 5),
                                  random_state=0, prefer_backend="tfidf",
                                  n_bootstrap=10)
    summary = cohort_summary(res)
    assert "bootstrap_jaccard" in summary.columns
    assert "stable" in summary.columns
    assert summary["stable"].all()
