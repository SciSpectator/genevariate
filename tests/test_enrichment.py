"""
Tests for the mean-contrast gene ranker (`rank_genes_by_condition`), covering
both the Welch t-stat path and the empirical-Bayes moderated-t path.
"""

from __future__ import annotations

import numpy as np
import pytest

from genevariate.core.analysis.enrichment import rank_genes_by_condition


def test_rank_genes_by_condition_welch_signs(tiny_expression_df):
    """Welch: true up-regulated genes in case must rank with positive stat."""
    df, labels = tiny_expression_df
    out = rank_genes_by_condition(
        df=df,
        labels=labels,
        case_label="case",
        control_label="control",
        moderated=False,
    )
    # Top-5 by t_stat should include the mean-shift genes (0..4)
    top = out.sort_values("t_stat", ascending=False).head(5).index.tolist()
    signal = {f"gene{i:02d}" for i in range(5)}
    assert len(signal & set(top)) >= 4, (
        f"Welch recovered only {len(signal & set(top))}/5 mean-shift genes in top-5 — {top}"
    )


def test_moderated_t_recovers_signal(tiny_expression_df):
    """Moderated-t must also recover the true mean-shift genes."""
    df, labels = tiny_expression_df
    out = rank_genes_by_condition(
        df=df,
        labels=labels,
        case_label="case",
        control_label="control",
        moderated=True,
    )
    top = out.sort_values("t_stat", ascending=False).head(5).index.tolist()
    signal = {f"gene{i:02d}" for i in range(5)}
    assert len(signal & set(top)) >= 4


def test_moderated_pvalue_column_exists(tiny_expression_df):
    df, labels = tiny_expression_df
    out = rank_genes_by_condition(
        df=df,
        labels=labels,
        case_label="case",
        control_label="control",
        moderated=True,
    )
    assert "p_value" in out.columns
    pv = out["p_value"].dropna().values
    assert len(pv) > 0
    assert ((pv >= 0) & (pv <= 1)).all()


def test_shuffle_null_mean_pvalue_welch(tiny_expression_df):
    """Random labels → p-values roughly uniform under the Welch test."""
    df, labels = tiny_expression_df
    shuf = labels.sample(frac=1.0, random_state=77).values
    shuffled = labels.copy(); shuffled[:] = shuf
    out = rank_genes_by_condition(
        df=df, labels=shuffled, case_label="case", control_label="control",
        moderated=False,
    )
    pv = out["p_value"].dropna().values
    assert 0.3 < float(pv.mean()) < 0.7


def test_shuffle_null_mean_pvalue_moderated(tiny_expression_df):
    """Random labels → p-values roughly uniform under moderated-t."""
    df, labels = tiny_expression_df
    shuf = labels.sample(frac=1.0, random_state=91).values
    shuffled = labels.copy(); shuffled[:] = shuf
    out = rank_genes_by_condition(
        df=df, labels=shuffled, case_label="case", control_label="control",
        moderated=True,
    )
    pv = out["p_value"].dropna().values
    assert 0.3 < float(pv.mean()) < 0.7


def test_moderated_vs_welch_on_small_sample_more_stable():
    """
    EB shrinkage should make the moderated-t more robust on tiny samples
    (n=4 per group): Welch blows up for a few genes where the within-group
    variance is near-zero; the moderated test pulls those toward the prior.

    Pass criterion: max |stat| under moderated ≤ max |stat| under Welch.
    """
    rng = np.random.default_rng(20260419)
    n = 4
    g = 30
    ctrl = rng.normal(0, 1, size=(n, g))
    case = rng.normal(0, 1, size=(n, g))
    case[:, 0:3] += 2.0  # real signal on a handful
    import pandas as pd
    gsms = [f"GSM{i}" for i in range(2 * n)]
    df = pd.DataFrame(np.vstack([ctrl, case]),
                      columns=[f"g{i:02d}" for i in range(g)])
    df.insert(0, "series_id", "T"); df.insert(0, "GSM", gsms)
    labels = pd.Series(["control"] * n + ["case"] * n, index=gsms)

    welch = rank_genes_by_condition(df, labels, "case", "control", moderated=False)
    mod = rank_genes_by_condition(df, labels, "case", "control", moderated=True)
    assert mod["t_stat"].abs().max() <= welch["t_stat"].abs().max() + 1e-6, (
        "Moderated-t should not exceed Welch's max |t_stat| (EB shrinks the tail)."
    )
