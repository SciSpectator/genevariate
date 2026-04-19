"""
Shared pytest fixtures for GeneVariate unit tests.

Synthetic expression matrices live here so individual test modules can
import them without duplicating the data-generation boilerplate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


RNG_SEED = 20260419  # frozen for determinism


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(RNG_SEED)


@pytest.fixture(scope="session")
def tiny_expression_df():
    """
    40 samples × 50 genes, with an embedded case/control contrast:

    - 20 case samples, 20 control samples.
    - Genes 0..4  — differential MEAN only  (case +2 SD)
    - Genes 5..9  — differential VARIANCE only (case 3× variance)
    - Genes 10..49 — pure noise, no group signal.

    Returns (df_canonical, labels_series). df_canonical has the canonical
    columns GSM | series_id | gene00..gene49.
    """
    rng = np.random.default_rng(RNG_SEED)
    n_per = 30
    n_genes = 50
    gene_cols = [f"gene{i:02d}" for i in range(n_genes)]

    ctrl = rng.normal(loc=0.0, scale=1.0, size=(n_per, n_genes))
    case = rng.normal(loc=0.0, scale=1.0, size=(n_per, n_genes))

    # Mean shift on genes 0..4 (strong signal)
    case[:, 0:5] += 3.0

    # Variance inflation on genes 5..9 (6× variance; big-enough to clear n=30 noise)
    case[:, 5:10] = rng.normal(loc=0.0, scale=np.sqrt(6.0),
                               size=(n_per, 5))

    X = np.vstack([ctrl, case])
    gsms = [f"GSM{i:04d}" for i in range(2 * n_per)]
    df = pd.DataFrame(X, columns=gene_cols)
    df.insert(0, "series_id", "GSE_TEST")
    df.insert(0, "GSM", gsms)

    labels = pd.Series(
        ["control"] * n_per + ["case"] * n_per,
        index=gsms,
        name="condition",
    )
    return df, labels


@pytest.fixture(scope="session")
def bimodal_expression_df():
    """
    300 samples × 3 genes — one bimodal, one unimodal-normal, one lognormal.
    Size chosen so the KDE peak-finder can resolve the two modes reliably.
    """
    rng = np.random.default_rng(RNG_SEED + 1)
    n = 300
    half = n // 2
    g_normal = rng.normal(0.0, 1.0, size=n)
    g_bimodal = np.concatenate([rng.normal(-5.0, 0.5, size=half),
                                rng.normal(5.0, 0.5, size=half)])
    rng.shuffle(g_bimodal)
    g_lognormal = rng.lognormal(0.0, 0.6, size=n)

    gsms = [f"GSM{i:04d}" for i in range(n)]
    df = pd.DataFrame({
        "GSM": gsms,
        "series_id": "GSE_BIMODAL",
        "gNormal": g_normal,
        "gBimodal": g_bimodal,
        "gLognormal": g_lognormal,
    })
    return df
