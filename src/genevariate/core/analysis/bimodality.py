"""
Bimodality-gated enrichment — restrict pathway testing to the subset of
genes that GeneVariate's Distribution Classifier tags as bimodal or
heavy-tailed. Asks a question standard enrichment cannot:

    "Which pathways are driven by stochastic on/off switches (bimodal)
     rather than graded mean shifts?"

This is a reporting and filtering layer on top of the standard enrichment
pipelines (mean-based or ΔVariance). It does not re-invent enrichment —
it re-defines what the *gene universe* is before enrichment runs.

Typical use:
    tags = classify_distributions(df)   # per-gene Bimodal/Multimodal/Normal/...
    ranked = rank_genes_by_condition(df, labels, "case", "ctrl")
    gated  = filter_ranked_by_distribution(ranked, tags,
                                           keep=("Bimodal", "Multimodal"))
    gsea   = run_prerank_gsea(gated, gene_sets=[...])
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import (gaussian_kde, norm, lognorm, gamma as gamma_dist,
                         cauchy, uniform as uniform_dist)


# Bimodal/heavy-tailed tags produced by the classifier below
BIMODAL_TAGS: Tuple[str, ...] = ("Bimodal", "Multimodal")
HEAVY_TAGS: Tuple[str, ...]   = ("Cauchy", "Lognormal")


def classify_gene_distribution(values: np.ndarray) -> str:
    """
    Classify a 1-D array of expression values into one of:
        Bimodal, Multimodal, Normal, Lognormal, Gamma, Cauchy, Uniform,
        Effectively Constant, Not Enough Data.

    This mirrors `BioAI_Engine.analyze_gene_distribution` used by the GUI's
    Distribution Classification tool, re-exposed so the analysis layer can
    run without importing the Tk GUI.
    """
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size < 20:
        return "Not Enough Data"
    if np.std(vals) < 1e-6:
        return "Effectively Constant"

    try:
        kde = gaussian_kde(vals)
        grid = np.linspace(vals.min(), vals.max(), min(300, vals.size))
        pdf = kde(grid)
        peaks, _ = find_peaks(pdf, prominence=0.15 * pdf.max())
        if len(peaks) >= 3:
            return "Multimodal"
        if len(peaks) >= 2:
            p1, p2 = peaks[0], peaks[1]
            valley = pdf[p1:p2 + 1].min()
            peak_min = min(pdf[p1], pdf[p2])
            if valley < 0.75 * peak_min:
                return "Bimodal"
    except Exception:
        pass

    scores: Dict[str, float] = {}
    try:
        mu, sigma = norm.fit(vals)
        scores["Normal"] = float(np.sum(norm.logpdf(vals, mu, sigma)))
    except Exception:
        pass
    if vals.min() > 0:
        try:
            s, loc, scale = lognorm.fit(vals, floc=0)
            scores["Lognormal"] = float(np.sum(lognorm.logpdf(vals, s, loc, scale)))
        except Exception:
            pass
        try:
            a, loc, scale = gamma_dist.fit(vals, floc=0)
            scores["Gamma"] = float(np.sum(gamma_dist.logpdf(vals, a, loc, scale)))
        except Exception:
            pass
    try:
        loc, scale = cauchy.fit(vals)
        scores["Cauchy"] = float(np.sum(cauchy.logpdf(vals, loc, scale)))
    except Exception:
        pass
    try:
        loc, scale = uniform_dist.fit(vals)
        scores["Uniform"] = float(np.sum(uniform_dist.logpdf(vals, loc, scale)))
    except Exception:
        pass

    if not scores:
        return "Normal"
    return max(scores, key=scores.get)


def classify_distributions(df: pd.DataFrame,
                           subset: Optional[Iterable[str]] = None) -> pd.Series:
    """
    Classify every gene column in a GeneVariate canonical DataFrame.
    Returns a Series indexed by gene symbol with string tags.
    """
    meta = [c for c in ("GSM", "series_id") if c in df.columns]
    gene_cols = [c for c in df.columns if c not in meta]
    if subset is not None:
        subset = set(str(g).upper() for g in subset)
        gene_cols = [c for c in gene_cols if str(c).upper() in subset]

    tags = {}
    for g in gene_cols:
        tags[g] = classify_gene_distribution(df[g].values)
    return pd.Series(tags, name="distribution_tag")


def filter_ranked_by_distribution(ranked: pd.DataFrame,
                                  tags: pd.Series,
                                  keep: Sequence[str] = BIMODAL_TAGS) -> pd.DataFrame:
    """
    Keep only rows of `ranked` whose gene is tagged with one of `keep`.
    Tag matching is case-insensitive. Unknown genes are dropped.
    """
    keep_set = {str(k).lower() for k in keep}
    tag_lower = tags.astype(str).str.lower()
    allowed = set(tag_lower[tag_lower.isin(keep_set)].index)
    allowed_upper = {str(g).upper() for g in allowed}
    out = ranked.copy()
    out.index = out.index.astype(str).str.upper()
    out = out[out.index.isin(allowed_upper)]
    return out


def distribution_summary(tags: pd.Series) -> pd.DataFrame:
    """Return a count table of how many genes fall into each distribution class."""
    counts = tags.value_counts().to_frame(name="n_genes")
    counts["fraction"] = counts["n_genes"] / counts["n_genes"].sum()
    return counts.reset_index(names="distribution")
