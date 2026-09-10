"""
Bimodality-gated enrichment - restrict pathway testing to the subset of
genes that GeneVariate's Distribution Classifier tags as bimodal or
heavy-tailed. Asks a question standard enrichment cannot:

    "Which pathways are driven by stochastic on/off switches (bimodal)
     rather than graded mean shifts?"

This is a reporting and filtering layer on top of the standard enrichment
pipelines (mean-based or ΔVariance). It does not re-invent enrichment -
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
                         cauchy, uniform as uniform_dist, skew, kstest)

try:  # Hartigan's dip test - the standard rigorous unimodality test.
    from diptest import diptest as _diptest
    _HAS_DIPTEST = True
except Exception:  # pragma: no cover - exercised only when dep absent
    _diptest = None
    _HAS_DIPTEST = False


# Bimodal/heavy-tailed tags produced by the classifier below
BIMODAL_TAGS: Tuple[str, ...] = ("Bimodal", "Multimodal")
HEAVY_TAGS: Tuple[str, ...]   = ("Cauchy", "Lognormal")

# Reject unimodality when the dip-test p-value falls below this.
DIP_ALPHA: float = 0.05

#: How much closer to the empirical CDF a shape must sit than the Normal
#: before it is reported, as a fraction of the Normal's own distance.
GOF_MARGIN: float = 0.90

#: Smallest share of samples a mode must hold to be named one. A dropout spike
#: of a few percent is a detection floor, not a subpopulation the design can
#: support, and naming it a mode puts the gene in the switch universe.
MIN_MODE_SHARE: float = 0.10

#: How many standard errors of skewness the sample must show before a
#: skewed family may be named. Under normality skewness has standard error
#: sqrt(6/n), and a three-parameter family tracks noise below that for free.
SKEW_Z: float = 3.0

#: How far the standard deviation must exceed the one implied by the sample's
#: own interquartile range before tails with no finite variance are named. A
#: bounded gene sits near 1 and a Cauchy runs orders of magnitude above it.
TAIL_RATIO: float = 2.0


def optimal_bins(data, method: str = 'auto') -> int:
    """Freedman-Diaconis bin count, bounded by what the sample supports.

    The bounds have to scale with n. A fixed floor of 80 bins is right for
    a microarray corpus of tens of thousands of samples, but the RNA-seq
    and single-cell routes deliver whole platforms of a few hundred
    samples, where 80 bins leaves under two samples per bin: the histogram
    becomes a picket fence and the drag-to-select region is then drawn over
    noise rather than over a distribution. Zero-inflated data fails the
    other way -- a spike at zero collapses the IQR, so the raw rule asks
    for hundreds of bins on a handful of distinct values.

    Both bounds therefore key on n, and the ceiling keeps at least three
    samples per bin, which is what a drag-selected region needs in order to
    rest on a distribution rather than on noise. The ceiling reduces to the
    previous 250 for any n >= 750, and the floor reaches the previous 80 at
    n >= 6400, so the microarray corpora this was tuned on (tens of
    thousands of samples) bin exactly as before; only small platforms
    change.

    This lives here rather than beside the window that first used it because
    the assistant has to draw the same histogram the user sees. The chart
    layer is deliberately Tk-free, so a bin rule kept in the GUI module can
    only be copied into it, and a copied rule is one that drifts: the two
    histograms would then disagree about the same numbers. One rule, imported
    by both, is what lets the assistant's figure be checked against the
    button that produces it.
    """
    arr = np.asarray(data, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n < 10:
        return max(10, n)
    q25, q75 = np.percentile(arr, [25, 75])
    iqr = q75 - q25
    fd_bins = 50
    if iqr > 0:
        bin_width = 2.0 * iqr / (n ** (1.0 / 3.0))
        data_range = arr.max() - arr.min()
        if data_range > 0 and bin_width > 0:
            fd_bins = int(np.ceil(data_range / bin_width))
    floor = int(min(80, max(10, np.sqrt(n))))
    ceiling = int(min(250, max(10, n // 3)))
    return max(floor, min(fd_bins, ceiling))


def robust_kde(values: np.ndarray) -> gaussian_kde:
    """Kernel density for *values*, with a bandwidth set by a robust scale.

    Scott's rule, which is what ``gaussian_kde`` uses when asked for nothing,
    scales the kernel by the standard deviation. A gene with a point mass of
    undetected samples at zero and an expressed hump far above it has a
    standard deviation set by the distance *between* the two states rather than
    by the width of either, so the kernel comes out wider than the structure it
    is meant to resolve and both states smear into one flat curve. Silverman's
    robust scale, the smaller of the standard deviation and the interquartile
    range's normal-equivalent, is set by the narrower state instead and leaves
    the two humps standing.

    The same estimator draws the curve on the histogram and counts the modes
    behind the label, so what is plotted is what was classified.
    """
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    sd = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
    q75, q25 = np.percentile(vals, [75, 25])
    iqr_scale = float(q75 - q25) / 1.349
    scale = min(sd, iqr_scale) if iqr_scale > 0 else sd
    if not np.isfinite(scale) or scale <= 0 or sd <= 0:
        return gaussian_kde(vals)
    # ``bw_method`` is a factor on the sample's own standard deviation, so the
    # bandwidth wanted has to be expressed relative to it.
    return gaussian_kde(vals, bw_method=0.9 * (scale / sd) * vals.size ** -0.2)


def density_grid(vals: np.ndarray, n: int = 512) -> np.ndarray:
    """Evaluation grid padded past the data range.

    Density peaks for the outermost modes sit at the extreme observations, and
    ``find_peaks`` never reports a boundary point, so an unpadded grid silently
    misses the very peaks that define a bimodal gene.
    """
    span = float(vals.max() - vals.min())
    pad = 0.10 * span if span > 0 else 1.0
    return np.linspace(vals.min() - pad, vals.max() + pad, n)


def _fit_gof(dist, vals, sample_skew=None):
    """KS distance from *vals* to the best fit of *dist*, None if it leans the wrong way."""
    try:
        params = dist.fit(vals)
        stat = float(kstest(vals, dist.cdf, args=params).statistic)
    except Exception:
        return None
    if not np.isfinite(stat):
        return None
    if not dist.shapes or sample_skew is None:
        return stat
    if abs(sample_skew) < SKEW_Z * np.sqrt(6.0 / vals.size):
        return None
    try:
        fitted_skew = float(dist.stats(*params, moments="s"))
    except Exception:
        return None
    if not np.isfinite(fitted_skew) or fitted_skew * sample_skew < 0.0:
        return None
    return stat


def _mode_shares(pdf: np.ndarray, grid: np.ndarray, vals: np.ndarray,
                 kept: Sequence[int]) -> np.ndarray:
    """Fraction of *vals* falling in each peak's basin, split at the valleys."""
    bounds = [float(grid[a + int(np.argmin(pdf[a:b + 1]))])
              for a, b in zip(kept, kept[1:])]
    idx = np.searchsorted(np.asarray(bounds), vals, side="right")
    return np.bincount(idx, minlength=len(kept)) / float(vals.size)


def _kept_peaks(pdf: np.ndarray, depth: float = 0.75,
                grid: Optional[np.ndarray] = None,
                vals: Optional[np.ndarray] = None,
                min_share: float = MIN_MODE_SHARE) -> List[int]:
    """Local maxima of *pdf*, merging any pair not separated by a real valley.

    A peak is kept only when the density between it and the previous one falls
    below *depth* of the smaller of the two, so a shoulder on one hump is not
    counted as a second mode. The comparison is against the smaller peak rather
    than the tallest one anywhere in the density: read counts put a spike of
    undetected samples at zero whose density is orders of magnitude above the
    expressed hump, and a threshold scaled to the global maximum would erase
    the expressed hump entirely.

    A valley says two peaks are separated; it does not say both are populated.
    When *grid* and *vals* are given, a surviving peak must also hold
    *min_share* of the samples, so a detection-floor spike is not a mode, and
    neither is a wobble in the tail of a kernel density.
    """
    peaks, _ = find_peaks(pdf)
    if peaks.size == 0:
        return [int(np.argmax(pdf))]
    kept = [int(peaks[0])]
    for p in peaks[1:]:
        p = int(p)
        prev = kept[-1]
        valley = pdf[prev:p + 1].min()
        if valley < depth * min(pdf[prev], pdf[p]):
            kept.append(p)
        elif pdf[p] > pdf[prev]:
            kept[-1] = p

    if grid is None or vals is None:
        return kept
    while len(kept) > 1:
        shares = _mode_shares(pdf, grid, vals, kept)
        weakest = int(np.argmin(shares))
        if shares[weakest] >= min_share:
            break
        kept.pop(weakest)
    return kept


def _count_density_modes(pdf: np.ndarray, depth: float = 0.75,
                         grid: Optional[np.ndarray] = None,
                         vals: Optional[np.ndarray] = None,
                         min_share: float = MIN_MODE_SHARE) -> int:
    """How many populated modes :func:`_kept_peaks` leaves standing."""
    return len(_kept_peaks(pdf, depth, grid, vals, min_share))


def density_modes(vals: np.ndarray) -> List[Tuple[float, float]]:
    """Where each populated mode of *vals* sits, and what share of the samples
    it holds, as ``[(location, share), ...]`` ordered along the axis.

    The classifier reduces this to a count and then to a name. A reader given
    only the name cannot check it against the picture, and on a gene whose two
    states differ in density by an order of magnitude -- an off state held by
    most samples and an on state held by a large minority -- the minor mode is
    drawn flat against the axis and the plot appears to contradict its own
    label. Returning the modes themselves lets the figure show what was
    counted, using the same density that did the counting.
    """
    grid = density_grid(vals)
    pdf = robust_kde(vals)(grid)
    kept = _kept_peaks(pdf, grid=grid, vals=vals)
    shares = _mode_shares(pdf, grid, vals, kept)
    return [(float(grid[i]), float(s)) for i, s in zip(kept, shares)]


def n_density_modes(vals: np.ndarray) -> int:
    """Populated modes in the density of *vals*.

    A Gaussian mixture chosen by BIC used to answer this. It cannot: BIC buys
    whatever components lower the likelihood, and on a zero-inflated gene the
    cheapest purchase is one broad filler Gaussian spread across the sparse
    range between the off state and the on state. That component plants a
    shallow local maximum where the histogram shows nothing, and the gene is
    named multimodal on the strength of a hump nobody can see. The density the
    curve is drawn from answers instead, so the label and the picture cannot
    disagree.
    """
    grid = density_grid(vals)
    return _count_density_modes(robust_kde(vals)(grid), grid=grid, vals=vals)


def _diptest_class(vals: np.ndarray) -> Optional[str]:
    """``Bimodal``/``Multimodal`` when the dip test rejects one hump AND the
    mixture density shows that many populated modes, else ``None``.

    Rejecting unimodality says the departure is unlikely under one hump; it
    does not say how many populated humps replace it. Returning ``None`` when
    the density holds a single mode lets a gene the dip flags -- a skew, or a
    detection-floor spike too small to be a subpopulation -- still reach the
    shape families, which is the only way a large platform can report anything
    other than Bimodal.
    """
    if not _HAS_DIPTEST:
        return None
    try:
        _, pval = _diptest(vals)
    except Exception:
        return None
    if pval >= DIP_ALPHA:
        return None
    n_modes = n_density_modes(vals)
    if n_modes >= 3:
        return "Multimodal"
    if n_modes == 2:
        return "Bimodal"
    return None


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

    # Preferred gate: Hartigan's dip test (rigorous unimodality test) + GMM/BIC
    # to count modes. Falls through to the KDE peak heuristic when diptest is
    # not installed, preserving the original behaviour.
    dip = _diptest_class(vals)
    if dip is not None:
        return dip

    # Without diptest installed there is no unimodality test to gate on, so the
    # density is asked directly. A prominence screen scaled to the tallest peak
    # anywhere used to run first; on a gene whose zero spike is orders of
    # magnitude denser than its expressed hump that screen deletes the hump,
    # which is the one thing the classifier exists to find. Separation is
    # already judged against the smaller of each pair of peaks, and a peak too
    # sparsely populated to be a subpopulation is already dropped by share.
    try:
        grid = density_grid(vals)
        n_modes = _count_density_modes(robust_kde(vals)(grid),
                                       grid=grid, vals=vals)
        if n_modes >= 3:
            return "Multimodal"
        if n_modes == 2:
            return "Bimodal"
    except Exception:
        pass

    sample_skew = float(skew(vals))
    q75, q25 = np.percentile(vals, [75, 25])
    iqr = float(q75 - q25)
    families = [("Normal", norm), ("Lognormal", lognorm),
                ("Gamma", gamma_dist), ("Uniform", uniform_dist)]
    if iqr <= 0 or float(np.std(vals)) / (iqr / 1.349) >= TAIL_RATIO:
        families.append(("Cauchy", cauchy))

    scores: Dict[str, float] = {}
    for name, dist in families:
        stat = _fit_gof(dist, vals, sample_skew)
        if stat is not None:
            scores[name] = stat

    if "Normal" not in scores:
        return min(scores, key=scores.get) if scores else "Normal"
    best = min(scores, key=scores.get)
    if scores[best] > GOF_MARGIN * scores["Normal"]:
        return "Normal"
    return best


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
