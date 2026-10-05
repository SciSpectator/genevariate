"""Bimodality-gated enrichment - restrict pathway testing to bimodal or heavy-tailed genes."""

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

GOF_MARGIN: float = 0.90

MIN_MODE_SHARE: float = 0.10

SKEW_Z: float = 3.0

TAIL_RATIO: float = 2.0


def optimal_bins(data, method: str = 'auto') -> int:
    """Freedman-Diaconis bin count, bounded by what the sample supports."""
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
    """Kernel density for *values*, with a bandwidth set by a robust scale."""
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    sd = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
    q75, q25 = np.percentile(vals, [75, 25])
    iqr_scale = float(q75 - q25) / 1.349
    scale = min(sd, iqr_scale) if iqr_scale > 0 else sd
    if not np.isfinite(scale) or scale <= 0 or sd <= 0:
        return gaussian_kde(vals)
    return gaussian_kde(vals, bw_method=0.9 * (scale / sd) * vals.size ** -0.2)


def density_grid(vals: np.ndarray, n: int = 512) -> np.ndarray:
    """Evaluation grid padded past the data range."""
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
    """Local maxima of *pdf*, merging any pair not separated by a real valley."""
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
    """Each populated mode of *vals* as ``[(location, share), ...]`` along the axis."""
    grid = density_grid(vals)
    pdf = robust_kde(vals)(grid)
    kept = _kept_peaks(pdf, grid=grid, vals=vals)
    shares = _mode_shares(pdf, grid, vals, kept)
    return [(float(grid[i]), float(s)) for i, s in zip(kept, shares)]


def n_density_modes(vals: np.ndarray) -> int:
    """Populated modes in the density of *vals*."""
    grid = density_grid(vals)
    return _count_density_modes(robust_kde(vals)(grid), grid=grid, vals=vals)


def _diptest_class(vals: np.ndarray) -> Optional[str]:
    """``Bimodal``/``Multimodal`` when the dip test rejects one hump, else ``None``."""
    if not _HAS_DIPTEST:
        raise RuntimeError(
            "diptest is required to classify a distribution as bimodal or "
            "multimodal and is not importable. Install it with "
            "`pip install diptest`; it is a declared dependency of this "
            "package.")
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
    """One of Bimodal, Multimodal, Normal, Lognormal, Gamma, Cauchy, Uniform, Effectively Constant, Not Enough Data."""
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size < 20:
        return "Not Enough Data"
    if np.std(vals) < 1e-6:
        return "Effectively Constant"

    # Hartigan's dip test decides whether the gene has more than one mode.
    dip = _diptest_class(vals)
    if dip is not None:
        return dip

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
    """Returns a Series indexed by gene symbol with string tags."""
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
    """Keep only rows of `ranked` whose gene is tagged with one of `keep`."""
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
