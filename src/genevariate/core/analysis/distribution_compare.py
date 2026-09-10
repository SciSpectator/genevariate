"""How far apart two groups of expression values are, and whether that is real.

This is the arithmetic behind the Compare Distributions window's Distance Matrix
and Statistics tabs: per-group descriptives, a pairwise rank-sum test, and three
distances that answer three different questions about the same pair -

* **Wasserstein** - how much probability mass has to be moved, in expression
  units, to turn one distribution into the other. It is on the data's own scale,
  so it can be quoted as "about one log2 unit apart".
* **Delta-mean** - the difference of the two means. Cheap, familiar, and blind:
  two groups with identical means and opposite spreads score zero.
* **Jensen-Shannon** - a bounded (0-1) divergence between the two histograms. It
  sees shape - bimodality, a shifted tail - that neither of the other two do,
  but it is unitless and depends on the binning, so it ranks pairs rather than
  measuring them.

They are reported together because no one of them is sufficient: a pair that
scores high on all three is genuinely separated, and a pair that scores high on
only one is telling you which kind of difference it has.

The window computed all of this inline, next to the widgets that displayed it,
so the assistant could not answer "how different are these two groups?" without
a second implementation - and a second implementation is a second set of
numbers. It lives here now; the window keeps the heatmaps.
"""
from __future__ import annotations

import itertools
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ranksums, wasserstein_distance

from .enrichment import benjamini_hochberg

__all__ = ["group_summary", "pairwise_distances", "distance_matrix",
           "summarize_comparison_stats", "JS_BINS", "DISTANCE_METRICS"]

#: Bin count for the Jensen-Shannon histograms. Fixed rather than chosen per
#: pair: the divergence depends on the binning, so a matrix whose cells were
#: binned differently would not be comparable cell to cell.
JS_BINS = 50

DISTANCE_METRICS = ("wasserstein", "delta_mean", "jensen_shannon")


def _clean(values) -> np.ndarray:
    v = pd.Series(values, dtype="float64").to_numpy()
    return v[np.isfinite(v)]


def _js(a: np.ndarray, b: np.ndarray) -> float:
    """Jensen-Shannon divergence between two samples, over a shared grid.

    Both are histogrammed on the same edges spanning their union, because a
    divergence between histograms on different supports is not a divergence
    between the distributions.
    """
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0
    bins = np.linspace(lo, hi, JS_BINS)
    p, _ = np.histogram(a, bins=bins, density=True)
    q, _ = np.histogram(b, bins=bins, density=True)
    p = p / (p.sum() + 1e-10)
    q = q / (q.sum() + 1e-10)
    return float(jensenshannon(p, q))


def group_summary(groups: Mapping[str, Sequence[float]]) -> pd.DataFrame:
    """N, mean, median, SD and IQR for each group, largest first.

    Empty groups are dropped rather than reported with NaN statistics: a group
    the filter left with no samples is a fact about the filter, and the caller
    is told about it by its absence from the frame.
    """
    rows = []
    for name, values in groups.items():
        v = _clean(values)
        if v.size == 0:
            continue
        q1, q3 = np.percentile(v, [25, 75])
        rows.append({
            "group": name,
            "n": int(v.size),
            "mean": float(v.mean()),
            "median": float(np.median(v)),
            "sd": float(v.std(ddof=1)) if v.size > 1 else np.nan,
            "iqr": float(q3 - q1),
            "min": float(v.min()),
            "max": float(v.max()),
        })
    if not rows:
        return pd.DataFrame(columns=["group", "n", "mean", "median", "sd",
                                     "iqr", "min", "max"])
    return (pd.DataFrame(rows)
            .sort_values("n", ascending=False, kind="stable")
            .reset_index(drop=True))


def pairwise_distances(groups: Mapping[str, Sequence[float]],
                       *, min_n: int = 3) -> pd.DataFrame:
    """Every pair of groups: three distances, a rank-sum test and a BH q.

    The correction is applied across every pair in this call, for the same
    reason it is applied across the whole grid in :mod:`.region_enrichment`:
    with *k* groups there are *k(k-1)/2* comparisons, and at ten groups that is
    forty-five, of which two will clear p<0.05 on noise alone. Quoting the raw
    p of the best pair out of forty-five is the multiplicity error this column
    exists to prevent.

    Groups with fewer than *min_n* finite values are skipped: a rank-sum test on
    two points has no power and a Wasserstein distance from them is noise.
    """
    usable = {}
    for name, values in groups.items():
        v = _clean(values)
        if v.size >= min_n:
            usable[name] = v

    rows = []
    for a, b in itertools.combinations(usable, 2):
        va, vb = usable[a], usable[b]
        try:
            z, p = ranksums(va, vb)
        except ValueError:
            z, p = np.nan, np.nan
        rows.append({
            "group_a": a,
            "group_b": b,
            "n_a": int(va.size),
            "n_b": int(vb.size),
            "mean_a": float(va.mean()),
            "mean_b": float(vb.mean()),
            "delta_mean": float(abs(va.mean() - vb.mean())),
            "wasserstein": float(wasserstein_distance(va, vb)),
            "jensen_shannon": _js(va, vb),
            "rank_sum_z": float(z),
            "p_value": float(p),
        })
    if not rows:
        return pd.DataFrame(columns=["group_a", "group_b", "n_a", "n_b",
                                     "mean_a", "mean_b", "delta_mean",
                                     "wasserstein", "jensen_shannon",
                                     "rank_sum_z", "p_value", "q_value",
                                     "significance", "separation"])

    out = pd.DataFrame(rows)
    out["q_value"] = benjamini_hochberg(out["p_value"].to_numpy())
    out["significance"] = [
        "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else "ns"
        for q in out["q_value"]]
    # The window's own wording for the Wasserstein magnitude, kept verbatim so
    # a reader who saw "High" in the tab sees "high" here for the same number.
    out["separation"] = ["high" if w > 1 else "moderate" if w > 0.5 else "low"
                         for w in out["wasserstein"]]
    return (out.sort_values("wasserstein", ascending=False, kind="stable")
            .reset_index(drop=True))


def distance_matrix(pairs: pd.DataFrame, metric: str = "wasserstein",
                    groups: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """The square, symmetric form of one column of :func:`pairwise_distances`.

    Built from the pair table rather than recomputed, so the heatmap and the
    pair list can never disagree. The diagonal is zero.
    """
    if metric not in DISTANCE_METRICS:
        raise ValueError(f"unknown metric {metric!r}; "
                         f"expected one of {DISTANCE_METRICS}")
    if groups is None:
        names = list(dict.fromkeys(
            list(pairs.get("group_a", [])) + list(pairs.get("group_b", []))))
    else:
        names = list(groups)
    mat = pd.DataFrame(np.zeros((len(names), len(names))),
                       index=names, columns=names, dtype=float)
    for _, r in pairs.iterrows():
        a, b = r["group_a"], r["group_b"]
        if a in mat.index and b in mat.columns:
            mat.loc[a, b] = mat.loc[b, a] = float(r[metric])
    return mat


def summarize_comparison_stats(summary: pd.DataFrame, pairs: pd.DataFrame,
                               *, value_label: str = "expression",
                               top: int = 8) -> str:
    """Markdown for a group comparison, leading with what separates and what
    does not."""
    lines = []
    if summary.empty:
        return "No group had any usable values."

    lines.append("## The groups\n")
    lines.append(f"{len(summary)} group(s), "
                 f"{int(summary['n'].sum()):,} samples in total.\n")
    for _, r in summary.head(top).iterrows():
        lines.append(f"- **{r['group']}**: n={int(r['n']):,}, "
                     f"median {r['median']:.3g}, mean {r['mean']:.3g} "
                     f"(SD {r['sd']:.3g}, IQR {r['iqr']:.3g})")
    if len(summary) > top:
        lines.append(f"- +{len(summary) - top} more group(s)")

    if pairs.empty:
        lines.append("\nOnly one group had enough samples to describe, so "
                     "there is nothing to compare it against.")
        return "\n".join(lines)

    n_sig = int((pairs["significance"] != "ns").sum())
    lines.append("\n## What separates\n")
    lines.append(
        f"{len(pairs)} pair(s) compared; {n_sig} differ after "
        f"Benjamini-Hochberg across all {len(pairs)} of them. The q values "
        f"below are corrected over every pair in this comparison - a raw p "
        f"from the best of {len(pairs)} pairs would not mean what it looks "
        f"like it means.\n")
    for _, r in pairs.head(top).iterrows():
        lines.append(
            f"- **{r['group_a']}** vs **{r['group_b']}**: Wasserstein "
            f"{r['wasserstein']:.3g} {value_label} units ({r['separation']} "
            f"separation), Jensen-Shannon {r['jensen_shannon']:.3f}, "
            f"means {r['mean_a']:.3g} vs {r['mean_b']:.3g} "
            f"(n={int(r['n_a']):,}/{int(r['n_b']):,}); "
            f"rank-sum q={r['q_value']:.2e} {r['significance']}")
    if len(pairs) > top:
        lines.append(f"- +{len(pairs) - top} more pair(s)")

    # A significant p on thousands of samples can sit on a distance nobody
    # would call a difference. Saying so is the point of reporting both.
    small = pairs[(pairs["significance"] != "ns") & (pairs["wasserstein"] <= 0.5)]
    if not small.empty:
        lines.append(
            f"\n{len(small)} pair(s) are significant but barely separated "
            f"(Wasserstein <= 0.5): with these sample sizes the test detects "
            f"differences too small to act on. Read the distance, not the star.")
    return "\n".join(lines)
