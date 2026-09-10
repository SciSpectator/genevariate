"""How a high-expression region is defined, in one place.

The Gene Distribution Explorer defines a region by dragging a rectangle across
the histogram, so what the window actually holds is a pair of numbers on the
expression axis. Everything downstream -- the enrichment table, the box model,
the comparison grid -- only ever needs that pair.

The assistant had no way to say it. Every region tool took a ``quantile`` and
nothing else, so a user who had brushed 3.828-7.848 in the window, or who asked
for the mean+3SD tail their earlier work was built on, could not restate either
one in a sentence: the request was silently answered at the tool's own default
quantile instead. A region rule that the window can express and the assistant
cannot is a rule the two halves of the program disagree about.

So a region is described here by :class:`RegionBounds` -- a low, a high and the
rule that produced them -- and the three ways of arriving at one are all
supported:

* **explicit** ``low``/``high``: the numbers a brush would have produced;
* **standard deviations** ``sd=k``: the mean+k*SD tail;
* **quantile** ``quantile=q``: the gene's own upper quantile.

The rule string travels with the bounds so a figure or a report can say which
of the three it was, rather than printing two numbers whose provenance the
reader has to guess.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

__all__ = ["RegionBounds", "resolve_bounds", "region_mask"]


@dataclass(frozen=True)
class RegionBounds:
    """A closed interval on the expression axis, and how it was chosen.

    ``low``/``high`` are in the platform's own measurement units, never a
    fraction: a region is compared against a distribution, and a fraction has
    no meaning once it leaves the gene it was computed on.
    """

    low: float
    high: float
    rule: str

    @property
    def empty(self) -> bool:
        """True when the rule placed the cut past the data it was given.

        A mean+3SD tail on a distribution whose maximum sits nearer the mean
        than that is a real answer -- the gene has no such tail -- and it must
        not be quietly widened until it catches something.
        """
        return self.low > self.high

    def describe(self) -> str:
        if self.empty:
            return (f"{self.rule}: cut at {self.low:.4g}, above the observed "
                    f"maximum {self.high:.4g} - no samples qualify")
        return f"{self.low:.4g} to {self.high:.4g} ({self.rule})"


def _finite(values: Sequence[float]) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    return v[np.isfinite(v)]


def resolve_bounds(values: Sequence[float], *,
                   low: Optional[float] = None,
                   high: Optional[float] = None,
                   sd: Optional[float] = None,
                   quantile: Optional[float] = None,
                   default_quantile: float = 0.8) -> RegionBounds:
    """Turn whichever region rule the caller gave into a concrete interval.

    Precedence is explicit bounds, then ``sd``, then ``quantile``, then the
    default quantile: a caller who names actual numbers means them, and should
    not have them overridden by a quantile that was only ever a fallback.

    A bound left open runs to the data's own edge, so ``low=3.828`` alone is
    "3.828 upwards" and ``sd=3`` is the upper tail rather than a band. Passing
    an ``sd`` or a ``quantile`` that the sample cannot support (an empty or
    constant vector) raises, instead of returning an interval that would select
    everything or nothing without saying so.
    """
    v = _finite(values)
    if v.size == 0:
        raise ValueError("no finite values to define a region on")
    vmin, vmax = float(v.min()), float(v.max())

    if low is not None or high is not None:
        lo = vmin if low is None else float(low)
        hi = vmax if high is None else float(high)
        if hi < lo:
            lo, hi = hi, lo
        return RegionBounds(lo, hi, "explicit bounds")

    if sd is not None:
        k = float(sd)
        std = float(v.std(ddof=1)) if v.size > 1 else 0.0
        if not np.isfinite(std) or std <= 0:
            raise ValueError(
                "the values have no spread, so a standard-deviation tail "
                "cannot be placed on them")
        cut = float(v.mean()) + k * std
        return RegionBounds(cut, vmax, f"mean {'+' if k >= 0 else '-'} "
                                       f"{abs(k):g} SD")

    q = default_quantile if quantile is None else float(quantile)
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"quantile must lie in [0, 1], got {q}")
    return RegionBounds(float(np.quantile(v, q)), vmax, f"quantile {q:g}")


def region_mask(values: Sequence[float], bounds: RegionBounds) -> np.ndarray:
    """Membership of ``values`` in ``bounds``, closed at both ends.

    Closed rather than half-open because the upper bound is routinely the
    sample maximum, and a half-open rule would drop the single most extreme
    sample out of the very region that was drawn to contain it.
    """
    v = np.asarray(values, dtype=float)
    return np.isfinite(v) & (v >= bounds.low) & (v <= bounds.high)
