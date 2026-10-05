"""How a high-expression region is defined, in one place."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

__all__ = ["RegionBounds", "resolve_bounds", "region_mask"]


@dataclass(frozen=True)
class RegionBounds:
    """A closed interval on the expression axis, and how it was chosen."""

    low: float
    high: float
    rule: str

    @property
    def empty(self) -> bool:
        """True when the rule placed the cut past the data it was given."""
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
    """Turn whichever region rule the caller gave into a concrete interval."""
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
    """Membership of ``values`` in ``bounds``, closed at both ends."""
    v = np.asarray(values, dtype=float)
    return np.isfinite(v) & (v >= bounds.low) & (v <= bounds.high)
