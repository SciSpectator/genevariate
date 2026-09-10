"""GeneVariate - RAM budget check for large allocations."""

from __future__ import annotations

import os
from typing import Optional

DEFAULT_MEM_FRACTION = 0.5


class AllocationTooLargeError(MemoryError):
    """A request whose dense form would not fit in available memory."""


def available_ram_bytes() -> Optional[int]:
    """Bytes of RAM available right now, or None if it cannot be read."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def mem_fraction(env_var: str = "GENEVARIATE_MEM_FRACTION") -> float:
    """Ceiling as a fraction of available RAM, from ``env_var`` or the default."""
    raw = os.environ.get(env_var, "").strip()
    if raw:
        try:
            val = float(raw)
            if 0.0 < val <= 1.0:
                return val
        except ValueError:
            pass
    return DEFAULT_MEM_FRACTION


def budget_bytes(env_var: str = "GENEVARIATE_MEM_FRACTION") -> Optional[int]:
    """Bytes one allocation may claim, or None when RAM cannot be measured."""
    avail = available_ram_bytes()
    if avail is None:
        return None
    return int(avail * mem_fraction(env_var))


def require_fits(n_bytes: int, what: str, *, advice: str = "",
                 env_var: str = "GENEVARIATE_MEM_FRACTION") -> None:
    """Raise :class:`AllocationTooLargeError` unless ``n_bytes`` fits.

    Passes when RAM cannot be measured. Never reduces the request.
    """
    budget = budget_bytes(env_var)
    if budget is None or n_bytes <= budget:
        return
    gib = 1024 ** 3
    avail = available_ram_bytes() or 0
    tail = f" {advice}" if advice else ""
    raise AllocationTooLargeError(
        f"{what} would need {n_bytes / gib:,.1f} GiB, but only "
        f"{avail / gib:,.1f} GiB of RAM is available "
        f"({budget / gib:,.1f} GiB usable for one allocation). Refusing "
        f"rather than silently reducing it, because a result computed over "
        f"less than what was asked for would still report success.{tail}"
    )
