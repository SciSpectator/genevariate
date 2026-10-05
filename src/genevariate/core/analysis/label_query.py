"""Selecting samples by more than one label at once, in one place."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, Optional, Tuple

import pandas as pd

__all__ = [
    "MISSING",
    "RESERVED_COLUMNS",
    "QueryResult",
    "queryable_columns",
    "column_values",
    "query_mask",
    "describe_criteria",
    "run_query",
]

MISSING = "N/A"

RESERVED_COLUMNS = frozenset({
    "gsm", "gene", "_platform", "series_id", "gpl", "platform",
})


@dataclass(frozen=True)
class QueryResult:
    """The outcome of a multi-label query."""

    mask: pd.Series
    criteria: Tuple[Tuple[str, str], ...]
    n_matched: int
    n_total: int
    description: str
    unknown_columns: Tuple[str, ...] = field(default=())
    unknown_values: Tuple[Tuple[str, str], ...] = field(default=())

    @property
    def ok(self) -> bool:
        """True when every criterion named a real column and a real value."""
        return not self.unknown_columns and not self.unknown_values


def queryable_columns(df: pd.DataFrame,
                      skip: Optional[Iterable[str]] = None) -> List[str]:
    """The label columns of ``df``, in the order the frame carries them."""
    extra = {str(c) for c in (skip or ())}
    out = []
    for c in df.columns:
        name = str(c)
        if name.lower() in RESERVED_COLUMNS or name in extra:
            continue
        if df[c].dtype == "object":
            out.append(name)
    return out


def _labels_per_sample(df: pd.DataFrame, column: str) -> pd.Series:
    """One row per label a sample carries, indexed by that sample's row."""
    from .region_enrichment import explode_multi_value
    filled = df[column].fillna(MISSING).astype(str)
    return explode_multi_value(filled).astype(str).str.strip()


def column_values(df: pd.DataFrame, column: str) -> List[str]:
    """Every distinct label of ``column``, sorted, unrecorded ones included."""
    if column not in df.columns:
        return []
    labels = _labels_per_sample(df, column)
    return sorted(set(labels.tolist()) - {""})


def _normalise(criteria) -> Tuple[Tuple[str, str], ...]:
    """Accept a mapping or a sequence of pairs; keep the caller's order."""
    if criteria is None:
        return ()
    if isinstance(criteria, Mapping):
        items = list(criteria.items())
    else:
        items = list(criteria)
    out = []
    for item in items:
        col, val = item
        col, val = str(col).strip(), str(val).strip()
        if col and val:
            out.append((col, val))
    return tuple(out)


def query_mask(df: pd.DataFrame, criteria) -> pd.Series:
    """The boolean mask for ``criteria``, ANDed across every criterion."""
    mask = pd.Series(True, index=df.index)
    for col, val in _normalise(criteria):
        if col in df.columns:
            labels = _labels_per_sample(df, col)
            hit = labels.index[labels == val]
            mask = mask & pd.Series(df.index.isin(hit), index=df.index)
    return mask


def describe_criteria(criteria) -> str:
    """``Tissue=Liver AND Condition=Cancer``, or a stated absence."""
    pairs = _normalise(criteria)
    if not pairs:
        return "(no criteria)"
    return " AND ".join(f"{c}={v}" for c, v in pairs)


def run_query(df: pd.DataFrame, criteria,
              skip: Optional[Iterable[str]] = None) -> QueryResult:
    """Run a multi-label query and report what it matched and what it could not."""
    pairs = _normalise(criteria)
    known = set(map(str, df.columns))
    unknown_cols = tuple(c for c, _ in pairs if c not in known)

    unknown_vals = []
    for col, val in pairs:
        if col in known and val not in set(column_values(df, col)):
            unknown_vals.append((col, val))

    usable = [(c, v) for c, v in pairs
              if c in known and (c, v) not in unknown_vals]
    mask = query_mask(df, usable)
    if unknown_cols or unknown_vals:
        mask = pd.Series(False, index=df.index)

    return QueryResult(
        mask=mask,
        criteria=pairs,
        n_matched=int(mask.sum()),
        n_total=int(len(df)),
        description=describe_criteria(pairs),
        unknown_columns=unknown_cols,
        unknown_values=tuple(unknown_vals),
    )
