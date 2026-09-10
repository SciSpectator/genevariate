"""Selecting samples by more than one label at once, in one place.

A multi-label query is the question "which samples are Tissue=Liver **and**
Condition=Cancer", and the program asked it in two windows that each carried
their own copy: the sample-grouping dialog of the extraction flow and the
Region Analysis window. The two agreed today by coincidence rather than by
construction -- same masking expression typed twice, next to two different
lists of which columns are label columns at all. A query that selects 310
samples in one window and 314 in the other is the failure this module exists
to make impossible, and it is the same failure the assistant would introduce a
third copy of.

The whole computation is an equality test per criterion, ANDed:

    mask &= df[column].fillna(MISSING).astype(str) == value

``fillna`` before ``astype`` is deliberate and is the part worth stating: a
column read from CSV holds ``NaN`` for a sample whose label was never
recorded, and ``str(nan)`` is ``"nan"``, which would then be selectable as
though it were a value someone wrote down. Filling first turns every unrecorded
sample into one visible bucket that the caller can see and choose to exclude,
rather than a silent one spelled differently in each window.

Nothing here touches Tkinter, and nothing here decides *which* query to run.
"""
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

#: What an unrecorded label is called once it is a string. Both windows used
#: this literal; it is centralised here so a query cannot mean two things.
MISSING = "N/A"

#: Columns that identify a sample rather than describe it. Compared
#: case-insensitively, because the same field arrives as ``GSM`` from one
#: loader and ``gsm`` from another.
RESERVED_COLUMNS = frozenset({
    "gsm", "gene", "_platform", "series_id", "gpl", "platform",
})


@dataclass(frozen=True)
class QueryResult:
    """The outcome of a multi-label query.

    ``mask`` is indexed like the frame it came from, so a caller can use it to
    slice any aligned structure. ``unknown_columns`` and ``unknown_values``
    are reported rather than raised: a query naming a column that this
    platform does not carry is a question worth answering with "it does not
    have that field", not with a traceback.
    """

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
    """The label columns of ``df``, in the order the frame carries them.

    A label column is an ``object`` column that is not one of the identifier
    fields. ``skip`` adds caller-specific exclusions -- the Region Analysis
    window drops the raw GEO metadata it displays elsewhere -- without those
    exclusions becoming part of what every caller means by "label column".
    """
    extra = {str(c) for c in (skip or ())}
    out = []
    for c in df.columns:
        name = str(c)
        if name.lower() in RESERVED_COLUMNS or name in extra:
            continue
        if df[c].dtype == "object":
            out.append(name)
    return out


def column_values(df: pd.DataFrame, column: str) -> List[str]:
    """Every distinct value of ``column``, sorted, unrecorded ones included.

    Every value, not a head: a column with hundreds of tissues is exactly the
    case a multi-label query is for, and a value past a cut is not selectable
    at all.
    """
    if column not in df.columns:
        return []
    return sorted(df[column].fillna(MISSING).astype(str).unique().tolist())


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
    """The boolean mask for ``criteria``, ANDed across every criterion.

    An empty query selects everything, which is what "no criteria" means; it
    is the caller's job to decide whether running that is useful.
    """
    mask = pd.Series(True, index=df.index)
    for col, val in _normalise(criteria):
        if col in df.columns:
            mask = mask & (df[col].fillna(MISSING).astype(str) == val)
    return mask


def describe_criteria(criteria) -> str:
    """``Tissue=Liver AND Condition=Cancer``, or a stated absence."""
    pairs = _normalise(criteria)
    if not pairs:
        return "(no criteria)"
    return " AND ".join(f"{c}={v}" for c, v in pairs)


def run_query(df: pd.DataFrame, criteria,
              skip: Optional[Iterable[str]] = None) -> QueryResult:
    """Run a multi-label query and report what it matched and what it could not.

    A criterion naming a column the frame does not carry is dropped from the
    mask and recorded in ``unknown_columns``; one naming a value the column
    does not hold is recorded in ``unknown_values``. Both leave ``ok`` False,
    so a caller can tell "nothing matched" from "you asked for a field that is
    not here" -- two answers that a bare count of zero would conflate.
    """
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
        # A query that could not be asked in full has matched nothing, however
        # many rows the criteria that *were* understood happen to select.
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
