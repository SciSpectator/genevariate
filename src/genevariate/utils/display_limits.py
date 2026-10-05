"""How much of a result reaches the user, decided by the user."""
from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

_STORE = pathlib.Path.home() / ".genevariate" / "display_limits.json"

#: key -> (default, what it limits, where the user meets it)
DISPLAY: Dict[str, Tuple[Optional[int], str, str]] = {
    "bars": (25, "Bars in a frequency or enrichment chart",
             "Region analysis: Frequency, Enrichment"),
    "enrichment_rows": (30, "Label values drawn on an enrichment panel",
                        "Region analysis: Enrichment; the assistant"),
    "groups": (20, "Coloured groups in a distribution",
               "Region analysis: Distributions; Compare distributions"),
    "summary_rows": (8, "Label values on a summary panel",
                     "Region analysis: Summary"),
    "summary_groups": (6, "Coloured groups in the summary histogram",
                       "Region analysis: Summary"),
    "summary_studies": (8, "Studies named on the summary study panel",
                        "Region analysis: Summary"),
    "table_columns": (25, "Columns listed in a samples table",
                      "Region analysis: Samples; Selected samples"),
    "table_rows": (2000, "Rows listed in a samples table",
                   "Region analysis: Samples; Compare distributions"),
    "legend_groups": (12, "Groups named in a scatter legend",
                      "Selected samples"),
    "categories": (30, "Categories before the rest become 'Other'",
                   "Interactive subset"),
    "assistant_rows": (25, "Rows in a table the assistant hands back",
                       "Assistant (chat)"),
}

#: key -> (default, what it decides, what changes when you raise it)
ANALYSIS: Dict[str, Tuple[Optional[int], str, str]] = {
    "enrichment_values": (
        30, "Label values tested per column before the rest are pooled",
        "A column with more values is binned: a numeric one into intervals, a "
        "categorical one into the commonest plus an 'Other' pool that is "
        "tested and reported like any other value. Raising this enlarges the "
        "Benjamini-Hochberg family, so every q moves."),
    "comparison_values": (
        24, "Label values entering the region comparison grid",
        "Values past this rank do not enter the grid at all and are not "
        "tested. Raising it enlarges the family every q in the grid is "
        "corrected over."),
    "discovery_results": (
        None, "Experiments each source returns in Discover Experiments",
        "Matches past this rank are discarded before the results are shown, "
        "so the search reports fewer experiments than it found. The "
        "GEOmetadb arm is never limited, and a limit set here applies to "
        "ARCHS4, CELLxGENE and Expression Atlas."),
    "projection_points": (
        10000, "Samples entering a PCA or t-SNE projection",
        "Points past this count are dropped by a seeded, group-proportional "
        "draw before the projection runs. t-SNE additionally stays at or "
        "below 5,000 unless this is cleared, because its cost grows with the "
        "square of the sample count."),
    "decision_graph_points": (
        5000, "Samples entering the density-peaks decision graph",
        "The decision graph needs the full distance matrix, so it is built "
        "from a seeded subsample of this size. Memory and time grow with the "
        "square of it; clearing it uses every sample."),
    "background_values": (
        50000, "Expression values drawn for a whole-platform background",
        "Comparing a group against its whole platform measures it against a "
        "seeded draw of this many values from the platform matrix. The draw "
        "is what the group's distribution is tested against, so raising it "
        "sharpens that background; clearing it uses every value."),
}

_current: Dict[str, Optional[int]] = {}


def _all() -> Dict[str, Tuple[Optional[int], str, str]]:
    return {**DISPLAY, **ANALYSIS}


def _load() -> None:
    global _current
    _current = {k: v[0] for k, v in _all().items()}
    try:
        saved = json.loads(_STORE.read_text())
    except Exception:
        return
    for k, v in (saved or {}).items():
        if k in _current and (v is None or (isinstance(v, int) and v > 0)):
            _current[k] = v


def get(key: str) -> Optional[int]:
    """The limit in force for ``key``; ``None`` when the user removed it."""
    if not _current:
        _load()
    if key in _current:
        return _current[key]
    spec = _all().get(key)
    return spec[0] if spec else None


def set_all(values: Dict[str, Optional[int]]) -> None:
    if not _current:
        _load()
    known = _all()
    for k, v in values.items():
        if k in known:
            _current[k] = v
    _save()


def reset() -> None:
    set_all({k: v[0] for k, v in _all().items()})


def _save() -> None:
    try:
        _STORE.parent.mkdir(parents=True, exist_ok=True)
        _STORE.write_text(json.dumps(_current, indent=1))
    except Exception:
        pass


def head(items: Sequence[Any], key: str) -> List[Any]:
    """The part of ``items`` the user asked to see."""
    n = get(key)
    return list(items) if n is None else list(items)[:n]


def head_series(series, key: str):
    """``value_counts()``-style head that honours the same setting."""
    n = get(key)
    return series if n is None else series.head(n)


def cap_note(shown: int, total: int, *, noun: str = "",
             how: str = "top") -> str:
    """``"top 25 of 340 values"`` when something was left out, else ``""``."""
    if total <= shown:
        return ""
    what = f" {noun}" if noun else ""
    if how == "top":
        return f"top {shown:,} of {total:,}{what}"
    return f"{shown:,} of {total:,}{what}, {how}"
