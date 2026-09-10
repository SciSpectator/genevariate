"""Which label values are over-represented inside a region, and at what q.

This is the test behind the Region Analysis window's Enrichment tab: for every
label value, a one-sided Fisher exact test of "in the region" against "the rest
of the samples the region was drawn from", Benjamini-Hochberg across the whole
grid that was tested, and -- for the hits that survive -- the study-clumping
diagnostics from :mod:`..overdispersion`.

It lived inside the window, interleaved with the Treeview that displayed it, so
the numbers could only be obtained by opening that window and clicking. The
assistant therefore could not answer the single most common question asked of a
region ("what is it enriched for, and what is the q value?") at all, and a
second implementation written for it would have been a second set of q values
for the same data. The statistics live here now; the window keeps the widgets.

**The multiplicity correction is applied across everything the caller tested in
one go.** That is deliberate and it is why :func:`region_label_enrichment` takes
the whole grid rather than one cell at a time: a session that tests more label
fields reports a larger q for an identical p, so a q value is only meaningful
together with the grid it was corrected over. Splitting the grid across calls
would quietly shrink every q.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from .enrichment import benjamini_hochberg
from .overdispersion import enrichment_diagnostics

__all__ = ["build_enrichment_cells",
           "region_label_enrichment", "summarize_region_enrichment",
           "region_composition", "summarize_region_composition",
           "NOT_SPECIFIED"]

#: Label values that record the absence of a label rather than a value. They are
#: never tested: "we did not extract a tissue for these samples" is a statement
#: about coverage, and letting it compete for significance would report the
#: extractor's blind spots as biology.
NOT_SPECIFIED = frozenset({
    "nan", "none", "n/a", "na", "not specified", "unknown", "not applicable",
    "",
})

#: Above this many distinct values a column is binned before testing. A free-text
#: age or a raw treatment string can carry hundreds of values, each with a
#: handful of samples, and testing them all buys nothing but a multiplicity
#: penalty that buries the columns that do carry signal.
def _max_values() -> int:
    """How many label values a column is tested at before the rest are pooled.

    This is the user's setting, not a constant: it decides the size of the
    Benjamini-Hochberg family, so a reader who wants every value tested has to
    be able to ask for that.
    """
    from genevariate.utils import display_limits
    n = display_limits.get("enrichment_values")
    return 10 ** 9 if n is None else n


MAX_VALUES = 30


def _bin_high_cardinality(combined: pd.Series,
                          max_values: Optional[int] = None) -> pd.Series:
    """Collapse a column with too many values, numerically when it is numeric.

    A numeric column (an age, a dose) is cut into intervals, because its values
    are ordered and neighbouring ones mean nearly the same thing. A categorical
    column keeps its commonest values and pools the rest into ``Other``, because
    its values are not ordered and averaging them would invent a category.
    """
    if max_values is None:
        max_values = _max_values()
    if combined.nunique() <= max_values:
        return combined

    numeric = pd.to_numeric(combined, errors="coerce")
    if numeric.notna().sum() > len(combined) * 0.5:
        try:
            n_bins = min(12, max(5, combined.nunique() // 5))
            return pd.cut(numeric, bins=n_bins,
                          duplicates="drop").astype(str).fillna("N/A")
        except Exception:
            pass
    top = combined.value_counts().head(max_values - 1).index
    return combined.where(combined.isin(top), "Other")


def build_enrichment_cells(
    labels_by_column: Mapping[str, pd.Series],
    selected_ids: Sequence,
    *,
    region_name: str,
    study_of: Optional[Mapping] = None,
    min_study_coverage: float = 0.5,
) -> Tuple[Dict[Tuple[str, str], Tuple[pd.Series, pd.Series]],
           Dict[Tuple[str, str], list]]:
    """Split resolved labels into the inside/outside pairs the test consumes.

    Resolving *where* a column's labels come from is the caller's job, and it
    genuinely differs: a window reads the frame it drew the region on and falls
    back to a label file, an assistant tool reads the merged platform frame.
    Splitting them is not caller-specific, and when each caller wrote its own
    split they disagreed - one dropped the samples the label does not describe
    and one cast them to the string ``"nan"``, which left them in the
    background as though they had been observed not to be the value under
    test. Every fold that path reported came out too high by the ratio of the
    two backgrounds. Doing the split in one place is what stops that.

    ``labels_by_column`` maps a label column to a Series indexed by sample id.
    ``selected_ids`` are the ids inside the region. ``study_of`` optionally
    maps an id to the study it came from; it is attached only when at least
    ``min_study_coverage`` of the samples in a cell have one, because a
    study-clumped statistic computed over mostly-unknown studies is worse than
    none.

    Returns ``(cells, groups)`` shaped for :func:`region_label_enrichment`.
    """
    inside_ids = set(selected_ids)
    cells: Dict[Tuple[str, str], Tuple[pd.Series, pd.Series]] = {}
    groups: Dict[Tuple[str, str], list] = {}

    for col, series in (labels_by_column or {}).items():
        if series is None or getattr(series, "empty", True):
            continue
        # Drop before the cast, never after: ``astype(str)`` would turn a
        # missing label into a value that survives every later filter.
        described = series.dropna().astype(str)
        if described.empty:
            continue
        in_mask = described.index.isin(list(inside_ids))
        inside, outside = described[in_mask], described[~in_mask]
        if inside.empty or outside.empty:
            continue

        key = (region_name, str(col))
        cells[key] = (inside, outside)
        if study_of:
            ids = list(inside.index) + list(outside.index)
            studies = [study_of.get(i) for i in ids]
            known = sum(1 for s in studies if s)
            if known >= len(studies) * min_study_coverage:
                groups[key] = [s if s else "_unknown" for s in studies]
    return cells, groups


def region_label_enrichment(
    cells: Mapping[Tuple[str, str], Tuple[pd.Series, pd.Series]],
    groups: Optional[Mapping[Tuple[str, str], Sequence]] = None,
    *,
    max_values: Optional[int] = None,
    n_boot: int = 500,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Test every label value of every region x label column in one grid.

    ``cells`` maps ``(region_name, label_column)`` to a pair of label Series:
    the values of the samples **inside** the region and the values of the
    samples **outside** it. Both are indexed by sample id. Resolving which
    frame a column's labels come from is the caller's job -- it differs between
    a loaded platform and a label file -- but the test must not.

    ``groups`` optionally maps the same key to a study id per sample, ordered
    ``inside`` then ``outside``, matching the concatenation of the two Series.
    Given it, each surviving hit also reports the number of contributing
    studies, the intra-study correlation, the effective sample size after that
    clumping, and a CI bootstrapped over studies rather than samples. GEO
    samples arrive in study-sized clumps, so without those a p-value is far
    more confident than the evidence warrants.

    Returns one row per tested value with ``p_value``, ``q_value`` and the
    counts behind them, sorted by p. An empty frame means nothing was testable.
    """
    rows: List[Dict[str, Any]] = []
    diag_inputs: Dict[Tuple[str, str], tuple] = {}
    row_index: Dict[Tuple[Tuple[str, str], str], Dict[str, Any]] = {}

    for key, pair in cells.items():
        region, column = key
        inside, outside = pair
        inside = pd.Series(inside).dropna().astype(str)
        outside = pd.Series(outside).dropna().astype(str)
        n_in, n_out = len(inside), len(outside)
        if n_in == 0 or n_out == 0:
            continue

        combined = pd.concat([inside, outside])
        binned = _bin_high_cardinality(combined, max_values)
        if binned is not combined:
            inside = binned.iloc[:n_in]
            outside = binned.iloc[n_in:]

        diag_inputs[key] = (
            np.r_[np.ones(n_in, dtype=bool), np.zeros(n_out, dtype=bool)],
            pd.concat([inside, outside]).to_numpy(),
            list(groups.get(key)) if groups and groups.get(key) is not None
            else None,
        )

        for value in sorted(inside.unique()):
            if str(value).strip().lower() in NOT_SPECIFIED:
                continue
            a = int((inside == value).sum())
            if a == 0:
                continue
            b, c = n_in - a, int((outside == value).sum())
            d = n_out - c
            try:
                odds, p = fisher_exact(np.array([[a, b], [c, d]]),
                                       alternative="greater")
            except Exception:
                odds, p = np.nan, 1.0
            in_frac, bg_frac = a / n_in, c / n_out
            row = {
                "region": region, "label_column": column, "value": value,
                "a": a, "n_region": n_in, "c": c, "n_background": n_out,
                "region_pct": in_frac * 100.0, "background_pct": bg_frac * 100.0,
                "enrichment": (in_frac / bg_frac) if bg_frac > 0 else np.inf,
                # The odds ratio of the very 2x2 the p came from. Reported
                # beside the enrichment because the two diverge once a region
                # is a large slice of the platform: one is a ratio of rates,
                # the other a ratio of odds.
                "odds_ratio": float(odds),
                "p_value": float(p),
                "n_gse": None, "rho": np.nan, "n_eff": np.nan,
                "ci_low": np.nan, "ci_high": np.nan,
            }
            rows.append(row)
            row_index[(key, value)] = row

    if not rows:
        return pd.DataFrame(columns=[
            "region", "label_column", "value", "a", "n_region", "c",
            "n_background", "region_pct", "background_pct", "enrichment",
            "odds_ratio", "p_value", "q_value", "significance", "n_gse",
            "rho", "n_eff", "ci_low", "ci_high", "replicated"])

    # One correction over the whole grid - see the module docstring.
    for row, q in zip(rows, benjamini_hochberg([r["p_value"] for r in rows])):
        q = float(q) if np.isfinite(q) else 1.0
        row["q_value"] = q
        row["significance"] = ("***" if q < 0.001 else "**" if q < 0.01
                               else "*" if q < 0.05 else "ns")

    # Clumping diagnostics only for the survivors: fitting a beta-binomial per
    # value is the expensive step and a column can hold hundreds of them.
    wanted: Dict[Tuple[str, str], List[str]] = {}
    for row in rows:
        if row["significance"] != "ns":
            wanted.setdefault((row["region"], row["label_column"]), []
                              ).append(row["value"])
    for key, values in wanted.items():
        inputs = diag_inputs.get(key)
        if inputs is None:
            continue
        try:
            diag = enrichment_diagnostics(*inputs, values=values,
                                          n_boot=n_boot, alpha=alpha)
        except Exception:
            continue
        for value, dv in diag.items():
            row = row_index.get((key, value))
            if row is None:
                continue
            row["n_gse"] = dv.get("n_gse")
            row["rho"] = dv.get("rho", np.nan)
            row["n_eff"] = dv.get("n_eff_sel", np.nan)
            row["ci_low"] = dv.get("ci_low", np.nan)
            row["ci_high"] = dv.get("ci_high", np.nan)

    out = pd.DataFrame(rows).sort_values("p_value", kind="stable")
    # ``None`` rather than ``True`` where nothing was diagnosed: the diagnostics
    # are only run for the survivors, so calling a non-significant row
    # "replicated" would report an answer to a question never asked of it.
    out["replicated"] = [
        None if r["significance"] == "ns" else not _is_thin(r)
        for _, r in out.iterrows()]
    return out.reset_index(drop=True)


def region_composition(
    cells: Mapping[Tuple[str, str], Tuple[pd.Series, pd.Series]],
    *,
    max_values: Optional[int] = None,
) -> pd.DataFrame:
    """What a region is made of: every label value counted, nothing tested.

    Takes the same ``cells`` mapping as :func:`region_label_enrichment` and
    answers the other half of the question. Enrichment asks which values are
    over-represented and has to discard the ones that record an absent label;
    composition asks what is actually *in* there and must keep them, because
    "42% of this region has no tissue recorded" is the first thing a reader
    needs to know before any q value from it means anything. A value that is
    unspecified for half the region is not a null result, it is the reason the
    result is thin, and only this table shows it.

    Values present in the background but absent from the region are reported
    too, with ``n_region`` of zero: a tissue the region *excludes* is as much
    a description of the region as one it concentrates.

    Returns one row per value, sorted by the share of the region it accounts
    for, with ``enrichment`` against the background for orientation only --
    it carries no p-value and must not be quoted as if it did.
    """
    rows: List[Dict[str, Any]] = []

    for (region, column), pair in cells.items():
        inside = pd.Series(pair[0]).fillna("N/A").astype(str)
        outside = pd.Series(pair[1]).fillna("N/A").astype(str)
        n_in, n_out = len(inside), len(outside)
        if n_in == 0:
            continue

        combined = pd.concat([inside, outside])
        binned = _bin_high_cardinality(combined, max_values)
        if binned is not combined:
            inside, outside = binned.iloc[:n_in], binned.iloc[n_in:]

        in_counts = inside.value_counts()
        out_counts = outside.value_counts()
        for value in set(in_counts.index) | set(out_counts.index):
            a = int(in_counts.get(value, 0))
            c = int(out_counts.get(value, 0))
            in_frac = a / n_in
            bg_frac = (c / n_out) if n_out else np.nan
            rows.append({
                "region": region, "label_column": column, "value": value,
                "n_region": a, "region_pct": in_frac * 100.0,
                "n_background": c,
                "background_pct": bg_frac * 100.0 if n_out else np.nan,
                "n_total": a + c,
                "enrichment": ((in_frac / bg_frac) if bg_frac
                               else (np.inf if a else np.nan)),
                "unspecified": str(value).strip().lower() in NOT_SPECIFIED
                or str(value) == "N/A",
            })

    if not rows:
        return pd.DataFrame(columns=[
            "region", "label_column", "value", "n_region", "region_pct",
            "n_background", "background_pct", "n_total", "enrichment",
            "unspecified"])
    out = pd.DataFrame(rows).sort_values(
        ["region", "label_column", "n_region"],
        ascending=[True, True, False], kind="stable")
    return out.reset_index(drop=True)


def summarize_region_composition(table: pd.DataFrame, top: int = 8) -> str:
    """Markdown for a composition table, leading with what is unrecorded."""
    if table is None or table.empty:
        return "The region holds no labelled sample to describe.\n"

    lines: List[str] = []
    for (region, column), grp in table.groupby(["region", "label_column"],
                                               sort=False):
        n_in = int(grp["n_region"].sum())
        miss = grp[grp["unspecified"]]["n_region"].sum()
        head = f"**{region} - `{column}`** ({n_in:,} samples in the region"
        if miss:
            head += f"; {100.0 * miss / n_in:.1f}% carry no value"
        lines.append(head + ")\n")
        shown = grp[grp["n_region"] > 0].head(top)
        for _, r in shown.iterrows():
            bg = ("" if not np.isfinite(r["background_pct"]) else
                  f" vs {r['background_pct']:.1f}% outside")
            lines.append(f"- {r['value']}: {r['n_region']:,} "
                         f"({r['region_pct']:.1f}%{bg})")
        rest = int((grp["n_region"] > 0).sum()) - len(shown)
        if rest > 0:
            lines.append(f"- ...and {rest} further value(s)")
        absent = grp[(grp["n_region"] == 0) & (grp["n_background"] > 0)]
        if len(absent):
            top_absent = absent.nlargest(3, "n_background")["value"].tolist()
            lines.append(f"- Absent from the region entirely: "
                         f"{', '.join(map(str, top_absent))}"
                         + (f" and {len(absent) - 3} more"
                            if len(absent) > 3 else ""))
        lines.append("")
    lines.append("These are counts, not tests. For which of them are "
                 "over-represented beyond chance, and at what q, use the "
                 "enrichment test.\n")
    return "\n".join(lines)


def _is_thin(row: Mapping[str, Any]) -> bool:
    """True when a hit is not backed by replicated, study-independent evidence.

    Either it rests on fewer than three studies, or the study-bootstrap CI on
    the enrichment ratio still covers 1.0. Both mean the p-value is far more
    confident than the data warrant, and a hit like that should not be read as
    a finding about biology rather than about one submitter's experiment.
    """
    n_gse = row.get("n_gse")
    if n_gse is not None and n_gse < 3:
        return True
    lo, hi = row.get("ci_low", np.nan), row.get("ci_high", np.nan)
    if np.isfinite(lo) and np.isfinite(hi) and lo <= 1.0 <= hi:
        return True
    return False


def summarize_region_enrichment(table: pd.DataFrame, top: int = 15) -> str:
    """Markdown for an enrichment table, naming what is and is not replicated."""
    if table is None or table.empty:
        return "No label value could be tested for enrichment in this region.\n"
    sig = table[table["significance"] != "ns"]
    lines = [
        f"**{len(sig)} of {len(table)} tested label values are enriched at "
        f"q<0.05** (Fisher exact, one-sided, Benjamini-Hochberg across all "
        f"{len(table)} tests).\n",
    ]
    if sig.empty:
        lines.append("Nothing survives the correction over this grid.\n")
        return "\n".join(lines)

    thin = int((sig["replicated"] == False).sum())  # noqa: E712 - object dtype
    for _, r in sig.head(top).iterrows():
        ci = ""
        if np.isfinite(r["ci_low"]) and np.isfinite(r["ci_high"]):
            ci = f", study-bootstrap CI {r['ci_low']:.2f}-{r['ci_high']:.2f}"
        n_gse = r["n_gse"]
        studies = (f", {int(n_gse)} studies"
                   if n_gse is not None and np.isfinite(n_gse) else "")
        flag = "" if r["replicated"] else "  **not replicated**"
        lines.append(
            f"- `{r['label_column']}` = **{r['value']}** - "
            f"{r['region_pct']:.1f}% of the region vs "
            f"{r['background_pct']:.1f}% of the background "
            f"({r['enrichment']:.2f}x), q={r['q_value']:.3g}"
            f"{studies}{ci}{flag}")
    if thin:
        lines.append(
            f"\n{thin} of the {len(sig)} hits rest on fewer than three studies "
            f"or have a CI covering 1.0, so they are not evidence that the "
            f"effect generalises beyond the studies that carry it.")
    return "\n".join(lines) + "\n"
