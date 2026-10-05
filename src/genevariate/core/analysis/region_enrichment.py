"""Which label values are over-represented inside a region, and at what q."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from .enrichment import benjamini_hochberg
from .overdispersion import enrichment_diagnostics

__all__ = ["build_enrichment_cells", "explode_multi_value",
           "region_label_enrichment", "summarize_region_enrichment",
           "region_composition", "summarize_region_composition",
           "NOT_SPECIFIED"]

NOT_SPECIFIED = frozenset({
    "nan", "none", "n/a", "na", "not specified", "unknown", "not applicable",
    "",
})

def _max_values() -> int:
    """How many label values a column is tested at before the rest are pooled."""
    from genevariate.utils import display_limits
    n = display_limits.get("enrichment_values")
    return 10 ** 9 if n is None else n


def _bin_high_cardinality(combined: pd.Series,
                          max_values: Optional[int] = None) -> pd.Series:
    """Collapse a column with too many values, numerically when it is numeric."""
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


def _parts_outside_brackets(text: str) -> List[str]:
    """The parts of *text* separated by the semicolons that sit outside brackets."""
    parts: List[str] = []
    buf: List[str] = []
    depth = 0
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        if ch == ";" and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return parts


def explode_multi_value(described: pd.Series) -> pd.Series:
    """Give a sample carrying several labels one row per label."""
    if described.empty:
        return described
    parts = described.map(_parts_outside_brackets)
    if not parts.map(len).gt(1).any():
        return described
    out = parts.explode().str.strip()
    out = out[out != ""]
    pair = pd.MultiIndex.from_arrays([out.index, out.to_numpy()])
    return out[~pair.duplicated()]


def build_enrichment_cells(
    labels_by_column: Mapping[str, pd.Series],
    selected_ids: Sequence,
    *,
    region_name: str,
    study_of: Optional[Mapping] = None,
    min_study_coverage: float = 0.5,
) -> Tuple[Dict[Tuple[str, str], Tuple[pd.Series, pd.Series]],
           Dict[Tuple[str, str], list]]:
    """Returns ``(cells, groups)`` shaped for :func:`region_label_enrichment`."""
    inside_ids = set(selected_ids)
    cells: Dict[Tuple[str, str], Tuple[pd.Series, pd.Series]] = {}
    groups: Dict[Tuple[str, str], list] = {}

    for col, series in (labels_by_column or {}).items():
        if series is None or getattr(series, "empty", True):
            continue
        described = explode_multi_value(series.dropna().astype(str))
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
    """One row per tested value with ``p_value``, ``q_value`` and the counts behind them."""
    rows: List[Dict[str, Any]] = []
    diag_inputs: Dict[Tuple[str, str], tuple] = {}
    row_index: Dict[Tuple[Tuple[str, str], str], Dict[str, Any]] = {}

    for key, pair in cells.items():
        region, column = key
        inside, outside = pair
        inside = pd.Series(inside).dropna().astype(str)
        outside = pd.Series(outside).dropna().astype(str)
        rows_in = len(inside)
        n_in, n_out = inside.index.nunique(), outside.index.nunique()
        if n_in == 0 or n_out == 0:
            continue

        combined = pd.concat([inside, outside])
        binned = _bin_high_cardinality(combined, max_values)
        if binned is not combined:
            inside = binned.iloc[:rows_in]
            outside = binned.iloc[rows_in:]

        diag_inputs[key] = (
            np.r_[np.ones(len(inside), dtype=bool),
                  np.zeros(len(outside), dtype=bool)],
            pd.concat([inside, outside]).to_numpy(),
            list(groups.get(key)) if groups and groups.get(key) is not None
            else None,
        )

        for value in sorted(inside.unique()):
            if str(value).strip().lower() in NOT_SPECIFIED:
                continue
            a = int(inside.index[inside == value].nunique())
            if a == 0:
                continue
            b = n_in - a
            c = int(outside.index[outside == value].nunique())
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

    for row, q in zip(rows, benjamini_hochberg([r["p_value"] for r in rows])):
        q = float(q) if np.isfinite(q) else 1.0
        row["q_value"] = q
        row["significance"] = ("***" if q < 0.001 else "**" if q < 0.01
                               else "*" if q < 0.05 else "ns")

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
    out["replicated"] = [
        None if r["significance"] == "ns" else not _is_thin(r)
        for _, r in out.iterrows()]
    return out.reset_index(drop=True)


def region_composition(
    cells: Mapping[Tuple[str, str], Tuple[pd.Series, pd.Series]],
    *,
    max_values: Optional[int] = None,
) -> pd.DataFrame:
    """What a region is made of: every label value counted, nothing tested."""
    rows: List[Dict[str, Any]] = []

    for (region, column), pair in cells.items():
        inside = pd.Series(pair[0]).fillna("N/A").astype(str)
        outside = pd.Series(pair[1]).fillna("N/A").astype(str)
        rows_in = len(inside)
        n_in, n_out = inside.index.nunique(), outside.index.nunique()
        if n_in == 0:
            continue

        combined = pd.concat([inside, outside])
        binned = _bin_high_cardinality(combined, max_values)
        if binned is not combined:
            inside, outside = binned.iloc[:rows_in], binned.iloc[rows_in:]

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
    """True when a hit is not backed by replicated, study-independent evidence."""
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
