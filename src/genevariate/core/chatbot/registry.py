"""
Tool registry for the GeneVariate assistant.

``build_registry(app)`` returns a ``{name: Tool}`` map whose executors call the
*existing* analysis API (``genevariate.core.analysis``) against the platforms
already loaded in ``app.gpl_datasets``. Everything here is Tk-free; executors
run on a worker thread and return a :class:`ToolResult`.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from genevariate.core import label_entities

from .tools import Tool, ToolParam, ToolResult, history as tool_history


# -----------------------------------------------------------------
# Shared resolver helpers
# -----------------------------------------------------------------
def _platforms(app) -> Dict[str, pd.DataFrame]:
    return getattr(app, "gpl_datasets", {}) or {}


def _modalities(app, keys) -> Dict[str, str]:
    """Real modality per source, from the app's GEOmetadb-backed technology.

    The accession alone cannot say whether a GPL is an array or a sequencer,
    so ask the app rather than letting the name heuristic guess. Sources the
    app knows nothing about are simply left out, and the analysis layer falls
    back to its own inference for those.
    """
    from genevariate.core.analysis.cross_modality import modality_from_category
    facts = getattr(app, "_platform_facts", None)
    if not callable(facts):
        return {}
    out: Dict[str, str] = {}
    for k in keys:
        try:
            mod = modality_from_category(facts(k).get("category"))
        except Exception:
            mod = None
        if mod:
            out[k] = mod
    return out


def session_state_text(app) -> str:
    """What is loaded right now, as a block for the agent's system prompt.

    The agent used to have to call ``list_platforms`` to discover this, and a
    turn that skipped it would re-acquire data that was already in the session.
    For a GEO platform that is merely wasteful - the reload is byte-identical.
    For a single-cell census it is wrong: the census is re-queried live and a
    different cell budget yields a different pseudo-bulk matrix, so the sample
    count, the regions derived from it and every downstream number move.

    Reads the same ``_platforms``/``_modalities`` accessors the tools use, so
    it cannot drift from what ``list_platforms`` would report.
    """
    plats = _platforms(app)
    if not plats:
        return ("\n\nCURRENT SESSION STATE\nNo platforms are loaded yet - "
                "acquire the one the user described before analysing.\n")
    mods = _modalities(app, list(plats.keys()))
    lines = []
    for key, frame in plats.items():
        rows, cols = getattr(frame, "shape", (0, 0))[:2]
        mod = mods.get(key)
        lines.append(f"  - {key}: {rows:,} samples x {cols:,} columns"
                     + (f" ({mod})" if mod else ""))
    return ("\n\nCURRENT SESSION STATE\nAlready loaded - use these directly:\n"
            + "\n".join(lines)
            + "\nIf the data you need is listed above it is ALREADY in the "
              "session. Do NOT call load_geo_platform, fetch_single_cell, "
              "load_single_cell_file or add_custom_platform for it, and do not "
              "call list_platforms to re-check this list. Re-fetching a "
              "single-cell census does NOT reproduce the loaded one: the "
              "census is queried live and a different cell budget gives a "
              "different pseudo-bulk matrix under a different name.\n")


def _match_platform(app, name: Optional[str]) -> Optional[str]:
    """Map a friendly/partial platform name to a real gpl_datasets key."""
    keys = list(_platforms(app).keys())
    if not keys:
        return None
    if not name:
        return keys[0]
    name = str(name).strip()
    if name in keys:
        return name
    low = name.lower()
    for k in keys:
        if k.lower() == low:
            return k
    for k in keys:
        if low in k.lower() or k.lower() in low:
            return k
    return keys[0]


# Label columns follow two conventions: the legacy "Classified_" prefix and the
# plain canonical names the app stores today (the label loader strips the prefix,
# and geo_extract_driver emits plain Sex/Age/Tissue/Condition/Treatment). Match
# both, and the kind columns derived from the entity links as well: a Tissue that
# resolved to a catalogued cell line is a group the assistant must be able to
# split on, and anything not recognised here is handed to the ranker as a gene.
_LABEL_FIELDS_LOWER = frozenset(
    f.lower() for f in
    ("Condition", "Tissue", "Treatment", "Age", "Sex")) | frozenset(
    f"{f.lower()}{label_entities.KIND_SUFFIX}"
    for f in label_entities.ENTITY_FIELDS)


#: What a tool may be asked to group or predict by. The derived kind columns are
#: offered alongside the extracted fields so a user can ask for the cell-line
#: split by name without knowing how it is spelled.
LABEL_CHOICES = ("Tissue", "Condition", "Treatment", "Sex", "Age") + tuple(
    label_entities.kind_column_name(f) for f in label_entities.ENTITY_FIELDS)


def _is_label_col(c) -> bool:
    s = str(c)
    return s.startswith("Classified_") or s.lower() in _LABEL_FIELDS_LOWER


def _classified_cols(df: pd.DataFrame) -> List[str]:
    """Label/metadata columns of a merged frame.

    ``_labelled`` records what it merged, which is the only way a curator's own
    column name is told apart from a gene symbol.
    """
    recorded = [c for c in (df.attrs.get("label_columns") or [])
                if c in df.columns]
    return recorded or [c for c in df.columns if _is_label_col(c)]


def _pick_condition_column(df: pd.DataFrame, requested: Optional[str]) -> Optional[str]:
    cols = _classified_cols(df)
    if requested:
        r = str(requested).strip()
        if r in df.columns:
            return r
        low = r.lower()
        for c in cols:
            if c.lower() == low or low in c.lower():
                return c
    return cols[0] if cols else None


def _labels_from_column(df: pd.DataFrame, column: str) -> Dict[str, str]:
    gsm = df["GSM"].astype(str).str.upper()
    return dict(zip(gsm.values, df[column].astype(str).values))


def _entity_frames(app, key: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """Per platform, the frame that still carries the accession columns.

    The expression frame only ever receives the label *values*, so the MeSH and
    Cellosaurus accessions live in the label file the user loaded (or the one
    the in-program extraction wrote), which the app keeps in
    ``platform_labels``. Fall back to the expression frame for the case where a
    caller merged the two.
    """
    out: Dict[str, pd.DataFrame] = {}
    labels = getattr(app, "platform_labels", {}) or {}
    for name, df in list(labels.items()) + list(_platforms(app).items()):
        if name in out or df is None or getattr(df, "empty", True):
            continue
        if label_entities.has_entity_links(df):
            out[name] = df
    if key:
        return {k: v for k, v in out.items() if k == key}
    return out


def _labelled(app, key: str) -> Optional[pd.DataFrame]:
    """A platform's expression frame with the loaded label columns merged in.

    Labels live in their own per-platform frames -- a file the user added, or
    the one an in-program extraction wrote -- and an expression matrix read
    from GEO carries none. Every label-aware tool reads its groups off the
    single frame it is handed, so without this merge the assistant answers "no
    label column to group by" in a session where the labels are plainly loaded
    and visible in the window beside it.

    Only the value column of each field is merged, plus the kind columns
    derived from the entity links. The accessions stay behind: a column of MeSH
    ids is metadata about a label, and merging it would offer it as a group of
    its own.
    """
    df = _platforms(app).get(key)
    if df is None or "GSM" not in getattr(df, "columns", []):
        return df
    labels = (getattr(app, "platform_labels", {}) or {}).get(key)
    if labels is None or getattr(labels, "empty", True) \
            or "GSM" not in labels.columns:
        return df
    label_entities.add_kind_columns(labels)
    cols = [c for c in label_entities.label_value_columns(labels)
            if c not in df.columns]
    if not cols:
        return df
    sub = labels[["GSM", *cols]].copy()
    sub["GSM"] = sub["GSM"].astype(str).str.upper()
    sub = sub.drop_duplicates(subset="GSM")
    # The stage prefix is dropped so a tool asked for "Tissue" finds it
    # whatever pass the file came from.
    sub = sub.rename(columns={
        c: next((c[len(p):] for p in label_entities.LABEL_STAGE_PREFIXES
                 if c.startswith(p)), c) for c in cols})
    out = df.copy()
    out["GSM"] = out["GSM"].astype(str).str.upper()
    out = out.merge(sub, on="GSM", how="left")
    out.attrs["label_columns"] = [c for c in sub.columns if c != "GSM"]
    return out


def _enrichment_figures(table, cap: Optional[int] = None):
    """The Enrichment tab's own charts for a ``region_label_enrichment`` frame.

    Drawn by the routine the window uses, so a tool returns the chart the
    button shows rather than a second one made for the assistant.
    """
    fig, figs, cdesc = None, {}, {}
    try:
        from matplotlib.figure import Figure
        from genevariate.core.analysis.figures import (
            draw_enrichment, enrichment_rows,
        )
        groups: Dict[tuple, list] = {}
        for row in enrichment_rows(table):
            if row["Sig"] != "ns":
                groups.setdefault(
                    (row["Region"], row["Label Column"]), []).append(row)
        for (reg, lcol), rws in groups.items():
            # As many values as the user asked to see, not a number this
            # module chose for them.
            from genevariate.utils import display_limits
            n = cap if cap is not None else display_limits.get("enrichment_rows")
            rws = rws if n is None else rws[:n]
            f = Figure(figsize=(16, max(4, min(20, 0.45 * len(rws) + 1.5))))
            a1, a2 = f.subplots(1, 2, gridspec_kw={"width_ratios": [3, 2]})
            d = draw_enrichment(a1, a2, rws, label_column=lcol)
            f.suptitle(f"{reg} - Fisher Enrichment: {lcol}", fontsize=11,
                       weight="bold", y=0.995)
            try:
                f.tight_layout(rect=(0, 0, 1, 0.96))
            except Exception:
                pass
            if fig is None:
                fig, cdesc = f, d
            figs[f"{reg} x {lcol}"] = f
    except Exception:
        return None, {}, ""
    if not cdesc.get("n"):
        return fig, figs, ""
    return fig, figs, (
        "\n\n## What the chart shows\n"
        f"{cdesc['n']} value(s) of {cdesc['label_column']} survive the "
        f"correction in a foreground of {cdesc['n_selected']} samples against "
        f"{cdesc['n_background']:,}. {cdesc['single_study']} of them rest on "
        "fewer than three studies, so the bar is tall and the interval wide: "
        "that is one experiment, not a population.\n")


_META_PREFIXES = ("GSM", "series_id")


def _find_gene_column(df: pd.DataFrame, gene: str) -> Optional[str]:
    """Case-insensitive match of a gene symbol to a platform gene column."""
    if not gene:
        return None
    g = str(gene).strip().upper()
    meta = {c for c in df.columns
            if c in _META_PREFIXES or _is_label_col(c)}
    for c in df.columns:
        if c in meta:
            continue
        if str(c).upper() == g:
            return c
    return None


def _gene_vector(df: pd.DataFrame, gene: str) -> Optional[np.ndarray]:
    col = _find_gene_column(df, gene)
    if col is None:
        return None
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _gene_stats(values: np.ndarray) -> Dict[str, float]:
    from genevariate.core.analysis.bimodality import classify_gene_distribution
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "std": float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "distribution": classify_gene_distribution(v),
    }


def _gsea_term_count(gsea: Optional[pd.DataFrame]) -> int:
    """Count real enriched terms, excluding per-library error rows.

    ``run_prerank_gsea`` returns rows with an ``error`` column when a library
    fails (e.g. no gene overlap); those must not be counted as enriched terms.
    """
    if gsea is None or gsea.empty:
        return 0
    if "error" in gsea.columns:
        return int(gsea["error"].isna().sum())
    return int(len(gsea))


def _manifest(tool: str, resolved: Dict[str, Any],
              inputs: Optional[Dict[str, Any]] = None,
              seed: Optional[int] = None) -> Dict[str, Any]:
    """Build a per-run reproducibility manifest for a tool result. Never raises."""
    try:
        from genevariate.core.reproducibility import build_manifest
        return build_manifest(tool, params=resolved, inputs=inputs, seed=seed)
    except Exception:
        return {}


def _append_manifest(report: str, manifest: Dict[str, Any]) -> str:
    """Append the reproducibility manifest block to a markdown report."""
    if not manifest:
        return report
    try:
        from genevariate.core.reproducibility import manifest_to_markdown
        block = manifest_to_markdown(manifest)
    except Exception:
        return report
    return (report + "\n\n" + block) if report else block


def _default_case_control(df: pd.DataFrame, column: str) -> Tuple[str, str]:
    vals = [v for v in df[column].astype(str) if v and v.lower() != "nan"]
    common = [v for v, _ in Counter(vals).most_common()]
    if len(common) >= 2:
        return common[0], common[1]
    if common:
        return common[0], common[0]
    return "case", "control"


# -----------------------------------------------------------------
# Registry
# -----------------------------------------------------------------
def build_registry(app) -> Dict[str, Tool]:
    from genevariate.core.analysis import (
        rank_genes_by_condition,
        run_prerank_gsea,
        DEFAULT_LIBRARIES,
    )

    libs_default = ",".join(DEFAULT_LIBRARIES)

    # ---- list_platforms -----------------------------------------
    def _list_resolver(app, raw):
        return {}

    def _list_exec(app, resolved, progress_cb):
        plats = _platforms(app)
        if not plats:
            return ToolResult("No platforms are loaded yet.", ok=False)
        rows = [{"platform": k,
                 "samples": getattr(v, "shape", (0, 0))[0],
                 "columns": getattr(v, "shape", (0, 0))[1]}
                for k, v in plats.items()]
        tbl = pd.DataFrame(rows)
        return ToolResult(f"{len(plats)} platform(s) loaded.", table=tbl,
                          payload={"platforms": list(plats.keys())})

    # ---- shared condition/case-control resolver -----------------
    def _cond_resolver(app, raw):
        key = _match_platform(app, raw.get("platform"))
        out = dict(raw)
        out["platform"] = key
        if key:
            df = _labelled(app, key)
            col = _pick_condition_column(df, raw.get("condition_column"))
            out["condition_column"] = col
            if col and (not raw.get("case_label") or not raw.get("control_label")):
                case, control = _default_case_control(df, col)
                out.setdefault("case_label", raw.get("case_label") or case)
                out.setdefault("control_label", raw.get("control_label") or control)
        out.setdefault("libraries", libs_default)
        return out

    def _prep_labels(app, resolved):
        key = resolved.get("platform")
        plats = _platforms(app)
        if not key or key not in plats:
            raise ValueError("No matching platform is loaded.")
        df = _labelled(app, key)
        if "GSM" not in df.columns:
            raise ValueError(f"Platform {key!r} is not in canonical (GSM) format.")
        col = resolved.get("condition_column") or _pick_condition_column(df, None)
        if not col:
            raise ValueError(
                f"Platform {key!r} has no label column to group by "
                f"(e.g. Condition/Tissue/Sex).")
        labels = _labels_from_column(df, col)
        # The canonical ranker treats every non-GSM/series_id column as a gene,
        # so drop the label/metadata columns before handing it over.
        drop = _classified_cols(df)
        expr_df = df.drop(columns=drop) if drop else df
        return expr_df, labels

    # ---- condition_enrichment -----------------------------------
    def _cond_exec(app, resolved, progress_cb):
        from genevariate.core.analysis import enrichment_report_markdown
        df, labels = _prep_labels(app, resolved)
        case = str(resolved.get("case_label"))
        control = str(resolved.get("control_label"))
        libs = [s.strip() for s in str(resolved.get("libraries", libs_default)).split(",")
                if s.strip()]
        progress_cb(20.0, "Ranking genes by condition…")
        ranked = rank_genes_by_condition(df, labels, case, control,
                                         moderated=bool(resolved.get("moderated", False)))
        progress_cb(60.0, "Running prerank GSEA…")
        gsea = run_prerank_gsea(ranked, gene_sets=libs)
        n = _gsea_term_count(gsea)
        top = gsea.head(15) if n else ranked.head(15)
        comparison = f"{case} vs {control} on {resolved.get('platform')}"
        manifest = _manifest("condition_enrichment", resolved,
                             inputs={"platform": df}, seed=42)
        try:
            report = enrichment_report_markdown(None, gsea, comparison)
        except Exception:
            report = ""
        report = _append_manifest(report, manifest)
        from . import charts
        fig, cdesc = charts.fig_from_enrichment(ranked, gsea, comparison)
        report += charts.describe_bar_block(cdesc)
        return ToolResult(
            f"Condition enrichment ({case} vs {control}) on "
            f"{resolved.get('platform')}: {n} enriched term(s).",
            table=top, report=report, manifest=manifest, figure=fig,
            payload={"ranked": ranked, "gsea": gsea, "chart": cdesc,
                     "report": report})

    # ---- rank_genes (no GSEA) -----------------------------------
    # ---- region_comparison (regions against each other) ---------
    def _regcmp_resolver(app, raw):
        key = _match_platform(app, raw.get("platform"))
        out = dict(raw)
        out["platform"] = key
        if key:
            out["condition_column"] = _pick_condition_column(
                _labelled(app, key), raw.get("condition_column"))
        out.update(_region_params(raw))
        return out

    def _regcmp_exec(app, resolved, progress_cb):
        from genevariate.core.analysis import (
            enrichment_matrix, heterogeneity, pairwise_differential,
            summarize_comparison,
        )
        key = resolved.get("platform")
        plats = _platforms(app)
        if not key or key not in plats:
            return ToolResult("No matching platform is loaded. Load one first.",
                              ok=False)
        df = _labelled(app, key)
        col = resolved.get("condition_column")
        if not col or col not in df.columns:
            return ToolResult(
                f"Platform {key!r} has no label column to compare regions on.",
                ok=False)

        genes = [g.strip() for g in str(resolved.get("genes") or "").split(",")
                 if g.strip()]
        if len(genes) < 2:
            return ToolResult(
                "Comparing regions needs at least two genes - one region each.",
                ok=False)

        # Same selector as the other region tools, so "the top decile" or a
        # brushed pair of bounds means the identical set of samples whichever
        # of them the user reaches for.
        by_col, cuts, problems = _region_masks(df, genes,
                                              _region_params(resolved))
        masks = {}
        for gene_col, mask in by_col.items():
            lo, hi = cuts[gene_col]
            masks[f"{str(gene_col).upper()} {lo:.3g}-{hi:.3g}"] = mask
        if len(masks) < 2:
            return ToolResult(
                f"Only {len(masks)} of {len(genes)} gene(s) yielded a region "
                f"on {key!r} - two are the minimum."
                + ("\n- " + "\n- ".join(problems) if problems else ""),
                ok=False)

        labels = df[col].astype(str).to_numpy()
        # GEO samples arrive in study-sized clumps; without series_id every
        # number below would silently treat them as independent draws.
        groups = (df["series_id"].astype(str).to_numpy()
                  if "series_id" in df.columns else None)

        progress_cb(25.0, "Building the region x label grid…")
        matrix = enrichment_matrix(masks, labels, groups, n_boot=300)
        progress_cb(55.0, "Testing regions against each other…")
        pairs = pairwise_differential(masks, labels, groups,
                                      values=matrix["values"], n_boot=300)
        progress_cb(80.0, "Checking whether the effects are region-specific…")
        het = heterogeneity(masks, labels, groups, values=matrix["values"])

        report = summarize_comparison(matrix, het, pairs)
        manifest = _manifest("region_comparison", resolved,
                             inputs={"platform": df}, seed=0)
        report = _append_manifest(f"# Region comparison - {key} ({col})\n\n"
                                  + report, manifest)

        tbl = pd.DataFrame([
            {"value": r["value"], "region_a": r["region_a"],
             "rate_a": round(r["rate_a"], 4), "region_b": r["region_b"],
             "rate_b": round(r["rate_b"], 4),
             "odds_ratio": round(float(np.exp(r["log_or"])), 3),
             "q": r["q"], "jaccard": round(r["jaccard"], 3)}
            for r in pairs[:25]])
        n_sig = sum(1 for r in pairs
                    if np.isfinite(r["q"]) and r["q"] < 0.05)
        note = "" if groups is not None else " (no series_id: study clumping uncorrected)"
        return ToolResult(
            f"Compared {len(masks)} regions on {key} ({col}): {n_sig} of "
            f"{len(pairs)} region pairs separate at q<0.05{note}.",
            table=tbl, report=report, manifest=manifest,
            payload={"matrix": matrix, "pairs": pairs, "heterogeneity": het,
                     "report": report})

    # ---- shared region setup for the box tools ------------------
    def _region_params(raw):
        """Read whichever region rule the caller used off a raw argument dict.

        The window's brush produces two numbers, and earlier work on this data
        was cut at a multiple of the standard deviation, so both have to be
        sayable in a sentence. They are passed straight through to
        ``resolve_bounds``, which decides the precedence; nothing is defaulted
        here, because a quantile filled in at this level would outrank the
        explicit bounds the user actually named.
        """
        def num(key):
            val = raw.get(key)
            if val in (None, ""):
                return None
            try:
                return float(val)
            except (TypeError, ValueError):
                return None

        return {"low": num("low"), "high": num("high"), "sd": num("sd"),
                "quantile": num("quantile")}

    def _region_masks(df, genes, spec):
        """One high-expression slab per gene, plus the bounds that define it.

        ``spec`` is the dict ``_region_params`` returns: explicit bounds, an SD
        tail or a quantile. Each gene's bounds are resolved against *its own*
        distribution, which is what makes a single rule ("the top decile")
        comparable across genes and across platforms that do not share a scale.
        """
        from genevariate.core.analysis import region_mask, resolve_bounds

        masks, cuts, problems = {}, {}, []
        for g in genes:
            v = _gene_vector(df, g)
            if v is None:
                problems.append(f"{g}: not measured on this platform")
                continue
            finite = v[np.isfinite(v)]
            if finite.size < 50:
                problems.append(f"{g}: only {finite.size} finite value(s), "
                                f"too few to place a region on")
                continue
            col = _find_gene_column(df, g)
            try:
                bounds = resolve_bounds(finite, **spec)
            except ValueError as exc:
                problems.append(f"{g}: {exc}")
                continue
            if bounds.empty:
                # Not a missing gene - a threshold that lands past the data.
                # Reporting it as "missing" sends the user looking for the
                # gene instead of at the cut they chose.
                problems.append(f"{g}: {bounds.describe()}")
                continue
            masks[col] = region_mask(v, bounds)
            cuts[col] = (bounds.low, bounds.high)
        return masks, cuts, problems

    # ---- region_enrichment (what is a region enriched for?) -----
    def _regenrich_resolver(app, raw):
        out = dict(raw)
        out["platform"] = _match_platform(app, raw.get("platform"))
        out.update(_region_params(raw))
        return out

    def _regenrich_exec(app, resolved, progress_cb):
        from genevariate.core.analysis import (
            region_label_enrichment, summarize_region_enrichment,
        )
        key = resolved.get("platform")
        if not key or key not in _platforms(app):
            return ToolResult("No matching platform is loaded. Load one first.",
                              ok=False)
        df = _labelled(app, key)
        genes = [g.strip() for g in str(resolved.get("genes") or "").split(",")
                 if g.strip()]
        if not genes:
            return ToolResult("Name at least one gene to define a region on.",
                              ok=False)

        # Which label columns to test. Testing every one of them in a single
        # call is not a convenience: the correction is applied across the whole
        # grid, so splitting the columns over several calls would report a
        # smaller q for each of them than the session actually earned.
        wanted = [c.strip() for c in
                  str(resolved.get("label_columns") or "").split(",") if c.strip()]
        cols = [c for c in (wanted or _classified_cols(df)) if c in df.columns]
        # A column whose samples all carry the same value cannot be enriched:
        # the region and the background hold it in identical proportion by
        # construction, so the test is decided before it runs. Keeping it adds
        # no hit and one slot to the multiple-testing grid, which lowers every
        # other q in the same call. The Region Analysis window drops these
        # (`platform_label_cols`, nunique > 1) and the assistant did not: the
        # two agreed on GPL570, which has no such column, and differed by
        # exactly the number of them everywhere else - one on GPL24676, four
        # on the single-cell census. An explicit `label_columns` request is
        # honoured as asked.
        if not wanted:
            cols = [c for c in cols if df[c].nunique(dropna=True) > 1]
        if not cols:
            return ToolResult(
                f"Platform {key!r} carries no label column to test the region "
                f"against.", ok=False)

        masks, cuts, problems = _region_masks(df, genes,
                                              _region_params(resolved))
        if not masks:
            return ToolResult(
                f"No region could be placed on {key!r} for {', '.join(genes)}."
                + ("\n- " + "\n- ".join(problems) if problems else ""),
                ok=False)

        gse = (df["series_id"].astype(str).to_numpy()
               if "series_id" in df.columns else None)

        # The split is the program's, not this tool's: `build_enrichment_cells`
        # is the one the Region Analysis window uses too, so an enrichment the
        # assistant reports and the same enrichment reached by clicking are
        # the same computation and not two that happen to agree.
        from genevariate.core.analysis import build_enrichment_cells

        ids = df["GSM"].astype(str).to_numpy()
        study_of = (dict(zip(ids, gse)) if gse is not None else None)

        cells, groups = {}, {}
        for gene_col, mask in masks.items():
            lo, hi = cuts[gene_col]
            region = f"{str(gene_col).upper()} {lo:.3g}-{hi:.3g}"
            by_col = {c: pd.Series(df[c].to_numpy(), index=ids) for c in cols}
            c_cells, c_groups = build_enrichment_cells(
                by_col, ids[mask], region_name=region, study_of=study_of)
            cells.update(c_cells)
            groups.update(c_groups)
        if not cells:
            return ToolResult(
                f"The region on {key!r} left no background to test against.",
                ok=False)

        progress_cb(30.0, f"Testing {len(cells)} region x label grid(s)…")
        table = region_label_enrichment(cells, groups or None)
        if table.empty:
            return ToolResult(
                f"Nothing in {key!r} was testable: every label value in the "
                f"region records an absent label rather than a value.",
                ok=False)

        report = summarize_region_enrichment(table)
        manifest = _manifest("region_enrichment", resolved,
                             inputs={"platform": df}, seed=0)
        header = "# Region enrichment - " + key + "\n\nRegion(s): " + "; ".join(
            sorted({r for r, _ in cells})) + "\nLabel columns tested: " \
            + ", ".join(cols) + "\n\n"
        if problems:
            header += "Not tested:\n- " + "\n- ".join(problems) + "\n\n"
        note = ("" if gse is not None else
                "\nNo series_id on this platform, so no study-clumping "
                "diagnostics: every q below treats the samples as independent "
                "draws, which they are not.\n")
        report = _append_manifest(header + report + note, manifest)

        show = table[table["significance"] != "ns"]
        show = (show if not show.empty else table).head(25)
        tbl = show[["region", "label_column", "value", "a", "n_region",
                    "region_pct", "background_pct", "enrichment", "q_value",
                    "significance", "n_gse", "replicated"]].copy()
        for c in ("region_pct", "background_pct", "enrichment"):
            tbl[c] = tbl[c].round(2)

        n_sig = int((table["significance"] != "ns").sum())
        fig, figs, block = _enrichment_figures(table)
        report += block
        return ToolResult(
            f"{n_sig} of {len(table)} tested label values are enriched at "
            f"q<0.05 in the {key} region (Fisher exact, one-sided, BH across "
            f"all {len(table)} tests){note.rstrip()}",
            table=tbl, report=report, manifest=manifest, figure=fig,
            figures=figs,
            payload={"enrichment": table, "report": report,
                     "region_window": {
                         "platform": key,
                         "cuts": {c: [float(lo), float(hi)]
                                  for c, (lo, hi) in cuts.items()}}})

    # ---- label-value foreground (Label Enrichment window, 4th strategy) -----
    def _lblenrich_resolver(app, raw):
        out = dict(raw)
        out["platform"] = _match_platform(app, raw.get("platform"))
        if out["platform"]:
            out["label_column"] = _pick_condition_column(
                _labelled(app, out["platform"]), raw.get("label_column"))
        return out

    def _lblenrich_exec(app, resolved, progress_cb):
        from genevariate.core.analysis import (
            region_label_enrichment, summarize_region_enrichment,
        )
        key = resolved.get("platform")
        if not key or key not in _platforms(app):
            return ToolResult("No matching platform is loaded. Load one first.",
                              ok=False)
        df = _labelled(app, key)
        col = resolved.get("label_column")
        value = str(resolved.get("label_value") or "").strip()
        if not col or col not in df.columns:
            return ToolResult(
                f"Platform {key!r} has no label column to take a foreground "
                "from.", ok=False)
        if not value:
            seen = df[col].astype(str).value_counts().head(8)
            return ToolResult(
                f"Name the {col} value to use as the foreground. On {key} the "
                "commonest are: "
                + ", ".join(f"{v} ({n:,})" for v, n in seen.items()) + ".",
                ok=False)

        series = df[col].astype(str)
        mask = series.str.strip().str.lower() == value.lower()
        if not mask.any():
            near = [v for v in series.unique()
                    if value.lower() in str(v).lower()][:6]
            return ToolResult(
                f"No sample on {key} has {col} = {value!r}."
                + (f" Did you mean: {', '.join(near)}?" if near else ""),
                ok=False)
        if (~mask).sum() == 0:
            return ToolResult(
                f"Every sample on {key} has {col} = {value!r}, so there is no "
                "background to test against.", ok=False)

        # Everything except the column the foreground was cut from. Testing
        # Tissue=Liver against Tissue would report that liver samples are
        # enriched for liver, at q=0, as a finding.
        others = [c for c in _classified_cols(df) if c != col and c in df.columns]
        if not others:
            return ToolResult(
                f"{col} is the only label column on {key}, and a foreground "
                f"cut from it can only be tested against the others. Load a "
                "label file carrying more fields.", ok=False)

        gse = (df["series_id"].astype(str).to_numpy()
               if "series_id" in df.columns else None)
        region = f"{col}={value}"
        cells, groups = {}, {}
        for c in others:
            labels = df[c].astype(str)
            cells[(region, c)] = (labels[mask], labels[~mask])
            if gse is not None:
                groups[(region, c)] = list(gse[mask]) + list(gse[~mask])

        progress_cb(35.0, f"Testing {len(others)} label column(s)…")
        table = region_label_enrichment(cells, groups or None)
        if table.empty:
            return ToolResult(
                f"Nothing was testable: the {int(mask.sum()):,} samples with "
                f"{col}={value!r} record no value in any other label column.",
                ok=False)

        manifest = _manifest("label_value_enrichment", resolved,
                             inputs={"platform": df}, seed=0)
        header = (f"# What else describes the {col}={value} samples on {key}\n\n"
                  f"Foreground: {int(mask.sum()):,} samples; background: "
                  f"{int((~mask).sum()):,}.\nColumns tested: "
                  + ", ".join(others) + "\n\n")
        note = ("" if gse is not None else
                "\nNo series_id on this platform, so no study-clumping "
                "diagnostics: every q below treats the samples as independent "
                "draws, which they are not.\n")
        report = _append_manifest(
            header + summarize_region_enrichment(table) + note, manifest)

        show = table[table["significance"] != "ns"]
        show = (show if not show.empty else table).head(25)
        tbl = show[["label_column", "value", "a", "n_region", "region_pct",
                    "background_pct", "enrichment", "q_value", "significance",
                    "n_gse", "replicated"]].copy()
        for c in ("region_pct", "background_pct", "enrichment"):
            tbl[c] = tbl[c].round(2)
        n_sig = int((table["significance"] != "ns").sum())
        fig2, figs2, block2 = _enrichment_figures(table)
        report += block2
        return ToolResult(
            f"{n_sig} of {len(table)} label values are enriched at q<0.05 "
            f"among the {int(mask.sum()):,} {col}={value} samples on {key} "
            f"(one-sided Fisher, BH across all {len(table)} tests)"
            f"{note.rstrip()}",
            table=tbl, report=report, manifest=manifest, figure=fig2,
            figures=figs2,
            payload={"enrichment": table, "n_foreground": int(mask.sum())})

    # ---- compare_distributions (Compare Distributions window) --------------
    def _cmpdist_resolver(app, raw):
        out = dict(raw)
        out["platform"] = _match_platform(app, raw.get("platform"))
        df = _labelled(app, out["platform"]) if out["platform"] else None
        out["group_by"] = _pick_condition_column(
            df, raw.get("group_by")) if df is not None else raw.get("group_by")
        out["max_groups"] = int(raw.get("max_groups") or 12)
        out["min_n"] = int(raw.get("min_n") or 3)
        return out

    def _cmpdist_exec(app, resolved, progress_cb):
        from genevariate.core.analysis import (
            distance_matrix, group_summary, pairwise_distances,
            summarize_comparison_stats,
        )
        from genevariate.core.analysis.region_enrichment import NOT_SPECIFIED
        key = resolved.get("platform")
        if not key or key not in _platforms(app):
            return ToolResult("No matching platform is loaded. Load one first.",
                              ok=False)
        df = _labelled(app, key)
        gene = str(resolved.get("gene") or "").strip()
        vec = _gene_vector(df, gene)
        if vec is None:
            return ToolResult(f"{gene!r} is not a column on {key}.", ok=False)
        col = resolved.get("group_by")
        if not col or col not in df.columns:
            return ToolResult(
                f"{key} carries no label column to group by"
                + (f" (asked for {col!r})." if col else "."), ok=False)

        # Groups are the values of one label column. Values that record the
        # ABSENCE of a label are dropped: "we did not extract a tissue for
        # these 400 samples" is a group of the extractor's making, and letting
        # it into the matrix would put a coverage gap on the same footing as a
        # tissue.
        series = df[col].astype(str)
        groups, dropped_unspec = {}, 0
        for value, idx in series.groupby(series).groups.items():
            if str(value).strip().lower() in NOT_SPECIFIED:
                dropped_unspec += len(idx)
                continue
            v = vec[df.index.get_indexer(idx)]
            v = v[np.isfinite(v)]
            if v.size:
                groups[str(value)] = v
        if len(groups) < 2:
            return ToolResult(
                f"{col!r} on {key} yields fewer than two groups with values, "
                f"so there is nothing to compare.", ok=False)

        # Only the largest groups are compared. Every group added costs the
        # whole matrix a multiplicity penalty, and a group of four samples
        # cannot earn one back.
        order = sorted(groups, key=lambda g: groups[g].size, reverse=True)
        max_groups = int(resolved.get("max_groups") or 12)
        kept, hidden = order[:max_groups], order[max_groups:]
        groups = {g: groups[g] for g in kept}

        progress_cb(40.0, f"Comparing {len(groups)} groups…")
        summary = group_summary(groups)
        pairs = pairwise_distances(groups, min_n=int(resolved.get("min_n") or 3))
        if pairs.empty:
            return ToolResult(
                f"No pair of {col!r} groups on {key} had enough samples to "
                f"compare (need at least {resolved.get('min_n', 3)} each).",
                ok=False)

        try:
            value_label = app.platform_measurement_label(key)
        except Exception:
            value_label = "expression"

        from . import charts
        fig, vdesc = charts.fig_group_violin(
            groups, f"{gene} by {col} - {key}", value_label=value_label)
        # One heatmap per metric, as the window's Distance Matrix tab draws
        # them: the three answer different questions and reading only one of
        # them is how a shape difference gets reported as a shift.
        mats, heatmaps = {}, {}
        for metric in ("wasserstein", "jensen_shannon", "delta_mean"):
            mats[metric] = distance_matrix(pairs, metric, groups=list(groups))
            hfig, _ = charts.fig_distance_heatmap(
                mats[metric],
                f"{metric.replace('_', '-').title()} - {gene} by {col}")
            if hfig is not None:
                heatmaps[metric] = hfig
        matrix = mats["wasserstein"]

        report = summarize_comparison_stats(summary, pairs,
                                            value_label=value_label)
        report += charts.describe_group_violin_block(vdesc)
        notes = []
        if dropped_unspec:
            notes.append(f"{dropped_unspec:,} sample(s) whose {col} records no "
                         f"value were excluded - they are a coverage gap, not "
                         f"a group.")
        if hidden:
            notes.append(f"{len(hidden)} smaller group(s) were not compared: "
                         + ", ".join(hidden[:8])
                         + (", …" if len(hidden) > 8 else "") + ".")
        header = f"# {gene} across {col} - {key}\n\n"
        if notes:
            header += "\n".join(f"- {n}" for n in notes) + "\n\n"
        manifest = _manifest("compare_distributions", resolved,
                             inputs={"platform": df}, seed=0)
        report = _append_manifest(header + report, manifest)

        tbl = pairs[["group_a", "group_b", "n_a", "n_b", "wasserstein",
                     "jensen_shannon", "delta_mean", "p_value", "q_value",
                     "significance", "separation"]].head(40).copy()
        for c in ("wasserstein", "jensen_shannon", "delta_mean"):
            tbl[c] = tbl[c].round(4)

        n_sig = int((pairs["significance"] != "ns").sum())
        far = pairs.iloc[0]
        return ToolResult(
            f"{len(groups)} {col} groups compared on {key}; {n_sig} of "
            f"{len(pairs)} pairs differ at q<0.05 (BH across all pairs). "
            f"Furthest apart: {far['group_a']} vs {far['group_b']}, "
            f"Wasserstein {far['wasserstein']:.3g}.",
            table=tbl, report=report, manifest=manifest, figure=fig,
            figures=heatmaps,
            payload={"summary": summary, "pairs": pairs, "matrix": matrix,
                     "matrices": mats, "chart": vdesc, "report": report})

    # ---- region_composition (what is in there, before any test) -
    def _regcomp_resolver(app, raw):
        out = dict(raw)
        out["platform"] = _match_platform(app, raw.get("platform"))
        out.update(_region_params(raw))
        return out

    def _regcomp_exec(app, resolved, progress_cb):
        from genevariate.core.analysis import (
            region_composition, summarize_region_composition,
        )
        key = resolved.get("platform")
        if not key or key not in _platforms(app):
            return ToolResult("No matching platform is loaded. Load one first.",
                              ok=False)
        df = _labelled(app, key)
        genes = [g.strip() for g in str(resolved.get("genes") or "").split(",")
                 if g.strip()]
        if not genes:
            return ToolResult("Name at least one gene to define a region on.",
                              ok=False)
        wanted = [c.strip() for c in
                  str(resolved.get("label_columns") or "").split(",") if c.strip()]
        cols = [c for c in (wanted or _classified_cols(df)) if c in df.columns]
        if not cols:
            return ToolResult(f"Platform {key!r} carries no label column.",
                              ok=False)

        masks, cuts, problems = _region_masks(df, genes,
                                              _region_params(resolved))
        if not masks:
            return ToolResult(
                f"No region could be placed on {key!r} for {', '.join(genes)}."
                + ("\n- " + "\n- ".join(problems) if problems else ""),
                ok=False)

        cells, studies = {}, {}
        for gene_col, mask in masks.items():
            lo, hi = cuts[gene_col]
            region = f"{str(gene_col).upper()} {lo:.3g}-{hi:.3g}"
            for c in cols:
                labels = df[c].astype(str)
                cells[(region, c)] = (labels[mask], labels[~mask])
            # Which studies the region is actually made of. A region that is
            # 80% one GSE is one experiment, whatever its label composition
            # says, and that has to be visible before the composition is read.
            if "series_id" in df.columns:
                s = df["series_id"][mask].astype(str).value_counts()
                studies[region] = s

        progress_cb(50.0, "Counting the region's composition…")
        table = region_composition(cells)
        report = summarize_region_composition(table)

        # The window's Distributions tab shows the cut against the histogram it
        # was taken from. Two numbers in prose are a claim; the same two numbers
        # drawn on the distribution are one the reader can check.
        from . import charts
        first_gene = next(iter(masks))
        try:
            x_label = app.platform_measurement_label(key)
        except Exception:
            x_label = "Expression"
        vec = _gene_vector(df, first_gene)
        fig, fdesc = charts.fig_histogram(
            vec, str(first_gene), label=key,
            dist_class=_gene_stats(vec).get("distribution", ""),
            x_label=x_label, bounds=cuts[first_gene])

        study_lines = []
        for region, s in studies.items():
            n = int(s.sum())
            top = s.head(5)
            share = 100.0 * float(top.iloc[0]) / n if n else 0.0
            study_lines.append(
                f"**{region}** draws its {n:,} samples from {len(s):,} "
                f"studies; the largest contributes {int(top.iloc[0]):,} "
                f"({share:.1f}%).")
            study_lines.append("  " + ", ".join(
                f"{g} ({int(v):,})" for g, v in top.items())
                + (f", +{len(s) - len(top)} more" if len(s) > len(top) else ""))
        if study_lines:
            report = ("## Which studies the region is made of\n\n"
                      + "\n".join(study_lines) + "\n\n"
                      + "## What the region is made of\n\n" + report)

        manifest = _manifest("region_composition", resolved,
                             inputs={"platform": df}, seed=0)
        header = f"# Region composition - {key}\n\n"
        if problems:
            header += "Not described:\n- " + "\n- ".join(problems) + "\n\n"
        report = _append_manifest(header + report, manifest)

        present = table[table["n_region"] > 0]
        tbl = present[["region", "label_column", "value", "n_region",
                       "region_pct", "n_background", "background_pct",
                       "enrichment", "unspecified"]].head(40).copy()
        for c in ("region_pct", "background_pct", "enrichment"):
            tbl[c] = tbl[c].round(2)

        n_in = int(next(iter(masks.values())).sum())
        total_read = int(table["n_region"].sum()) or 1
        unspec = int(table[table["unspecified"]]["n_region"].sum())
        return ToolResult(
            f"The {key} region holds {n_in:,} samples described over "
            f"{len(cols)} label column(s); {100.0 * unspec / total_read:.1f}% "
            f"of those label readings record no value at all.",
            table=tbl, report=report, manifest=manifest, figure=fig,
            payload={"composition": table, "chart": fdesc,
                     "studies": {k: v.to_dict() for k, v in studies.items()},
                     "report": report})

    # ---- pooled_enrichment (same region rule, every platform) ---
    def _pooled_resolver(app, raw):
        out = dict(raw)
        plats = list(_platforms(app))
        wanted = [p.strip() for p in
                  str(raw.get("platforms") or "").split(",") if p.strip()]
        if wanted:
            picked, seen = [], set()
            for w in wanted:
                k = _match_platform(app, w)
                if k and k not in seen:
                    seen.add(k)
                    picked.append(k)
            out["platforms"] = ",".join(picked)
        else:
            out["platforms"] = ",".join(plats)
        first = plats[0] if plats else None
        if first:
            out["label_column"] = _pick_condition_column(
                _labelled(app, first), raw.get("label_column"))
        out.update(_region_params(raw))
        return out

    def _pooled_exec(app, resolved, progress_cb):
        from genevariate.core.analysis import (
            pooled_label_enrichment, summarize_pooled_enrichment,
        )
        keys = [k for k in str(resolved.get("platforms") or "").split(",")
                if k and k in _platforms(app)]
        if len(keys) < 2:
            return ToolResult(
                "Pooling needs at least two platforms loaded; "
                f"{len(keys)} matched. Load another platform first.", ok=False)
        gene = str(resolved.get("gene") or "").strip()
        if not gene:
            return ToolResult("Name the gene whose region is to be pooled.",
                              ok=False)
        lcol = resolved.get("label_column")
        if not lcol:
            return ToolResult("No label column to pool the region against.",
                              ok=False)

        spec = _region_params(resolved)
        tech = _modalities(app, keys)

        # Each platform is its own stratum: its own samples, its own background,
        # its own study ids and its own cut. Nothing is concatenated - a
        # microarray corpus and a single-cell census are not rows of one table,
        # and stratifying is exactly the refusal to treat them as if they were.
        strata, skipped = {}, []
        for k in keys:
            df = _labelled(app, k)
            if lcol not in df.columns:
                skipped.append(f"{k}: has no '{lcol}' column")
                continue
            masks, cuts, problems = _region_masks(df, [gene], spec)
            if not masks:
                skipped.append(f"{k}: " + (problems[0] if problems
                                           else "no region could be placed"))
                continue
            col = next(iter(masks))
            mask = masks[col]
            keep = df[lcol].notna().to_numpy()
            mask, labels = mask[keep], df[lcol][keep].astype(str).to_numpy(
                dtype=object)
            if not mask.any() or mask.all():
                lo_c, hi_c = cuts[col]
                why = (f"{k}: the region {lo_c:.4g} to {hi_c:.4g} covers "
                       f"{'every' if mask.all() else 'no'} labelled sample, "
                       f"so there is no contrast to test")
                if spec.get("low") is not None or spec.get("high") is not None:
                    why += ("; those are absolute expression bounds and every "
                            "platform here was given the same ones. Pool with "
                            "`quantile` instead, which places the cut on each "
                            "platform's own distribution")
                skipped.append(why)
                continue
            groups = None
            if "series_id" in df.columns:
                g = df["series_id"][keep].astype(str).to_numpy()
                groups = list(g)
            strata[k] = {"in_region": mask, "labels": labels, "groups": groups,
                         "technology": tech.get(k, "unknown"),
                         "_cut": cuts[col]}
        if len(strata) < 2:
            return ToolResult(
                f"Only {len(strata)} platform(s) could contribute a {gene} "
                f"region tested on '{lcol}' - pooling needs two."
                + ("\n- " + "\n- ".join(skipped) if skipped else ""),
                ok=False)

        cuts_by_plat = {k: s.pop("_cut") for k, s in strata.items()}
        progress_cb(40.0, f"Pooling {len(strata)} platforms…")
        res = pooled_label_enrichment(strata)
        rows = res.get("rows") or []
        if not rows:
            return ToolResult(
                f"No label value of '{lcol}' occurs often enough on at least "
                f"two of these platforms to be pooled.", ok=False)

        report = summarize_pooled_enrichment(res, lcol)
        manifest = _manifest("pooled_enrichment", resolved,
                             inputs={k: _labelled(app, k) for k in strata},
                             seed=0)
        header = (f"# Pooled region enrichment - {gene} / {lcol}\n\n"
                  + "Per-platform region:\n"
                  + "\n".join(f"- {k}: {lo:.4g} to {hi:.4g}"
                              for k, (lo, hi) in sorted(cuts_by_plat.items()))
                  + "\n\nThe cut is resolved against each platform's own "
                    "distribution, which is what lets one rule mean the same "
                    "thing on scales that do not compare.\n\n")
        if skipped:
            header += "Not pooled:\n- " + "\n- ".join(skipped) + "\n\n"
        report = _append_manifest(header + report, manifest)

        tbl = pd.DataFrame([
            {"value": r["value"], "k": r["k"],
             "pooled_or": round(r["pooled_or"], 3),
             "ci_low": round(r["ci_low"], 3), "ci_high": round(r["ci_high"], 3),
             "q": r["q"], "i2": round(r["i2"], 1),
             "same_sign": f"{r['n_same_sign']}/{r['k']}",
             "concordant": r["concordant"]}
            for r in rows[:25]])
        n_sig = sum(1 for r in rows if np.isfinite(r["q"]) and r["q"] < 0.05)
        n_conc = sum(1 for r in rows if r["concordant"])
        return ToolResult(
            f"Pooled {gene} regions over {len(strata)} platforms on '{lcol}': "
            f"{n_sig} of {len(rows)} values pass FDR, {n_conc} of them "
            f"concordant across every platform tested.",
            table=tbl, report=report, manifest=manifest,
            payload={"pooled": res, "cuts": cuts_by_plat, "report": report})

    # ---- region_box_model (how well does the box predict a label?)
    # ---- classify_distributions (modality landscape) ------------
    def _modality_resolver(app, raw):
        out = dict(raw)
        out["platform"] = _match_platform(app, raw.get("platform"))
        try:
            out["max_genes"] = int(raw.get("max_genes") or 2000)
        except (TypeError, ValueError):
            out["max_genes"] = 2000
        return out

    def _modality_exec(app, resolved, progress_cb):
        from genevariate.core.analysis import (
            classify_distributions, distribution_summary,
        )
        key = resolved.get("platform")
        plats = _platforms(app)
        if not key or key not in plats:
            return ToolResult("No matching platform is loaded. Load one first.",
                              ok=False)
        df = plats[key]
        # classify_distributions treats every non-GSM/series_id column as a
        # gene, so drop the Classified_* metadata columns first.
        drop = _classified_cols(df)
        expr_df = df.drop(columns=drop) if drop else df
        # Each gene costs a dip test plus a GMM fit, so a whole transcriptome
        # is minutes of work for a class-proportion answer a sample settles.
        # Subsample deterministically, and say so rather than implying the
        # landscape was measured exhaustively.
        max_genes = int(resolved.get("max_genes") or 0)
        genes = [c for c in expr_df.columns if c not in ("GSM", "series_id")]
        subset = None
        sampled = max_genes > 0 and len(genes) > max_genes
        if sampled:
            rng = np.random.default_rng(0)
            subset = [genes[i] for i in
                      sorted(rng.choice(len(genes), max_genes, replace=False))]
        progress_cb(30.0, f"Classifying gene distributions on {key}…")
        tags = classify_distributions(expr_df, subset=subset)
        if tags is None or tags.empty:
            return ToolResult(f"No gene columns to classify on {key!r}.", ok=False)
        summary = distribution_summary(tags)
        progress_cb(80.0, "Summarising modality landscape…")
        n_genes = int(len(tags))
        scope = (f"a random {n_genes:,} of {len(genes):,} genes" if sampled
                 else f"{n_genes:,} genes")
        top = summary.sort_values("n_genes", ascending=False)
        parts = [f"{r['distribution']} {r['fraction'] * 100:.1f}%"
                 for _, r in top.head(4).iterrows()]
        lines = [f"# Distribution landscape - {key}\n",
                 f"Classified **{scope}** into "
                 f"{len(summary)} modality class(es).\n"]
        for _, r in top.iterrows():
            lines.append(f"- **{r['distribution']}**: {int(r['n_genes']):,} genes "
                         f"({r['fraction'] * 100:.1f}%)")
        report = "\n".join(lines)
        return ToolResult(
            f"Modality landscape on {key}: {scope} - "
            + ", ".join(parts) + ".",
            table=top, report=report,
            payload={"platform": key, "tags": tags, "summary": summary,
                     "report": report})

    # ---- label_entities -----------------------------------------
    def _ent_resolver(app, raw):
        out = dict(raw)
        out["platform"] = _match_platform(app, raw.get("platform"))
        return out

    def _ent_exec(app, resolved, progress_cb):
        frames = _entity_frames(app, resolved.get("platform") or None)
        if not frames:
            return ToolResult(
                "No loaded labels carry entity links. Phase 1 and phase 1b "
                "record the value verbatim; the MeSH / Cellosaurus accessions "
                "are written by the phase 2 normalization pass, so re-run the "
                "extractor with normalization or load a normalized label file.",
                ok=False)

        progress_cb(40.0, "Reading the entity links\u2026")
        parts = []
        for name, df in sorted(frames.items()):
            tbl = label_entities.entity_table(df)
            if tbl.empty:
                continue
            tbl.insert(0, "platform", name)
            parts.append(tbl)
        if not parts:
            return ToolResult("The loaded labels resolved to no entities.",
                              ok=False)
        table = pd.concat(parts, ignore_index=True)

        field = str(resolved.get("field") or "").strip()
        if field:
            table = table[table["Field"].str.lower() == field.lower()]
        source = str(resolved.get("source") or "").strip()
        if source:
            table = table[table["Source"].str.lower().str.startswith(
                source.lower())]
        if table.empty:
            return ToolResult("No label value matches that field/vocabulary.",
                              ok=False)

        cells = table[table["Source"] == label_entities.CELLOSAURUS]
        counts = table.groupby("Source")["n"].sum().to_dict()
        stage = ", ".join(sorted({label_entities.linked_stage(d)
                                  for d in frames.values()} - {""}))
        summ = (f"{len(table)} distinct label value(s) across "
                f"{len(frames)} platform(s), linked at {stage or 'phase2'}: "
                + ", ".join(f"{int(v):,} samples {k}"
                            for k, v in sorted(counts.items()))
                + f". {cells['Value'].nunique()} value(s) are catalogued cell "
                  "lines, not tissue.")
        lines = [f"# Entity links ({stage or 'phase2'})\n",
                 "A label value is not only a word: the normalization pass "
                 "records which vocabulary answered. `CVCL_*` is a Cellosaurus "
                 "registration, so the sample is a catalogued **cell line** "
                 "rather than a piece of tissue; `D######` is a MeSH heading; "
                 "an `ART-*` identifier was minted locally because nothing "
                 "recognised the value.\n"]
        if not cells.empty:
            lines.append("**Cell lines found in Tissue:**\n")
            for r in cells.sort_values("n", ascending=False).itertuples(
                    index=False):
                lines.append(f"- {r.Value} ({r.Accession}) - {int(r.n):,} "
                             f"samples on {r.platform}")
            lines.append(
                f"\nSplit them from tissue with the derived label column "
                f"`{label_entities.kind_column_name()}` "
                f"(`{label_entities.KIND_TISSUE}` / "
                f"`{label_entities.KIND_CELL_LINE}` / "
                f"`{label_entities.KIND_MIXED}` / "
                f"`{label_entities.KIND_UNRESOLVED}`), for example "
                f"`gene_distribution(by_label=\""
                f"{label_entities.kind_column_name()}\")`.")
        return ToolResult(
            summ, table=table.head(50), report="\n".join(lines),
            payload={"entities": table,
                     "cell_line_values": sorted(set(cells["Value"])),
                     "kind_column": label_entities.kind_column_name(),
                     "stage": stage})

    # The three interchangeable ways of naming a region, offered by every tool
    # that takes one. They are listed as separate optional parameters rather
    # than one string so the model cannot invent a syntax: it either gives two
    # numbers, a count of standard deviations, or a quantile.
    def _region_toolparams(what: str) -> List[ToolParam]:
        return [
            ToolParam("low", "float", required=False,
                      help=f"Lower expression bound of {what}, in the "
                           f"platform's own units - the number a drag-selection "
                           f"in the Gene Distribution Explorer would give. "
                           f"Overrides `sd` and `quantile`."),
            ToolParam("high", "float", required=False,
                      help=f"Upper expression bound of {what}. Defaults to the "
                           f"gene's observed maximum."),
            ToolParam("sd", "float", required=False,
                      help=f"Define {what} as the mean + this many standard "
                           f"deviations, up to the maximum (e.g. 2 or 3). "
                           f"Overrides `quantile`."),
            ToolParam("quantile", "float", required=False, default=0.8,
                      help=f"Define {what} as the gene's own upper quantile "
                           f"(0.9 = top decile). Used when no bounds or `sd` "
                           f"are given."),
        ]

    tools: Dict[str, Tool] = {}

    tools["label_entities"] = Tool(
        name="label_entities",
        description="Report what each extracted label resolved to in phase 2 "
                    "(MeSH heading, Cellosaurus cell line, or a locally minted "
                    "out-of-vocabulary id), and name which Tissue values are "
                    "catalogued cell lines rather than tissue.",
        params=[
            ToolParam("platform", "platform", required=False,
                      help="Platform whose labels to read (default: all)."),
            ToolParam("field", "str", required=False,
                      choices=label_entities.ENTITY_FIELDS,
                      help="Restrict to one label field."),
            ToolParam("source", "str", required=False,
                      choices=(label_entities.MESH,
                               label_entities.CELLOSAURUS,
                               label_entities.LOCAL),
                      help="Restrict to one vocabulary."),
        ],
        resolver=_ent_resolver, executor=_ent_exec,
        examples=("which tissue labels are cell lines",
                  "show the cellosaurus entity links",
                  "what did phase 2 resolve the labels to",
                  "list the mesh ids for the tissue labels",
                  "which samples are cell lines rather than tissue"))

    tools["list_platforms"] = Tool(
        name="list_platforms",
        description="List the gene-expression platforms currently loaded.",
        params=[],
        resolver=_list_resolver, executor=_list_exec,
        examples=("what platforms are loaded", "list datasets",
                  "show my platforms"))

    cond_params = [
        ToolParam("platform", "platform", help="Loaded platform to analyse."),
        ToolParam("condition_column", "str", required=False,
                  help="Label column defining the groups "
                       "(e.g. Condition/Tissue/Sex)."),
        ToolParam("case_label", "str", help="Group treated as 'case'."),
        ToolParam("control_label", "str", help="Group treated as 'control'."),
        ToolParam("libraries", "str", required=False, default=libs_default,
                  help="Comma-separated Enrichr gene-set libraries."),
        ToolParam("moderated", "bool", required=False, default=False,
                  help="Use empirical-Bayes moderated variance."),
    ]
    tools["condition_enrichment"] = Tool(
        name="condition_enrichment",
        description="Rank genes case-vs-control and run prerank GSEA.",
        params=cond_params,
        resolver=_cond_resolver, executor=_cond_exec,
        examples=("run condition enrichment on GPL570 tumor vs normal",
                  "gsea case vs control", "enrichment tumour versus healthy"))

    tools["region_comparison"] = Tool(
        name="region_comparison",
        description="Compare the high-expression regions of several genes "
                    "against each other: assigns each label to the region it "
                    "belongs to, tests the regions against one another rather "
                    "than only against the background, and flags region-specific "
                    "effects carried by a few studies.",
        params=[
            ToolParam("platform", "platform", required=False,
                      help="Platform to compare on (defaults to the first loaded)."),
            ToolParam("genes", "str",
                      help="Comma-separated gene symbols; each becomes one region."),
            ToolParam("condition_column", "str", required=False,
                      help="Label column the regions are compared on."),
            *_region_toolparams("each gene's region"),
        ],
        resolver=_regcmp_resolver, executor=_regcmp_exec,
        examples=("compare regions for TP53, MKI67 and GAPDH on GPL570",
                  "region comparison across gene regions",
                  "compare the high expression regions of these genes"))

    tools["region_enrichment"] = Tool(
        name="region_enrichment",
        description="Report what a gene's high-expression region is enriched "
                    "for and at what q: a one-sided Fisher exact test of every "
                    "label value inside the region against the rest of the "
                    "loaded samples, Benjamini-Hochberg across the whole grid "
                    "tested, and for each surviving hit the number of "
                    "contributing studies and an enrichment interval "
                    "bootstrapped over studies rather than samples. This is the "
                    "Region Analysis window's Enrichment tab.",
        params=[
            ToolParam("platform", "platform", required=False,
                      help="Platform to test on (defaults to the first loaded)."),
            ToolParam("genes", "str",
                      help="Comma-separated gene symbols; each defines one "
                           "region, tested separately."),
            ToolParam("label_columns", "str", required=False,
                      help="Comma-separated label columns to test (default: "
                           "every label column on the platform). The correction "
                           "is applied across all of them at once, so a q value "
                           "only means anything alongside the grid it was "
                           "corrected over."),
            *_region_toolparams("the region"),
        ],
        resolver=_regenrich_resolver, executor=_regenrich_exec,
        # Examples steer routing, so they are phrased as shapes of question
        # rather than as one worked case. Four variants of a single gene,
        # platform and pair of bounds teach the router that case instead of
        # the pattern, and a session about a different gene is then routed on
        # the strength of a resemblance it does not have.
        examples=("what is this gene's high-expression region enriched for",
                  "region enrichment q values for a gene on a platform",
                  "which tissues are over-represented in the top decile of a gene",
                  "enrichment of the region between two expression bounds",
                  "which labels are over-represented above the mean + 2 SD"))

    tools["label_value_enrichment"] = Tool(
        name="label_value_enrichment",
        description="Take one group of samples defined by a LABEL VALUE - "
                    "every sample whose Tissue is liver, whose Condition is "
                    "tumour - and report what ELSE describes them: every "
                    "value of every OTHER label column tested against the "
                    "rest of the platform with a one-sided Fisher exact test, "
                    "BH-corrected over the whole grid and widened for study "
                    "clumping. Answers 'what are the liver samples also', "
                    "'what conditions come with this tissue', 'describe the "
                    "tumour samples'. The foreground is a label here, not an "
                    "expression region - for a gene's high-expression region "
                    "use region_enrichment.",
        params=[
            ToolParam("platform", "platform", required=False,
                      help="Platform to test on (defaults to the first loaded)."),
            ToolParam("label_column", "str", required=False,
                      choices=LABEL_CHOICES,
                      help="Column the foreground is cut from."),
            ToolParam("label_value", "str",
                      help="Value of that column defining the foreground "
                           "(e.g. liver, tumour, female)."),
        ],
        resolver=_lblenrich_resolver, executor=_lblenrich_exec,
        examples=("what else describes the liver samples on GPL96",
                  "what conditions come with Tissue=Lung",
                  "characterise the tumour samples",
                  "what are the female samples enriched for",
                  "label enrichment for Condition=control"))

    tools["compare_distributions"] = Tool(
        name="compare_distributions",
        description="Compare one gene's distribution across the groups a label "
                    "column defines: per-group N/mean/median/SD/IQR, and for "
                    "every pair the Wasserstein distance (in expression "
                    "units), the Jensen-Shannon divergence (shape), the "
                    "difference of means, a Wilcoxon rank-sum test and a "
                    "BH q corrected across every pair. Returns the violin plot "
                    "and the distance matrix. This is the Compare "
                    "Distributions window's Distance Matrix, Separation and "
                    "Statistics tabs. Use it for 'how different are these "
                    "groups' / 'which tissues separate'.",
        params=[
            ToolParam("platform", "platform", required=False,
                      help="Platform to compare on (defaults to the first loaded)."),
            ToolParam("gene", "str",
                      help="Gene whose distribution is compared."),
            ToolParam("group_by", "str", required=False, choices=LABEL_CHOICES,
                      help="Label column whose values become the groups."),
            ToolParam("max_groups", "int", required=False, default=12,
                      help="Compare only the N largest groups (default 12)."),
            ToolParam("min_n", "int", required=False, default=3,
                      help="Skip groups with fewer than this many samples."),
        ],
        resolver=_cmpdist_resolver, executor=_cmpdist_exec,
        examples=("compare the ALB distribution across tissues",
                  "how different is ALB between conditions",
                  "which tissues separate on ALB expression",
                  "pairwise distance matrix for ALB by tissue",
                  "violin plot of ALB by condition",
                  "wasserstein distance between the groups"))

    tools["region_composition"] = Tool(
        name="region_composition",
        description="Describe what a gene's high-expression region is actually "
                    "made of, before any test: every label value counted inside "
                    "and outside, the share of the region that carries no value "
                    "at all, the values the region excludes entirely, and which "
                    "studies contribute its samples. This is the Region "
                    "Analysis window's Frequency and Samples tabs. Use it to "
                    "describe a region; use `region_enrichment` to test it.",
        params=[
            ToolParam("platform", "platform", required=False,
                      help="Platform to describe (defaults to the first loaded)."),
            ToolParam("genes", "str",
                      help="Comma-separated gene symbols; each defines one region."),
            ToolParam("label_columns", "str", required=False,
                      help="Comma-separated label columns to break down "
                           "(default: every label column on the platform)."),
            *_region_toolparams("the region"),
        ],
        resolver=_regcomp_resolver, executor=_regcomp_exec,
        examples=("what is in the ALB high-expression region",
                  "breakdown of the samples in the ALB region",
                  "how many samples in the region have no tissue recorded",
                  "which studies make up this region",
                  "frequency of every label in the top decile of ALB"))

    tools["pooled_enrichment"] = Tool(
        name="pooled_enrichment",
        description="Apply one region rule to the same gene on every loaded "
                    "platform and pool the label enrichments across them: "
                    "DerSimonian-Laird random-effects odds ratio per label "
                    "value, its BH-adjusted q, the between-platform "
                    "heterogeneity (tau^2, I^2, Cochran's Q), and whether every "
                    "platform points the same way. Each platform is its own "
                    "stratum with its own cut and background - nothing is "
                    "concatenated across scales. Define the region with "
                    "`quantile` here: absolute bounds are one platform's "
                    "numbers and would place the same interval on every "
                    "scale, which selects nothing on the others.",
        params=[
            ToolParam("gene", "str",
                      help="Gene whose high-expression region is pooled."),
            ToolParam("platforms", "str", required=False,
                      help="Comma-separated platforms to pool (default: every "
                           "loaded platform). At least two are needed."),
            ToolParam("label_column", "str", required=False,
                      choices=LABEL_CHOICES,
                      help="Label column the region is tested against on every "
                           "platform."),
            *_region_toolparams("the region on each platform"),
        ],
        resolver=_pooled_resolver, executor=_pooled_exec,
        examples=("pool the ALB region across platforms",
                  "is the liver enrichment consistent across platforms",
                  "cross-platform pooled odds ratio for the ALB region",
                  "does this region replicate on the other platform"))

    # ---- cross-platform: what the platforms share, and where they disagree --
    def _xplat_keys(app, raw):
        """The platforms named, or every loaded one, as real dataset keys."""
        plats = list(_platforms(app))
        wanted = [p.strip() for p in
                  str(raw.get("platforms") or "").split(",") if p.strip()]
        if not wanted:
            return plats
        picked, seen = [], set()
        for w in wanted:
            k = _match_platform(app, w)
            if k and k not in seen:
                seen.add(k)
                picked.append(k)
        return picked

    def _xplat_resolver(app, raw):
        out = dict(raw)
        out["platforms"] = ",".join(_xplat_keys(app, raw))
        return out

    def _overlap_exec(app, resolved, progress_cb):
        from genevariate.core.analysis import gene_inventory
        keys = [k for k in str(resolved.get("platforms") or "").split(",")
                if k and k in _platforms(app)]
        if len(keys) < 2:
            return ToolResult(
                "Comparing gene coverage needs at least two platforms loaded; "
                f"{len(keys)} matched. Load another platform first.", ok=False)

        maps = getattr(app, "gpl_gene_mappings", {}) or {}
        gene_sets = {k: set(maps.get(k, {}).keys()) for k in keys}
        missing = [k for k in keys if not gene_sets[k]]
        if missing:
            return ToolResult(
                f"No gene map for {', '.join(missing)} - the platform is loaded "
                "but its probes have not been mapped to symbols, so there is "
                "nothing to intersect.", ok=False)

        progress_cb(40.0, "Intersecting gene inventories…")
        inv = gene_inventory(gene_sets)

        tbl = pd.DataFrame(
            [{"platform": k, "genes": len(gene_sets[k]),
              "only_here": len(inv["unique_genes"][k]),
              "common_to_all": len(inv["common_all"])} for k in keys])

        # A count of shared genes is not a rate. Jaccard is, and it is the one
        # number that says whether two platforms are close to interchangeable
        # or merely both large.
        mat = pd.DataFrame(0.0, index=keys, columns=keys)
        for a in keys:
            for b in keys:
                if a == b:
                    mat.loc[a, b] = 1.0
                else:
                    union = len(gene_sets[a] | gene_sets[b])
                    mat.loc[a, b] = (len(gene_sets[a] & gene_sets[b]) / union
                                     if union else 0.0)

        from . import charts
        fig, _ = charts.fig_distance_heatmap(
            mat, "Gene-inventory overlap (Jaccard)")
        bar, _ = charts.fig_bar(
            [f"{k} only" for k in keys],
            [len(inv["unique_genes"][k]) for k in keys],
            "Genes carried by one platform alone", xlabel="genes",
            top=len(keys))

        n_all = len(inv["all_genes"])
        n_common = len(inv["common_all"])
        report = [f"# Gene coverage across {len(keys)} platforms\n",
                  f"- {n_all:,} distinct symbols in total; {n_common:,} "
                  f"({n_common / max(n_all, 1):.1%}) are measured on every "
                  f"platform.",
                  "- Only those common genes can carry a claim made about all "
                  "of them at once. Anything below is measured on a subset, "
                  "and a gene missing from a platform is missing, not absent "
                  "from the biology."]
        for k in keys:
            uni = len(inv["unique_genes"][k])
            report.append(f"- {k}: {len(gene_sets[k]):,} genes, {uni:,} of "
                          f"which no other loaded platform carries.")
        report.append("\n## Pairwise\n")
        for (a, b), shared in inv["pairwise_overlap"].items():
            union = len(gene_sets[a] | gene_sets[b])
            report.append(f"- {a} & {b}: {len(shared):,} shared, Jaccard "
                          f"{len(shared) / max(union, 1):.3f}.")
        manifest = _manifest("platform_gene_overlap", resolved,
                             inputs={k: _platforms(app)[k] for k in keys})
        return ToolResult(
            f"{n_common:,} of {n_all:,} genes are measured on all "
            f"{len(keys)} platforms ("
            f"{n_common / max(n_all, 1):.1%}).",
            table=tbl, report=_append_manifest("\n".join(report), manifest),
            manifest=manifest, figure=fig, figures={"unique_genes": bar},
            payload={"common_all": sorted(inv["common_all"]),
                     "unique_genes": {k: sorted(v) for k, v
                                      in inv["unique_genes"].items()},
                     "jaccard": mat})

    def _xplat_de_resolver(app, raw):
        out = _xplat_resolver(app, raw)
        method = str(raw.get("batch_method") or "none").strip().lower()
        if method not in ("none", "median_centering", "quantile_normalization",
                          "combat"):
            method = "none"
        out["batch_method"] = method
        out["reference"] = str(raw.get("reference") or "(auto)").strip()
        out["pval_threshold"] = float(raw.get("pval_threshold") or 0.05)
        out["delta_threshold"] = float(raw.get("delta_threshold") or 0.5)
        return out

    def _xplat_de_exec(app, resolved, progress_cb):
        from genevariate.core.analysis import (
            analyze_platforms, summarize_cross_platform,
        )
        keys = [k for k in str(resolved.get("platforms") or "").split(",")
                if k and k in _platforms(app)]
        if len(keys) < 2:
            return ToolResult(
                "A cross-platform comparison needs at least two platforms "
                f"loaded; {len(keys)} matched. Load another first.", ok=False)

        maps = getattr(app, "gpl_gene_mappings", {}) or {}
        if any(not maps.get(k) for k in keys):
            return ToolResult(
                "At least one platform has no gene map, so its columns cannot "
                "be matched to the others by symbol.", ok=False)

        ref = resolved.get("reference") or "(auto)"
        if ref != "(auto)":
            ref = _match_platform(app, ref) or "(auto)"

        facts = getattr(app, "_platform_facts", None)
        tech = {}
        for k in keys:
            try:
                tech[k] = facts(k).get("category", "unknown")
            except Exception:
                tech[k] = "unknown"
        species = {}
        for k in keys:
            try:
                species[k] = app.platform_species(k)
            except Exception:
                species[k] = "unknown"

        results = analyze_platforms(
            keys,
            {k: _platforms(app)[k] for k in keys},
            {k: maps.get(k, {}) for k in keys},
            species=species, technology=tech,
            labels={k: (getattr(app, "platform_labels", {}) or {}).get(k)
                    for k in keys},
            reference=ref,
            batch_method=resolved.get("batch_method", "none"),
            pval_threshold=float(resolved.get("pval_threshold", 0.05)),
            delta_threshold=float(resolved.get("delta_threshold", 0.5)),
            progress_cb=progress_cb)

        if not results["gene_stats"]:
            return ToolResult(
                "No gene was measured on two of these platforms with enough "
                "samples to test. Check that the platforms share a gene map.",
                ok=False)

        rows = []
        for g in results["de_genes"][:40]:
            rows.append({"gene": g["gene"], "q": round(g["adj_pval"], 6),
                         "max_abs_delta": round(g["max_abs_delta"], 4),
                         "n_platforms": g["n_platforms"],
                         "reference": g["ref_platform"]})
        tbl = pd.DataFrame(rows) if rows else pd.DataFrame(
            [{"gene": g["gene"], "max_pval": round(g["max_pval"], 4),
              "max_abs_delta": round(g["max_abs_delta"], 4),
              "n_platforms": g["n_platforms"]}
             for g in results["conserved_genes"][:40]])

        rho = pd.DataFrame(np.nan, index=keys, columns=keys)
        for k in keys:
            rho.loc[k, k] = 1.0
        for (a, b), info in results["plat_correlations"].items():
            rho.loc[a, b] = rho.loc[b, a] = info["spearman"]

        from . import charts
        # A pair with no correlation to report is left blank rather than drawn
        # as 0.0, which on this scale reads as "measured, and unrelated"
        # instead of "not measured".
        fig, _ = charts.fig_distance_heatmap(
            rho, "Platform agreement (Spearman rho on gene means)")
        bar, _ = charts.fig_bar(
            [g["gene"] for g in results["de_genes"][:12]],
            [g["max_abs_delta"] for g in results["de_genes"][:12]],
            "Largest cross-platform differences", xlabel="max |delta mean|")

        manifest = _manifest("cross_platform_comparison", resolved,
                             inputs={k: _platforms(app)[k] for k in keys})
        report = _append_manifest(summarize_cross_platform(results), manifest)

        headline = (f"{results['n_tested']:,} genes tested across {len(keys)} "
                    f"platforms: {results['n_de']:,} differ (BH q < "
                    f"{results['pval_threshold']}), "
                    f"{results['n_conserved']:,} are conserved.")
        if results.get("is_cross_technology"):
            headline += (" These platforms measure different quantities, so "
                         "read the rankings, not the values.")
        figs = {"largest_differences": bar} if bar is not None else {}
        return ToolResult(
            headline, table=tbl, report=report, manifest=manifest,
            figure=fig, figures=figs,
            payload={"n_de": results["n_de"],
                     "n_conserved": results["n_conserved"],
                     "reference": results["reference"],
                     "batch_effect_score": results["batch_effect_score"],
                     "batch_correction_used": results["batch_correction_used"],
                     "de_genes": [g["gene"] for g in results["de_genes"]],
                     "conserved_genes": [g["gene"]
                                         for g in results["conserved_genes"]],
                     "spearman": rho})

    tools["platform_gene_overlap"] = Tool(
        name="platform_gene_overlap",
        description="Compare what the loaded platforms MEASURE, before any "
                    "expression is read: how many gene symbols each carries, "
                    "how many are common to all of them, the pairwise overlap "
                    "and Jaccard index, and the genes only one platform "
                    "carries. Cheap - it touches the gene maps, not the data. "
                    "Use it for 'which genes do these platforms share', 'how "
                    "much do the platforms overlap', or before any claim made "
                    "about all platforms at once.",
        params=[
            ToolParam("platforms", "str", required=False,
                      help="Comma-separated platforms to compare (default: "
                           "every loaded platform). At least two are needed."),
        ],
        resolver=_xplat_resolver, executor=_overlap_exec,
        examples=("which genes do GPL570 and GPL96 share",
                  "how much do the loaded platforms overlap",
                  "gene coverage across platforms",
                  "what is unique to GPL570",
                  "how many genes are measured on every platform"))

    tools["cross_platform_comparison"] = Tool(
        name="cross_platform_comparison",
        description="Run the full Cross-Platform Analysis over the loaded "
                    "platforms: for every shared gene, each platform is tested "
                    "against a reference with Mann-Whitney and KS, the p is "
                    "widened by the study design effect, the k-1 tests per gene "
                    "are Sidak-corrected into one, and BH runs across genes. "
                    "Returns the genes that DIFFER between platforms, the ones "
                    "CONSERVED across them, the batch-effect score, and the "
                    "Spearman agreement between every pair. Optionally applies "
                    "median centering, per-gene quantile normalization or "
                    "ComBat (protecting the extracted labels). Use it for "
                    "'do these platforms agree', 'which genes are platform-"
                    "specific', 'is there a batch effect', 'which genes are "
                    "conserved across platforms'.",
        params=[
            ToolParam("platforms", "str", required=False,
                      help="Comma-separated platforms (default: every loaded "
                           "one). At least two are needed."),
            ToolParam("reference", "str", required=False, default="(auto)",
                      help="Platform every other is tested against. Default "
                           "'(auto)' picks the one with the most mapped genes."),
            ToolParam("batch_method", "str", required=False, default="none",
                      choices=("none", "median_centering",
                               "quantile_normalization", "combat"),
                      help="Correction to apply first. 'none' compares the "
                           "values as loaded. quantile_normalization removes "
                           "differences in distribution shape, which is what "
                           "this program measures elsewhere - use it "
                           "deliberately."),
            ToolParam("pval_threshold", "float", required=False, default=0.05,
                      help="BH q below which a gene counts as differing."),
            ToolParam("delta_threshold", "float", required=False, default=0.5,
                      help="Minimum |difference of means| for a gene to count "
                           "as differing, on top of the q."),
        ],
        resolver=_xplat_de_resolver, executor=_xplat_de_exec,
        examples=("do GPL570 and GPL96 agree",
                  "which genes are platform-specific",
                  "cross-platform differential expression",
                  "which genes are conserved across platforms",
                  "is there a batch effect between the platforms",
                  "run the cross-platform analysis with combat"))

    # ---- bimodality-gated enrichment ---------------------------------------
    # ---- pseudo-cohorts ----------------------------------------------------
    # ---- over-representation of a gene list --------------------------------
    tools["classify_distributions"] = Tool(
        name="classify_distributions",
        description="Classify genes on a platform by their expression "
                    "distribution (unimodal/bimodal/heavy-tailed) and summarise "
                    "the modality landscape over a random sample of genes.",
        params=[
            ToolParam("platform", "platform", required=False,
                      help="Platform to profile (defaults to the first loaded)."),
            ToolParam("max_genes", "int", required=False, default=2000,
                      help="Cap on genes classified; a random sample is taken "
                           "above it. 0 classifies the whole platform, which "
                           "takes minutes on a transcriptome."),
        ],
        resolver=_modality_resolver, executor=_modality_exec,
        examples=("classify the gene distributions on GPL570",
                  "show the modality landscape of my platform",
                  "how many genes are bimodal"))

    # ---- gene_distribution --------------------------------------
    def _dist_resolver(app, raw):
        out = dict(raw)
        out["platform"] = _match_platform(app, raw.get("platform"))
        out.update(_region_params(raw))
        return out

    def _dist_exec(app, resolved, progress_cb):
        gene = str(resolved.get("gene") or "").strip()
        if not gene:
            return ToolResult("Which gene? Provide a gene symbol.", ok=False)
        key = resolved.get("platform")
        plats = _platforms(app)
        if not key or key not in plats:
            return ToolResult("No matching platform is loaded. Load one first "
                              "(load_geo_platform / fetch_single_cell).", ok=False)
        by_label = str(resolved.get("by_label") or "").strip()
        if by_label:
            from genevariate.core.analysis import (
                stratified_distribution, fig_stratified,
            )
            progress_cb(40.0, f"Stratifying {gene} by {by_label} on {key}…")
            try:
                res = stratified_distribution(_labelled(app, key), gene, None,
                                              by_label)
            except ValueError as e:
                return ToolResult(str(e), ok=False)
            summ = (f"{gene} on {key} stratified by {res['label']}: "
                    f"Kruskal-Wallis p={res['kruskal']['p']:.3g} "
                    f"({'significant' if res['significant'] else 'n.s.'}) "
                    f"across {len(res['groups'])} groups.")
            return ToolResult(summ, table=res["table"], report=res["report"],
                              figure=fig_stratified(res),
                              payload={"gene": gene, "platform": key,
                                       "label": res["label"],
                                       "kruskal": res["kruskal"],
                                       "anova": res["anova"],
                                       "significant": res["significant"],
                                       "report": res["report"]})
        progress_cb(40.0, f"Profiling {gene} on {key}…")
        vec = _gene_vector(plats[key], gene)
        if vec is None:
            return ToolResult(f"Gene {gene!r} not found on platform {key!r}.",
                              ok=False)
        stats = _gene_stats(vec)
        tbl = pd.DataFrame([{"platform": key, "gene": gene, **stats}])
        summ = (f"{gene} on {key}: {stats.get('distribution', '?')} "
                f"(n={stats.get('n', 0)}, mean={stats.get('mean', float('nan')):.3g}, "
                f"median={stats.get('median', float('nan')):.3g}, "
                f"std={stats.get('std', float('nan')):.3g}).")
        cv = (stats.get("std", 0.0) / stats.get("mean", 1.0)
              if stats.get("mean") else float("nan"))
        report = (f"# {gene} distribution - {key}\n\n"
                  f"- **Class**: {stats.get('distribution', '?')}\n"
                  f"- **Samples**: {stats.get('n', 0):,}\n"
                  f"- **Mean / median**: {stats.get('mean', float('nan')):.3g} / "
                  f"{stats.get('median', float('nan')):.3g}\n"
                  f"- **Std (CV)**: {stats.get('std', float('nan')):.3g} "
                  f"({cv:.2f})\n"
                  f"- **Range**: {stats.get('min', float('nan')):.3g} - "
                  f"{stats.get('max', float('nan')):.3g}\n")
        from . import charts
        # The Explorer names its x axis after what the platform measures (an
        # array intensity, a sequencing count, a sum or a mean of cells), and a
        # figure that says only "expression" hides the very difference the
        # cross-platform comparisons turn on.
        try:
            x_label = app.platform_measurement_label(key)
        except Exception:
            x_label = "Expression"
        # A region asked for on the histogram has to be drawn *on* the
        # histogram. ``fig_histogram`` already shades one; not passing the
        # bounds here is what made this tool answer "here is the distribution"
        # to a question that said "with that region marked on top of it".
        bounds = None
        spec = _region_params(resolved)
        if any(spec.get(k) is not None for k in ("low", "high", "sd")):
            finite = vec[np.isfinite(vec)] if hasattr(vec, "__len__") else vec
            try:
                from genevariate.core.analysis import resolve_bounds
                rb = resolve_bounds(np.asarray(finite, dtype=float), **spec)
                if not rb.empty:
                    bounds = (rb.low, rb.high)
            except (ValueError, TypeError) as exc:
                report += f"\n\n_Region not drawn: {exc}_\n"

        fig, desc = charts.fig_histogram(vec, gene, label=key,
                                         dist_class=stats.get("distribution", ""),
                                         x_label=x_label, bounds=bounds)
        report += charts.describe_distribution_block(desc)
        if bounds is not None:
            # The count has to be in the sentence, not only in the payload.
            # A caller reading this text is told the region was drawn and is
            # given no number for it, so the only figure in reach is the
            # distribution's own n - which is how a region of 310 samples ends
            # up reported as 3,093.
            reg = (desc or {}).get("region") or {}
            n_in, pct = reg.get("n_in"), reg.get("pct_in")
            summ += (f" Region {bounds[0]:.4g}-{bounds[1]:.4g} marked on the "
                     f"distribution")
            summ += (f": {n_in:,} of {len(vec):,} samples fall inside it "
                     f"({pct:.2f}%)." if n_in is not None else ".")
        return ToolResult(summ, table=tbl, report=report, figure=fig,
                          payload={"gene": gene, "platform": key,
                                   "values": vec, "stats": stats,
                                   "chart": desc, "report": report})

    tools["gene_distribution"] = Tool(
        name="gene_distribution",
        description="Profile one gene's distribution on a platform "
                    "(class + mean/median/std/modality). Give `low`/`high` "
                    "(or `sd`) to SHADE that region on top of the full "
                    "distribution - use this when asked to show a region "
                    "against the whole distribution rather than on its own.",
        params=[
            ToolParam("gene", "str", help="Gene symbol, e.g. TP53."),
            ToolParam("platform", "platform", required=False,
                      help="Platform to profile (defaults to the first loaded)."),
            *_region_toolparams("the region marked on the distribution"),
            ToolParam("by_label", "str", required=False,
                      choices=LABEL_CHOICES,
                      help="Optional label to stratify the gene by "
                           "(Kruskal-Wallis / ANOVA across label groups). "
                           "Tissue_kind splits catalogued cell lines from "
                           "tissue."),
        ],
        resolver=_dist_resolver, executor=_dist_exec,
        examples=("analyze the distribution of TP53",
                  "distribution of gene BRCA1 on GPL570",
                  "profile EGFR expression",
                  "distribution of APP stratified by Tissue"))

    # ---- compare_gene -------------------------------------------
    def _cmp_resolver(app, raw):
        out = dict(raw)
        plats_arg = raw.get("platforms")
        if isinstance(plats_arg, str):
            plats_arg = [s.strip() for s in plats_arg.replace(";", ",").split(",")
                         if s.strip()]
        if plats_arg:
            resolved = []
            for p in plats_arg:
                m = _match_platform(app, p)
                # Keep an unmatched GPL id verbatim so the executor can
                # auto-load/download it (full automation).
                resolved.append(m if m else str(p).strip().upper())
            out["platforms"] = [p for p in dict.fromkeys(resolved) if p]
        else:
            out["platforms"] = list(_platforms(app).keys())
        return out

    def _cmp_exec(app, resolved, progress_cb):
        from scipy.stats import ks_2samp
        gene = str(resolved.get("gene") or "").strip()
        keys = resolved.get("platforms") or []
        if not gene:
            return ToolResult("Which gene should I compare?", ok=False)
        if len(keys) < 2:
            return ToolResult("Need at least two platforms/sources to compare. "
                              "Load or fetch them first.", ok=False)
        # Auto-load (and download when missing) any GPL platform not yet in
        # memory so the comparison is fully automated.
        for key in list(keys):
            if key not in _platforms(app) and str(key).upper().startswith("GPL"):
                progress_cb(8.0, f"Loading {key}…")
                _load_exec(app, {"platform": key, "download": True, "max_gse": 0},
                           progress_cb)
        plats = _platforms(app)
        rows, vecs = [], {}
        for i, key in enumerate(keys):
            progress_cb(20.0 + 60.0 * i / max(len(keys), 1),
                        f"Extracting {gene} from {key}…")
            if key not in plats:
                continue
            vec = _gene_vector(plats[key], gene)
            if vec is None:
                rows.append({"source": key, "gene": gene, "n": 0,
                             "note": "gene not found"})
                continue
            vecs[key] = vec[np.isfinite(vec)]
            rows.append({"source": key, "gene": gene, **_gene_stats(vec)})
        table = pd.DataFrame(rows)
        # pairwise KS test between the first two usable sources
        ks_txt = ""
        usable = [k for k in keys if k in vecs and vecs[k].size > 1]
        if len(usable) >= 2:
            a, b = usable[0], usable[1]
            stat, p = ks_2samp(vecs[a], vecs[b])
            ks_txt = (f" KS({a} vs {b}) D={stat:.3f}, p={p:.3g}"
                      f" - {'differ' if p < 0.05 else 'no significant difference'}.")
        summ = (f"Compared {gene} across {len(usable)} source(s): "
                + ", ".join(usable) + "." + ks_txt)
        rlines = [f"# {gene} across sources\n"]
        for r in rows:
            if r.get("n"):
                rlines.append(
                    f"- **{r['source']}**: {r.get('distribution', '?')}, "
                    f"mean={r.get('mean', float('nan')):.3g}, "
                    f"median={r.get('median', float('nan')):.3g}, "
                    f"n={r.get('n', 0):,}")
            else:
                rlines.append(f"- **{r['source']}**: {r.get('note', 'no data')}")
        if ks_txt:
            rlines.append("\n**Two-sample test:**" + ks_txt)
        report = "\n".join(rlines)
        from . import charts
        fig, desc = charts.fig_overlay(vecs, gene)
        report += charts.describe_overlay_block(desc)
        return ToolResult(summ, table=table, report=report, figure=fig,
                          payload={"gene": gene, "sources": usable,
                                   "vectors": vecs, "chart": desc,
                                   "report": report})

    tools["compare_gene"] = Tool(
        name="compare_gene",
        description="Compare one gene's distribution across two or more "
                    "platforms/sources (stats + KS test).",
        params=[
            ToolParam("gene", "str", help="Gene symbol to compare."),
            ToolParam("platforms", "list", required=False,
                      help="Platforms/sources to compare (defaults to all loaded)."),
        ],
        resolver=_cmp_resolver, executor=_cmp_exec,
        examples=("compare TP53 across single cell and GEO",
                  "compare distribution of BRCA1 between GPL570 and GPL96",
                  "how does EGFR differ across platforms"))

    # ---- compare_modalities (same gene, harmonised across modalities) --
    def _cross_resolver(app, raw):
        out = _cmp_resolver(app, raw)  # reuse platforms parsing
        m = str(raw.get("method") or "zscore").strip().lower()
        out["method"] = m if m in ("zscore", "rank", "none", "combat") else "zscore"
        return out

    def _cross_exec(app, resolved, progress_cb):
        from genevariate.core.analysis import compare_gene_across_modalities
        gene = str(resolved.get("gene") or "").strip()
        keys = resolved.get("platforms") or []
        if not gene:
            return ToolResult("Which gene should I compare across modalities?",
                              ok=False)
        if len(keys) < 2:
            return ToolResult("Need at least two sources/modalities to compare. "
                              "Load or fetch them first.", ok=False)
        plats = _platforms(app)
        sources = {k: plats[k] for k in keys if k in plats}
        if len(sources) < 2:
            return ToolResult("Fewer than two of those sources are loaded.",
                              ok=False)
        progress_cb(40.0, f"Harmonising {gene} across modalities…")
        res = compare_gene_across_modalities(
            sources, gene, method=str(resolved.get("method", "zscore")),
            modalities=_modalities(app, sources))
        tbl = res["table"]
        n_found = int(tbl["n"].fillna(0).gt(0).sum()) if "n" in tbl.columns else 0
        if n_found == 0:
            return ToolResult(f"{gene!r} was not found in any source.", ok=False)
        from . import charts
        harm = {k: v for k, v in (res.get("harmonized") or {}).items()
                if v is not None and getattr(v, "size", 0) > 1}
        fig, desc = charts.fig_overlay(harm, gene)
        report = res["report"] + charts.describe_overlay_block(desc)
        return ToolResult(res["summary"], table=res["table"],
                          report=report, figure=fig,
                          payload={"gene": gene, "pairwise": res["pairwise"],
                                   "harmonized": res["harmonized"],
                                   "concordant": res["concordant"],
                                   "chart": desc, "report": report})

    tools["compare_modalities"] = Tool(
        name="compare_modalities",
        description="Compare the SAME gene across different data MODALITIES "
                    "(microarray / RNA-seq / single-cell) on a HARMONISED scale "
                    "(z-score or rank), then test whether its distribution shape "
                    "is consistent across modalities. ALWAYS pick this tool when "
                    "the request says 'modalities', 'harmonise/harmonize', "
                    "'z-score', 'rank scale', or 'batch correction' - even if it "
                    "also names specific platforms like GPL570/GPL96.",
        params=[
            ToolParam("gene", "str", help="Gene symbol to compare."),
            ToolParam("platforms", "list", required=False,
                      help="Sources/modalities to compare (defaults to all loaded)."),
            ToolParam("method", "str", required=False, default="zscore",
                      choices=("zscore", "rank", "none", "combat"),
                      help="Scale-harmonisation method ('combat' does real "
                           "batch-effect correction on shared genes)."),
        ],
        resolver=_cross_resolver, executor=_cross_exec,
        examples=("compare TP53 across microarray and rna-seq modalities",
                  "is EGFR consistent between single cell and bulk",
                  "harmonise and compare BRCA1 across platforms"))

    # ---- load_geo_platform (headless, no dialogs) ---------------
    def _load_resolver(app, raw):
        out = dict(raw)
        out.setdefault("download", True)
        out.setdefault("max_gse", 0)
        return out

    def _download_platform(app, key_up, resolved, progress_cb):
        """Auto-download a GPL from GEO when it isn't on disk.

        Returns the saved CSV path (str) on success, or a failing
        ``ToolResult`` describing why the download could not proceed.
        """
        gds_conn = getattr(app, "gds_conn", None)
        data_dir = getattr(app, "data_dir", None)
        if gds_conn is None or not data_dir:
            return ToolResult(
                f"No local file for {key_up} and no GEO metadata database is "
                "open to download it. Open GEOmetadb (or add the CSV to the "
                "data directory) and try again.", ok=False)
        try:
            from genevariate.core.gpl_downloader import GPLDownloader
        except Exception as exc:
            return ToolResult(f"GPL downloader unavailable: {exc}", ok=False)
        try:
            downloader = GPLDownloader(gds_conn=gds_conn, output_base_dir=data_dir)
            downloader.check_dependencies()
        except Exception as exc:
            return ToolResult(
                f"Cannot auto-download {key_up} (missing dependency): {exc}. "
                "Install GEOparse or place the CSV in the data directory.",
                ok=False)
        try:
            max_gse = int(resolved.get("max_gse") or 0)
        except Exception:
            max_gse = 0
        if max_gse < 0:
            max_gse = 0  # 0 == fetch every GSE series (whole platform)

        info = None
        query = getattr(app, "_query_gpl_info_local", None)
        try:
            if callable(query):
                info = query(key_up)
        except Exception as exc:
            return ToolResult(
                f"{key_up} was not found in the GEO metadata database: {exc}",
                ok=False)

        scope = f"up to {max_gse} series" if max_gse else "all series"
        progress_cb(20.0,
                    f"Downloading {key_up} from GEO ({scope})…")

        def cb(pct, stage, msg):
            try:
                if pct is not None:
                    progress_cb(20.0 + float(pct) * 0.55, f"{key_up}: {msg}")
            except Exception:
                pass

        try:
            if info is not None:
                res = downloader.run_with_info(info, max_gse=max_gse, callback=cb)
            else:
                res = downloader.run(key_up, max_gse=max_gse, callback=cb)
        except Exception as exc:
            return ToolResult(
                f"Auto-download of {key_up} failed: {exc}", ok=False)

        fpath = res.get("filepath") if isinstance(res, dict) else None
        if not fpath or not os.path.exists(fpath):
            return ToolResult(
                f"Auto-download of {key_up} produced no usable file.", ok=False)
        return fpath

    def _load_exec(app, resolved, progress_cb):
        name = str(resolved.get("platform") or "").strip()
        if not name:
            return ToolResult("Which platform (e.g. GPL570)?", ok=False)
        key_up = name.upper()
        if key_up in _platforms(app):
            df = _platforms(app)[key_up]
            return ToolResult(f"{key_up} already loaded ({df.shape[0]} samples).",
                              payload={"platform": key_up})
        progress_cb(15.0, f"Locating {key_up}…")
        try:
            available = app._discover_available_platforms()
        except Exception:
            available = {}
        path = available.get(key_up) or available.get(name)

        # Not on disk → download it straight from GEO (full automation).
        if not path and bool(resolved.get("download", True)):
            dl = _download_platform(app, key_up, resolved, progress_cb)
            if isinstance(dl, ToolResult):
                return dl  # surface the download failure
            path = dl

        if not path:
            return ToolResult(
                f"No local file found for {key_up} and auto-download is off. "
                "Enable download, or place its CSV in the data directory.",
                ok=False)
        progress_cb(80.0, f"Reading {key_up}…")
        comp = "gzip" if str(path).endswith(".gz") else "infer"
        df = pd.read_csv(path, compression=comp, low_memory=False)
        # canonicalise: ensure a GSM column
        if "GSM" not in df.columns:
            gsm_col = next((c for c in df.columns
                            if str(c).strip().upper() in ("GSM", "SAMPLE", "SAMPLES")),
                           None)
            if gsm_col is None and df.shape[1]:
                gsm_col = df.columns[0]
            df = df.rename(columns={gsm_col: "GSM"})
        df["GSM"] = df["GSM"].astype(str)
        # Register through the app rather than writing the dict directly: the
        # windows resolve a gene symbol through ``gpl_gene_mappings``, so a
        # platform the assistant put straight into ``gpl_datasets`` is one the
        # user then finds has no genes in any window they open. Loading a
        # platform must leave the program in the state a manual load leaves it.
        app.register_platform_frame(key_up, df)
        try:
            app.after(0, app._update_platform_status)
        except Exception:
            pass
        return ToolResult(f"Loaded {key_up}: {df.shape[0]} samples × "
                          f"{df.shape[1] - 1} columns.",
                          payload={"platform": key_up})

    # ---- multi_label_query (several label conditions at once) --------------
    def _mlq_resolver(app, raw):
        out = dict(raw)
        plats = list(_platforms(app) or {})
        if not str(out.get("platform") or "").strip() and len(plats) == 1:
            out["platform"] = plats[0]
        return out

    def _parse_criteria(text):
        """``Tissue=Liver AND Condition=Cancer`` -> [(col, val), ...].

        Accepts ``AND``, commas or semicolons between criteria, because the
        model writes whichever the user's sentence suggested.
        """
        import re
        if isinstance(text, dict):
            return [(str(k), str(v)) for k, v in text.items()]
        if isinstance(text, (list, tuple)):
            out = []
            for item in text:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    out.append((str(item[0]), str(item[1])))
                elif isinstance(item, str) and "=" in item:
                    c, v = item.split("=", 1)
                    out.append((c.strip(), v.strip()))
            return out
        s = str(text or "")
        parts = [p for p in re.split(r"\s+AND\s+|[,;]", s, flags=re.I)
                 if p.strip()]
        out = []
        for p in parts:
            if "=" in p:
                c, v = p.split("=", 1)
                if c.strip() and v.strip():
                    out.append((c.strip(), v.strip()))
        return out

    def _mlq_exec(app, resolved, progress_cb):
        from genevariate.core.analysis import (
            run_query, queryable_columns, column_values)

        plats = list(_platforms(app) or {})
        if not plats:
            return ToolResult("No platform is loaded to query.", ok=False)
        key = str(resolved.get("platform") or "").strip()
        if not key:
            return ToolResult(
                "Name the platform to query. Loaded: " + ", ".join(plats) + ".",
                ok=False)
        if key not in plats:
            return ToolResult(
                f"{key} is not loaded. Loaded: " + ", ".join(plats) + ".",
                ok=False)

        df = _labelled(app, key)
        if df is None or getattr(df, "empty", True):
            return ToolResult(f"{key} carries no samples to query.", ok=False)

        cols = queryable_columns(df)
        if not cols:
            return ToolResult(
                f"{key} has no label columns to query. Load a label file for "
                "it, or run the extraction pipeline first.", ok=False)

        crit = _parse_criteria(resolved.get("criteria"))
        if not crit:
            return ToolResult(
                "Give at least one condition, written as field=value, for "
                f"example Tissue=Liver. Label fields on {key}: "
                + ", ".join(cols[:15]) + ".", ok=False)

        res = run_query(df, crit)

        if res.unknown_columns:
            missing = ", ".join(res.unknown_columns)
            return ToolResult(
                f"{key} has no label field called {missing}. Its fields are: "
                + ", ".join(cols[:15]) + ".", ok=False)
        if res.unknown_values:
            col, val = res.unknown_values[0]
            vals = column_values(df, col)[:10]
            return ToolResult(
                f"No sample on {key} has {col} = '{val}'. The commonest "
                f"values of {col} are: " + ", ".join(vals) + ".", ok=False)
        if res.n_matched == 0:
            return ToolResult(
                f"No sample on {key} satisfies {res.description}. Each "
                "condition matches on its own; together they do not.",
                ok=False)

        sel = df.loc[res.mask]
        keep = ["GSM"] if "GSM" in sel.columns else []
        keep += [c for c, _ in res.criteria if c in sel.columns]
        rest = [c for c in cols if c not in keep][:4]
        table = sel[keep + rest].head(50)

        pct = 100.0 * res.n_matched / max(res.n_total, 1)
        lines = [f"# {res.description} on {key}\n",
                 f"**{res.n_matched:,} of {res.n_total:,} samples "
                 f"({pct:.1f}%)** satisfy every condition.\n"]
        n_gse = None
        if "series_id" in sel.columns:
            n_gse = int(sel["series_id"].astype(str).nunique())
            # A selection carried by one study is one observation however
            # many samples that study deposited.
            lines.append(f"They come from **{n_gse} "
                         f"stud{'y' if n_gse == 1 else 'ies'}**"
                         + (". A selection that rests on a single study is one "
                            "observation, not the sample count."
                            if n_gse == 1 else ".") + "\n")
        for c in rest:
            vc = sel[c].fillna("N/A").astype(str).value_counts().head(4)
            if len(vc):
                lines.append(f"- {c}: " + ", ".join(
                    f"{i} ({n})" for i, n in vc.items()))

        manifest = _manifest("multi_label_query", resolved,
                             inputs={"platform": key})
        return ToolResult(
            f"{res.n_matched:,} of {res.n_total:,} samples on {key} match "
            f"{res.description}"
            + (f", from {n_gse} studies." if n_gse is not None else "."),
            table=table,
            payload={"platform": key, "n_matched": res.n_matched,
                     "n_total": res.n_total,
                     "criteria": [list(c) for c in res.criteria]},
            report=_append_manifest("\n".join(lines), manifest),
            manifest=manifest)

    tools["multi_label_query"] = Tool(
        name="multi_label_query",
        description="Select the samples that satisfy several label conditions "
                    "at once, ANDed - Tissue=Liver and Condition=Cancer - and "
                    "report how many there are and which studies they come "
                    "from. Use this whenever the request is to select, filter, "
                    "subset or count samples by their labels, including when "
                    "it names the platform to do it on: the platform is "
                    "already loaded and this never loads or downloads "
                    "anything. It filters by label values; it does not test "
                    "whether a label is enriched anywhere.",
        params=[
            ToolParam("criteria", "str",
                      help="Conditions as field=value, joined by AND or commas."),
            ToolParam("platform", "str", required=False,
                      help="Platform to query; assumed when only one is loaded."),
        ],
        resolver=_mlq_resolver, executor=_mlq_exec,
        examples=("how many samples are Tissue=Liver and Condition=Cancer",
                  "select the female liver samples on GPL570"))

    # ---- load_label_file (a prepared label table, not the LLM pipeline) ----
    def _labels_resolver(app, raw):
        out = dict(raw)
        out["path"] = str(out.get("path") or "").strip()
        return out

    def _labels_exec(app, resolved, progress_cb):
        import os
        import re
        import pandas as pd

        path = resolved.get("path")
        if not path:
            return ToolResult(
                "Name the label file to read. It is a CSV with a GSM column "
                "and one column per label field, and the platform it belongs "
                "to has to appear in the file name.", ok=False)
        if not os.path.exists(path):
            return ToolResult(f"No file at {path}.", ok=False)

        fname = os.path.basename(path)
        if not re.search(r"GPL\d+", fname, re.I):
            # Not a formality: the loader takes the platform from the file
            # name, so a file without one cannot be attached to anything.
            return ToolResult(
                f"{fname} does not carry a platform accession in its name, and "
                "that is where the loader reads it from. Rename it like "
                "GPL570_labels.csv so the labels attach to a platform.",
                ok=False)

        try:
            head = pd.read_csv(path, nrows=5,
                               compression="gzip" if str(path).endswith(".gz")
                               else "infer", low_memory=False)
        except Exception as exc:
            return ToolResult(f"{fname} could not be read as CSV: {exc}",
                              ok=False)
        if not any(str(c).strip().lower() in ("gsm", "id", "sample")
                   for c in head.columns) and \
                not _looks_like_gsm_column(head[head.columns[0]]):
            cols = ", ".join(str(c) for c in list(head.columns)[:8])
            return ToolResult(
                f"{fname} has no sample-id column to join on. The first "
                f"columns are: {cols}.", ok=False)

        loader = getattr(app, "_load_single_label_file", None)
        if not callable(loader):
            return ToolResult("This build cannot read label files.", ok=False)

        before = dict(getattr(app, "platform_labels", {}) or {})
        try:
            # skip_auto_check: the window's modal dialogs have no place in a
            # chat answer. The refresh they would have triggered is done below.
            loader(path, skip_auto_check=True)
        except Exception as exc:
            return ToolResult(f"Reading {fname} failed: {exc}", ok=False)

        after = getattr(app, "platform_labels", {}) or {}
        gained = [p for p in after if p not in before
                  or len(after[p]) != len(before.get(p, []))]
        if not gained:
            return ToolResult(
                f"{fname} was read but produced no labels. Either no column "
                "held a value worth keeping, or no sample id matched.",
                ok=False)

        for fn in ("_rebuild_merged_labels", "_refresh_labels_display"):
            try:
                getattr(app, fn, lambda: None)()
            except Exception:
                pass

        plat = gained[0]
        df = after[plat]
        from genevariate.core.analysis import queryable_columns
        label_cols = queryable_columns(df)
        report = (
            f"# Labels for {plat}\n\n"
            f"- file: `{path}`\n"
            f"- samples: {len(df):,}\n"
            f"- label columns: {len(label_cols)} "
            f"({', '.join(label_cols[:12])})\n\n"
            "These are read as written. Values are not harmonized on the way "
            "in - harmonization applies only to labels the extraction "
            "pipeline produced, so a prepared table stays the vocabulary its "
            "author chose.\n")
        manifest = _manifest("load_label_file", resolved, inputs={"path": path})
        return ToolResult(
            f"Loaded labels for {plat} from {fname}: {len(df):,} samples, "
            f"{len(label_cols)} label columns.",
            table=df.head(20),
            payload={"platform": plat, "samples": int(len(df)),
                     "label_columns": label_cols},
            report=_append_manifest(report, manifest), manifest=manifest)

    tools["load_label_file"] = Tool(
        name="load_label_file",
        description="Read a prepared sample-label table (CSV) and attach it to "
                    "the platform named in the file name, the way the 'read a "
                    "prepared table' control does. Use this when labels "
                    "already exist; use extract_labels to derive them with the "
                    "language model instead.",
        params=[
            ToolParam("path", "str",
                      help="Path to the label CSV, named like GPL570_labels.csv."),
        ],
        resolver=_labels_resolver, executor=_labels_exec,
        examples=("load the labels from /data/GPL570_labels.csv",
                  "read my prepared sample label table"))

    # ---- add_custom_platform (a local matrix, loaded as a platform) --------
    def _custom_resolver(app, raw):
        out = dict(raw)
        out["path"] = str(out.get("path") or "").strip()
        out["name"] = str(out.get("name") or "").strip()
        return out

    def _looks_like_gsm_column(series) -> bool:
        """The loader's own rule: a column is the sample id when it is called
        GSM, or when most of what it holds looks like a GEO sample accession."""
        import re
        vals = series.dropna().astype(str).head(50)
        if vals.empty:
            return False
        hits = sum(bool(re.fullmatch(r"GSM\d+", v.strip(), re.I)) for v in vals)
        return hits > len(vals) * 0.5

    def _custom_exec(app, resolved, progress_cb):
        import os
        import pandas as pd

        path, name = resolved.get("path"), resolved.get("name")
        if not path:
            return ToolResult(
                "Name the file to load. It must be a CSV or CSV.GZ with a GSM "
                "column of sample ids and one numeric column per gene.",
                ok=False)
        if not os.path.exists(path):
            return ToolResult(f"No file at {path}.", ok=False)
        if not name:
            return ToolResult(
                "Give the platform a name to load it under, for example "
                "MyStudy_GPL12345. It is how every later question refers to it.",
                ok=False)

        loaded = getattr(app, "gpl_datasets", {}) or {}
        if name in loaded:
            # The window refuses rather than silently replacing, because a
            # platform already carries results that were computed from it.
            return ToolResult(
                f"A platform called {name} is already loaded "
                f"({len(loaded[name]):,} samples). Choose another name.",
                ok=False)

        # Read a few rows first, so a file that cannot be a platform is
        # refused in words rather than through the loader's error dialog.
        try:
            head = pd.read_csv(path, nrows=5,
                               compression="gzip" if str(path).endswith(".gz")
                               else "infer", low_memory=False)
        except Exception as exc:
            return ToolResult(f"{os.path.basename(path)} could not be read as "
                              f"CSV: {exc}", ok=False)
        if head.empty or not len(head.columns):
            return ToolResult(f"{os.path.basename(path)} has no columns.",
                              ok=False)

        has_gsm = any(str(c).strip().lower() in ("gsm", "id", "sample")
                      for c in head.columns) or \
            _looks_like_gsm_column(head[head.columns[0]])
        if not has_gsm:
            cols = ", ".join(str(c) for c in list(head.columns)[:8])
            return ToolResult(
                f"{os.path.basename(path)} has no sample-id column. One column "
                "must be called GSM and hold the sample accessions; the first "
                f"columns here are: {cols}.", ok=False)

        if progress_cb:
            try:
                progress_cb(10, f"Loading {os.path.basename(path)}...")
            except Exception:
                pass

        try:
            app._load_gpl_data(name, path)
        except Exception as exc:
            return ToolResult(f"Loading {name} failed: {exc}", ok=False)

        df = (getattr(app, "gpl_datasets", {}) or {}).get(name)
        if df is None:
            return ToolResult(
                f"The loader did not register {name}. The file was read but "
                "produced no usable platform - most often no column could be "
                "matched to a gene.", ok=False)

        genes = len((getattr(app, "gpl_gene_mappings", {}) or {}).get(name, {}))
        report = (
            f"# {name} loaded from a local file\n\n"
            f"- source: `{path}`\n"
            f"- samples: {len(df):,}\n"
            f"- gene columns indexed: {genes:,}\n\n"
            "The platform is loaded exactly as one downloaded from GEO, so "
            "every analysis reaches it by the same name. Nothing was "
            "normalized on the way in: a matrix is used as the file states "
            "it.\n")
        manifest = _manifest("add_custom_platform", resolved,
                             inputs={"path": path})
        return ToolResult(
            f"Loaded {name} from {os.path.basename(path)}: {len(df):,} "
            f"samples, {genes:,} gene columns.",
            payload={"platform": name, "samples": int(len(df)),
                     "genes": int(genes)},
            report=_append_manifest(report, manifest), manifest=manifest)

    tools["add_custom_platform"] = Tool(
        name="add_custom_platform",
        description="Load a local CSV/CSV.GZ expression matrix as a platform, "
                    "the way the Add Custom Platform control does. The file "
                    "needs a GSM column of sample ids and one numeric column "
                    "per gene. Use this for data that is not in GEO; use "
                    "load_geo_platform for a GPL accession.",
        params=[
            ToolParam("path", "str", help="Path to the CSV or CSV.GZ matrix."),
            ToolParam("name", "str",
                      help="Name to load it under, e.g. MyStudy_GPL12345."),
        ],
        resolver=_custom_resolver, executor=_custom_exec,
        examples=("load my own matrix from /data/counts.csv.gz as MyStudy",
                  "add a custom platform from a local file"))

    # ---- normalize_platform (the explicit, standalone correction step) -----
    def _norm_resolver(app, raw):
        out = dict(raw)
        out.setdefault("modality", "auto")
        return out

    def _raw_platforms(app):
        """{gpl: (raw_path, normalized_or_None)} exactly as the window sees it.

        The Normalize window already answers "what is on disk and does it have
        a normalized twin"; asking it rather than re-deriving the answer is
        what keeps the assistant from offering to normalize a platform the
        window would not list.
        """
        finder = getattr(app, "_find_raw_platforms", None)
        if callable(finder):
            try:
                return finder() or {}
            except Exception:
                return {}
        return {}

    def _norm_exec(app, resolved, progress_cb):
        import os
        from genevariate.core.gpl_downloader import normalize_platform

        found = _raw_platforms(app)
        if not found:
            return ToolResult(
                "No raw platform matrix is on disk to normalize. A platform "
                "has to be downloaded before it can be corrected; the "
                "normalized file is written beside the raw one.", ok=False)

        want = str(resolved.get("platform") or "").strip().upper()
        if not want:
            return ToolResult(
                "Name the platform to normalize. Raw matrices on disk: "
                + ", ".join(sorted(found)) + ".", ok=False)
        if want not in found:
            return ToolResult(
                f"There is no raw matrix for {want} on disk. Raw matrices "
                "available: " + ", ".join(sorted(found)) + ".", ok=False)

        modality = str(resolved.get("modality") or "auto").lower()
        if modality not in ("auto", "counts", "intensity"):
            return ToolResult(
                f"modality must be auto, counts or intensity, not '{modality}'.",
                ok=False)

        raw_path, existing = found[want]
        out_dir = os.path.dirname(raw_path)

        def cb(pct, stage, msg):
            if progress_cb:
                try:
                    progress_cb(int(pct), f"{stage}: {msg}")
                except Exception:
                    pass

        try:
            res = normalize_platform(want, out_dir, callback=cb,
                                     modality=modality)
        except Exception as exc:
            return ToolResult(f"Normalization of {want} failed: {exc}",
                              ok=False)

        rows, cols = (res.get("shape") or (0, 0))[:2]
        kind = "RNA-seq counts" if res.get("counts") else "array intensity"
        steps = ("TMM effective library size, CPM, then log2"
                 if res.get("counts")
                 else ("log2 then NaN-aware quantile normalization"
                       if res.get("applied_log2")
                       else "NaN-aware quantile normalization "
                            "(the matrix was already on a log scale)"))
        dropped = int(res.get("genes_dropped") or 0)

        lines = [
            f"Normalized {want} as {kind}: {steps}.",
            f"Wrote {rows:,} x {cols:,} to "
            f"{os.path.basename(str(res.get('output_path')))}.",
        ]
        if dropped:
            lines.append(f"{dropped:,} genes were dropped below the minimum "
                         "total count.")
        if existing:
            # The raw file is never modified, so re-running is safe -- but a
            # reader comparing two results has to know one replaced the other.
            lines.append("This replaced an existing normalized matrix; the "
                         "raw file was not touched.")

        report = (
            f"# Normalization of {want}\n\n"
            f"- modality: **{kind}** "
            f"({'declared' if modality != 'auto' else 'read from the values'})\n"
            f"- correction: {steps}\n"
            f"- result: {rows:,} samples x {cols:,} columns\n"
            f"- genes dropped: {dropped:,}\n"
            f"- raw: `{res.get('raw_path')}`\n"
            f"- normalized: `{res.get('output_path')}`\n\n"
            "The raw matrix is left as downloaded, so this step can be re-run "
            "with a different modality without going back to GEO. Loading the "
            "platform afterwards picks up the normalized file.\n")
        manifest = _manifest("normalize_platform", resolved,
                             inputs={"raw_path": res.get("raw_path")})
        return ToolResult(" ".join(lines),
                          payload={"platform": want, **res},
                          report=_append_manifest(report, manifest),
                          manifest=manifest)

    tools["normalize_platform"] = Tool(
        name="normalize_platform",
        description="Normalize a platform's raw downloaded matrix and write "
                    "the corrected matrix beside it. RNA-seq counts get TMM "
                    "effective library size, CPM and log2; array intensities "
                    "get log2 and NaN-aware quantile normalization. The "
                    "modality is read from the values unless stated. This "
                    "corrects a raw file on disk - it does not rescale a "
                    "platform that is already loaded.",
        params=[
            ToolParam("platform", "str", help="Platform id, e.g. GPL570."),
            ToolParam("modality", "str", required=False, default="auto",
                      help="auto, counts (RNA-seq) or intensity (arrays)."),
        ],
        resolver=_norm_resolver, executor=_norm_exec,
        examples=("normalize GPL570",
                  "apply TMM and CPM to the raw GPL24676 counts"))

    tools["load_geo_platform"] = Tool(
        name="load_geo_platform",
        description="Load a GEO/GPL microarray platform into memory so it can be "
                    "analysed. If the platform isn't already on disk the whole "
                    "platform is downloaded automatically from GEO (every GSE "
                    "series). Set max_gse only to cap the number of series.",
        params=[
            ToolParam("platform", "str", help="Platform id, e.g. GPL570."),
            ToolParam("download", "bool", required=False, default=True,
                      help="Auto-download from GEO when not found locally."),
            ToolParam("max_gse", "int", required=False, default=0,
                      help="Max GEO series to fetch; 0 = the whole platform."),
        ],
        resolver=_load_resolver, executor=_load_exec,
        examples=("load GPL570", "load the GEO platform GPL96",
                  "bring in microarray platform GPL10558"))

    def _census_cache_path(app, organism, tissue, gene, max_cells):
        """Where a census query is kept between sessions, or None.

        The filename carries every argument that changes what comes back, so a
        liver query and a lung query cannot read each other's cells, and a
        gene-restricted fetch cannot be served to a caller who asked for the
        whole transcriptome. ``max_cells`` is part of it too: a 20,000-cell
        subsample is a different dataset from the full query, not a smaller
        view of it.

        Returns None when the program has no data directory, in which case the
        caller simply fetches as before.
        """
        base = getattr(app, "data_dir", None)
        if not base:
            return None
        parts = [str(organism or "any"),
                 str(tissue or "all_tissues"),
                 (gene or "all_genes"),
                 (f"{int(max_cells)}cells" if max_cells else "allcells")]
        stem = "_".join(re.sub(r"[^A-Za-z0-9]+", "-", p).strip("-").lower()
                        for p in parts)
        return os.path.join(str(base), "single_cell", f"census_{stem}.h5ad")

    # ---- fetch_single_cell (CELLxGENE census -> pseudobulk) -----
    def _sc_resolver(app, raw):
        out = dict(raw)
        out.setdefault("organism", "homo_sapiens")
        out.setdefault("max_cells", 0)  # 0 == all matching cells (no subsample)
        out.setdefault("name", "scRNA")
        return out

    def _sc_exec(app, resolved, progress_cb):
        try:
            from genevariate.sources.cellxgene import (CensusClient,
                                                       CensusTooLargeError)
            from genevariate.utils.pseudobulk import aggregate_to_platform
        except Exception as exc:
            return ToolResult(
                "Single-cell fetch needs the CELLxGENE extra "
                f"(cellxgene_census + anndata): {exc}", ok=False)
        gene = str(resolved.get("gene") or "").strip()
        tissue = resolved.get("tissue") or None
        organism = resolved.get("organism", "homo_sapiens")
        try:
            max_cells = int(resolved.get("max_cells") or 0)
        except (TypeError, ValueError):
            max_cells = 0
        # 0/None → fetch every matching cell so the pseudobulk is unbiased.
        max_cells = max_cells if max_cells > 0 else None

        # "Every matching cell" is only a sane default while something is
        # doing the matching. With no tissue and no gene the query matches the
        # whole census - tens of millions of cells across every organ - and the
        # fetch keeps allocating until the machine's memory manager kills the
        # program. That is what happened here: a request with no tissue and
        # max_cells=0 took down a 503 GB machine, and the only trace was the
        # word "Terminated". Refusing costs the caller one more turn and says
        # what to supply; the alternative is losing the whole session.
        if not tissue and not gene:
            # A cell budget is not a substitute for a filter and must not be
            # offered as one. `max_cells` decides how many cells come back;
            # `tissue` decides which. Capping an unfiltered query returns a
            # random mixture of every organ, which answers no question anyone
            # asked and looks like a result. An earlier version of this message
            # listed a budget beside the filters as an equal option and quoted
            # a number, and the caller took the number: it retried with a cap,
            # still no tissue, and would have aggregated liver against a
            # sample of the whole body. No number is suggested here for the
            # same reason.
            return ToolResult(
                "This query has no filter, so it selects the whole census - "
                "tens of millions of cells across every organ. Say which "
                "tissue you want (or which gene). A max_cells budget will not "
                "fix it: capping an unfiltered query returns a random mixture "
                "of every organ rather than the tissue you are asking about.",
                ok=False)

        # Asking for a name that is already taken used to register a second
        # platform under a suffixed key. That is the wrong default here: the
        # census is queried live, so the same tissue asked for twice with
        # different cell budgets yields two differently-sized matrices, and a
        # caller who said 'liver_census' and silently received 'liver_census_3'
        # goes on to apply bounds derived from the first to the second. Hand
        # back what is loaded and let the caller ask for a new name if they
        # genuinely wanted a second copy.
        wanted = str(resolved.get("name") or "").strip()
        if wanted and wanted in _platforms(app):
            frame = _platforms(app)[wanted]
            return ToolResult(
                f"Platform {wanted!r} is already loaded "
                f"({getattr(frame, 'shape', (0, 0))[0]:,} samples) and was "
                f"reused as-is - the census was not re-queried. Pass a "
                f"different `name` if you meant to fetch a second, separate "
                f"copy.",
                table=frame.head(20) if hasattr(frame, "head") else None,
                payload={"platform": wanted, "reused": True})

        # A census query is minutes of network for a result that does not
        # change between calls, so it is kept on disk next to the GPL
        # platforms and re-read instead of re-fetched. The key is every
        # argument that changes the answer; anything else would serve one
        # query's cells under another query's name.
        cache_path = _census_cache_path(app, organism, tissue, gene, max_cells)
        adata = None
        if cache_path and os.path.exists(cache_path):
            try:
                import anndata as _ad
                progress_cb(20.0, f"Reading cached census {os.path.basename(cache_path)}…")
                adata = _ad.read_h5ad(cache_path)
                progress_cb(60.0, f"Cached census: {adata.n_obs:,} cells")
            except Exception as exc:
                # A corrupt or half-written cache must not be fatal - fall
                # through to the network rather than failing the tool.
                progress_cb(20.0, f"Cache unreadable ({exc}); re-fetching…")
                adata = None

        from_cache = adata is not None
        if adata is None:
            progress_cb(15.0, "Querying CELLxGENE Census…")
            client = CensusClient()
            try:
                adata = client.fetch(
                    organism=organism,
                    genes=[gene] if gene else None,
                    tissue=tissue,
                    max_cells=max_cells,
                    progress_callback=lambda m: progress_cb(35.0, str(m)),
                )
            except CensusTooLargeError as exc:
                return ToolResult(str(exc), ok=False)
            finally:
                try:
                    client.close()
                except Exception:
                    pass
            if cache_path:
                try:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    # Write beside the target and rename, so an interrupted
                    # write never leaves a file that later reads as a cache.
                    tmp = cache_path + ".part"
                    adata.write_h5ad(tmp)
                    os.replace(tmp, cache_path)
                    progress_cb(65.0, f"Cached to {cache_path}")
                except Exception as exc:
                    progress_cb(65.0, f"Could not cache the census: {exc}")

        progress_cb(70.0, "Pseudo-bulking cells…")
        # One function, used by the browser window too: the grouping, the
        # aggregation and the cell threshold are decisions the program makes,
        # not decisions this tool makes again. When each side chose for itself
        # they chose differently - mean vs sum, tissue vs donor, 5 cells vs 10
        # - and the same query produced platforms on different scales with
        # different numbers of profiles.
        df, pb = aggregate_to_platform(adata)
        name = str(resolved.get("name") or "scRNA")
        key = name if name not in _platforms(app) else f"{name}_{len(_platforms(app))}"
        # Stash the cells first, then register: the app reads a single-cell
        # platform's organism and its measurement label back off
        # ``scrna_datasets``, and registration indexes the gene columns that
        # every window resolves a symbol through. Doing either by hand is what
        # made an assistant-fetched census behave unlike a browser-fetched one.
        if not hasattr(app, "scrna_datasets") or app.scrna_datasets is None:
            app.scrna_datasets = {}
        app.scrna_datasets[key] = {"cells": adata, "pseudobulk": pb}
        app.register_platform_frame(key, df)
        try:
            app.after(0, app._update_platform_status)
        except Exception:
            pass
        # What scale the aggregates ended up on is not a detail: every bound a
        # user quotes is in these units, and a caller told only "registered"
        # has no way to know whether it got log2 CPM or a raw mean of counts.
        pbinfo = dict(pb.uns.get("pseudobulk", {}))
        norm = str(pbinfo.get("normalization") or "unknown")
        # The aggregation is chosen inside `aggregate_to_platform` now, so it
        # is read back from the record it writes rather than kept in a second
        # variable here that could disagree with what actually ran.
        agg = str(pbinfo.get("agg") or "unknown")
        note = ""
        if pbinfo.get("normalization_skipped"):
            note = (f" NOT normalized ({pbinfo.get('normalization_skip_reason', '')})"
                    f" - values are on a raw scale and are NOT comparable with "
                    f"log2 bulk platforms.")
        return ToolResult(
            f"Fetched {adata.n_obs:,} cells → pseudo-bulked ({agg}) to "
            f"{df.shape[0]} samples; normalization: {norm}; registered as "
            f"platform {key!r}.{note}",
            payload={"platform": key, "aggregation": agg,
                     "normalization": norm,
                     "normalization_skipped": bool(
                         pbinfo.get("normalization_skipped", False))})

    # ---- cell-level single-cell tools --------------------------
    # ---- load_single_cell_file (local .h5ad) --------------------
    def _h5ad_resolver(app, raw):
        out = dict(raw)
        out.setdefault("name", "")
        return out

    def _h5ad_exec(app, resolved, progress_cb):
        path_s = str(resolved.get("path") or "").strip()
        if not path_s:
            return ToolResult("Give the path to a .h5ad file.", ok=False)
        from pathlib import Path as _P
        path = _P(path_s).expanduser()
        if not path.exists():
            return ToolResult(f"No file at {path}.", ok=False)
        if path.suffix.lower() != ".h5ad":
            return ToolResult(
                f"{path.name} is not a .h5ad. This loads AnnData files; an "
                "expression table goes through load_geo_platform.", ok=False)
        try:
            from genevariate.utils.anndata_io import load_h5ad
            from genevariate.utils.pseudobulk import aggregate_to_platform
        except Exception as exc:
            return ToolResult(
                f"Reading .h5ad needs the single-cell extra (anndata): {exc}",
                ok=False)
        progress_cb(15.0, f"Reading {path.name}…")
        try:
            adata = load_h5ad(path)
        except Exception as exc:
            return ToolResult(f"Could not read {path.name}: {exc}", ok=False)

        # Pseudo-bulk on the same fields, with the same minimum and the same
        # normalisation the census fetch uses, so a local file and a census
        # query of the same cells register on one scale rather than two.
        progress_cb(60.0, f"Pseudo-bulking {adata.n_obs:,} cells…")
        # The same function the Census fetch and the browser window use. A
        # file off disk is aggregated the way a fetched query is, or the two
        # produce platforms that cannot be compared with each other.
        df, pb = aggregate_to_platform(adata)

        name = str(resolved.get("name") or "").strip() or path.stem
        key = name if name not in _platforms(app) else \
            f"{name}_{len(_platforms(app))}"
        if not hasattr(app, "scrna_datasets") or app.scrna_datasets is None:
            app.scrna_datasets = {}
        app.scrna_datasets[key] = {"cells": adata, "pseudobulk": pb}
        app.register_platform_frame(key, df)
        try:
            app.after(0, app._update_platform_status)
        except Exception:
            pass
        # Read back from the record the shared aggregator writes, so what is
        # reported is what ran rather than a second copy of the decision.
        pbinfo = dict(pb.uns.get("pseudobulk", {}))
        grouped = "+".join(str(g) for g in (pbinfo.get("groupby") or [])) or "?"
        return ToolResult(
            f"Loaded {adata.n_obs:,} cells x {adata.n_vars:,} genes from "
            f"{path.name}, grouped by {grouped} ({pbinfo.get('agg', '?')}) into "
            f"{df.shape[0]} pseudo-bulk samples; normalization: "
            f"{pbinfo.get('normalization', 'unknown')}; registered as platform "
            f"{key!r}. The cells themselves stay available to the cell-level "
            "tools.",
            payload={"platform": key, "n_cells": int(adata.n_obs),
                     "aggregation": pbinfo.get("agg"),
                     "normalization": pbinfo.get("normalization")})

    # ---- cell_markers (dot plot) --------------------------------
    # ---- cell_embedding (UMAP / PCA over cells) -----------------
    # ---- read_source (find out what the program can already do) ----
    tools["fetch_single_cell"] = Tool(
        name="fetch_single_cell",
        description="Fetch single-cell RNA-seq from the CELLxGENE Census, "
                    "pseudo-bulk it, and register it as a platform.",
        params=[
            ToolParam("gene", "str", required=False,
                      help="Restrict fetch to this gene (optional)."),
            ToolParam("tissue", "str", required=False,
                      help="Tissue filter, e.g. lung."),
            ToolParam("organism", "str", required=False, default="homo_sapiens",
                      help="Census organism."),
            ToolParam("max_cells", "int", required=False, default=0,
                      help="Optional cap on cells fetched; 0 = all matching cells."),
            ToolParam("name", "str", required=False, default="scRNA",
                      help="Platform name to register under."),
        ],
        resolver=_sc_resolver, executor=_sc_exec,
        examples=("fetch single cell data for TP53 in lung",
                  "get scRNA-seq from cellxgene for brain",
                  "pull single-cell data for EGFR"))

    tools["load_single_cell_file"] = Tool(
        name="load_single_cell_file",
        description="Load a local .h5ad (AnnData) file of single cells, "
                    "pseudo-bulk it and register it as a platform. Use for a "
                    "file already on disk; fetch_single_cell queries the "
                    "CELLxGENE Census over the network instead.",
        params=[
            ToolParam("path", "str", required=True,
                      help="Path to the .h5ad file."),
            ToolParam("name", "str", required=False,
                      help="Platform name to register under; "
                           "defaults to the file name."),
        ],
        resolver=_h5ad_resolver, executor=_h5ad_exec,
        examples=("load the h5ad file at ~/data/liver.h5ad",
                  "open my local single cell file census_liver.h5ad",
                  "read this anndata file and register it"))

    # ---- label_markers (supervised label predictors) ------------
    # ---- cross_modality_gene (single-cell vs bulk) --------------
    def _xmod_resolver(app, raw):
        out = dict(raw)
        loaded = list(_platforms(app).keys())
        plats_arg = raw.get("platforms")
        if isinstance(plats_arg, str):
            plats_arg = [s.strip() for s in plats_arg.replace(";", ",").split(",")
                         if s.strip()]
        # keep only names that genuinely match a loaded source; generic words
        # the LLM may invent ("single-cell", "bulk") won't match and are dropped
        matched = []
        for p in (plats_arg or []):
            low = str(p).strip().lower()
            hit = next((k for k in loaded if k.lower() == low
                        or low in k.lower() or k.lower() in low), None)
            if hit:
                matched.append(hit)
        matched = list(dict.fromkeys(matched))
        # too few real matches -> compare across every loaded source
        out["platforms"] = matched if len(matched) >= 2 else loaded
        return out

    def _xmod_exec(app, resolved, progress_cb):
        from genevariate.core.analysis import cross_modality_gene
        gene = str(resolved.get("gene") or "").strip()
        keys = resolved.get("platforms") or []
        if not gene:
            return ToolResult("Which gene should I compare across modalities?",
                              ok=False)
        plats = _platforms(app)
        sources = {k: plats[k] for k in keys if k in plats}
        if len(sources) < 2:
            return ToolResult("Need at least two loaded sources (e.g. a bulk "
                              "platform and a single-cell one) to compare. "
                              "Load/fetch them first.", ok=False)
        method = str(resolved.get("method") or "rank").lower()
        if method not in ("rank", "zscore"):
            method = "rank"
        progress_cb(40.0, f"Harmonizing {gene} across {len(sources)} sources…")
        try:
            res = cross_modality_gene(sources, gene, method=method,
                                      modalities=_modalities(app, sources))
        except Exception as e:
            return ToolResult(f"Cross-modality comparison failed: {e}", ok=False)
        table = res.get("table")
        fig = None
        if table is not None and not table.empty:
            from . import charts
            vecs = {}
            for k in sources:
                v = _gene_vector(sources[k], gene)
                if v is not None:
                    vecs[k] = v[np.isfinite(v)]
            if len(vecs) >= 2:
                try:
                    fig, _ = charts.fig_overlay(vecs, gene)
                except Exception:
                    fig = None
        summ = f"{gene} across {len(sources)} modalities: {res.get('summary', '')}"
        return ToolResult(summ, table=table, figure=fig,
                          payload={"gene": gene, "sources": list(sources),
                                   "table": table,
                                   "summary": res.get("summary", "")})

    tools["cross_modality_gene"] = Tool(
        name="cross_modality_gene",
        description="Compare one gene's expression across modalities "
                    "(single-cell pseudo-bulk vs bulk vs microarray) on a "
                    "harmonised scale - the SC-vs-bulk view of the same gene.",
        params=[
            ToolParam("gene", "str", help="Gene symbol, e.g. TP53."),
            ToolParam("platforms", "list", required=False,
                      help="Sources to compare "
                           "(defaults to all loaded platforms)."),
            ToolParam("method", "str", required=False, default="rank",
                      choices=("rank", "zscore"),
                      help="Harmonisation method across modalities."),
        ],
        resolver=_xmod_resolver, executor=_xmod_exec,
        examples=("compare TP53 between single cell and bulk",
                  "cross modality expression of EGFR",
                  "how does APP differ between scRNA and GPL570"))

    # ---- search_experiments (GEOmetadb GSE keyword search) ------------------
    def _search_resolver(app, raw):
        return {"query": str(raw.get("query") or "").strip(),
                "limit": raw.get("limit", 25)}

    def _search_exec(app, resolved, progress_cb):
        query = str(resolved.get("query") or "").strip()
        if not query:
            return ToolResult("What should I search GEO experiments for "
                              "(e.g. 'breast cancer', 'alzheimer')?", ok=False)
        conn = getattr(app, "gds_conn", None)
        if conn is None:
            return ToolResult(
                "No GEO metadata database is open. Open GEOmetadb first so I "
                "can search experiments (GSE) by keyword.", ok=False)
        try:
            limit = int(resolved.get("limit") or 25)
        except Exception:
            limit = 25
        limit = max(1, min(limit, 500))

        # Normalise tokens the way the app's own search does (symbol-strip +
        # root form) so "alzheimer's" matches "alzheimers"/"alzheimer".
        import re as _re
        raw_tokens = {t.strip().lower()
                      for t in _re.split(r"[,\s]+", query) if t.strip()}
        tokens = set()
        for t in raw_tokens:
            tokens.add(t)
            cleaned = _re.sub(r"[^a-z0-9]", "", t)
            if cleaned:
                tokens.add(cleaned)
            if cleaned.endswith("s") and len(cleaned) > 3:
                tokens.add(cleaned[:-1])
        tokens = {t for t in tokens if len(t) >= 3}
        if not tokens:
            return ToolResult("The search term is too short to look up.", ok=False)

        progress_cb(20.0, f"Searching GEO experiments for '{query}'…")
        try:
            gse_cols = [r[1] for r in conn.execute("PRAGMA table_info(gse)").fetchall()]
        except Exception as exc:
            return ToolResult(f"Could not read the GEO 'gse' table: {exc}", ok=False)
        if not gse_cols:
            return ToolResult("The GEO database has no 'gse' table to search.",
                              ok=False)
        # Prefer the descriptive text columns; fall back to any text-ish column.
        pref = [c for c in ("title", "summary", "overall_design")
                if c in gse_cols]
        search_cols = pref or [c for c in gse_cols
                               if c.lower() not in ("id", "gse", "status")]
        if not search_cols:
            return ToolResult("No searchable text columns in the GEO 'gse' table.",
                              ok=False)

        like_parts, params = [], []
        for tok in tokens:
            for col in search_cols:
                like_parts.append(f"LOWER({col}) LIKE ?")
                params.append(f"%{tok}%")
        sel = ", ".join(dict.fromkeys(["gse"] + search_cols))
        # Cap the SQL scan generously; we rank + trim after.
        q = (f"SELECT {sel} FROM gse WHERE {' OR '.join(like_parts)} "
             f"LIMIT {max(limit * 20, 500)}")
        try:
            hits = pd.read_sql_query(q, conn, params=params)
        except Exception as exc:
            return ToolResult(f"GEO experiment search failed: {exc}", ok=False)
        if hits.empty:
            return ToolResult(
                f"No GEO experiments matched '{query}'.",
                report=f"# Experiment search: '{query}'\n\nNo GSE series matched.",
                payload={"query": query, "gses": []}, ok=True)

        # Rank by how many distinct tokens appear in each series' text blob.
        def _score(row):
            blob = " ".join(str(row.get(c, "")) for c in search_cols).lower()
            return sum(1 for t in tokens if t in blob)
        hits = hits.copy()
        hits["_matches"] = hits.apply(_score, axis=1)
        hits = hits.sort_values("_matches", ascending=False).head(limit)

        title_col = "title" if "title" in hits.columns else None
        rows = []
        for _, r in hits.iterrows():
            title = str(r.get(title_col, "")) if title_col else ""
            if len(title) > 140:
                title = title[:137] + "…"
            rows.append({"gse": str(r.get("gse", "")),
                         "title": title,
                         "matches": int(r["_matches"])})
        table = pd.DataFrame(rows, columns=["gse", "title", "matches"])

        lines = [f"# Experiment search: '{query}'", "",
                 f"Top {len(table)} matching GEO series (of the scanned set):", ""]
        for _, r in table.iterrows():
            lines.append(f"- **{r['gse']}** ({r['matches']} term"
                         f"{'s' if r['matches'] != 1 else ''}): {r['title']}")
        report = "\n".join(lines)
        manifest = _manifest("search_experiments", resolved,
                             inputs={"tokens": sorted(tokens)})
        report = _append_manifest(report, manifest)
        return ToolResult(
            f"Found {len(table)} GEO experiment(s) matching '{query}'.",
            table=table, report=report, manifest=manifest,
            payload={"query": query, "gses": list(table["gse"])})

    tools["search_experiments"] = Tool(
        name="search_experiments",
        description="Search GEO for EXPERIMENTS/series (GSE) by free-text keyword "
                    "(a disease, tissue, treatment or topic) using the open "
                    "GEOmetadb database - e.g. 'find experiments about breast "
                    "cancer'. Returns a ranked list of matching GSE ids with "
                    "titles. Read-only; downloads nothing. Use load_geo_platform "
                    "to actually load a platform once the user picks one.",
        params=[
            ToolParam("query", "str",
                      help="Free-text topic, e.g. 'breast cancer' or 'alzheimer'."),
            ToolParam("limit", "int", required=False, default=25,
                      help="Max experiments to return (1-500)."),
        ],
        resolver=_search_resolver, executor=_search_exec,
        examples=("find experiments about breast cancer",
                  "search GEO for alzheimer studies",
                  "which datasets study liver fibrosis",
                  "identify experiments involving glioblastoma"))

    # ---- extract_labels (REMOTE-ONLY LLM sample labelling) ------------------
    def _extract_labels_resolver(app, raw):
        fields = raw.get("fields")
        if isinstance(fields, str):
            fields = [f.strip() for f in fields.replace(",", " ").split() if f.strip()]
        return {"platform": _match_platform(app, raw.get("platform")),
                "fields": fields or None,
                "limit": raw.get("limit", 500),
                "url": raw.get("url", ""),
                "model": raw.get("model", "")}

    def _extract_labels_exec(app, resolved, progress_cb):
        from genevariate.core import geo_extract_driver as _drv
        key = resolved.get("platform")
        plats = _platforms(app)
        if not key or key not in plats:
            return ToolResult("Load a platform first (e.g. GPL570), then I can "
                              "extract sample labels for it.", ok=False)
        # HARD SAFETY: never spin up a local model on this machine. Require a
        # reachable remote OpenAI-compatible endpoint; refuse otherwise.
        picked = _drv.resolve_backend(str(resolved.get("url") or ""),
                                      str(resolved.get("model") or ""))
        url, model = picked["url"], picked["model"]
        os.environ.setdefault("GEO_NO_AUTOSCALE", "1")
        progress_cb(8.0, "Checking the extraction backend…")
        try:
            reachable = _drv.backend_reachable(url, timeout=4.0)
        except Exception:
            reachable = False
        if not reachable:
            return ToolResult(
                "Label extraction needs a running LLM backend and none is "
                f"reachable at {url}. Start the extraction server (or point it "
                "at a remote endpoint) and try again - I will not launch a "
                "local model here.", ok=False)

        want = resolved.get("fields") or list(_drv.ALL_FIELDS)
        want = [f for f in want if f in _drv.ALL_FIELDS] or list(_drv.ALL_FIELDS)
        try:
            limit = int(resolved.get("limit") or 500)
        except Exception:
            limit = 500
        limit = max(1, min(limit, 5000))

        df = plats[key]
        if "GSM" not in df.columns:
            return ToolResult(
                f"Platform {key!r} has no GSM column, so there are no GEO "
                "samples to look up metadata for. Label extraction needs a "
                "GEO-derived platform.", ok=False)
        gsms = df["GSM"].astype(str).tolist()[:limit]
        n = len(gsms)

        # Pull sample metadata text from GEOmetadb (the expression frame has no
        # free-text description columns); fall back to bare gsm rows.
        samples = [{"gsm": g} for g in gsms]
        conn = getattr(app, "gds_conn", None)
        if conn is not None:
            progress_cb(15.0, f"Fetching metadata for {n} sample(s)…")
            try:
                gcols = [r[1] for r in conn.execute("PRAGMA table_info(gsm)").fetchall()]
                meta_cols = [c for c in ("gsm", "title", "source_name_ch1",
                                         "characteristics_ch1", "treatment_protocol_ch1",
                                         "description") if c in gcols]
                if "gsm" in meta_cols and len(meta_cols) > 1:
                    # Chunk the IN() list - SQLite caps bound params
                    # (SQLITE_MAX_VARIABLE_NUMBER, 999 on older builds).
                    by_gsm = {}
                    sel = ", ".join(meta_cols)
                    for j in range(0, len(gsms), 900):
                        chunk = gsms[j:j + 900]
                        marks = ",".join("?" * len(chunk))
                        mq = f"SELECT {sel} FROM gsm WHERE gsm IN ({marks})"
                        meta = pd.read_sql_query(mq, conn, params=chunk)
                        for _, r in meta.iterrows():
                            by_gsm[str(r["gsm"])] = dict(r)
                    samples = [dict(by_gsm.get(g, {"gsm": g}), gsm=g) for g in gsms]
            except Exception:
                pass

        def _prog(i, total, gsm):
            frac = 20.0 + (float(i) / max(total, 1)) * 75.0
            progress_cb(frac, f"Labelling {gsm} ({i}/{total})…")

        try:
            recs = _drv.extract_labels(samples, fields=want, url=url,
                                       model=model, progress=_prog)
        except Exception as exc:
            return ToolResult(f"Label extraction failed: {exc}", ok=False)

        out = pd.DataFrame(recs)
        if "gsm" in out.columns:
            out = out.rename(columns={"gsm": "GSM"})
        # Persist so downstream analyses (enrichment, comparisons) can use it.
        try:
            app.platform_labels[key] = out.copy()
            app.after(0, app._update_platform_status)
        except Exception:
            pass

        # Small per-field coverage summary (Not-Specified rate).
        cover = {}
        for f in want:
            if f in out.columns:
                spec = out[f].astype(str).ne(_drv.NS).sum()
                cover[f] = f"{spec}/{len(out)}"
        lines = [f"# Extracted labels for {key}", "",
                 f"Samples labelled: **{len(out)}** (of {df.shape[0]} in {key})",
                 f"Fields: {', '.join(want)}", "", "Coverage (specified/total):"]
        lines += [f"- {f}: {c}" for f, c in cover.items()]
        report = "\n".join(lines)
        manifest = _manifest("extract_labels", resolved,
                             inputs={"n_samples": len(out), "fields": want,
                                     "backend": url, "model": model})
        report = _append_manifest(report, manifest)
        cap = out.head(2000)
        return ToolResult(
            f"Labelled {len(out)} sample(s) in {key} ({', '.join(want)}).",
            table=cap, report=report, manifest=manifest,
            payload={"platform": key, "fields": want, "n": len(out)})

    tools["extract_labels"] = Tool(
        name="extract_labels",
        description="Extract sample labels (Sex/Age/Tissue/Condition/Treatment) "
                    "for a loaded platform's samples using the LLM label "
                    "extractor, and store them as that platform's labels. Needs a "
                    "running REMOTE extraction backend - it will refuse if none is "
                    "reachable (it never launches a local model). Use this when "
                    "the user asks to 'label samples' or 'extract conditions'.",
        params=[
            ToolParam("platform", "platform",
                      help="Loaded platform whose samples to label, e.g. GPL570."),
            ToolParam("fields", "list", required=False, default=None,
                      help="Subset of Sex/Age/Tissue/Condition/Treatment "
                           "(default: all five)."),
            ToolParam("limit", "int", required=False, default=500,
                      help="Max samples to label (1-5000)."),
            ToolParam("url", "str", required=False, default="",
                      help="OpenAI-compatible backend URL (optional)."),
            ToolParam("model", "str", required=False, default="",
                      help="Model name served by the backend (optional)."),
        ],
        resolver=_extract_labels_resolver, executor=_extract_labels_exec,
        examples=("extract sample labels for GPL570",
                  "label the samples in this platform",
                  "annotate conditions and tissue for GPL96",
                  "run label extraction on the loaded platform"))

    # ---- export_results (write what the assistant has produced) -------------
    def _export_resolver(app, raw):
        out = dict(raw)
        folder = str(raw.get("folder") or "").strip()
        out["folder"] = os.path.expanduser(folder) if folder else str(
            Path.home() / "genevariate_exports")
        out["which"] = str(raw.get("which") or "last").strip().lower()
        return out

    def _export_exec(app, resolved, progress_cb):
        runs = [r for r in tool_history()
                if r.tool != "export_results"
                and (r.result.figure is not None or r.result.figures
                     or getattr(r.result.table, "empty", True) is False
                     or (r.result.report or "").strip())]
        if not runs:
            return ToolResult(
                "Nothing has been produced yet in this session, so there is "
                "nothing to write. Run an analysis first, then ask again.",
                ok=False)

        which = resolved.get("which", "last")
        if which in ("all", "everything", "session"):
            chosen = runs
        elif which in ("last", "latest", ""):
            chosen = runs[-1:]
        else:
            chosen = [r for r in runs if r.tool == which]
            if not chosen:
                return ToolResult(
                    f"No {which!r} result has been produced this session. "
                    "What is available: "
                    + ", ".join(sorted({r.tool for r in runs})) + ".",
                    ok=False)

        out_dir = Path(resolved["folder"])
        # Make the export folder, but not the tree above it. A caller who names
        # a folder inside a directory that does not exist has mistyped the path
        # far more often than they have meant to found a new tree there, and
        # creating it silently is unrecoverable in practice: an agent asked for
        # .../Genevariate_Paper/... re-cased it to .../GeneVariate_Paper/... and
        # four sessions' exports went to a parallel directory while the one they
        # were supposed to land in stayed empty, each run reporting success with
        # the path it had invented.
        parent = out_dir.parent
        if not parent.exists():
            # The mistyped component is rarely the last one, so walk up to the
            # deepest directory that does exist and look for a name that
            # differs only in case. Checking only the immediate parent finds
            # nothing when a whole branch is wrong, which is the usual shape of
            # this mistake: .../GeneVariate_Paper/Results/x for
            # .../Genevariate_Paper/Results/x.
            corrected = None
            try:
                anc, missing = parent, []
                while not anc.exists() and anc != anc.parent:
                    missing.append(anc.name)
                    anc = anc.parent
                if missing and anc.exists():
                    want = missing[-1]
                    same = [p.name for p in anc.iterdir()
                            if p.is_dir() and p.name.lower() == want.lower()
                            and p.name != want]
                    # Exactly one case-variant is not a guess: there is only
                    # one directory it could have meant. Telling the caller to
                    # retry does not work - a model handed the corrected path
                    # verbatim re-sent its own spelling and then gave up - so
                    # resolve it here and say so in the result.
                    if len(same) == 1:
                        fixed = anc / same[0]
                        for part in reversed(missing[:-1]):
                            fixed = fixed / part
                        corrected = fixed / out_dir.name
            except OSError:
                pass
            if corrected is None:
                return ToolResult(
                    f"Not exporting: {parent} does not exist, so {out_dir} "
                    f"would be a new directory tree rather than the one you "
                    f"meant.", ok=False)
            # Stated as an instruction, not a footnote: handed a corrected
            # path as a passing remark, the agent reported the folder it had
            # asked for rather than the one the files are in, which sends a
            # reader to a directory that does not exist.
            note = (f"The folder you asked for ({out_dir}) does not exist. "
                    f"The files are in {corrected} - report THAT path, not "
                    f"the one you requested.")
            out_dir = corrected
        else:
            note = ""
        try:
            out_dir.mkdir(exist_ok=True)
        except OSError as exc:
            return ToolResult(f"Cannot write to {out_dir}: {exc}", ok=False)

        # The program's own writer, the same one the Label Enrichment window's
        # "Export all plots" button uses: 300 dpi, tight bounding box, PNG for
        # reading and PDF for submission. A figure the assistant saved and the
        # same figure saved from a window are then the same file, not two
        # renderings of one result that a reader has to reconcile.
        from genevariate.utils.export_manager import PlotExportManager
        mgr = PlotExportManager(base_dir=out_dir)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        subdir = f"assistant_{stamp}"
        figures = {}
        seen = Counter()
        for run in chosen:
            seen[run.tool] += 1
            n = seen[run.tool]
            stem = run.tool if n == 1 else f"{run.tool}_{n}"
            if run.result.figure is not None:
                figures[stem] = run.result.figure
            for extra, fig in (run.result.figures or {}).items():
                figures[f"{stem}_{extra}"] = fig

        progress_cb(30.0, f"Writing {len(figures)} figure(s)…")
        pngs = mgr.export_batch(figures, analysis_id="genevariate", ext="png",
                                subdir=subdir)
        pdfs = mgr.export_batch(figures, analysis_id="genevariate", ext="pdf",
                                subdir=subdir)
        written = list(pngs.values()) + list(pdfs.values())

        progress_cb(70.0, "Writing tables…")
        target = out_dir / subdir
        target.mkdir(parents=True, exist_ok=True)
        seen.clear()
        for run in chosen:
            seen[run.tool] += 1
            stem = run.tool if seen[run.tool] == 1 else f"{run.tool}_{seen[run.tool]}"
            table = run.result.table
            if table is not None and not getattr(table, "empty", True):
                path = target / f"{stem}.csv"
                try:
                    table.to_csv(path, index=False)
                    written.append(path)
                except OSError:
                    pass
            # No markdown. The program's own Export buttons write figures,
            # tables and the windows' text; nothing in it writes a .md, so an
            # assistant that wrote one would be producing a file the user
            # cannot obtain by clicking, which is the one thing an export
            # meant to be checkable must not do. The report stays where the
            # user already reads it: in the chat.

        if not written:
            return ToolResult(
                f"Nothing could be written to {target}. Check that the folder "
                "is writable and has free space.", ok=False)

        index = None
        if pngs:
            try:
                index = mgr.write_html_index(
                    pngs, title="GeneVariate assistant results", subdir=subdir)
                written.append(index)
            except OSError:
                pass

        listing = "\n".join(f"- `{p.name}`" for p in written)
        report = (f"# Exported to `{target}`\n\n"
                  f"{len(figures)} figure(s) written as both PNG and PDF at "
                  f"{mgr.dpi} dpi with a tight bounding box, the same "
                  f"settings the program's Export buttons use.\n\n"
                  f"From: {', '.join(r.tool for r in chosen)}\n\n"
                  f"{listing}\n")
        if index is not None:
            report += f"\nOpen `{index}` to browse the figures.\n"
        return ToolResult(
            f"Wrote {len(written)} file(s) to {target}."
            + (f" {note}" if note else ""),
            report=(f"{note}\n\n{report}" if note else report),
            payload={"folder": str(target), "files": [str(p) for p in written],
                     "tools": [r.tool for r in chosen], "path_corrected": note})

    tools["export_results"] = Tool(
        name="export_results",
        description="Write the figures, tables and reports the assistant has "
                    "already produced this session to a folder on disk, using "
                    "the program's own export writer: 300 dpi PNG plus vector "
                    "PDF and the table as CSV. Use this when the user asks "
                    "to save or export a plot, a table or a result. It writes "
                    "what was produced - it never re-runs the analysis, and it "
                    "writes no file the program's own Export buttons do not.",
        params=[
            ToolParam("folder", "str", required=False,
                      help="Directory to write into (default: "
                           "~/genevariate_exports). Created if missing."),
            ToolParam("which", "str", required=False, default="last",
                      help="'last' (default), 'all' for everything produced "
                           "this session, or the name of a tool to export only "
                           "its results."),
        ],
        resolver=_export_resolver, executor=_export_exec,
        examples=("save that plot", "export the results",
                  "write the figure to a file",
                  "save everything from this session to ~/Desktop/figs",
                  "save this table as csv",
                  "export the region enrichment as a publication figure"))

    return tools
