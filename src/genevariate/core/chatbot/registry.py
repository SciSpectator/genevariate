"""Tool registry for the GeneVariate assistant."""
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
from genevariate.utils.anndata_io import ONTOLOGY_FIELDS

from .tools import Tool, ToolParam, ToolResult, history as tool_history


# -----------------------------------------------------------------
# Shared resolver helpers
# -----------------------------------------------------------------
def _platforms(app) -> Dict[str, pd.DataFrame]:
    """Every platform the session holds, by either route."""
    out = dict(getattr(app, "gpl_gene_cache", {}) or {})
    out.update(getattr(app, "gpl_datasets", {}) or {})
    return out


def _gene_map(app, plat: str) -> Dict[str, str]:
    """The probe-to-symbol map for a platform, by either load route."""
    maps = getattr(app, "gpl_gene_mappings", {}) or {}
    return maps.get(plat) or maps.get(f"_cache_{plat}") or {}


def _modalities(app, keys) -> Dict[str, str]:
    """Real modality per source, from the app's GEOmetadb-backed technology."""
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
    """What is loaded right now, as a block for the agent's system prompt."""
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


_LABEL_FIELDS_LOWER = frozenset(
    f.lower() for f in
    ("Condition", "Tissue", "Treatment", "Age", "Sex")) | frozenset(
    f"{f.lower()}{label_entities.KIND_SUFFIX}"
    for f in label_entities.ENTITY_FIELDS) | frozenset(
    f.lower() for f in ONTOLOGY_FIELDS)


#: What a tool may be asked to group or predict by.
LABEL_CHOICES = ("Tissue", "Condition", "Treatment", "Sex", "Age") + tuple(
    label_entities.kind_column_name(f) for f in label_entities.ENTITY_FIELDS)


def _is_label_col(c) -> bool:
    s = str(c)
    return s.startswith("Classified_") or s.lower() in _LABEL_FIELDS_LOWER


def _classified_cols(df: pd.DataFrame) -> List[str]:
    """Label/metadata columns of a merged frame."""
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
    """Per platform, the frame that still carries the accession columns."""
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
    """A platform's expression frame with the loaded label columns merged in."""
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
    sub = sub.rename(columns={
        c: next((c[len(p):] for p in label_entities.LABEL_STAGE_PREFIXES
                 if c.startswith(p)), c) for c in cols})
    out = df.copy()
    out["GSM"] = out["GSM"].astype(str).str.upper()
    out = out.merge(sub, on="GSM", how="left")
    out.attrs["label_columns"] = [c for c in sub.columns if c != "GSM"]
    return out


def _enrichment_figures(table, cap: Optional[int] = None):
    """The Enrichment tab's own charts for a ``region_label_enrichment`` frame."""
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
    """Count real enriched terms, excluding per-library error rows."""
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
        try:
            ranked = rank_genes_by_condition(
                df, labels, case, control,
                moderated=bool(resolved.get("moderated", False)))
        except ValueError as exc:
            have = pd.Series(list(labels.values())).astype(str).value_counts()
            offer = ", ".join(f"{v} ({n})" for v, n in have.head(8).items())
            return ToolResult(
                f"{exc} Values on "
                f"{resolved.get('condition_column')}: {offer}.", ok=False)
        gate_note = ""
        if resolved.get("bimodality_gated"):
            from genevariate.core.analysis import (
                classify_distributions, filter_ranked_by_distribution)
            progress_cb(35.0, "Classifying gene distributions…")
            tags = classify_distributions(df)
            gated = filter_ranked_by_distribution(
                ranked, tags, keep=("Bimodal", "Multimodal"))
            if len(gated) < 15:
                return ToolResult(
                    f"Only {len(gated)} gene(s) on "
                    f"{resolved.get('platform')} are bimodal or multimodal, "
                    f"which is too few to rank. Ask for the ungated method or "
                    f"use a larger platform.", ok=False)
            gate_note = (f" Ranked over the {len(gated):,} of {len(ranked):,} "
                         f"genes whose distribution is bimodal or multimodal.")
            ranked = gated
        progress_cb(60.0, "Running prerank GSEA…")
        gsea = run_prerank_gsea(ranked, gene_sets=libs)
        ora = None
        if resolved.get("over_representation", True):
            from genevariate.core.analysis import run_enrichr
            try:
                topn = max(10, int(resolved.get("top_genes") or 250))
            except (TypeError, ValueError):
                topn = 250
            progress_cb(75.0, "Running Enrichr over-representation…")
            try:
                ora = run_enrichr(list(ranked.head(topn).index),
                                  gene_sets=libs)
            except Exception as exc:
                gate_note += (f" Over-representation was skipped: {exc}."
                              if str(exc) else "")
                ora = None
        n = _gsea_term_count(gsea)
        top = gsea.head(15) if n else ranked.head(15)
        comparison = f"{case} vs {control} on {resolved.get('platform')}"
        manifest = _manifest("condition_enrichment", resolved,
                             inputs={"platform": df}, seed=42)
        try:
            report = enrichment_report_markdown(ora, gsea, comparison)
        except Exception:
            report = ""
        report = _append_manifest(report, manifest)
        from . import charts
        fig, cdesc = charts.fig_from_enrichment(ranked, gsea, comparison)
        report += charts.describe_bar_block(cdesc)
        n_ora = 0 if ora is None else int(len(ora))
        return ToolResult(
            f"Condition enrichment ({case} vs {control}) on "
            f"{resolved.get('platform')}: {n} enriched term(s) from the ranked "
            f"test"
            + (f" and {n_ora} from over-representation" if n_ora else "")
            + "." + gate_note,
            table=top, report=report, manifest=manifest, figure=fig,
            payload={"ranked": ranked, "gsea": gsea, "ora": ora,
                     "chart": cdesc, "report": report})

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
        groups = (df["series_id"].astype(str).to_numpy()
                  if "series_id" in df.columns else None)

        progress_cb(25.0, "Building the region x label grid…")
        matrix = enrichment_matrix(masks, labels, groups, n_boot=300)
        try:
            from genevariate.core.analysis import cluster_regions
            region_order = cluster_regions(matrix).get("order")
        except Exception:
            region_order = None
        progress_cb(55.0, "Testing regions against each other…")
        pairs = pairwise_differential(masks, labels, groups,
                                      values=matrix["values"], n_boot=300)
        progress_cb(80.0, "Checking whether the effects are region-specific…")
        het = heterogeneity(masks, labels, groups, values=matrix["values"])
        try:
            from genevariate.core.analysis import region_separability
            _lab_cols = [c for c in _classified_cols(df) if c in df.columns]
            sep = region_separability(masks, df[_lab_cols or [col]], groups)
        except Exception as exc:
            self_log = getattr(app, "enqueue_log", lambda _m: None)
            self_log(f"[region_comparison] separability unavailable: {exc}")
            sep = None

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
        sep_note = ""
        if isinstance(sep, dict) and sep.get("auc") is not None:
            sep_note = (f" Fitted on folds split by study, the regions are "
                        f"separable at AUC {float(sep['auc']):.2f}.")
        _n_pairs = max(1, len(masks) * (len(masks) - 1) // 2)
        return ToolResult(
            f"Compared {len(masks)} regions on {key} ({col}) over "
            f"{_n_pairs} region pair(s): {n_sig} of {len(pairs)} label value "
            f"comparisons differ at q<0.05{note}." + sep_note,
            table=tbl, report=report, manifest=manifest,
            payload={"matrix": matrix, "pairs": pairs, "heterogeneity": het,
                     "separability": sep, "region_order": region_order,
                     "report": report})

    # ---- shared region setup for the box tools ------------------
    def _region_params(raw):
        """Read whichever region rule the caller used off a raw argument dict."""
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
        """One high-expression slab per gene, plus the bounds that define it."""
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

        wanted = [c.strip() for c in
                  str(resolved.get("label_columns") or "").split(",") if c.strip()]
        cols = [c for c in (wanted or _classified_cols(df)) if c in df.columns]
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
        show = (show if not show.empty else table).head(_assistant_rows())
        tbl = show[["region", "label_column", "value", "a", "n_region",
                    "region_pct", "background_pct", "enrichment",
                    "ci_low", "ci_high", "n_eff", "q_value",
                    "significance", "n_gse", "replicated"]].copy()
        for c in ("region_pct", "background_pct", "enrichment",
                  "ci_low", "ci_high", "n_eff"):
            if c in tbl.columns:
                tbl[c] = pd.to_numeric(tbl[c], errors="coerce").round(2)

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
        from genevariate.core.analysis.region_enrichment import (
            explode_multi_value, _parts_outside_brackets)
        series = df[col].astype(str)
        if not value:
            seen = explode_multi_value(series).value_counts().head(8)
            return ToolResult(
                f"Name the {col} value to use as the foreground. On {key} the "
                "commonest are: "
                + ", ".join(f"{v} ({n:,})" for v, n in seen.items()) + ".",
                ok=False)

        want = value.lower()

        def _carries(v):
            v = str(v)
            return (v.strip().lower() == want
                    or any(p.strip().lower() == want
                           for p in _parts_outside_brackets(v)))

        mask = series.map(_carries)
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

        others = [c for c in _classified_cols(df) if c != col and c in df.columns]
        if not others:
            return ToolResult(
                f"{col} is the only label column on {key}, and a foreground "
                f"cut from it can only be tested against the others. Load a "
                "label file carrying more fields.", ok=False)

        gse = (df["series_id"].astype(str).to_numpy()
               if "series_id" in df.columns else None)
        region = f"{col}={value}"

        from genevariate.core.analysis import build_enrichment_cells

        ids = (df["GSM"].astype(str).to_numpy() if "GSM" in df.columns
               else df.index.to_numpy())
        study_of = (dict(zip(ids, gse)) if gse is not None else None)
        by_col = {c: pd.Series(df[c].to_numpy(), index=ids) for c in others}
        cells, groups = build_enrichment_cells(
            by_col, ids[mask.to_numpy()], region_name=region,
            study_of=study_of)

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
        show = (show if not show.empty else table).head(_assistant_rows())
        tbl = show[["label_column", "value", "a", "n_region", "region_pct",
                    "background_pct", "enrichment", "ci_low", "ci_high",
                    "n_eff", "q_value", "significance",
                    "n_gse", "replicated"]].copy()
        for c in ("region_pct", "background_pct", "enrichment",
                  "ci_low", "ci_high", "n_eff"):
            if c in tbl.columns:
                tbl[c] = pd.to_numeric(tbl[c], errors="coerce").round(2)
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

        tbl = pairs[["group_a", "group_b", "n_a", "n_b", "mean_a", "mean_b",
                     "wasserstein", "jensen_shannon", "delta_mean", "p_value",
                     "q_value", "significance",
                     "separation"]].head(_assistant_rows()).copy()
        for c in ("mean_a", "mean_b"):
            tbl[c] = tbl[c].round(4)
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
            if "series_id" in df.columns:
                s = df["series_id"][mask].astype(str).value_counts()
                studies[region] = s

        progress_cb(50.0, "Counting the region's composition…")
        table = region_composition(cells)
        report = summarize_region_composition(table)

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
                       "enrichment",
                       "unspecified"]].head(_assistant_rows()).copy()
        for c in ("region_pct", "background_pct", "enrichment",
                  "ci_low", "ci_high", "n_eff"):
            if c in tbl.columns:
                tbl[c] = pd.to_numeric(tbl[c], errors="coerce").round(2)

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
        drop = _classified_cols(df)
        expr_df = df.drop(columns=drop) if drop else df
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
            summ, table=table.head(_assistant_rows()),
            report="\n".join(lines),
            payload={"entities": table,
                     "cell_line_values": sorted(set(cells["Value"])),
                     "kind_column": label_entities.kind_column_name(),
                     "stage": stage})

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
        ToolParam("bimodality_gated", "bool", required=False, default=False,
                  help="Rank only over genes whose own distribution is "
                       "bimodal or multimodal, which is the window's third "
                       "method."),
        ToolParam("over_representation", "bool", required=False, default=True,
                  help="Also run Enrichr over-representation on the top of "
                       "the ranking, as the window does."),
        ToolParam("top_genes", "int", required=False, default=250,
                  help="How many top-ranked genes go into "
                       "over-representation."),
    ]
    tools["condition_enrichment"] = Tool(
        name="condition_enrichment",
        description="Rank genes case-vs-control, run prerank GSEA and Enrichr "
                    "over-representation, optionally over bimodal genes only, "
                    "which are the three methods the Label Enrichment window "
                    "offers.",
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

        maps = {k: _gene_map(app, k) for k in _platforms(app)}
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
        bar, _ = charts.fig_unique_gene_counts(
            {k: inv["unique_genes"][k] for k in keys})

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
                          "combat", "harmony"):
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

        maps = {k: _gene_map(app, k) for k in _platforms(app)}
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
        fig, _ = charts.fig_distance_heatmap(
            rho, "Platform agreement (Spearman rho on gene means)")
        bar, _ = charts.fig_top_differences(results["de_genes"])

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
                               "quantile_normalization", "combat", "harmony"),
                      help="Correction to apply first. 'none' compares the "
                           "values as loaded. quantile_normalization removes "
                           "differences in distribution shape, which is what "
                           "this program measures elsewhere - use it "
                           "deliberately. harmony returns a joint embedding "
                           "and leaves the per-gene values untouched."),
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
            progress_cb(40.0, f"Comparing {gene} across {by_label} on {key}…")
            return _cmpdist_exec(
                app,
                _cmpdist_resolver(app, {"platform": key, "gene": gene,
                                        "group_by": by_label}),
                progress_cb)
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
        try:
            x_label = app.platform_measurement_label(key)
        except Exception:
            x_label = "Expression"
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
                           "(Wilcoxon rank-sum between label groups). "
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
        """Auto-download a GPL from GEO when it isn't on disk."""
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
        """``Tissue=Liver AND Condition=Cancer`` -> [(col, val), ...]."""
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
        table = sel[keep + rest].head(_assistant_rows())

        pct = 100.0 * res.n_matched / max(res.n_total, 1)
        lines = [f"# {res.description} on {key}\n",
                 f"**{res.n_matched:,} of {res.n_total:,} samples "
                 f"({pct:.1f}%)** satisfy every condition.\n"]
        n_gse = None
        if "series_id" in sel.columns:
            n_gse = int(sel["series_id"].astype(str).nunique())
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

        # The table is a view of the matched samples, so it says it is one.
        from genevariate.utils.display_limits import cap_note as _cap_note
        _note = _cap_note(len(table), int(res.n_matched),
                          noun="matched samples")
        if _note:
            lines.append(f"\nThe table lists the {_note}; raise "
                         f"'assistant_rows' under Display limits for more.")

        manifest = _manifest("multi_label_query", resolved,
                             inputs={"platform": key})
        return ToolResult(
            f"{res.n_matched:,} of {res.n_total:,} samples on {key} match "
            f"{res.description}"
            + (f", from {n_gse} studies." if n_gse is not None else ".")
            + (f" {_note.capitalize()} listed." if _note else ""),
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
            loader(path, skip_auto_check=True)
        except Exception as exc:
            return ToolResult(f"Reading {fname} failed: {exc}", ok=False)

        after = getattr(app, "platform_labels", {}) or {}
        gained = [p for p in after if p not in before
                  or len(after[p]) != len(before.get(p, []))]
        if not gained:
            plat = re.search(r"GPL\d+", fname, re.I)
            already = plat and any(
                p.upper() == plat.group(0).upper() for p in after)
            if already:
                key = next(p for p in after
                           if p.upper() == plat.group(0).upper())
                return ToolResult(
                    f"{key} already carries {len(after[key]):,} labelled "
                    f"sample(s) from this file, so nothing changed.",
                    payload={"platform": key, "rows": int(len(after[key]))})
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
        """A column is the sample id when most of its values are GSM accessions."""
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
            return ToolResult(
                f"A platform called {name} is already loaded "
                f"({len(loaded[name]):,} samples). Choose another name.",
                ok=False)

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

        genes = len(_gene_map(app, name))
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
        """{gpl: (raw_path, normalized_or_None)} exactly as the window sees it."""
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
        """Where a census query is kept between sessions, or None."""
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

        if not tissue and not gene:
            return ToolResult(
                "This query has no filter, so it selects the whole census - "
                "tens of millions of cells across every organ. Say which "
                "tissue you want (or which gene). A max_cells budget will not "
                "fix it: capping an unfiltered query returns a random mixture "
                "of every organ rather than the tissue you are asking about.",
                ok=False)

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

        cache_path = _census_cache_path(app, organism, tissue, gene, max_cells)
        adata = None
        if cache_path and os.path.exists(cache_path):
            try:
                import anndata as _ad
                progress_cb(20.0, f"Reading cached census {os.path.basename(cache_path)}…")
                adata = _ad.read_h5ad(cache_path)
                progress_cb(60.0, f"Cached census: {adata.n_obs:,} cells")
            except Exception as exc:
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
                    tmp = cache_path + ".part"
                    adata.write_h5ad(tmp)
                    os.replace(tmp, cache_path)
                    progress_cb(65.0, f"Cached to {cache_path}")
                except Exception as exc:
                    progress_cb(65.0, f"Could not cache the census: {exc}")

        progress_cb(70.0, "Pseudo-bulking cells…")
        df, pb = aggregate_to_platform(adata)
        name = str(resolved.get("name") or "scRNA")
        key = name if name not in _platforms(app) else f"{name}_{len(_platforms(app))}"
        if df.shape[0] == 0:
            _pbi = dict(pb.uns.get("pseudobulk", {}))
            _grp = _pbi.get("n_groups_before_filter")
            _min = _pbi.get("min_cells")
            _by = " x ".join(_pbi.get("groupby") or ()) or "groups"
            _why = ""
            if _grp is not None and _min is not None:
                _why = (f" The {adata.n_obs:,} cells fall into {_grp:,} "
                        f"{_by} groups and none of them reaches the "
                        f"{_min}-cell minimum a profile needs.")
            return ToolResult(
                f"Fetched {adata.n_obs:,} cells, and pseudo-bulking them left "
                f"no profiles, so nothing was registered.{_why} Fetch more "
                f"cells, or narrow the tissue so the cells you do fetch land "
                f"in fewer groups.", ok=False,
                payload={"cells": int(adata.n_obs), "samples": 0})
        if not hasattr(app, "scrna_datasets") or app.scrna_datasets is None:
            app.scrna_datasets = {}
        app.scrna_datasets[key] = {"cells": adata, "pseudobulk": pb}
        app.register_platform_frame(key, df)
        try:
            app.after(0, app._update_platform_status)
        except Exception:
            pass
        pbinfo = dict(pb.uns.get("pseudobulk", {}))
        norm = str(pbinfo.get("normalization") or "unknown")
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

        progress_cb(60.0, f"Pseudo-bulking {adata.n_obs:,} cells…")
        df, pb = aggregate_to_platform(adata)

        name = str(resolved.get("name") or "").strip() or path.stem
        key = name if name not in _platforms(app) else \
            f"{name}_{len(_platforms(app))}"
        if df.shape[0] == 0:
            _pbi = dict(pb.uns.get("pseudobulk", {}))
            _grp = _pbi.get("n_groups_before_filter")
            _min = _pbi.get("min_cells")
            _by = " x ".join(_pbi.get("groupby") or ()) or "groups"
            _why = ""
            if _grp is not None and _min is not None:
                _why = (f" The {adata.n_obs:,} cells fall into {_grp:,} "
                        f"{_by} groups and none of them reaches the "
                        f"{_min}-cell minimum a profile needs.")
            return ToolResult(
                f"Fetched {adata.n_obs:,} cells, and pseudo-bulking them left "
                f"no profiles, so nothing was registered.{_why} Fetch more "
                f"cells, or narrow the tissue so the cells you do fetch land "
                f"in fewer groups.", ok=False,
                payload={"cells": int(adata.n_obs), "samples": 0})
        if not hasattr(app, "scrna_datasets") or app.scrna_datasets is None:
            app.scrna_datasets = {}
        app.scrna_datasets[key] = {"cells": adata, "pseudobulk": pb}
        app.register_platform_frame(key, df)
        try:
            app.after(0, app._update_platform_status)
        except Exception:
            pass
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

    # ---- cross_modality_gene (single-cell vs bulk) --------------
    def _xmod_resolver(app, raw):
        out = dict(raw)
        loaded = list(_platforms(app).keys())
        plats_arg = raw.get("platforms")
        if isinstance(plats_arg, str):
            plats_arg = [s.strip() for s in plats_arg.replace(";", ",").split(",")
                         if s.strip()]
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
        from genevariate.core.analysis import compare_gene_across_modalities
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
            res = compare_gene_across_modalities(
                sources, gene, method=method,
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
        inc = raw.get("include_samples", True)
        if isinstance(inc, str):
            inc = inc.strip().lower() not in ("0", "false", "no", "off")
        return {"query": str(raw.get("query") or "").strip(),
                "limit": raw.get("limit", 25),
                "include_samples": bool(inc)}

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

        from genevariate.core import corpus_search as _cs
        tokens = _cs.search_tokens(query)
        if not tokens:
            return ToolResult("The search term is too short to look up.", ok=False)
        patterns = _cs.word_patterns(tokens)

        progress_cb(20.0, f"Searching GEO experiments for '{query}'…")
        samples = bool(resolved.get("include_samples", True))
        try:
            found = _cs.search_corpus(
                conn, tokens, patterns,
                log=lambda m: progress_cb(50.0, m.replace("[search] ", "")))
        except Exception as exc:
            return ToolResult(f"GEO experiment search failed: {exc}", ok=False)
        search_cols = found["series_columns"]
        hits = found["series_rows"]
        by_sample = set(found["gse_from_samples"]) - set(found["gse_from_series"])
        total_matched = len(found["experiments"]) if samples else len(hits or [])
        if hits is None or hits.empty:
            if samples and by_sample:
                extra = sorted(by_sample)
                return ToolResult(
                    f"{len(extra):,} GEO experiment(s) reach '{query}' through "
                    f"their samples, none through their own description.",
                    table=pd.DataFrame({"gse": extra[:limit]}),
                    payload={"query": query, "gses": extra[:limit]}, ok=True)
            return ToolResult(
                f"No GEO experiments matched '{query}'.",
                report=f"# Experiment search: '{query}'\n\nNo GSE series matched.",
                payload={"query": query, "gses": []}, ok=True)

        def _score(row):
            blob = " ".join(str(row.get(c, "")) for c in search_cols).lower()
            clean = re.sub(r"[^a-z0-9\s]", "", blob)
            return len(_cs.tokens_present(blob, clean, tokens, patterns))
        n_series = len(hits)
        hits = hits.copy()
        hits["_matches"] = hits.apply(_score, axis=1)
        hits["_acc"] = pd.to_numeric(
            hits["gse"].astype(str).str.extract(r"(\d+)", expand=False),
            errors="coerce").fillna(0)
        hits = hits.sort_values(["_matches", "_acc"],
                                ascending=[False, False]).head(limit)

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

        shown = (f"All {n_series} experiment(s) whose own description carries "
                 f"the term:" if len(table) >= n_series else
                 f"The {len(table)} best of {n_series:,} experiment(s) whose "
                 f"own description carries the term, ranked by how many search "
                 f"terms each has and then by recency. Ask for a higher limit "
                 f"to see more:")
        lines = [f"# Experiment search: '{query}'", "", shown, ""]
        for _, r in table.iterrows():
            lines.append(f"- **{r['gse']}** ({r['matches']} term"
                         f"{'s' if r['matches'] != 1 else ''}): {r['title']}")
        if samples and by_sample:
            lines += ["",
                      f"A further {len(by_sample):,} experiment(s) carry the "
                      f"term only in their samples' metadata and are counted "
                      f"in the total. Discover Experiments in the window lists "
                      f"them sample by sample."]
        report = "\n".join(lines)
        manifest = _manifest("search_experiments", resolved,
                             inputs={"tokens": sorted(tokens)})
        report = _append_manifest(report, manifest)
        summary = (f"{total_matched:,} GEO experiment(s) match '{query}': "
                   f"{n_series:,} in their own description"
                   + (f" and {len(by_sample):,} through their samples' "
                      f"metadata alone" if samples and by_sample else "")
                   + f". The {len(table)} listed are those carrying the most "
                     f"search terms, most recent first among equals.")
        return ToolResult(
            summary, table=table, report=report, manifest=manifest,
            payload={"query": query, "gses": list(table["gse"]),
                     "n_total": int(total_matched),
                     "n_from_series": int(n_series),
                     "n_from_samples_only": int(len(by_sample))})

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
                      help="How many experiments to list. The total found is "
                           "reported whatever this is set to."),
            ToolParam("include_samples", "bool", required=False, default=True,
                      help="Search the samples' metadata as well as the "
                           "series description, which is what the window's "
                           "Discover Experiments does. Turning it off is "
                           "faster and finds fewer experiments."),
        ],
        resolver=_search_resolver, executor=_search_exec,
        examples=("find experiments about breast cancer",
                  "search GEO for alzheimer studies",
                  "which datasets study liver fibrosis",
                  "identify experiments involving glioblastoma"))

    def _unconfirmed(resolved, what, cost=""):
        """A ToolResult that asks first, or ``None`` when the user has agreed."""
        ok = resolved.get("confirm")
        if isinstance(ok, str):
            ok = ok.strip().lower() in ("1", "true", "yes", "y", "on")
        if ok:
            return None
        return ToolResult(
            f"This will {what}." + (f" {cost}" if cost else "")
            + " Tell me to go ahead and I will run it.",
            ok=False,
            payload={"needs_confirmation": True, "action": what})

    _CONFIRM_PARAM = ToolParam(
        "confirm", "bool", required=False, default=False,
        help="Set once the user has agreed. The tool describes what it would "
             "do and changes nothing until this is set.")

    def _assistant_rows():
        """How many rows a tool hands back, as the user set it."""
        from genevariate.utils import display_limits as _dl
        n = _dl.get("assistant_rows")
        return 10 ** 9 if n is None else int(n)

    # ---- the buttons that change the session --------------------------------
    def _on_main(app, fn_name):
        fn = getattr(app, fn_name, None)
        if not callable(fn):
            return False
        after = getattr(app, "after", None)
        if callable(after):
            after(0, fn)
        else:
            fn()
        return True

    def _button_tool(name, handler, what, cost, description, examples,
                     ready=None):
        def _resolver(app, raw):
            out = dict(raw)
            out.setdefault("confirm", False)
            return out

        def _exec(app, resolved, progress_cb):
            missing = ready(app) if ready else None
            if missing:
                return ToolResult(missing, ok=False)
            gate = _unconfirmed(resolved, what, cost)
            if gate is not None:
                return gate
            if not _on_main(app, handler):
                return ToolResult(f"This build has no {handler} to run.",
                                  ok=False)
            return ToolResult(
                f"Started: {what}. It runs in the window, so watch it there.",
                payload={"handler": handler})

        tools[name] = Tool(
            name=name, description=description,
            params=[_CONFIRM_PARAM],
            resolver=_resolver, executor=_exec, examples=examples)

    def _needs_step2(app):
        if getattr(app, "step2_data_df", None) is not None:
            return None
        if (getattr(app, "step1_results_df", None) is not None
                and getattr(app, "gse_to_keep_for_step2", None)):
            return None
        return ("There is nothing to label yet. Either run Discover "
                "Experiments and keep some, or load a CSV of samples with "
                "load_label_file.")

    def _needs_selection(app):
        if getattr(app, "gse_to_keep_for_step2", None):
            return None
        return ("No experiments are selected. Run Discover Experiments first "
                "and keep the ones you want.")

    _button_tool(
        "manual_labeling", "run_manual_labeling",
        "open the manual labelling window over the samples now loaded",
        "It writes labels that later analyses read.",
        "Open the manual sample labelling window on the samples currently "
        "loaded, which is what the Manual Labeling button does.",
        ("let me label the samples by hand", "open manual labelling"),
        ready=_needs_step2)

    _button_tool(
        "download_expression", "_download_selected_expression",
        "download the expression matrices for the experiments now selected",
        "That is a network transfer and can run to many gigabytes.",
        "Download expression data for the platforms the selected experiments "
        "use, which is what Step 1.5 does.",
        ("download the data for the experiments I kept",
         "fetch the expression matrices"),
        ready=_needs_selection)

    _button_tool(
        "curate_labels", "_open_llm_curator",
        "open the label curation window",
        "Curation rewrites the loaded labels.",
        "Open the label curation window, which is what the Label curation "
        "button does.",
        ("open label curation", "let me curate the labels"))

    _button_tool(
        "load_archs4", "_open_archs4_window",
        "open the ARCHS4 ingestion window",
        "Loading a platform from ARCHS4 is a network transfer.",
        "Open the ARCHS4 RNA-seq ingestion window, which is what Load RNA-seq "
        "from ARCHS4 does.",
        ("load RNA-seq from ARCHS4", "get a platform from ARCHS4"))

    # ---- cell_composition / cell_qc / marker_dotplot ------------------------
    def _adata_of(app, want=""):
        sets = getattr(app, "scrna_datasets", None) or {}
        want = str(want or "").strip()
        for k in sets:
            if want and want.lower() in str(k).lower():
                return k, (sets[k] or {}).get("cells")
        for k, v in sets.items():
            if (v or {}).get("cells") is not None:
                return k, v["cells"]
        return None, None

    def _no_cells():
        return ToolResult(
            "No cell-level single-cell data is open. fetch_single_cell or the "
            "Census browser loads one, and its cells stay available for the "
            "rest of the session.", ok=False)

    def _obs_pick(adata, requested, prefer):
        obs = getattr(adata, "obs", None)
        cols = [] if obs is None else list(getattr(obs, "columns", []))
        if requested and requested in cols:
            return requested
        for c in prefer:
            if c in cols:
                return c
        return cols[0] if cols else None

    def _cellcomp_resolver(app, raw):
        return {"platform": raw.get("platform", ""),
                "group_by": raw.get("group_by", ""),
                "stack_by": raw.get("stack_by", "")}

    def _cellcomp_exec(app, resolved, progress_cb):
        from genevariate.core.analysis import cell_composition
        key, adata = _adata_of(app, resolved.get("platform"))
        if adata is None:
            return _no_cells()
        group = _obs_pick(adata, resolved.get("group_by"),
                          ("tissue", "donor_id", "dataset_id", "disease"))
        stack = _obs_pick(adata, resolved.get("stack_by"),
                          ("cell_type", "disease", "assay"))
        if not group or not stack:
            return ToolResult(f"{key} carries no cell annotation to cross.",
                              ok=False)
        progress_cb(40.0, f"Composing {stack} within {group}…")
        try:
            res = cell_composition(adata, group, stack)
        except Exception as exc:
            return ToolResult(f"Composition failed: {exc}", ok=False)
        table = getattr(res, "table", None)
        if table is None:
            table = getattr(res, "counts", None)
        manifest = _manifest("cell_composition", resolved,
                             inputs={"platform": key})
        return ToolResult(
            f"{stack} within {group} on {key}.",
            table=(table.head(_assistant_rows()) if table is not None else None),
            report=_append_manifest(
                f"# {stack} within {group} - {key}\n", manifest),
            manifest=manifest,
            payload={"platform": key, "group_by": group, "stack_by": stack,
                     "result": res})

    tools["cell_composition"] = Tool(
        name="cell_composition",
        description="Cross-tabulate one cell annotation within another over "
                    "the open single-cell data, which is what the Composition "
                    "tab of the single-cell window draws.",
        params=[
            ToolParam("platform", "str", required=False,
                      help="Single-cell dataset (defaults to the open one)."),
            ToolParam("group_by", "str", required=False,
                      help="Annotation the bars are grouped by, e.g. tissue."),
            ToolParam("stack_by", "str", required=False,
                      help="Annotation stacked inside each bar, e.g. cell_type."),
        ],
        resolver=_cellcomp_resolver, executor=_cellcomp_exec,
        examples=("what cell types are in each tissue",
                  "show the composition of the single-cell data"))

    def _cellqc_resolver(app, raw):
        return {"platform": raw.get("platform", ""),
                "group_by": raw.get("group_by", ""),
                "mito_prefix": raw.get("mito_prefix", "MT-")}

    def _cellqc_exec(app, resolved, progress_cb):
        from genevariate.core.analysis import cell_qc, summarize_cell_qc
        key, adata = _adata_of(app, resolved.get("platform"))
        if adata is None:
            return _no_cells()
        group = resolved.get("group_by") or None
        if group:
            group = _obs_pick(adata, group, ())
        prefix = str(resolved.get("mito_prefix") or "MT-")
        progress_cb(40.0, f"Measuring cell quality on {key}…")
        try:
            qc = cell_qc(adata, mito_prefix=prefix, group_by=group)
            summary = summarize_cell_qc(qc, mito_prefix=prefix, group_by=group)
        except Exception as exc:
            return ToolResult(f"Quality measurement failed: {exc}", ok=False)
        manifest = _manifest("cell_qc", resolved, inputs={"platform": key})
        med = summary.get("median") if isinstance(summary, dict) else None
        head = (f"{len(qc):,} cells of {key} measured for genes detected, "
                f"total counts and mitochondrial share")
        if isinstance(med, dict) and med:
            head += ": median " + ", ".join(
                f"{k} {v:.3g}" for k, v in list(med.items())[:3])
        return ToolResult(
            head + ".", table=qc.head(_assistant_rows()),
            report=_append_manifest(f"# Cell quality - {key}\n", manifest),
            manifest=manifest,
            payload={"platform": key, "summary": summary})

    tools["cell_qc"] = Tool(
        name="cell_qc",
        description="Per-cell genes detected, total counts and mitochondrial "
                    "share for the open single-cell data, which is what the "
                    "QC tab of the single-cell window reports.",
        params=[
            ToolParam("platform", "str", required=False,
                      help="Single-cell dataset (defaults to the open one)."),
            ToolParam("group_by", "str", required=False,
                      help="Cell annotation to report the quality per."),
            ToolParam("mito_prefix", "str", required=False, default="MT-",
                      help="Gene-name prefix that marks mitochondrial genes."),
        ],
        resolver=_cellqc_resolver, executor=_cellqc_exec,
        examples=("check the quality of the cells",
                  "what is the mitochondrial fraction per cell type"))

    def _dotplot_resolver(app, raw):
        genes = raw.get("genes")
        if isinstance(genes, str):
            genes = [g.strip() for g in genes.replace(",", " ").split()
                     if g.strip()]
        return {"platform": raw.get("platform", ""),
                "genes": genes or [],
                "group_by": raw.get("group_by", "")}

    def _dotplot_exec(app, resolved, progress_cb):
        from genevariate.core.analysis import marker_dotplot
        key, adata = _adata_of(app, resolved.get("platform"))
        if adata is None:
            return _no_cells()
        genes = resolved.get("genes") or []
        if not genes:
            return ToolResult("Name the genes to show, for example the markers "
                              "you expect to separate the cell types.",
                              ok=False)
        group = _obs_pick(adata, resolved.get("group_by"),
                          ("cell_type", "tissue", "disease"))
        if not group:
            return ToolResult(f"{key} carries no cell annotation to group by.",
                              ok=False)
        progress_cb(40.0, f"Measuring {len(genes)} gene(s) per {group}…")
        try:
            res = marker_dotplot(adata, genes, group)
        except Exception as exc:
            return ToolResult(f"Dot plot failed: {exc}", ok=False)
        table = getattr(res, "table", None)
        manifest = _manifest("marker_dotplot", resolved,
                             inputs={"platform": key, "genes": genes})
        return ToolResult(
            f"{len(genes)} gene(s) summarised per {group} on {key}, as mean "
            f"expression and the fraction of cells expressing.",
            table=(table.head(_assistant_rows()) if table is not None else None),
            report=_append_manifest(f"# Markers per {group} - {key}\n", manifest),
            manifest=manifest,
            payload={"platform": key, "group_by": group, "result": res})

    tools["marker_dotplot"] = Tool(
        name="marker_dotplot",
        description="Mean expression and expressing fraction of named genes "
                    "per cell annotation on the open single-cell data, which "
                    "is what the Dot plot tab of the single-cell window draws.",
        params=[
            ToolParam("genes", "str",
                      help="Genes to show, comma or space separated."),
            ToolParam("platform", "str", required=False,
                      help="Single-cell dataset (defaults to the open one)."),
            ToolParam("group_by", "str", required=False,
                      help="Cell annotation on the other axis, e.g. cell_type."),
        ],
        resolver=_dotplot_resolver, executor=_dotplot_exec,
        examples=("dot plot of ALB and CD3E by cell type",
                  "which cell types express my markers"))

    # ---- cell_embedding -----------------------------------------------------
    def _cellembed_resolver(app, raw):
        sets = getattr(app, "scrna_datasets", None) or {}
        want = str(raw.get("platform") or "").strip()
        key = None
        for k in sets:
            if want and want.lower() in str(k).lower():
                key = k
                break
        if key is None:
            key = next((k for k, v in sets.items()
                        if (v or {}).get("cells") is not None), None)
        return {"platform": key,
                "colour_by": raw.get("colour_by") or raw.get("color_by") or "",
                "max_cells": raw.get("max_cells", 20000)}

    def _cellembed_exec(app, resolved, progress_cb):
        from genevariate.core.analysis import cell_embedding
        sets = getattr(app, "scrna_datasets", None) or {}
        key = resolved.get("platform")
        adata = (sets.get(key) or {}).get("cells") if key else None
        if adata is None:
            return ToolResult(
                "No cell-level single-cell data is open. fetch_single_cell or "
                "the Census browser loads one, and its cells stay available "
                "for as long as the session does.", ok=False)
        try:
            max_cells = max(100, int(resolved.get("max_cells") or 20000))
        except (TypeError, ValueError):
            max_cells = 20000
        progress_cb(40.0, f"Embedding {int(adata.n_obs):,} cells of {key}…")
        try:
            coords, method, idx = cell_embedding(adata, max_cells=max_cells,
                                                 seed=0)
        except Exception as exc:
            return ToolResult(f"Embedding failed: {exc}", ok=False)
        coords = np.asarray(coords)
        if coords.ndim == 1:
            coords = coords.reshape(-1, 1)
        if coords.shape[1] < 2:
            coords = np.column_stack([coords[:, 0], np.zeros(len(coords))])
            method += " (one dimension only, the second axis is flat)"

        obs = getattr(adata, "obs", None)
        colour = str(resolved.get("colour_by") or "").strip()
        groups, colour_name = ["all"] * len(idx), ""
        if obs is not None and len(getattr(obs, "columns", [])):
            pick = colour if colour in obs.columns else None
            if pick is None:
                for cand in ("cell_type", "tissue", "disease", "assay"):
                    if cand in obs.columns:
                        pick = cand
                        break
            if pick is not None:
                groups = obs[pick].iloc[idx].astype(str).tolist()
                colour_name = pick
        from . import charts
        title = f"{key}: {int(len(idx)):,} cells, {method}"
        fig, vdesc = charts.fig_scatter(coords, groups, title, "dim 1", "dim 2")
        counts = pd.Series(groups).value_counts()
        table = counts.rename_axis(colour_name or "group").reset_index(
            name="cells")
        note = ("" if int(adata.n_obs) <= len(idx) else
                f" {int(adata.n_obs) - len(idx):,} further cell(s) were left "
                f"out by the max_cells setting, which you can raise.")
        report = (f"# {title}\n\n"
                  f"- **Cells drawn**: {len(idx):,} of {int(adata.n_obs):,}\n"
                  f"- **Layout**: {method}\n"
                  f"- **Coloured by**: {colour_name or 'nothing'}\n")
        manifest = _manifest("cell_embedding", resolved,
                             inputs={"platform": key})
        return ToolResult(
            f"{len(idx):,} cells of {key} laid out by {method}"
            + (f", coloured by {colour_name}" if colour_name else "")
            + "." + note,
            table=table.head(_assistant_rows()),
            report=_append_manifest(report, manifest), manifest=manifest,
            figure=fig,
            payload={"platform": key, "method": method,
                     "n_cells": int(len(idx)), "chart": vdesc})

    tools["cell_embedding"] = Tool(
        name="cell_embedding",
        description="Lay out the individual cells of an open single-cell "
                    "dataset in two dimensions, preferring the embedding its "
                    "depositor published, and colour them by a cell-level "
                    "annotation such as cell type or tissue.",
        params=[
            ToolParam("platform", "str", required=False,
                      help="Single-cell dataset to embed (defaults to the one "
                           "that is open)."),
            ToolParam("colour_by", "str", required=False,
                      help="Cell-level column to colour by, e.g. cell_type."),
            ToolParam("max_cells", "int", required=False, default=20000,
                      help="Cells drawn before a seeded subsample is taken."),
        ],
        resolver=_cellembed_resolver, executor=_cellembed_exec,
        examples=("show the UMAP of the cells",
                  "embed the single-cell data coloured by cell type",
                  "what does the cell space look like"))

    # ---- list_genes ---------------------------------------------------------
    def _list_genes_resolver(app, raw):
        return {"platform": _match_platform(app, raw.get("platform")),
                "contains": str(raw.get("contains") or "").strip(),
                "limit": raw.get("limit", 100)}

    def _list_genes_exec(app, resolved, progress_cb):
        key = resolved.get("platform")
        plats = _platforms(app)
        if not key or key not in plats:
            return ToolResult("No matching platform is loaded. Load one first.",
                              ok=False)
        df = plats[key]
        genes = [c for c in df.columns
                 if c not in ("GSM", "series_id")
                 and pd.api.types.is_numeric_dtype(df[c])]
        needle = resolved.get("contains", "").lower()
        matched = [g for g in genes if needle in str(g).lower()] if needle \
            else list(genes)
        try:
            limit = max(1, int(resolved.get("limit") or 100))
        except (TypeError, ValueError):
            limit = 100
        table = pd.DataFrame({"gene": matched[:limit]})
        if needle:
            head = (f"{len(matched):,} of the {len(genes):,} genes on {key} "
                    f"contain {needle!r}")
        else:
            head = f"{key} carries {len(genes):,} gene(s)"
        if len(matched) > limit:
            head += f". The first {limit} are listed"
        report = (f"# Genes on {key}\n\n- **Total**: {len(genes):,}\n"
                  + (f"- **Matching {needle!r}**: {len(matched):,}\n"
                     if needle else "")
                  + "\n" + ", ".join(f"`{g}`" for g in matched[:limit]) + "\n")
        manifest = _manifest("list_genes", resolved, inputs={"platform": key})
        return ToolResult(head + ".", table=table,
                          report=_append_manifest(report, manifest),
                          manifest=manifest,
                          payload={"platform": key, "n_genes": len(genes),
                                   "matched": matched[:limit]})

    tools["list_genes"] = Tool(
        name="list_genes",
        description="List or search the genes a loaded platform measures. "
                    "Answers whether a gene is present before anything is run "
                    "on it, and what the platform covers.",
        params=[
            ToolParam("platform", "platform", required=False,
                      help="Platform to list (defaults to the first loaded)."),
            ToolParam("contains", "str", required=False,
                      help="Only genes whose name contains this text."),
            ToolParam("limit", "int", required=False, default=100,
                      help="How many names to list."),
        ],
        resolver=_list_genes_resolver, executor=_list_genes_exec,
        examples=("does GPL570 have ALB",
                  "list the genes on my platform",
                  "which genes start with HLA"))

    # ---- project_samples / cluster_samples ----------------------------------
    def _structure_frame(app, resolved):
        """(feature matrix, group labels, gene names, platform key) or a refusal."""
        key = resolved.get("platform")
        if not key or key not in _platforms(app):
            return None, ToolResult("No matching platform is loaded. Load one "
                                    "first.", ok=False)
        df = _labelled(app, key)
        wanted = resolved.get("genes") or []
        if isinstance(wanted, str):
            wanted = [g.strip() for g in wanted.replace(",", " ").split()
                      if g.strip()]
        cols = []
        for g in wanted:
            c = _find_gene_column(df, g)
            if c is not None and c not in cols:
                cols.append(c)
        if wanted and not cols:
            return None, ToolResult(
                f"None of {', '.join(map(str, wanted))} is a column on {key}.",
                ok=False)
        if not cols:
            num = [c for c in df.columns
                   if c not in ("GSM", "series_id")
                   and pd.api.types.is_numeric_dtype(df[c])]
            cols = num[:200]
        if len(cols) < 1:
            return None, ToolResult(f"{key} carries no numeric column to "
                                    "project.", ok=False)
        X = df[cols].apply(pd.to_numeric, errors="coerce")
        keep = X.notna().all(axis=1).to_numpy()
        if keep.sum() < 3:
            return None, ToolResult(
                "Fewer than three samples carry a value for every gene asked "
                "for, which is not enough to project. Name fewer genes.",
                ok=False)
        X = X[keep]
        colour = resolved.get("colour_by")
        if colour and colour in df.columns:
            groups = df.loc[keep, colour].fillna("N/A").astype(str).tolist()
            colour_name = colour
        elif "series_id" in df.columns:
            groups = df.loc[keep, "series_id"].fillna("N/A").astype(str).tolist()
            colour_name = "series_id"
        else:
            groups = ["all"] * int(keep.sum())
            colour_name = ""
        return (X.to_numpy(dtype=float), groups, cols, key, colour_name), None

    def _structure_resolver(app, raw):
        out = dict(raw)
        out["platform"] = _match_platform(app, raw.get("platform"))
        df = _labelled(app, out["platform"]) if out["platform"] else None
        req = raw.get("colour_by") or raw.get("color_by") or raw.get("group_by")
        out["colour_by"] = (_pick_condition_column(df, req)
                            if df is not None and req else req)
        return out

    def _project_exec(app, resolved, progress_cb):
        from genevariate.core.analysis import project, scale_features
        built, refusal = _structure_frame(app, resolved)
        if refusal is not None:
            return refusal
        X, groups, cols, key, colour_name = built
        method = str(resolved.get("method") or "pca").lower()
        progress_cb(40.0, f"Projecting {X.shape[0]:,} samples of {key}…")
        try:
            res = project(scale_features(X), method=method)
        except Exception as exc:
            return ToolResult(f"Projection failed: {exc}", ok=False)
        ev = res.get("explained")
        if res["method"] == "single feature":
            xl, yl = str(cols[0]), ""
            axes = ("One feature was asked for, so there is nothing to "
                    "project. The values are drawn on a single axis and the "
                    "second is flat, not invented.")
        elif res["method"] == "pca" and ev:
            xl, yl = f"PC1 ({ev[0]:.1%})", f"PC2 ({ev[1]:.1%})"
            axes = (f"PC1 carries {ev[0]:.1%} of the variance and PC2 "
                    f"{ev[1]:.1%}.")
        elif res["method"] == "tsne":
            xl, yl = "t-SNE 1", "t-SNE 2"
            axes = (f"t-SNE axes carry no variance share, and distance between "
                    f"clusters on them means nothing. Perplexity "
                    f"{res.get('perplexity', 0):.0f}.")
        else:
            xl, yl = "UMAP 1", "UMAP 2"
            axes = ("UMAP axes carry no variance share and no distance "
                    "meaning beyond the neighbourhood each point sits in.")
        asked = method
        note = ("" if asked != "umap" or res["method"] == "umap" else
                " UMAP was asked for and is unavailable here, so this is PCA.")
        from . import charts
        title = f"{res['method'].upper()} of {key}"
        fig, vdesc = charts.fig_scatter(res["coords"], groups, title, xl, yl)
        coords = pd.DataFrame(res["coords"], columns=["dim1", "dim2"])
        coords.insert(0, colour_name or "group", groups)
        report = (f"# {title}\n\n"
                  f"- **Samples**: {X.shape[0]:,}\n"
                  f"- **Features**: {len(cols)} numeric column(s)\n"
                  f"- **Coloured by**: {colour_name or 'nothing'}\n"
                  f"- **Axes**: {axes}\n")
        manifest = _manifest("project_samples", resolved,
                             inputs={"platform": key, "features": cols[:20]})
        report = _append_manifest(report, manifest)
        return ToolResult(
            f"{X.shape[0]:,} samples of {key} projected by "
            f"{res['method'].upper()} over {len(cols)} feature(s). {axes}{note}",
            table=coords.head(_assistant_rows()),
            report=report, manifest=manifest, figure=fig,
            payload={"platform": key, "method": res["method"],
                     "explained": ev, "coords": coords, "chart": vdesc})

    def _cluster_exec(app, resolved, progress_cb):
        from genevariate.core.analysis import (
            cluster, cluster_composition, density_peaks, project,
            scale_features,
        )
        built, refusal = _structure_frame(app, resolved)
        if refusal is not None:
            return refusal
        X, groups, cols, key, colour_name = built
        method = str(resolved.get("method") or "kmeans").lower()
        progress_cb(40.0, f"Clustering {X.shape[0]:,} samples of {key}…")
        X_s = scale_features(X)
        from . import charts

        if method in ("density_peaks", "dpc", "density"):
            try:
                res = density_peaks(X_s)
            except ValueError as exc:
                return ToolResult(str(exc), ok=False)
            fig, vdesc = charts.fig_scatter(
                np.column_stack([res["rho"], res["delta"]]), groups,
                f"Density peak decision graph - {key}",
                "local density", "distance to a denser sample")
            tbl = pd.DataFrame({"rho": res["rho"], "delta": res["delta"]})
            tbl.insert(0, colour_name or "group", groups)
            tbl = tbl.sort_values("delta", ascending=False)
            report = (f"# Density peaks - {key}\n\n"
                      "A cluster centre is dense and far from anything denser, "
                      "so it sits alone towards the top right. No cluster count "
                      "is assumed and none is returned.\n")
            manifest = _manifest("cluster_samples", resolved,
                                 inputs={"platform": key})
            return ToolResult(
                f"Decision graph for {X.shape[0]:,} samples of {key}. The "
                f"{int((res['delta'] > np.percentile(res['delta'], 99)).sum())} "
                f"sample(s) in the top percentile of distance are the "
                f"candidate centres.",
                table=tbl.head(_assistant_rows()),
                report=_append_manifest(report, manifest), manifest=manifest,
                figure=fig, payload={"platform": key, "chart": vdesc})

        kw = {}
        for name in ("eps", "min_samples", "max_k"):
            if resolved.get(name) not in (None, ""):
                try:
                    kw[name] = (float(resolved[name]) if name == "eps"
                                else int(resolved[name]))
                except (TypeError, ValueError):
                    pass
        try:
            res = cluster(X_s, method=method, **kw)
        except Exception as exc:
            return ToolResult(f"Clustering failed: {exc}", ok=False)
        coords = project(X_s)["coords"]
        fig, vdesc = charts.fig_scatter(
            coords, res["names"],
            f"{res['method'].upper()} on {key}", "PC1", "PC2")
        comp = cluster_composition(res["names"], groups)
        if res["method"] == "kmeans":
            headline = (f"{res['k']} cluster(s) on {key}, chosen by silhouette"
                        + (f" ({res['silhouette']:.2f})"
                           if res["silhouette"] is not None else "") + ".")
        else:
            headline = (f"{res['n_clusters']} cluster(s) on {key}, with "
                        f"{res['n_noise']:,} sample(s) left as noise.")
        report = (f"# {res['method'].upper()} - {key}\n\n"
                  f"- **Samples**: {X.shape[0]:,}\n"
                  f"- **Features**: {len(cols)} numeric column(s)\n"
                  f"- **Compared against**: {colour_name or 'nothing'}\n\n"
                  "A cluster means nothing on its own, so the table gives what "
                  "each one is made of in the labels the source published.\n")
        manifest = _manifest("cluster_samples", resolved,
                             inputs={"platform": key, "features": cols[:20]})
        # The composition table is a view of the whole crossing, so it says so.
        from genevariate.utils.display_limits import cap_note as _cap_note
        _note = _cap_note(min(len(comp), _assistant_rows()), len(comp),
                          noun="cluster and label pairs")
        return ToolResult(
            headline + (f" {_note.capitalize()} listed." if _note else ""),
            table=comp.head(_assistant_rows()),
            report=_append_manifest(report, manifest), manifest=manifest,
            figure=fig,
            payload={"platform": key, "method": res["method"],
                     "k": res.get("k"), "silhouette": res.get("silhouette"),
                     "composition": comp, "chart": vdesc})

    _STRUCT_PARAMS = [
        ToolParam("platform", "platform", required=False,
                  help="Platform to work on (defaults to the first loaded)."),
        ToolParam("genes", "str", required=False,
                  help="Genes to use as features. Every numeric column is "
                       "used when this is left out."),
        ToolParam("colour_by", "str", required=False,
                  help="Label column the points are coloured by, or compared "
                       "against. Defaults to the study identifier."),
    ]

    tools["project_samples"] = Tool(
        name="project_samples",
        description="Project the samples of a platform onto two axes by PCA or "
                    "UMAP, coloured by a label or by study. Answers what the "
                    "samples look like before any group is assumed.",
        params=_STRUCT_PARAMS + [
            ToolParam("method", "str", required=False, default="pca",
                      help="pca, umap or tsne."),
        ],
        resolver=_structure_resolver, executor=_project_exec,
        examples=("run a PCA of GPL570 coloured by tissue",
                  "project the samples with UMAP",
                  "what does the sample space look like"))

    tools["cluster_samples"] = Tool(
        name="cluster_samples",
        description="Group the samples of a platform without being told the "
                    "groups, by k-means with the cluster count chosen on "
                    "silhouette, by DBSCAN, or as a density peak decision "
                    "graph, and report what each cluster is made of.",
        params=_STRUCT_PARAMS + [
            ToolParam("method", "str", required=False, default="kmeans",
                      help="kmeans, dbscan or density_peaks."),
            ToolParam("max_k", "int", required=False,
                      help="Largest cluster count k-means may choose."),
            ToolParam("eps", "float", required=False,
                      help="DBSCAN neighbourhood radius. Raise it when "
                           "everything comes back as noise."),
            ToolParam("min_samples", "int", required=False,
                      help="Smallest DBSCAN neighbourhood that counts as a "
                           "cluster."),
        ],
        resolver=_structure_resolver, executor=_cluster_exec,
        examples=("cluster the samples on GPL570",
                  "run DBSCAN over the samples and tell me what is noise",
                  "show the density peak decision graph"))

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
        parent = out_dir.parent
        if not parent.exists():
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
                                subdir=subdir, timestamp=False)
        pdfs = mgr.export_batch(figures, analysis_id="genevariate", ext="pdf",
                                subdir=subdir, timestamp=False)
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
