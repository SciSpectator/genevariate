"""
Tool registry for the GeneVariate assistant.

``build_registry(app)`` returns a ``{name: Tool}`` map whose executors call the
*existing* analysis API (``genevariate.core.analysis``) against the platforms
already loaded in ``app.gpl_datasets``. Everything here is Tk-free; executors
run on a worker thread and return a :class:`ToolResult`.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .tools import Tool, ToolParam, ToolResult


# -----------------------------------------------------------------
# Shared resolver helpers
# -----------------------------------------------------------------
def _platforms(app) -> Dict[str, pd.DataFrame]:
    return getattr(app, "gpl_datasets", {}) or {}


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


def _classified_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if str(c).startswith("Classified_")]


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


_META_PREFIXES = ("GSM", "series_id")


def _find_gene_column(df: pd.DataFrame, gene: str) -> Optional[str]:
    """Case-insensitive match of a gene symbol to a platform gene column."""
    if not gene:
        return None
    g = str(gene).strip().upper()
    meta = {c for c in df.columns
            if c in _META_PREFIXES or str(c).startswith("Classified_")}
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
        rank_genes_by_variability,
        run_variability_gsea,
        DEFAULT_LIBRARIES,
        VARIABILITY_METHODS,
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
            df = _platforms(app)[key]
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
        df = plats[key]
        if "GSM" not in df.columns:
            raise ValueError(f"Platform {key!r} is not in canonical (GSM) format.")
        col = resolved.get("condition_column") or _pick_condition_column(df, None)
        if not col:
            raise ValueError(f"Platform {key!r} has no Classified_* condition column.")
        labels = _labels_from_column(df, col)
        # The canonical ranker treats every non-GSM/series_id column as a gene,
        # so drop the Classified_* metadata columns before handing it over.
        drop = [c for c in df.columns if str(c).startswith("Classified_")]
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
        return ToolResult(
            f"Condition enrichment ({case} vs {control}) on "
            f"{resolved.get('platform')}: {n} enriched term(s).",
            table=top, report=report, manifest=manifest,
            payload={"ranked": ranked, "gsea": gsea, "report": report})

    # ---- variability_enrichment ---------------------------------
    def _var_exec(app, resolved, progress_cb):
        from genevariate.core.analysis import (
            variability_report_markdown, VARIABILITY_DEFAULT_METHOD,
        )
        df, labels = _prep_labels(app, resolved)
        case = str(resolved.get("case_label"))
        control = str(resolved.get("control_label"))
        method = str(resolved.get("method") or "").strip()
        if method not in VARIABILITY_METHODS:
            method = None
        libs = [s.strip() for s in str(resolved.get("libraries", libs_default)).split(",")
                if s.strip()]
        progress_cb(20.0, "Ranking genes by variability…")
        kwargs = {"method": method} if method else {}
        ranked = rank_genes_by_variability(df, labels, case, control, **kwargs)
        progress_cb(60.0, "Running variability GSEA…")
        gsea = run_variability_gsea(ranked, gene_sets=libs)
        n = _gsea_term_count(gsea)
        top = gsea.head(15) if n else ranked.head(15)
        comparison = f"{case} vs {control} on {resolved.get('platform')}"
        try:
            report = variability_report_markdown(
                ranked, gsea, comparison, method or VARIABILITY_DEFAULT_METHOD)
        except Exception:
            report = ""
        return ToolResult(
            f"Variability enrichment ({case} vs {control}) on "
            f"{resolved.get('platform')}: {n} enriched term(s).",
            table=top, report=report,
            payload={"ranked": ranked, "gsea": gsea, "report": report})

    # ---- rank_genes (no GSEA) -----------------------------------
    def _rank_exec(app, resolved, progress_cb):
        df, labels = _prep_labels(app, resolved)
        case = str(resolved.get("case_label"))
        control = str(resolved.get("control_label"))
        progress_cb(40.0, "Ranking genes…")
        ranked = rank_genes_by_condition(df, labels, case, control,
                                         moderated=bool(resolved.get("moderated", False)))
        return ToolResult(
            f"Top-ranked genes ({case} vs {control}) on {resolved.get('platform')}.",
            table=ranked.head(25), payload={"ranked": ranked})

    # ---- run_ngs_de (raw counts -> DESeq2 -> GSEA) --------------
    def _ngs_resolver(app, raw):
        out = dict(raw)
        out.setdefault("libraries", libs_default)
        out.setdefault("min_count", 10)
        return out

    def _ngs_exec(app, resolved, progress_cb):
        import os
        from genevariate.core.count_io import load_counts
        from genevariate.core.analysis import (
            run_deseq2, deseq_results_to_ranked,
        )
        path = resolved.get("counts_path")
        if not path:
            return ToolResult("counts_path is required (CSV, 10x dir, or h5ad).",
                              ok=False)
        if not os.path.exists(path):
            return ToolResult(f"Count file not found: {path!r}. Provide a valid "
                              "CSV/TSV, 10x directory, or .h5ad path.", ok=False)
        column = resolved.get("condition_column")
        case = resolved.get("case_label")
        control = resolved.get("control_label")
        if not (column and case and control):
            return ToolResult(
                "condition_column, case_label and control_label are required "
                "for DESeq2.", ok=False)
        progress_cb(15.0, "Loading counts…")
        try:
            counts, meta = load_counts(path)
        except Exception as exc:
            return ToolResult(f"Could not read counts from {path!r}: {exc}",
                              ok=False)
        if meta is None or column not in meta.columns:
            return ToolResult(
                f"Sample metadata has no column {column!r}; DESeq2 needs a "
                "design factor from the count file's metadata.", ok=False)
        progress_cb(45.0, "Running DESeq2…")
        res = run_deseq2(counts, meta, (column, str(case), str(control)),
                         min_count=int(resolved.get("min_count", 10)))
        progress_cb(75.0, "Ranking + GSEA…")
        ranked = deseq_results_to_ranked(res)
        libs = [s.strip() for s in str(resolved.get("libraries", libs_default)).split(",")
                if s.strip()]
        gsea = run_prerank_gsea(ranked, gene_sets=libs)
        n = _gsea_term_count(gsea)

        # Optionally publish log-CPM-normalised counts as a platform so the NGS
        # result becomes a modality you can compare / connect against the others.
        registered = None
        register_as = str(resolved.get("register_as") or "").strip()
        if register_as:
            from genevariate.core.analysis import (
                cpm_normalize, counts_to_platform_df,
            )
            progress_cb(90.0, f"Registering {register_as} as a platform…")
            norm = cpm_normalize(counts, log=True)
            meta2 = meta.copy()
            if column in meta2.columns:
                meta2 = meta2.rename(columns={column: f"Classified_{column}"})
            plat_df = counts_to_platform_df(norm, sample_meta=meta2)
            key = (register_as if register_as not in _platforms(app)
                   else f"{register_as}_{len(_platforms(app))}")
            app.gpl_datasets[key] = plat_df
            registered = key
            try:
                app.after(0, app._update_platform_status)
            except Exception:
                pass

        extra = f" Registered normalised counts as platform {registered!r}." \
            if registered else ""
        manifest = _manifest("run_ngs_de", resolved,
                             inputs={"counts": path}, seed=42)
        report = _append_manifest("", manifest)
        return ToolResult(
            f"DESeq2 DE + GSEA ({case} vs {control}): {len(res)} genes tested, "
            f"{n} enriched term(s).{extra}",
            table=(gsea.head(15) if n else ranked.head(15)),
            report=report, manifest=manifest,
            payload={"deseq": res, "ranked": ranked, "gsea": gsea,
                     "registered": registered})

    # ---- classify_distributions (modality landscape) ------------
    def _modality_resolver(app, raw):
        out = dict(raw)
        out["platform"] = _match_platform(app, raw.get("platform"))
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
        progress_cb(30.0, f"Classifying gene distributions on {key}…")
        tags = classify_distributions(expr_df)
        if tags is None or tags.empty:
            return ToolResult(f"No gene columns to classify on {key!r}.", ok=False)
        summary = distribution_summary(tags)
        progress_cb(80.0, "Summarising modality landscape…")
        n_genes = int(len(tags))
        top = summary.sort_values("n_genes", ascending=False)
        parts = [f"{r['distribution']} {r['fraction'] * 100:.1f}%"
                 for _, r in top.head(4).iterrows()]
        lines = [f"# Distribution landscape — {key}\n",
                 f"Classified **{n_genes:,}** gene(s) into "
                 f"{len(summary)} modality class(es).\n"]
        for _, r in top.iterrows():
            lines.append(f"- **{r['distribution']}**: {int(r['n_genes']):,} genes "
                         f"({r['fraction'] * 100:.1f}%)")
        report = "\n".join(lines)
        return ToolResult(
            f"Modality landscape on {key}: {n_genes:,} genes — "
            + ", ".join(parts) + ".",
            table=top, report=report,
            payload={"platform": key, "tags": tags, "summary": summary,
                     "report": report})

    # ---- meta_enrichment (cross-platform consensus) -------------
    def _meta_resolver(app, raw):
        out = dict(raw)
        plats_arg = raw.get("platforms")
        if isinstance(plats_arg, str):
            plats_arg = [s.strip() for s in plats_arg.replace(";", ",").split(",")
                         if s.strip()]
        if plats_arg:
            resolved = [_match_platform(app, p) for p in plats_arg]
            out["platforms"] = [p for p in dict.fromkeys(resolved) if p]
        else:
            out["platforms"] = list(_platforms(app).keys())
        out.setdefault("libraries", libs_default)
        m = str(raw.get("method") or "rank_product").strip().lower()
        out["method"] = (m if m in ("rank_product", "stouffer", "random_effects")
                         else "rank_product")
        return out

    def _meta_exec(app, resolved, progress_cb):
        from genevariate.core.analysis import (
            combine_ranks, run_meta_enrichment_gsea,
            meta_enrichment_report_markdown,
        )
        keys = resolved.get("platforms") or []
        if len(keys) < 2:
            return ToolResult("Meta-enrichment needs at least two platforms. "
                              "Load or fetch them first.", ok=False)
        case = resolved.get("case_label")
        control = resolved.get("control_label")
        method = resolved.get("method", "rank_product")
        libs = [s.strip() for s in str(resolved.get("libraries", libs_default)).split(",")
                if s.strip()]
        per_platform: Dict[str, pd.DataFrame] = {}
        used: List[str] = []
        for i, key in enumerate(keys):
            progress_cb(10.0 + 50.0 * i / max(len(keys), 1),
                        f"Ranking {key}…")
            try:
                df, labels = _prep_labels(app, {
                    "platform": key,
                    "condition_column": resolved.get("condition_column"),
                })
                col = _pick_condition_column(_platforms(app)[key],
                                             resolved.get("condition_column"))
                c_case, c_control = case, control
                if not (c_case and c_control):
                    c_case, c_control = _default_case_control(
                        _platforms(app)[key], col)
                ranked = rank_genes_by_condition(df, labels, str(c_case),
                                                 str(c_control))
                per_platform[key] = ranked
                used.append(key)
            except Exception:
                continue
        if len(per_platform) < 2:
            return ToolResult(
                "Could not rank at least two platforms for meta-enrichment "
                "(need a shared Classified_* condition column on each).",
                ok=False)
        progress_cb(65.0, f"Combining ranks ({method})…")
        combined = combine_ranks(per_platform, method=method)
        gsea = None
        try:
            progress_cb(80.0, "Running consensus GSEA…")
            gsea = run_meta_enrichment_gsea(combined, gene_sets=libs)
        except Exception:
            gsea = None
        comparison = (f"{case} vs {control}" if case and control
                      else "case vs control")
        manifest = _manifest("meta_enrichment", resolved,
                             inputs={k: _platforms(app)[k] for k in used
                                     if k in _platforms(app)}, seed=42)
        try:
            report = meta_enrichment_report_markdown(
                combined, gsea, used, comparison, method)
        except Exception:
            report = ""
        report = _append_manifest(report, manifest)
        n = _gsea_term_count(gsea)
        top = gsea.head(15) if n else combined.head(15)
        return ToolResult(
            f"Cross-platform meta-enrichment ({method}) over {len(used)} "
            f"platform(s): {n} consensus term(s).",
            table=top, report=report, manifest=manifest,
            payload={"combined": combined, "gsea": gsea,
                     "platforms": used, "report": report})

    tools: Dict[str, Tool] = {}

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
                  help="Classified_* metadata column defining the groups."),
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

    var_params = list(cond_params) + [
        ToolParam("method", "str", required=False,
                  choices=tuple(VARIABILITY_METHODS),
                  help="Variability ranking method."),
    ]
    tools["variability_enrichment"] = Tool(
        name="variability_enrichment",
        description="Rank genes by differential variability and run GSEA.",
        params=var_params,
        resolver=_cond_resolver, executor=_var_exec,
        examples=("run variability enrichment on GPL96 case vs control",
                  "differential variability gsea", "delta variance analysis"))

    tools["rank_genes"] = Tool(
        name="rank_genes",
        description="Rank genes case-vs-control (no GSEA); show the top table.",
        params=cond_params,
        resolver=_cond_resolver, executor=_rank_exec,
        examples=("rank genes tumor vs normal", "top differential genes",
                  "which genes change case vs control"))

    tools["run_ngs_de"] = Tool(
        name="run_ngs_de",
        description="DESeq2 differential expression on a raw-count file, then GSEA.",
        params=[
            ToolParam("counts_path", "str",
                      help="Path to counts CSV/TSV, 10x dir, or .h5ad."),
            ToolParam("condition_column", "str",
                      help="Design factor column in the count metadata."),
            ToolParam("case_label", "str", help="Test level."),
            ToolParam("control_label", "str", help="Reference level."),
            ToolParam("libraries", "str", required=False, default=libs_default,
                      help="Comma-separated gene-set libraries."),
            ToolParam("min_count", "int", required=False, default=10,
                      help="Drop genes with total count below this."),
            ToolParam("register_as", "str", required=False,
                      help="If set, publish log-CPM counts under this platform "
                           "name so the NGS result can be compared/connected "
                           "against other modalities."),
        ],
        resolver=_ngs_resolver, executor=_ngs_exec,
        examples=("run deseq2 on counts.csv treated vs control",
                  "ngs differential expression from h5ad",
                  "raw count rna-seq de"))

    tools["classify_distributions"] = Tool(
        name="classify_distributions",
        description="Classify every gene on a platform by its expression "
                    "distribution (unimodal/bimodal/heavy-tailed) and summarise "
                    "the modality landscape.",
        params=[
            ToolParam("platform", "platform", required=False,
                      help="Platform to profile (defaults to the first loaded)."),
        ],
        resolver=_modality_resolver, executor=_modality_exec,
        examples=("classify the gene distributions on GPL570",
                  "show the modality landscape of my platform",
                  "how many genes are bimodal"))

    tools["meta_enrichment"] = Tool(
        name="meta_enrichment",
        description="Cross-platform consensus enrichment: rank each platform "
                    "case-vs-control, combine the rankings (rank-product or "
                    "Stouffer), then run GSEA on the consensus.",
        params=[
            ToolParam("platforms", "list", required=False,
                      help="Platforms to combine (defaults to all loaded)."),
            ToolParam("condition_column", "str", required=False,
                      help="Classified_* column shared across platforms."),
            ToolParam("case_label", "str", required=False,
                      help="Group treated as 'case'."),
            ToolParam("control_label", "str", required=False,
                      help="Group treated as 'control'."),
            ToolParam("method", "str", required=False, default="rank_product",
                      choices=("rank_product", "stouffer", "random_effects"),
                      help="Rank-combination method (random_effects = "
                           "DerSimonian-Laird effect-size meta-analysis)."),
            ToolParam("libraries", "str", required=False, default=libs_default,
                      help="Comma-separated Enrichr gene-set libraries."),
        ],
        resolver=_meta_resolver, executor=_meta_exec,
        examples=("run meta enrichment across GPL570 and GPL96 tumor vs normal",
                  "cross-platform consensus enrichment case vs control",
                  "combine rankings across platforms and run gsea"))

    # ---- gene_distribution --------------------------------------
    def _dist_resolver(app, raw):
        out = dict(raw)
        out["platform"] = _match_platform(app, raw.get("platform"))
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
        report = (f"# {gene} distribution — {key}\n\n"
                  f"- **Class**: {stats.get('distribution', '?')}\n"
                  f"- **Samples**: {stats.get('n', 0):,}\n"
                  f"- **Mean / median**: {stats.get('mean', float('nan')):.3g} / "
                  f"{stats.get('median', float('nan')):.3g}\n"
                  f"- **Std (CV)**: {stats.get('std', float('nan')):.3g} "
                  f"({cv:.2f})\n"
                  f"- **Range**: {stats.get('min', float('nan')):.3g} – "
                  f"{stats.get('max', float('nan')):.3g}\n")
        return ToolResult(summ, table=tbl, report=report,
                          payload={"gene": gene, "platform": key,
                                   "values": vec, "stats": stats,
                                   "report": report})

    tools["gene_distribution"] = Tool(
        name="gene_distribution",
        description="Profile one gene's distribution on a platform "
                    "(class + mean/median/std/modality).",
        params=[
            ToolParam("gene", "str", help="Gene symbol, e.g. TP53."),
            ToolParam("platform", "platform", required=False,
                      help="Platform to profile (defaults to the first loaded)."),
        ],
        resolver=_dist_resolver, executor=_dist_exec,
        examples=("analyze the distribution of TP53",
                  "distribution of gene BRCA1 on GPL570",
                  "profile EGFR expression"))

    # ---- compare_gene -------------------------------------------
    def _cmp_resolver(app, raw):
        out = dict(raw)
        plats_arg = raw.get("platforms")
        if isinstance(plats_arg, str):
            plats_arg = [s.strip() for s in plats_arg.replace(";", ",").split(",")
                         if s.strip()]
        if plats_arg:
            resolved = [_match_platform(app, p) for p in plats_arg]
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
                      f" — {'differ' if p < 0.05 else 'no significant difference'}.")
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
        return ToolResult(summ, table=table, report=report,
                          payload={"gene": gene, "sources": usable,
                                   "vectors": vecs, "report": report})

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
            sources, gene, method=str(resolved.get("method", "zscore")))
        tbl = res["table"]
        n_found = int(tbl["n"].fillna(0).gt(0).sum()) if "n" in tbl.columns else 0
        if n_found == 0:
            return ToolResult(f"{gene!r} was not found in any source.", ok=False)
        return ToolResult(res["summary"], table=res["table"],
                          report=res["report"],
                          payload={"gene": gene, "pairwise": res["pairwise"],
                                   "harmonized": res["harmonized"],
                                   "concordant": res["concordant"],
                                   "report": res["report"]})

    tools["compare_modalities"] = Tool(
        name="compare_modalities",
        description="Compare the SAME gene across different data modalities "
                    "(microarray / RNA-seq / single-cell) on a harmonised scale "
                    "(z-score or rank), then test whether its distribution shape "
                    "is consistent across modalities.",
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

    # ---- gene_connections (co-expression links) -----------------
    def _conn_resolver(app, raw):
        out = _cmp_resolver(app, raw)  # platforms parsing
        m = str(raw.get("method") or "pearson").strip().lower()
        out["method"] = m if m in ("pearson", "spearman", "rho") else "pearson"
        try:
            out["top_n"] = int(raw.get("top_n", 25))
        except (TypeError, ValueError):
            out["top_n"] = 25
        try:
            out["min_abs"] = float(raw.get("min_abs", 0.3))
        except (TypeError, ValueError):
            out["min_abs"] = 0.3
        return out

    def _conn_exec(app, resolved, progress_cb):
        from genevariate.core.analysis import (
            gene_coexpression, coexpression_consensus,
        )
        gene = str(resolved.get("gene") or "").strip()
        keys = resolved.get("platforms") or []
        if not gene:
            return ToolResult("Which gene's connections should I find?", ok=False)
        plats = _platforms(app)
        keys = [k for k in keys if k in plats]
        if not keys:
            return ToolResult("No matching source is loaded. Load one first.",
                              ok=False)
        method = str(resolved.get("method", "pearson"))
        top_n = int(resolved.get("top_n", 25))
        # >=2 sources -> cross-modality consensus; else within one source.
        if len(keys) >= 2:
            progress_cb(40.0, f"Finding consensus connections for {gene}…")
            sources = {k: plats[k] for k in keys}
            res = coexpression_consensus(
                sources, gene, method=method, top_n=top_n,
                min_abs=float(resolved.get("min_abs", 0.3)))
            if res["table"].empty:
                return ToolResult(res["summary"], ok=False)
            return ToolResult(res["summary"], table=res["table"],
                              report=res["report"],
                              payload={"gene": gene, "sources": res["sources"],
                                       "report": res["report"]})
        key = keys[0]
        progress_cb(40.0, f"Correlating {gene} against all genes on {key}…")
        try:
            tbl = gene_coexpression(plats[key], gene, method=method,
                                    top_n=top_n,
                                    min_abs=float(resolved.get("min_abs", 0.0)))
        except ValueError as exc:
            return ToolResult(str(exc), ok=False)
        if tbl.empty:
            return ToolResult(f"No co-expression partners found for {gene} on "
                              f"{key}.", ok=False)
        pos = tbl[tbl["r"] > 0].head(8)
        neg = tbl[tbl["r"] < 0].head(8)
        lines = [f"# {gene} connections — {key}\n",
                 f"Top co-expressed genes ({method}):\n"]
        for g, row in pos.iterrows():
            lines.append(f"- **{g}**: r={row['r']:.3f} (positive)")
        for g, row in neg.iterrows():
            lines.append(f"- **{g}**: r={row['r']:.3f} (inverse)")
        report = "\n".join(lines)
        top_names = ", ".join(list(tbl.index[:8]))
        return ToolResult(
            f"{gene} on {key}: {len(tbl)} connection(s). Top: {top_names}.",
            table=tbl, report=report,
            payload={"gene": gene, "platform": key, "report": report})

    tools["gene_connections"] = Tool(
        name="gene_connections",
        description="Find genes connected to a query gene by co-expression "
                    "(Pearson/Spearman) within one source; with two or more "
                    "sources, keep only the partners whose connection holds "
                    "across modalities (consensus, consistent sign).",
        params=[
            ToolParam("gene", "str", help="Query gene symbol."),
            ToolParam("platforms", "list", required=False,
                      help="Source(s) to search (defaults to all loaded); "
                           ">=2 gives a cross-modality consensus."),
            ToolParam("method", "str", required=False, default="pearson",
                      choices=("pearson", "spearman", "rho"),
                      help="Association method ('rho' = Lovell proportionality, "
                           "robust to compositional bias on normalised data)."),
            ToolParam("top_n", "int", required=False, default=25,
                      help="How many partners to return."),
            ToolParam("min_abs", "float", required=False, default=0.3,
                      help="Minimum |r| to keep a connection."),
        ],
        resolver=_conn_resolver, executor=_conn_exec,
        examples=("what genes are connected to TP53 on GPL570",
                  "find co-expression partners of EGFR consistent across platforms",
                  "which genes correlate with BRCA1"))

    # ---- load_geo_platform (headless, no dialogs) ---------------
    def _load_resolver(app, raw):
        return dict(raw)

    def _load_exec(app, resolved, progress_cb):
        name = str(resolved.get("platform") or "").strip()
        if not name:
            return ToolResult("Which platform (e.g. GPL570)?", ok=False)
        key_up = name.upper()
        if key_up in _platforms(app):
            df = _platforms(app)[key_up]
            return ToolResult(f"{key_up} already loaded ({df.shape[0]} samples).",
                              payload={"platform": key_up})
        progress_cb(20.0, f"Locating {key_up}…")
        try:
            available = app._discover_available_platforms()
        except Exception:
            available = {}
        path = available.get(key_up) or available.get(name)
        if not path:
            return ToolResult(
                f"No local file found for {key_up}. Download it via the GPL "
                "downloader, or place its CSV in the data directory.", ok=False)
        progress_cb(50.0, f"Reading {key_up}…")
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
        app.gpl_datasets[key_up] = df
        try:
            app.after(0, app._update_platform_status)
        except Exception:
            pass
        return ToolResult(f"Loaded {key_up}: {df.shape[0]} samples × "
                          f"{df.shape[1] - 1} columns.",
                          payload={"platform": key_up})

    tools["load_geo_platform"] = Tool(
        name="load_geo_platform",
        description="Load a locally-available GEO/GPL microarray platform "
                    "into memory so it can be analysed.",
        params=[ToolParam("platform", "str", help="Platform id, e.g. GPL570.")],
        resolver=_load_resolver, executor=_load_exec,
        examples=("load GPL570", "load the GEO platform GPL96",
                  "bring in microarray platform GPL10558"))

    # ---- fetch_single_cell (CELLxGENE census -> pseudobulk) -----
    def _sc_resolver(app, raw):
        out = dict(raw)
        out.setdefault("organism", "homo_sapiens")
        out.setdefault("max_cells", 20000)
        out.setdefault("name", "scRNA")
        return out

    def _sc_exec(app, resolved, progress_cb):
        try:
            from genevariate.sources.cellxgene import CensusClient
            from genevariate.utils.pseudobulk import (
                pseudobulk, pseudobulk_to_platform_df,
            )
        except Exception as exc:
            return ToolResult(
                "Single-cell fetch needs the CELLxGENE extra "
                f"(cellxgene_census + anndata): {exc}", ok=False)
        gene = str(resolved.get("gene") or "").strip()
        tissue = resolved.get("tissue") or None
        organism = resolved.get("organism", "homo_sapiens")
        try:
            max_cells = int(resolved.get("max_cells", 20000))
        except (TypeError, ValueError):
            max_cells = 20000
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
        finally:
            try:
                client.close()
            except Exception:
                pass
        progress_cb(70.0, "Pseudo-bulking cells…")
        groupby = [c for c in ("tissue", "cell_type") if c in adata.obs.columns]
        if not groupby:
            groupby = list(adata.obs.columns[:1])
        pb = pseudobulk(adata, groupby=tuple(groupby), min_cells=5)
        df = pseudobulk_to_platform_df(pb)
        if "GSM" not in df.columns:
            df = df.reset_index().rename(columns={df.columns[0]: "GSM"})
            df["GSM"] = df["GSM"].astype(str)
        name = str(resolved.get("name") or "scRNA")
        key = name if name not in _platforms(app) else f"{name}_{len(_platforms(app))}"
        app.gpl_datasets[key] = df
        try:
            app.after(0, app._update_platform_status)
        except Exception:
            pass
        return ToolResult(
            f"Fetched {adata.n_obs:,} cells → pseudo-bulked to {df.shape[0]} "
            f"samples; registered as platform {key!r}.",
            payload={"platform": key})

    # ---- run_analysis_code (sandboxed Python) -------------------
    def _code_resolver(app, raw):
        out = dict(raw)
        try:
            out["timeout"] = float(raw.get("timeout", 20.0))
        except (TypeError, ValueError):
            out["timeout"] = 20.0
        return out

    def _code_exec(app, resolved, progress_cb):
        from .code_exec import run_user_code
        code = str(resolved.get("code") or "").strip()
        if not code:
            return ToolResult("Provide a Python snippet in `code`. The loaded "
                              "platforms are available as `platforms` (a dict of "
                              "DataFrames); assign your answer to `result`.",
                              ok=False)
        progress_cb(30.0, "Running analysis code (sandboxed)…")
        res = run_user_code(code, _platforms(app),
                            timeout=float(resolved.get("timeout", 20.0)))
        if not res["ok"]:
            return ToolResult(f"Code did not run: {res['error']}", ok=False,
                              payload={"stdout": res.get("stdout", "")})
        stdout = res.get("stdout") or ""
        result = res.get("result")
        parts = []
        if stdout.strip():
            parts.append("output:\n" + stdout.strip())
        if result is not None and not isinstance(result, (pd.DataFrame, pd.Series)):
            parts.append(f"result = {result!r}")
        summary = "Code ran." + (" " + " | ".join(parts) if parts else
                                 " (no output; assign to `result` to return a value)")
        report_lines = ["# Analysis code result\n", "```python", code, "```"]
        if stdout.strip():
            report_lines += ["\n**Output:**\n```", stdout.rstrip(), "```"]
        return ToolResult(summary[:2000], table=res.get("result_table"),
                          report="\n".join(report_lines),
                          payload={"stdout": stdout, "result": result})

    tools["run_analysis_code"] = Tool(
        name="run_analysis_code",
        description="Run a short sandboxed Python snippet against the loaded "
                    "platforms for open-ended analysis. `platforms` is a dict of "
                    "canonical DataFrames; `pd`, `np` and the GeneVariate "
                    "analysis functions are available; assign your answer to "
                    "`result`. Imports, file/network access and dunder access "
                    "are blocked.",
        params=[
            ToolParam("code", "str",
                      help="Python snippet; use `platforms`, set `result`."),
            ToolParam("timeout", "float", required=False, default=20.0,
                      help="Wall-clock seconds before the snippet is abandoned."),
        ],
        resolver=_code_resolver, executor=_code_exec,
        examples=("compute the mean of TP53 across every loaded platform",
                  "how many genes overlap between GPL570 and GPL96",
                  "correlate the library sizes of all samples"))

    # ---- activity_inference (TF / pathway activities via decoupleR) ----
    def _activity_resolver(app, raw):
        out = dict(raw)
        out["platform"] = _match_platform(app, raw.get("platform"))
        k = str(raw.get("kind") or "tf").strip().lower()
        out["kind"] = k if k in ("tf", "pathway") else "tf"
        out.setdefault("organism", "human")
        return out

    def _activity_exec(app, resolved, progress_cb):
        try:
            from genevariate.core.analysis import run_activity
        except Exception as exc:
            return ToolResult(f"Activity inference unavailable: {exc}", ok=False)
        key = resolved.get("platform")
        plats = _platforms(app)
        if not key or key not in plats:
            return ToolResult("No matching platform is loaded. Load one first.",
                              ok=False)
        kind = str(resolved.get("kind", "tf"))
        organism = str(resolved.get("organism", "human"))
        progress_cb(30.0, f"Inferring {kind} activity on {key}…")
        try:
            res = run_activity(plats[key], kind=kind, organism=organism)
        except RuntimeError as exc:
            return ToolResult(str(exc), ok=False)
        manifest = _manifest("activity_inference", resolved,
                             inputs={"platform": plats[key]})
        report = _append_manifest(res.get("report", ""), manifest)
        return ToolResult(res["summary"], table=res["ranked"].head(15),
                          report=report, manifest=manifest,
                          payload={"activities": res["activities"],
                                   "ranked": res["ranked"], "report": report})

    tools["activity_inference"] = Tool(
        name="activity_inference",
        description="Infer transcription-factor (CollecTRI) or pathway "
                    "(PROGENy) activity per sample via decoupleR — the "
                    "mechanistic step beyond over-represented gene lists.",
        params=[
            ToolParam("platform", "platform", required=False,
                      help="Platform to score (defaults to the first loaded)."),
            ToolParam("kind", "str", required=False, default="tf",
                      choices=("tf", "pathway"),
                      help="'tf' for CollecTRI TF activities, 'pathway' for "
                           "PROGENy pathway activities."),
            ToolParam("organism", "str", required=False, default="human",
                      help="Organism for the regulatory network."),
        ],
        resolver=_activity_resolver, executor=_activity_exec,
        examples=("infer TF activity on GPL570",
                  "run progeny pathway activity on my platform",
                  "which transcription factors are active"))

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
            ToolParam("max_cells", "int", required=False, default=20000,
                      help="Cap on cells fetched."),
            ToolParam("name", "str", required=False, default="scRNA",
                      help="Platform name to register under."),
        ],
        resolver=_sc_resolver, executor=_sc_exec,
        examples=("fetch single cell data for TP53 in lung",
                  "get scRNA-seq from cellxgene for brain",
                  "pull single-cell data for EGFR"))

    return tools
