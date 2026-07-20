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
        top = gsea.head(15) if gsea is not None and not gsea.empty else ranked.head(15)
        n = 0 if gsea is None else len(gsea)
        return ToolResult(
            f"Condition enrichment ({case} vs {control}) on "
            f"{resolved.get('platform')}: {n} enriched term(s).",
            table=top, payload={"ranked": ranked, "gsea": gsea})

    # ---- variability_enrichment ---------------------------------
    def _var_exec(app, resolved, progress_cb):
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
        top = gsea.head(15) if gsea is not None and not gsea.empty else ranked.head(15)
        n = 0 if gsea is None else len(gsea)
        return ToolResult(
            f"Variability enrichment ({case} vs {control}) on "
            f"{resolved.get('platform')}: {n} enriched term(s).",
            table=top, payload={"ranked": ranked, "gsea": gsea})

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
        from genevariate.core.count_io import load_counts
        from genevariate.core.analysis import (
            run_deseq2, deseq_results_to_ranked,
        )
        path = resolved.get("counts_path")
        if not path:
            return ToolResult("counts_path is required (CSV, 10x dir, or h5ad).",
                              ok=False)
        column = resolved.get("condition_column")
        case = resolved.get("case_label")
        control = resolved.get("control_label")
        if not (column and case and control):
            return ToolResult(
                "condition_column, case_label and control_label are required "
                "for DESeq2.", ok=False)
        progress_cb(15.0, "Loading counts…")
        counts, meta = load_counts(path)
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
        n = 0 if gsea is None else len(gsea)
        return ToolResult(
            f"DESeq2 DE + GSEA ({case} vs {control}): {len(res)} genes tested, "
            f"{n} enriched term(s).",
            table=(gsea.head(15) if gsea is not None and not gsea.empty
                   else ranked.head(15)),
            payload={"deseq": res, "ranked": ranked, "gsea": gsea})

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
        ],
        resolver=_ngs_resolver, executor=_ngs_exec,
        examples=("run deseq2 on counts.csv treated vs control",
                  "ngs differential expression from h5ad",
                  "raw count rna-seq de"))

    return tools
