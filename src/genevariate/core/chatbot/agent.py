"""
Agentic planner/executor for the GeneVariate assistant.

Where ``router.route`` maps a prompt to ONE tool, the agent turns a *goal*
("analyze the distribution of TP53 across single-cell and GEO data, then
compare them") into an ordered **plan** of tool calls, executes them in
sequence while narrating progress, threads context between steps (a platform
loaded in step 1 is analysed in step 2), and synthesises a final answer.

Everything here is Tk-free and never raises: a local LLM proposes the plan when
Ollama is up, otherwise a deterministic heuristic planner covers the common
"profile / compare a gene across sources" goals. The sidebar shows the plan and
only runs it after the user approves — the human confirmation gate stays.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .tools import Tool, ToolResult


# -----------------------------------------------------------------
# Plan value objects
# -----------------------------------------------------------------
@dataclass
class Step:
    tool: str
    params: Dict[str, Any] = field(default_factory=dict)
    note: str = ""


@dataclass
class Plan:
    goal: str
    steps: List[Step] = field(default_factory=list)
    source: str = "heuristic"      # "llm" | "heuristic"
    message: str = ""


@dataclass
class StepOutcome:
    step: Step
    result: Optional[ToolResult]
    error: str = ""


@dataclass
class AgentRun:
    plan: Plan
    outcomes: List[StepOutcome] = field(default_factory=list)
    summary: str = ""


# -----------------------------------------------------------------
# Prompt-parsing helpers (shared by the heuristic planner)
# -----------------------------------------------------------------
_GPL = re.compile(r"\bGPL\d+\b", re.IGNORECASE)
_STOPWORDS = {
    "ANALYZE", "ANALYSE", "DISTRIBUTION", "COMPARE", "GENE", "GENES", "DATA",
    "SINGLE", "CELL", "GEO", "FROM", "ACROSS", "BETWEEN", "THEM", "THE", "AND",
    "OF", "IN", "ON", "TO", "WITH", "RUN", "SHOW", "PLOT", "PROFILE", "EXPRESSION",
    "PLATFORM", "PLATFORMS", "SCRNA", "RNASEQ", "TISSUE", "VS", "VERSUS", "FOR",
    "CELLXGENE", "MICROARRAY", "BULK", "COUNTS", "DATASET", "DATASETS",
    # common filler / pronouns that must never be read as a gene symbol
    "MY", "YOUR", "OUR", "ME", "US", "IT", "ITS", "FULL", "ANALYSIS", "ANALYSES",
    "DO", "GET", "PLEASE", "ALL", "AN", "IS", "ARE", "LET", "NOW", "HELP", "WHAT",
    "WHICH", "HOW", "THAT", "THIS", "THESE", "THOSE", "SET", "ANY", "EACH",
}
_GENE_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,9}\b")


def _extract_gene(goal: str) -> Optional[str]:
    """Best-effort gene-symbol extraction from a free-text goal."""
    # explicit "gene X" phrasing wins
    m = re.search(r"\bgene\s+([A-Za-z][A-Za-z0-9\-]{1,9})\b", goal, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\bof\s+([A-Za-z][A-Za-z0-9\-]{1,9})\b", goal, re.IGNORECASE)
    if m and m.group(1).upper() not in _STOPWORDS:
        return m.group(1).upper()
    # Otherwise only accept tokens the user actually wrote in UPPER case: gene
    # symbols are conventionally uppercase (TP53, EGFR), whereas ordinary words
    # ("load", "condition", "run") are lowercase — matching the ORIGINAL text
    # (not an upper-cased copy) avoids reading verbs/nouns as genes.
    for tok in _GENE_RE.findall(goal):
        if tok not in _STOPWORDS and not _GPL.fullmatch(tok):
            return tok
    return None


def _wants_single_cell(goal: str) -> bool:
    g = goal.lower()
    return any(k in g for k in ("single cell", "single-cell", "scrna",
                                "cellxgene", "census"))


def _wants_geo(goal: str) -> bool:
    g = goal.lower()
    return bool(_GPL.search(goal)) or "geo" in g or "microarray" in g


def _tissue(goal: str) -> Optional[str]:
    m = re.search(r"\bin\s+([a-z][a-z ]{2,20})\b", goal, re.IGNORECASE)
    if m:
        cand = m.group(1).strip()
        if cand.upper() not in _STOPWORDS:
            return cand.split()[0]
    return None


# -----------------------------------------------------------------
# Heuristic planner
# -----------------------------------------------------------------
def _heuristic_plan(goal: str, app, registry: Dict[str, Tool]) -> Plan:
    gene = _extract_gene(goal)
    steps: List[Step] = []
    produced: List[str] = []

    loaded = list(getattr(app, "gpl_datasets", {}) or {})

    # 1) acquire data sources
    if _wants_single_cell(goal) and "fetch_single_cell" in registry:
        params: Dict[str, Any] = {"name": "scRNA"}
        if gene:
            params["gene"] = gene
        t = _tissue(goal)
        if t:
            params["tissue"] = t
        steps.append(Step("fetch_single_cell", params,
                          "Fetch single-cell data from CELLxGENE"))
        produced.append("scRNA")

    if _wants_geo(goal) and "load_geo_platform" in registry:
        gpls = list(dict.fromkeys(g.upper() for g in _GPL.findall(goal)))
        if gpls:
            for gpl in gpls:
                steps.append(Step("load_geo_platform", {"platform": gpl},
                                  f"Load GEO platform {gpl}"))
                produced.append(gpl)
        elif loaded:
            produced.append(loaded[0])

    # if nothing to acquire, fall back to whatever is already loaded
    if not produced:
        produced = loaded[:2] if loaded else []

    # 2) profile the gene on each source
    if gene and "gene_distribution" in registry:
        for src in produced:
            steps.append(Step("gene_distribution",
                              {"gene": gene, "platform": src},
                              f"Profile {gene} on {src}"))

    # 3) compare across sources when asked (or when >1 source and a gene)
    wants_compare = "compare" in goal.lower() or len(produced) >= 2
    if gene and wants_compare and len(produced) >= 2 and "compare_gene" in registry:
        steps.append(Step("compare_gene",
                          {"gene": gene, "platforms": list(produced)},
                          f"Compare {gene} across {', '.join(produced)}"))

    # 4) analytical intent — enrichment / ranking / modality / meta.
    # Reuse the deterministic keyword router to pick the tool + params, then
    # target whatever platform(s) we just loaded, so an offline goal like
    # "run condition enrichment on GPL570 tumor vs normal" runs the real
    # analysis instead of degrading to a bare gene profile.
    from .router import _keyword_route
    _ANALYTIC = {"condition_enrichment", "variability_enrichment", "rank_genes",
                 "classify_distributions", "meta_enrichment"}
    if not any(s.tool in _ANALYTIC for s in steps):
        act = _keyword_route(goal, registry)
        if act.tool in _ANALYTIC and act.tool in registry:
            p = dict(act.params)
            if act.tool == "meta_enrichment":
                if not p.get("platforms") and len(produced) >= 2:
                    p["platforms"] = list(produced)
            else:
                if not p.get("platform") and produced:
                    p["platform"] = produced[0]
            steps.append(Step(act.tool, p, f"Run {act.tool}"))

    msg = ""
    if not steps:
        msg = ("I couldn't turn that into a plan. Try e.g. “analyze the "
               "distribution of TP53 across single-cell and GPL570, then "
               "compare them”.")
    return Plan(goal=goal, steps=steps, source="heuristic", message=msg)


# -----------------------------------------------------------------
# LLM planner
# -----------------------------------------------------------------
def _plan_system_prompt(registry: Dict[str, Tool]) -> str:
    lines = [
        "You are an analysis planner. Break the user's GOAL into an ordered",
        "list of steps, each calling ONE tool. Reply with ONLY a JSON array:",
        '[{"tool": <name>, "params": {..}, "note": <short>}]',
        "Chain steps: a platform loaded/fetched earlier can be analysed later.",
        "Available tools:",
    ]
    for t in registry.values():
        pnames = ", ".join(p.name for p in t.params) or "(none)"
        lines.append(f"- {t.name}: {t.description} params: {pnames}")
    lines.append(
        'Example goal "distribution of TP53 in single-cell and GPL570, compare":')
    lines.append(
        '[{"tool":"fetch_single_cell","params":{"gene":"TP53","name":"scRNA"}},'
        '{"tool":"load_geo_platform","params":{"platform":"GPL570"}},'
        '{"tool":"gene_distribution","params":{"gene":"TP53","platform":"scRNA"}},'
        '{"tool":"gene_distribution","params":{"gene":"TP53","platform":"GPL570"}},'
        '{"tool":"compare_gene","params":{"gene":"TP53",'
        '"platforms":["scRNA","GPL570"]}}]')
    return "\n".join(lines)


def _extract_json_array(text: str) -> Optional[list]:
    if not text:
        return None
    start = text.find("[")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, list):
                            return obj
                    except (json.JSONDecodeError, ValueError):
                        break
        start = text.find("[", start + 1)
    return None


def _llm_plan(goal: str, registry: Dict[str, Tool]) -> Optional[Plan]:
    try:
        from genevariate.core import ollama_manager as om
        from genevariate.core.llm_extractor import llm_backend
    except Exception:
        return None
    model = getattr(om, "DEFAULT_MODEL", "gemma4:e2b")
    try:
        if not om.ollama_server_ok() or not om.model_available(model):
            return None
    except Exception:
        return None
    messages = [
        {"role": "system", "content": _plan_system_prompt(registry)},
        {"role": "user", "content": goal},
    ]
    try:
        text = llm_backend.chat(messages, model=model, temperature=0.0,
                                num_predict=512, think=False, timeout=45)
    except Exception:
        return None
    arr = _extract_json_array(text)
    if not arr:
        return None
    steps: List[Step] = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        name = item.get("tool")
        if name not in registry:
            continue
        params = item.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        steps.append(Step(name, registry[name].coerce(params),
                          str(item.get("note", ""))))
    if not steps:
        return None
    return Plan(goal=goal, steps=steps, source="llm")


# -----------------------------------------------------------------
# Public planning entry point
# -----------------------------------------------------------------
def plan(goal: str, app, registry: Dict[str, Tool]) -> Plan:
    """Produce an ordered execution plan for ``goal`` (LLM first, else heuristic)."""
    if not goal or not goal.strip():
        return Plan(goal=goal or "", steps=[], message="Empty goal.")
    llm = _llm_plan(goal, registry)
    if llm is not None and llm.steps:
        # If the small model omitted obvious steps, keep it — the user can edit.
        return llm
    return _heuristic_plan(goal, app, registry)


# -----------------------------------------------------------------
# Executor
# -----------------------------------------------------------------
def _inject_context(tool: Tool, params: Dict[str, Any],
                    context: Dict[str, Any]) -> Dict[str, Any]:
    names = {p.name for p in tool.params}
    raw = dict(params)
    if "gene" in names and not raw.get("gene") and context.get("gene"):
        raw["gene"] = context["gene"]
    if "platform" in names and not raw.get("platform") and context.get("last_platform"):
        raw["platform"] = context["last_platform"]
    if "platforms" in names and not raw.get("platforms") and context.get("platforms"):
        raw["platforms"] = list(context["platforms"])
    return raw


def _update_context(raw: Dict[str, Any], result: Optional[ToolResult],
                    context: Dict[str, Any]) -> None:
    if raw.get("gene"):
        context["gene"] = raw["gene"]
    if result is not None and getattr(result, "payload", None):
        p = result.payload.get("platform")
        if p:
            context["last_platform"] = p
            plats = context.setdefault("platforms", [])
            if p not in plats:
                plats.append(p)


def run_plan(app, plan_obj: Plan, registry: Dict[str, Tool],
             on_event: Callable[[str, str, Optional[ToolResult]], None],
             *, progress_cb: Optional[Callable[[float, str], None]] = None,
             should_stop: Optional[Callable[[], bool]] = None,
             context: Optional[Dict[str, Any]] = None) -> AgentRun:
    """Execute ``plan_obj`` step by step, narrating through ``on_event``.

    ``on_event(kind, text, result)`` kinds: ``step_start``, ``step_result``,
    ``step_error``, ``final``. ``progress_cb(value, text)`` drives the shared
    progress bar; ``should_stop()`` lets the UI cancel between steps.
    """
    context = context or {}
    run = AgentRun(plan=plan_obj)
    total = max(len(plan_obj.steps), 1)

    for i, step in enumerate(plan_obj.steps):
        if should_stop and should_stop():
            on_event("final", "Stopped by user.", None)
            run.summary = "Stopped by user."
            return run
        tool = registry.get(step.tool)
        base = 100.0 * i / total
        if progress_cb:
            progress_cb(base, step.note or step.tool)
        if tool is None:
            out = StepOutcome(step, None, f"Unknown tool {step.tool!r}")
            run.outcomes.append(out)
            on_event("step_error", out.error, None)
            continue
        on_event("step_start", step.note or f"Running {tool.name}", None)
        try:
            raw = _inject_context(tool, step.params, context)
            resolved = tool.coerce(tool.resolver(app, raw))

            def _step_progress(v: float, t: str) -> None:
                if progress_cb:
                    progress_cb(base + (v / total), t)

            result = tool.executor(app, resolved, _step_progress)
            _update_context(raw, result, context)
            run.outcomes.append(StepOutcome(step, result))
            on_event("step_result", result.summary if result else "done", result)
        except Exception as exc:  # never raise out of the agent
            import traceback
            tb = traceback.format_exc(limit=2)
            out = StepOutcome(step, None, f"{exc}\n{tb}")
            run.outcomes.append(out)
            on_event("step_error", f"{tool.name} failed: {exc}", None)

    run.summary = _synthesize(run)
    on_event("final", run.summary, None)
    if progress_cb:
        progress_cb(100.0, "Done")
    return run


def _synthesize(run: AgentRun) -> str:
    ok = [o for o in run.outcomes if o.result and o.result.ok]
    fail = [o for o in run.outcomes if o.error or (o.result and not o.result.ok)]
    lines = [f"Completed {len(ok)}/{len(run.outcomes)} step(s) for: "
             f"{run.plan.goal}"]
    for o in run.outcomes:
        if o.result is not None:
            mark = "✓" if o.result.ok else "•"
            lines.append(f"  {mark} {o.result.summary}")
        elif o.error:
            lines.append(f"  ✗ {o.step.tool}: {o.error.splitlines()[0]}")
    if fail:
        lines.append("Some steps need attention (see above).")
    return "\n".join(lines)
