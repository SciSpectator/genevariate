"""
LangChain-powered reasoning agent for the GeneVariate assistant.

Unlike :mod:`router` (one JSON-named tool per prompt) and :mod:`agent` (an
up-front plan the user approves), this module wires the *existing* registry
tools into a genuine tool-calling reasoning loop: the LLM reads the user's goal,
decides which analysis tools to call and in what order, observes each result,
and keeps going until it can answer. It is built on LangChain 1.x's unified
``create_agent`` (a LangGraph ReAct agent).

The chat backend is **pluggable** via ``GENEVARIATE_AGENT_BACKEND``:

* ``groq`` (default) - Groq's free hosted API (Llama-3.3-70B), the strongest
  and fastest tool-caller; needs a free ``GROQ_API_KEY`` (console.groq.com/keys).
* ``ollama`` - a fully local model (default ``qwen2.5:7b``), private + offline,
  auto-installed and auto-pulled on first use.
* anything else - an OpenAI-compatible endpoint (OpenRouter, Gemini-via-OpenAI,
  Cerebras, NVIDIA NIM) via ``GENEVARIATE_AGENT_BASE_URL`` +
  ``GENEVARIATE_AGENT_API_KEY``.

Because the agent drives GeneVariate's *Python API* (each tool wraps a real
analysis function), a reliable tool-caller matters far more than raw size - no
screen/vision model is involved.

Everything degrades gracefully: if the stack or a key is missing,
:func:`agent_available` returns ``False`` and callers fall back to the heuristic
planner in :mod:`agent`. Nothing here imports Tkinter.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .tools import Tool, ToolResult

# ---- base LangChain stack (import-guarded; backend clients are lazy) --
try:
    from langchain.agents import create_agent  # LangChain 1.x unified agent
    from langchain_core.tools import StructuredTool
    from langgraph.checkpoint.memory import InMemorySaver
    from pydantic import BaseModel, Field, create_model
    _HAS_LANGCHAIN = True
    _IMPORT_ERROR = ""
except Exception as _exc:  # pragma: no cover - exercised only w/o extras
    create_agent = None  # type: ignore
    StructuredTool = None  # type: ignore
    InMemorySaver = None  # type: ignore
    BaseModel = object  # type: ignore
    Field = None  # type: ignore
    create_model = None  # type: ignore
    _HAS_LANGCHAIN = False
    _IMPORT_ERROR = repr(_exc)


# ---- conversation memory --------------------------------------------
# One saver for the process, keyed by the caller's thread id. A run is a fresh
# graph over the live registry, so without this every turn starts blank and a
# follow-up ("now split that gene by it") has nothing to refer back to. Held
# here rather than in the window because the graph is rebuilt on each call.
_MEMORY = None


def _memory():
    """The shared checkpointer, or ``None`` when the stack is missing."""
    global _MEMORY
    if _MEMORY is None and InMemorySaver is not None:
        _MEMORY = InMemorySaver()
    return _MEMORY


# ---- backend + model configuration ----------------------------------
DEFAULT_BACKEND = "ollama"
# The local default is the 12B: it is already Q4-quantized (~7.5 GB, fits an
# 11 GB card) and it calls tools reliably. The e2b it replaced answers faster
# but misses tool calls often enough that the run ends with no analysis at all.
# Tool-calling accuracy is the whole assistant now that nothing stands behind
# it, so it is the property to pick a model on.
_BACKEND_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "ollama": "gemma4:12b-it-q4_K_M",
    "openai": "gpt-4o-mini",
}
DEFAULT_AGENT_MODEL = _BACKEND_MODELS[DEFAULT_BACKEND]


def _backend() -> str:
    return (os.environ.get("GENEVARIATE_AGENT_BACKEND") or DEFAULT_BACKEND) \
        .strip().lower() or DEFAULT_BACKEND


def _default_model(backend: Optional[str] = None) -> str:
    """Model the agent drives, overridable via ``GENEVARIATE_AGENT_MODEL``."""
    env = os.environ.get("GENEVARIATE_AGENT_MODEL", "").strip()
    if env:
        return env
    return _BACKEND_MODELS.get(backend or _backend(), _BACKEND_MODELS["ollama"])


# ---- persisted API keys (so a free key is entered only once) --------
_CFG_DIR = pathlib.Path.home() / ".genevariate"
_CFG_FILE = _CFG_DIR / "agent.json"


def _load_persisted_keys() -> None:
    try:
        data = json.loads(_CFG_FILE.read_text())
    except Exception:
        return
    for env_name, val in (data or {}).items():
        if val and not os.environ.get(env_name):
            os.environ[env_name] = str(val)


def persist_api_key(env_name: str, key: str) -> None:
    """Set ``env_name`` for this session and save it under ~/.genevariate."""
    key = (key or "").strip()
    if not key:
        return
    os.environ[env_name] = key
    try:
        _CFG_DIR.mkdir(parents=True, exist_ok=True)
        data: Dict[str, str] = {}
        if _CFG_FILE.exists():
            try:
                data = json.loads(_CFG_FILE.read_text())
            except Exception:
                data = {}
        data[env_name] = key
        _CFG_FILE.write_text(json.dumps(data))
        try:
            os.chmod(_CFG_FILE, 0o600)
        except Exception:
            pass
    except Exception:
        pass


_load_persisted_keys()


def api_key_prompt(backend: Optional[str] = None) -> Optional[Dict[str, str]]:
    """Describe the free key a hosted backend needs, if it isn't set yet.

    Returns ``{"backend","label","env","url"}`` or ``None`` when no key is
    required (local ``ollama``) or one is already present.
    """
    backend = backend or _backend()
    if backend == "groq":
        if not os.environ.get("GROQ_API_KEY"):
            return {"backend": "groq", "label": "Groq", "env": "GROQ_API_KEY",
                    "url": "https://console.groq.com/keys"}
        return None
    if backend == "ollama":
        return None
    if not (os.environ.get("GENEVARIATE_AGENT_API_KEY")
            or os.environ.get("OPENAI_API_KEY")):
        return {"backend": backend, "label": backend, "url": "",
                "env": "GENEVARIATE_AGENT_API_KEY"}
    return None


# ---- import (re)enable after an on-demand install -------------------
def _try_enable() -> bool:
    """(Re)attempt the base imports after an on-demand pip install."""
    global create_agent, StructuredTool, BaseModel
    global Field, create_model, _HAS_LANGCHAIN, _IMPORT_ERROR
    if _HAS_LANGCHAIN:
        return True
    try:
        import importlib
        importlib.invalidate_caches()
        from langchain.agents import create_agent as _ca
        from langchain_core.tools import StructuredTool as _st
        from pydantic import BaseModel as _bm, Field as _f, create_model as _cm
        create_agent, StructuredTool = _ca, _st
        BaseModel, Field, create_model = _bm, _f, _cm
        _HAS_LANGCHAIN = True
        _IMPORT_ERROR = ""
        return True
    except Exception as exc:  # pragma: no cover - only w/o the extra
        _IMPORT_ERROR = repr(exc)
        return False


def _module_present(name: str) -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


SYSTEM_PROMPT = (
    "You are GeneVariate's analysis agent. You help a bioinformatician analyse "
    "gene-expression data by CALLING TOOLS - you never invent numbers.\n\n"
    "ANSWER EVERY REQUEST FROM A FRESH TOOL CALL. Earlier turns in this "
    "conversation are context, not results you may reuse: a question that looks "
    "like one you already answered will usually differ in the platform, the "
    "gene or the interval, and quoting the earlier numbers would silently "
    "report the wrong dataset. If you find yourself about to answer because "
    "'I computed this above', call the tool again with the parameters of the "
    "CURRENT request. The exception is a request that is explicitly about the "
    "session itself - 'summarise what you found', 'which of these could you "
    "not answer' - where restating earlier results is the whole point. There, "
    "report what actually happened, including every question you could not "
    "answer and why; a gap you leave out reads as a result you obtained.\n\n"
    "Workflow:\n"
    "1. Understand the user's goal (a gene, a platform, single-cell vs GEO, a "
    "comparison, an enrichment).\n"
    "2. REASON about WHICH dataset the user actually wants, then obtain it "
    "yourself before analysing. Choose the right source for the intent - it is "
    "NOT always a GEO platform:\n"
    "   • a GEO/GPL microarray id (e.g. GPL570) → `load_geo_platform` (it "
    "AUTO-DOWNLOADS from GEO when the platform isn't on disk);\n"
    "   • single-cell / scRNA-seq / a cell type / tissue / 'CELLxGENE' → "
    "`fetch_single_cell` (pulls the census and pseudo-bulks it);\n"
    "   • a path to the user's OWN expression matrix → `add_custom_platform` "
    "(needs a name to load it under);\n"
    "   • a path to a prepared table of sample LABELS → `load_label_file` "
    "(the platform is read from the file name, so it must contain a GPL id).\n"
    "   What is already loaded is listed under CURRENT SESSION STATE at the end "
    "of these instructions - read it there rather than calling `list_platforms`, "
    "and never re-acquire something it lists. If the data "
    "the user described is NOT loaded, DECIDE ON YOUR OWN which acquisition tool "
    "fits and call it - infer the source from the wording, don't assume GEO. "
    "NEVER stop to ask the user to download data and NEVER tell them a file is "
    "missing; obtain it and continue. Only report a data problem if the tool "
    "itself returns an error you cannot work around.\n"
    "3. Run the analysis: `gene_distribution` to profile one gene on one "
    "platform (pass `by_label` to split it by an extracted label - "
    "Tissue/Sex/Condition/… - with a Kruskal-Wallis/ANOVA test); "
    "`compare_gene` to contrast a gene across two or more sources; "
    "`condition_enrichment` for case-vs-control ranking + GSEA.\n"
    "3a. ENTITY-LINKED LABELS. The extractor's phase 2 resolves every Tissue/"
    "Condition/Treatment value against a vocabulary and records the accession: "
    "`D######` is a MeSH heading, `CVCL_*` is a Cellosaurus registration - the "
    "sample is a catalogued CELL LINE, not a piece of tissue - and `ART-*` was "
    "minted locally because nothing recognised the value. Phase 1/1b are "
    "verbatim and carry no accession. When the user asks which labels are cell "
    "lines, what a label was resolved to, or about MeSH/Cellosaurus/OOV ids, "
    "call `label_entities` - never guess an accession. To ANALYSE the "
    "distinction rather than list it, pass the derived label `Tissue_kind` "
    "(values Tissue / Cell line / Mixed / Unresolved) as `by_label` to "
    "`gene_distribution`.\n"
    "3aa. REGIONS. A region is the samples whose expression of one or more "
    "genes falls inside an interval on that gene's own scale. Every region "
    "tool takes the interval THREE interchangeable ways, and you must use the "
    "one the user actually described: `low`/`high` for explicit expression "
    "bounds (the numbers a drag-selection in the Explorer gives - use these "
    "whenever the user quotes real values such as 'between 3.828 and 7.848'); "
    "`sd` for a mean + k*SD tail ('the 3SD tail' -> sd=3); `quantile` for the "
    "gene's own upper quantile ('the top decile' -> quantile=0.9, 'the top "
    "20%' -> quantile=0.8). Explicit bounds beat `sd`, and `sd` beats "
    "`quantile`. Never silently substitute a different rule or a different "
    "number from the one asked for, and state the interval you used. "
    "`region_enrichment` is the one to reach for when the user asks what a "
    "region IS enriched for, or asks for q values: it is the Region Analysis "
    "window's Enrichment tab, a one-sided Fisher exact test per label value "
    "with BH across every label column tested in that one call. Pass all the "
    "label columns of interest in a single call - the correction is applied "
    "across the whole grid, so splitting them over several calls would report "
    "a smaller q for each than the data earned. "
    "`region_comparison` puts several "
    "single-gene regions in one grid and tests them against each other, with "
    "BH across the whole grid and the effective sample size beside every cell. "
    "`pooled_enrichment` answers whether a "
    "region's enrichment holds on more than one platform: the same rule and "
    "the same gene on every loaded platform, each keeping its own cut and its "
    "own background, pooled with a random-effects odds ratio. Quote I^2 and "
    "how many platforms agree in sign, never a pooled OR on its own. "
    "GEO samples "
    "arrive in study-sized clumps, so quote the effective sample size and the "
    "number of contributing studies whenever you quote a region effect.\n"
    "3a2. COMPARING GROUPS WITHIN ONE PLATFORM is `compare_distributions`, "
    "not `compare_gene`. `compare_gene` puts one gene side by side across "
    "several PLATFORMS; `compare_distributions` splits ONE platform by a label "
    "column (Tissue/Condition/Sex/Treatment) and compares those groups. It is "
    "the Compare Distributions window: per-group N/mean/median/SD/IQR, and per "
    "pair the Wasserstein distance (in expression units), the Jensen-Shannon "
    "divergence (shape), the difference of means, a Wilcoxon rank-sum test and "
    "a BH q corrected across EVERY pair - with k groups there are k(k-1)/2 of "
    "them, so quote the q, never the raw p. Distance and test answer different "
    "questions: a pair can be far apart and still ns because both groups are "
    "small, so quote both. Samples whose label records no value are dropped as "
    "a coverage gap, not kept as a group.\n"
    "3a3. COMPARING WHOLE PLATFORMS. `platform_gene_overlap` answers what the "
    "platforms MEASURE - gene counts, what is common to all, pairwise Jaccard, "
    "what only one carries. It reads the gene maps, never the expression, so "
    "use it first whenever a claim is about to be made about all platforms at "
    "once: a gene absent from a platform is unmeasured there, not switched "
    "off. `cross_platform_comparison` answers whether the platforms AGREE on "
    "the genes they share - every shared gene tested against a reference, p "
    "widened by the study design effect, the k-1 tests per gene Sidak-"
    "corrected into one, then BH across genes; it returns the genes that "
    "differ, the ones conserved, the batch-effect score and the pairwise "
    "Spearman. Keep the three apart: `compare_gene` is ONE gene across "
    "platforms, `cross_platform_comparison` is EVERY shared gene, and "
    "`pooled_enrichment` is one REGION's label enrichment pooled over "
    "platforms. Two cautions. Report the batch-effect score as a batch effect "
    "only when the tool does; across different technologies (array intensity "
    "vs sequencing counts) it is a difference of units, and the tool says so "
    "- repeat that instead. And a correction is not free: quantile "
    "normalization erases the distribution shape this program measures "
    "elsewhere, so leave batch_method at 'none' unless the user asks.\n"
    "3a4. WHICH ENRICHMENT. `condition_enrichment` needs a case/control split "
    "and ranks the genes that separate them, so it is the pathway question. "
    "`region_enrichment`, `label_value_enrichment` and `pooled_enrichment` "
    "test LABELS, not pathways - do not reach for them when asked what a set "
    "of genes does.\n"
    "3a5b. SELECTING vs TESTING. `multi_label_query` SELECTS the samples that "
    "satisfy several label conditions at once (Tissue=Liver AND Sex=female) "
    "and reports how many there are and from how many studies. It answers "
    "'how many' and 'which'. It does NOT test whether anything is enriched: "
    "'what is this region enriched for' is `region_enrichment`, and 'what "
    "else describes these samples' is `label_value_enrichment`. A count is "
    "not evidence of association, so never present a query's percentage as a "
    "lift.\n"
    "3a5c. PREPARING DATA. `normalize_platform` corrects a RAW matrix on disk "
    "and writes the normalized file beside it - TMM/CPM/log2 for counts, "
    "log2 + quantile for intensities - and it does NOT rescale a platform "
    "that is already loaded in this session. Only reach for it when the user "
    "asks for normalization of raw data; loading a platform already picks up "
    "the normalized file when one exists.\n"
    "3a6. CELLS vs SAMPLES. `load_single_cell_file` opens a local .h5ad and "
    "`fetch_single_cell` queries the CELLxGENE Census over the network. Both "
    "enter the session the same way every other source does: as a pseudo-bulk "
    "platform of group means, one row per group. So a cell-level question - "
    "'how many cells', 'which cells express this' - cannot be answered off "
    "it, because a row is a group, not a cell. Say that instead of counting "
    "the rows. Cells are also not independent replicates, so a group abundant "
    "in one donor is not thereby general.\n"
    "4. When comparing single-cell and GEO for a gene, first load/fetch BOTH "
    "sources, then call `compare_gene` (or `cross_modality_gene` for a "
    "harmonised SC-vs-bulk view) with that gene and both platform names.\n"
    "5. WHEN NO TOOL FITS, DO NOT GUESS. Every tool you have is a function the "
    "program already runs behind a button, so an answer you cannot reach with "
    "one is an answer the program does not compute. Say which part is missing "
    "rather than inventing a number or declining outright.\n"
    "6. To PLOT or when the user asks for a chart/figure/'the plot', call a "
    "figure-producing tool - `gene_distribution` (single-gene histogram), "
    "`compare_gene` (overlaid distributions) or an enrichment tool. Those "
    "return the same figure the window draws.\n"
    "6a. TO SAVE OR EXPORT what you have already produced, call "
    "`export_results` - never re-run the analysis to 'regenerate' a figure. It "
    "writes the exact figures and tables of this session through the program's "
    "own export writer (300 dpi PNG plus vector PDF, CSV, and the report with "
    "its manifest), so an assistant export and a window export are the same "
    "file. Pass `which='all'` for the whole session or a tool name for one "
    "result; default is the most recent.\n"
    "7. EXPRESSION RANGES:stored values are on a log2/quantile scale (roughly "
    "0.5-50), NOT 0-1. So when a user gives a fractional range like 'between 0.9 "
    "and 1.0' they mean a RELATIVE band of the gene's OWN values - 90-100% of "
    "that gene's max expression (equivalently the top decile) - NOT literal "
    "0.9-1.0 raw values (which would select nothing). REASON about it this way: "
    "compute the gene's max (or quantiles) and translate the fraction to real "
    "thresholds (0.9×max … max) before filtering/plotting. State the concrete "
    "expression thresholds you used in the answer.\n\n"
    "Rules: call one tool at a time, read its result, then decide the next step. "
    "Prefer real platform names returned by earlier tool calls. When you have "
    "enough information, STOP calling tools and write a short, concrete summary "
    "of the findings (distribution class, key statistics, whether sources differ "
    "and how). Keep the final answer factual and grounded in tool outputs. "
    "Never end a turn without either calling a tool or writing a summary of a "
    "tool result you already received. Do NOT ask the user to clarify a metadata "
    "column, case/control labels, or gene-set libraries - every tool auto-selects "
    "sensible defaults, so just call the tool. Never reload a platform that "
    "`list_platforms` already shows as loaded."
)


# ---- registry Tool -> LangChain StructuredTool ----------------------
_PYTYPE = {
    "str": str,
    "platform": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": List[str],
}


#: Parameter spellings the registry uses interchangeably across tools. Nine
#: tools take ``gene`` and seven take ``genes``, which no caller can be
#: expected to keep straight - and a model that guesses wrong does not fail
#: once and move on. The rejected call comes back as a validation error, the
#: model re-emits the same arguments, and the turn burns its entire step
#: budget: three whole analyses were lost this way before the loop was logged
#: and the mismatched spelling showed up in it.
_PARAM_ALIASES = {"genes": "gene", "gene": "genes"}


def _apply_aliases(tool: Tool, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Fold an accepted alias onto the spelling this tool actually declares."""
    declared = {p.name for p in tool.params}
    out = dict(kwargs)
    for canonical, alias in _PARAM_ALIASES.items():
        if canonical in declared and alias not in declared:
            supplied = out.pop(alias, None)
            if supplied not in (None, "") and out.get(canonical) in (None, ""):
                out[canonical] = supplied
    return out


def _args_schema(tool: Tool):
    """Build a pydantic model describing a tool's parameters."""
    fields: Dict[str, Any] = {}
    declared = {p.name for p in tool.params}
    for p in tool.params:
        pytype = _PYTYPE.get(p.type, str)
        # A required parameter that has an accepted alias is declared optional
        # here on purpose. Pydantic validates before the alias is folded in, so
        # a model that wrote `gene` for `genes` was rejected with "Field
        # required", rewrote the same call, and span to the recursion limit.
        # A genuinely missing value still fails, in the executor, with a
        # sentence that names the tool and what it needs.
        if p.required and not _PARAM_ALIASES.get(p.name):
            fields[p.name] = (pytype, Field(description=p.help or p.name))
        elif p.required:
            fields[p.name] = (
                Optional[pytype],
                Field(default=None,
                      description=(p.help or p.name)
                      + f" (may also be given as `{_PARAM_ALIASES[p.name]}`)"),
            )
        else:
            fields[p.name] = (
                Optional[pytype],
                Field(default=p.default, description=p.help or p.name),
            )
        alias = _PARAM_ALIASES.get(p.name)
        if alias and alias not in declared and alias not in fields:
            fields[alias] = (
                Optional[pytype],
                Field(default=None,
                      description=f"Alias for `{p.name}`; either spelling is "
                                  f"accepted."),
            )
    if not fields:
        return create_model(f"{tool.name}_Args")
    return create_model(f"{tool.name}_Args", **fields)


def _observation(result: ToolResult) -> str:
    """Render a ToolResult as a compact text observation for the LLM."""
    lines = [result.summary]
    tbl = getattr(result, "table", None)
    if tbl is not None:
        try:
            preview = tbl.head(10).to_string(index=False, max_cols=8)
            lines.append("Result table (head):\n" + preview)
        except Exception:
            pass
    report = getattr(result, "report", "")
    if report:
        lines.append("Analysis:\n" + report[:1500])
    if not result.ok:
        lines.append("(this step did not succeed - adjust and try another tool)")
    return "\n".join(lines)


# Llama-3.x sometimes emits a tool call in its *native* text format
# (``<function=name>{json}</function>`` or ``<function/name>{json}</function>``,
# occasionally wrapped in a ``<|python_tag|>``) inside the assistant content
# instead of routing it through the structured tool-calls API. Groq/LangChain
# then hand it back as a plain final answer and the tool never runs - most often
# for the tool with the most parameters. We detect and execute that leaked call
# so the reasoning loop still produces a real result.
_LEAKED_TOOL_RE = re.compile(
    r"<function[/=]\s*([A-Za-z0-9_]+)\s*>\s*(\{.*?\})\s*</function>",
    re.DOTALL,
)


def _find_leaked_tool_call(text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Return ``(tool_name, params)`` if ``text`` is a leaked native tool call."""
    if not text or "<function" not in text:
        return None
    m = _LEAKED_TOOL_RE.search(text)
    if not m:
        return None
    name = m.group(1)
    try:
        params = json.loads(m.group(2))
    except (ValueError, TypeError):
        return None
    if not isinstance(params, dict):
        return None
    return name, params


def build_langchain_tools(
    app,
    registry: Dict[str, Tool],
    on_event: Callable[[str, str, Optional[ToolResult]], None],
    progress_cb: Optional[Callable[[float, str], None]],
    sink: Dict[str, List[ToolResult]],
) -> List[Any]:
    """Wrap each registry :class:`Tool` as a LangChain ``StructuredTool``."""
    _progress = progress_cb or (lambda v, t: None)

    def _make(tool: Tool):
        def _run(**kwargs: Any) -> str:
            raw = {k: v for k, v in kwargs.items() if v not in (None, "")}
            raw = _apply_aliases(tool, raw)
            on_event("tool_start", f"{tool.name}({raw})", None)
            try:
                resolved = tool.resolver(app, raw)
                resolved = tool.coerce(resolved)
                result = tool.run(app, resolved, _progress)
            except Exception as exc:  # keep the loop alive; tell the model
                on_event("tool_error", f"{tool.name} failed: {exc}", None)
                return f"ERROR from {tool.name}: {exc}"
            sink["results"].append(result)
            on_event("tool_result", result.summary, result)
            return _observation(result)

        return StructuredTool.from_function(
            func=_run,
            name=tool.name,
            description=tool.description,
            args_schema=_args_schema(tool),
        )

    return [_make(t) for t in registry.values()]


# ---- chat-model construction per backend ----------------------------
# ---- local-inference (GGUF) optimisation ----------------------------
# The local backend runs a *quantized* GGUF model through Ollama/llama.cpp.
# Quantization (GGUF Q4_K_M by default; also AWQ/GPTQ for GPU-only runtimes)
# shrinks the 7B model to ~4-5 GB so it fits the user's ~8 GB VRAM and roughly
# doubles token throughput versus fp16 with a negligible tool-calling-accuracy
# hit. ``keep_alive`` holds the model resident between turns so we pay the
# multi-second load only once - the single biggest latency win for an
# interactive agent. Groq is a hosted LPU service and is already optimally
# served, so no client-side tuning applies there.
_DEFAULT_QUANT = "q4_K_M"


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, "")).strip() or default)
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _resolve_ollama_model(model: str) -> str:
    """Append the requested GGUF quantization to a bare Ollama tag.

    ``GENEVARIATE_AGENT_QUANT`` (default ``q4_K_M``) selects the GGUF level,
    e.g. ``q4_K_M`` (smallest/fastest), ``q5_K_M`` (balanced), ``q8_0``
    (highest fidelity). A model tag that already pins a quant is left as-is.
    """
    quant = (os.environ.get("GENEVARIATE_AGENT_QUANT") or "").strip()
    if not quant or ":" not in model:
        return model
    tag = model.rsplit(":", 1)[-1].lower()
    if "q" in tag and any(ch.isdigit() for ch in tag) and "_" in tag:
        return model  # tag already carries an explicit quant
    return f"{model}-{quant}"


def _ollama_options() -> Dict[str, Any]:
    """Latency/throughput knobs for the local GGUF model (all env-overridable).

    ``reasoning`` is off because the default local model is a thinking model.
    With thinking on it spends the whole generation budget on reasoning tokens
    and the turn ends with no ``content`` and no ``tool_calls`` at all, which
    the loop can only read as an empty turn: the run then ends with "the agent
    finished without a textual answer" and no analysis, because the plan the
    model is reasoning its way towards is never emitted. A tool call is a
    structured decision
    rather than a chain of thought, so nothing is lost by asking for it
    directly.

    ``num_ctx`` has to hold the system prompt and every tool schema - about
    6,000 tokens before the conversation starts - plus one observation per tool
    the agent runs. At 8,192 the second tool result pushes the tool definitions
    out of the window and the agent forgets what it can call.
    """
    return {
        "temperature": 0,
        "reasoning": _env_bool("GENEVARIATE_AGENT_REASONING", False),
        "keep_alive": os.environ.get("GENEVARIATE_AGENT_KEEP_ALIVE", "30m"),
        "num_ctx": _env_int("GENEVARIATE_AGENT_NUM_CTX", 32768),
        "num_predict": _env_int("GENEVARIATE_AGENT_NUM_PREDICT", 2048),
    }


_DEBUG_LOOP = _env_bool("GENEVARIATE_AGENT_DEBUG_LOOP", False)


def _log_step(node, msg) -> None:
    """Print one raw reasoning step to stderr (GENEVARIATE_AGENT_DEBUG_LOOP)."""
    kind = type(msg).__name__
    calls = getattr(msg, "tool_calls", None) or []
    text = (_message_text(msg) or "").replace("\n", " ")
    if calls:
        what = "; ".join(f"{c.get('name')}({c.get('args')})" for c in calls)
    elif kind == "ToolMessage":
        what = f"-> {str(getattr(msg, 'content', ''))[:200]}"
    else:
        what = f"text[{len(text)}]: {text[:200]}"
    print(f"[loop] {node}/{kind}: {what}", file=sys.stderr, flush=True)


def _context_middleware():
    """Drop stale tool observations before they crowd out the tool schemas.

    A long session fails in a way that looks nothing like running out of room:
    the model simply stops emitting usable tool calls and the graph spins until
    it hits the recursion limit. What has actually happened is that the
    accumulated observations pushed the tool definitions towards the edge of
    ``num_ctx``. It shows up first on the tool with the largest schema and the
    longest report - ``region_enrichment`` looped at the eighth turn of a
    session while a smaller tool, asked immediately afterwards with the
    same history, still worked.

    Clearing the older tool results keeps the conversation itself intact: the
    assistant's own answers stay, so a closing "summarise what you found" turn
    can still attribute every number, and the full result was never in the
    conversation to begin with - it is in the window and in the export.

    Trigger is a fraction of the window rather than the library default of
    100,000 tokens, which a 32,768-token local model can never reach.
    """
    try:
        from langchain.agents.middleware import (ClearToolUsesEdit,
                                                 ContextEditingMiddleware)
    except ImportError:
        return []
    opts = _ollama_options()
    num_ctx = int(opts.get("num_ctx") or 32768)
    reserve = int(opts.get("num_predict") or 2048) + 6000  # answer + schemas
    trigger = _env_int("GENEVARIATE_AGENT_CLEAR_AT",
                       max(4096, num_ctx - reserve))
    keep = _env_int("GENEVARIATE_AGENT_KEEP_TOOL_RESULTS", 3)
    return [ContextEditingMiddleware(
        edits=[ClearToolUsesEdit(trigger=trigger, keep=keep)],
    )]


def _build_llm(model: str, backend: str):
    if backend == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=0)
    if backend == "ollama":
        from langchain_ollama import ChatOllama
        # ``OLLAMA_URL`` is the program's setting for where Ollama is. Left to
        # itself ChatOllama goes to localhost:11434 regardless, so a user who
        # points the program at another server gets told the backend is ready -
        # the probe reached it - and then the agent hangs against a port with
        # nothing on it. The probe and the agent have to address the same
        # server or "ready" means nothing.
        from genevariate.core import llm_client
        return ChatOllama(model=_resolve_ollama_model(model),
                          base_url=llm_client.ollama_url(),
                          **_ollama_options())
    # OpenAI-compatible: OpenRouter / Gemini-openai / Cerebras / NIM / ...
    from langchain_openai import ChatOpenAI
    base = os.environ.get("GENEVARIATE_AGENT_BASE_URL") or None
    key = (os.environ.get("GENEVARIATE_AGENT_API_KEY")
           or os.environ.get("OPENAI_API_KEY"))
    return ChatOpenAI(model=model, temperature=0, base_url=base, api_key=key)


# ---- availability ---------------------------------------------------
def _backend_ready(backend: str, model: str) -> Tuple[bool, str]:
    if backend == "groq":
        if not _module_present("langchain_groq"):
            return False, "the Groq client (langchain-groq) is not installed yet."
        if not os.environ.get("GROQ_API_KEY"):
            return False, ("no Groq API key set - get a free one at "
                           "console.groq.com/keys.")
        return True, ""
    if backend == "ollama":
        if not _module_present("langchain_ollama"):
            return False, "the Ollama client (langchain-ollama) is not installed yet."
        try:
            from genevariate.core import ollama_manager as om
            if not om.ollama_server_ok():
                return False, "no Ollama server is running yet."
            tag = _resolve_ollama_model(model)
            if not om.model_available(tag):
                return False, f"model {tag!r} has not been pulled yet."
        except Exception as exc:
            return False, f"could not verify the Ollama backend: {exc}"
        return True, ""
    if not _module_present("langchain_openai"):
        return False, "the OpenAI-compatible client (langchain-openai) is not installed yet."
    if not (os.environ.get("GENEVARIATE_AGENT_API_KEY")
            or os.environ.get("OPENAI_API_KEY")):
        return False, "no API key set (GENEVARIATE_AGENT_API_KEY)."
    return True, ""


def agent_available(model: Optional[str] = None) -> bool:
    """True only if LangChain and the selected backend are both ready."""
    if not _HAS_LANGCHAIN:
        return False
    backend = _backend()
    ok, _ = _backend_ready(backend, model or _default_model(backend))
    return ok


def unavailable_reason(model: Optional[str] = None) -> str:
    """Human-readable explanation of why the LangChain agent can't run."""
    if not _HAS_LANGCHAIN:
        return ("the reasoning stack (langchain) is not installed yet. "
                + _IMPORT_ERROR)
    backend = _backend()
    ok, why = _backend_ready(backend, model or _default_model(backend))
    return "" if ok else why


# ---- on-demand provisioning (auto-install, no manual steps) ---------
def _pip_install(pkgs: List[str], log: Callable[[str], None]) -> bool:
    """pip-install ``pkgs`` into the running interpreter's environment."""
    import sys
    import subprocess

    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    base = [sys.executable, "-m", "pip", "install", "--upgrade"]
    attempts: List[List[str]]
    if in_venv:
        attempts = [base + list(pkgs)]
    else:
        attempts = [
            base + ["--user"] + list(pkgs),
            base + ["--user", "--break-system-packages"] + list(pkgs),
        ]
    for cmd in attempts:
        log("$ " + " ".join(cmd))
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    log(line)
            proc.wait()
            if proc.returncode == 0:
                return True
        except Exception as exc:  # try the next flag set
            log(f"pip attempt failed: {exc}")
    return False


def ensure_agent_ready(
    log: Optional[Callable[[str], None]] = None,
    *,
    model: Optional[str] = None,
    backend: Optional[str] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Tuple[bool, str]:
    """Provision everything the selected backend needs, on first use.

    Installs the base LangChain stack + the backend client automatically. For
    the local ``ollama`` backend it also installs/starts Ollama and pulls the
    model. Hosted backends still need a free API key (:func:`api_key_prompt`
    surfaces that to the UI). Returns ``(ok, message)`` and never raises.
    """
    log = log or (lambda *_: None)
    stopped = (lambda: bool(should_stop and should_stop()))
    backend = backend or _backend()
    model = model or _default_model(backend)

    # 1) base reasoning stack ----------------------------------------
    if not _HAS_LANGCHAIN and not _try_enable():
        log("Installing the reasoning stack (langchain)…")
        if not _pip_install(["langchain>=1.0"], log):
            return False, ("Could not auto-install langchain. Install it once "
                           "with `pip install genevariate[agent]`.")
        if not _try_enable():
            return False, ("Installed the reasoning stack - please restart "
                           "GeneVariate to finish enabling the agent.")
        log("Reasoning stack ready.")
    if stopped():
        return False, "Setup cancelled."

    # 2) backend client + resources ----------------------------------
    if backend == "groq":
        if not _module_present("langchain_groq"):
            log("Installing the Groq client (langchain-groq)…")
            if not _pip_install(["langchain-groq>=0.2"], log):
                return False, "Could not auto-install langchain-groq."
        if not os.environ.get("GROQ_API_KEY"):
            return False, ("Add a free Groq API key (console.groq.com/keys) to "
                           "use the hosted agent.")
        return True, f"Agent ready - Groq {model}."

    if backend == "ollama":
        if not _module_present("langchain_ollama"):
            log("Installing the Ollama client (langchain-ollama)…")
            if not _pip_install(["langchain-ollama>=0.3"], log):
                return False, "Could not auto-install langchain-ollama."
        try:
            from genevariate.core import ollama_manager as om
        except Exception as exc:
            return False, f"Ollama manager unavailable: {exc}"
        if not om.ollama_binary_exists():
            log("Installing Ollama…")
            try:
                om.install_ollama_blocking(log)
            except Exception as exc:
                return False, f"Ollama install failed: {exc}"
        if not om.ollama_server_ok():
            log("Starting the Ollama server…")
            try:
                om.start_ollama_server_blocking(log)
            except Exception as exc:
                return False, f"Could not start Ollama: {exc}"
        if stopped():
            return False, "Setup cancelled."
        # Pull the same quantized GGUF tag that _build_llm will load.
        tag = _resolve_ollama_model(model)
        if not om.model_available(tag):
            log(f"Pulling model {tag} (first time only; several GB)…")
            try:
                om.pull_model_blocking(tag, log)
            except Exception as exc:
                return False, f"Model pull failed: {exc}"
        if not om.model_available(tag):
            return False, f"Model {tag!r} is still unavailable after pulling."
        return True, f"Agent ready - local {tag}."

    # OpenAI-compatible backend
    if not _module_present("langchain_openai"):
        log("Installing the OpenAI-compatible client (langchain-openai)…")
        if not _pip_install(["langchain-openai>=0.2"], log):
            return False, "Could not auto-install langchain-openai."
    if not (os.environ.get("GENEVARIATE_AGENT_API_KEY")
            or os.environ.get("OPENAI_API_KEY")):
        return False, "Set GENEVARIATE_AGENT_API_KEY for the hosted agent."
    return True, f"Agent ready - {backend} {model}."


# ---- run ------------------------------------------------------------
@dataclass
class AgentReply:
    """Outcome of a full reasoning run for the sidebar to render."""
    goal: str
    summary: str = ""
    results: List[ToolResult] = field(default_factory=list)
    stopped: bool = False
    ok: bool = True
    source: str = "langchain"


def _message_text(msg: Any) -> str:
    content = getattr(msg, "content", "")
    if isinstance(content, list):  # some backends return content blocks
        parts = []
        for blk in content:
            if isinstance(blk, dict):
                parts.append(str(blk.get("text", "")))
            else:
                parts.append(str(blk))
        return " ".join(p for p in parts if p).strip()
    return str(content).strip()


def run_agent(
    app,
    registry: Dict[str, Tool],
    goal: str,
    on_event: Callable[[str, str, Optional[ToolResult]], None],
    *,
    model: Optional[str] = None,
    backend: Optional[str] = None,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    max_steps: int = 12,
    thread_id: Optional[str] = None,
) -> AgentReply:
    """Run the LangChain reasoning loop against the live tool registry.

    ``on_event(kind, text, result)`` narrates with ``kind`` in
    ``{"start","thought","tool_start","tool_result","tool_error","final"}``.
    Returns an :class:`AgentReply`; never raises.

    Passing ``thread_id`` continues an existing conversation: the earlier turns
    and their tool observations are replayed into the model, so a follow-up can
    say "now split it by that" instead of restating the whole question. Omit it
    for a one-shot run with no history.
    """
    reply = AgentReply(goal=goal)
    if not _HAS_LANGCHAIN:
        reply.ok = False
        reply.summary = unavailable_reason(model)
        return reply

    backend = backend or _backend()
    model_name = model or _default_model(backend)
    sink: Dict[str, List[ToolResult]] = {"results": []}
    on_event("start", f"Reasoning about: {goal}", None)

    try:
        lc_tools = build_langchain_tools(app, registry, on_event, progress_cb, sink)
        llm = _build_llm(model_name, backend)
        from genevariate.core.chatbot.registry import session_state_text
        graph = create_agent(model=llm, tools=lc_tools,
                             system_prompt=SYSTEM_PROMPT + session_state_text(app),
                             middleware=_context_middleware(),
                             checkpointer=_memory() if thread_id else None)
    except Exception as exc:
        reply.ok = False
        reply.summary = f"Could not start the {backend} agent: {exc}"
        on_event("tool_error", reply.summary, None)
        return reply

    final_text = ""
    try:
        config: Dict[str, Any] = {"recursion_limit": max(4, max_steps * 2)}
        if thread_id:
            config["configurable"] = {"thread_id": str(thread_id)}
        stream = graph.stream(
            {"messages": [("user", goal)]}, config, stream_mode="updates",
        )
        for chunk in stream:
            if should_stop and should_stop():
                reply.stopped = True
                on_event("final", "Stopped by user.", None)
                break
            for _node, update in (chunk or {}).items():
                msgs = (update or {}).get("messages", []) \
                    if isinstance(update, dict) else []
                for msg in msgs:
                    if _DEBUG_LOOP:
                        # A turn that ends on the recursion limit shows nothing
                        # in the transcript: every step is an AIMessage the UI
                        # has no reason to render, so the loop looks idle from
                        # outside. Printing the raw step is the only way to see
                        # whether the model is emitting no calls, malformed
                        # calls, or the same call over and over.
                        _log_step(_node, msg)
                    if type(msg).__name__ != "AIMessage":
                        continue
                    text = _message_text(msg)
                    tool_calls = getattr(msg, "tool_calls", None) or []
                    if text:
                        if tool_calls:
                            on_event("thought", text, None)
                        else:
                            final_text = text
    except Exception as exc:
        reply.ok = False
        reply.summary = f"Agent run failed: {exc}"
        reply.results = sink["results"]
        on_event("tool_error", reply.summary, None)
        return reply

    reply.results = sink["results"]

    # Recover a tool call the model leaked as text instead of calling properly.
    leaked = _find_leaked_tool_call(final_text)
    if leaked and leaked[0] in registry:
        name, params = leaked
        tool = registry[name]
        raw = {k: v for k, v in params.items() if v not in (None, "")}
        on_event("tool_start", f"{name}({raw})", None)
        try:
            resolved = tool.coerce(tool.resolver(app, raw))
            result = tool.run(app, resolved, progress_cb)
            reply.results.append(result)
            on_event("tool_result", result.summary, result)
            final_text = _observation(result)
        except Exception as exc:
            on_event("tool_error", f"{name} failed: {exc}", None)
            final_text = f"ERROR from {name}: {exc}"

    # There is deliberately no fallback here. When the loop ends without a
    # successful analysis - a tool error, or the model asking which gene the
    # user meant - that text is the answer. The keyword router that used to
    # stand in ran a tool the model had declined to run, on parameters the user
    # never supplied, and its output was indistinguishable from the analysis
    # that was actually asked for.
    if not final_text:
        oks = [r for r in reply.results if r.ok]
        final_text = (oks[-1].summary if oks else
                      "The agent finished without a textual answer.")
    reply.summary = final_text
    if not reply.stopped:
        on_event("final", final_text, None)
    return reply
