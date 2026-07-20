"""
LangChain-powered reasoning agent for the GeneVariate assistant.

Unlike :mod:`router` (single-tool keyword/JSON routing) and :mod:`agent`
(deterministic heuristic planner), this module wires the *existing* registry
tools into a genuine tool-calling reasoning loop: the LLM reads the user's goal,
decides which analysis tools to call and in what order, observes each result,
and keeps going until it can answer. It is built on LangChain 1.x's unified
``create_agent`` (a LangGraph ReAct agent) driving ``ChatOllama``.

Everything degrades gracefully: if ``langchain`` / ``langchain-ollama`` are not
installed, or no ollama server / model is available, :func:`agent_available`
returns ``False`` and callers fall back to the heuristic planner in
:mod:`agent`. Nothing here imports Tkinter; tool execution reuses the headless
executors from :mod:`registry`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .tools import Tool, ToolResult

# ---- optional LangChain stack (import-guarded) ----------------------
try:
    from langchain.agents import create_agent  # LangChain 1.x unified agent
    from langchain_ollama import ChatOllama
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field, create_model
    _HAS_LANGCHAIN = True
    _IMPORT_ERROR = ""
except Exception as _exc:  # pragma: no cover - exercised only w/o extras
    create_agent = None  # type: ignore
    ChatOllama = None  # type: ignore
    StructuredTool = None  # type: ignore
    BaseModel = object  # type: ignore
    Field = None  # type: ignore
    create_model = None  # type: ignore
    _HAS_LANGCHAIN = False
    _IMPORT_ERROR = repr(_exc)


DEFAULT_AGENT_MODEL = "llama3.1:8b"


def _default_model() -> str:
    """Model the agent drives, overridable via ``GENEVARIATE_AGENT_MODEL``.

    Small instruct models (e.g. gemma:2b) call tools unreliably, so the default
    is a mid-size model with solid native tool-calling. Pull it once with
    ``ollama pull llama3.1:8b``.
    """
    return os.environ.get("GENEVARIATE_AGENT_MODEL", DEFAULT_AGENT_MODEL).strip() \
        or DEFAULT_AGENT_MODEL


SYSTEM_PROMPT = (
    "You are GeneVariate's analysis agent. You help a bioinformatician analyse "
    "gene-expression data by CALLING TOOLS — you never invent numbers.\n\n"
    "Workflow:\n"
    "1. Understand the user's goal (a gene, a platform, single-cell vs GEO, a "
    "comparison, an enrichment).\n"
    "2. Make sure the required data is loaded BEFORE analysing it: use "
    "`load_geo_platform` for a GEO/GPL microarray id, or `fetch_single_cell` to "
    "pull and pseudo-bulk CELLxGENE single-cell data. Use `list_platforms` to "
    "see what is already loaded.\n"
    "3. Run the analysis: `gene_distribution` to profile one gene on one "
    "platform; `compare_gene` to contrast a gene across two or more sources; "
    "`condition_enrichment` / `variability_enrichment` / `rank_genes` for "
    "case-vs-control ranking + GSEA; `run_ngs_de` for raw-count DESeq2.\n"
    "4. When comparing single-cell and GEO for a gene, first load/fetch BOTH "
    "sources, then call `compare_gene` with that gene and both platform names.\n\n"
    "Rules: call one tool at a time, read its result, then decide the next step. "
    "Prefer real platform names returned by earlier tool calls. When you have "
    "enough information, STOP calling tools and write a short, concrete summary "
    "of the findings (distribution class, key statistics, whether sources differ "
    "and how). Keep the final answer factual and grounded in tool outputs."
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


def _args_schema(tool: Tool):
    """Build a pydantic model describing a tool's parameters."""
    fields: Dict[str, Any] = {}
    for p in tool.params:
        pytype = _PYTYPE.get(p.type, str)
        if p.required:
            fields[p.name] = (pytype, Field(description=p.help or p.name))
        else:
            fields[p.name] = (
                Optional[pytype],
                Field(default=p.default, description=p.help or p.name),
            )
    if not fields:
        # create_model needs at least an empty model; give it none.
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
    if not result.ok:
        lines.append("(this step did not succeed — adjust and try another tool)")
    return "\n".join(lines)


def build_langchain_tools(
    app,
    registry: Dict[str, Tool],
    on_event: Callable[[str, str, Optional[ToolResult]], None],
    progress_cb: Optional[Callable[[float, str], None]],
    sink: Dict[str, List[ToolResult]],
) -> List[Any]:
    """Wrap each registry :class:`Tool` as a LangChain ``StructuredTool``.

    The wrapper resolves + coerces params via the tool's own contract, runs the
    headless executor, records the :class:`ToolResult` in ``sink['results']``,
    narrates through ``on_event``, and returns a text observation to the model.
    """
    _progress = progress_cb or (lambda v, t: None)

    def _make(tool: Tool):
        def _run(**kwargs: Any) -> str:
            raw = {k: v for k, v in kwargs.items() if v not in (None, "")}
            on_event("tool_start", f"{tool.name}({raw})", None)
            try:
                resolved = tool.resolver(app, raw)
                resolved = tool.coerce(resolved)
                result = tool.executor(app, resolved, _progress)
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


# ---- availability ---------------------------------------------------
def agent_available(model: Optional[str] = None) -> bool:
    """True only if LangChain, an ollama server, and the model are all ready."""
    if not _HAS_LANGCHAIN:
        return False
    try:
        from genevariate.core import ollama_manager as om
        if not om.ollama_server_ok():
            return False
        return om.model_available(model or _default_model())
    except Exception:
        return False


def unavailable_reason(model: Optional[str] = None) -> str:
    """Human-readable explanation of why the LangChain agent can't run."""
    if not _HAS_LANGCHAIN:
        return ("LangChain is not installed — run "
                "`pip install genevariate[agent]`. " + _IMPORT_ERROR)
    try:
        from genevariate.core import ollama_manager as om
        if not om.ollama_server_ok():
            return "No ollama server is reachable (start it with `ollama serve`)."
        m = model or _default_model()
        if not om.model_available(m):
            return f"Model {m!r} is not available — run `ollama pull {m}`."
    except Exception as exc:
        return f"Could not verify the ollama backend: {exc}"
    return ""


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
    progress_cb: Optional[Callable[[float, str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    max_steps: int = 12,
) -> AgentReply:
    """Run the LangChain reasoning loop against the live tool registry.

    ``on_event(kind, text, result)`` is called for narration with ``kind`` in
    ``{"start","thought","tool_start","tool_result","tool_error","final"}``.
    Returns an :class:`AgentReply`. Never raises — errors surface as a failed
    reply so the caller can fall back to the heuristic planner.
    """
    reply = AgentReply(goal=goal)
    if not _HAS_LANGCHAIN:
        reply.ok = False
        reply.summary = unavailable_reason(model)
        return reply

    model_name = model or _default_model()
    sink: Dict[str, List[ToolResult]] = {"results": []}
    on_event("start", f"Reasoning about: {goal}", None)

    try:
        lc_tools = build_langchain_tools(app, registry, on_event, progress_cb, sink)
        llm = ChatOllama(model=model_name, temperature=0)
        graph = create_agent(model=llm, tools=lc_tools, system_prompt=SYSTEM_PROMPT)
    except Exception as exc:
        reply.ok = False
        reply.summary = f"Could not start the LangChain agent: {exc}"
        on_event("tool_error", reply.summary, None)
        return reply

    final_text = ""
    seen_ai = 0
    try:
        stream = graph.stream(
            {"messages": [("user", goal)]},
            {"recursion_limit": max(4, max_steps * 2)},
            stream_mode="updates",
        )
        for chunk in stream:
            if should_stop and should_stop():
                reply.stopped = True
                on_event("final", "Stopped by user.", None)
                break
            for _node, update in (chunk or {}).items():
                msgs = (update or {}).get("messages", []) if isinstance(update, dict) else []
                for msg in msgs:
                    cls = type(msg).__name__
                    if cls == "AIMessage":
                        seen_ai += 1
                        text = _message_text(msg)
                        tool_calls = getattr(msg, "tool_calls", None) or []
                        if text:
                            # intermediate reasoning vs the final answer
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
    if not final_text:
        # fall back to the last successful tool summary
        oks = [r for r in reply.results if r.ok]
        final_text = (oks[-1].summary if oks else
                      "The agent finished without a textual answer.")
    reply.summary = final_text
    if not reply.stopped:
        on_event("final", final_text, None)
    return reply
