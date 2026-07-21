"""
LangChain-powered reasoning agent for the GeneVariate assistant.

Unlike :mod:`router` (single-tool keyword/JSON routing) and :mod:`agent`
(deterministic heuristic planner), this module wires the *existing* registry
tools into a genuine tool-calling reasoning loop: the LLM reads the user's goal,
decides which analysis tools to call and in what order, observes each result,
and keeps going until it can answer. It is built on LangChain 1.x's unified
``create_agent`` (a LangGraph ReAct agent).

The chat backend is **pluggable** via ``GENEVARIATE_AGENT_BACKEND``:

* ``groq`` (default) — Groq's free hosted API (Llama-3.3-70B), the strongest
  and fastest tool-caller; needs a free ``GROQ_API_KEY`` (console.groq.com/keys).
* ``ollama`` — a fully local model (default ``qwen2.5:7b``), private + offline,
  auto-installed and auto-pulled on first use.
* anything else — an OpenAI-compatible endpoint (OpenRouter, Gemini-via-OpenAI,
  Cerebras, NVIDIA NIM) via ``GENEVARIATE_AGENT_BASE_URL`` +
  ``GENEVARIATE_AGENT_API_KEY``.

Because the agent drives GeneVariate's *Python API* (each tool wraps a real
analysis function), a reliable tool-caller matters far more than raw size — no
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
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .tools import Tool, ToolResult

# ---- base LangChain stack (import-guarded; backend clients are lazy) --
try:
    from langchain.agents import create_agent  # LangChain 1.x unified agent
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field, create_model
    _HAS_LANGCHAIN = True
    _IMPORT_ERROR = ""
except Exception as _exc:  # pragma: no cover - exercised only w/o extras
    create_agent = None  # type: ignore
    StructuredTool = None  # type: ignore
    BaseModel = object  # type: ignore
    Field = None  # type: ignore
    create_model = None  # type: ignore
    _HAS_LANGCHAIN = False
    _IMPORT_ERROR = repr(_exc)


# ---- backend + model configuration ----------------------------------
DEFAULT_BACKEND = "groq"
_BACKEND_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "ollama": "qwen2.5:7b",
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
    "gene-expression data by CALLING TOOLS — you never invent numbers.\n\n"
    "Workflow:\n"
    "1. Understand the user's goal (a gene, a platform, single-cell vs GEO, a "
    "comparison, an enrichment).\n"
    "2. REASON about WHICH dataset the user actually wants, then obtain it "
    "yourself before analysing. Choose the right source for the intent — it is "
    "NOT always a GEO platform:\n"
    "   • a GEO/GPL microarray id (e.g. GPL570) → `load_geo_platform` (it "
    "AUTO-DOWNLOADS from GEO when the platform isn't on disk);\n"
    "   • single-cell / scRNA-seq / a cell type / tissue / 'CELLxGENE' → "
    "`fetch_single_cell` (pulls the census and pseudo-bulks it);\n"
    "   • raw NGS / RNA-seq counts / a counts file → `run_ngs_de`.\n"
    "   Call `list_platforms` first to see what is already loaded. If the data "
    "the user described is NOT loaded, DECIDE ON YOUR OWN which acquisition tool "
    "fits and call it — infer the source from the wording, don't assume GEO. "
    "NEVER stop to ask the user to download data and NEVER tell them a file is "
    "missing; obtain it and continue. Only report a data problem if the tool "
    "itself returns an error you cannot work around.\n"
    "3. Run the analysis: `gene_distribution` to profile one gene on one "
    "platform; `compare_gene` to contrast a gene across two or more sources; "
    "`condition_enrichment` / `variability_enrichment` / `rank_genes` for "
    "case-vs-control ranking + GSEA; `run_ngs_de` for raw-count DESeq2.\n"
    "4. When comparing single-cell and GEO for a gene, first load/fetch BOTH "
    "sources, then call `compare_gene` with that gene and both platform names.\n"
    "5. For an open-ended computation with no dedicated tool, use "
    "`run_analysis_code` (sandboxed Python: read `platforms`/`params`, set "
    "`result`). If that snippet solves a task the user is likely to repeat, "
    "call `save_learned_tool` to PROMOTE it into a new persisted named tool — "
    "you can extend your own toolset this way.\n\n"
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
        lines.append("(this step did not succeed — adjust and try another tool)")
    return "\n".join(lines)


# Llama-3.x sometimes emits a tool call in its *native* text format
# (``<function=name>{json}</function>`` or ``<function/name>{json}</function>``,
# occasionally wrapped in a ``<|python_tag|>``) inside the assistant content
# instead of routing it through the structured tool-calls API. Groq/LangChain
# then hand it back as a plain final answer and the tool never runs — most often
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


# ---- chat-model construction per backend ----------------------------
# ---- local-inference (GGUF) optimisation ----------------------------
# The local backend runs a *quantized* GGUF model through Ollama/llama.cpp.
# Quantization (GGUF Q4_K_M by default; also AWQ/GPTQ for GPU-only runtimes)
# shrinks the 7B model to ~4–5 GB so it fits the user's ~8 GB VRAM and roughly
# doubles token throughput versus fp16 with a negligible tool-calling-accuracy
# hit. ``keep_alive`` holds the model resident between turns so we pay the
# multi-second load only once — the single biggest latency win for an
# interactive agent. Groq is a hosted LPU service and is already optimally
# served, so no client-side tuning applies there.
_DEFAULT_QUANT = "q4_K_M"


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, "")).strip() or default)
    except (TypeError, ValueError):
        return default


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
    """Latency/throughput knobs for the local GGUF model (all env-overridable)."""
    return {
        "temperature": 0,
        "keep_alive": os.environ.get("GENEVARIATE_AGENT_KEEP_ALIVE", "30m"),
        "num_ctx": _env_int("GENEVARIATE_AGENT_NUM_CTX", 8192),
        "num_predict": _env_int("GENEVARIATE_AGENT_NUM_PREDICT", 1024),
    }


def _build_llm(model: str, backend: str):
    if backend == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=0)
    if backend == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=_resolve_ollama_model(model),
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
            return False, ("no Groq API key set — get a free one at "
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
            return False, ("Installed the reasoning stack — please restart "
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
        return True, f"Agent ready — Groq {model}."

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
        return True, f"Agent ready — local {tag}."

    # OpenAI-compatible backend
    if not _module_present("langchain_openai"):
        log("Installing the OpenAI-compatible client (langchain-openai)…")
        if not _pip_install(["langchain-openai>=0.2"], log):
            return False, "Could not auto-install langchain-openai."
    if not (os.environ.get("GENEVARIATE_AGENT_API_KEY")
            or os.environ.get("OPENAI_API_KEY")):
        return False, "Set GENEVARIATE_AGENT_API_KEY for the hosted agent."
    return True, f"Agent ready — {backend} {model}."


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
) -> AgentReply:
    """Run the LangChain reasoning loop against the live tool registry.

    ``on_event(kind, text, result)`` narrates with ``kind`` in
    ``{"start","thought","tool_start","tool_result","tool_error","final"}``.
    Returns an :class:`AgentReply`; never raises.
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
        graph = create_agent(model=llm, tools=lc_tools, system_prompt=SYSTEM_PROMPT)
    except Exception as exc:
        reply.ok = False
        reply.summary = f"Could not start the {backend} agent: {exc}"
        on_event("tool_error", reply.summary, None)
        return reply

    final_text = ""
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
                msgs = (update or {}).get("messages", []) \
                    if isinstance(update, dict) else []
                for msg in msgs:
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
            result = tool.executor(app, resolved, progress_cb or (lambda v, t: None))
            reply.results.append(result)
            on_event("tool_result", result.summary, result)
            final_text = _observation(result)
        except Exception as exc:
            on_event("tool_error", f"{name} failed: {exc}", None)
            final_text = f"ERROR from {name}: {exc}"

    if not final_text:
        oks = [r for r in reply.results if r.ok]
        final_text = (oks[-1].summary if oks else
                      "The agent finished without a textual answer.")
    reply.summary = final_text
    if not reply.stopped:
        on_event("final", final_text, None)
    return reply
