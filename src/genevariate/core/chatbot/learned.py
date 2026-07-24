"""
Self-extending "learned tools" for the GeneVariate assistant.

The built-in registry (``registry.build_registry``) is static. When the agent
solves something novel with the sandboxed ``run_analysis_code`` tool, it can
*promote* that working snippet into a **named, persisted tool** via the
``save_learned_tool`` registry tool. Learned tools are stored as JSON on disk
and re-loaded on every ``build_registry`` call, so they appear to the LangChain
agent exactly like the built-ins on the next run and survive across sessions.

Safety: a learned tool is nothing more than a *named* ``run_analysis_code``
snippet — it runs through the SAME AST-validated, import-blocked, timed sandbox
(:mod:`genevariate.core.chatbot.code_exec`). The code is validated once at save
time (so unsafe snippets can never be persisted) and again at every execution.
Nothing here imports Tkinter.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from .code_exec import CodeValidationError, _validate, run_user_code
from .tools import Tool, ToolParam, ToolResult

_ALLOWED_PARAM_TYPES = ("str", "int", "float", "bool", "list", "platform")
_NAME_RE = re.compile(r"[^a-z0-9_]+")


def _learned_dir(app) -> str:
    """Directory holding the persisted learned-tool JSON files."""
    base = getattr(app, "data_dir", None) or os.getcwd()
    return os.path.join(str(base), "chatbot_learned")


def _sanitize_name(name: str) -> str:
    n = _NAME_RE.sub("_", str(name or "").strip().lower()).strip("_")
    return n or "learned_tool"


def _normalize_params(params: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in params or []:
        if not isinstance(p, dict) or not p.get("name"):
            continue
        t = str(p.get("type", "str")).lower()
        out.append({
            "name": _sanitize_name(p["name"]),
            "type": t if t in _ALLOWED_PARAM_TYPES else "str",
            "required": bool(p.get("required", False)),
            "default": p.get("default"),
            "help": str(p.get("help", "")),
        })
    return out


def save_learned_tool_record(app, name: str, description: str, code: str,
                             params: Optional[List[Dict[str, Any]]] = None,
                             examples: Optional[List[str]] = None) -> str:
    """Validate and persist a learned tool. Returns the JSON file path.

    Raises ``CodeValidationError`` if the snippet fails the sandbox AST checks or
    the metadata is incomplete, so unsafe/blank tools are never written.
    """
    name = _sanitize_name(name)
    code = str(code or "").strip()
    if not code:
        raise CodeValidationError("a learned tool needs a non-empty code snippet")
    if "result" not in code:
        raise CodeValidationError(
            "the snippet must assign its answer to `result`")
    _validate(code)  # re-runs the same guards as run_analysis_code
    record = {
        "name": name,
        "description": str(description or "").strip() or f"Learned tool '{name}'.",
        "code": code,
        "params": _normalize_params(params),
        "examples": [str(e) for e in (examples or []) if str(e).strip()],
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    d = _learned_dir(app)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{name}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    return path


def load_learned_records(app) -> List[Dict[str, Any]]:
    """Read every persisted learned-tool record. Never raises."""
    d = _learned_dir(app)
    records: List[Dict[str, Any]] = []
    try:
        names = sorted(f for f in os.listdir(d) if f.endswith(".json"))
    except OSError:
        return records
    for fn in names:
        try:
            with open(os.path.join(d, fn), encoding="utf-8") as fh:
                rec = json.load(fh)
            if isinstance(rec, dict) and rec.get("name") and rec.get("code"):
                records.append(rec)
        except Exception:
            continue  # skip corrupt files, keep the rest
    return records


def delete_learned_tool(app, name: str) -> bool:
    """Remove a persisted learned tool by name. Returns True if a file was deleted."""
    path = os.path.join(_learned_dir(app), f"{_sanitize_name(name)}.json")
    try:
        os.remove(path)
        return True
    except OSError:
        return False


def _make_learned_tool(rec: Dict[str, Any]) -> Tool:
    """Wrap a stored record as an executable :class:`Tool`."""
    name = rec["name"]
    code = rec["code"]
    param_specs = _normalize_params(rec.get("params"))
    tool_params = [
        ToolParam(p["name"], p["type"], required=p["required"],
                  default=p["default"], help=p["help"])
        for p in param_specs
    ]
    platform_params = [p["name"] for p in param_specs if p["type"] == "platform"]

    def _resolver(app, raw, _params=tool_params, _plats=platform_params):
        # local import to avoid a registry <-> learned import cycle
        from .registry import _match_platform
        resolved: Dict[str, Any] = {}
        by_name = {p.name: p for p in _params}
        for p in _params:
            if p.name in raw and raw[p.name] not in (None, ""):
                resolved[p.name] = raw[p.name]
            elif p.default is not None:
                resolved[p.name] = p.default
        for key in _plats:
            resolved[key] = _match_platform(app, resolved.get(key))
        return resolved

    def _executor(app, resolved, progress_cb, _code=code, _name=name):
        from .registry import _platforms
        progress_cb(30.0, f"Running learned tool '{_name}' (sandboxed)…")
        res = run_user_code(_code, _platforms(app), timeout=30.0,
                            params=dict(resolved))
        if not res["ok"]:
            return ToolResult(f"Learned tool '{_name}' failed: {res['error']}",
                              ok=False, payload={"stdout": res.get("stdout", "")})
        stdout = res.get("stdout") or ""
        result = res.get("result")
        parts = []
        if stdout.strip():
            parts.append("output:\n" + stdout.strip())
        if result is not None and not hasattr(result, "columns") \
                and not hasattr(result, "index"):
            parts.append(f"result = {result!r}")
        summary = f"Learned tool '{_name}' ran." + (
            " " + " | ".join(parts) if parts else "")
        report = f"# Learned tool `{_name}`\n\n```python\n{_code}\n```"
        return ToolResult(summary[:2000], table=res.get("result_table"),
                          report=report,
                          payload={"stdout": stdout, "result": result,
                                   "learned": True})

    desc = rec.get("description") or f"Learned tool '{name}'."
    return Tool(
        name=name,
        description="[learned] " + desc,
        params=tool_params,
        resolver=_resolver,
        executor=_executor,
        examples=tuple(rec.get("examples") or ()),
    )


def build_learned_tools(app) -> Dict[str, Tool]:
    """Load every persisted learned tool as a ``{name: Tool}`` map. Never raises."""
    out: Dict[str, Tool] = {}
    for rec in load_learned_records(app):
        try:
            _validate(rec["code"])  # skip any record that no longer validates
            out[rec["name"]] = _make_learned_tool(rec)
        except Exception:
            continue
    return out
