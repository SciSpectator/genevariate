"""
Restricted Python execution for the GeneVariate assistant.

A fixed tool registry can only recombine pre-written analyses; giving the agent a
*sandboxed code cell* lets it answer open-ended questions against the loaded
platforms (the pattern used by Biomni, CellVoyager, K-Dense Analyst). This module
runs a snippet against the in-memory platform DataFrames behind several
guardrails:

  - **AST validation** rejects imports, dunder access, and a denylist of escape
    hatches (``open``, ``eval``, ``exec``, ``getattr`` …) *before* execution.
  - **Restricted builtins** — only a safe numeric/collection subset is exposed;
    no filesystem, network, or process primitives.
  - **A curated namespace** — ``pd``, ``np``, a copy of ``platforms`` and the
    read-only GeneVariate analysis functions; no ``app`` handle, so the snippet
    cannot mutate GUI state.
  - **A wall-clock timeout** run on a daemon thread.

It is deliberately conservative: this is a convenience for trusted local analysis,
not a hardened multi-tenant sandbox. Nothing here imports Tkinter.
"""
from __future__ import annotations

import ast
import contextlib
import io
import threading
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Names that must never be called or referenced in a snippet.
_DENY_NAMES = frozenset({
    "open", "eval", "exec", "compile", "__import__", "input", "exit", "quit",
    "globals", "locals", "vars", "getattr", "setattr", "delattr", "breakpoint",
    "memoryview", "help", "dir", "object", "type", "super", "classmethod",
    "staticmethod", "property", "os", "sys", "subprocess", "socket", "shutil",
    "importlib", "builtins", "__builtins__",
})

# The only builtins a snippet may use.
_SAFE_BUILTINS: Dict[str, Any] = {
    k: __builtins__[k] if isinstance(__builtins__, dict)  # type: ignore[index]
    else getattr(__builtins__, k)
    for k in (
        "abs", "min", "max", "sum", "len", "range", "enumerate", "zip",
        "sorted", "list", "dict", "set", "tuple", "float", "int", "str",
        "bool", "round", "print", "any", "all", "map", "filter", "reversed",
        "divmod", "pow", "isinstance", "frozenset", "repr", "format",
    )
}


class CodeValidationError(ValueError):
    """Raised when a snippet fails the AST safety checks."""


def _validate(code: str) -> None:
    """Reject unsafe constructs before execution. Raises CodeValidationError."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise CodeValidationError(f"syntax error: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise CodeValidationError("imports are not allowed")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise CodeValidationError(
                    f"access to dunder attribute {node.attr!r} is not allowed")
        if isinstance(node, ast.Name):
            if node.id in _DENY_NAMES or node.id.startswith("__"):
                raise CodeValidationError(f"use of {node.id!r} is not allowed")
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            raise CodeValidationError("global/nonlocal are not allowed")


def _build_namespace(platforms: Dict[str, pd.DataFrame],
                     params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The read-only environment a snippet runs against.

    ``params`` (used by *learned tools*) is exposed as a plain ``params`` dict so
    a saved snippet can read its typed inputs, e.g. ``params['gene']``.
    """
    from genevariate.core import analysis as A

    exposed = {}
    for name in getattr(A, "__all__", []):
        obj = getattr(A, name, None)
        if callable(obj):
            exposed[name] = obj

    safe_platforms = {k: v.copy() for k, v in (platforms or {}).items()}
    ns: Dict[str, Any] = {
        "__builtins__": _SAFE_BUILTINS,
        "pd": pd,
        "np": np,
        "platforms": safe_platforms,
        "params": dict(params or {}),
        "result": None,
    }
    ns.update(exposed)
    return ns


def run_user_code(code: str, platforms: Dict[str, pd.DataFrame],
                  timeout: float = 20.0,
                  params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute ``code`` against ``platforms`` behind the guardrails above.

    Returns a dict with ``ok``, ``stdout``, ``result`` (the snippet's ``result``
    variable), ``error`` and ``result_table`` (set when ``result`` is a
    DataFrame/Series). Never raises — validation and runtime errors are reported
    in the returned dict. ``params`` is exposed to the snippet as ``params``.
    """
    code = str(code or "").strip()
    if not code:
        return {"ok": False, "error": "empty code", "stdout": "", "result": None}
    try:
        _validate(code)
    except CodeValidationError as exc:
        return {"ok": False, "error": f"blocked: {exc}", "stdout": "",
                "result": None}

    ns = _build_namespace(platforms, params)
    buf = io.StringIO()
    box: Dict[str, Any] = {"error": None}

    def _target() -> None:
        try:
            with contextlib.redirect_stdout(buf):
                compiled = compile(code, "<assistant_snippet>", "exec")
                exec(compiled, ns, ns)  # noqa: S102 - guarded above
        except Exception as exc:  # surface runtime errors to the caller
            box["error"] = f"{type(exc).__name__}: {exc}"

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        return {"ok": False, "error": f"timed out after {timeout:.0f}s",
                "stdout": buf.getvalue(), "result": None}
    if box["error"]:
        return {"ok": False, "error": box["error"], "stdout": buf.getvalue(),
                "result": None}

    result = ns.get("result")
    out: Dict[str, Any] = {"ok": True, "error": None,
                           "stdout": buf.getvalue(), "result": result,
                           "result_table": None}
    if isinstance(result, pd.DataFrame):
        out["result_table"] = result.head(50)
    elif isinstance(result, pd.Series):
        out["result_table"] = result.head(50).to_frame()
    return out
