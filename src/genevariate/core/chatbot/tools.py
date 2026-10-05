"""Tk-free tool abstraction for the GeneVariate conversational assistant."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence


# -----------------------------------------------------------------
# Parameter / result / action value objects
# -----------------------------------------------------------------
@dataclass
class ToolParam:
    """One typed input to a tool."""
    name: str
    type: str = "str"
    required: bool = True
    default: Any = None
    choices: Optional[Sequence[str]] = None
    help: str = ""


@dataclass
class ToolResult:
    """Headless outcome of a tool run, marshalled back to the sidebar."""
    summary: str
    table: Any = None          # optional pandas.DataFrame preview
    payload: Dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    report: str = ""           # optional markdown description/analysis
    manifest: Dict[str, Any] = field(default_factory=dict)  # reproducibility record
    figure: Any = None         # optional matplotlib Figure (Agg) for the results window
    figures: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Tool:
    name: str
    description: str
    params: List[ToolParam]
    resolver: Callable[[Any, Dict[str, Any]], Dict[str, Any]]
    executor: Callable[[Any, Dict[str, Any], Callable[[float, str], None]], ToolResult]
    examples: Sequence[str] = field(default_factory=tuple)
    main_thread: bool = False  # executor must run on the Tk main thread

    def run(self, app, resolved: Dict[str, Any],
            progress_cb: Optional[Callable[[float, str], None]] = None) -> ToolResult:
        """Execute the tool and record what it produced."""
        result = self.executor(app, resolved,
                               progress_cb or (lambda value, text: None))
        record(self.name, resolved, result)
        return result

    def coerce(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Type-coerce a raw param dict against this tool's declared params."""
        out: Dict[str, Any] = {}
        by_name = {p.name: p for p in self.params}
        for p in self.params:
            if p.name in raw and raw[p.name] not in (None, ""):
                out[p.name] = _coerce_value(raw[p.name], p.type)
            elif p.default is not None:
                out[p.name] = p.default
        # keep any extra recognised keys the resolver may consume
        for k, v in raw.items():
            if k not in by_name and v not in (None, ""):
                out[k] = v
        return out


@dataclass
class Recorded:
    """One completed tool run, kept so it can be exported or re-shown."""
    tool: str
    params: Dict[str, Any]
    result: ToolResult


# The figures and tables the assistant has produced this session, oldest first.
_HISTORY: List[Recorded] = []
_HISTORY_LIMIT = 40


def record(tool: str, params: Optional[Dict[str, Any]],
           result: ToolResult) -> None:
    _HISTORY.append(Recorded(tool, dict(params or {}), result))
    del _HISTORY[:-_HISTORY_LIMIT]


def history() -> List[Recorded]:
    """Every run so far, most recent last."""
    return list(_HISTORY)


@dataclass
class Action:
    """Router output: the chosen tool + proposed params (pre-resolution)."""
    tool: Optional[str]
    params: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    source: str = "llm"         # "llm" | "none"
    message: str = ""


# -----------------------------------------------------------------
# Coercion helper
# -----------------------------------------------------------------
def _coerce_value(value: Any, type_name: str) -> Any:
    try:
        if type_name == "int":
            return int(float(value))
        if type_name == "float":
            return float(value)
        if type_name == "bool":
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in ("1", "true", "yes", "y", "on")
        if type_name == "list":
            if isinstance(value, (list, tuple)):
                return [str(x).strip() for x in value if str(x).strip()]
            return [x.strip() for x in str(value).replace(";", ",").split(",")
                    if x.strip()]
        # "str" and "platform" pass through as strings
        return str(value)
    except (TypeError, ValueError):
        return value
