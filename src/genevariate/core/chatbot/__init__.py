"""Tk-free conversational-assistant core for GeneVariate.

Turns a natural-language prompt into a proposed analysis tool + params
(``route``), executed only after the user confirms in the chat sidebar.
Executors call the existing ``genevariate.core.analysis`` API.
"""
from .tools import Tool, ToolParam, ToolResult, Action
from .registry import build_registry
from .router import route

__all__ = [
    "Tool",
    "ToolParam",
    "ToolResult",
    "Action",
    "build_registry",
    "route",
]
