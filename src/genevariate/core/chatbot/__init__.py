"""Tk-free conversational-assistant core for GeneVariate.

Turns a natural-language prompt into a proposed analysis tool + params
(``route``), executed only after the user confirms in the chat sidebar.
Executors call the existing ``genevariate.core.analysis`` API.
"""
from .tools import Tool, ToolParam, ToolResult, Action
from .registry import build_registry
from .router import route
from .agent import plan, run_plan, Plan, Step, AgentRun
from .langchain_agent import (
    run_agent,
    agent_available,
    unavailable_reason,
    AgentReply,
    DEFAULT_AGENT_MODEL,
)

__all__ = [
    "Tool",
    "ToolParam",
    "ToolResult",
    "Action",
    "build_registry",
    "route",
    "plan",
    "run_plan",
    "Plan",
    "Step",
    "AgentRun",
    "run_agent",
    "agent_available",
    "unavailable_reason",
    "AgentReply",
    "DEFAULT_AGENT_MODEL",
]
