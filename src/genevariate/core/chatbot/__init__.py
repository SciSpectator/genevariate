"""Tk-free conversational-assistant core for GeneVariate."""
from .tools import Tool, ToolParam, ToolResult, Action
from .registry import build_registry
from .router import route
from .agent import plan, run_plan, Plan, Step, AgentRun
from .langchain_agent import (
    run_agent,
    agent_available,
    unavailable_reason,
    ensure_agent_ready,
    api_key_prompt,
    persist_api_key,
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
    "ensure_agent_ready",
    "api_key_prompt",
    "persist_api_key",
    "AgentReply",
    "DEFAULT_AGENT_MODEL",
]
