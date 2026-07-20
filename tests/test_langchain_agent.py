"""
Offline tests for the LangChain reasoning agent (``core.chatbot.langchain_agent``).

These run without the ``agent`` extra installed: they exercise the availability
gate, the graceful-degradation path (``run_agent`` returns a failed reply that
callers fall back on), model configuration, and — only when LangChain is
actually present — the registry→StructuredTool wrapping. No network, no ollama.
"""
from __future__ import annotations

import pytest

from genevariate.core.chatbot import build_registry
from genevariate.core.chatbot import langchain_agent as la


class FakeApp:
    def __init__(self, platforms=None):
        self.gpl_datasets = platforms or {}


def test_default_model_env_override(monkeypatch):
    monkeypatch.delenv("GENEVARIATE_AGENT_MODEL", raising=False)
    assert la._default_model() == la.DEFAULT_AGENT_MODEL
    monkeypatch.setenv("GENEVARIATE_AGENT_MODEL", "qwen2.5:7b")
    assert la._default_model() == "qwen2.5:7b"
    monkeypatch.setenv("GENEVARIATE_AGENT_MODEL", "  ")
    assert la._default_model() == la.DEFAULT_AGENT_MODEL


def test_agent_unavailable_without_langchain(monkeypatch):
    """No LangChain (or no ollama) ⇒ unavailable + a helpful reason."""
    monkeypatch.setattr(la, "_HAS_LANGCHAIN", False)
    assert la.agent_available() is False
    reason = la.unavailable_reason()
    assert "pip install" in reason.lower()


def test_run_agent_degrades_gracefully(monkeypatch):
    """When the stack is missing, run_agent never raises; it flags ok=False."""
    monkeypatch.setattr(la, "_HAS_LANGCHAIN", False)
    reg = build_registry(FakeApp())
    events = []
    reply = la.run_agent(
        FakeApp(), reg, "compare TP53 across sources",
        lambda kind, text, result: events.append((kind, text)))
    assert isinstance(reply, la.AgentReply)
    assert reply.ok is False
    assert reply.summary  # carries the reason
    assert reply.results == []


def test_agent_available_needs_server(monkeypatch):
    """Even with LangChain present, a dead ollama server means unavailable."""
    monkeypatch.setattr(la, "_HAS_LANGCHAIN", True)
    import genevariate.core.ollama_manager as om
    monkeypatch.setattr(om, "ollama_server_ok", lambda *a, **k: False)
    assert la.agent_available() is False
    assert "ollama" in la.unavailable_reason().lower()


# ---- these need the real LangChain stack; skipped without the extra ----
_needs_lc = pytest.mark.skipif(
    not la._HAS_LANGCHAIN, reason="agent extra (langchain-ollama) not installed")


@_needs_lc
def test_build_langchain_tools_wraps_registry():
    reg = build_registry(FakeApp())
    sink = {"results": []}
    events = []
    tools = la.build_langchain_tools(
        FakeApp(), reg,
        lambda kind, text, result: events.append((kind, text)),
        None, sink)
    names = {t.name for t in tools}
    assert {"list_platforms", "gene_distribution", "compare_gene"} <= names


@_needs_lc
def test_wrapped_tool_executes_and_narrates():
    reg = build_registry(FakeApp())
    sink = {"results": []}
    events = []
    tools = la.build_langchain_tools(
        FakeApp(), reg,
        lambda kind, text, result: events.append((kind, text)),
        None, sink)
    lp = next(t for t in tools if t.name == "list_platforms")
    obs = lp.invoke({})
    assert "No platforms" in obs
    assert sink["results"] and sink["results"][0].ok is False
    kinds = [k for k, _ in events]
    assert "tool_start" in kinds and "tool_result" in kinds
