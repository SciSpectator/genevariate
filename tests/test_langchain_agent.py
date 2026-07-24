"""
Offline tests for the LangChain reasoning agent (``core.chatbot.langchain_agent``).

These run without any backend client or network: they exercise backend/model
selection, the availability gate per backend (Groq / Ollama), graceful
degradation, API-key surfacing + persistence, and — when the base LangChain
stack is present — the registry→StructuredTool wrapping. No ollama, no Groq.
"""
from __future__ import annotations

import json
import os

import pytest

from genevariate.core.chatbot import build_registry
from genevariate.core.chatbot import langchain_agent as la


class FakeApp:
    def __init__(self, platforms=None):
        self.gpl_datasets = platforms or {}


def test_default_model_follows_backend(monkeypatch):
    monkeypatch.delenv("GENEVARIATE_AGENT_BACKEND", raising=False)
    monkeypatch.delenv("GENEVARIATE_AGENT_MODEL", raising=False)
    # default backend is local ollama on gemma4 (no API key)
    assert la._default_model() == la.DEFAULT_AGENT_MODEL
    assert la.DEFAULT_BACKEND == "ollama"
    assert la.DEFAULT_AGENT_MODEL == "gemma4:e2b"
    # switching backend switches the default model
    monkeypatch.setenv("GENEVARIATE_AGENT_BACKEND", "groq")
    assert "llama-3.3-70b" in la._default_model()
    # explicit model override wins; blank falls back
    monkeypatch.setenv("GENEVARIATE_AGENT_MODEL", "qwen2.5:14b")
    assert la._default_model() == "qwen2.5:14b"
    monkeypatch.setenv("GENEVARIATE_AGENT_MODEL", "  ")
    assert "llama-3.3-70b" in la._default_model()


def test_resolve_ollama_model_appends_quant(monkeypatch):
    monkeypatch.delenv("GENEVARIATE_AGENT_QUANT", raising=False)
    # no quant env -> tag unchanged
    assert la._resolve_ollama_model("qwen2.5:7b") == "qwen2.5:7b"
    monkeypatch.setenv("GENEVARIATE_AGENT_QUANT", "q5_K_M")
    assert la._resolve_ollama_model("qwen2.5:7b") == "qwen2.5:7b-q5_K_M"
    # a tag that already pins a quant is left alone
    assert (la._resolve_ollama_model("qwen2.5:7b-instruct-q4_K_M")
            == "qwen2.5:7b-instruct-q4_K_M")
    # bare name (no colon) is left alone
    assert la._resolve_ollama_model("qwen2.5") == "qwen2.5"


def test_ollama_options_are_env_overridable(monkeypatch):
    monkeypatch.delenv("GENEVARIATE_AGENT_KEEP_ALIVE", raising=False)
    monkeypatch.delenv("GENEVARIATE_AGENT_NUM_CTX", raising=False)
    monkeypatch.delenv("GENEVARIATE_AGENT_NUM_PREDICT", raising=False)
    opts = la._ollama_options()
    assert opts["temperature"] == 0
    assert opts["keep_alive"] == "30m"
    assert opts["num_ctx"] == 8192 and opts["num_predict"] == 1024
    monkeypatch.setenv("GENEVARIATE_AGENT_KEEP_ALIVE", "1h")
    monkeypatch.setenv("GENEVARIATE_AGENT_NUM_CTX", "4096")
    monkeypatch.setenv("GENEVARIATE_AGENT_NUM_PREDICT", "bad")
    opts = la._ollama_options()
    assert opts["keep_alive"] == "1h" and opts["num_ctx"] == 4096
    # non-int falls back to the default
    assert opts["num_predict"] == 1024


def test_agent_unavailable_without_langchain(monkeypatch):
    monkeypatch.setattr(la, "_HAS_LANGCHAIN", False)
    assert la.agent_available() is False
    assert "langchain" in la.unavailable_reason().lower()


def test_run_agent_degrades_gracefully(monkeypatch):
    monkeypatch.setattr(la, "_HAS_LANGCHAIN", False)
    reg = build_registry(FakeApp())
    events = []
    reply = la.run_agent(
        FakeApp(), reg, "compare TP53 across sources",
        lambda kind, text, result: events.append((kind, text)))
    assert isinstance(reply, la.AgentReply)
    assert reply.ok is False
    assert reply.summary
    assert reply.results == []


def test_groq_backend_needs_key(monkeypatch):
    monkeypatch.setattr(la, "_HAS_LANGCHAIN", True)
    monkeypatch.setenv("GENEVARIATE_AGENT_BACKEND", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert la.agent_available() is False
    assert "groq" in la.unavailable_reason().lower()
    prompt = la.api_key_prompt()
    assert prompt and prompt["env"] == "GROQ_API_KEY"
    assert prompt["url"]


def test_ollama_backend_needs_server(monkeypatch):
    monkeypatch.setattr(la, "_HAS_LANGCHAIN", True)
    monkeypatch.setenv("GENEVARIATE_AGENT_BACKEND", "ollama")
    import genevariate.core.ollama_manager as om
    monkeypatch.setattr(om, "ollama_server_ok", lambda *a, **k: False)
    assert la.agent_available() is False
    assert "ollama" in la.unavailable_reason().lower()
    # local backend never asks for an API key
    assert la.api_key_prompt() is None


def test_persist_api_key_round_trip(monkeypatch, tmp_path):
    monkeypatch.setattr(la, "_CFG_DIR", tmp_path)
    monkeypatch.setattr(la, "_CFG_FILE", tmp_path / "agent.json")
    monkeypatch.delenv("FAKE_TEST_KEY", raising=False)
    la.persist_api_key("FAKE_TEST_KEY", "abc123")
    assert os.environ["FAKE_TEST_KEY"] == "abc123"
    saved = json.loads((tmp_path / "agent.json").read_text())
    assert saved["FAKE_TEST_KEY"] == "abc123"


# ---- these need the base LangChain stack; skipped if it is absent ----
_needs_lc = pytest.mark.skipif(
    not la._HAS_LANGCHAIN, reason="base langchain not installed")


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
