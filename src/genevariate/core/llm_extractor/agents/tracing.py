"""Per-agent OpenTelemetry tracing for Arize Phoenix.

Each agent gets its OWN ``TracerProvider`` whose Resource carries a
distinct ``project_name``. Phoenix uses ``project_name`` as the
project key, so traces from different agents land in different
Phoenix projects — letting evaluators score each agent in isolation.

Layout in Phoenix:
    Project: agent.router         (RouterAgent spans)
    Project: agent.collapser-0    (CollapserAgent-0 spans)
    Project: agent.collapser-1    (CollapserAgent-1 spans)
    Project: agent.verifier-0     (VerifierAgent spans)
    Project: agent.ood_mesh_minter   (OodMeshMinterAgent spans)

OpenInference span kinds used:
    AGENT   — handler dispatch in AgentBase.run (the agent "step")
    LLM     — Ollama /api/chat calls inside LLM skills
    CHAIN   — Skill.run wrapper (multi-step skill orchestration)
    RETRIEVER — RAG / mesh-exact retrieval skills

Endpoint: ``PHOENIX_COLLECTOR_ENDPOINT`` env var (default
``http://localhost:6006/v1/traces``). gRPC also works on :4317 but
HTTP is simpler and avoids needing an extra exporter package.
"""
from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from typing import Any, Dict, Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)
from openinference.semconv.trace import (
    SpanAttributes,
    OpenInferenceSpanKindValues,
)


PHOENIX_HTTP_ENDPOINT = os.environ.get(
    "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces",
)

_PROVIDERS: Dict[str, TracerProvider] = {}
_LOCK = threading.Lock()


def _project_name(agent_name: str) -> str:
    """Map an agent name to its Phoenix project_name."""
    return f"agent.{agent_name}"


def get_tracer(agent_name: str) -> trace.Tracer:
    """Return a Tracer whose spans land in Phoenix project ``agent.<name>``.

    Idempotent — the same agent_name always reuses the same
    TracerProvider, so repeated calls are cheap.
    """
    proj = _project_name(agent_name)
    with _LOCK:
        provider = _PROVIDERS.get(proj)
        if provider is None:
            resource = Resource.create({
                "service.name":  proj,
                "project.name":  proj,        # Phoenix legacy attr
                # Phoenix reads `openinference.project.name` first if present.
                "openinference.project.name": proj,
            })
            provider = TracerProvider(resource=resource)
            exporter = OTLPSpanExporter(endpoint=PHOENIX_HTTP_ENDPOINT)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            _PROVIDERS[proj] = provider
    return provider.get_tracer(proj)


# ─────────────────────────────────────────────────────────────────────────────
# Span helpers (OpenInference semantic conventions)
# ─────────────────────────────────────────────────────────────────────────────
def _set_kind(span: trace.Span, kind: OpenInferenceSpanKindValues) -> None:
    span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, kind.value)


def _set_io(span: trace.Span, inp: Any, out: Any) -> None:
    if inp is not None:
        try:
            span.set_attribute(SpanAttributes.INPUT_VALUE, json.dumps(inp, default=str))
            span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, "application/json")
        except Exception:                                       # noqa: BLE001
            span.set_attribute(SpanAttributes.INPUT_VALUE, str(inp))
    if out is not None:
        try:
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps(out, default=str))
            span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "application/json")
        except Exception:                                       # noqa: BLE001
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(out))


@contextmanager
def agent_span(agent_name: str, method: str, params: Dict[str, Any]):
    """Wrap an agent handler dispatch as an AGENT span."""
    tracer = get_tracer(agent_name)
    with tracer.start_as_current_span(f"{agent_name}.{method}") as span:
        _set_kind(span, OpenInferenceSpanKindValues.AGENT)
        span.set_attribute("agent.name",   agent_name)
        span.set_attribute("agent.method", method)
        _set_io(span, params, None)
        ctx: Dict[str, Any] = {"output": None, "error": None}
        try:
            yield ctx
            _set_io(span, params, ctx.get("output"))
            if ctx.get("error"):
                span.set_status(trace.Status(trace.StatusCode.ERROR, ctx["error"]))
        except Exception as e:                                  # noqa: BLE001
            span.set_status(trace.Status(trace.StatusCode.ERROR, f"{type(e).__name__}: {e}"))
            raise


@contextmanager
def llm_span(agent_name: str, skill_name: str, model: str,
             prompt: Any, *, temperature: float = 0.0):
    """Wrap an LLM call as an LLM span."""
    tracer = get_tracer(agent_name)
    with tracer.start_as_current_span(f"{skill_name}.llm") as span:
        _set_kind(span, OpenInferenceSpanKindValues.LLM)
        span.set_attribute(SpanAttributes.LLM_MODEL_NAME,            model)
        span.set_attribute(SpanAttributes.LLM_PROVIDER,              "ollama")
        span.set_attribute("llm.invocation_parameters",
                           json.dumps({"temperature": temperature}))
        _set_io(span, prompt, None)
        ctx: Dict[str, Any] = {"output": None}
        try:
            yield ctx
            _set_io(span, prompt, ctx.get("output"))
        except Exception as e:                                  # noqa: BLE001
            span.set_status(trace.Status(trace.StatusCode.ERROR, f"{type(e).__name__}: {e}"))
            raise


@contextmanager
def skill_span(agent_name: str, skill_name: str,
               kind: OpenInferenceSpanKindValues,
               inp: Any):
    """Generic skill span (CHAIN / RETRIEVER / TOOL)."""
    tracer = get_tracer(agent_name)
    with tracer.start_as_current_span(f"{skill_name}") as span:
        _set_kind(span, kind)
        span.set_attribute("skill.name", skill_name)
        _set_io(span, inp, None)
        ctx: Dict[str, Any] = {"output": None}
        try:
            yield ctx
            _set_io(span, inp, ctx.get("output"))
        except Exception as e:                                  # noqa: BLE001
            span.set_status(trace.Status(trace.StatusCode.ERROR, f"{type(e).__name__}: {e}"))
            raise


def shutdown() -> None:
    """Flush all per-agent providers (call before process exit)."""
    with _LOCK:
        for p in _PROVIDERS.values():
            try:
                p.shutdown()
            except Exception:                                   # noqa: BLE001
                pass
