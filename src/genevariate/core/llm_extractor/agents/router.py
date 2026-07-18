"""RouterAgent — the planner brain of the multi-agent extraction system.

Given a (raw, col) pair, this agent decides which Phase 2 strategy to
invoke. It owns the LLMRouterSkill and is consulted by CollapserAgent
before the cascade runs.

Strategies returned (today):
    full_cascade : default — episodic → mesh-exact → RAG → picker → verify
    mint_direct  : skip retrieval, mint directly into the out-of-distribution
                   (OOD) mesh

Why a router helps despite a single-source (MeshDB) backend:
    The 10-sample baseline minted 14/20 unique labels — 75% OOD-mesh rate —
    because labels like 'intrathymic T progenitor (ITTP) cells' are not
    in MeSH. The cascade ran BioLORD retrieval + LLM picker + verifier
    for each (~720ms) before falling through to mint. The router can
    short-circuit those, saving ~720ms × 14 ≈ 10s on this run.

Failure mode: the router is best-effort. If LLMRouterSkill errors or the
verdict is unparseable, the agent returns full_cascade (the safe default).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .base import AgentBase, AgentMessage, MessageBus
from .skills import LLMRouterSkill


class RouterAgent(AgentBase):
    """Capability: ``route(raw, col) -> {strategy, reason, elapsed_s}``."""

    def __init__(self, bus: MessageBus,
                 ollama_url: str = "http://localhost:11434",
                 model: str = "gemma4-e2b-text:latest",
                 name: str = "router") -> None:
        super().__init__(name=name, bus=bus)
        self.skill = LLMRouterSkill(ollama_url=ollama_url, model=model,
                                    agent_name=name)
        self.handlers = {"route": self._route}
        self._finalize_routes()

    def _route(self, msg: AgentMessage) -> Dict[str, Any]:
        raw = msg.params.get("raw", "")
        col = msg.params.get("col", "")
        result = self.skill.run(raw=raw, col=col)
        if not result.ok:
            # Safe default — keep the pipeline running even if the
            # router skill faulted.
            return {
                "strategy":  "full_cascade",
                "reason":    f"router_error:{result.error}",
                "elapsed_s": result.elapsed_s,
            }
        return {
            "strategy":  result.data["strategy"],
            "reason":    result.data.get("raw_response", ""),
            "elapsed_s": result.elapsed_s,
        }
