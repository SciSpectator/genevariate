"""A2A (agent-to-agent) coordination layer for the multi-agent LLM
extractor system.

Module layout:

    base            AgentMessage, AgentBase, MessageBus
    skills          Skill ABC + concrete skills (RAG, Mesh-exact, LLM-picker, etc.)
    ood_mesh_minter OodMeshMinterAgent (singleton OOD-mesh namespace owner)
    collapser       CollapserAgent (Phase 2 resolver)
    verifier        VerifierAgent (Tier 4.5 self-check)
    router          RouterAgent (planner — picks Phase 2 strategy)
    coordinator     Coordinator (boots fleet, routes requests)

Transport is pluggable. The default ``MessageBus`` uses ``queue.Queue``
for in-process fan-out so the same code path runs on a workstation and
on a cluster. For multi-host deployment, swap ``MessageBus`` for a
Redis Streams / NATS / actual A2A-protocol implementation — agents
themselves are transport-agnostic.
"""
from .base        import AgentBase, AgentMessage, MessageBus
from .skills      import (
    Skill, SkillResult,
    EpisodicLookupSkill, MeshExactLookupSkill, OodMeshLookupSkill,
    RAGSkill, OodMeshMintSkill,
    LLMRouterSkill, LLMPickerSkill, LLMVerifierSkill,
)
from .ood_mesh_minter import OodMeshMinterAgent
from .collapser   import CollapserAgent
from .verifier    import VerifierAgent
from .router      import RouterAgent
from .coordinator import Coordinator

__all__ = [
    "AgentBase", "AgentMessage", "MessageBus",
    "Skill", "SkillResult",
    "EpisodicLookupSkill", "MeshExactLookupSkill", "OodMeshLookupSkill",
    "RAGSkill", "OodMeshMintSkill",
    "LLMRouterSkill", "LLMPickerSkill", "LLMVerifierSkill",
    "OodMeshMinterAgent", "CollapserAgent", "VerifierAgent",
    "RouterAgent", "Coordinator",
]
