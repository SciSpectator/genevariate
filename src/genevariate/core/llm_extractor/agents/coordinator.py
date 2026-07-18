"""Coordinator — boots the agent fleet and exposes a synchronous
``collapse(raw, col)`` API for callers that just want labels.

Topology:

    [client] ── collapse ──> CollapserAgent(s)
                                ├── route          ──> RouterAgent (planner)
                                ├── verify_pick    ──> VerifierAgent(s)
                                └── mint_ood_mesh  ──> OodMeshMinterAgent (singleton)

Scale: spin up N collapsers + M verifiers + 1 router + 1 minter. The
MessageBus load-balances multiple agents claiming the same method by
registering each under a unique name; for fan-out, replace this default
bus with a Redis-Streams-backed implementation that consumer-groups
across collapser pods.

Set ``use_router=True`` to enable the planner — collapsers consult it
before running the retrieval cascade and may short-circuit to
``mint_direct`` for labels likely outside MeSH coverage.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from mesh_lookup import MeshDB
from .base import AgentBase, AgentMessage, MessageBus
from .ood_mesh_minter import OodMeshMinterAgent
from .collapser    import CollapserAgent
from .verifier     import VerifierAgent
from .router       import RouterAgent


class Coordinator:
    """Owns the bus + the fleet. ``start()`` then ``collapse()``."""

    def __init__(self, n_collapsers: int = 2, n_verifiers: int = 1,
                 use_router: bool = False,
                 db: Optional[MeshDB] = None,
                 cache: Any = None) -> None:
        """``cache`` is an optional ``GSEContextCache`` shared with all
        collapsers. When set, Phase 2 enables per-GSE sibling consistency
        — semantically equivalent raws within one experiment converge on
        the same canonical id (Tier 1.5/1.6 in ``Phase2Mesh._resolve_one``).
        """
        self.bus  = MessageBus()
        self.db   = db or MeshDB()
        self.use_router = use_router
        self.cache = cache
        self.agents: List[AgentBase] = []

        # Singleton minter — owns the ART namespace in the OOD mesh.
        self.agents.append(OodMeshMinterAgent(self.bus, db=self.db))

        # Verifier pool.
        for i in range(max(1, n_verifiers)):
            self.agents.append(VerifierAgent(
                self.bus, db=self.db, name=f"verifier-{i}",
            ))

        # Optional planner / router — singleton for now. Cheap LLM call,
        # so one instance is plenty unless we cluster-scale.
        if use_router:
            self.agents.append(RouterAgent(self.bus))

        # Collapser pool — each delegates verify+mint+route over the bus.
        self._collapser_names: List[str] = []
        for i in range(max(1, n_collapsers)):
            cname = f"collapser-{i}"
            self.agents.append(CollapserAgent(
                self.bus, db=self.db, name=cname,
                delegate_router=use_router,
                delegate_verifier=True, delegate_minter=True,
                cache=self.cache,
            ))
            self._collapser_names.append(cname)
        self._rr = 0     # round-robin index for collapser dispatch

        # NOTE: VerifierAgent and CollapserAgent both register
        # ``verify_pick``/``collapse`` capabilities. The bus's routing
        # table keeps the LAST registered agent for each method, but
        # we explicitly target collapsers by name in ``collapse()``
        # below to round-robin across the pool.

    def start(self) -> None:
        for a in self.agents:
            a.start()

    def stop(self) -> None:
        for a in self.agents:
            a.stop()
        for a in self.agents:
            a.join(timeout=2.0)

    # ── client API ──────────────────────────────────────────────────────
    def collapse(self, raw: str, col: str, context: str = "",
                 gse_id: Optional[str] = None,
                 timeout: float = 300.0) -> Dict[str, Any]:
        """Round-robin dispatch a collapse request to the collapser pool.

        ``context`` is optional GSE-level free-form text (title + summary +
        overall_design) forwarded to the picker / verifier so study-defined
        abbreviations and brand names can be expanded.

        ``gse_id`` opts into per-GSE sibling consistency — when set AND the
        Coordinator was constructed with a ``cache``, sibling raws in the
        same experiment converge on the same canonical id without
        re-running the LLM cascade.
        """
        target = self._collapser_names[self._rr % len(self._collapser_names)]
        self._rr += 1
        params: Dict[str, Any] = {"raw": raw, "col": col, "context": context}
        if gse_id:
            params["gse_id"] = gse_id
        resp = self.bus.call(AgentMessage(
            method="collapse", recipient=target,
            sender="__coordinator__",
            params=params,
        ), timeout=timeout)
        if resp.error:
            raise RuntimeError(f"collapse failed: {resp.error}")
        return resp.result or {}
