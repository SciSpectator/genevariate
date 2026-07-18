"""CollapserAgent — Phase 2 resolver as an A2A worker.

Capability: ``collapse(raw, col)`` — returns canonical name(s) and
component breakdown, exactly the same payload as
``Phase2Mesh.collapse``.

Routing: this agent can delegate three pieces of the cascade over the
bus to specialist agents:
    delegate_router=True    → consult RouterAgent first, may short-circuit
    delegate_verifier=True  → KEEP/REJECT picks via VerifierAgent
    delegate_minter=True    → mint into the out-of-distribution (OOD) mesh
                              via OodMeshMinterAgent (singleton)

For local / single-process deployment with all three flags off, the
agent runs Phase2Mesh.collapse directly with no bus traffic.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from mesh_lookup import MeshDB
from phase2_mesh import Phase2Mesh, NS
from .base import AgentBase, AgentMessage, MessageBus


class CollapserAgent(AgentBase):
    def __init__(self, bus: MessageBus, db: Optional[MeshDB] = None,
                 name: str = "collapser",
                 delegate_router:   bool = False,
                 delegate_verifier: bool = False,
                 delegate_minter:   bool = False,
                 cache: Any = None) -> None:
        super().__init__(name=name, bus=bus)
        self.db = db or MeshDB()
        self.delegate_router   = delegate_router
        self.delegate_verifier = delegate_verifier
        self.delegate_minter   = delegate_minter

        # Phase 2 driver gets a PRIVATE MeshDB when we delegate minting,
        # so monkey-patching create_ood_mesh on _p2.db cannot corrupt the
        # shared MeshDB instance the OodMeshMinterAgent uses (which
        # would cause a re-entrant mint_ood_mesh deadlock). The
        # underlying SQLite file is shared via WAL — only the Python
        # wrapper is per-agent.
        p2_db = MeshDB() if delegate_minter else self.db
        self._p2 = Phase2Mesh(
            db=p2_db, use_verifier=not delegate_verifier,
            cache=cache,
        )
        # Install delegations once — _p2 is per-collapser, p2_db is
        # private when delegating, so patching is safe.
        if self.delegate_verifier:
            self._p2._verify_pick = self._verify_via_bus      # type: ignore[method-assign]
        if self.delegate_minter:
            self._p2.db.create_ood_mesh = self._mint_via_bus  # type: ignore[method-assign]
        self.handlers = {"collapse": self._collapse}
        self._finalize_routes()

    # ── main entry ──────────────────────────────────────────────────────
    def _collapse(self, msg: AgentMessage) -> Dict[str, Any]:
        raw     = msg.params.get("raw", "")
        col     = msg.params.get("col", "")
        context = msg.params.get("context", "")
        gse_id  = (msg.params.get("gse_id") or msg.params.get("gse")
                   or msg.params.get("series_id") or None)

        # Trivial NS short-circuit — saves a router LLM call.
        if not raw or raw.strip().lower() == NS.lower():
            return {"label": NS, "components": [], "id": ""}

        # NOTE: column-scope gate intentionally DISABLED — minimal binary
        # classifier skills cannot reliably gate without false-NS rejections
        # of valid abbreviations / dosed compounds / cancer cell lines.
        # Per-label scope skills remain wired in VerifierAgent for traceable
        # advisory use, but no longer short-circuit collapse to NS.
        # See feedback_minimal_prompts.md + feedback_per_label_skills.md.

        # Consult the router first when delegated. The router decides
        # whether to skip the cascade entirely (mint_direct) for labels
        # known to be out-of-MeSH-vocabulary, saving ~720ms of retrieval
        # + LLM-picker per such case.
        if self.delegate_router:
            decision = self._route_via_bus(raw, col)
            strategy = decision.get("strategy", "full_cascade")
            if strategy == "mint_direct":
                return self._mint_direct(raw, col)
            # else fall through to full cascade

        return self._p2.collapse(raw, col, context, gse_id)

    def _scope_via_bus(self, raw: str, col: str) -> Dict[str, Any]:
        try:
            resp = self.bus.call(AgentMessage(
                method="verify_scope", sender=self.name,
                params={"raw": raw, "col": col},
            ), timeout=15.0)
        except Exception:                                 # noqa: BLE001
            return {"in_scope": True, "reason": "scope_unreachable"}
        if resp.error or not resp.result:
            return {"in_scope": True, "reason": f"scope_err:{resp.error}"}
        return resp.result

    # ── strategy: mint_direct ───────────────────────────────────────────
    def _mint_direct(self, raw: str, col: str) -> Dict[str, Any]:
        """Skip the retrieval cascade and mint each component directly
        into the out-of-distribution (OOD) mesh.

        Still consults the episodic cache and the existing OOD-mesh
        lookup before minting, so re-runs are idempotent and don't
        allocate duplicate ART-* IDs.
        """
        parts = [p.strip() for p in re.split(r";\s*", raw) if p.strip()]
        comps: List[Dict[str, Any]] = []
        for p in parts:
            comps.append(self._mint_direct_one(p, col))
        return {
            "label":      "; ".join(c["name"] for c in comps),
            "id":         "; ".join(c["id"]   for c in comps),
            "components": comps,
        }

    def _mint_direct_one(self, raw: str, col: str) -> Dict[str, Any]:
        # Episodic cache hit?
        history = self.db.get_resolution_history(raw, col, k=1)
        if history:
            h = history[0]
            return {"raw": raw, "id": h["output_id"], "name": h["output_name"],
                    "source": f"episodic:{h['source']}"}
        # Already-minted OOD-mesh entry?
        existing = self.db.lookup_ood_mesh(raw, col)
        if existing:
            self.db.record_resolution(raw, col, existing["id"],
                                      existing["label"], "ood-mesh-existing")
            return {"raw": raw, "id": existing["id"],
                    "name": existing["label"], "source": "ood-mesh-existing"}
        # Mint (over bus when delegated).
        if self.delegate_minter:
            minted = self._mint_via_bus(raw, col)
        else:
            minted = self.db.create_ood_mesh(raw, col)
        self.db.record_resolution(raw, col, minted["id"], minted["label"],
                                  "ood-mesh-minted")
        return {"raw": raw, "id": minted["id"], "name": minted["label"],
                "source": "ood-mesh-minted"}

    # ── A2A delegations ─────────────────────────────────────────────────
    def _route_via_bus(self, raw: str, col: str) -> Dict[str, Any]:
        try:
            resp = self.bus.call(AgentMessage(
                method="route", sender=self.name,
                params={"raw": raw, "col": col},
            ), timeout=15.0)
        except Exception:                                 # noqa: BLE001
            return {"strategy": "full_cascade", "reason": "router_unreachable"}
        if resp.error or not resp.result:
            return {"strategy": "full_cascade", "reason": f"router_err:{resp.error}"}
        return resp.result

    def _verify_via_bus(self, label: str, col: str, candidate: dict,
                        context: str = "") -> bool:
        resp = self.bus.call(AgentMessage(
            method="verify_pick", sender=self.name,
            params={"label": label, "col": col, "candidate": candidate,
                    "context": context},
        ), timeout=180.0)
        if resp.error or not resp.result:
            return False
        return resp.result.get("verdict") == "KEEP"

    def _mint_via_bus(self, label: str, col: str) -> Dict[str, Any]:
        resp = self.bus.call(AgentMessage(
            method="mint_ood_mesh", sender=self.name,
            params={"label": label, "col": col},
        ), timeout=30.0)
        if resp.error or not resp.result:
            raise RuntimeError(f"mint_ood_mesh via bus failed: {resp.error}")
        return resp.result
