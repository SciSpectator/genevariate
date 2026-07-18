"""OodMeshMinterAgent — singleton owner of the ART-{T,C,X}-##### namespace
in the out-of-distribution (OOD) mesh.

A2A pattern: every CollapserAgent that needs to mint a new
out-of-distribution-mesh entry sends a ``mint_ood_mesh(label, col)``
request to this one agent. That serializes mint requests through one
writer, eliminating the race where two collapsers both read MAX(id)
and both INSERT ART-T-00042.

The agent is a thin wrapper around ``MeshDB.create_ood_mesh``, which
is itself transaction-safe via BEGIN IMMEDIATE — but the A2A funnel
keeps contention off the SQLite write lock and gives a single
audit point for namespace allocation.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from mesh_lookup import MeshDB
from .base import AgentBase, AgentMessage, MessageBus


class OodMeshMinterAgent(AgentBase):
    """Singleton agent: ``mint_ood_mesh``, ``lookup_ood_mesh``."""

    def __init__(self, bus: MessageBus, db: Optional[MeshDB] = None,
                 name: str = "ood-mesh-minter") -> None:
        super().__init__(name=name, bus=bus)
        self.db = db or MeshDB()
        self.handlers = {
            "mint_ood_mesh":   self._mint_ood_mesh,
            "lookup_ood_mesh": self._lookup_ood_mesh,
        }
        self._finalize_routes()

    def _mint_ood_mesh(self, msg: AgentMessage) -> Dict[str, Any]:
        label = msg.params.get("label", "")
        col   = msg.params.get("col", "")
        return self.db.create_ood_mesh(label, col)

    def _lookup_ood_mesh(self, msg: AgentMessage) -> Dict[str, Any]:
        label = msg.params.get("label", "")
        col   = msg.params.get("col", "")
        hit = self.db.lookup_ood_mesh(label, col)
        return {"hit": hit}
