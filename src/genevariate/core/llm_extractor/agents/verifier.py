"""VerifierAgent — Tier 4.5 self-check as an independent A2A worker.

Pattern: CollapserAgent proposes a (raw, col, picked) triple via
``verify_pick``. The VerifierAgent consults the shared
``verifier_decisions`` cache; on miss, runs the verifier LLM; writes
the verdict through; replies KEEP or REJECT.

Independence buys us: (a) the verifier scales separately from the
picker fleet, (b) verifier prompt updates redeploy this pod alone,
(c) verifier failures don't crash the picker — collapser sees a
REJECT (or error) and falls through to mint.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from mesh_lookup import MeshDB
from phase2_mesh import Phase2Mesh, _PROMPT_VERSION
from .base import AgentBase, AgentMessage, MessageBus
from .skills import SCOPE_SKILLS


class VerifierAgent(AgentBase):
    """Wraps Phase2Mesh._verify_pick + a column-scope input validator.

    Two capabilities:
      * ``verify_pick``  — KEEP/REJECT a picked MeSH candidate (post-pick).
      * ``verify_scope`` — Is the RAW input plausibly a value of COL?
        Pre-cascade gate; rejects type-invalid inputs (e.g. tissue spans
        leaked into Condition) so collapser can short-circuit to NS
        instead of minting spurious OOD-mesh entries.
    """

    def __init__(self, bus: MessageBus, db: Optional[MeshDB] = None,
                 name: str = "verifier") -> None:
        super().__init__(name=name, bus=bus)
        self.db = db or MeshDB()
        self._p2 = Phase2Mesh(
            db=self.db, use_episodic=False, use_picker=False,
            use_pubtator=False, use_verifier=True,
        )
        # Per-label scope skills — one instance per column. Each has its
        # own narrow few-shot prompt, independently tunable.
        self._scope_skills = {
            col: cls(agent_name=name) for col, cls in SCOPE_SKILLS.items()
        }
        self.handlers = {
            "verify_pick":  self._verify_pick,
            "verify_scope": self._verify_scope,
        }
        self._finalize_routes()

    def _verify_pick(self, msg: AgentMessage) -> Dict[str, Any]:
        label = msg.params["label"]
        col   = msg.params["col"]
        candidate = msg.params["candidate"]
        keep = self._p2._verify_pick(label, col, candidate)
        return {
            "verdict":        "KEEP" if keep else "REJECT",
            "prompt_version": _PROMPT_VERSION,
        }

    def _verify_scope(self, msg: AgentMessage) -> Dict[str, Any]:
        raw = msg.params.get("raw", "")
        col = msg.params.get("col", "")
        skill = self._scope_skills.get(col)
        if skill is None:
            # Unknown column — fail open so the cascade still runs.
            return {"in_scope": True, "reason": f"no_scope_skill:{col}"}
        result = skill.run(raw=raw)
        if not result.ok:
            # Fail-open: if the scope skill errors, let the cascade run.
            return {"in_scope": True, "reason": f"scope_err:{result.error}"}
        return {
            "in_scope": bool(result.data.get("in_scope", True)),
            "reason":   result.data.get("raw_response", ""),
        }
