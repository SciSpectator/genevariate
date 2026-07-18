"""Skill primitives — single-step units the agents compose into pipelines.

A "skill" is one atomic operation: an LLM call, a DB lookup, a vector
retrieval, an HTTP call, etc. Skills are reusable across agents and
testable in isolation. Higher-level agents (Router, Collapser) orchestrate
skills based on routing decisions.

Skills here wrap existing primitives in mesh_lookup / phase2_mesh / ollama,
so behavior is preserved — the abstraction is purely organizational.

Skills implemented:
    - EpisodicLookupSkill   : past-resolution cache (Tier 1)
    - MeshExactLookupSkill  : exact MeSH match (Tier 2)
    - OodMeshLookupSkill    : existing out-of-distribution (OOD) mesh lookup (Tier 3)
    - RAGSkill              : BioLORD-2023 semantic retrieval over MeSH
    - LLMPickerSkill        : pick best candidate from a list
    - LLMVerifierSkill      : KEEP / REJECT a picked candidate
    - LLMRouterSkill        : the planner brain — picks a strategy
    - OodMeshMintSkill      : mint a new ART-* cluster into the OOD mesh (Tier 5)

The RAG Skill is the highest-value piece: it is what gives MeSH coverage
beyond exact match. The Router uses LLMRouterSkill to decide whether to
invoke RAG (full_cascade) or skip it (mint_direct) per (raw, col).
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mesh_lookup import MeshDB


# ─────────────────────────────────────────────────────────────────────────────
# Skill base
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SkillResult:
    """Uniform return type for every skill."""
    ok:         bool
    data:       Any                    = None
    elapsed_s:  float                  = 0.0
    error:      Optional[str]          = None
    metadata:   Dict[str, Any]         = field(default_factory=dict)


class Skill(ABC):
    """Abstract single-step unit.

    Subclasses implement ``_run(**kwargs)``. The base ``run`` adds timing
    and exception capture so callers (agents) get a uniform SkillResult.
    """
    name: str = "skill"

    def run(self, **kwargs: Any) -> SkillResult:
        t0 = time.time()
        try:
            data = self._run(**kwargs)
            return SkillResult(ok=True, data=data,
                               elapsed_s=time.time() - t0)
        except Exception as e:                              # noqa: BLE001
            return SkillResult(ok=False, data=None,
                               elapsed_s=time.time() - t0,
                               error=f"{type(e).__name__}: {e}")

    @abstractmethod
    def _run(self, **kwargs: Any) -> Any: ...


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval skills (deterministic, sub-50ms)
# ─────────────────────────────────────────────────────────────────────────────
class EpisodicLookupSkill(Skill):
    """Tier 1 — past resolution cache."""
    name = "episodic_lookup"

    def __init__(self, db: MeshDB) -> None:
        self.db = db

    def _run(self, raw: str, col: str) -> Optional[Dict[str, Any]]:
        history = self.db.get_resolution_history(raw, col, k=1)
        if not history:
            return None
        h = history[0]
        return {"raw": raw, "id": h["output_id"], "name": h["output_name"],
                "source": f"episodic:{h['source']}"}


class MeshExactLookupSkill(Skill):
    """Tier 2 — exact MeSH match (case-insensitive, scope-gated by col)."""
    name = "mesh_exact"

    def __init__(self, db: MeshDB) -> None:
        self.db = db

    def _run(self, raw: str, col: str) -> List[Dict[str, Any]]:
        return self.db.lookup_mesh(raw, col)


class OodMeshLookupSkill(Skill):
    """Tier 3 — already-minted out-of-distribution (OOD) mesh lookup."""
    name = "ood_mesh_lookup"

    def __init__(self, db: MeshDB) -> None:
        self.db = db

    def _run(self, raw: str, col: str) -> Optional[Dict[str, Any]]:
        return self.db.lookup_ood_mesh(raw, col)


class RAGSkill(Skill):
    """BioLORD-2023 semantic retrieval over MeSH (top-K).

    This IS the RAG step. Without it, anything past exact-match would have
    no candidates to pick from and would mint immediately. The Router
    decides whether to invoke this skill (cost ~20ms + ~700ms picker LLM)
    or skip it for labels likely to be out-of-vocabulary.
    """
    name = "rag_mesh"

    def __init__(self, db: MeshDB, top_k: int = 20) -> None:
        self.db = db
        self.top_k = top_k

    def _run(self, raw: str, col: str,
             k: Optional[int] = None) -> List[Dict[str, Any]]:
        k = k or self.top_k
        return self.db.find_similar_mesh(raw, col, k=k)


class OodMeshMintSkill(Skill):
    """Tier 5 — mint a new ART-* cluster into the out-of-distribution
    (OOD) mesh. Atomic via BEGIN IMMEDIATE."""
    name = "ood_mesh_mint"

    def __init__(self, db: MeshDB) -> None:
        self.db = db

    def _run(self, raw: str, col: str) -> Dict[str, Any]:
        minted = self.db.create_ood_mesh(raw, col)
        self.db.record_resolution(raw, col, minted["id"],
                                  minted["label"], "ood-mesh-minted")
        return {"raw": raw, "id": minted["id"], "name": minted["label"],
                "source": "ood-mesh-minted"}


# ─────────────────────────────────────────────────────────────────────────────
# LLM skills (Ollama / gemma4:e2b)
# ─────────────────────────────────────────────────────────────────────────────
ROUTER_SYSTEM = (
    "You classify metadata labels into TWO routing strategies for a "
    "MeSH-based normalizer:\n"
    "- mint_direct : label is novel / non-MeSH (very-specific cell "
    "populations, sub-type abbreviations, paper-specific IDs, molecule "
    "types like 'genomic DNA').\n"
    "- full_cascade : label is plausibly in MeSH (common organs, "
    "diseases, drugs, broad cell types).\n\n"
    "Reply with EXACTLY one word: mint_direct or full_cascade.\n\n"
    "Examples:\n"
    "RAW: Liver | COLUMN: Tissue -> full_cascade\n"
    "RAW: asthma | COLUMN: Condition -> full_cascade\n"
    "RAW: cisplatin | COLUMN: Treatment -> full_cascade\n"
    "RAW: intrathymic T progenitor (ITTP) cells | COLUMN: Condition -> mint_direct\n"
    "RAW: double positive thymocytes (DP) | COLUMN: Condition -> mint_direct\n"
    "RAW: SP4 | COLUMN: Condition -> mint_direct\n"
    "RAW: genomic DNA | COLUMN: Tissue -> mint_direct\n"
    "RAW: total RNA | COLUMN: Tissue -> mint_direct"
)


class LLMRouterSkill(Skill):
    """The planner brain.

    Calls gemma4:e2b via Ollama's raw HTTP /api/chat with ``think:false``
    and a few-shot prompt to keep latency low (~400-600ms). Falls back to
    'full_cascade' on any error so the pipeline never blocks on the
    router being unavailable.
    """
    name = "llm_router"

    def __init__(self, ollama_url: str = "http://localhost:11434",
                 model: str = "gemma4-e2b-text:latest",
                 timeout_s: float = 10.0,
                 agent_name: str = "router") -> None:
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.agent_name = agent_name
        # Use requests directly — gemma4's `think` flag isn't exposed in
        # ollama-python 0.2.x, but the HTTP API supports it natively.
        import requests
        self._requests = requests

    def _run(self, raw: str, col: str) -> Dict[str, Any]:
        try:
            from .tracing import llm_span
        except Exception:                                # noqa: BLE001
            llm_span = None                              # type: ignore[assignment]

        prompt = {
            "system": ROUTER_SYSTEM,
            "user":   f"RAW: {raw} | COLUMN: {col} ->",
        }
        # think=True is the pipeline-wide default per user policy.
        # Override only via THINK_MODE=false env var (GUI toggles this).
        import os as _os
        _think = _os.environ.get("THINK_MODE", "true").lower() in ("1","true","yes")
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": ROUTER_SYSTEM},
                {"role": "user",
                 "content": f"RAW: {raw} | COLUMN: {col} ->"},
            ],
            "stream": False,
            "think":  _think,
            "keep_alive": -1,
            "options": {"temperature": 0.0,
                        "num_predict": 256 if _think else 4,
                        "seed": 0},
        }

        def _call() -> Dict[str, Any]:
            r = self._requests.post(
                self.ollama_url + "/api/chat", json=body, timeout=self.timeout_s,
            )
            r.raise_for_status()
            out = (r.json().get("message", {}) or {}).get("content", "").strip().lower()
            strategy = "mint_direct" if "mint_direct" in out else "full_cascade"
            return {"strategy": strategy, "raw_response": out}

        if llm_span is None:
            return _call()
        with llm_span(self.agent_name, "llm_router", self.model, prompt) as ctx:
            ctx["output"] = _call()
            return ctx["output"]


# ─────────────────────────────────────────────────────────────────────────────
# Per-label scope skills — one skill per extractor agent (Tissue / Condition /
# Treatment). Each skill carries its OWN narrow few-shot system prompt so
# tuning examples for one label can never dilute the others.
#
# Architectural rule (memory: feedback_per_label_skills.md): skills MUST be
# per-label. NEVER build a single generic skill that switches behavior on a
# `col` parameter — few-shots for one label regress the others.
# ─────────────────────────────────────────────────────────────────────────────
TISSUE_SCOPE_SYSTEM = (
    "Is RAW a tissue, organ, cell type, cell line, abbreviation for "
    "one (e.g. BM, PBMC), or biological sample type (incl. composites "
    "like 'Rod; Retina' or cancer cell lines like 'Prostate cancer')? "
    "If unsure, reply YES. Reply YES or NO.\n"
    "Liver -> YES\n"
    "BM -> YES\n"
    "Rod; Retina -> YES\n"
    "Prostate cancer -> YES\n"
    "cisplatin -> NO\n"
    "male -> NO"
)

CONDITION_SCOPE_SYSTEM = (
    "Is RAW a disease, disease abbreviation (e.g. ETP-ALL, AML), "
    "phenotype, disease stage, cancer type, or explicit "
    "healthy/control clinical state? If unsure, reply YES. "
    "Reply YES or NO.\n"
    "glioblastoma -> YES\n"
    "ovarian cancer -> YES\n"
    "knee osteoarthritis -> YES\n"
    "ETP-ALL -> YES\n"
    "male -> NO\n"
    "wild type -> NO"
)

TREATMENT_SCOPE_SYSTEM = (
    "Is RAW a drug, antibiotic, drug combination (e.g. Oxacillin/"
    "Rifampin), dose-prefixed compound (e.g. 8% D-xylose), genetic "
    "perturbation, surgery, dietary intervention, or experimental "
    "intervention? If unsure, reply YES. Reply YES or NO.\n"
    "cisplatin -> YES\n"
    "Oxacillin/Rifampin -> YES\n"
    "8% D-xylose -> YES\n"
    "CRISPR/Cas9 Edited -> YES\n"
    "library prep -> NO\n"
    "male -> NO"
)


class _BaseScopeSkill(Skill):
    """Shared transport for per-label scope skills.

    Subclasses set ``COL`` and ``SYSTEM_PROMPT`` so each label has its
    own narrow few-shot prompt independently tunable. Single LLM call
    (~400ms, think:false, num_predict=4) returns
    ``{in_scope: bool, raw_response: str}``.
    """
    COL: str = ""
    SYSTEM_PROMPT: str = ""

    def __init__(self, ollama_url: str = "http://localhost:11434",
                 model: str = "gemma4-e2b-text:latest",
                 timeout_s: float = 10.0,
                 agent_name: str = "verifier") -> None:
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.agent_name = agent_name
        import requests
        self._requests = requests

    def _run(self, raw: str, col: str = "") -> Dict[str, Any]:
        # `col` is accepted for caller-symmetry but ignored — each subclass
        # is bound to a single column via its class-level COL.
        try:
            from .tracing import llm_span
        except Exception:                                # noqa: BLE001
            llm_span = None                              # type: ignore[assignment]
        user_msg = f"RAW: {raw} ->"
        prompt = {"system": self.SYSTEM_PROMPT, "user": user_msg}
        # think=True is the pipeline-wide default. Env override only.
        import os as _os
        _think = _os.environ.get("THINK_MODE", "true").lower() in ("1","true","yes")
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            "stream": False, "think": _think, "keep_alive": -1,
            "options": {"temperature": 0.0,
                        "num_predict": 256 if _think else 4,
                        "seed": 42},
        }

        def _call() -> Dict[str, Any]:
            r = self._requests.post(
                self.ollama_url + "/api/chat", json=body,
                timeout=self.timeout_s,
            )
            r.raise_for_status()
            out = (r.json().get("message", {}) or {}).get(
                "content", "").strip().lower()
            in_scope = out.startswith("yes") or (
                "yes" in out and "no" not in out)
            return {"in_scope": in_scope, "raw_response": out, "col": self.COL}

        if llm_span is None:
            return _call()
        with llm_span(self.agent_name, self.name, self.model, prompt) as ctx:
            ctx["output"] = _call()
            return ctx["output"]


class TissueScopeSkill(_BaseScopeSkill):
    """Scope gate for the Tissue extractor agent."""
    name = "tissue_scope"
    COL  = "Tissue"
    SYSTEM_PROMPT = TISSUE_SCOPE_SYSTEM


class ConditionScopeSkill(_BaseScopeSkill):
    """Scope gate for the Condition extractor agent."""
    name = "condition_scope"
    COL  = "Condition"
    SYSTEM_PROMPT = CONDITION_SCOPE_SYSTEM


class TreatmentScopeSkill(_BaseScopeSkill):
    """Scope gate for the Treatment extractor agent."""
    name = "treatment_scope"
    COL  = "Treatment"
    SYSTEM_PROMPT = TREATMENT_SCOPE_SYSTEM


# Dispatch table for callers that want to look up a scope skill by column.
SCOPE_SKILLS = {
    "Tissue":    TissueScopeSkill,
    "Condition": ConditionScopeSkill,
    "Treatment": TreatmentScopeSkill,
}


class LLMPickerSkill(Skill):
    """Pick the best candidate from a list (uses Phase2Mesh._pick prompts)."""
    name = "llm_picker"

    def __init__(self, p2_driver: Any) -> None:
        self._p2 = p2_driver

    def _run(self, raw: str, col: str,
             candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        return self._p2._pick(raw, col, candidates)


class LLMVerifierSkill(Skill):
    """KEEP / REJECT a picked candidate (uses cache + Phase2Mesh._verify_pick)."""
    name = "llm_verifier"

    def __init__(self, p2_driver: Any) -> None:
        self._p2 = p2_driver

    def _run(self, raw: str, col: str, candidate: Dict[str, Any]) -> bool:
        return self._p2._verify_pick(raw, col, candidate)
