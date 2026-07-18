"""Phase 1c — context-normalisation sibling-consensus curator (no LLM).

Reads from :pyclass:`gse_context_cache.GSEContextCache`. Pure SQL +
deterministic rules. Replaces the heavyweight BioLORD / n-gram /
JS-divergence path of ``phase1c_semantic.py`` for the common case.

Rules (all universal, no entity hardcoding):

    R1. KEY-VALUE LEAK
        If value matches ``<word>(\\.<word>)*\\s*:\\s*\\S+`` and looks
        like a substring of the GSM's raw metadata blob (title,
        source_name, characteristics, treatment_protocol, description),
        demote to NS.

    R2. SIBLING NS-CONSENSUS DEMOTION
        If ≥THRESHOLD of GSE siblings (same field) are NS AND the value
        is NOT a substring of this GSM's own raw metadata blob,
        demote to NS.

    R3. SIBLING VALUE-CONSENSUS PROMOTION
        If value is NS AND ≥THRESHOLD of GSE siblings agree on a single
        non-NS value, promote the NS to that majority value.

    R4. STRUCTURAL-TWIN VETO
        If R2 / R3 fires but THIS GSM has a different
        ``characteristics_ch1`` key set than the dominant siblings,
        downgrade the action from ``demote/promote`` to ``flag-only``
        (keep value, log discrepancy).

The curator does NOT call an LLM. It is intended as a fast in-pipeline
gate; ``phase1c_semantic.py`` remains available for the harder cases
(parser-marker leaks, composite-split, name-vs-value).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from gse_context_cache import GSEContextCache, LABEL_COLS, NS, _is_ns

# Matches ``foo: 12.3``  or  ``foo.bar.baz: blah-2``  -- a raw key:value
# substring (no spaces in the key, dots allowed in the key).
_KV_LEAK_RE = re.compile(r"^[A-Za-z][\w.]*\s*:\s*\S+", re.UNICODE)

# Universal whitespace + delimiter normaliser. Collapses any run of
# whitespace OR common delimiters (`; , | / \ tab newline`) to a single
# space so leak-detection is robust to surface variation between the
# raw blob (often tab-separated) and the model's parsed output (often
# semicolon-separated). No entity-specific tokens.
_NORM_RE = re.compile(r"[\s;,|/\\]+")


def _norm_for_blob_check(s: str) -> str:
    return _NORM_RE.sub(" ", (s or "").lower()).strip()


class ConsensusCurator:
    """Stateless Phase 1c — purely deterministic, cache-backed."""

    def __init__(self, cache: GSEContextCache,
                 threshold: float = 0.80,
                 require_struct_twin: bool = True):
        self.cache = cache
        self.threshold = threshold
        self.require_struct_twin = require_struct_twin

    # ── per-sample ────────────────────────────────────────────────────────
    def curate_sample(self, gse: str, gsm: str,
                      values: Dict[str, str]) -> Dict[str, dict]:
        """Returns ``{field: {final_value, action, reason}}`` for one GSM.

        ``values`` is the Phase 1b output ``{Tissue, Condition, Treatment}``.

        Pure function — does NOT write the verdict to the cache. The
        caller is responsible for the final ``upsert_phase_value("p1c")``
        once any post-processing (e.g. semantic LLM curator) has run.
        """
        out: Dict[str, dict] = {}
        ctx = self.cache.get_context(gse, gsm)
        gsm_blob_lc = ""
        if ctx:
            gsm_blob_lc = " | ".join(filter(None, [
                ctx.get("title"), ctx.get("source_name"),
                ctx.get("characteristics"),
                ctx.get("treatment_protocol"),
                ctx.get("description")])).lower()
        # Pre-normalise once: delimiter-collapsed blob for the R1
        # substring check. Cheap and stable across samples in this GSE.
        gsm_blob_norm = _norm_for_blob_check(gsm_blob_lc)

        for field in LABEL_COLS:
            v = values.get(field, NS)
            out[field] = self._decide(gse, gsm, field, v,
                                      gsm_blob_lc, gsm_blob_norm, ctx)
        return out

    # ── per-GSE ───────────────────────────────────────────────────────────
    def curate_gse(self, gse: str, samples: List[dict]) -> Dict[str, Dict[str, dict]]:
        """Iterate over a GSE worth of samples.

        ``samples`` items must have ``gsm`` and ``phase1b`` (or ``phase1``)
        dicts. Returns ``{gsm: {field: decision}}``.
        """
        out: Dict[str, Dict[str, dict]] = {}
        for s in samples:
            gsm = s.get("gsm")
            if not gsm:
                continue
            p1b = s.get("phase1b") or s.get("phase1") or {}
            out[gsm] = self.curate_sample(gse, gsm, p1b)
        return out

    # ── decision core ─────────────────────────────────────────────────────
    def _decide(self, gse: str, gsm: str, field: str,
                value: str, gsm_blob_lc: str, gsm_blob_norm: str,
                ctx: Optional[dict]) -> dict:
        v = (value or "").strip()
        # R1: key:value leak — value parsed straight from the raw blob,
        # not a real label. Check the full blob (title, source,
        # characteristics, treatment_protocol, description), not just
        # characteristics. Substring check runs on a normalised view of
        # both sides (whitespace + ;,|/\ collapsed to single space) so
        # tab-vs-semicolon mismatches between raw metadata and model
        # output don't defeat detection.
        if not _is_ns(v) and _KV_LEAK_RE.match(v):
            v_norm = _norm_for_blob_check(v)
            if (gsm_blob_norm and v_norm and v_norm in gsm_blob_norm) \
                    or (gsm_blob_lc and v.lower() in gsm_blob_lc):
                return {"final_value": NS,
                        "action": "demote_to_NS",
                        "reason": "key_value_leak_in_raw_blob",
                        "from": value, "support": 1.0}

        # R2 / R3: sibling consensus
        verdict = self.cache.consensus_verdict(
            gse, gsm, field, v, threshold=self.threshold)

        if verdict["action"] == "demote_to_NS":
            # R4 structural-twin veto
            if self.require_struct_twin and verdict["struct_twin_count"] < 3:
                return {"final_value": v or NS,
                        "action": "flag_only",
                        "reason": ("would_demote_but_no_struct_twin"
                                   f" twins={verdict['struct_twin_count']}"),
                        "from": value, "support": verdict["support"]}
            return {"final_value": NS,
                    "action": "demote_to_NS",
                    "reason": verdict["reason"],
                    "from": value, "support": verdict["support"]}

        if verdict["action"] == "promote_to_majority":
            if self.require_struct_twin and verdict["struct_twin_count"] < 3:
                return {"final_value": NS,
                        "action": "flag_only",
                        "reason": ("would_promote_but_no_struct_twin"
                                   f" twins={verdict['struct_twin_count']}"),
                        "from": value,
                        "support": verdict["support"]}
            return {"final_value": verdict["majority_value"],
                    "action": "promote_to_majority",
                    "reason": verdict["reason"],
                    "from": value, "support": verdict["support"]}

        return {"final_value": v or NS,
                "action": "keep",
                "reason": verdict["reason"],
                "from": value, "support": verdict["support"]}
