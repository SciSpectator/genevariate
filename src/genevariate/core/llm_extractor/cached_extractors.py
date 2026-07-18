"""Episodic-memory wrappers for Phase 1 / Phase 1b.

CachedPhase1Agent / CachedPhase1bAgent are drop-in replacements for the
underlying ``Phase1Agent`` / ``Phase1bAgent`` whose ``extract`` /
``infer_sample`` methods first consult :pyclass:`gse_context_cache.
GSEContextCache` and only call the LLM on a miss. Cache hits are O(1)
SQL reads; misses go through the wrapped agent and are persisted.

Cache keys include both the model name (``EXTRACTION_MODEL``) and the
prompt-version of the underlying phase, so an upgrade to either
invalidates the affected rows lazily (a stale row is only ignored, not
deleted).

Universal — no entity hardcoding.
"""
from __future__ import annotations

import hashlib
import os
from collections import Counter
from typing import Dict, Optional

from gse_context_cache import GSEContextCache, LABEL_COLS, NS

# Per-phase prompt-version sentinels. Bump any of these when the
# underlying prompt changes; old cached rows are then ignored.
PHASE1_PROMPT_VERSION  = os.environ.get("PHASE1_PROMPT_VERSION",  "v5-protocol-polarity-2026-05-08")
PHASE1B_PROMPT_VERSION = os.environ.get("PHASE1B_PROMPT_VERSION", "v10-treatment-class1-class2-2026-05-08")
DEFAULT_MODEL          = os.environ.get("PHASE1_MODEL", "gemma4-e2b-text:latest")


def _counter_to_dict(c) -> Dict[str, int]:
    """Stable dict from Counter or {value: count} mapping."""
    if c is None:
        return {}
    if isinstance(c, Counter):
        return dict(c)
    if isinstance(c, dict):
        return dict(c)
    return {}


class CachedPhase1Agent:
    """Wraps any agent exposing ``.extract(raw)`` and ``.extract_field``.

    The wrapped object should set ``model_version`` / ``prompt_version``
    on construction; if not provided, defaults are used.
    """

    def __init__(self, base, cache: GSEContextCache,
                 *,
                 model_version: str = DEFAULT_MODEL,
                 prompt_version: str = PHASE1_PROMPT_VERSION,
                 gse_resolver=None):
        """``gse_resolver(gsm) -> gse_id`` is optional. When provided,
        the cached extraction is also persisted into the per-(GSE, GSM)
        context table for downstream Phase 1c consensus."""
        self.base = base
        self.cache = cache
        self.model_version = model_version
        self.prompt_version = prompt_version
        self.gse_resolver = gse_resolver
        # Counters for visibility
        self.hits = 0
        self.misses = 0

    def extract(self, raw: Dict[str, str], gsm: Optional[str] = None,
                gse: Optional[str] = None,
                force_refresh: bool = False) -> Dict[str, str]:
        """Returns ``{Tissue, Condition, Treatment}`` for ``raw``.

        Episodic-cached per (raw_hash, field, model, prompt). Per-field
        granularity so a single missing field doesn't invalidate the
        other two.

        ``force_refresh=True`` bypasses the cache read (always calls the
        underlying agent) but still writes the fresh result back. Useful
        for benchmarking and for re-runs after a prompt change that
        wasn't reflected in ``prompt_version``.
        """
        h = self.cache.hash_raw(raw)
        out: Dict[str, str] = {}
        missing_fields = []
        if force_refresh:
            missing_fields = list(LABEL_COLS)
        else:
            for f in LABEL_COLS:
                v = self.cache.get_phase1_episodic(
                    h, f, self.model_version, self.prompt_version)
                if v is None:
                    missing_fields.append(f)
                else:
                    out[f] = v
                    self.hits += 1

        if missing_fields:
            # Could be partial-recovery; cheapest is to call full extract.
            fresh = self.base.extract(raw)
            for f in missing_fields:
                v = fresh.get(f, NS)
                self.cache.set_phase1_episodic(
                    h, f, v, self.model_version, self.prompt_version)
                out[f] = v
                self.misses += 1

        # Side-effects: write context + p1 values into the GSE/GSM cache.
        if gsm:
            resolved_gse = gse or (self.gse_resolver(gsm) if self.gse_resolver else None)
            if resolved_gse:
                self.cache.upsert_context(resolved_gse, gsm, raw)
                for f in LABEL_COLS:
                    self.cache.upsert_phase_value(
                        resolved_gse, gsm, f, "p1", out.get(f, NS))
        return out


class CachedPhase1bAgent:
    """Wraps :pyclass:`phase1b.Phase1bAgent`.

    ``infer_sample(gsm, raw, labels, gse_id, gse_ctx, sibling_dist)`` is
    cached on (gsm, hash(gse_ctx + sibling_dist), field). Per-field
    granularity again.
    """

    def __init__(self, base, cache: GSEContextCache,
                 *,
                 model_version: str = DEFAULT_MODEL,
                 prompt_version: str = PHASE1B_PROMPT_VERSION):
        self.base = base
        self.cache = cache
        self.model_version = model_version
        self.prompt_version = prompt_version
        self.hits = 0
        self.misses = 0

    def infer_sample(self, gsm: str, raw: dict,
                     labels: Dict[str, str],
                     gse_id: str, gse_ctx: dict,
                     sibling_dist=None,
                     force_refresh: bool = False) -> Dict[str, str]:
        sib_dict = {f: _counter_to_dict((sibling_dist or {}).get(f, {}))
                    for f in LABEL_COLS}
        # Include this GSM's Phase-1 labels in the cache key. The Tissue
        # branch reads them as cross-field signal (v5 fallback), so a
        # change to this GSM's Phase-1 Condition/Treatment must
        # invalidate the cached Phase-1b row for this gsm.
        gse_hash = self.cache.hash_gse_state(gse_ctx, sib_dict, labels)

        out: Dict[str, str] = {}
        missing = []
        if force_refresh:
            missing = list(LABEL_COLS)
        else:
            for f in LABEL_COLS:
                cached = self.cache.get_phase1b_episodic(
                    gsm, gse_hash, f, self.model_version, self.prompt_version)
                if cached is None:
                    missing.append(f)
                else:
                    out[f] = cached
                    self.hits += 1

        if missing:
            fresh = self.base.infer_sample(
                gsm, raw, labels, gse_id, gse_ctx, sibling_dist=sibling_dist)
            for f in missing:
                v = fresh.get(f, labels.get(f, NS))
                self.cache.set_phase1b_episodic(
                    gsm, gse_hash, f, labels.get(f, NS), v,
                    self.model_version, self.prompt_version)
                out[f] = v
                self.misses += 1

        # Persist into the per-(GSE, GSM) results table.
        for f in LABEL_COLS:
            self.cache.upsert_phase_value(gse_id, gsm, f, "p1b",
                                          out.get(f, NS))
        return out

    @property
    def stats(self) -> Dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}
