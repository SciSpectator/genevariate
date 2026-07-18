"""
GeneVariate — External metadata enrichment for the GSE extraction pipeline.

Purpose
-------
The existing GSE extraction (Step 1 / 1.5 in ``gse_worker.py``) reads free-text
metadata fields from GEOmetadb (``title``, ``source_name``, ``characteristics``)
and asks an LLM to extract Tissue / Condition / Treatment labels. The quality
of those labels is bounded by how well the submitter described the sample.

For many GEO samples, other public projects have already **curated** the same
samples into ontology-backed labels:

* **ARCHS4** (Ma'ayan Lab) — re-aligned & harmonised metadata for most GEO
  bulk RNA-seq samples (and some scRNA-seq). Contains a clean
  ``characteristics_ch1`` and ``source_name_ch1`` for every GSM.
* **CELLxGENE Discover Census** (CZI) — single-cell datasets ingested into
  CELLxGENE carry fully ontology-resolved ``tissue``, ``disease``,
  ``cell_type``, ``assay``, ``development_stage``, ``sex`` values.
* **Expression Atlas** (EMBL-EBI) — curated factor values per experiment.

This module provides a single ``ExternalEnricher`` that, given a
``(gsm, gse, raw)`` triple, queries the available external sources and
returns:

* a short **enrichment text block** (≤ ~500 chars) suitable to *prepend*
  to the Step-1 extractor prompt, so the LLM sees curated labels alongside
  the raw GEOmetadb text; and
* a list of **sibling label candidates** for each of Tissue/Condition —
  these are fed into Step-1.5's ``phase15_collapse`` as extra targets, so
  the deterministic collapser can snap LLM outputs onto curated names.

All values surfaced here are real curated measurements from public
resources. Nothing is fabricated or inferred by this module — it only
looks things up and forwards them.

Design notes
------------
* Every external source is consulted **lazily** and **per-GSE** (results
  are cached so that a full platform pass doesn't make one network call
  per GSM).
* Any network / import failure short-circuits silently and returns an
  empty enrichment — Step 1 still runs as before, so extraction is never
  worse than the GEOmetadb-only baseline.
* No external source can *override* the LLM's output — enrichment only
  adds material that the LLM or the Phase 1.5 collapser may choose to
  use.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_MAX_BLOCK_CHARS = 800   # keep the prompt augment short
_MAX_SIBLINGS = 25       # cap Step-1.5 candidate list size


# ────────────────────────────────────────────────────────────────────────────
# Result struct
# ────────────────────────────────────────────────────────────────────────────
class EnrichmentResult:
    """Bundle returned for a single GSM."""

    __slots__ = ("block", "tissue_candidates", "condition_candidates",
                 "treatment_candidates", "sources")

    def __init__(self):
        self.block: str = ""
        self.tissue_candidates: List[str] = []
        self.condition_candidates: List[str] = []
        self.treatment_candidates: List[str] = []
        self.sources: List[str] = []

    def is_empty(self) -> bool:
        return (not self.block
                and not self.tissue_candidates
                and not self.condition_candidates
                and not self.treatment_candidates)

    def __repr__(self) -> str:
        return (f"EnrichmentResult(sources={self.sources}, "
                f"block={self.block[:60]!r}, "
                f"tissue={self.tissue_candidates}, "
                f"condition={self.condition_candidates})")


# ────────────────────────────────────────────────────────────────────────────
# Per-source enrichers (each is independently optional)
# ────────────────────────────────────────────────────────────────────────────
class _ARCHS4Enricher:
    """Look up curated metadata for a GSM in the ARCHS4 matrix."""

    def __init__(self, organism: str = "human"):
        self.organism = organism
        self._client = None
        self._gse_cache: Dict[str, Any] = {}
        self._disabled = False
        self._lock = threading.Lock()

    def _get_client(self):
        if self._disabled:
            return None
        if self._client is None:
            try:
                from genevariate.sources.archs4 import ARCHS4Client
                self._client = ARCHS4Client(organism=self.organism)
            except Exception as exc:
                log.info("ARCHS4 enrichment disabled: %s", exc)
                self._disabled = True
                return None
        return self._client

    def enrich(self, gsm: str, gse: str, result: EnrichmentResult) -> None:
        client = self._get_client()
        if client is None:
            return
        try:
            rec = client.metadata_for_gsm(gsm)
        except Exception as exc:
            log.info("ARCHS4 enrichment failed for %s: %s", gsm, exc)
            return
        if not rec:
            return

        lines = []
        for field in ("source_name_ch1", "characteristics_ch1",
                       "extract_protocol_ch1", "title"):
            v = (rec.get(field) or "").strip()
            if v and v.lower() != "not available":
                lines.append(f"{field}: {v}")
        if lines:
            result.block += "[ARCHS4 harmonised metadata]\n" + "\n".join(lines) + "\n\n"
            result.sources.append("ARCHS4")

        # Parse ARCHS4 characteristics as sibling candidates for Phase 1.5
        char = (rec.get("characteristics_ch1") or "")
        for key, val in _split_characteristics(char):
            k = key.lower()
            if "tissue" in k or "organ" in k:
                _add_unique(result.tissue_candidates, val)
            elif "disease" in k or "condition" in k or "diagnosis" in k:
                _add_unique(result.condition_candidates, val)
            elif "treatment" in k or "drug" in k or "stimul" in k:
                _add_unique(result.treatment_candidates, val)


class _CensusEnricher:
    """Look up a GSE's dataset in CELLxGENE Census (if present).

    CELLxGENE Census does not index by GSE directly, but ingested scRNA
    submissions carry their CELLxGENE collection in ``obs``. This
    enricher performs a light match: does the Census know anything about
    this series?
    """

    def __init__(self):
        self._client = None
        self._disabled = False
        self._gse_cache: Dict[str, Optional[Dict[str, Any]]] = {}

    def _get_client(self):
        if self._disabled:
            return None
        if self._client is None:
            try:
                from genevariate.sources.cellxgene import CensusClient
                self._client = CensusClient()
            except Exception as exc:
                log.info("CELLxGENE enrichment disabled: %s", exc)
                self._disabled = True
                return None
        return self._client

    def enrich(self, gsm: str, gse: str, result: EnrichmentResult) -> None:
        if not gse:
            return
        client = self._get_client()
        if client is None:
            return
        if gse in self._gse_cache:
            cached = self._gse_cache[gse]
            if not cached:
                return
            self._apply(cached, result)
            return
        try:
            # Look up any cells whose dataset_id contains this GSE.
            census = client._open()
            exp = census["census_data"]["homo_sapiens"]
            tbl = exp.obs.read(
                column_names=["dataset_id", "tissue", "disease",
                               "cell_type", "assay", "development_stage"],
                value_filter=f"dataset_id == '{gse}'",
            ).concat().to_pandas()
        except Exception as exc:
            log.info("CELLxGENE lookup failed for %s: %s", gse, exc)
            self._gse_cache[gse] = None
            return
        if tbl is None or len(tbl) == 0:
            self._gse_cache[gse] = None
            return
        summary = {
            "n_cells":  int(len(tbl)),
            "tissues":  list(tbl["tissue"].astype(str).value_counts().head(5).index),
            "diseases": list(tbl["disease"].astype(str).value_counts().head(5).index),
            "cells":    list(tbl["cell_type"].astype(str).value_counts().head(5).index),
            "assays":   list(tbl["assay"].astype(str).value_counts().head(3).index),
        }
        self._gse_cache[gse] = summary
        self._apply(summary, result)

    def _apply(self, summary: Dict[str, Any], result: EnrichmentResult) -> None:
        lines = [f"[CELLxGENE Census — {summary['n_cells']:,} curated cells]"]
        if summary["tissues"]:
            lines.append("tissue: " + "; ".join(summary["tissues"]))
        if summary["diseases"]:
            lines.append("disease: " + "; ".join(summary["diseases"]))
        if summary["cells"]:
            lines.append("cell_type: " + "; ".join(summary["cells"]))
        if summary["assays"]:
            lines.append("assay: " + "; ".join(summary["assays"]))
        result.block += "\n".join(lines) + "\n\n"
        result.sources.append("CELLxGENE")
        for t in summary["tissues"]:
            _add_unique(result.tissue_candidates, t)
        for d in summary["diseases"]:
            _add_unique(result.condition_candidates, d)


class _AtlasEnricher:
    """Expression Atlas REST — factor values for a GSE experiment."""

    def __init__(self):
        self._gse_cache: Dict[str, Optional[Dict[str, Any]]] = {}
        self._disabled = False

    def enrich(self, gsm: str, gse: str, result: EnrichmentResult) -> None:
        if self._disabled or not gse:
            return
        if gse in self._gse_cache:
            cached = self._gse_cache[gse]
            if not cached:
                return
            self._apply(cached, result)
            return
        try:
            import requests  # type: ignore
        except Exception:
            self._disabled = True
            return
        try:
            r = requests.get(
                f"https://www.ebi.ac.uk/gxa/experiments/{gse}",
                headers={"Accept": "application/json"},
                timeout=4.0,
            )
            if r.status_code != 200:
                self._gse_cache[gse] = None
                return
            data = r.json()
        except Exception as exc:
            log.info("Expression Atlas lookup failed for %s: %s", gse, exc)
            self._gse_cache[gse] = None
            return
        fv = data.get("experiment", {}) if isinstance(data, dict) else {}
        factors = fv.get("experimentalFactors") or []
        if not factors:
            self._gse_cache[gse] = None
            return
        summary = {"factors": factors[:8]}
        self._gse_cache[gse] = summary
        self._apply(summary, result)

    def _apply(self, summary: Dict[str, Any], result: EnrichmentResult) -> None:
        factors = summary.get("factors") or []
        if not factors:
            return
        result.block += "[Expression Atlas factors]\n" + "; ".join(
            [str(f) for f in factors]) + "\n\n"
        result.sources.append("ExpressionAtlas")
        for f in factors:
            fl = str(f).lower()
            if "organism part" in fl or "tissue" in fl:
                _add_unique(result.tissue_candidates, str(f))
            elif "disease" in fl:
                _add_unique(result.condition_candidates, str(f))
            elif "compound" in fl or "treatment" in fl:
                _add_unique(result.treatment_candidates, str(f))


# ────────────────────────────────────────────────────────────────────────────
# Public façade
# ────────────────────────────────────────────────────────────────────────────
class ExternalEnricher:
    """Aggregates ARCHS4 + CELLxGENE + Expression Atlas per (gsm, gse).

    The enricher caches per-GSE results so that a platform-wide pass only
    fires one network call per series per source. Any single source
    failing is logged and ignored.
    """

    def __init__(
        self,
        *,
        enable_archs4: bool = True,
        enable_census: bool = True,
        enable_atlas: bool = False,  # off by default (slow EBI endpoint)
        organism: str = "human",
    ):
        self._archs4 = _ARCHS4Enricher(organism=organism) if enable_archs4 else None
        self._census = _CensusEnricher() if enable_census else None
        self._atlas = _AtlasEnricher() if enable_atlas else None

    def enrich(self, gsm: str, gse: str, raw: Optional[Dict[str, Any]] = None
               ) -> EnrichmentResult:
        result = EnrichmentResult()
        if not gsm and not gse:
            return result
        for enricher in (self._archs4, self._census, self._atlas):
            if enricher is None:
                continue
            try:
                enricher.enrich(gsm, gse, result)
            except Exception as exc:
                log.info("Enricher %s raised: %s",
                         enricher.__class__.__name__, exc)
        # Enforce size caps
        if len(result.block) > _MAX_BLOCK_CHARS:
            result.block = result.block[:_MAX_BLOCK_CHARS] + "…\n"
        result.tissue_candidates   = result.tissue_candidates[:_MAX_SIBLINGS]
        result.condition_candidates = result.condition_candidates[:_MAX_SIBLINGS]
        result.treatment_candidates = result.treatment_candidates[:_MAX_SIBLINGS]
        return result


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────
def _split_characteristics(char: str) -> List[Tuple[str, str]]:
    """Parse GEO characteristics fields like ``tissue: lung; age: 32``."""
    out: List[Tuple[str, str]] = []
    if not char:
        return out
    for chunk in str(char).replace("\n", ";").split(";"):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        k, v = chunk.split(":", 1)
        k = k.strip().lower()
        v = v.strip()
        if k and v:
            out.append((k, v))
    return out


def _add_unique(dst: List[str], val: str) -> None:
    if not val:
        return
    v = str(val).strip()
    if not v or v.lower() in {"not specified", "not available", "n/a", "na"}:
        return
    if v not in dst:
        dst.append(v)
