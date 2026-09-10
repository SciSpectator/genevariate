"""GeneVariate - modality-aware external curation for label extraction.

What this adds
--------------
The extractor reads a sample's own GEO free text (``title``,
``source_name_ch1``, ``characteristics_ch1``, ``treatment_protocol_ch1``,
``description``) and asks the model for Sex / Age / Tissue / Condition /
Treatment. The ceiling on that is how well the submitter described the sample.
For part of GEO, someone else has already curated the same study into
ontology-backed terms. This module fetches that curation and offers it to the
prompt as one extra field, so the model sees the curated vocabulary next to the
submitter's prose.

Why the source depends on the modality
--------------------------------------
The curating projects do not overlap, and each covers one kind of assay. The
split below is measured from the two indexes, not assumed:

* **Expression Atlas** (EMBL-EBI) publishes 4,562 experiments, 2,540 of them
  imported from GEO as ``E-GEOD-<n>``. Of those: 2,029 one-colour microarray,
  511 RNA-Seq mRNA, **0 single-cell**. So it is the source for microarray and
  bulk RNA-seq, and useless for single-cell.
* **CELLxGENE Discover** (CZI) publishes 388 collections, 225 of which link a
  ``GSE`` accession as their raw data. Every dataset carries fully
  ontology-resolved ``tissue`` (UBERON), ``disease`` (MONDO/PATO),
  ``cell_type`` (CL), ``sex`` (PATO) and ``development_stage`` (HsapDv). It
  covers single-cell and nothing else.
* **Methylation** and the peak-based assays (ChIP/ATAC/DNase/CUT&RUN/Hi-C)
  have no equivalent third-party curation keyed by GEO accession. They are
  reported as having no source rather than being handed a resource that does
  not describe them; the study-level GEO context that phase 1b can scrape
  (``geo_extract_driver.set_gse_scrape``) remains their only extra material.

Deliberately not used: **ARCHS4**. Its remote metadata call
(``archs4py.data.meta_remote``) reads every sample's metadata array over S3 and
then returns *expression counts*, not metadata - ``ARCHS4Client.metadata_for_gsm``
can never answer. Even if it did, ARCHS4 republishes GEO's own
``characteristics_ch1`` text, which the extractor already reads. It would add a
very expensive copy of the input, not curation.

Honesty of the block
--------------------
All of this curation is **study-level**: it says which values occur somewhere in
the study, not which one belongs to the sample in hand. The block says so in as
many words, and no value here can override the model - it is additional context
in the prompt, nothing more. Every lookup is cached per GSE, so a platform-wide
pass makes one request per study per source, and any failure degrades to an
empty enrichment rather than to a worse label.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

#: The five fields the extractor publishes; enrichment speaks their names so
#: the model does not have to map a curator's vocabulary onto them itself.
FIELDS: Tuple[str, ...] = ("Sex", "Age", "Tissue", "Condition", "Treatment")

#: Column the enrichment is published under, both on a sample table handed to
#: the pipeline and as a prompt field name.
ENRICHMENT_COLUMN = "external_curation"

_MAX_BLOCK_CHARS = 900
_MAX_VALUES_PER_FIELD = 6
_TIMEOUT = 20.0

_ATLAS_INDEX_URL = "https://www.ebi.ac.uk/gxa/json/experiments"
_ATLAS_EXPERIMENT_URL = "https://www.ebi.ac.uk/gxa/json/experiments/{acc}"
_CXG_COLLECTIONS_URL = "https://api.cellxgene.cziscience.com/curation/v1/collections"
_CXG_COLLECTION_URL = _CXG_COLLECTIONS_URL + "/{cid}"

_EMPTY_VALUES = {"", "n/a", "na", "none", "not available", "not specified",
                 "unknown", "not applicable"}


# ────────────────────────────────────────────────────────────────────────────
# Which source covers which modality
# ────────────────────────────────────────────────────────────────────────────
#: Technology category (``gpl_downloader.classify_technology``) -> source names.
SOURCES_BY_CATEGORY: Dict[str, Tuple[str, ...]] = {
    "microarray":       ("expression-atlas",),
    "bulk-rna-seq":     ("expression-atlas",),
    "single-cell":      ("cellxgene",),
    "methylation":      (),
    "sequencing-other": (),
    "other":            ("expression-atlas",),
}

#: Why a category has no source, said out loud instead of returning a silent
#: empty result.
NO_SOURCE_REASON: Dict[str, str] = {
    "methylation": "no third-party project curates methylation series by GEO "
                   "accession; Expression Atlas and CELLxGENE are expression only",
    "sequencing-other": "peak-based assays (ChIP/ATAC/DNase/CUT&RUN/Hi-C) are "
                        "not curated per GEO accession by any of these projects",
}


def sources_for(category: str) -> Tuple[str, ...]:
    """Source names that actually cover ``category``; empty tuple if none do."""
    return SOURCES_BY_CATEGORY.get(str(category or "").strip().lower(),
                                   ("expression-atlas",))


# ────────────────────────────────────────────────────────────────────────────
# Result
# ────────────────────────────────────────────────────────────────────────────
class EnrichmentResult:
    """Curated study-level material found for one sample."""

    __slots__ = ("candidates", "context", "sources", "note")

    def __init__(self) -> None:
        self.candidates: Dict[str, List[str]] = {f: [] for f in FIELDS}
        self.context: List[str] = []
        self.sources: List[str] = []
        self.note: str = ""

    def add(self, field: str, value: Any) -> None:
        if field not in self.candidates:
            return
        text = _clean(value)
        if not text:
            return
        bucket = self.candidates[field]
        if text not in bucket and len(bucket) < _MAX_VALUES_PER_FIELD:
            bucket.append(text)

    def is_empty(self) -> bool:
        return not any(self.candidates.values()) and not self.context

    @property
    def block(self) -> str:
        """The text offered to the prompt, or ``""`` when nothing was found."""
        if self.is_empty():
            return ""
        head = (f"[Curated by {', '.join(self.sources)} - values recorded "
                f"anywhere in this study, not necessarily in this sample]")
        lines = [head]
        for field in FIELDS:
            values = self.candidates[field]
            if values:
                lines.append(f"{field}: " + " | ".join(values))
        lines.extend(self.context)
        text = "\n".join(lines)
        if len(text) > _MAX_BLOCK_CHARS:
            text = text[:_MAX_BLOCK_CHARS].rstrip() + " …"
        return text

    def __repr__(self) -> str:
        return (f"EnrichmentResult(sources={self.sources}, "
                f"candidates={ {k: v for k, v in self.candidates.items() if v} })")


# ────────────────────────────────────────────────────────────────────────────
# Sources
# ────────────────────────────────────────────────────────────────────────────
class _HttpSource:
    """Shared plumbing: one index fetch, then one fetch per study, both cached."""

    name = ""

    def __init__(self, timeout: float = _TIMEOUT, session: Any = None):
        self.timeout = timeout
        self._session = session
        self._index: Optional[Dict[str, Any]] = None
        self._studies: Dict[str, Optional[Dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def _get(self, url: str) -> Optional[Any]:
        try:
            if self._session is not None:
                response = self._session.get(url, timeout=self.timeout)
            else:
                import requests
                response = requests.get(url, timeout=self.timeout)
            if response.status_code != 200:
                return None
            return response.json()
        except Exception as exc:                       # network, JSON, import
            log.info("%s: %s failed: %s", self.name, url, exc)
            return None

    def index(self) -> Dict[str, Any]:
        """``{GSE: handle}`` for every study this source has, fetched once."""
        with self._lock:
            if self._index is None:
                self._index = self._build_index() or {}
            return self._index

    def study(self, gse: str) -> Optional[Dict[str, Any]]:
        """Curated summary for one series, or ``None`` if it is not covered."""
        key = str(gse or "").strip().upper()
        if not key:
            return None
        handle = self.index().get(key)
        if handle is None:
            return None
        with self._lock:
            if key not in self._studies:
                self._studies[key] = self._fetch_study(handle)
            return self._studies[key]

    def _build_index(self) -> Dict[str, Any]:
        raise NotImplementedError

    def _fetch_study(self, handle: Any) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def apply(self, summary: Dict[str, Any], result: EnrichmentResult) -> None:
        raise NotImplementedError


# ── Expression Atlas - microarray and bulk RNA-seq ───────────────────────
#: Curator property names -> extractor field. Single words match a whole word
#: of the property name, multi-word keys match as a substring, so "dosage" is
#: not read as "age" and "organism part" is not read as "organism".
_ATLAS_PROPERTY_FIELDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("Sex",       ("sex", "gender")),
    ("Age",       ("age", "developmental stage", "development stage")),
    ("Tissue",    ("organism part", "tissue", "cell type", "cell line",
                   "sampling site", "organism region")),
    ("Condition", ("disease", "phenotype", "diagnosis", "genotype",
                   "disease staging", "clinical information")),
    ("Treatment", ("compound", "treatment", "dose", "irradiate", "infect",
                   "stimulus", "growth condition")),
)


class _ExpressionAtlasSource(_HttpSource):
    """EMBL-EBI Expression Atlas: curated experiment design per GEO series."""

    name = "Expression Atlas"

    def _build_index(self) -> Dict[str, Any]:
        payload = self._get(_ATLAS_INDEX_URL)
        experiments = (payload or {}).get("experiments") or []
        index: Dict[str, Any] = {}
        for entry in experiments:
            accession = str(entry.get("experimentAccession") or "")
            if not accession.startswith("E-GEOD-"):
                continue
            index["GSE" + accession[len("E-GEOD-"):]] = entry
        log.info("Expression Atlas index: %d GEO series", len(index))
        return index

    def _fetch_study(self, handle: Any) -> Optional[Dict[str, Any]]:
        accession = str(handle.get("experimentAccession") or "")
        payload = self._get(_ATLAS_EXPERIMENT_URL.format(acc=accession))
        if not isinstance(payload, dict):
            return None
        # Baseline experiments describe assay groups, differential ones describe
        # contrasts, and the properties hang off a different key in each.
        properties: List[Dict[str, Any]] = []
        for header in payload.get("columnHeaders") or []:
            for owner in ("assayGroupSummary", "contrastSummary"):
                summary = header.get(owner) or {}
                properties.extend(summary.get("properties") or [])
        return {
            "accession": accession,
            "properties": properties,
            "technology": handle.get("technologyType") or [],
            "species": handle.get("species") or "",
        }

    def apply(self, summary: Dict[str, Any], result: EnrichmentResult) -> None:
        for prop in summary.get("properties") or []:
            name = str(prop.get("propertyName") or "").strip().lower()
            field = _atlas_field(name)
            if not field:
                continue
            result.add(field, prop.get("testValue"))
            result.add(field, prop.get("referenceValue"))
        technology = "; ".join(str(t) for t in summary.get("technology") or [])
        if technology:
            result.context.append(f"assay: {technology}")


def _atlas_field(property_name: str) -> str:
    words = set(re.split(r"[^a-z0-9]+", property_name))
    for field, keys in _ATLAS_PROPERTY_FIELDS:
        for key in keys:
            if (key in words) if " " not in key else (key in property_name):
                return field
    return ""


# ── CELLxGENE Discover - single-cell ─────────────────────────────────────
#: Dataset key -> extractor field. ``cell_type`` and ``assay`` are context, not
#: candidates: a cell type is not the sample's tissue, and naming it as one
#: would invite the model to answer the wrong question.
_CXG_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("tissue", "Tissue"),
    ("disease", "Condition"),
    ("sex", "Sex"),
    ("development_stage", "Age"),
)
_GSE_IN_URL = re.compile(r"GSE\d+", re.IGNORECASE)


class _CellxGeneSource(_HttpSource):
    """CZI CELLxGENE Discover: ontology-resolved obs terms per collection."""

    name = "CELLxGENE Discover"

    def _build_index(self) -> Dict[str, Any]:
        collections = self._get(_CXG_COLLECTIONS_URL)
        index: Dict[str, Any] = {}
        for collection in collections or []:
            cid = collection.get("collection_id")
            if not cid:
                continue
            for link in collection.get("links") or []:
                for accession in _GSE_IN_URL.findall(str(link.get("link_url") or "")):
                    index.setdefault(accession.upper(), cid)
        log.info("CELLxGENE index: %d GEO series", len(index))
        return index

    def _fetch_study(self, handle: Any) -> Optional[Dict[str, Any]]:
        payload = self._get(_CXG_COLLECTION_URL.format(cid=handle))
        if not isinstance(payload, dict):
            return None
        terms: Dict[str, List[str]] = {}
        cells = 0
        for dataset in payload.get("datasets") or []:
            cells += int(dataset.get("cell_count") or 0)
            for key in ("tissue", "disease", "sex", "development_stage",
                        "cell_type", "assay"):
                for term in dataset.get(key) or []:
                    label = _clean(term.get("label") if isinstance(term, dict)
                                   else term)
                    if label and label not in terms.setdefault(key, []):
                        terms[key].append(label)
        if not terms:
            return None
        return {"collection_id": handle, "terms": terms, "cells": cells}

    def apply(self, summary: Dict[str, Any], result: EnrichmentResult) -> None:
        terms = summary.get("terms") or {}
        for key, field in _CXG_FIELDS:
            for label in terms.get(key) or []:
                result.add(field, label)
        cell_types = terms.get("cell_type") or []
        if cell_types:
            result.context.append("cell types present: "
                                  + "; ".join(cell_types[:6]))
        assays = terms.get("assay") or []
        if assays:
            result.context.append("assay: " + "; ".join(assays[:3]))


_SOURCE_CLASSES = {
    "expression-atlas": _ExpressionAtlasSource,
    "cellxgene": _CellxGeneSource,
}


# ────────────────────────────────────────────────────────────────────────────
# Façade
# ────────────────────────────────────────────────────────────────────────────
class ExternalEnricher:
    """Curated study-level context for one technology category.

    ``category`` is a ``gpl_downloader`` technology category; it selects the
    sources that actually cover that assay (see :data:`SOURCES_BY_CATEGORY`).
    ``sources`` overrides the selection with ready-made source objects, which is
    how the tests run without a network.
    """

    def __init__(self, category: str = "microarray", *,
                 sources: Optional[Sequence[Any]] = None,
                 timeout: float = _TIMEOUT, session: Any = None):
        self.category = str(category or "").strip().lower()
        if sources is not None:
            self.sources = list(sources)
            self.note = ""
        else:
            names = sources_for(self.category)
            self.sources = [_SOURCE_CLASSES[n](timeout=timeout, session=session)
                            for n in names if n in _SOURCE_CLASSES]
            self.note = "" if self.sources else NO_SOURCE_REASON.get(
                self.category, "no curated external source covers this category")

    def enrich(self, gsm: str = "", gse: str = "") -> EnrichmentResult:
        """Curated material for the study ``gse``; ``gsm`` is accepted for
        symmetry with the extractor's per-sample call but is not looked up -
        none of these projects publishes per-GSM curation."""
        result = EnrichmentResult()
        result.note = self.note
        if not gse:
            return result
        for source in self.sources:
            try:
                summary = source.study(gse)
            except Exception as exc:
                log.info("%s raised for %s: %s",
                         getattr(source, "name", source), gse, exc)
                continue
            if not summary:
                continue
            try:
                source.apply(summary, result)
            except Exception as exc:
                log.info("%s could not render %s: %s",
                         getattr(source, "name", source), gse, exc)
                continue
            name = getattr(source, "name", "")
            if name and name not in result.sources:
                result.sources.append(name)
        return result


def annotate_samples(df, category: str, *, log_func=None,
                     enricher: Optional[ExternalEnricher] = None) -> List[str]:
    """Add the enrichment column to a sample table; return the columns added.

    ``df`` is modified in place. The returned list is what the caller appends to
    the extractor's metadata-column selection, and it is empty when nothing was
    found - an empty column is worse than no column, because it would spend
    prompt budget on a field that is always blank.
    """
    def say(message: str) -> None:
        if log_func:
            log_func(message)

    series_column = next((c for c in ("series_id", "gse", "GSE", "Series")
                          if c in getattr(df, "columns", [])), "")
    if not series_column:
        say("[Enrich] No series column on the sample table - skipped.")
        return []

    worker = enricher or ExternalEnricher(category)
    if not worker.sources:
        say(f"[Enrich] {category}: {worker.note}.")
        return []

    blocks: Dict[str, str] = {}
    for gse in sorted({str(v).strip().upper() for v in df[series_column]
                       if str(v).strip()}):
        blocks[gse] = worker.enrich(gse=gse).block
    hits = sum(1 for b in blocks.values() if b)
    if not hits:
        say(f"[Enrich] {category}: none of the {len(blocks)} series are in "
            f"{', '.join(s.name for s in worker.sources)}.")
        return []

    df[ENRICHMENT_COLUMN] = [
        blocks.get(str(v).strip().upper(), "") for v in df[series_column]]
    covered = int((df[ENRICHMENT_COLUMN] != "").sum())
    say(f"[Enrich] {category}: curated context for {hits}/{len(blocks)} series "
        f"({covered:,} samples) from "
        f"{', '.join(s.name for s in worker.sources)}.")
    return [ENRICHMENT_COLUMN]


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────
def _clean(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    text = re.sub(r"\s+", " ", text)
    if text.lower() in _EMPTY_VALUES:
        return ""
    return text


__all__ = [
    "FIELDS", "ENRICHMENT_COLUMN", "SOURCES_BY_CATEGORY", "NO_SOURCE_REASON",
    "sources_for", "EnrichmentResult", "ExternalEnricher", "annotate_samples",
]
