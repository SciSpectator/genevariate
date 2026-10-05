"""GeneVariate - cross-database experiment discovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set

import pandas as pd  # type: ignore


LogFn = Callable[[str], None]


def _noop(_msg: str) -> None:
    return


# ────────────────────────────────────────────────────────────────────────────
# Result record
# ────────────────────────────────────────────────────────────────────────────
@dataclass
class DiscoveryHit:
    source: str           # "ARCHS4" | "CELLxGENE" | "Expression Atlas"
    accession: str        # dataset/series identifier in the source
    title: str = ""
    summary: str = ""
    organism: str = ""
    platform: str = ""    # or assay for scRNA-seq
    n_samples: int = 0
    url: str = ""
    matched: List[str] = field(default_factory=list)


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────
def _normalize_keywords(raw: str | Iterable[str]) -> List[str]:
    """Split a comma-string or iterable of keywords into lowercase tokens."""
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",")]
    else:
        parts = [str(p).strip() for p in raw]
    return [p.lower() for p in parts if p]


def _match_tokens(text: str, tokens: Sequence[str]) -> List[str]:
    t = (text or "").lower()
    return [tok for tok in tokens if tok in t]


# ────────────────────────────────────────────────────────────────────────────
# ARCHS4
# ────────────────────────────────────────────────────────────────────────────
_ARCHS4_META_CACHE: Dict[str, Dict[str, list]] = {}
_ARCHS4_META_FIELDS = (
    "geo_accession", "series_id", "title",
    "source_name_ch1", "characteristics_ch1",
    "library_strategy", "instrument_model",
)


def _load_archs4_meta(organism: str, log: LogFn = _noop):
    """Populate and return the cached metadata columns for ``organism``."""
    if organism in _ARCHS4_META_CACHE:
        return _ARCHS4_META_CACHE[organism]

    try:
        from genevariate.sources.archs4 import ARCHS4Client
        import archs4py as a4  # type: ignore
    except Exception as exc:
        log(f"[ARCHS4] unavailable: {exc}")
        return None

    try:
        client = ARCHS4Client(organism=organism)
        s3_url, endpoint = a4.data.resolve_url(client.url)
    except Exception as exc:
        log(f"[ARCHS4] URL resolution failed: {exc}")
        return None

    out: Dict[str, list] = {}
    for field in _ARCHS4_META_FIELDS:
        field_path = f"meta/samples/{field}"
        log(f"[ARCHS4] caching {field} column (remote HDF5)…")
        try:
            arr = a4.data.fetch_meta_remote(field_path, s3_url, endpoint)
            out[field] = [str(x) for x in list(arr)]
        except Exception as exc:
            log(f"[ARCHS4] field {field!r} read failed: {exc}")
            out[field] = []

    _ARCHS4_META_CACHE[organism] = out
    n = len(out.get("geo_accession") or [])
    log(f"[ARCHS4] metadata cached: {n:,} samples across "
        f"{sum(1 for v in out.values() if v)} fields")
    return out


def _limit_hits(
    hits: List["DiscoveryHit"],
    limit: Optional[int],
    source: str,
    log: LogFn,
) -> List["DiscoveryHit"]:
    """Apply the user's result limit, and say so when it removes anything."""
    if limit is None or len(hits) <= limit:
        return hits
    log(f"[{source}] {len(hits)} matched, {limit} kept: the result is "
        f"limited by 'Experiments each source returns in Discover "
        f"Experiments', which is cleared from Display limits...")
    return hits[:limit]


def search_archs4(
    keywords: str | Iterable[str],
    *,
    organism: str = "human",
    max_results: Optional[int] = None,
    log: LogFn = _noop,
) -> List[DiscoveryHit]:
    """Keyword-search ARCHS4 harmonised metadata (remote, no 30 GB download)."""
    tokens = _normalize_keywords(keywords)
    if not tokens:
        return []

    meta = _load_archs4_meta(organism, log=log)
    if not meta:
        return []

    titles   = meta.get("title") or []
    chars    = meta.get("characteristics_ch1") or []
    sources  = meta.get("source_name_ch1") or []
    series   = meta.get("series_id") or []
    samples  = meta.get("geo_accession") or []

    n = len(samples) or len(titles)
    if not n:
        log("[ARCHS4] metadata empty - skipping")
        return []

    strategy = meta.get("library_strategy") or []
    instrument = meta.get("instrument_model") or []

    def _col(name, i):
        col = {"library_strategy": strategy,
               "instrument_model": instrument}.get(name) or []
        return str(col[i]).strip() if i < len(col) else ""

    hits: Dict[str, DiscoveryHit] = {}
    for i in range(n):
        t = titles[i] if i < len(titles) else ""
        c = chars[i]  if i < len(chars)  else ""
        s = sources[i] if i < len(sources) else ""
        blob_l = (t + " " + c + " " + s).lower()
        matched = [tok for tok in tokens if tok in blob_l]
        if not matched:
            continue
        sid_raw = (series[i] if i < len(series) else "") or ""
        if not sid_raw:
            continue
        for sid in [x.strip() for x in str(sid_raw).split(",") if x.strip()]:
            hit = hits.get(sid)
            if hit is None:
                hits[sid] = DiscoveryHit(
                    source="ARCHS4",
                    accession=sid,
                    title=t,
                    summary=c or s,
                    organism=organism,
                    n_samples=1,
                    platform=", ".join(
                        x for x in (_col("library_strategy", i),
                                    _col("instrument_model", i)) if x),
                    url=f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={sid}",
                    matched=sorted(set(matched)),
                )
            else:
                hit.n_samples += 1
                hit.matched = sorted(set(hit.matched) | set(matched))

    log(f"[ARCHS4] {len(hits)} experiment(s) matched")
    return _limit_hits(list(hits.values()), max_results, "ARCHS4", log)


# ────────────────────────────────────────────────────────────────────────────
# CELLxGENE Discover Census
# ────────────────────────────────────────────────────────────────────────────
_CXG_CURATION_CACHE: Dict[str, object] = {"data": None, "ts": 0.0}
_CXG_CURATION_TTL_SECONDS = 60 * 30  # 30 minutes


class DiscoverySourceError(RuntimeError):
    """A source could not be reached, or answered with something unusable."""


def _fetch_cxg_curation(log: LogFn = _noop) -> List[Dict]:
    """Return the CELLxGENE Discover curation-API dataset list (cached)."""
    import time
    now = time.time()
    cached = _CXG_CURATION_CACHE.get("data")
    ts = float(_CXG_CURATION_CACHE.get("ts") or 0.0)
    if cached is not None and (now - ts) < _CXG_CURATION_TTL_SECONDS:
        return cached  # type: ignore[return-value]

    import urllib.request
    import json
    req = urllib.request.Request(
        "https://api.cellxgene.cziscience.com/curation/v1/datasets",
        headers={"Accept": "application/json",
                  "User-Agent": "GeneVariate/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        log(f"[CELLxGENE] curation API fetch failed: {exc}")
        raise DiscoverySourceError(f"CELLxGENE could not be reached: {exc}") from exc
    if not isinstance(data, list):
        log("[CELLxGENE] unexpected curation API response shape")
        raise DiscoverySourceError(
            "CELLxGENE answered with an unexpected response shape")
    _CXG_CURATION_CACHE["data"] = data
    _CXG_CURATION_CACHE["ts"] = now
    return data


def _cxg_labels(field) -> List[str]:
    """Flatten a curation-API facet field (list of {label,...} or scalar) to labels."""
    if field is None:
        return []
    if isinstance(field, list):
        out = []
        for item in field:
            if isinstance(item, dict):
                lab = item.get("label")
                if lab:
                    out.append(str(lab))
            elif item:
                out.append(str(item))
        return out
    if isinstance(field, dict):
        lab = field.get("label")
        return [str(lab)] if lab else []
    return [str(field)]


def search_cellxgene(
    keywords: str | Iterable[str],
    *,
    organism: str = "homo_sapiens",
    tissue: str = "",
    disease: str = "",
    assay: str = "",
    max_results: Optional[int] = None,
    log: LogFn = _noop,
) -> List[DiscoveryHit]:
    """Keyword-search CELLxGENE Discover datasets."""
    tokens = _normalize_keywords(keywords)
    if not tokens:
        return []

    data = _fetch_cxg_curation(log=log)
    if not data:
        return []

    # Normalize organism
    org_map = {
        "human": "Homo sapiens",
        "homo_sapiens": "Homo sapiens",
        "homo sapiens": "Homo sapiens",
        "mouse": "Mus musculus",
        "mus_musculus": "Mus musculus",
        "mus musculus": "Mus musculus",
    }
    org_canonical = org_map.get((organism or "").lower().strip(), organism)
    org_canonical_l = (org_canonical or "").lower()

    tissue_l  = (tissue  or "").lower().strip()
    disease_l = (disease or "").lower().strip()
    assay_l   = (assay   or "").lower().strip()

    hits: List[DiscoveryHit] = []
    for rec in data:
        # Skip withdrawn/tombstoned datasets - website hides these
        if rec.get("tombstone"):
            continue
        # Organism filter
        org_labels = _cxg_labels(rec.get("organism"))
        if org_canonical_l and not any(org_canonical_l in o.lower()
                                        for o in org_labels):
            continue

        title = str(rec.get("title") or "")
        coll = str(rec.get("collection_name") or "")
        tissues  = _cxg_labels(rec.get("tissue"))
        diseases = _cxg_labels(rec.get("disease"))
        assays   = _cxg_labels(rec.get("assay"))
        celltypes = _cxg_labels(rec.get("cell_type"))

        tissues_s  = " | ".join(tissues).lower()
        diseases_s = " | ".join(diseases).lower()
        assays_s   = " | ".join(assays).lower()
        celltypes_s = " | ".join(celltypes).lower()

        # Strict sub-filters - each must hit its facet column
        if tissue_l and tissue_l not in tissues_s:
            continue
        if disease_l and disease_l not in diseases_s:
            continue
        if assay_l and assay_l not in assays_s:
            continue

        # Keyword match across title, collection, harmonized facets
        blob = " ".join([title.lower(), coll.lower(),
                          tissues_s, diseases_s, assays_s, celltypes_s])
        matched = _match_tokens(blob, tokens)
        if not matched:
            continue

        ds_id = str(rec.get("dataset_id") or "")
        coll_id = str(rec.get("collection_id") or "")
        n_cells = int(rec.get("cell_count") or 0)
        url = (rec.get("explorer_url")
               or (f"https://cellxgene.cziscience.com/collections/{coll_id}"
                   if coll_id else ""))
        # Build a compact summary showing the actual facet hits
        summary_parts = []
        if tissues:
            summary_parts.append("tissue: " + ", ".join(tissues[:3]))
        if diseases:
            summary_parts.append("disease: " + ", ".join(diseases[:3]))
        if assays:
            summary_parts.append("assay: " + ", ".join(assays[:3]))
        summary = " | ".join(summary_parts)

        hits.append(DiscoveryHit(
            source="CELLxGENE",
            accession=ds_id,
            title=title or coll,
            summary=summary or (f"Collection: {coll}" if coll else ""),
            organism=org_canonical or organism,
            platform=", ".join(assays[:2]) or "scRNA-seq",
            n_samples=n_cells,
            url=str(url),
            matched=sorted(set(matched)),
        ))

    log(f"[CELLxGENE] {len(hits)} dataset(s) matched")
    return _limit_hits(hits, max_results, "CELLxGENE", log)


# ────────────────────────────────────────────────────────────────────────────
# Expression Atlas (EMBL-EBI)
# ────────────────────────────────────────────────────────────────────────────
def search_atlas(
    keywords: str | Iterable[str],
    *,
    species: str = "",
    max_results: Optional[int] = None,
    log: LogFn = _noop,
) -> List[DiscoveryHit]:
    """Keyword-search Expression Atlas experiments via the public REST API."""
    tokens = _normalize_keywords(keywords)
    if not tokens:
        return []
    species_l = species.lower().strip() if species else ""

    try:
        import urllib.request
        import json
        req = urllib.request.Request(
            "https://www.ebi.ac.uk/gxa/json/experiments",
            headers={"Accept": "application/json",
                      "User-Agent": "GeneVariate/1.0"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        log(f"[Atlas] fetch failed: {exc}")
        return []

    experiments = data.get("experiments") or data.get("aaData") or []
    if not experiments:
        return []

    hits: List[DiscoveryHit] = []
    for exp in experiments:
        title = str(exp.get("experimentDescription") or exp.get("description") or "")
        acc = str(exp.get("experimentAccession") or exp.get("accession") or "")
        organism = str(exp.get("species") or "")
        factors = exp.get("experimentalFactors") or []
        if isinstance(factors, list):
            factors_s = ", ".join(str(f) for f in factors)
        else:
            factors_s = str(factors)
        if species_l and species_l not in organism.lower():
            continue
        blob = " ".join([title, acc, organism, factors_s])
        matched = _match_tokens(blob, tokens)
        if not matched:
            continue
        n = int(exp.get("numberOfAssays") or exp.get("numberOfSamples") or 0)
        hits.append(DiscoveryHit(
            source="Expression Atlas",
            accession=acc,
            title=title,
            summary=factors_s,
            organism=organism,
            platform=str(exp.get("technologyType") or ""),
            n_samples=n,
            url=f"https://www.ebi.ac.uk/gxa/experiments/{acc}" if acc else "",
            matched=sorted(set(matched)),
        ))

    log(f"[Atlas] {len(hits)} experiment(s) matched")
    return _limit_hits(hits, max_results, "Atlas", log)


# ────────────────────────────────────────────────────────────────────────────
# DataFrame shaping (compatible with GEOmetadb output)
# ────────────────────────────────────────────────────────────────────────────
def hits_to_dataframe(hits: Sequence[DiscoveryHit]) -> pd.DataFrame:
    """Shape DiscoveryHits into the columns ExtractionThread emits."""
    rows = []
    for h in hits:
        rows.append({
            "GSM": "",  # external sources: no GEO sample accession
            "series_id": h.accession,
            "gpl": h.platform or h.source,
            "title": h.title,
            "source_name_ch1": "",
            "characteristics_ch1": h.summary,
            "organism_ch1": h.organism,
            "Source": h.source,
            "Token_Match": 1 if h.matched else 0,
            "Matched_Tokens": list(h.matched),
        })
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────────────
# Unified entry-point
# ────────────────────────────────────────────────────────────────────────────
KNOWN_SOURCES = ("archs4", "cellxgene", "atlas")


def search_sources(
    keywords: str | Iterable[str],
    sources: Set[str] | Sequence[str],
    *,
    organism: str = "human",
    max_per_source: Optional[int] = None,
    log: LogFn = _noop,
    subfilters: Optional[Dict[str, str]] = None,
    status: Optional[Dict[str, str]] = None,
) -> Dict[str, List[DiscoveryHit]]:
    """Run the requested external searches and return per-source hit lists."""
    wanted = {s.lower().strip() for s in sources}
    sf = dict(subfilters or {})
    out: Dict[str, List[DiscoveryHit]] = {}
    st = status if status is not None else {}

    def _run(name, fn):
        try:
            out[name] = fn()
        except Exception as exc:
            out[name] = []
            st[name] = f"failed: {exc}"
            log(f"[Discovery] {name} did not answer, so its empty result "
                f"means unreachable and not 'nothing matched': {exc}")
        else:
            st[name] = "ok"

    if "archs4" in wanted:
        a4_org = sf.get("archs4_organism") or organism
        _run("archs4", lambda: search_archs4(
            keywords, organism=a4_org,
            max_results=max_per_source, log=log))
    if "cellxgene" in wanted:
        cx_org = "homo_sapiens" if organism == "human" else (
            "mus_musculus" if organism == "mouse" else organism)
        _run("cellxgene", lambda: search_cellxgene(
            keywords, organism=cx_org,
            tissue=sf.get("cx_tissue") or "",
            disease=sf.get("cx_disease") or "",
            assay=sf.get("cx_assay") or "",
            max_results=max_per_source, log=log))
    if "atlas" in wanted:
        _run("atlas", lambda: search_atlas(
            keywords, species=sf.get("atlas_species") or "",
            max_results=max_per_source, log=log))
    return out
