"""
GeneVariate — ARCHS4 (Ma'ayan Lab) data source.

Provides two capabilities without requiring the user to download the
~30 GB H5 file:

1. ``metadata_for_gsm(gsm)`` — fetch the harmonised ARCHS4 metadata for a
   single GEO sample (tissue, cell_line, series_id, …). Uses remote
   range-read calls in ``archs4py`` so the transfer is kilobytes.

2. ``gene_slice(gene, organism="human", max_samples=None)`` — fetch a
   single gene's expression vector across all ARCHS4 samples. Transfers
   ~2 MB per gene (one HDF5 row) instead of the whole matrix.

All values returned are real measurements aggregated from GEO/SRA by the
ARCHS4 pipeline — nothing is simulated or inferred.

Docs
----
* ARCHS4 project:   https://maayanlab.cloud/archs4/
* archs4py:         https://pypi.org/project/archs4py/
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ────────────────────────────────────────────────────────────────────────────
# Lazy import — archs4py isn't a hard dep of the whole program
# ────────────────────────────────────────────────────────────────────────────
def _require_archs4py():
    try:
        import archs4py as a4
        return a4
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "archs4py is required for ARCHS4 access.\n"
            "Install with:  pip install archs4py"
        ) from exc


# Known organisms — matches the keys in archs4py's config
ORGANISMS = ("human", "mouse")

# Which archs4py meta fields we extract when enriching labels
_META_FIELDS = [
    "geo_accession", "series_id", "title",
    "source_name_ch1", "characteristics_ch1",
    "extract_protocol_ch1",
]


def _resolve_url(organism: str = "human", version: str = "latest") -> str:
    """Return the remote H5 URL archs4py will use."""
    a4 = _require_archs4py()
    cfg = a4.utils.get_config()
    key = organism.upper()
    try:
        entry = cfg["GENE_COUNTS"][key][version]
    except KeyError as exc:
        raise KeyError(
            f"No ARCHS4 URL for organism={organism!r}, version={version!r}. "
            f"Known organisms: {list(cfg['GENE_COUNTS'])}"
        ) from exc
    return entry["primary"]


# ────────────────────────────────────────────────────────────────────────────
# Client — thin cache wrapper around archs4py's remote calls
# ────────────────────────────────────────────────────────────────────────────
class ARCHS4Client:
    """Remote-first client; caches small results in-memory across calls.

    The client never downloads the full H5 file. All queries are
    range-read HTTPS requests against the S3-hosted matrix.
    """

    def __init__(self, organism: str = "human", version: str = "latest"):
        self.organism = organism
        self.version = version
        self.url = _resolve_url(organism, version)
        self._meta_cache: Dict[str, Dict[str, str]] = {}

    # ───── Metadata look-up ─────────────────────────────────────────
    def metadata_for_gsm(self, gsm: str) -> Optional[Dict[str, str]]:
        """Return ARCHS4's harmonised metadata for one GSM, or None.

        The returned dict has string values for the fields in
        ``_META_FIELDS``. Missing → None.
        """
        if not gsm:
            return None
        gsm_norm = str(gsm).strip()
        if gsm_norm in self._meta_cache:
            return self._meta_cache[gsm_norm] or None

        a4 = _require_archs4py()
        try:
            df = a4.data.meta_remote(
                self.url, gsm_norm,
                meta_fields=_META_FIELDS, silent=True)
        except Exception:
            self._meta_cache[gsm_norm] = {}
            return None
        if df is None or len(df) == 0:
            self._meta_cache[gsm_norm] = {}
            return None

        # archs4py returns a DataFrame; pick the first exact-match row
        import pandas as pd  # type: ignore
        hit = None
        if "geo_accession" in df.columns:
            m = df["geo_accession"].astype(str) == gsm_norm
            if m.any():
                hit = df[m].iloc[0]
        if hit is None:
            hit = df.iloc[0]
        rec = {k: ("" if pd.isna(v) else str(v)) for k, v in hit.items()}
        self._meta_cache[gsm_norm] = rec
        return rec

    def metadata_for_series(self, gse: str, max_rows: int = 200
                            ) -> Optional["pandas.DataFrame"]:
        """Return all ARCHS4 samples for a GSE (series_id) as a DataFrame."""
        if not gse:
            return None
        a4 = _require_archs4py()
        try:
            df = a4.data.meta_remote(
                self.url, gse,
                meta_fields=_META_FIELDS, silent=True)
        except Exception:
            return None
        if df is None or len(df) == 0:
            return None
        if "series_id" in df.columns:
            df = df[df["series_id"].astype(str).str.contains(str(gse),
                                                              na=False)]
        if max_rows is not None and len(df) > max_rows:
            df = df.head(max_rows)
        return df

    # ───── Gene-slice fetch ─────────────────────────────────────────
    def gene_slice(
        self,
        gene: str,
        *,
        max_samples: Optional[int] = None,
    ) -> Optional["pandas.Series"]:
        """Return expression for one gene across ARCHS4 samples.

        Uses a single HDF5 row fetch (~2 MB for human, ~1M samples).
        Returned Series is indexed by GSM accession; values are raw
        ARCHS4 expression counts (same units as the matrix).

        If ``max_samples`` is set, the returned Series is truncated to
        the first ``max_samples`` entries to keep memory bounded.
        """
        a4 = _require_archs4py()
        import numpy as np   # type: ignore
        import pandas as pd  # type: ignore

        try:
            # archs4py.data.index_remote takes sample_idx and gene_idx;
            # passing empty sample_idx means "all samples".
            # archs4py's meta module provides the gene index lookup.
            gene_names = a4.meta.get_meta_gene_field(self.url, "symbol",
                                                      silent=True)
            gene_names = [str(g) for g in list(gene_names)]
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read gene index from {self.url}: {exc}"
            ) from exc
        if gene not in gene_names:
            return None
        j = gene_names.index(gene)

        try:
            gsm_ids = a4.meta.get_meta_sample_field(self.url,
                                                     "geo_accession",
                                                     silent=True)
            gsm_ids = [str(g) for g in list(gsm_ids)]
        except Exception:
            gsm_ids = None

        # Fetch row
        sample_idx = list(range(len(gsm_ids))) if gsm_ids else []
        if max_samples is not None and gsm_ids and len(gsm_ids) > max_samples:
            sample_idx = sample_idx[:max_samples]
            gsm_ids = gsm_ids[:max_samples]

        try:
            expr = a4.data.index_remote(self.url,
                                         sample_idx=sample_idx,
                                         gene_idx=[j],
                                         silent=True)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch gene row {gene!r}: {exc}"
            ) from exc

        if expr is None:
            return None
        arr = np.asarray(expr).ravel().astype(float)
        if gsm_ids and len(gsm_ids) == len(arr):
            return pd.Series(arr, index=pd.Index(gsm_ids, name="GSM"),
                              name=gene)
        return pd.Series(arr, name=gene)
