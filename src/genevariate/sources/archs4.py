"""GeneVariate - ARCHS4 (Ma'ayan Lab) data source."""

from __future__ import annotations

from typing import (Any, Dict, Iterable, List, Optional, Sequence,
                    Tuple, TYPE_CHECKING)

if TYPE_CHECKING:
    import pandas


# ────────────────────────────────────────────────────────────────────────────
# Lazy import - archs4py isn't a hard dep of the whole program
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


# Known organisms - matches the keys in archs4py's config
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
# Client - thin cache wrapper around archs4py's remote calls
# ────────────────────────────────────────────────────────────────────────────
class ARCHS4Client:
    """Remote-first client; caches small results in-memory across calls."""

    def __init__(self, organism: str = "human", version: str = "latest"):
        self.organism = organism
        self.version = version
        self.url = _resolve_url(organism, version)
        self._meta_cache: Dict[str, Dict[str, str]] = {}

    # ───── Metadata look-up ─────────────────────────────────────────
    def metadata_for_gsm(self, gsm: str) -> Optional[Dict[str, str]]:
        """Return ARCHS4's harmonised metadata for one GSM, or None."""
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

    # ───── Gene-slice fetch ─────────────────────────────────────────
    def gene_slice(
        self,
        gene: str,
        *,
        max_samples: Optional[int] = None,
    ) -> Optional["pandas.Series"]:
        """Return expression for one gene across ARCHS4 samples, indexed by GSM."""
        a4 = _require_archs4py()
        import numpy as np   # type: ignore
        import pandas as pd  # type: ignore

        try:
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
