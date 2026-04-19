"""
ARCHS4 data source — uniformly-processed bulk RNA-seq from GEO/SRA.

Uses `archs4py` to pull gene-level counts for a GEO Series (GSE) or a list
of GSM accessions from the ARCHS4 HDF5 mirror, normalizes with log-quantile
(matching GeneVariate's microarray pipeline), and emits canonical-format CSV.

Species are autodetected by trying human first, then mouse.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

try:
    import archs4py as a4
    _HAS_A4 = True
except Exception:
    a4 = None
    _HAS_A4 = False

from .base import BaseSource, SourceInfo


class Archs4Source(BaseSource):
    """
    Pull RNA-seq counts for GEO accessions from ARCHS4.

    Typical usage:
        src = Archs4Source(cache_dir="~/.genevariate/archs4", species="human")
        df  = src.fetch("GSE123456")   # GSE → all samples in series
        df  = src.fetch(["GSM111", "GSM222"])  # list of GSMs
        path = Archs4Source.save_csv(df, out_dir, "archs4_gse123456")
    """

    info = SourceInfo(
        name="ARCHS4",
        technology="bulk-rna-seq",
        species="human",
        description="MaayanLab uniformly-processed GEO/SRA bulk RNA-seq counts",
    )

    def __init__(self,
                 cache_dir: str = "~/.genevariate/archs4",
                 species: str = "human"):
        if not _HAS_A4:
            raise RuntimeError(
                "archs4py is not installed. Install with "
                "`pip install archs4py --break-system-packages --user`"
            )
        self.cache_dir = os.path.expanduser(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.species = species.lower()
        if self.species not in ("human", "mouse"):
            raise ValueError("species must be 'human' or 'mouse'")

    # ------------------------------------------------------------------
    # File management
    # ------------------------------------------------------------------
    def _locate_h5(self, species: str) -> Optional[str]:
        """
        Look for an existing ARCHS4 HDF5 in the cache for this species.
        ARCHS4 file names include the version string (e.g. human_gene_v2.5.h5),
        so we glob rather than hardcode. Returns the most recently modified
        match (proxy for "latest version") or None.
        """
        pattern = os.path.join(self.cache_dir, f"{species}_gene*.h5")
        matches = [p for p in glob.glob(pattern)
                   if os.path.getsize(p) > 10 * 1024 * 1024]
        if not matches:
            return None
        matches.sort(key=os.path.getmtime, reverse=True)
        return matches[0]

    def ensure_h5(self, species: Optional[str] = None,
                  progress: Optional[Callable[[str, float], None]] = None) -> str:
        """
        Download the ARCHS4 HDF5 bundle for `species` to the cache if missing.
        Returns the file path.
        """
        species = (species or self.species).lower()
        cached = self._locate_h5(species)
        if cached:
            return cached
        if progress:
            progress(f"Downloading ARCHS4 {species} gene counts (≈30GB, one-time)...", 0.0)
        a4.download.counts(species, path=self.cache_dir, type="GENE_COUNTS")
        cached = self._locate_h5(species)
        if cached is None:
            raise FileNotFoundError("ARCHS4 download completed but file not located")
        return cached

    @staticmethod
    def _file_version(path: str) -> Optional[str]:
        """Best-effort extract the version segment from the filename."""
        base = os.path.basename(path)
        m = re.search(r"v([0-9][0-9.\-]*)", base)
        return m.group(1) if m else None

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------
    def fetch(self,
              query,
              normalize: str = "log_quantile",
              progress: Optional[Callable[[str, float], None]] = None,
              **kwargs) -> pd.DataFrame:
        """
        query: a GSE accession (str like 'GSE123456') or a list of GSM accessions.
        Returns a canonical-format DataFrame (GSM | series_id | gene columns).
        """
        h5 = self.ensure_h5(progress=progress)

        if isinstance(query, str) and query.upper().startswith("GSE"):
            if progress:
                progress(f"Fetching ARCHS4 samples for {query}...", 0.25)
            counts = a4.data.series(h5, query)
            meta = a4.meta.series(h5, query, silent=True)
            series_col = {str(g).upper(): query.upper()
                          for g in counts.columns}
        else:
            samples = query if isinstance(query, (list, tuple)) else [query]
            samples = [str(s).strip().upper() for s in samples]
            if progress:
                progress(f"Fetching {len(samples)} ARCHS4 samples...", 0.25)
            counts = a4.data.samples(h5, samples)
            meta = a4.meta.samples(h5, samples, silent=True)
            series_col = {}
            if meta is not None and "series_id" in meta.columns:
                for g in counts.columns:
                    g_up = str(g).upper()
                    s = meta.loc[meta.get("geo_accession", pd.Series()).astype(str).str.upper() == g_up,
                                 "series_id"]
                    if not s.empty:
                        series_col[g_up] = str(s.iloc[0]).split(",")[0].strip().upper()

        if counts is None or counts.empty:
            raise ValueError(f"ARCHS4 returned no data for query={query!r}")

        if progress:
            progress("Normalizing counts (log-quantile)...", 0.7)
        normed = a4.normalize(counts, method=normalize)

        if progress:
            progress("Assembling canonical DataFrame...", 0.9)
        df = self.to_canonical(normed, gsm_to_series=series_col)
        df.attrs["provenance"] = {
            "source": "ARCHS4",
            "archs4py_version": getattr(a4, "__version__", "unknown"),
            "h5_filename": os.path.basename(h5),
            "h5_version": self._file_version(h5),
            "species": self.species,
            "query": query if isinstance(query, str) else list(query),
            "normalize_method": normalize,
            "n_samples": int(df.shape[0]),
            "n_genes": int(df.shape[1] - len(("GSM", "series_id"))),
        }
        return df
