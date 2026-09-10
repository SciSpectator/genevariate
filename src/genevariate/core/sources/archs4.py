"""
ARCHS4 data source - uniformly-processed bulk RNA-seq from GEO/SRA.

Uses `archs4py` to pull gene-level counts for a GEO Series (GSE) or a list
of GSM accessions from the ARCHS4 HDF5 mirror, normalizes with log-quantile
(matching GeneVariate's microarray pipeline), and emits canonical-format CSV.

Species are autodetected by trying human first, then mouse.
"""

from __future__ import annotations

import glob
import os
import re
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import requests

try:
    import archs4py as a4
    _HAS_A4 = True
except Exception:
    a4 = None
    _HAS_A4 = False

from .base import BaseSource, SourceInfo


def _sample_meta_frame(meta: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Reshape ARCHS4's metadata table into the columns extraction reads.

    ARCHS4 returns samples as rows indexed by accession, already using GEO's
    own ``*_ch1`` field names, so the only rename needed is the accession
    itself. Fields ARCHS4 does not carry (description, treatment protocol)
    are simply absent rather than blank, so a later GEOmetadb lookup can
    still supply them.
    """
    if meta is None or len(meta) == 0:
        return pd.DataFrame(columns=["gsm"])
    out = meta.copy()
    if "geo_accession" not in out.columns:
        out.insert(0, "geo_accession", [str(i) for i in out.index])
    out = out.rename(columns={"geo_accession": "gsm"})
    out["gsm"] = out["gsm"].astype(str).str.strip().str.upper()
    cols = ["gsm"] + [c for c in out.columns if c != "gsm"]
    return out[cols].reset_index(drop=True)


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

        archs4py's own downloader is not used here: it wraps ``wget.download``,
        which cannot resume. These matrices are 50-60 GB, so a connection that
        drops at 90% would restart from zero, and the transfer would in
        practice never complete on an ordinary link.
        """
        species = (species or self.species).lower()
        cached = self._locate_h5(species)
        if cached:
            return cached

        cfg = a4.utils.get_config()["GENE_COUNTS"][species.upper()]["latest"]
        urls = [u for u in (cfg.get("primary"), cfg.get("fallback")) if u]
        dest = os.path.join(self.cache_dir, os.path.basename(urls[0]))
        last: Optional[Exception] = None
        for url in urls:
            try:
                self._download_resumable(url, dest, progress=progress)
                return dest
            except Exception as exc:
                last = exc
        raise RuntimeError(
            f"ARCHS4 {species} download failed from all mirrors: {last}")

    @staticmethod
    def _download_resumable(url: str, dest: str,
                            progress: Optional[Callable[[str, float], None]] = None,
                            attempts: int = 100,
                            chunk: int = 8 << 20) -> str:
        """Stream `url` to `dest`, continuing where an interrupted run stopped.

        Bytes already on disk are re-requested with a Range header rather than
        refetched. The partial file carries a ``.part`` suffix so that a
        half-written matrix can never be mistaken for a usable cache entry by
        ``_locate_h5``, which globs for ``*_gene*.h5``.
        """
        part = dest + ".part"
        total: Optional[int] = None
        try:
            head = requests.head(url, allow_redirects=True, timeout=60)
            if head.ok:
                total = int(head.headers.get("Content-Length") or 0) or None
        except Exception:
            pass

        def _say(done: int):
            if not progress:
                return
            if total:
                progress(f"Downloading ARCHS4 {os.path.basename(dest)} - "
                         f"{done / 1e9:.1f} / {total / 1e9:.1f} GB "
                         f"(one-time, resumable)",
                         min(0.20, done / total * 0.20))
            else:
                progress(f"Downloading ARCHS4 {os.path.basename(dest)} - "
                         f"{done / 1e9:.1f} GB", 0.0)

        last: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            have = os.path.getsize(part) if os.path.exists(part) else 0
            if total is not None and have >= total:
                break
            try:
                headers = {"Range": f"bytes={have}-"} if have else {}
                with requests.get(url, headers=headers, stream=True,
                                  timeout=120, allow_redirects=True) as r:
                    r.raise_for_status()
                    # A server that ignores Range answers 200 with the whole
                    # file; appending that to what we already have would
                    # silently produce a corrupt matrix.
                    if have and r.status_code != 206:
                        have = 0
                    mode = "ab" if have else "wb"
                    if total is None:
                        cl = int(r.headers.get("Content-Length") or 0)
                        total = (have + cl) if cl else None
                    marker = have
                    _say(have)
                    with open(part, mode) as fh:
                        for block in r.iter_content(chunk_size=chunk):
                            if not block:
                                continue
                            fh.write(block)
                            have += len(block)
                            if have - marker >= (64 << 20):
                                marker = have
                                _say(have)
            except Exception as exc:
                last = exc
                if attempt == attempts:
                    raise
                time.sleep(min(30, 2 * attempt))

        size = os.path.getsize(part) if os.path.exists(part) else 0
        if total is not None and size != total:
            raise IOError(
                f"ARCHS4 download incomplete: {size:,} of {total:,} bytes "
                f"({os.path.basename(dest)}). Rerun to resume.")
        if size <= 0:
            raise IOError(f"ARCHS4 download produced no data ({last})")
        os.replace(part, dest)
        if progress:
            progress(f"ARCHS4 matrix ready ({size / 1e9:.1f} GB)", 0.22)
        return dest

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

        # ARCHS4 harmonises the submitter's own text fields alongside the
        # counts, and they are the only description of these samples that
        # travels with the matrix. Label extraction otherwise falls back to
        # GEOmetadb, which is a dated snapshot -- any GSM newer than it would
        # reach the extractor as a bare accession with nothing to read.
        df.attrs["sample_meta"] = _sample_meta_frame(meta)

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
