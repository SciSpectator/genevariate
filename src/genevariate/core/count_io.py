"""
Raw-count readers for the NGS / RNA-seq pipeline.

Every reader returns a **genes x samples** integer-ish DataFrame (rows = genes,
columns = samples) — the convention used by `genevariate.core.analysis.rnaseq`.
`load_counts` dispatches on the path (file extension or directory) and also
returns an optional per-sample metadata DataFrame (indexed by sample) carrying
any ``Classified_*`` / ``series_id`` columns.

Supported inputs:
  * CSV / TSV gene-count table (genes in the first column, samples as columns)
  * 10x Genomics MTX directory (``matrix.mtx[.gz]`` + features + barcodes)
  * AnnData ``.h5ad`` (``X`` = samples x genes counts; ``obs`` = sample meta)

Only scipy (a core dependency) is required for the MTX path — scanpy is *not*
needed. The h5ad path uses the existing anndata adapter and is import-guarded.
"""
from __future__ import annotations

import gzip
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------
# CSV / TSV
# -----------------------------------------------------------------
def read_counts_csv(path, sep: Optional[str] = None,
                    gene_col: int = 0) -> pd.DataFrame:
    """Read a gene-count table into a genes x samples DataFrame.

    ``sep=None`` autodetects the delimiter (pandas' python engine). ``gene_col``
    is the 0-based column holding gene identifiers (used as the index).
    """
    df = pd.read_csv(path, sep=sep, engine="python", index_col=gene_col)
    # Keep only numeric sample columns (drop stray annotation columns).
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        # Fall back: coerce everything to numeric.
        num = df.apply(pd.to_numeric, errors="coerce")
    num.index = num.index.astype(str)
    num = num[~num.index.duplicated(keep="first")]
    return num.fillna(0)


# -----------------------------------------------------------------
# 10x Genomics MTX
# -----------------------------------------------------------------
def _read_tsv_first_col(path: Path) -> list:
    opener = gzip.open if str(path).endswith(".gz") else open
    out = []
    with opener(path, "rt") as fh:
        for line in fh:
            out.append(line.rstrip("\n").split("\t")[0])
    return out


def _find(dir_path: Path, *names) -> Optional[Path]:
    for n in names:
        p = dir_path / n
        if p.exists():
            return p
    return None


def read_10x_mtx(dir_path) -> pd.DataFrame:
    """Read a 10x MTX directory into a genes x samples (cells) DataFrame.

    Uses ``scipy.io.mmread`` directly (scipy is a core dep); no scanpy needed.
    Accepts gzipped or plain features/genes/barcodes files.
    """
    from scipy.io import mmread

    d = Path(dir_path)
    mtx = _find(d, "matrix.mtx.gz", "matrix.mtx")
    if mtx is None:
        raise FileNotFoundError(f"No matrix.mtx[.gz] in {d}")
    features = _find(d, "features.tsv.gz", "features.tsv",
                     "genes.tsv.gz", "genes.tsv")
    barcodes = _find(d, "barcodes.tsv.gz", "barcodes.tsv")
    if features is None or barcodes is None:
        raise FileNotFoundError(f"Missing features/barcodes in {d}")

    m = mmread(str(mtx))  # genes x cells (10x convention)
    dense = np.asarray(m.todense()) if hasattr(m, "todense") else np.asarray(m)
    genes = _read_tsv_first_col(features)
    cells = _read_tsv_first_col(barcodes)
    if dense.shape != (len(genes), len(cells)):
        # Some exports store cells x genes — transpose if that matches.
        if dense.shape == (len(cells), len(genes)):
            dense = dense.T
        else:
            raise ValueError(
                f"MTX shape {dense.shape} matches neither "
                f"({len(genes)},{len(cells)}) nor its transpose")
    df = pd.DataFrame(dense, index=[str(g) for g in genes],
                      columns=[str(c) for c in cells])
    df.index = df.index.astype(str)
    return df[~df.index.duplicated(keep="first")]


# -----------------------------------------------------------------
# AnnData h5ad
# -----------------------------------------------------------------
def read_counts_h5ad(path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read counts from an ``.h5ad`` file.

    Returns (counts genes x samples, sample_meta indexed by sample). ``X`` is
    assumed to be samples x genes raw counts. Requires anndata (import-guarded
    inside the adapter).
    """
    from genevariate.utils.anndata_io import load_h5ad, _coerce_gene_index

    adata = load_h5ad(path)
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X)
    genes = [str(g) for g in _coerce_gene_index(adata.var)]
    samples = [str(s) for s in adata.obs.index]
    counts = pd.DataFrame(X.T, index=genes, columns=samples)  # genes x samples
    counts.index = counts.index.astype(str)
    counts = counts[~counts.index.duplicated(keep="first")]
    meta = adata.obs.copy()
    meta.index = meta.index.astype(str)
    return counts, meta


# -----------------------------------------------------------------
# Dispatch
# -----------------------------------------------------------------
def load_counts(path) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Load raw counts from a CSV/TSV file, a 10x MTX directory, or an h5ad.

    Returns (counts genes x samples, sample_meta or None).
    """
    p = Path(path)
    if p.is_dir():
        return read_10x_mtx(p), None
    suffix = "".join(p.suffixes).lower()
    if suffix.endswith(".h5ad"):
        return read_counts_h5ad(p)
    if p.name.lower().startswith("matrix.mtx"):
        return read_10x_mtx(p.parent), None
    return read_counts_csv(p), None
