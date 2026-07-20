"""
Batch-effect integration across sources/modalities.

Cross-modality comparison by per-source z-scoring only *rescales* platform
effects; it does not remove them. This module provides real batch correction so
microarray / RNA-seq / single-cell values can be placed in a genuinely shared
space before comparison:

  - :func:`combat_correct` — ComBat (Johnson et al. 2007) empirical-Bayes batch
    correction on the matrix of genes shared across sources, treating each source
    as a batch. Wraps whichever ComBat implementation is installed
    (``inmoose``/``pycombat``/``combat``); raises a clear error if none is.
  - :func:`harmony_embed` — Harmony (`harmonypy`) joint embedding of the shared-
    gene matrix, returning batch-corrected PCA coordinates for visualisation.

Both consume GeneVariate's canonical platform DataFrames and are optional: import
guards keep the base app working without these packages.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_META_PREFIXES = ("GSM", "series_id")


def _is_meta(col) -> bool:
    return col in _META_PREFIXES or str(col).startswith("Classified_")


def _gene_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical platform DataFrame -> genes x samples numeric matrix (upper genes)."""
    genes = [c for c in df.columns if not _is_meta(c)]
    idx = (df["GSM"].astype(str).values if "GSM" in df.columns
           else df.index.astype(str))
    mat = df[genes].apply(pd.to_numeric, errors="coerce")
    mat.index = idx                       # samples
    mat.columns = [str(c).upper() for c in mat.columns]
    return mat.T                          # genes x samples


def common_gene_matrix(sources: Dict[str, pd.DataFrame]
                       ) -> Tuple[pd.DataFrame, pd.Series]:
    """Build a genes x samples matrix on the genes shared by all sources.

    Sample columns are prefixed with the source name to stay unique. Returns
    ``(matrix, batch)`` where ``batch`` is a per-sample Series of source names.
    """
    if len(sources) < 2:
        raise ValueError("Need at least two sources to integrate.")
    mats = {name: _gene_matrix(df) for name, df in sources.items()}
    shared = set.intersection(*[set(m.index) for m in mats.values()])
    if not shared:
        raise ValueError("Sources share no common genes.")
    shared = sorted(shared)
    blocks, batch = [], []
    for name, m in mats.items():
        sub = m.loc[shared].copy()
        sub.columns = [f"{name}|{c}" for c in sub.columns]
        blocks.append(sub)
        batch.extend([name] * sub.shape[1])
    matrix = pd.concat(blocks, axis=1)
    return matrix, pd.Series(batch, index=matrix.columns, name="batch")


def _combat_impl():
    """Return a ``callable(matrix_genes_x_samples, batch_series) -> corrected``.

    Tries inmoose, pycombat, then the ``combat`` package, adapting each to a
    genes x samples DataFrame in / out. Raises RuntimeError if none is present.
    """
    try:
        from inmoose.pycombat import pycombat_norm  # inmoose (maintained fork)

        def _run(mat: pd.DataFrame, batch: pd.Series) -> pd.DataFrame:
            out = pycombat_norm(mat.values, batch.values)
            return pd.DataFrame(out, index=mat.index, columns=mat.columns)
        return _run
    except Exception:
        pass
    try:
        from pycombat import Combat  # CoAxLab pycombat (samples x genes)

        def _run(mat: pd.DataFrame, batch: pd.Series) -> pd.DataFrame:
            c = Combat()
            corrected = c.fit_transform(
                Y=mat.T.values,
                b=pd.factorize(batch.values)[0])
            return pd.DataFrame(corrected.T, index=mat.index, columns=mat.columns)
        return _run
    except Exception:
        pass
    try:
        from combat.pycombat import pycombat  # combat package (genes x samples)

        def _run(mat: pd.DataFrame, batch: pd.Series) -> pd.DataFrame:
            out = pycombat(mat, batch)
            return pd.DataFrame(np.asarray(out), index=mat.index,
                                columns=mat.columns)
        return _run
    except Exception:
        pass
    raise RuntimeError(
        "No ComBat implementation installed. Install one of: `inmoose`, "
        "`pycombat`, or `combat` to run batch correction.")


def combat_correct(sources: Dict[str, pd.DataFrame]
                   ) -> Dict[str, pd.DataFrame]:
    """ComBat-correct the shared-gene matrix, treating each source as a batch.

    Returns ``{source_name -> corrected canonical platform DataFrame}`` on the
    shared genes, so the corrected values can flow back into every analysis.
    """
    matrix, batch = common_gene_matrix(sources)
    corrected = _combat_impl()(matrix, batch)
    out: Dict[str, pd.DataFrame] = {}
    for name in sources:
        cols = [c for c in corrected.columns if c.startswith(f"{name}|")]
        sub = corrected[cols]
        sub.columns = [c.split("|", 1)[1] for c in cols]
        plat = sub.T.reset_index().rename(columns={"index": "GSM"})
        plat["GSM"] = plat["GSM"].astype(str)
        out[name] = plat
    return out


def harmony_embed(sources: Dict[str, pd.DataFrame], n_pcs: int = 30
                  ) -> Dict[str, object]:
    """Harmony batch-corrected joint embedding of the shared-gene matrix.

    Returns ``{"embedding": DataFrame(samples x PCs), "batch": Series}``.
    Requires ``harmonypy`` and ``scikit-learn``.
    """
    try:
        import harmonypy
    except Exception as exc:
        raise RuntimeError(
            "harmonypy is not installed. `pip install harmonypy` for Harmony "
            "integration.") from exc
    from sklearn.decomposition import PCA

    matrix, batch = common_gene_matrix(sources)
    X = matrix.T.fillna(0.0).values                     # samples x genes
    k = int(min(n_pcs, min(X.shape) - 1, 50))
    pcs = PCA(n_components=max(k, 2), random_state=0).fit_transform(X)
    meta = pd.DataFrame({"batch": batch.values}, index=matrix.columns)
    ho = harmonypy.run_harmony(pcs, meta, ["batch"])
    corrected = np.asarray(ho.Z_corr).T                 # samples x PCs
    emb = pd.DataFrame(corrected, index=matrix.columns,
                       columns=[f"PC{i + 1}" for i in range(corrected.shape[1])])
    return {"embedding": emb, "batch": batch}
