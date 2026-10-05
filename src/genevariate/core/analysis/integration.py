"""Batch-effect integration across sources/modalities."""
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
    out = mat.T                           # genes x samples
    if out.index.has_duplicates:
        out = out.groupby(level=0).mean()
    return out


def common_gene_matrix(sources: Dict[str, pd.DataFrame]
                       ) -> Tuple[pd.DataFrame, pd.Series]:
    """Returns ``(matrix, batch)`` where ``batch`` is a per-sample Series of source names."""
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
    """Return a ``callable(matrix_genes_x_samples, batch_series) -> corrected``."""
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


def combat_correct(sources: Dict[str, pd.DataFrame],
                   log=None) -> Dict[str, pd.DataFrame]:
    """Returns ``{source_name -> corrected canonical platform DataFrame}`` on the shared genes."""
    matrix, batch = common_gene_matrix(sources)
    blank = matrix.isna().any(axis=0).to_numpy()
    if blank.any():
        keep = ~blank
        if not keep.any():
            raise ValueError(
                f"Every one of the {matrix.shape[1]:,} samples has an "
                f"unmeasured gene among the {matrix.shape[0]:,} shared, so "
                f"ComBat has no complete block to work on.")
        note = (f"ComBat needs no gaps: {int(blank.sum()):,} of "
                f"{matrix.shape[1]:,} samples carry an unmeasured gene and "
                f"were set aside, keeping all {matrix.shape[0]:,} shared "
                f"genes on {int(keep.sum()):,} samples.")
        if log:
            log(note)
        matrix = matrix.loc[:, keep]
        batch = batch[keep]
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


def harmony_embed(sources: Dict[str, pd.DataFrame], n_pcs: int = 30,
                  log=None) -> Dict[str, object]:
    """Returns ``{"embedding": DataFrame(samples x PCs), "batch": Series}``."""
    try:
        import harmonypy
    except Exception as exc:
        raise RuntimeError(
            "harmonypy is not installed. `pip install harmonypy` for Harmony "
            "integration.") from exc
    from sklearn.decomposition import PCA

    matrix, batch = common_gene_matrix(sources)
    blank = matrix.isna().any(axis=0).to_numpy()
    if blank.any():
        keep = ~blank
        if not keep.any():
            raise ValueError(
                f"Every one of the {matrix.shape[1]:,} samples has an "
                f"unmeasured gene among the {matrix.shape[0]:,} shared, so "
                f"there is no complete block to embed.")
        note = (f"Harmony needs no gaps: {int(blank.sum()):,} of "
                f"{matrix.shape[1]:,} samples carry an unmeasured gene and "
                f"were set aside, keeping all {matrix.shape[0]:,} shared "
                f"genes on {int(keep.sum()):,} samples.")
        if log:
            log(note)
        matrix = matrix.loc[:, keep]
        batch = batch[keep]
    X = matrix.T.values                                 # samples x genes
    k = int(min(n_pcs, min(X.shape) - 1, 50))
    pcs = PCA(n_components=max(k, 2), random_state=0).fit_transform(X)
    meta = pd.DataFrame({"batch": batch.values}, index=matrix.columns)
    nclust = int(min(max(2, round(X.shape[0] / 30)), 100, max(X.shape[0] - 1, 2)))
    ho = harmonypy.run_harmony(pcs, meta, ["batch"], nclust=nclust)
    # Z_corr is (PCs x samples) before harmonypy 2, (samples x PCs) after
    corrected = np.asarray(ho.Z_corr)
    if corrected.shape[0] != X.shape[0]:
        corrected = corrected.T                         # -> samples x PCs
    emb = pd.DataFrame(corrected, index=matrix.columns,
                       columns=[f"PC{i + 1}" for i in range(corrected.shape[1])])
    return {"embedding": emb, "batch": batch}
