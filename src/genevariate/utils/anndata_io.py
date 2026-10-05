"""GeneVariate - AnnData / GeneVariate platform DataFrame adapters."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# ────────────────────────────────────────────────────────────────────────────
# Lazy anndata import
# ────────────────────────────────────────────────────────────────────────────
def _require_anndata():
    try:
        import anndata as ad  # noqa: F401
        return ad
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "anndata is required for single-cell support. "
            "Install it with: pip install anndata"
        ) from exc


# ────────────────────────────────────────────────────────────────────────────
# CELLxGENE / HCA ontology fields
# ────────────────────────────────────────────────────────────────────────────
ONTOLOGY_FIELDS: Tuple[str, ...] = (
    "cell_type",
    "tissue",
    "disease",
    "assay",
    "development_stage",
    "sex",
    "self_reported_ethnicity",
    "organism",
)



ID_COLS: Tuple[str, ...] = (
    "donor_id", "sample_id", "library_id", "dataset_id", "batch",
    "cell_id", "suspension_type", "is_primary_data",
)


def _coerce_gene_index(var: pd.DataFrame) -> pd.Index:
    """Choose a stable, human-readable gene identifier for ``var``."""
    for cand in ("feature_name", "gene_symbol", "hgnc_symbol", "Symbol", "symbol"):
        if cand in var.columns:
            syms = var[cand].astype(str)
            dup = syms.duplicated(keep=False)
            if dup.any():
                for idcol in ("feature_id", "gene_ids", "gene_id", "ensembl_id"):
                    if idcol in var.columns:
                        tag = var[idcol].astype(str)
                        break
                else:
                    tag = pd.Series(var.index.astype(str), index=var.index)
                syms = syms.mask(dup, syms + "_" + tag)
            return pd.Index(syms.values, name=cand)
    return pd.Index(var.index.astype(str), name=var.index.name or "gene_id")


def _normalize_label_columns(obs: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``obs`` with its ontology fields read as plain text."""
    out = obs.copy()
    for src in ONTOLOGY_FIELDS:
        val = out.get(src)
        if val is not None and hasattr(val, "astype"):
            out[src] = val.astype(object).astype(str)
    return out


# ────────────────────────────────────────────────────────────────────────────
# AnnData → platform DataFrame (cell-level; each row is a cell)
# ────────────────────────────────────────────────────────────────────────────
def anndata_to_platform_df(
    adata,
    *,
    gene_col: Optional[Sequence[str]] = None,
    include_obs: Optional[Sequence[str]] = None,
    max_genes: Optional[int] = None,
) -> pd.DataFrame:
    """Convert an AnnData to GeneVariate's platform-DataFrame shape."""
    ad = _require_anndata()  # noqa: F841 (import side-effect)
    if adata is None:
        raise ValueError("adata is None")

    obs = _normalize_label_columns(adata.obs)
    if include_obs is not None:
        obs_keep = [c for c in include_obs if c in obs.columns]
        obs = obs[obs_keep]

    # Gene selection
    var_index = _coerce_gene_index(adata.var)
    if gene_col is None:
        if max_genes is not None and adata.n_vars > max_genes:
            # Pick highest-variance genes
            X = adata.X
            if hasattr(X, "toarray"):
                # sparse - compute column variances efficiently
                n = X.shape[0]
                mean = np.asarray(X.mean(axis=0)).ravel()
                mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
                var = np.clip(mean_sq - mean**2, 0.0, None)
            else:
                var = np.nanvar(np.asarray(X), axis=0)
            top = np.argsort(-var)[:max_genes]
            sel = sorted(top.tolist())
        else:
            sel = list(range(adata.n_vars))
    else:
        wanted = set(map(str, gene_col))
        sel = [i for i, g in enumerate(var_index) if str(g) in wanted]
        if not sel:
            raise ValueError("None of the requested genes are in adata.var")

    # Densify the selected subset only
    X_sub = adata.X[:, sel]
    if hasattr(X_sub, "toarray"):
        X_sub = X_sub.toarray()
    else:
        X_sub = np.asarray(X_sub)

    gene_names = [str(g) for g in var_index[sel]]
    expr_df = pd.DataFrame(X_sub, index=adata.obs.index, columns=gene_names)

    df = pd.concat([obs, expr_df], axis=1, copy=False)
    return df


# ────────────────────────────────────────────────────────────────────────────
# h5ad load / save
# ────────────────────────────────────────────────────────────────────────────
def load_h5ad(path: Union[str, Path]):
    """Load a .h5ad file (produced by scanpy / CELLxGENE export)."""
    ad = _require_anndata()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return ad.read_h5ad(str(p))


def save_h5ad(adata, path: Union[str, Path], *, compression: str = "gzip") -> Path:
    """Save an AnnData to .h5ad with gzip compression (default)."""
    _require_anndata()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(str(p), compression=compression)
    return p


# ────────────────────────────────────────────────────────────────────────────
# Summary helpers (used by the browser window before committing a large fetch)
# ────────────────────────────────────────────────────────────────────────────
def summarize_adata(adata) -> Dict[str, Any]:
    """Return a small dict summarizing an AnnData (for UI display)."""
    _require_anndata()
    summary = {
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "layers": sorted(list(adata.layers.keys())),
        "obs_cols": list(adata.obs.columns),
        "var_cols": list(adata.var.columns),
    }
    # Ontology-ish categorical distributions
    for key in ("cell_type", "tissue", "disease", "assay",
                 "development_stage", "sex"):
        if key in adata.obs.columns:
            vc = adata.obs[key].astype(str).value_counts()
            summary[key] = {str(k): int(v) for k, v in vc.head(10).items()}
            summary[f"{key}_n_unique"] = int(adata.obs[key].nunique())
    return summary
