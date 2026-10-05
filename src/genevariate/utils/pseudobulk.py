"""GeneVariate - Pseudo-bulk aggregation of AnnData."""

from __future__ import annotations

from typing import Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

AggMethod = Literal["mean", "sum", "median"]


# ────────────────────────────────────────────────────────────────────────────
# Lazy anndata import
# ────────────────────────────────────────────────────────────────────────────
def _require_anndata():
    try:
        import anndata as ad
        return ad
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "anndata is required for pseudo-bulk aggregation. "
            "Install with: pip install anndata"
        ) from exc


# ────────────────────────────────────────────────────────────────────────────
# Core aggregation
# ────────────────────────────────────────────────────────────────────────────
def pseudobulk(
    adata,
    groupby: Sequence[str] = ("donor_id", "cell_type"),
    agg: AggMethod = "mean",
    *,
    min_cells: int = 10,
    layer: Optional[str] = None,
    use_raw: bool = False,
    return_obs: Optional[Sequence[str]] = None,
):
    """Aggregate cells of ``adata`` into synthetic-sample AnnData."""
    ad_mod = _require_anndata()
    if adata is None:
        raise ValueError("adata is None")
    groupby = list(groupby)
    missing = [g for g in groupby if g not in adata.obs.columns]
    if missing:
        raise KeyError(f"groupby columns not found in adata.obs: {missing}")

    # Choose the expression matrix - always real measured data
    if use_raw:
        if adata.raw is None:
            raise ValueError("use_raw=True but adata.raw is None")
        X = adata.raw.X
        var = adata.raw.var.copy()
    elif layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"layer {layer!r} not present in adata.layers")
        X = adata.layers[layer]
        var = adata.var.copy()
    else:
        X = adata.X
        var = adata.var.copy()

    obs = adata.obs[groupby].copy()
    # Composite group key for indexing the aggregation
    group_key = obs.astype(str).agg("§".join, axis=1)
    groups = pd.Categorical(group_key)
    codes = groups.codes
    n_groups = len(groups.categories)

    # Group sizes (real cell counts)
    sizes = np.bincount(codes, minlength=n_groups)

    # Aggregate gene-wise
    if hasattr(X, "toarray"):
        from scipy import sparse as _sparse
        if agg == "mean":
            # Mean = group sum / group size. Use sparse matmul for the sum.
            ind_mtx = _sparse.csr_matrix(
                (np.ones(X.shape[0], dtype=np.float32),
                 (codes, np.arange(X.shape[0]))),
                shape=(n_groups, X.shape[0]),
            )
            summed = (ind_mtx @ X).toarray().astype(np.float32)
            with np.errstate(invalid="ignore"):
                agg_mat = summed / sizes[:, None].astype(np.float32)
        elif agg == "sum":
            ind_mtx = _sparse.csr_matrix(
                (np.ones(X.shape[0], dtype=np.float32),
                 (codes, np.arange(X.shape[0]))),
                shape=(n_groups, X.shape[0]),
            )
            agg_mat = (ind_mtx @ X).toarray().astype(np.float32)
        elif agg == "median":
            agg_mat = np.zeros((n_groups, X.shape[1]), dtype=np.float32)
            for g in range(n_groups):
                rows = np.where(codes == g)[0]
                if rows.size == 0:
                    continue
                block = X[rows, :]
                if hasattr(block, "toarray"):
                    block = block.toarray()
                agg_mat[g] = np.median(block, axis=0)
        else:
            raise ValueError(f"Unknown agg method: {agg!r}")
    else:
        X_arr = np.asarray(X, dtype=np.float32)
        if agg == "mean":
            agg_mat = np.zeros((n_groups, X_arr.shape[1]), dtype=np.float32)
            for g in range(n_groups):
                m = codes == g
                if m.any():
                    agg_mat[g] = X_arr[m].mean(axis=0)
        elif agg == "sum":
            agg_mat = np.zeros((n_groups, X_arr.shape[1]), dtype=np.float32)
            for g in range(n_groups):
                m = codes == g
                if m.any():
                    agg_mat[g] = X_arr[m].sum(axis=0)
        elif agg == "median":
            agg_mat = np.zeros((n_groups, X_arr.shape[1]), dtype=np.float32)
            for g in range(n_groups):
                m = codes == g
                if m.any():
                    agg_mat[g] = np.median(X_arr[m], axis=0)
        else:
            raise ValueError(f"Unknown agg method: {agg!r}")

    # Build obs for aggregated AnnData
    cat_strs = list(groups.categories)
    parsed = [s.split("§") for s in cat_strs]
    obs_out = pd.DataFrame(parsed, columns=groupby)
    obs_out["n_cells"] = sizes
    obs_out["pseudobulk_method"] = agg
    obs_out.index = pd.Index(
        ["|".join(row) for row in parsed], name="pseudobulk_group"
    )

    from .anndata_io import ONTOLOGY_FIELDS
    for src in ONTOLOGY_FIELDS:
        if src in obs_out.columns or src not in adata.obs.columns:
            continue
        majority = (adata.obs.assign(_g=group_key.values)
                     .groupby("_g")[src]
                     .agg(lambda s: s.astype(str).mode().iloc[0]
                          if len(s.mode()) else ""))
        obs_out[src] = [majority.get(c, "") for c in cat_strs]

    # Carry the study each group's cells came from
    for src in ("dataset_id", "series_id", "gse"):
        if src in obs_out.columns:
            break
        if src in adata.obs.columns:
            majority = (adata.obs.assign(_g=group_key.values)
                        .groupby("_g")[src]
                        .agg(lambda s: s.astype(str).mode().iloc[0]
                             if len(s.mode()) else ""))
            obs_out[src] = [majority.get(c, "") for c in cat_strs]
            break

    # Carry extra obs fields on request
    if return_obs:
        for col in return_obs:
            if col in obs_out.columns or col not in adata.obs.columns:
                continue
            s = adata.obs[col]
            if pd.api.types.is_numeric_dtype(s):
                majority = (pd.DataFrame({"_g": group_key.values, col: s.values})
                             .groupby("_g")[col].mean())
            else:
                majority = (pd.DataFrame({"_g": group_key.values, col: s.astype(str).values})
                             .groupby("_g")[col]
                             .agg(lambda vals: vals.mode().iloc[0] if len(vals.mode()) else ""))
            obs_out[col] = [majority.get(c, np.nan) for c in cat_strs]

    keep = obs_out["n_cells"].to_numpy() >= int(min_cells)
    agg_mat = agg_mat[keep]
    obs_out = obs_out.loc[keep].copy()

    # uns carries provenance so downstream users can see how it was made
    uns = {
        "pseudobulk": {
            "agg": agg,
            "groupby": list(groupby),
            "min_cells": int(min_cells),
            "n_groups_before_filter": int(n_groups),
            "n_groups_after_filter": int(obs_out.shape[0]),
            "n_cells_input": int(adata.n_obs),
            "n_cells_kept": int(obs_out["n_cells"].sum()),
            "source": "real single-cell measurements aggregated by groupby",
        }
    }
    return ad_mod.AnnData(X=agg_mat, obs=obs_out, var=var, uns=uns)


# ────────────────────────────────────────────────────────────────────────────
# Scale harmonisation with the rest of the program
# ────────────────────────────────────────────────────────────────────────────
def is_raw_counts(X) -> bool:
    """True when a matrix holds raw read counts rather than normalized values."""
    arr = np.asarray(X.data if hasattr(X, "data") else X, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return False
    return bool(arr.min() >= 0.0 and np.all(arr == np.floor(arr)))


def normalize_pseudobulk(adata_pb):
    """Put a raw-count aggregate on the scale the rest of GeneVariate reasons on."""
    ad_mod = _require_anndata()
    X = adata_pb.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    if X.size == 0 or not is_raw_counts(X):
        info = dict(adata_pb.uns.get("pseudobulk", {}))
        info["normalization"] = "none (input is not raw counts)"
        info["normalization_skipped"] = True
        info["normalization_skip_reason"] = (
            "matrix is empty" if X.size == 0 else
            "values are not whole numbers - they are already normalized, or "
            "they are a mean of counts; aggregate raw counts with agg='sum' "
            "to produce a library this can normalize")
        adata_pb.uns["pseudobulk"] = info
        return adata_pb

    from genevariate.core.gpl_downloader import normalize_counts

    frame = pd.DataFrame(X.T, index=np.arange(X.shape[1]),
                         columns=[str(s) for s in adata_pb.obs_names])
    norm, n_dropped = normalize_counts(frame)
    keep = norm.index.to_numpy()

    uns = dict(adata_pb.uns)
    info = dict(uns.get("pseudobulk", {}))
    info["normalization"] = "TMM -> CPM -> log2(CPM+1)"
    info["normalization_scope"] = "this platform only"
    info["genes_dropped_low_count"] = int(n_dropped)
    uns["pseudobulk"] = info

    return ad_mod.AnnData(X=norm.to_numpy(dtype=np.float32).T,
                          obs=adata_pb.obs.copy(),
                          var=adata_pb.var.iloc[keep].copy(),
                          uns=uns)


# ────────────────────────────────────────────────────────────────────────────
# AnnData (pseudobulked) → GPL platform DataFrame for GeneVariate
# ────────────────────────────────────────────────────────────────────────────
def pseudobulk_to_platform_df(adata_pb) -> pd.DataFrame:
    """Flatten a pseudobulked AnnData into GeneVariate's platform DataFrame."""
    _require_anndata()
    X = adata_pb.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X)

    # Gene names - prefer feature_name if present
    from .anndata_io import _coerce_gene_index
    gene_names = [str(g) for g in _coerce_gene_index(adata_pb.var)]

    expr = pd.DataFrame(X, index=adata_pb.obs.index, columns=gene_names)
    meta = adata_pb.obs.copy()
    if "series_id" not in meta.columns:
        for src in ("dataset_id", "gse"):
            if src in meta.columns:
                meta.insert(0, "series_id", meta[src].astype(str))
                break
    if "GSM" not in meta.columns:
        meta.insert(0, "GSM", [str(i) for i in adata_pb.obs.index])
    # Order: metadata first, gene columns after.
    return pd.concat([meta, expr], axis=1, copy=False)


#: Fields a cell-level frame may use for the donor and the cell type, in the
#: order they are preferred.
DONOR_FIELDS = ("donor_id", "sample_id", "patient_id", "patient", "batch")
CELLTYPE_FIELDS = ("cell_type", "celltype", "CellType", "cell_ontology_class",
                   "cluster", "leiden", "louvain")
#: Groups thinner than this are dropped.
MIN_CELLS = 10


def choose_groupby(adata) -> list:
    """Donor x cell type for *adata*, falling back to what it does carry."""
    obs = [str(c) for c in adata.obs.columns]
    donor = next((c for c in DONOR_FIELDS if c in obs), None)
    ctype = next((c for c in CELLTYPE_FIELDS if c in obs), None)
    picked = [c for c in (donor, ctype) if c]
    if picked:
        return picked
    picked = [c for c in ("tissue", "cell_type") if c in obs]
    return picked or list(adata.obs.columns[:1])


def aggregate_to_platform(adata, *, groupby=None, agg=None,
                          min_cells: int = MIN_CELLS):
    """Returns ``(frame, pb)`` - the platform DataFrame and the pseudo-bulk AnnData."""
    if groupby is None:
        groupby = choose_groupby(adata)
    if agg is None:
        agg = "sum" if is_raw_counts(adata.X) else "mean"
    pb = pseudobulk(adata, groupby=tuple(groupby), min_cells=int(min_cells),
                    agg=agg)
    pb = normalize_pseudobulk(pb)
    frame = pseudobulk_to_platform_df(pb)
    if "GSM" not in frame.columns:
        frame = frame.reset_index()
        frame = frame.rename(columns={frame.columns[0]: "GSM"})
        frame["GSM"] = frame["GSM"].astype(str)
    return frame, pb


# ────────────────────────────────────────────────────────────────────────────
# Convenience: describe what a pseudo-bulk *would* produce without running it
# ────────────────────────────────────────────────────────────────────────────
def describe_pseudobulk(
    adata,
    groupby: Sequence[str] = ("donor_id", "cell_type"),
    min_cells: int = 10,
) -> Dict[str, Union[int, float, List]]:
    """Return a preview of group sizes without aggregating anything."""
    groupby = list(groupby)
    missing = [g for g in groupby if g not in adata.obs.columns]
    if missing:
        raise KeyError(f"groupby columns not found: {missing}")
    sizes = (adata.obs[groupby].astype(str)
             .agg(" | ".join, axis=1)
             .value_counts()
             .sort_values(ascending=False))
    surviving = sizes[sizes >= int(min_cells)]
    return {
        "n_groups_total": int(len(sizes)),
        "n_groups_kept": int(len(surviving)),
        "n_groups_dropped": int(len(sizes) - len(surviving)),
        "largest_group_cells": int(sizes.iloc[0]) if len(sizes) else 0,
        "median_group_cells": float(sizes.median()) if len(sizes) else 0.0,
        "smallest_kept_group_cells": int(surviving.iloc[-1]) if len(surviving) else 0,
        "top_groups": [
            {"group": str(name), "cells": int(n)}
            for name, n in sizes.head(10).items()
        ],
    }
