"""
GeneVariate — Pseudo-bulk aggregation of AnnData.

What this module does, in one paragraph
----------------------------------------
Pseudo-bulk is not fake data. It is a groupby-aggregation of real,
measured single-cell expression values — the same kind of operation that
already produces a single "bulk" expression value from millions of cells
in a tube (that is what bulk RNA-seq is). Here we aggregate explicitly,
under user control, so that per-cell measurements from CELLxGENE / HCA /
etc. can be compared to GEO bulk data on the same axes.

Every value produced by this module is the mean, sum, or median of real
measurements that came directly from a public database. Nothing is
simulated.

The helpers here return objects with transparent provenance: every
aggregated row carries an ``n_cells`` column telling the user how many
real cells contributed to it.
"""

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
    """Aggregate cells of ``adata`` into synthetic-sample AnnData.

    Parameters
    ----------
    adata
        Cell-level AnnData (from CELLxGENE / HCA / local .h5ad).
    groupby
        Columns in ``adata.obs`` to group by. The resulting AnnData has
        one row per unique combination of values in these columns.
    agg
        How to collapse cells within a group: mean (default), sum, or
        median. ``mean`` is the standard choice for normalized data;
        ``sum`` is correct for raw counts and downstream DESeq2/edgeR.
    min_cells
        Drop groups whose cell count is below this threshold (default
        10). Prevents statistically meaningless 1–2 cell "samples".
    layer
        If given, aggregate ``adata.layers[layer]`` instead of
        ``adata.X``. ``adata.raw.X`` is used when ``use_raw=True``.
    use_raw
        If True, aggregate ``adata.raw.X``.
    return_obs
        Extra ``obs`` columns to carry into the resulting AnnData's obs
        table (in addition to the groupby columns and ``n_cells``).
        Values are aggregated per-group by majority vote for strings and
        mean for numerics.

    Returns
    -------
    A new AnnData whose:
        * n_obs = number of groups surviving ``min_cells``
        * obs   = one row per group, with:
                    - the groupby columns,
                    - ``n_cells`` (count of real cells in this group),
                    - ``pseudobulk_method`` (what agg was used),
                    - Classified_* aliases carried through from normalized fields,
                    - any ``return_obs`` fields
        * var   = same as input adata.var (genes)
        * X     = aggregated expression matrix (dense float32)
    """
    ad_mod = _require_anndata()
    if adata is None:
        raise ValueError("adata is None")
    groupby = list(groupby)
    missing = [g for g in groupby if g not in adata.obs.columns]
    if missing:
        raise KeyError(f"groupby columns not found in adata.obs: {missing}")

    # Choose the expression matrix — always real measured data
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
        # Sparse → densify per-group by slicing the matrix (memory-safe:
        # we never densify the whole thing).
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
            # Median requires full dense per-group; do it group by group to
            # keep peak memory bounded.
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
    # Unpack the composite group key back into the original groupby columns
    cat_strs = list(groups.categories)
    parsed = [s.split("§") for s in cat_strs]
    obs_out = pd.DataFrame(parsed, columns=groupby)
    obs_out["n_cells"] = sizes
    obs_out["pseudobulk_method"] = agg
    obs_out.index = pd.Index(
        ["|".join(row) for row in parsed], name="pseudobulk_group"
    )

    # Forward Classified_* aliases the same way anndata_io does
    from .anndata_io import ONTOLOGY_LABEL_MAP
    for src, alias in ONTOLOGY_LABEL_MAP.items():
        if src in groupby and alias not in obs_out.columns:
            obs_out[alias] = obs_out[src]
        elif src in adata.obs.columns and alias not in obs_out.columns:
            # Aggregate non-groupby ontology fields by majority vote per group
            majority = (adata.obs.assign(_g=group_key.values)
                         .groupby("_g")[src]
                         .agg(lambda s: s.astype(str).mode().iloc[0]
                              if len(s.mode()) else ""))
            obs_out[alias] = [majority.get(c, "") for c in cat_strs]
            obs_out[src] = obs_out[alias]

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

    # Apply min_cells filter AFTER building obs (so we never drop data silently)
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
# AnnData (pseudobulked) → GPL platform DataFrame for GeneVariate
# ────────────────────────────────────────────────────────────────────────────
def pseudobulk_to_platform_df(adata_pb) -> pd.DataFrame:
    """Flatten a pseudobulked AnnData into GeneVariate's platform DataFrame.

    Output columns:
        * gene columns (from ``adata_pb.var_names``)
        * obs columns (groupby fields, ``n_cells``, ``pseudobulk_method``,
          Classified_* aliases, …)
    """
    _require_anndata()
    X = adata_pb.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X)

    # Gene names — prefer feature_name if present
    from .anndata_io import _coerce_gene_index
    gene_names = [str(g) for g in _coerce_gene_index(adata_pb.var)]

    expr = pd.DataFrame(X, index=adata_pb.obs.index, columns=gene_names)
    meta = adata_pb.obs.copy()
    # Order: metadata first (so LabelEnrichmentWindow finds Classified_* early),
    # gene columns after.
    return pd.concat([meta, expr], axis=1, copy=False)


# ────────────────────────────────────────────────────────────────────────────
# Convenience: describe what a pseudo-bulk *would* produce without running it
# ────────────────────────────────────────────────────────────────────────────
def describe_pseudobulk(
    adata,
    groupby: Sequence[str] = ("donor_id", "cell_type"),
    min_cells: int = 10,
) -> Dict[str, Union[int, float, List]]:
    """Return a preview of group sizes without aggregating anything.

    Used by the UI to show: "This will produce N groups (dropping M below
    min_cells); largest group = X cells, median = Y."
    """
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
