"""Cell-level analysis of an AnnData matrix, free of any GUI."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "gene_index", "categorical_obs_columns",
    "CompositionResult", "cell_composition",
    "cell_embedding",
    "DotPlotResult", "marker_dotplot",
    "cell_qc", "summarize_cell_qc",
]

# obs columns worth offering first
PREFERRED_OBS = (
    "cell_type", "tissue", "tissue_general", "disease", "assay", "donor_id",
    "sex", "development_stage", "self_reported_ethnicity", "dataset_id",
    "suspension_type", "is_primary_data",
)


def gene_index(adata) -> pd.Index:
    """The gene symbols of ``adata.var``, under the same rules as loading."""
    from genevariate.utils.anndata_io import _coerce_gene_index
    return _coerce_gene_index(adata.var)


def categorical_obs_columns(adata) -> List[str]:
    """obs columns that can be grouped by, preferred census names first."""
    try:
        cols = list(adata.obs.columns.astype(str))
    except Exception:
        return []
    out = [c for c in PREFERRED_OBS if c in cols]
    for c in cols:
        if c in out:
            continue
        dt = adata.obs[c].dtype
        if dt == "object" or str(dt).startswith("category"):
            out.append(c)
    return out


def _dense(block) -> np.ndarray:
    """A dense float32 view of a possibly-sparse expression block."""
    shape = getattr(block, "shape", None)
    if shape is not None and len(shape) == 2:
        from genevariate.utils.memory import require_fits
        n_rows, n_cols = int(shape[0]), int(shape[1])
        require_fits(
            n_rows * n_cols * 4,
            f"Densifying {n_rows:,} cells x {n_cols:,} genes",
            advice="Subsample the cells, or ask for fewer genes.",
        )
    if hasattr(block, "toarray"):
        block = block.toarray()
    return np.asarray(block, dtype=np.float32)


# ---------------------------------------------------------------- composition
@dataclass
class CompositionResult:
    """Cell counts of ``stack_by`` within each ``group_by``."""
    counts: pd.DataFrame
    fractions: pd.DataFrame
    group_by: str
    stack_by: str
    n_cells: int
    dropped_groups: int = 0
    n_groups_total: int = 0
    n_stack_total: int = 0
    other_column: bool = False


def cell_composition(adata, group_by: str, stack_by: str, *,
                     top_stack: int = 12, max_groups: int = 40,
                     ) -> CompositionResult:
    """Cross-tabulate ``stack_by`` within ``group_by`` over the cells."""
    obs = adata.obs
    for name, col in (("group_by", group_by), ("stack_by", stack_by)):
        if col not in obs.columns:
            raise KeyError(f"{name}={col!r} is not an obs column. "
                           f"Available: {', '.join(map(str, obs.columns[:12]))}")
    pair = obs[[group_by, stack_by]].astype(str)
    ct = pair.groupby([group_by, stack_by]).size().unstack(fill_value=0)

    n_stack_total = int(ct.shape[1])
    totals = ct.sum(axis=0).sort_values(ascending=False)
    other_column = False
    if n_stack_total > top_stack:
        keep = totals.head(top_stack).index.tolist()
        other = ct.drop(columns=keep).sum(axis=1)
        ct = ct[keep]
        ct["other"] = other
        other_column = True
    else:
        ct = ct[totals.index]

    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]
    n_groups_total = int(len(ct))
    dropped = max(0, n_groups_total - max_groups)
    if dropped:
        ct = ct.iloc[:max_groups]

    counts = ct.astype(int)
    fractions = counts.div(counts.sum(axis=1).replace(0, 1), axis=0)
    return CompositionResult(
        counts=counts, fractions=fractions, group_by=group_by,
        stack_by=stack_by, n_cells=int(adata.n_obs),
        dropped_groups=dropped, n_groups_total=n_groups_total,
        n_stack_total=n_stack_total, other_column=other_column)


# ------------------------------------------------------------------ embedding
def cell_embedding(adata, *, max_cells: int = 20_000, seed: int = 0,
                   ) -> Tuple[np.ndarray, str, np.ndarray]:
    """A 2-D layout of the cells, as ``(coords, method, cell_indices)``."""
    n_obs = int(adata.n_obs)
    if n_obs > max_cells:
        rs = np.random.default_rng(seed)
        idx = np.sort(rs.choice(n_obs, size=max_cells, replace=False))
    else:
        idx = np.arange(n_obs)

    obsm = getattr(adata, "obsm", None)
    if obsm is not None and "X_umap" in obsm:
        return np.asarray(obsm["X_umap"])[idx], "X_umap (from AnnData)", idx

    X = _dense(adata.X[idx])
    if X.size and X.min() >= 0.0 and X.max() > 50:
        X = np.log1p(X)

    k = min(30, min(X.shape) - 1) if min(X.shape) > 1 else 0
    if k <= 2:
        return X[:, :2], "raw (too few features for PCA)", idx

    from sklearn.decomposition import TruncatedSVD
    pcs = TruncatedSVD(n_components=k, random_state=seed).fit_transform(X)
    try:
        import umap
        coords = umap.UMAP(n_components=2, random_state=seed,
                           n_neighbors=min(15, pcs.shape[0] - 1)
                           ).fit_transform(pcs)
        return coords, f"UMAP on {k} PCs (subset={len(idx):,})", idx
    except Exception:
        return (pcs[:, :2],
                f"PCA[1,2] (subset={len(idx):,} - install umap-learn for UMAP)",
                idx)


# -------------------------------------------------------------------- dot plot
@dataclass
class DotPlotResult:
    """Per group, the mean expression and the fraction of cells expressing."""
    mean: pd.DataFrame
    fraction: pd.DataFrame
    group_by: str
    genes: List[str]
    missing: List[str] = field(default_factory=list)
    dropped_groups: int = 0
    n_groups_total: int = 0

    @property
    def table(self) -> pd.DataFrame:
        """One row per group x gene, long form."""
        m = self.mean.stack().rename("mean_expression")
        f = self.fraction.stack().rename("fraction_expressing")
        out = pd.concat([m, f], axis=1).reset_index()
        out.columns = [self.group_by, "gene",
                       "mean_expression", "fraction_expressing"]
        return out


def marker_dotplot(adata, genes: Sequence[str], group_by: str, *,
                   max_groups: int = 40) -> DotPlotResult:
    """Mean expression and expressing fraction of ``genes`` per ``group_by``."""
    if group_by not in adata.obs.columns:
        raise KeyError(f"{group_by!r} is not an obs column.")
    wanted = [str(g).strip() for g in genes if str(g).strip()]
    if not wanted:
        raise ValueError("No gene symbols given.")

    var_list = [str(v) for v in gene_index(adata)]
    pos = {g: i for i, g in enumerate(var_list)}
    found = [g for g in wanted if g in pos]
    missing = [g for g in wanted if g not in pos]
    if not found:
        raise KeyError(f"None of {wanted} are in this dataset.")

    X_sub = _dense(adata.X[:, [pos[g] for g in found]])
    groups = adata.obs[group_by].astype(str).to_numpy()
    vc = pd.Series(groups).value_counts()
    n_groups_total = int(len(vc))
    cats = list(vc.head(max_groups).index)

    mean_mat = np.zeros((len(cats), len(found)), dtype=float)
    frac_mat = np.zeros((len(cats), len(found)), dtype=float)
    for gi, c in enumerate(cats):
        m = groups == c
        if not m.any():
            continue
        block = X_sub[m]
        mean_mat[gi] = block.mean(axis=0)
        frac_mat[gi] = (block > 0).mean(axis=0)

    return DotPlotResult(
        mean=pd.DataFrame(mean_mat, index=cats, columns=found),
        fraction=pd.DataFrame(frac_mat, index=cats, columns=found),
        group_by=group_by, genes=found, missing=missing,
        dropped_groups=max(0, n_groups_total - len(cats)),
        n_groups_total=n_groups_total)


# -------------------------------------------------------------------------- QC
def cell_qc(adata, *, mito_prefix: str = "MT-",
            group_by: Optional[str] = None) -> pd.DataFrame:
    """Per-cell ``n_genes``, ``total_counts`` and ``pct_mito``."""
    X = adata.X
    if hasattr(X, "toarray"):
        total = np.asarray(X.sum(axis=1)).ravel().astype(float)
        n_genes = np.asarray((X > 0).sum(axis=1)).ravel().astype(float)
    else:
        Xa = np.asarray(X, dtype=float)
        total = Xa.sum(axis=1)
        n_genes = (Xa > 0).sum(axis=1).astype(float)

    pct_mito = np.zeros_like(total)
    prefix = str(mito_prefix or "").strip()
    if prefix:
        var_list = [str(g) for g in gene_index(adata)]
        cols = [i for i, g in enumerate(var_list)
                if g.upper().startswith(prefix.upper())]
        if cols:
            sub = X[:, cols]
            mito_total = (np.asarray(sub.sum(axis=1)).ravel()
                          if hasattr(sub, "toarray")
                          else np.asarray(sub).sum(axis=1))
            with np.errstate(invalid="ignore", divide="ignore"):
                pct_mito = 100.0 * mito_total / np.where(total > 0, total, 1.0)

    out = pd.DataFrame({"n_genes": n_genes, "total_counts": total,
                        "pct_mito": pct_mito}, index=adata.obs.index)
    if group_by and group_by in adata.obs.columns:
        out.insert(0, group_by, adata.obs[group_by].astype(str).to_numpy())
    return out


def summarize_cell_qc(qc: pd.DataFrame, *, mito_prefix: str = "MT-",
                      group_by: Optional[str] = None) -> Dict[str, Any]:
    """Medians and quartiles of the QC columns, per group when asked."""
    metrics = ["n_genes", "total_counts", "pct_mito"]
    out: Dict[str, Any] = {
        "n_cells": int(len(qc)),
        "mito_prefix": mito_prefix,
        "mito_detected": bool(qc["pct_mito"].to_numpy().any()),
        "overall": qc[metrics].describe().loc[
            ["min", "25%", "50%", "75%", "max"]].to_dict(),
    }
    if group_by and group_by in qc.columns:
        out["per_group"] = qc.groupby(group_by)[metrics].median()
    return out
