"""
GeneVariate — CELLxGENE Discover Census data source.

Fetches real single-cell RNA-seq data from the CZI CELLxGENE Census
(~50M cells, harmonized to the CELLxGENE schema), lazily, via TileDB-SOMA.

Every value returned by this module is a real measurement from a public
scRNA-seq submission. No data is fabricated, simulated, or generated.

Docs
----
* Census overview:   https://chanzuckerberg.github.io/cellxgene-census/
* Schema v5:         https://github.com/chanzuckerberg/single-cell-curation/

Typical use
-----------
>>> from genevariate.sources.cellxgene import CensusClient
>>> cx = CensusClient()
>>> preview = cx.preview(tissue="lung", disease="normal", max_rows=50_000)
>>> preview["n_cells"]
47382
>>> adata = cx.fetch(tissue="lung", disease="normal",
...                  max_cells=20_000, genes=["CD19", "CD8A", "EPCAM"])
>>> adata.n_obs, adata.n_vars
(20000, 3)
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ────────────────────────────────────────────────────────────────────────────
# Lazy imports
# ────────────────────────────────────────────────────────────────────────────
def _require_census():
    try:
        import cellxgene_census
        return cellxgene_census
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "cellxgene_census is required for CELLxGENE Census access.\n"
            "Install with:  pip install cellxgene-census"
        ) from exc


def _require_anndata():
    try:
        import anndata  # noqa: F401
        return anndata
    except Exception as exc:
        raise RuntimeError(
            "anndata is required. Install with: pip install anndata"
        ) from exc


# ────────────────────────────────────────────────────────────────────────────
# Known organisms
# ────────────────────────────────────────────────────────────────────────────
ORGANISMS = (
    "homo_sapiens",
    "mus_musculus",
    "callithrix_jacchus",
    "macaca_mulatta",
    "pan_troglodytes",
)

# Canonical obs fields that are useful in GeneVariate's UI pickers
FILTERABLE_FIELDS = (
    "tissue_general", "tissue", "disease", "assay",
    "cell_type", "sex", "self_reported_ethnicity",
    "development_stage", "suspension_type", "is_primary_data",
)


# ────────────────────────────────────────────────────────────────────────────
# Filter-expression builder
# ────────────────────────────────────────────────────────────────────────────
def _quote(val: str) -> str:
    v = str(val).replace("'", "\\'")
    return f"'{v}'"


def build_obs_filter(**kwargs: Any) -> str:
    """Build a CELLxGENE obs_value_filter string from kwargs.

    Each kwarg is one of ``FILTERABLE_FIELDS`` (or any obs column). Values
    may be a single string or an iterable of strings. Unrecognized keys
    raise. ``is_primary_data`` accepts a bool.

    Returns the empty string if no filters were supplied.
    """
    parts: List[str] = []
    for key, val in kwargs.items():
        if val is None:
            continue
        if isinstance(val, bool):
            parts.append(f"{key} == {'True' if val else 'False'}")
        elif isinstance(val, str):
            parts.append(f"{key} == {_quote(val)}")
        elif isinstance(val, (list, tuple, set)):
            vals = [str(v) for v in val if v is not None and str(v) != ""]
            if not vals:
                continue
            if len(vals) == 1:
                parts.append(f"{key} == {_quote(vals[0])}")
            else:
                joined = ", ".join(_quote(v) for v in vals)
                parts.append(f"{key} in [{joined}]")
        elif isinstance(val, (int, float)):
            parts.append(f"{key} == {val}")
        else:
            raise TypeError(f"Unsupported filter value for {key!r}: {type(val)}")
    return " and ".join(parts)


# ────────────────────────────────────────────────────────────────────────────
# Census client
# ────────────────────────────────────────────────────────────────────────────
class CensusClient:
    """Thin context-managing wrapper around ``cellxgene_census``.

    The Census is opened lazily the first time a method is called and
    reused across calls on the same instance. Call :meth:`close` or use
    it as a context manager to release the underlying TileDB handle.
    """

    def __init__(self, census_version: str = "stable"):
        self.census_version = census_version
        self._census = None

    # Context-manager sugar
    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *exc):
        self.close()

    def _open(self):
        if self._census is None:
            cxg = _require_census()
            self._census = cxg.open_soma(census_version=self.census_version)
        return self._census

    def close(self):
        if self._census is not None:
            try:
                self._census.close()
            except Exception:
                pass
            self._census = None

    # ───── Metadata browsing ─────────────────────────────────────────────
    def organisms(self) -> List[str]:
        """Return organism keys present in this Census release."""
        census = self._open()
        df = census["census_info"]["organisms"].read().concat().to_pandas()
        return df["organism"].tolist()

    def unique_values(
        self,
        column: str,
        organism: str = "homo_sapiens",
        *,
        prefilter: Optional[str] = None,
        limit: Optional[int] = 1000,
    ) -> List[str]:
        """List unique values of an ``obs`` column (with optional prefilter).

        ``prefilter`` can be used to narrow e.g. "all cell_types that appear
        in tissue=='lung'". If ``limit`` is given (default 1000) the result
        is truncated — the Census' ``tissue`` alone has ~700 distinct values,
        which is fine, but ``cell_type`` has thousands.
        """
        census = self._open()
        exp = census["census_data"][organism]
        kwargs = {"column_names": [column]}
        if prefilter:
            kwargs["value_filter"] = prefilter
        tbl = exp.obs.read(**kwargs).concat().to_pandas()
        vals = sorted(tbl[column].dropna().astype(str).unique().tolist())
        if limit is not None:
            vals = vals[:limit]
        return vals

    def obs_schema(self, organism: str = "homo_sapiens") -> List[str]:
        """Return the list of obs column names for the given organism."""
        census = self._open()
        schema = census["census_data"][organism].obs.schema
        return [f.name for f in schema]

    # ───── Previews ──────────────────────────────────────────────────────
    def preview(
        self,
        organism: str = "homo_sapiens",
        *,
        tissue: Optional[Any] = None,
        disease: Optional[Any] = None,
        cell_type: Optional[Any] = None,
        assay: Optional[Any] = None,
        sex: Optional[Any] = None,
        is_primary_data: Optional[bool] = True,
        extra: Optional[Dict[str, Any]] = None,
        max_rows: int = 200_000,
    ) -> Dict[str, Any]:
        """Return a summary of how many cells match a filter, without
        downloading expression data.

        ``is_primary_data`` defaults to True to exclude duplicated cells
        from multi-study reanalyses.
        """
        filt = build_obs_filter(
            tissue=tissue, disease=disease, cell_type=cell_type,
            assay=assay, sex=sex, is_primary_data=is_primary_data,
            **(extra or {}),
        )
        census = self._open()
        exp = census["census_data"][organism]
        read_kwargs = {
            "column_names": [
                "dataset_id", "cell_type", "tissue", "disease",
                "assay", "donor_id", "sex",
            ],
        }
        if filt:
            read_kwargs["value_filter"] = filt
        tbl = exp.obs.read(**read_kwargs).concat().to_pandas()
        n_total = int(len(tbl))
        if n_total > max_rows:
            tbl = tbl.sample(n=max_rows, random_state=0)
        summary: Dict[str, Any] = {
            "filter": filt or "(none)",
            "n_cells_matched": n_total,
            "n_cells_previewed": int(len(tbl)),
            "n_datasets": int(tbl["dataset_id"].nunique()),
            "n_donors": int(tbl["donor_id"].nunique()),
        }
        for col in ("cell_type", "tissue", "disease", "assay", "sex"):
            if col in tbl.columns:
                vc = tbl[col].astype(str).value_counts()
                summary[col] = [(str(k), int(v)) for k, v in vc.head(10).items()]
                summary[f"{col}_n_unique"] = int(tbl[col].nunique())
        return summary

    # ───── The main fetch ───────────────────────────────────────────────
    def fetch(
        self,
        organism: str = "homo_sapiens",
        *,
        tissue: Optional[Any] = None,
        disease: Optional[Any] = None,
        cell_type: Optional[Any] = None,
        assay: Optional[Any] = None,
        sex: Optional[Any] = None,
        is_primary_data: Optional[bool] = True,
        extra: Optional[Dict[str, Any]] = None,
        genes: Optional[Sequence[str]] = None,
        max_cells: Optional[int] = None,
        random_seed: int = 0,
        progress_callback=None,
    ):
        """Materialize an AnnData matching the given filter.

        Parameters
        ----------
        organism
            Organism key (see :data:`ORGANISMS`).
        tissue / disease / cell_type / assay / sex
            Obs-field filters; each may be a string or a list.
        is_primary_data
            Default True — exclude duplicate cells from reanalyses.
        extra
            Any additional obs-column filter, e.g. ``{"development_stage":
            "adult"}``.
        genes
            Optional gene subset (HGNC symbols or Ensembl IDs); if omitted,
            returns all genes in the organism.
        max_cells
            Optional upper cap on the number of cells materialized. Defaults
            to ``None`` — every matching cell is fetched so results are not
            silently subsampled. Set an integer only to cap for memory, in
            which case a random subsample is drawn when the filter matches
            more.
        random_seed
            Seed for the subsample. Fixed for reproducibility.
        progress_callback
            ``callable(message: str)`` invoked with status strings so a
            Tkinter window can show progress.
        """
        _require_anndata()
        cxg = _require_census()
        census = self._open()

        def _log(msg: str):
            if progress_callback:
                try:
                    progress_callback(msg)
                except Exception:
                    pass

        obs_filter = build_obs_filter(
            tissue=tissue, disease=disease, cell_type=cell_type,
            assay=assay, sex=sex, is_primary_data=is_primary_data,
            **(extra or {}),
        )
        _log(f"Counting matching cells ({obs_filter or 'all'})…")

        # Optional random subsample — we need to know total count first
        if max_cells is not None:
            exp = census["census_data"][organism]
            read_kwargs = {"column_names": ["soma_joinid"]}
            if obs_filter:
                read_kwargs["value_filter"] = obs_filter
            joinids = exp.obs.read(**read_kwargs).concat().to_pandas()
            n_match = int(len(joinids))
            _log(f"{n_match:,} cells matched.")
            if n_match > max_cells:
                import numpy as np
                rs = np.random.default_rng(random_seed)
                chosen = rs.choice(joinids["soma_joinid"].to_numpy(),
                                    size=max_cells, replace=False)
                _log(f"Subsampling to {max_cells:,} cells for memory safety.")
                coords_obs = sorted(chosen.tolist())
            else:
                coords_obs = None
        else:
            coords_obs = None

        _log("Fetching expression (this uses real measurements from CELLxGENE)…")
        get_kwargs: Dict[str, Any] = {
            "census": census,
            "organism": organism,
        }
        if obs_filter:
            get_kwargs["obs_value_filter"] = obs_filter
        if coords_obs is not None:
            get_kwargs["obs_coords"] = coords_obs
        if genes:
            get_kwargs["var_value_filter"] = (
                "feature_name in ["
                + ", ".join(_quote(g) for g in genes)
                + "]"
            )
        adata = cxg.get_anndata(**get_kwargs)
        _log(f"Fetched AnnData: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
        # Carry provenance in uns
        adata.uns["source"] = {
            "origin":       "CELLxGENE Discover Census",
            "organism":     organism,
            "census_version": self.census_version,
            "obs_filter":   obs_filter or "(none)",
            "max_cells":    max_cells,
            "is_primary_data_only": bool(is_primary_data),
            "note":         "All expression values are real measurements from "
                            "public scRNA-seq submissions harmonized by the "
                            "CELLxGENE Census.",
        }
        return adata
