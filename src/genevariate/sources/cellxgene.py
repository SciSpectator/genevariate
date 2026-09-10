"""
GeneVariate - CELLxGENE Discover Census data source.

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
# Memory guard
# ────────────────────────────────────────────────────────────────────────────
CENSUS_DENSE_ITEMSIZE = 8
CENSUS_MEM_FRACTION = 0.5


class CensusTooLargeError(MemoryError):
    """A census query whose dense matrix would not fit in available memory."""


def _available_ram_bytes() -> Optional[int]:
    """Bytes of RAM available right now, or None if it cannot be read."""
    from genevariate.utils.memory import available_ram_bytes
    return available_ram_bytes()


def _mem_fraction() -> float:
    """Ceiling as a fraction of available RAM (``GENEVARIATE_CENSUS_MEM_FRACTION``)."""
    from genevariate.utils.memory import mem_fraction
    return mem_fraction("GENEVARIATE_CENSUS_MEM_FRACTION")


def estimate_fetch_bytes(n_cells: int, n_genes: int) -> int:
    """Bytes the dense matrix for ``n_cells x n_genes`` would occupy."""
    return int(n_cells) * int(n_genes) * CENSUS_DENSE_ITEMSIZE


def check_fetch_fits(n_cells: int, n_genes: int, *,
                     capped: bool = False) -> None:
    """Raise :class:`CensusTooLargeError` if this fetch cannot fit in RAM.

    ``capped`` says the count already reflects a caller-supplied ``max_cells``.
    """
    need = estimate_fetch_bytes(n_cells, n_genes)
    avail = _available_ram_bytes()
    if avail is None:
        return
    budget = int(avail * _mem_fraction())
    if need <= budget:
        return

    gib = 1024 ** 3
    fits = max(1, int(budget // (int(n_genes) * CENSUS_DENSE_ITEMSIZE)))
    advice = (
        f"Narrow the filter (tissue, cell_type, disease, assay), or ask for "
        f"specific genes, or set max_cells explicitly - about {fits:,} cells "
        f"would fit right now."
        if not capped else
        f"Lower max_cells to about {fits:,}, narrow the filter, or ask for "
        f"specific genes."
    )
    raise CensusTooLargeError(
        f"This query would materialize {n_cells:,} cells x {n_genes:,} genes "
        f"= {need / gib:,.1f} GiB, but only {avail / gib:,.1f} GiB of RAM is "
        f"available ({budget / gib:,.1f} GiB usable for one fetch). Refusing "
        f"rather than subsampling, because a silent subsample would change "
        f"the result without saying so. {advice}"
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
        # The dated release the alias resolved to, filled on open.
        self.resolved_version = census_version

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
            self.resolved_version = self._resolve_version(cxg)
        return self._census

    def _resolve_version(self, cxg) -> str:
        """The dated release this handle opened, not the alias asked for.

        ``stable`` and ``latest`` are moving pointers: the same call a month
        apart opens different cells. Recording the alias as provenance
        therefore records nothing a reader can act on - they cannot get back
        the data the run used. The alias stays as the request; the date is
        what goes in the record.
        """
        asked = str(self.census_version)
        try:
            directory = cxg.get_census_version_directory()
        except Exception:
            return asked
        if not isinstance(directory, dict):
            return asked
        entry = directory.get(asked)
        if isinstance(entry, dict):
            for field in ("release_build", "census_version", "alias"):
                val = entry.get(field)
                if isinstance(val, str) and val and val != asked:
                    return val
        # Otherwise find the dated key whose alias is the one we asked for.
        for key, val in directory.items():
            if key == asked or not isinstance(val, dict):
                continue
            aliases = val.get("alias")
            aliases = [aliases] if isinstance(aliases, str) else (aliases or [])
            if asked in aliases:
                return str(key)
        return asked

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
        is truncated - the Census' ``tissue`` alone has ~700 distinct values,
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
        # Distinct dataset/donor counts are taken over every matching row.
        # They used to be computed after the ``max_rows`` subsample below, so a
        # broad filter reported the sample's counts under a label the UI reads
        # as a total ("Distinct datasets") - understated, with nothing saying
        # so. Only the per-column breakdowns are sampled, which is what
        # ``n_cells_previewed`` describes.
        n_datasets = int(tbl["dataset_id"].nunique())
        n_donors = int(tbl["donor_id"].nunique())
        if n_total > max_rows:
            tbl = tbl.sample(n=max_rows, random_state=0)
        summary: Dict[str, Any] = {
            "filter": filt or "(none)",
            "n_cells_matched": n_total,
            "n_cells_previewed": int(len(tbl)),
            "n_datasets": n_datasets,
            "n_donors": n_donors,
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
            Default True - exclude duplicate cells from reanalyses.
        extra
            Any additional obs-column filter, e.g. ``{"development_stage":
            "adult"}``.
        genes
            Optional gene subset (HGNC symbols or Ensembl IDs); if omitted,
            returns all genes in the organism.
        max_cells
            Optional upper cap on the number of cells materialized. Defaults
            to ``None`` - every matching cell is fetched so results are not
            silently subsampled. A query too large for the machine raises
            :class:`CensusTooLargeError` rather than being capped.
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

        exp = census["census_data"][organism]
        read_kwargs = {"column_names": ["soma_joinid"]}
        if obs_filter:
            read_kwargs["value_filter"] = obs_filter
        joinids = exp.obs.read(**read_kwargs).concat().to_pandas()
        n_match = int(len(joinids))
        _log(f"{n_match:,} cells matched.")

        if genes:
            n_genes_est = len(list(genes))
        else:
            try:
                n_genes_est = int(exp.ms["RNA"].var.count)
            except Exception:
                n_genes_est = 60_530

        if max_cells is not None and n_match > max_cells:
            import numpy as np
            rs = np.random.default_rng(random_seed)
            chosen = rs.choice(joinids["soma_joinid"].to_numpy(),
                                size=max_cells, replace=False)
            _log(f"Subsampling to {max_cells:,} cells (max_cells).")
            coords_obs = sorted(chosen.tolist())
            n_fetch = int(max_cells)
            capped = True
        else:
            coords_obs = None
            n_fetch = n_match
            capped = max_cells is not None

        if n_fetch == 0:
            raise ValueError(
                f"No cells match this filter ({obs_filter or 'all'}). "
                "Check the tissue/cell_type spelling against preview()."
            )

        check_fetch_fits(n_fetch, n_genes_est, capped=capped)
        _log(f"Estimated {estimate_fetch_bytes(n_fetch, n_genes_est) / 1024**3:,.1f} "
             f"GiB for {n_fetch:,} cells × {n_genes_est:,} genes.")

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
            # Both: what was asked for, and what that turned out to be. A
            # record that says only "stable" cannot be replayed.
            "census_version": self.census_version,
            "census_release": self.resolved_version,
            "obs_filter":   obs_filter or "(none)",
            "max_cells":    max_cells,
            "is_primary_data_only": bool(is_primary_data),
            "note":         "All expression values are real measurements from "
                            "public scRNA-seq submissions harmonized by the "
                            "CELLxGENE Census.",
        }
        return adata
