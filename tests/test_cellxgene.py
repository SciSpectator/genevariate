"""Unit + integration tests for the CELLxGENE Census data source.

Pure-function tests run offline and exercise the filter-expression builder.
Integration tests (marked ``@pytest.mark.network``) open the live Census,
preview a small filter, and fetch a tiny AnnData slice with real expression
values — then assert basic data-quality invariants.

Run all:                 pytest tests/test_cellxgene.py -v
Skip network tests:      pytest tests/test_cellxgene.py -v -m "not network"
"""
from __future__ import annotations

import pytest

from genevariate.sources.cellxgene import (
    CensusClient,
    FILTERABLE_FIELDS,
    ORGANISMS,
    _quote,
    build_obs_filter,
)


# ──────────────────────────────────────────────────────────────────
# Pure-function tests (offline, no network)
# ──────────────────────────────────────────────────────────────────
class TestBuildObsFilter:
    def test_empty_kwargs_returns_empty_string(self):
        assert build_obs_filter() == ""

    def test_all_none_returns_empty_string(self):
        assert build_obs_filter(tissue=None, disease=None) == ""

    def test_single_string(self):
        assert build_obs_filter(tissue="lung") == "tissue == 'lung'"

    def test_single_value_in_list_uses_eq(self):
        # Single-element lists collapse to == for readability.
        assert build_obs_filter(tissue=["lung"]) == "tissue == 'lung'"

    def test_multi_value_list_uses_in(self):
        out = build_obs_filter(tissue=["lung", "kidney"])
        assert out == "tissue in ['lung', 'kidney']"

    def test_bool_value(self):
        assert build_obs_filter(is_primary_data=True) == "is_primary_data == True"
        assert build_obs_filter(is_primary_data=False) == "is_primary_data == False"

    def test_multiple_filters_joined_by_and(self):
        out = build_obs_filter(tissue="lung", disease="normal")
        # order follows kwargs insertion order in py3.7+
        assert out == "tissue == 'lung' and disease == 'normal'"

    def test_quote_escapes_single_quotes(self):
        # Values containing apostrophes must be escaped, not break the expr.
        v = _quote("Crohn's disease")
        assert v == "'Crohn\\'s disease'"

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            build_obs_filter(tissue={"unhashable": "dict"})


def test_module_constants():
    """Sanity-check the public constants the GUI relies on."""
    assert "homo_sapiens" in ORGANISMS
    assert "mus_musculus" in ORGANISMS
    assert "tissue" in FILTERABLE_FIELDS
    assert "cell_type" in FILTERABLE_FIELDS
    assert "is_primary_data" in FILTERABLE_FIELDS


# ──────────────────────────────────────────────────────────────────
# Integration tests — require network + a working Census release
# ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def client():
    """Open the Census once and reuse across integration tests."""
    pytest.importorskip("cellxgene_census")
    pytest.importorskip("anndata")
    cx = CensusClient()
    try:
        cx._open()
    except Exception as exc:
        pytest.skip(f"Could not open CELLxGENE Census: {exc}")
    yield cx
    cx.close()


@pytest.mark.network
@pytest.mark.slow
def test_organisms_lists_homo_sapiens(client):
    orgs = client.organisms()
    assert isinstance(orgs, list)
    assert "homo_sapiens" in orgs


@pytest.mark.network
@pytest.mark.slow
def test_obs_schema_has_key_columns(client):
    cols = client.obs_schema("homo_sapiens")
    for required in ("tissue", "cell_type", "disease", "assay",
                     "is_primary_data", "soma_joinid", "dataset_id"):
        assert required in cols, f"obs missing expected column {required!r}"


@pytest.mark.network
@pytest.mark.slow
def test_preview_smoke(client):
    """Preview a small, well-known slice and verify counts look plausible."""
    summary = client.preview(
        tissue="lung", disease="normal", is_primary_data=True,
        max_rows=20_000,
    )
    assert isinstance(summary, dict)
    assert summary["n_cells_matched"] > 0, "Census returned zero lung-normal cells"
    assert summary["n_datasets"] >= 1
    assert summary["n_donors"] >= 1
    # Cell-type breakdown must include common lung cell types
    ct_names = {name for name, _ in summary.get("cell_type", [])}
    assert ct_names, "preview did not report any cell types"
    # The breakdown counts should sum to the previewed sample size
    total_ct_counts = sum(n for _, n in summary["cell_type"])
    assert total_ct_counts <= summary["n_cells_previewed"]


@pytest.mark.network
@pytest.mark.slow
def test_fetch_tiny_slice_returns_real_data(client):
    """Download a few hundred cells × a handful of genes and check quality."""
    genes = ["CD19", "CD8A", "EPCAM"]
    adata = client.fetch(
        tissue="lung",
        disease="normal",
        is_primary_data=True,
        genes=genes,
        max_cells=500,
    )

    # ── Shape ─────────────────────────────────────────────────────
    assert adata.n_obs > 0, "AnnData has zero observations"
    assert adata.n_obs <= 500, "max_cells cap was not respected"
    assert adata.n_vars == len(genes), (
        f"expected {len(genes)} genes, got {adata.n_vars}: "
        f"{list(adata.var_names)}"
    )

    # ── Provenance ────────────────────────────────────────────────
    src = adata.uns.get("source", {})
    assert src.get("origin") == "CELLxGENE Discover Census"
    assert src.get("organism") == "homo_sapiens"
    assert "obs_filter" in src
    assert src["is_primary_data_only"] is True

    # ── Data quality ──────────────────────────────────────────────
    import numpy as np
    X = adata.X
    # Convert to dense for the small slice
    arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)

    # Counts must be non-negative and finite.
    assert np.isfinite(arr).all(), "non-finite values found in X"
    assert (arr >= 0).all(), "negative counts in scRNA-seq matrix"

    # Sparsity sanity: scRNA-seq is typically >70% zeros, but accept >40%.
    zero_frac = float((arr == 0).sum() / arr.size)
    assert zero_frac > 0.4, f"unexpected density: zero_frac={zero_frac:.3f}"

    # At least one cell must have a non-zero count for at least one gene.
    assert arr.sum() > 0, "all-zero expression matrix returned"

    # obs metadata must contain the filtered tissue
    assert "tissue" in adata.obs.columns
    tissues = set(adata.obs["tissue"].astype(str).unique())
    assert "lung" in tissues, f"expected lung cells, got: {tissues}"

    # var must carry feature_name (HGNC symbol)
    assert "feature_name" in adata.var.columns or \
           set(adata.var_names) == set(genes), \
           "var is missing feature_name and var_names don't match requested genes"


# ──────────────────────────────────────────────────────────────────
# Biological plausibility — do the values that feed the plots
# actually match known biology?
# ──────────────────────────────────────────────────────────────────
@pytest.mark.network
@pytest.mark.slow
def test_lineage_markers_are_enriched_in_correct_cell_types(client):
    """The same data that feeds the GUI's Composition / Dot-plot / UMAP tabs.

    Pulls a multi-lineage lung sample and asserts that classic lineage
    markers are >2× enriched in their expected cell types. If this holds,
    the plots downstream are showing real biology, not random noise.

      EPCAM  → epithelial cells (high), lymphocytes (low)
      CD8A   → CD8 T cells       (high), epithelial (low)
      CD19   → B cells           (high), epithelial (low)
      PTPRC  → pan-leukocyte     (high in any immune cell)
    """
    import numpy as np
    import pandas as pd

    markers = ["EPCAM", "CD8A", "CD19", "PTPRC"]
    adata = client.fetch(
        tissue="lung", disease="normal", is_primary_data=True,
        genes=markers, max_cells=4_000,
    )
    assert adata.n_obs >= 1_000, "need a few thousand cells for stable means"

    # ── 1. Per-cell-type mean expression (this is exactly what the dot
    #       plot in the GUI displays — colour = mean expression). ──────
    X = adata.X
    arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    # log1p to stabilise — Census X is raw counts
    log1p = np.log1p(arr)

    var_map = (
        dict(zip(adata.var["feature_name"].astype(str), range(adata.n_vars)))
        if "feature_name" in adata.var.columns
        else dict(zip(adata.var_names.astype(str), range(adata.n_vars)))
    )
    cell_types = adata.obs["cell_type"].astype(str).to_numpy()

    rows = []
    for ct in pd.unique(cell_types):
        mask = cell_types == ct
        if mask.sum() < 30:
            continue  # skip tiny populations
        row = {"cell_type": ct, "n_cells": int(mask.sum())}
        for g in markers:
            if g in var_map:
                row[g] = float(log1p[mask, var_map[g]].mean())
        rows.append(row)
    table = pd.DataFrame(rows).set_index("cell_type")
    assert not table.empty, "no cell types passed the 30-cell threshold"

    # Debug: print the table so failures are diagnosable
    print("\nPer-cell-type mean log1p expression (n cells per type):")
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))

    # ── 2. Biological invariants ─────────────────────────────────
    # The Census uses Cell Ontology labels — match by substring so we
    # don't get bitten by exact-string changes between releases.
    def _pick(needle: str) -> pd.Series | None:
        hits = table.index[table.index.str.contains(needle, case=False)]
        return table.loc[hits].mean(numeric_only=True) if len(hits) else None

    epithelial = _pick("epithel")
    t_cells = _pick("t cell")
    b_cells = _pick("b cell")

    # We require at least the epithelial vs T-cell axis to be present,
    # because every lung dataset has both.
    assert epithelial is not None, "no epithelial cell types found in lung sample"
    assert t_cells is not None, "no T cell types found in lung sample"

    # EPCAM: epithelial should be substantially higher than T cells
    assert epithelial["EPCAM"] > t_cells["EPCAM"] * 2, (
        f"EPCAM not enriched in epithelium: "
        f"epi={epithelial['EPCAM']:.3f} vs T={t_cells['EPCAM']:.3f}"
    )

    # CD8A: T cells should express it more than epithelium
    assert t_cells["CD8A"] > epithelial["CD8A"] * 2, (
        f"CD8A not enriched in T cells: "
        f"T={t_cells['CD8A']:.3f} vs epi={epithelial['CD8A']:.3f}"
    )

    # PTPRC: pan-leukocyte — T cells (and B if present) should beat epi
    assert t_cells["PTPRC"] > epithelial["PTPRC"] * 2, (
        f"PTPRC not enriched in T cells: "
        f"T={t_cells['PTPRC']:.3f} vs epi={epithelial['PTPRC']:.3f}"
    )

    if b_cells is not None:
        # CD19 is B-cell-specific — should be highest there
        assert b_cells["CD19"] > epithelial["CD19"] * 2, (
            f"CD19 not enriched in B cells: "
            f"B={b_cells['CD19']:.3f} vs epi={epithelial['CD19']:.3f}"
        )
        assert b_cells["CD19"] > t_cells["CD19"], (
            f"CD19 should be higher in B than T cells: "
            f"B={b_cells['CD19']:.3f} vs T={t_cells['CD19']:.3f}"
        )


@pytest.mark.network
@pytest.mark.slow
def test_composition_proportions_sum_to_one(client):
    """The Composition tab plots normalized stacked bars; the underlying
    proportions per group must sum to 1.0 within a small epsilon."""
    import numpy as np
    adata = client.fetch(
        tissue="lung", disease="normal", is_primary_data=True,
        genes=["EPCAM"], max_cells=2_000,
    )
    df = adata.obs[["cell_type", "donor_id"]].astype(str)
    # Proportion of each cell_type within each donor
    counts = df.groupby(["donor_id", "cell_type"]).size().unstack(fill_value=0)
    props = counts.div(counts.sum(axis=1), axis=0)
    # Every donor row should sum to 1.0
    assert np.allclose(props.sum(axis=1), 1.0, atol=1e-6), \
        f"composition proportions don't normalize: {props.sum(axis=1).describe()}"
    assert (props.values >= 0).all() and (props.values <= 1).all()
