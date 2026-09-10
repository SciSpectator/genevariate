"""GeneVariate downstream-analysis modules (enrichment, variability, meta, etc.).

Submodules are imported **lazily** (PEP 562): accessing a public symbol imports
only the submodule that provides it, so the heavy, optional dependencies some of
them carry - ``decoupler`` (activity),
``inmoose``/``harmonypy`` (integration) - are pulled into memory only when a
feature that needs them is actually used. Importing ``genevariate.core.analysis``
(or a light path such as enrichment or bimodality) no longer drags the GPU/tensor
stack in at startup.
"""
from importlib import import_module

# exported symbol -> submodule that defines it (attribute name is identical)
_EXPORTS = {
    # enrichment (numpy/pandas/scipy; gseapy optional, itself lazy)
    "run_enrichr": "enrichment",
    "run_prerank_gsea": "enrichment",
    "rank_genes_by_condition": "enrichment",
    "enrichment_report_markdown": "enrichment",
    "benjamini_hochberg": "enrichment",
    "DEFAULT_LIBRARIES": "enrichment",
    # overdispersion / study-clumping corrections (numpy + scipy)
    "group_counts": "overdispersion",
    "estimate_rho": "overdispersion",
    "effective_sample_size": "overdispersion",
    "design_effect": "overdispersion",
    "enrichment_diagnostics": "overdispersion",
    # comparing many regions against each other (numpy + scipy; sklearn for
    # region_separability only)
    "overlap_matrix": "region_comparison",
    "enrichment_matrix": "region_comparison",
    "pairwise_differential": "region_comparison",
    "heterogeneity": "region_comparison",
    "cluster_regions": "region_comparison",
    "region_separability": "region_comparison",
    "summarize_comparison": "region_comparison",
    # regions on different platforms: one stratum each, effects pooled
    # (numpy + scipy)
    "pooled_label_enrichment": "pooled_enrichment",
    "summarize_pooled_enrichment": "pooled_enrichment",
    # calibrated P(label | genes) over a box (scikit-learn)
    "BoxLabelModel": "box_model",
    "fit_label_model": "box_model",
    "reliability_curve": "box_model",
    "integrate_box": "box_model",
    "relaxation_attribution": "box_model",
    # meta-enrichment
    # bimodality (diptest/sklearn optional, lazily probed in the module)
    "classify_gene_distribution": "bimodality",
    "classify_distributions": "bimodality",
    "filter_ranked_by_distribution": "bimodality",
    "distribution_summary": "bimodality",
    "BIMODAL_TAGS": "bimodality",
    "HEAVY_TAGS": "bimodality",
    "optimal_bins": "bimodality",
    # region selection - explicit bounds / SD tail / quantile, the three ways
    # the window's brush and a sentence can both name the same interval
    "RegionBounds": "region_select",
    "resolve_bounds": "region_select",
    "region_mask": "region_select",
    # selecting samples by several labels at once - one masking rule for the
    # grouping dialog, the Region Analysis window and the assistant (pandas)
    "MISSING": "label_query",
    "RESERVED_COLUMNS": "label_query",
    "QueryResult": "label_query",
    "queryable_columns": "label_query",
    "column_values": "label_query",
    "query_mask": "label_query",
    "describe_criteria": "label_query",
    "run_query": "label_query",
    # what a region is enriched for - the Fisher/BH grid behind the Region
    # Analysis window's Enrichment tab (numpy + scipy)
    "build_enrichment_cells": "region_enrichment",
    "region_label_enrichment": "region_enrichment",
    "summarize_region_enrichment": "region_enrichment",
    "region_composition": "region_enrichment",
    "summarize_region_composition": "region_enrichment",
    # comparing named groups of one gene's values - the Compare Distributions
    # window's Distance Matrix and Statistics tabs (numpy + scipy)
    "group_summary": "distribution_compare",
    "pairwise_distances": "distribution_compare",
    "distance_matrix": "distribution_compare",
    "summarize_comparison_stats": "distribution_compare",
    "DISTANCE_METRICS": "distribution_compare",
    # comparing whole platforms - the Cross-Platform Analysis window's gene
    # inventory, per-gene DE/conserved tests, batch metrics and corrections
    "gene_inventory": "cross_platform",
    "median_center_normalize": "cross_platform",
    "quantile_normalize_cross": "cross_platform",
    "platform_similarity": "cross_platform",
    "analyze_platforms": "cross_platform",
    "summarize_cross_platform": "cross_platform",
    # pseudo-cohorts
    # cross-modality
    "infer_modality": "cross_modality",
    "modality_from_category": "cross_modality",
    "harmonize_vectors": "cross_modality",
    "compare_gene_across_modalities": "cross_modality",
    "gene_coexpression": "cross_modality",
    "coexpression_consensus": "cross_modality",
    # integration (inmoose/harmonypy optional)
    "common_gene_matrix": "integration",
    "harmony_embed": "integration",
    # label-aware ML (scikit-learn; umap optional)
    # cell-level single-cell analysis (numpy/pandas; sklearn + umap only for
    # cell_embedding, and only when the AnnData carries no X_umap)
    "gene_index": "single_cell",
    "categorical_obs_columns": "single_cell",
    "cell_composition": "single_cell",
    "CompositionResult": "single_cell",
    "cell_embedding": "single_cell",
    "marker_dotplot": "single_cell",
    "DotPlotResult": "single_cell",
    "cell_qc": "single_cell",
    "summarize_cell_qc": "single_cell",
}

# exported name -> (submodule, differing attribute name in that submodule)
_ALIASES = {
    "VARIABILITY_METHODS": ("variability", "SUPPORTED_METHODS"),
    "VARIABILITY_DEFAULT_METHOD": ("variability", "RECOMMENDED_METHOD"),
    # Two different ComBats, exported under two different names. Both
    # submodules call theirs ``combat_correct``, but they are not
    # interchangeable: the cross_platform one takes a genes x samples matrix
    # with one batch id per column and returns (corrected, method), while the
    # integration one takes {source -> frame} and returns a corrected dict.
    # Mapping one exported name to both left the later entry silently winning,
    # so a snippet asking for a matrix ComBat was handed the dict one.
    "combat_correct_matrix": ("cross_platform", "combat_correct"),
    "combat_correct_sources": ("integration", "combat_correct"),
}

__all__ = sorted(list(_EXPORTS) + list(_ALIASES))


def __getattr__(name):  # PEP 562 lazy attribute loading
    if name in _ALIASES:
        mod, attr = _ALIASES[name]
        return getattr(import_module(f"{__name__}.{mod}"), attr)
    mod = _EXPORTS.get(name)
    if mod is not None:
        return getattr(import_module(f"{__name__}.{mod}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
