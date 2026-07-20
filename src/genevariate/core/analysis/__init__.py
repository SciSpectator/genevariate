"""GeneVariate downstream-analysis modules (enrichment, variability, meta, etc.).

Submodules are imported **lazily** (PEP 562): accessing a public symbol imports
only the submodule that provides it, so the heavy, optional dependencies some of
them carry — ``pydeseq2``/``torch`` (rnaseq), ``decoupler`` (activity),
``inmoose``/``harmonypy`` (integration) — are pulled into memory only when a
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
    # variability
    "rank_genes_by_variability": "variability",
    "run_variability_gsea": "variability",
    "variability_report_markdown": "variability",
    # meta-enrichment
    "combine_ranks": "meta_enrichment",
    "run_meta_enrichment_gsea": "meta_enrichment",
    "meta_enrichment_report_markdown": "meta_enrichment",
    # bimodality (diptest/sklearn optional, lazily probed in the module)
    "classify_gene_distribution": "bimodality",
    "classify_distributions": "bimodality",
    "filter_ranked_by_distribution": "bimodality",
    "distribution_summary": "bimodality",
    "BIMODAL_TAGS": "bimodality",
    "HEAVY_TAGS": "bimodality",
    # pseudo-cohorts
    "embed_labels": "pseudo_cohorts",
    "discover_pseudo_cohorts": "pseudo_cohorts",
    "cohort_summary": "pseudo_cohorts",
    "cohort_pairs": "pseudo_cohorts",
    "PseudoCohortResult": "pseudo_cohorts",
    # rnaseq (pydeseq2/torch — loaded only when run_deseq2 is called)
    "compute_qc_metrics": "rnaseq",
    "cpm_normalize": "rnaseq",
    "deseq2_size_factors": "rnaseq",
    "run_deseq2": "rnaseq",
    "deseq_results_to_ranked": "rnaseq",
    "counts_to_platform_df": "rnaseq",
    # cross-modality
    "infer_modality": "cross_modality",
    "harmonize_vectors": "cross_modality",
    "compare_gene_across_modalities": "cross_modality",
    "gene_coexpression": "cross_modality",
    "coexpression_consensus": "cross_modality",
    # integration (inmoose/harmonypy optional)
    "common_gene_matrix": "integration",
    "combat_correct": "integration",
    "harmony_embed": "integration",
    # activity (decoupler optional)
    "tf_activity": "activity",
    "pathway_activity": "activity",
    "run_activity": "activity",
}

# exported name -> (submodule, differing attribute name in that submodule)
_ALIASES = {
    "VARIABILITY_METHODS": ("variability", "SUPPORTED_METHODS"),
    "VARIABILITY_DEFAULT_METHOD": ("variability", "RECOMMENDED_METHOD"),
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
