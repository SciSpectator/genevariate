"""GeneVariate downstream-analysis modules (enrichment, variability, meta, etc.)."""

from .enrichment import (
    run_enrichr,
    run_prerank_gsea,
    rank_genes_by_condition,
    enrichment_report_markdown,
    DEFAULT_LIBRARIES,
)
from .variability import (
    rank_genes_by_variability,
    run_variability_gsea,
    variability_report_markdown,
    SUPPORTED_METHODS as VARIABILITY_METHODS,
    RECOMMENDED_METHOD as VARIABILITY_DEFAULT_METHOD,
)
from .meta_enrichment import (
    combine_ranks,
    run_meta_enrichment_gsea,
    meta_enrichment_report_markdown,
)
from .bimodality import (
    classify_gene_distribution,
    classify_distributions,
    filter_ranked_by_distribution,
    distribution_summary,
    BIMODAL_TAGS,
    HEAVY_TAGS,
)
from .pseudo_cohorts import (
    embed_labels,
    discover_pseudo_cohorts,
    cohort_summary,
    cohort_pairs,
    PseudoCohortResult,
)
from .rnaseq import (
    compute_qc_metrics,
    cpm_normalize,
    deseq2_size_factors,
    run_deseq2,
    deseq_results_to_ranked,
    counts_to_platform_df,
)
from .cross_modality import (
    infer_modality,
    harmonize_vectors,
    compare_gene_across_modalities,
    gene_coexpression,
    coexpression_consensus,
)

__all__ = [
    "run_enrichr",
    "run_prerank_gsea",
    "rank_genes_by_condition",
    "enrichment_report_markdown",
    "DEFAULT_LIBRARIES",
    "rank_genes_by_variability",
    "run_variability_gsea",
    "variability_report_markdown",
    "VARIABILITY_METHODS",
    "VARIABILITY_DEFAULT_METHOD",
    "combine_ranks",
    "run_meta_enrichment_gsea",
    "meta_enrichment_report_markdown",
    "classify_gene_distribution",
    "classify_distributions",
    "filter_ranked_by_distribution",
    "distribution_summary",
    "BIMODAL_TAGS",
    "HEAVY_TAGS",
    "compute_qc_metrics",
    "cpm_normalize",
    "deseq2_size_factors",
    "run_deseq2",
    "deseq_results_to_ranked",
    "counts_to_platform_df",
    "infer_modality",
    "harmonize_vectors",
    "compare_gene_across_modalities",
    "gene_coexpression",
    "coexpression_consensus",
]
