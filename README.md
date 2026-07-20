<p align="center">
  <img src="docs/logo.png" alt="GeneVariate" width="260">
</p>

<p align="center">
  <strong>GeneVariate</strong><br>
  <em>Variability-aware cross-technology gene-expression analysis with LLM-curated labels</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-2.1.0-blueviolet?style=for-the-badge" alt="v2.1.0">
</p>

<p align="center">
  <a href="#license"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Backend-Ollama-orange.svg?logo=ollama&logoColor=white" alt="Ollama">
  <img src="https://img.shields.io/badge/Data-Microarray%20%7C%20RNA--seq%20%7C%20Methylation-1565C0" alt="Technologies">
  <img src="https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey" alt="Platform">
  <img src="https://img.shields.io/badge/Docker-Supported-2496ED?logo=docker&logoColor=white" alt="Docker">
</p>

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Features](#features)
4. [Architecture](#architecture)
5. [Usage](#usage)
6. [Novel Analysis Methods](#novel-analysis-methods)
7. [Development](#development)
8. [Citation](#citation)
9. [License](#license)

---

## Overview

**GeneVariate** is a local-first gene-expression analysis platform. It ingests datasets from
multiple technologies (Affymetrix / Illumina microarrays, bulk RNA-seq via
[ARCHS4](https://maayanlab.cloud/archs4/), methylation peaks, scRNA-seq pseudobulk) into a
single canonical format, then lets you ask pathway questions that standard tools cannot —
*which pathways change in variance rather than mean*, *which are driven by bimodal on/off
switches*, *which survive cross-platform meta-analysis*.

The biological metadata attached to each sample (tissue, condition, treatment) is extracted
automatically by a local LLM (`gemma4:e2b` via [Ollama](https://ollama.com/)), following the
[LLM-Label-Extractor v2.2](https://github.com/SciSpectator/LLM-Label-Extractor) prompt design
with multi-value and coded-value support.

**All inference runs on your hardware.** No API keys, no cloud, no data exfiltration.

---

## Quick Start

```bash
# 1. Clone
git lfs install
git clone https://github.com/SciSpectator/genevariate.git
cd genevariate

# 2. Install
python3 -m venv venv && source venv/bin/activate
pip install -e ".[analysis]"

# 3. Install Ollama and pull models
curl -fsSL https://ollama.com/install.sh | sh     # Linux/macOS
ollama pull gemma4:e2b
ollama pull nomic-embed-text

# 4. Launch
genevariate
```

Per-OS walkthroughs (including Docker, Windows, Homebrew) live in [INSTALL.md](INSTALL.md).

---

## Features

### Data ingestion (cross-technology)

| Source | Technology | Notes |
|---|---|---|
| GEOmetadb | Microarray catalogue | Any GPL; queried from disk on low-RAM devices |
| ARCHS4 | Bulk RNA-seq | Uniformly-processed GEO/SRA counts via `archs4py` |
| GEO Series (GPL) | Microarray matrices | Auto probe-to-gene mapping + quantile normalization |
| Raw NGS counts | RNA-seq | CSV/TSV, 10x MTX dir, or `.h5ad` → QC → DESeq2 → GSEA (`core/count_io.py`) |
| scRNA-seq pseudobulk | Single-cell → bulk | Via the canonical loader |
| Methylation / peaks | β-values / intensities | Normalised through the same base class |

All sources emit the canonical format `GSM | series_id | GENE1 | GENE2 | …`, so every
downstream tool consumes them identically.

### LLM label extraction

- Unified `gemma4:e2b` model, 32k-token context, **unlimited output tokens** (`num_predict=-1`)
- Multi-phase pipeline: raw extraction → deterministic collapse → ReAct collapse agent
- Multi-value support (`"Whole Blood; Bone Marrow"`) and coded-value disambiguation (`0/1`, `Y/N`)
- 4-tier persistent memory (cluster map, semantic RAG, episodic log, knowledge graph)

### Novel enrichment methods

- **ΔVariance GSEA** — rank genes by log-variance z-test instead of mean shift
- **Bimodality-gated GSEA** — restrict testing to genes flagged bimodal/heavy-tailed
- **Cross-platform meta-enrichment** — rank-product or Stouffer combination across GPLs
- **Embedding-clustered pseudo-cohorts** — auto-discover case/control groups from LLM labels

See [Novel Analysis Methods](#novel-analysis-methods) for the statistical detail.

### NGS raw-count differential expression

- **RNA-seq DE window** (Analysis Tools → *RNA-seq DE (raw counts)*) — load a raw count
  matrix (CSV/TSV, 10x MTX directory, or `.h5ad`), run QC (library size, genes detected,
  %mito), DESeq2 median-of-ratios normalisation, negative-binomial DE via
  [`pydeseq2`](https://github.com/owkin/PyDESeq2), then GSEA on the Wald statistic
- Register the normalised matrix as a platform so every other window can reuse it
- Headless API in `core/analysis/rnaseq.py` (`compute_qc_metrics`, `cpm_normalize`,
  `deseq2_size_factors`, `run_deseq2`, `deseq_results_to_ranked`, `counts_to_platform_df`)
- Optional extra: `pip install genevariate[rnaseq]` (pulls `pydeseq2` + `anndata`)

### Conversational assistant (confirm-before-run)

- Collapsible chat sidebar (**Ctrl+/** or Tools → *Assistant*) — type a request such as
  *“run condition enrichment on GPL570 tumor vs normal”* and the assistant proposes ONE
  tool + parameters
- **Nothing runs until you confirm**: an editable card shows the resolved parameters; you
  click *Run* to execute on the shared progress bar
- Local LLM routing via Ollama when available, with a deterministic keyword fallback when
  it is not — the app works either way, no cloud calls
- Tk-free core in `core/chatbot/` (`build_registry`, `route`); every tool calls the
  existing analysis API rather than reimplementing it

### Infrastructure

- Resource-aware worker scaling (1–210 threads) driven by live CPU/RAM/VRAM/thermal metrics
- GPU auto-detection (NVIDIA / AMD) with automatic CPU fallback
- Low-RAM mode: GEOmetadb queried directly from disk (WAL + indexes + mmap), no OOM
- Docker image with bundled Ollama and automatic model pulling

---

## Architecture

<p align="center">
  <img src="docs/architecture.svg" alt="GeneVariate pipeline" width="100%">
</p>

**Three-layer design:**

1. **Ingestion** (`core/sources/`, `core/db_loader.py`, `core/gpl_downloader.py`) — pulls
   data from GEO, ARCHS4, or local files into the canonical sample × gene matrix.
2. **Label curation** (`core/extraction.py`, `core/gse_worker.py`, `core/gse_context.py`,
   `core/memory_agent.py`, `core/ns_repair_pipeline.py`) — LLM extraction + 4-tier memory.
3. **Analysis** (`core/analysis/`, `core/statistics.py`, `core/ai_engine.py`, `gui/`) —
   variability, enrichment, distribution classification, interactive exploration.

### Module map

| Module | Purpose |
|---|---|
| `core/sources/base.py` | Canonical-format contract + shared CSV writer |
| `core/sources/archs4.py` | ARCHS4 bulk RNA-seq ingestion |
| `core/count_io.py` | Raw-count readers (CSV/TSV, 10x MTX, h5ad) → genes × samples |
| `core/analysis/rnaseq.py` | QC + CPM + DESeq2 size factors/DE + GSEA bridge |
| `core/chatbot/` | Tk-free assistant: tool registry + LLM/keyword router |
| `core/db_loader.py` | Shared GEOmetadb loader (decompress once, tier-adapted cache) |
| `core/gpl_downloader.py` | GPL annotation download, probe-to-gene, quantile normalization |
| `core/extraction.py` | LLM prompts, parsers, Phase 1.5 deterministic rules |
| `core/gse_worker.py` | Autonomous per-GSE extraction agent |
| `core/gse_context.py` | MemGPT-style rolling per-experiment context |
| `core/memory_agent.py` | 4-tier persistent memory (SQLite, WAL) |
| `core/ns_repair_pipeline.py` | Multi-phase NS repair orchestrator |
| `core/ollama_manager.py` | Watchdog, thermal guard, GPU detection, Ollama lifecycle |
| `core/analysis/variability.py` | ΔVariance ranking + GSEA prerank |
| `core/analysis/enrichment.py` | Mean-based Enrichr / GSEA wrappers |
| `core/analysis/meta_enrichment.py` | Rank-product / Stouffer cross-platform combination |
| `core/analysis/bimodality.py` | Distribution-gated gene filtering |
| `core/analysis/pseudo_cohorts.py` | Embedding-clustered auto-cohorts |
| `core/ai_engine.py` | 8-class distribution classifier, outliers, transform recommender |
| `core/statistics.py` | Wilcoxon, Welch t, Wasserstein, Cohen's d, Cliff's delta |
| `gui/app.py` | Main 3-step workflow application |

Full file tree is in [INSTALL.md](INSTALL.md#project-layout).

---

## Usage

### GUI — 3-step workflow

1. **Search** — pick a GPL platform (or ARCHS4 bulk RNA-seq), query GEO, select experiments
2. **Extract** — watch the multi-phase LLM pipeline label every sample in real time
3. **Analyse** — histograms, PCA, region selection, group comparison, enrichment

### Headless / CLI

```bash
genevariate --ns-repair                     # batch label extraction
genevariate-bench --help                    # reproducible benchmark harness
```

### Programmatic — novel enrichment

```python
from genevariate.core.analysis import (
    rank_genes_by_variability, run_variability_gsea,
    rank_genes_by_condition, run_prerank_gsea,
    classify_distributions, filter_ranked_by_distribution,
    combine_ranks, run_meta_enrichment_gsea,
    embedding_pseudo_cohorts,
)

# ΔVariance enrichment
ranked = rank_genes_by_variability(df, labels, "case", "ctrl", method="logvar_z")
gsea   = run_variability_gsea(ranked, gene_sets=["KEGG_2021_Human"])

# Bimodality-gated enrichment
tags   = classify_distributions(df)
gated  = filter_ranked_by_distribution(ranked, tags, keep=("Bimodal", "Multimodal"))
gsea   = run_prerank_gsea(gated, gene_sets=["KEGG_2021_Human"])

# Cross-platform meta-enrichment
per_plat = {"GPL570": r570, "GPL96": r96, "GPL13534": rmeth}
combined = combine_ranks(per_plat, method="stouffer")
meta     = run_meta_enrichment_gsea(combined, gene_sets=["KEGG_2021_Human"])
```

---

## Novel Analysis Methods

### ΔVariance GSEA (`logvar_z`)

Classical GSEA ranks genes by a mean-shift statistic. GeneVariate's default ΔVariance
ranker uses the formally directional **log-variance z-test**:

```
z = (log s²_case − log s²_ctrl) / sqrt( 2/(n_c−1) + 2/(n_k−1) )
```

For each gene, `log(s²) ~ N(log σ², 2/(n−1))` asymptotically (Bartlett 1937; Cochran 1941).
Unlike signed Levene / KS, this is natively **directional and two-sided**, making it a
legitimate GSEA prerank. Auxiliary methods (`levene`, `bf`, `ks`, `wasserstein`,
`logvar_ratio`) are retained behind opt-in flags for sensitivity analysis.

### Bimodality-gated enrichment

The `BioAI_Engine` distribution classifier tags each gene as Normal / Lognormal / Bimodal /
Multimodal / Heavy-tailed / Uniform / Skewed / Mixed. `filter_ranked_by_distribution`
restricts the gene universe before enrichment, answering:

> *Which pathways are driven by stochastic on/off switches rather than graded mean shifts?*

### Cross-platform meta-enrichment

Combines per-platform rankings **before** running enrichment so pathway calls survive
GPL batch effects. Two combiners:

- **rank-product** — geometric mean of per-platform ranks (Breitling 2004); non-parametric
- **Stouffer** — weighted-z combination of signed t-statistics; preserves direction

### Embedding-clustered pseudo-cohorts

Uses `nomic-embed-text` (same backbone as `MemoryAgent`) to vectorise LLM-curated condition
labels and cluster samples via KMeans — no manual case/control assignment needed. Falls
back to TF-IDF char n-grams when Ollama is unavailable.

---

## Development

### Running tests

```bash
pip install -e ".[dev,analysis]"
pytest
```

The suite covers every `core/analysis/` module and the cross-technology source loaders
(`tests/test_variability.py`, `test_enrichment.py`, `test_meta_enrichment.py`,
`test_bimodality.py`, `test_pseudo_cohorts.py`, `test_sources.py`).

### System requirements

| Resource | Minimum (low-RAM mode) | Recommended |
|---|---|---|
| CPU | 2 cores | 8+ cores |
| RAM | 4 GB | 16+ GB |
| Disk | 3 GB | 10+ GB |
| GPU | Not required | NVIDIA 6+ GB VRAM |
| OS | Linux / macOS / Windows 10+ | Ubuntu 22.04+ / macOS 13+ |
| Python | 3.10+ | 3.11+ |

At startup GeneVariate auto-detects your tier:

| Tier | RAM | GEOmetadb | Max workers | Batch size |
|---|---|---|---|---|
| Low | ≤ 6 GB | Disk (WAL + mmap) | 4 | 50 |
| Medium | 6–14 GB | Disk or RAM | 20 | 100 |
| High | ≥ 14 GB | Full in-memory | 210 | 200 |

### Contributing

Open an issue or pull request on GitHub. Tests must pass; new analysis methods should land
in `core/analysis/` with a matching test module.

---

## Citation

```bibtex
@software{genevariate2026,
  title   = {GeneVariate: Variability-aware Cross-technology Gene-expression Analysis
             with LLM-curated Labels},
  author  = {Szczepaniak, Mateusz},
  year    = {2026},
  url     = {https://github.com/SciSpectator/genevariate},
  note    = {Paper in preparation}
}
```

---

## License

MIT — see [LICENSE](LICENSE).

<p align="center">
  <sub>Built with Ollama + gemma4:e2b · Runs entirely on your hardware · No data leaves your machine</sub>
</p>
