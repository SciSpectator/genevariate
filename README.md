<p align="center">
  <img src="docs/logo.png" alt="GeneVariate" width="220">
</p>

# GeneVariate

[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Data](https://img.shields.io/badge/Data-Microarray%20%7C%20RNA--seq%20%7C%20Single--cell-1565C0)](docs/architecture.md)
[![Labels](https://img.shields.io/badge/Labels-MeSH%20%7C%20Cellosaurus-8A2BE2)](docs/output-schema.md)
[![Docker](https://img.shields.io/badge/Docker-supported-2496ED?logo=docker&logoColor=white)](Dockerfile)
![preprint](https://img.shields.io/badge/preprint-in%20preparation-6E4B9E)

*Label-driven comparison of gene expression regions across measurement
technologies.*

A desktop program that takes a gene, a range of its expression, and asks which
sample labels live in that range. The samples can come from a microarray
platform, a bulk RNA-seq corpus and a single-cell atlas at once, because the
question is answered by counting labelled samples rather than by comparing
expression values, and a count of samples is the same quantity on every
technology.

This repository is the implementation accompanying the manuscript
*Label-driven comparison of gene expression regions across measurement
technologies*, in preparation.

The labels it consumes are produced by
[LLM-GEO-Label-Extractor](https://github.com/SciSpectator/LLM-GEO-Label-Extractor),
which is vendored here so the extraction can be driven from inside the program.
Five fields are used and no others: **Tissue, Condition, Treatment, Sex, Age**.

## Workflow

You load one or more platforms, attach the label files, brush a region on a
gene's distribution, and read what the region contains. Every analysis in the
program takes a brushed region and a label column as its input.

[![Architecture](docs/architecture.svg)](docs/architecture.svg)

For the step-by-step detail see [docs/architecture.md](docs/architecture.md).

1. **Ingestion.** A GEO platform is downloaded and probe-to-gene mapped, an
   ARCHS4 slice is pulled as uniformly reprocessed counts, a CELLxGENE Census
   query is aggregated into per-donor per-cell-type profiles, or your own CSV is
   read directly. Everything lands in one shape, `GSM | series_id | GENE1 |
   GENE2 ...`, so every later step consumes the sources identically.
2. **Labels.** A per-platform label file is attached, or the vendored extractor
   is run to produce one. Tissue, Condition and Treatment carry a controlled
   vocabulary accession beside the value, and the three namespaces stay
   separate, because a MeSH heading, a Cellosaurus cell line and a locally
   minted cluster identifier are not the same kind of claim.
3. **Region.** A range on a gene's distribution is brushed, on as many genes and
   platforms as you like. The brush defines a set of samples and nothing more.
4. **Comparison.** The region window answers what is in the region: which label
   values are enriched, whether two regions differ from each other, whether a
   combination of genes does more than the genes do apart, and whether any of it
   survives the fact that GEO samples arrive in study-sized clumps.

The program is local first. It reads GEO, ARCHS4 and CELLxGENE over the network
and sends nothing anywhere else. The assistant runs against a local model by
default and against a hosted endpoint only if you point it at one.

## Why counting labels crosses technologies

Expression is not comparable across technologies. A log2 array intensity, a
log-CPM and a mean single-cell count are three different quantities, so a
difference of means between them is a difference of units before it is a
difference of biology.

A label count is not affected by this. The region is defined inside one
platform, on that platform's own scale, and what crosses is the answer to
"which labels are in it", which is a count of samples. Each platform is its own
stratum with its own background, and the strata are combined by
DerSimonian-Laird random effects pooling rather than concatenated, so a large
platform cannot impose its label mix on a small one.

The honesty layer sits underneath. Samples from GEO are not independent draws,
they arrive in studies, so a region holding hundreds of samples from four
experiments carries far less information than the count suggests. The program
estimates the intra-cluster correlation across studies, converts it into Kish's
design effect, and reports an effective sample size beside every raw count.
Confidence intervals resample studies rather than samples. Details in
[docs/methods.md](docs/methods.md).

## Quick start

```bash
git clone https://github.com/SciSpectator/genevariate.git
cd genevariate
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[analysis]"
genevariate
```

Per-OS walkthroughs, including Docker, Windows and Homebrew, are in
[INSTALL.md](INSTALL.md). A guided first session is in
[docs/tutorial.md](docs/tutorial.md).

The program needs no model to run. The label extractor and the assistant do,
and both talk to an OpenAI-compatible endpoint over HTTP, so any local server
such as vLLM, SGLang or Ollama will serve them.

## The interface

The main window is three steps, left to right.

1. **Load platforms.** Preset buttons for common GPLs, **Download Platform** for
   any other, **Load RNA-seq from ARCHS4**, **Single-cell (CELLxGENE)**, or
   **Add Custom Platform** for your own matrix. **Discover Experiments** searches
   GEOmetadb, ARCHS4, CELLxGENE and Expression Atlas by keyword.
2. **Attach labels.** Point at per-platform label files, run the vendored
   extractor, or label by hand. The **Label Entities** window shows which values
   resolved to MeSH, which to Cellosaurus, and which were only clustered
   locally.
3. **Analyse.** **Gene Distribution Explorer** plots a gene on every selected
   platform and is where regions are brushed. **Label Enrichment**,
   **Distribution Analysis**, **Cross-Platform Analysis** and the **Tools** menu
   cover the rest.

### The region window

Brushing a range and pressing **Analyze Selected Range** opens the region
window, which is where the program does its work. Ten tabs, each taking the same
region and label column:

| Tab | Question it answers |
|---|---|
| Distributions | what the brushed region looks like on each platform |
| Labels | which label values are present, and in what proportion |
| Frequency | the raw counts behind the proportions |
| Enrichment | which values are over-represented against the platform background, with BH-FDR, study count, rho and effective n on every row |
| Gene Synergy | whether a multi-gene box does more than the genes do separately, as a k-way log-linear interaction odds ratio against the multiplicative null |
| Box Model | a cross-fitted, isotonic-calibrated P(label given expression), read by Monte Carlo integration so it stays defined when the box holds no samples, with per-gene relaxation attribution |
| Comparison | the regions against each other, one FDR over the whole grid, pairwise odds ratios with Jaccard overlap, and design-effect-corrected heterogeneity. Across platforms it pools instead |
| Label ML | how well a classifier separates the label from expression, cross-fitted by study |
| Statistics | the numeric summary of the region on each platform |
| Samples | the samples themselves, exportable in full |

Every tab exports. **Export All** writes every tab of the window, including the
analyses that are computed on demand.

### Every plot is live

Hover to read what is under the cursor, scroll to zoom, right-drag to pan,
double-click to reset. Clicking a sample opens its record: the GEO study, the
five labels, the free-text characteristics and its expression values. This is
implemented on matplotlib's own event system, so there is no optional
dependency that can quietly be missing and turn interactivity into a no-op.

## The assistant

**Ctrl+/** opens an assistant that drives the same analysis API the buttons do,
not a reimplementation of it. State a goal in plain English and it loads or
downloads the data, runs the right tool, and writes the result into its own
window so the chat stays readable.

```text
load GPL570
analyse the distribution of ALB on GPL570
compare ALB across microarray and rna-seq modalities
which tissues are enriched in the top decile of ALB
run meta enrichment across GPL570 and GPL96 tumor vs normal
```

It has no hardcoded answers and no keyword rules deciding what an analysis
means. It is given the computed numbers and asked to explain them, never to
produce them. The tool list and the backend options are in
[docs/architecture.md](docs/architecture.md).

## Analysis methods

The full derivations are in [docs/methods.md](docs/methods.md). In brief:

| Module | What it provides |
|---|---|
| `core/analysis/overdispersion.py` | intra-cluster correlation across studies for both counts and measurements, Kish design effect, effective sample size, bootstrap by study |
| `core/analysis/pooled_enrichment.py` | DerSimonian-Laird random-effects pooling of per-platform label enrichment, with Cochran Q, I2 and tau2 |
| `core/analysis/region_comparison.py` | regions against each other, grid-wide FDR, pairwise with Jaccard, design-effect-corrected heterogeneity, GSE-split separability |
| `core/analysis/synergy.py` | multiplicative null and k-way log-linear interaction for conjunction boxes, not identified rather than guessed when a cell is empty |
| `core/analysis/box_model.py` | calibrated P(label given expression), cross-fitted by study, relaxation attribution |
| `core/analysis/variability.py` | ranking by log-variance z-test, a directional scale-shift statistic fit to be a GSEA prerank |
| `core/analysis/bimodality.py` | Hartigan dip test then Gaussian-mixture BIC, so a bimodal call rests on a test rather than on peak counting |
| `core/analysis/meta_enrichment.py` | rank-product, Stouffer and random-effects combination across platforms |
| `core/analysis/integration.py` | ComBat and Harmony over the shared genes, for selecting a batch-corrected region before labels are compared |
| `core/analysis/cross_modality.py` | one gene across technologies on a harmonised scale, and compositionally coherent co-expression by Lovell's proportionality |
| `core/analysis/activity.py` | transcription factor and pathway activity per sample through decoupleR |

## Validation on planted ground truth

Unit tests check that a function returns a number. They do not check that the
program, driven the way a user drives it, recovers an answer decided before the
analysis ran.

```bash
python3 tools/make_synthetic_platform.py
python3 tools/validate_synthetic.py
```

The generator plants an AND gate, a single-gene marker, a label confined to four
studies and driven by no gene at all, a gene that means nothing, and a label
column that is pure noise. `GROUND_TRUTH.md` states what each tab must report,
and the validator checks 24 claims against it: that the four-study label is
discounted, that BH-FDR kills the noise column, that synergy separates a real
conjunction from a decoy, that calibration does not get worse, and that
relaxation charges the meaningless gene nothing. The two CSVs load in the GUI,
so the numbers on screen can be read against the same document.

## Docker

```bash
docker compose build
docker compose run --rm genevariate
```

Model weights are not baked into the image. The container reaches an inference
server on the host through `host.docker.internal`.

## Documentation

- [Tutorial](docs/tutorial.md)
- [Architecture and data flow](docs/architecture.md)
- [Methods and theory](docs/methods.md)
- [Output schema](docs/output-schema.md)
- [Reproducibility and limitations](docs/reproducibility.md)
- [Installation](INSTALL.md)

## Security and privacy

No API keys, no `.env` files, no author filesystem paths and no private model
configuration are in this repository. GEO, ARCHS4 and CELLxGENE are public
resources, and you remain responsible for the governance of your own inputs and
of any endpoint you point the program at.

## Citation

```bibtex
@software{genevariate,
  title  = {GeneVariate: label-driven comparison of gene expression regions
            across measurement technologies},
  author = {Szczepaniak, Mateusz},
  year   = {2026},
  url    = {https://github.com/SciSpectator/genevariate},
  note   = {Manuscript in preparation}
}
```

## License

Released under the [MIT License](LICENSE).
