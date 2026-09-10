# Output schema

What GeneVariate reads, and what it writes. Every table it writes is written
exactly as it appears on screen, so an exported file and the interface can never
disagree.

## 1. Input: the canonical platform frame

Every loader produces the same shape, and every analysis consumes it.

```text
GSM | series_id | GENE1 | GENE2 | ...
```

| Column | Type | Meaning |
|---|---|---|
| `GSM` | string | one sample, one row |
| `series_id` | string | the GEO study the sample came from |
| gene columns | float | expression, on the platform's own scale |

`series_id` is optional but it is what every study-clumping correction depends
on. Without it the program reports design effect 1 and states that its p-values
assume every sample is an independent experiment.

Gene columns are named by gene symbol, upper case. Probe-to-gene collapse takes
the maximum probe per gene per sample. ARCHS4 counts are log-CPM transformed on
load. Single-cell data is aggregated to per-donor per-cell-type profiles before
it is written into this frame, so a row is never a cell.

Columns prefixed `Classified_` are treated as metadata rather than genes, for
compatibility with older label files.

### Custom platform CSV

Any CSV or gzipped CSV in the shape above loads through **Add Custom Platform**.
The filename must contain a `GPLxxxxx` accession, because that is how the
program pairs an expression file with its label file.

```text
GPL99999_myplatform.csv.gz
```

## 2. Input: the label file

A CSV or gzipped CSV keyed on sample, one file per platform, with the platform
accession in the filename.

```text
GPL570_labels.csv
conditions_GPL96.csv.gz
```

The sample column may be named `GSM`, `sample`, `sample_id` or `geo_accession`.
If none of those is present, the first column is used when most of its values
start with `GSM`.

Five value columns are used and no others:

| Column | Vocabulary |
|---|---|
| `Tissue` | MeSH, or Cellosaurus for a catalogued cell line, or a locally minted identifier |
| `Condition` | MeSH or local |
| `Treatment` | MeSH or local |
| `Sex` | free value, never normalized |
| `Age` | free value, never normalized |

Tissue, Condition and Treatment carry accession columns beside the value, in the
form the extractor publishes:

```text
final_Tissue
final_Tissue_id
final_Tissue_mesh_id
final_Tissue_cell_id
final_Tissue_stage
```

Stage prefixes, deepest first:

| Prefix | Meaning |
|---|---|
| `final_` | the value that ships, after curation |
| `phase2_` | the same normalization, before curation |
| `phase1b_` | verbatim second pass, no accession |
| `phase1_` | verbatim first pass, no accession |

A value is multi-span. Phase 1 returns every span it found joined by `"; "`, and
normalization keeps the accession lists positionally parallel to it, blanks
included, so span *i* of the value belongs to span *i* of every id column.

Three accession namespaces appear and they do not mean the same thing:

| Pattern | Namespace | Claim |
|---|---|---|
| `D008099` | MeSH | the value named a concept the vocabulary knows |
| `CVCL_0027` | Cellosaurus | the value named a catalogued cell line, permitted under Tissue only |
| `ART-T-00042` | local | no vocabulary recognised it, so spellings are grouped and nothing is asserted |

`core/label_entities.py` derives a `_kind` column per entity field so a cell
line can be told from a tissue programmatically and not only by eye. Its values
are `Tissue`, `Cell line`, `Mixed` and `Unresolved`.

Older files carrying a `Classified_` prefix are accepted and the prefix is
stripped on load.

## 3. Output: the region window

**Export All** writes one folder. Figures are written twice, PNG at 300 dpi for
reading and PDF for submission, because a journal wants line art as vector and a
raster figure cannot be rescaled afterwards.

```text
region_<Tab name>.csv          one per table shown in the window
region_<Tab name>.png
region_<Tab name>.pdf
region_<Tab name>.txt          the window's written interpretation, where it has one
region_samples.csv             every sample in every region, with a Region column
region_synergy.csv             the synergy rows, marginal lifts expanded to lift_<GENE> columns
```

Tabs that only draw when a button is pressed are computed during the export
rather than being omitted, and each run is written before the next one starts.
Those files carry the run in the name:

```text
region_pooled_<gene>_<label column>_<Tab name>.csv
region_boxmodel_<label column>_<value>_<Tab name>.csv
region_labelml_clusters_<Tab name>.csv
region_labelml_markers_<label column>_<Tab name>.csv
region_labelml_gene_by_<label column>_<Tab name>.csv
region_labelml_cross_modality_<Tab name>.csv
```

Names are sanitised to `[A-Za-z0-9_-]` and truncated at 60 characters.

### Enrichment table columns

| Column | Meaning |
|---|---|
| `Region` | the brushed region the row belongs to |
| `Column` | the label column tested |
| `Value` | the label value tested |
| `Sel` | samples in the region carrying the value |
| `Sel%` | percentage of the region carrying it |
| `BG%` | percentage of the platform background carrying it |
| `Enrichment` | ratio of the two |
| `95% CI (by study)` | bootstrap over studies, not over samples |
| `n_GSE` | contributing studies |
| `n_eff` | raw selection size after correcting for study clumping |
| `p-value` | Fisher exact |
| `q-value` | Benjamini-Hochberg across the grid |
| `Sig` | significance marker |

An `n_eff` far below `Sel` is a row whose signal is one experiment repeated. The
interface additionally prints the ratio in parentheses when `n_eff` falls below
half of `Sel`.

### Synergy table columns

Per label value in a k-gene box: the box count, the multiplicative-null
expectation, the k-way interaction odds ratio, its bootstrap-by-study confidence
interval, and one `lift_<GENE>` column per gene giving that gene's marginal
lift. When any cell of the 2^k table is empty the interaction is reported as not
identified rather than as a number.

### Box model columns

`p_support`, `n_support`, `p_uniform`, the calibration error before and after
the isotonic layer, and one relaxation attribution per gene.

## 4. Output: the cross-platform window

The prefix is built from the first three platform names.

```text
xplat_<PLAT1>_vs_<PLAT2>_vs_<PLAT3>_overview.txt
xplat_..._DE_genes.csv
xplat_..._conserved_genes.csv
xplat_..._all_gene_stats.csv
xplat_..._gene_overlap.csv
xplat_..._<PLATFORM>_unique_genes.txt
```

`all_gene_stats.csv`

| Column | Meaning |
|---|---|
| `gene` | gene symbol |
| `adj_pval` | Benjamini-Hochberg across genes, of the Sidak-combined within-gene p |
| `max_abs_delta` | largest difference in mean against the reference platform |
| `ref_mean`, `ref_std` | reference platform summary |
| `n_platforms` | platforms the gene was measured on |
| `is_de` | set from `adj_pval`, so it agrees with the tab on screen |

`DE_genes.csv` adds one block of columns per non-reference platform:
`<PLAT>_mean`, `<PLAT>_delta`, `<PLAT>_pval`, `<PLAT>_effect_size`. The
per-platform detail also carries `pval_unclustered`, `design_effect` and `n_eff`,
so the uncorrected number and the correction applied to it are both visible.

`conserved_genes.csv` carries `gene`, `max_pval`, `max_abs_delta` and
`ref_mean`. These are genes where no test found a difference **and** the largest
difference in means is under half the threshold. A large p is not proof of
agreement, so it is the small delta that qualifies them as candidate
normalization anchors.

`overview.txt` states the median design effect across the tested genes, or,
where no platform carried a `series_id`, that the p-values assume every sample
is an independent experiment.

## 5. Output: any single table or plot

Every `ttk.Treeview` in the program has a Save button, a right-click menu and
Ctrl+S. Tables write to CSV, TSV or XLSX. Plots write to PNG, PDF, SVG or TIFF
at 300 dpi. This is installed by wrapping the widget constructor at startup, so
it holds for tables that do not exist yet as well as for the ones that do.

## 6. Output: the assistant

Results are written into the assistant's own window rather than into the chat.
Every answer carries a manifest recording which tool ran and with which resolved
arguments, so any assistant result can be reproduced by hand through the
interface.
