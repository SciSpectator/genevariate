# Methods and theory

Every analysis in GeneVariate takes a brushed region and a label column. This
page states what each one computes and what it refuses to claim.

## 1. Why a label count crosses technologies

A log2 array intensity, a log-CPM and a mean single-cell count are three
different quantities. A difference of means between them is a difference of
units before it is a difference of biology, and no amount of normalization
turns an unknown monotone transform into a known one.

The 2x2 table behind a label enrichment counts **samples**:

|  | in region | outside region |
|---|---|---|
| label present | a | b |
| label absent | c | d |

Expression decides only who is in the region. It never enters the test. The
region is defined inside one platform, on that platform's own scale, and what
crosses to the next platform is the answer to "which labels are in it", which
is a count. That is the same quantity on every technology.

One condition has to be met by the caller rather than by the code: the region
must mean the same thing on every platform. Technologies are related at best by
an unknown monotone transform, and the only region definitions invariant to a
monotone transform are quantile based. "The top decile of this gene on this
platform" transfers. "Above 12" does not.

## 2. Study clumping

`core/analysis/overdispersion.py`

GEO samples are not independent draws. They arrive in study-sized clumps, so a
region holding 7,754 GSMs from a few hundred GSEs carries far less information
than 7,754 independent draws. Treating the counts as binomial gives intervals
that are fictitiously tight and p-values that are confidently wrong.

### Intra-cluster correlation of a label

`estimate_rho` fits a beta-binomial to the per-study counts of a yes/no label by
maximum likelihood and reports

```text
rho = 1 / (alpha + beta + 1)
```

which is 0 when the label is spread evenly across studies and 1 when it is
confined to a few.

### Intra-cluster correlation of a measurement

`icc_oneway` is the continuous twin. It is the one-way random-effects ANOVA
moment estimator over studies:

```text
rho = (MSB - MSW) / (MSB + (n0 - 1) * MSW)
n0  = (N - sum(n_i^2) / N) / (k - 1)
```

with `MSB` the between-study mean square, `MSW` the within-study mean square,
`k` the number of studies and `n_i` the study sizes. This is what lets an
expression value measured on study-sized clumps be given the same design effect
as a count.

### Design effect and effective sample size

Kish's design effect and the effective sample size follow directly:

```text
deff  = 1 + (m_bar - 1) * rho
n_eff = n / deff
```

`m_bar` is the mean number of samples per study. `n_eff` is reported beside
every raw count in the interface, and a region whose signal sits in three
studies is visibly not the same evidence as one spread over forty.

### Widening a p-value

A p-value computed under independence is converted to its normal deviate,
divided by the square root of the design effect, and converted back:

```text
z      = Phi^-1(1 - p/2)
p_corr = 2 * (1 - Phi(z / sqrt(deff)))
```

This is applied to the Mann-Whitney and Kolmogorov-Smirnov tests in the
cross-platform window, weighting the two sides' design effects by how many
samples each contributes. Where a platform carries no `series_id`, the design
effect is 1 and the interface says the test assumes independence rather than
implying a correction was made.

### Confidence intervals

Every study-aware interval in the program resamples **studies** with
replacement, not samples. A bootstrap over samples inside a clumped region
reproduces the clumping in every replicate and returns the same fictitiously
tight interval the binomial would have given.

## 3. Region enrichment

`core/analysis/enrichment.py`, diagnostics from `overdispersion.py`

Per label value, a Fisher exact test of the 2x2 table above, Benjamini-Hochberg
across all values tested in the grid. Each row carries the selection count, the
selection and background percentages, the enrichment ratio, a bootstrap-by-study
confidence interval, the number of contributing studies, rho and `n_eff`.

A row with a large ratio and an `n_eff` far below its raw count is a row whose
signal is one experiment repeated.

## 4. Pooling across platforms

`core/analysis/pooled_enrichment.py`

Two things must not be done with per-platform enrichments, and this module
exists so that neither is.

**Do not pool the samples.** Concatenating three corpora into one 2x2 table is
not a cross-platform test, it is a test on a corpus nobody assembled. The label
composition of the corpora differs by construction, since a liver single-cell
census is entirely liver and a microarray corpus is not, so a label can be
enriched in every platform and depleted in the concatenation. This is Simpson's
paradox and it is not hypothetical here. Each platform is a stratum with its own
background, and only the effect sizes are combined.

**Do not compare significance.** "Enriched on A at q = 0.001 and not on B at
q = 0.09" is a difference of significance, not a significant difference. The
corpora differ several-fold in size, so that comparison is mostly a comparison
of power. The statistic that answers "do the platforms agree" is Cochran's Q and
I^2 over the per-platform log odds ratios, and that is what is reported.

Pooling is DerSimonian-Laird random effects. Fixed-effect pooling assumes every
platform estimates one common odds ratio, which is exactly the assumption under
test. The random-effects weights carry tau^2, so a real between-platform spread
widens the interval instead of being averaged away. Woolf's variance assumes
independent samples, so each platform's variance is multiplied by that label's
design effect before pooling.

## 5. Regions against each other

`core/analysis/region_comparison.py`

Asking the per-region question five times leaves the reader to eyeball five
blocks of a table, and the comparison they actually wanted needs statistics the
per-region test does not produce.

- `enrichment_matrix` puts regions by label values in one grid, log2 lift with a
  Haldane correction so an empty cell is a number rather than negative infinity,
  and BH-FDR across the **whole** matrix. Five regions by forty values is two
  hundred tests, not forty. `n_eff` sits beside every cell.
- `pairwise_differential` tests region A against region B directly. Both regions
  can be enriched against the platform background while being
  indistinguishable from each other, and the per-region path cannot express
  that.
- `heterogeneity` reports Cochran's Q and I^2 per label across regions. A label
  enriched everywhere and a label specific to one region look identical in a
  list of per-region hits, and they are opposite findings.
- `region_separability` is a cross-fitted P(region given label profile), with
  folds split by study. Counting says how different the regions look. This says
  how much of that survives.
- `cluster_regions` says which regions carry the same enrichment profile, that
  is, which of them are redundant.

Regions are brushed independently on different genes, so a sample can belong to
several of them. Two overlapping regions are not independent groups and a
pairwise test between them is inflated. `overlap_matrix` reports the Jaccard of
every pair and `pairwise_differential` carries the overlap on each result, so
the caller can refuse to draw the conclusion rather than draw it quietly.

## 6. Bimodality gate

`core/analysis/bimodality.py`

A filtering layer over the enrichment pipelines that redefines the gene universe
before enrichment runs, so the question becomes which pathways are driven by
stochastic on/off switches rather than by graded mean shifts.

A bimodal call rests on Hartigan's dip test for unimodality first, and then on
Gaussian-mixture BIC mode counting. It does not rest on counting peaks in a
kernel density estimate, which is a decision about bandwidth dressed up as a
decision about biology.

## 7. Batch integration

`core/analysis/integration.py`

Per-source z-scoring rescales a platform effect, it does not remove it. This
module provides ComBat (Johnson et al. 2007) empirical-Bayes correction and a
Harmony joint embedding over the genes shared across sources, treating each
source as a batch, so a region can be selected in a genuinely shared space
before labels are compared.

The shared-gene matrix is complete case. A gene not measured on every platform
is dropped and the count of dropped genes is reported. Filling those cells with
zero would fabricate a plausible low value on log2 data, which is worse than
dropping the gene, because the fabricated value is indistinguishable from a
measurement.

## 8. Cross-modality

`core/analysis/cross_modality.py`

`compare_gene_across_modalities` harmonises each source to a common scale by
z-score or by rank, then compares the shape of the distributions and reports
whether they agree. This is a descriptive harmonisation, not a claim that the
values are now the same quantity, and it is why the enrichment path and not this
one is the basis of the cross-technology comparison.

`gene_coexpression` and `coexpression_consensus` find the partners of a query
gene within one source and keep only those whose connection holds with a
consistent sign across several sources. Compositional data is handled by
Lovell's proportionality rather than by a correlation on closed counts.

## 9. Cross-platform gene comparison

`gui/app.py`, `CrossPlatformAnalysisWindow`

Per gene, Mann-Whitney and Kolmogorov-Smirnov of each platform against a
reference platform, with both p-values widened by the pooled design effect as in
section 2. Within a gene, the k-1 comparisons are combined by Sidak,
`1 - (1 - p_min)^m`, because the minimum of several p-values is not itself a
p-value. Benjamini-Hochberg then corrects across genes only, and the
differential-expression flag is set from the corrected p, so the exported table
and the tab on screen cannot disagree.

Conserved genes are those where no test found a difference and the largest
difference in means is under half the threshold. A large p is not proof of
agreement, so it is the small delta that qualifies them as candidate
normalization anchors. The p only says that nothing argues against it.

## 10. What the program will not do

- It will not concatenate platforms into one background.
- It will not impute an unmeasured expression value.
- It will not report a design-effect-corrected p-value when the platform has no
  study identifier, and it says which case you are in.
- It will not treat a cell as a sample. Single-cell data is aggregated to
  per-donor per-cell-type profiles before it reaches any analysis.

## References

Bartlett MS (1937) Proc R Soc Lond A 160:268.
Box GEP, Hill WJ (1974) Technometrics 16:385.
Breitling R et al. (2004) FEBS Lett 573:83.
Cochran WG (1941) Ann Eugen 11:47.
DerSimonian R, Laird N (1986) Control Clin Trials 7:177.
Hartigan JA, Hartigan PM (1985) Ann Stat 13:70.
Johnson WE, Li C, Rabinovic A (2007) Biostatistics 8:118.
Kish L (1965) Survey Sampling. Wiley.
Korsunsky I et al. (2019) Nat Methods 16:1289.
Lovell D et al. (2015) PLoS Comput Biol 11:e1004075.
Stouffer SA et al. (1949) The American Soldier. Princeton University Press.
