"""Comparing whole platforms against one another.

This is the arithmetic behind the Cross-Platform Analysis window. It answers
four questions about a set of loaded platforms in one pass over the data:

* **What do they measure in common?** Set algebra over each platform's gene
  map - the genes on all of them, the pairwise overlaps, and the genes only one
  platform carries. Cheap: it touches the gene maps, never the expression.
* **Do they agree on a gene?** Per gene, every platform is tested against a
  reference with Mann-Whitney and a two-sample KS, and the p is widened by the
  design effect first, because GEO delivers samples in study-sized clumps and a
  platform whose samples come from one experiment is one reading repeated.
  A gene on *k* platforms is tested *k-1* times, so the smallest of those p's is
  Sidak-corrected back into a single p before BH runs across genes.
* **How far apart are the platforms overall?** Spearman between the platforms'
  per-gene mean vectors, plus the shift in pooled median/mean/SD from the
  reference.
* **Is any of that removable?** Median centering, per-gene cross-platform
  quantile normalisation, or ComBat with the extracted labels protected as
  biological covariates.

One caution runs through all of it. A batch effect is unwanted technical
variation *within* one assay - same measurand, different run - and it is
removable precisely because the quantity measured is the same. Array intensity,
sequencing counts and aggregated single-cell counts are different measurands, so
across technologies the batch score is a unit difference and is reported as one;
correcting it produces an axis to select a region on, not comparable
measurements.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu, norm, rankdata, spearmanr

from .overdispersion import design_effect, icc_oneway

__all__ = ["gene_inventory", "median_center_normalize",
           "quantile_normalize_cross", "combat_correct", "platform_similarity",
           "analyze_platforms", "summarize_cross_platform"]


#: Upper bound on the shared reference resolution, so a very large platform
#: cannot make the quantile-normalisation reference itself expensive to hold.
CROSS_QNORM_MAX_GRID = 200_000

#: Columns that are provenance or free text, never a biological label to
#: protect during ComBat.
_SKIP_COLS = {
    'GSM', 'gsm', 'series_id', 'gpl', 'platform', '_platform',
    'title', 'gsm_title', 'source_name', 'source_name_ch1',
    'characteristics', 'characteristics_ch1', 'description',
    'treatment_protocol', 'organism_ch1', 'geo_accession',
    'Token_Match', 'Matched_Tokens',
}


# -----------------------------------------------------------------
# 1. Gene inventory - what the platforms measure in common
# -----------------------------------------------------------------
def gene_inventory(gene_sets: Mapping[str, set]) -> Dict[str, Any]:
    """Overlap algebra over each platform's measured genes.

    Returns ``all_genes``, ``common_all`` (on every platform), the pairwise
    intersections, and per platform the genes no other platform carries.
    """
    platforms = list(gene_sets)
    all_genes: set = set()
    for gs in gene_sets.values():
        all_genes |= gs
    common_all = (set.intersection(*gene_sets.values()) if gene_sets else set())

    pairwise = {}
    for i, a in enumerate(platforms):
        for b in platforms[i + 1:]:
            pairwise[(a, b)] = gene_sets[a] & gene_sets[b]

    unique = {}
    for plat in platforms:
        others = [gene_sets[p] for p in platforms if p != plat]
        unique[plat] = gene_sets[plat] - (set.union(*others) if others else set())

    return {"all_genes": all_genes, "common_all": common_all,
            "pairwise_overlap": pairwise, "unique_genes": unique}


# -----------------------------------------------------------------
# 2. Putting platforms on one axis
# -----------------------------------------------------------------
def median_center_normalize(expr_dict: Mapping[str, pd.Series]
                            ) -> Dict[str, pd.Series]:
    """Per-platform median centering, one gene at a time.

    Each platform is shifted so its median sits on the global median. The
    weakest of the three corrections and the only one that leaves the shape of
    every distribution untouched.
    """
    all_vals = np.concatenate([s.values for s in expr_dict.values()])
    global_median = np.nanmedian(all_vals)
    return {plat: series - np.nanmedian(series.values) + global_median
            for plat, series in expr_dict.items()}


def quantile_normalize_cross(expr_dict: Mapping[str, pd.Series]
                             ) -> Dict[str, pd.Series]:
    """Cross-platform quantile normalization for a single gene.

    Forces all platforms to share one rank-distribution template:

    1. Stretch each platform's sorted values onto the shared relative position
       grid [0, 1] and average them into one reference.
    2. Map every value back through its average rank position, by interpolation
       against that reference.

    Platforms carry different numbers of samples, so positions are relative
    rather than absolute: without that, the platform with the fewest samples
    dictated the resolution and every other platform's values were quantized
    onto its rank slots. Ranks are tie-averaged and the lookup interpolates, so
    equal inputs map to one output and distinct inputs are not collapsed
    together by an integer cast. NaN is excluded from the reference and from
    the ranks, and returned as NaN.

    This aligns the entire distribution shape, not just the median - which also
    means it removes the differences in shape this program exists to measure.
    """
    if not expr_dict or len(expr_dict) < 2:
        return dict(expr_dict)

    # Only platforms with enough measurements can contribute a distribution
    # shape. A platform that is empty for this gene is skipped rather than
    # dragging every other platform down to median centering.
    clean = {p: pd.to_numeric(s, errors='coerce').dropna()
             for p, s in expr_dict.items()}
    usable = {p: s for p, s in clean.items() if len(s) >= 3}
    if len(usable) < 2:
        return median_center_normalize(expr_dict)

    # Resolve the reference at the largest platform's resolution, so that
    # platform's values land exactly on grid points and are not rounded by the
    # interpolation.
    n_grid = min(max(len(s) for s in usable.values()), CROSS_QNORM_MAX_GRID)
    grid = np.linspace(0.0, 1.0, n_grid)

    reference = np.zeros(n_grid, dtype=np.float64)
    for s in usable.values():
        vals = np.sort(s.to_numpy(dtype=np.float64))
        reference += np.interp(grid, np.linspace(0.0, 1.0, len(vals)), vals)
    reference /= len(usable)

    result = {}
    for plat, series in expr_dict.items():
        # copy=True is required: pd.to_numeric returns the caller's own series
        # when it is already numeric, and to_numpy() on that is a view, so the
        # writes below would overwrite the input the caller still holds.
        vals = pd.to_numeric(series, errors='coerce').to_numpy(
            dtype=np.float64, copy=True)
        finite = np.isfinite(vals)
        n_ok = int(finite.sum())

        if n_ok == 0:
            result[plat] = series.copy()
            continue
        if n_ok == 1:
            vals[finite] = np.interp(0.5, grid, reference)
        else:
            pos = (rankdata(vals[finite], method='average') - 1.0) / (n_ok - 1.0)
            vals[finite] = np.interp(pos, grid, reference)
        result[plat] = pd.Series(vals, index=series.index, name=series.name)
    return result


def combat_correct(expr_df: pd.DataFrame, batch_labels: Sequence[str],
                   bio_covariates: Optional[pd.DataFrame] = None):
    """ComBat batch correction with optional biological covariate protection.

    ``expr_df`` is genes x samples; ``batch_labels`` has one batch id per
    sample column. ``bio_covariates`` (samples x covariates) names the
    biological variables to PROTECT - when given, ComBat removes only technical
    batch variance and preserves the group differences those labels describe.

    Returns ``(corrected_df, method_string)``. Falls back to per-batch median
    centering if pycombat is unavailable or produces NaN.

    ComBat needs a multi-gene matrix (>= ~10 genes) for reliable empirical
    Bayes estimation; with fewer it degenerates and may produce NaN.
    """
    n_batches = len(set(batch_labels))
    if n_batches < 2:
        return expr_df.copy(), 'single_batch_noop'

    def _validate(corrected, original):
        if corrected is None:
            return False
        nan_frac = np.isnan(corrected.values).sum() / corrected.size
        orig_nan_frac = np.isnan(original.values).sum() / original.size
        return nan_frac < orig_nan_frac + 0.1 and nan_frac < 0.5

    # pycombat expects mod as a list (single covariate) or list of lists, each
    # with one numeric/categorical entry per sample.
    mod: List[List[int]] = []
    if bio_covariates is not None and not bio_covariates.empty:
        try:
            if len(bio_covariates) == expr_df.shape[1]:
                for col_name in bio_covariates.columns:
                    values = bio_covariates[col_name].astype(str).values
                    val_to_code = {v: i for i, v in enumerate(sorted(set(values)))}
                    mod.append([val_to_code[v] for v in values])
            else:
                mod = []
        except Exception:
            mod = []

    method_suffix = '_with_covariates' if mod else ''

    for _try_import in range(2):
        try:
            if _try_import == 0:
                from pycombat import pycombat as _pycombat_fn
            else:
                from combat.pycombat import pycombat as _pycombat_fn
            if mod:
                corrected = _pycombat_fn(expr_df, batch_labels, mod=mod)
            else:
                corrected = _pycombat_fn(expr_df, batch_labels)
            if _validate(corrected, expr_df):
                return corrected, f'combat{method_suffix}'
            break
        except ImportError:
            continue
        except Exception:
            # ComBat crashed - try without covariates as an intermediate step
            if mod:
                try:
                    corrected = _pycombat_fn(expr_df, batch_labels)
                    if _validate(corrected, expr_df):
                        return corrected, 'combat_no_covariates'
                except Exception:
                    pass
            break

    corrected = expr_df.copy()
    global_median = np.nanmedian(expr_df.values)
    for batch in set(batch_labels):
        mask = [b == batch for b in batch_labels]
        batch_vals = corrected.loc[:, mask].values
        batch_median = np.nanmedian(batch_vals)
        if not np.isnan(batch_median) and not np.isnan(global_median):
            corrected.loc[:, mask] = batch_vals - batch_median + global_median
    return corrected, 'median_centering_fallback'


# -----------------------------------------------------------------
# 3. Platform-level similarity
# -----------------------------------------------------------------
def platform_similarity(platforms: Sequence[str],
                        datasets: Mapping[str, pd.DataFrame],
                        gene_maps: Mapping[str, Mapping[str, str]],
                        common_genes: Sequence[str]) -> Dict[tuple, Dict[str, Any]]:
    """Spearman rho between each pair of platforms' per-gene mean vectors.

    Rank-based on purpose: two platforms can put a gene at the same rank while
    disagreeing on its value by an order of magnitude, and that ordering is the
    part a cross-platform claim can rest on.
    """
    sorted_common = sorted(common_genes)
    vectors = {}
    for plat in platforms:
        gmap = gene_maps.get(plat, {})
        df = datasets[plat]
        vec = []
        for gene in sorted_common:
            col = gmap.get(gene)
            if col and col in df.columns:
                vec.append(pd.to_numeric(df[col], errors='coerce').mean())
            else:
                vec.append(np.nan)
        vectors[plat] = np.array(vec)

    out = {}
    for i, a in enumerate(platforms):
        for b in platforms[i + 1:]:
            va, vb = vectors[a], vectors[b]
            mask = ~(np.isnan(va) | np.isnan(vb))
            if mask.sum() > 10:
                corr, cpval = spearmanr(va[mask], vb[mask])
            else:
                corr, cpval = np.nan, np.nan
            out[(a, b)] = {'spearman': corr, 'pval': cpval,
                           'n_genes_compared': int(mask.sum())}
    return out


# -----------------------------------------------------------------
# 4. The whole comparison
# -----------------------------------------------------------------
def _inflate_p(p: float, deff: float) -> float:
    """Widen a two-sided p by a design effect, via its normal deviate."""
    p = float(p)
    if not np.isfinite(p) or p >= 1.0 or deff <= 1.0:
        return min(max(p, 0.0), 1.0)
    z = norm.isf(max(p, 1e-300) / 2.0)
    return float(min(1.0, 2.0 * norm.sf(z / np.sqrt(deff))))


def _bh(pvals: np.ndarray) -> np.ndarray:
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.ones(n)
    for rank_i, idx in enumerate(order):
        adj[idx] = pvals[idx] * n / (rank_i + 1)
    for i in range(n - 2, -1, -1):
        adj[order[i]] = min(adj[order[i]],
                            adj[order[i + 1]] if i + 1 < n else 1.0)
    return np.clip(adj, 0, 1)


def _bio_covariates(platforms, datasets, labels, col_labels):
    """Biological label matrix for ComBat, one row per sample column.

    Every label column the extractor produced is used, standard and custom
    alike; a column with no variance, or one that is >80% Unknown, protects
    nothing and is dropped.
    """
    rows: List[dict] = []
    detected: set = set()
    has_labels = False

    for plat in platforms:
        df = datasets[plat]
        lbl_df = (labels or {}).get(plat)
        if lbl_df is not None and not lbl_df.empty:
            if not detected:
                detected = {c for c in lbl_df.columns
                            if c not in _SKIP_COLS and not c.startswith('_')
                            and lbl_df[c].dtype == 'object'}
            gsm_col = next((c for c in df.columns if c.upper() == 'GSM'), None)
            lbl_gsm = next((c for c in lbl_df.columns if c.upper() == 'GSM'), None)
            if gsm_col and lbl_gsm:
                indexed = lbl_df.drop_duplicates(lbl_gsm).set_index(lbl_gsm)
                for _, row in df.iterrows():
                    gsm = str(row.get(gsm_col, '')).strip()
                    cov = {}
                    for bc in detected:
                        val = 'Unknown'
                        if gsm in indexed.index and bc in indexed.columns:
                            v = indexed.at[gsm, bc]
                            if isinstance(v, pd.Series):
                                v = v.iloc[0]
                            v = str(v).strip()
                            if v and v.lower() not in ('nan', 'none',
                                                       'not specified', ''):
                                val = v
                        cov[bc] = val
                    rows.append(cov)
                    has_labels = True
                continue
        for _ in range(len(df)):
            rows.append({bc: 'Unknown' for bc in (detected or {'_none'})})

    if not (has_labels and detected):
        return None
    cov = pd.DataFrame(rows, index=col_labels).drop(columns=['_none'],
                                                    errors='ignore')
    for bc in list(cov.columns):
        if cov[bc].nunique() <= 1:
            cov = cov.drop(columns=[bc])
    for bc in list(cov.columns):
        if (cov[bc] == 'Unknown').sum() / len(cov) > 0.8:
            cov = cov.drop(columns=[bc])
    return None if cov.empty else cov


def analyze_platforms(platforms: Sequence[str],
                      datasets: Mapping[str, pd.DataFrame],
                      gene_maps: Mapping[str, Mapping[str, str]],
                      *,
                      species: Optional[Mapping[str, str]] = None,
                      technology: Optional[Mapping[str, str]] = None,
                      labels: Optional[Mapping[str, pd.DataFrame]] = None,
                      reference: str = "(auto)",
                      batch_method: str = "none",
                      pval_threshold: float = 0.05,
                      delta_threshold: float = 0.5,
                      progress_cb: Optional[Callable[[float, str], None]] = None,
                      log_cb: Optional[Callable[[str], None]] = None
                      ) -> Dict[str, Any]:
    """Compare a set of platforms gene by gene. Returns the full result dict.

    ``datasets`` maps platform -> samples x columns frame; ``gene_maps`` maps
    platform -> {gene symbol: column name}. ``species``/``technology`` are used
    only to decide whether the platforms measure a comparable quantity at all.
    """
    platforms = list(platforms)
    progress = progress_cb or (lambda v, t: None)
    log = log_cb or (lambda m: None)
    species = dict(species or {})
    technology = dict(technology or {})

    results: Dict[str, Any] = {
        'platforms': platforms,
        'batch_method': batch_method,
        'pval_threshold': pval_threshold,
        'delta_threshold': delta_threshold,
    }

    ref = reference
    if ref == '(auto)' or ref not in platforms:
        ref = max(platforms, key=lambda p: len(gene_maps.get(p, {})))
    results['reference'] = ref

    # ---- 1. gene inventory ----------------------------------------
    progress(5, "Building gene inventories...")
    gene_sets = {plat: set(gene_maps.get(plat, {}).keys()) for plat in platforms}
    results['gene_sets'] = gene_sets
    results.update(gene_inventory(gene_sets))
    common_all = results['common_all']

    progress(10, f"Analyzing {len(common_all):,} common genes across "
                 f"{len(platforms)} platforms...")
    log(f"{len(common_all):,} common genes, {len(results['all_genes']):,} "
        f"total across {len(platforms)} platforms")

    # ---- 2. can these platforms be compared at all? ---------------
    results['species_map'] = species
    # An organism we could not resolve is not evidence of a second organism, so
    # only species we actually know count towards the claim.
    known_species = {s for s in species.values() if s != 'unknown'}
    results['is_cross_species'] = len(known_species) > 1

    results['tech_map'] = technology
    known_tech = {t for t in technology.values()
                  if t not in ('unknown', 'custom')}
    results['is_cross_technology'] = len(known_tech) > 1

    # ---- 3. expression for the common genes -----------------------
    progress(15, "Extracting expression data for common genes...")
    gene_expr: Dict[str, Dict[str, np.ndarray]] = {}
    # The study each sample came from, in dataset row order. GEO delivers
    # samples in study-sized clumps, so the tests below have to know which
    # values are repeat readings of one experiment rather than independent
    # evidence about the platform.
    gene_groups: Dict[str, Dict[str, np.ndarray]] = {}
    plat_series = {}
    for plat in platforms:
        df = datasets[plat]
        scol = next((c for c in df.columns
                     if c.lower() in ('series_id', 'gse', 'series')), None)
        plat_series[plat] = df[scol].astype(str).values if scol else None

    platform_medians, platform_means, platform_stds = {}, {}, {}
    for plat in platforms:
        df = datasets[plat]
        gmap = gene_maps.get(plat, {})
        series = plat_series[plat]
        vals_all: List[float] = []
        for gene in common_all:
            col = gmap.get(gene)
            if not col or col not in df.columns:
                continue
            num = pd.to_numeric(df[col], errors='coerce')
            mask = num.notna().values
            expr = num.values[mask]
            if len(expr) == 0:
                continue
            gene_expr.setdefault(gene, {})
            gene_groups.setdefault(gene, {})
            gene_expr[gene][plat] = expr
            if series is not None:
                gene_groups[gene][plat] = series[mask]
            vals_all.extend(expr.tolist())
        platform_medians[plat] = np.nanmedian(vals_all) if vals_all else 0
        platform_means[plat] = np.nanmean(vals_all) if vals_all else 0
        platform_stds[plat] = np.nanstd(vals_all) if vals_all else 0

    results['platform_medians'] = platform_medians
    results['platform_means'] = platform_means
    results['platform_stds'] = platform_stds

    # ---- 4. batch effect ------------------------------------------
    progress(25, "Detecting batch effects...")
    results['batch_metrics'] = {
        plat: {
            'median': platform_medians[plat],
            'mean': platform_means[plat],
            'std': platform_stds[plat],
            'median_shift_from_ref': platform_medians[plat] - platform_medians.get(ref, 0),
            'mean_shift_from_ref': platform_means[plat] - platform_means.get(ref, 0),
            'variance_ratio_to_ref': (platform_stds[plat]
                                      / max(platform_stds.get(ref, 1), 1e-10)) ** 2,
        } for plat in platforms}

    median_vals = [platform_medians[p] for p in platforms]
    results['batch_effect_score'] = (
        np.std(median_vals) / max(np.mean(list(platform_stds.values())), 1e-10))
    # Across technologies this score compares the pooled median of array
    # intensities against that of sequencing counts - it compares units. It is
    # large before any data is seen and says nothing about batches, so it is
    # reported as the scale difference it measures and the claim is withheld.
    results['batch_score_is_unit_difference'] = results['is_cross_technology']
    results['batch_effect_detected'] = (not results['is_cross_technology']
                                        and results['batch_effect_score'] > 0.2)

    # ---- 5. optional correction -----------------------------------
    _batch_key = batch_method.replace(' (preserve biology)', '').strip()
    corrected_gene_expr = gene_expr

    # A correction across technologies still runs, because putting the
    # platforms on one axis is what lets a region be selected on all of them at
    # once. What it does not do is make the values mean the same thing, so the
    # caveat travels with the result instead of the result being suppressed.
    if results['is_cross_technology'] and _batch_key != 'none':
        results['batch_correction_caveat'] = (
            f"{_batch_key} was applied across " + ", ".join(sorted(known_tech))
            + ", which measure different quantities. A batch correction models "
            "the platform as an offset on a shared measurand, and there is none "
            "here, so the corrected values are an axis to select a region on "
            "rather than comparable measurements. Per-gene quantile "
            "normalization goes further and makes each gene's distribution "
            "identical on every platform, which removes the very differences in "
            "distribution shape this program measures. Compare the labels the "
            "region carries.")
        log(results['batch_correction_caveat'])

    if _batch_key != 'none' and gene_expr:
        progress(30, f"Applying batch correction ({batch_method})...")
        if _batch_key in ('median_centering', 'quantile_normalization'):
            fn = (median_center_normalize if _batch_key == 'median_centering'
                  else quantile_normalize_cross)
            corrected_gene_expr = {
                gene: {p: s.values for p, s in
                       fn({p: pd.Series(v) for p, v in plat_data.items()}).items()}
                for gene, plat_data in gene_expr.items()}
            results['batch_correction_used'] = _batch_key

        elif _batch_key == 'combat':
            progress(30, "Running ComBat batch correction...")
            col_labels, batch_labels = [], []
            for plat in platforms:
                n = len(datasets[plat])
                col_labels.extend([f"{plat}_{i}" for i in range(n)])
                batch_labels.extend([plat] * n)

            bio_cov = _bio_covariates(platforms, datasets, labels, col_labels)
            if bio_cov is not None:
                n_groups = sum(bio_cov[c].nunique() for c in bio_cov.columns)
                progress(30, f"ComBat: protecting {len(bio_cov.columns)} label "
                             f"fields ({n_groups} groups) from correction")
                log(f"ComBat with biological covariates: "
                    f"{list(bio_cov.columns)} ({n_groups} groups protected)")
            else:
                progress(30, "ComBat: no usable extracted labels - running "
                             "without covariate protection")

            # A missing measurement is missing. Filling it with 0 turns it into
            # a real value - and on log2 data 0 is a plausible-looking low
            # expression, so it moves the gene's mean and every delta computed
            # from it. ComBat needs a dense matrix, so genes not measured
            # everywhere are dropped rather than invented.
            gene_rows, dropped_incomplete = {}, 0
            candidates = sorted(gene_expr.keys())
            for gene in candidates:
                row, complete = [], True
                for plat in platforms:
                    col = gene_maps.get(plat, {}).get(gene)
                    df = datasets[plat]
                    if not col or col not in df.columns:
                        complete = False
                        break
                    vals = pd.to_numeric(df[col], errors='coerce').values
                    if not np.isfinite(vals).all():
                        complete = False
                        break
                    row.extend(vals.tolist())
                if complete:
                    gene_rows[gene] = row
                else:
                    dropped_incomplete += 1
            results['batch_correction_genes_dropped'] = dropped_incomplete
            if dropped_incomplete:
                log(f"ComBat: {dropped_incomplete:,} of {len(candidates):,} "
                    f"common genes are not measured on every sample of every "
                    f"platform and were left out of the correction rather than "
                    f"zero-filled.")

            expr_matrix = pd.DataFrame(gene_rows, index=col_labels).T
            progress(35, f"ComBat: {expr_matrix.shape[0]} genes x "
                         f"{expr_matrix.shape[1]} samples")
            corrected_matrix, method_used = combat_correct(
                expr_matrix, batch_labels, bio_covariates=bio_cov)

            nan_frac = (np.isnan(corrected_matrix.values).sum()
                        / max(1, corrected_matrix.size))
            if nan_frac > 0.5:
                progress(35, f"ComBat produced {nan_frac:.0%} NaN - falling "
                             f"back to median centering")
                method_used = 'median_centering_fallback'
                corrected_gene_expr = {
                    gene: {p: s.values for p, s in median_center_normalize(
                        {p: pd.Series(v) for p, v in plat_data.items()}).items()}
                    for gene, plat_data in gene_expr.items()}
            else:
                # Genes ComBat could not correct are NOT filled back in from
                # the uncorrected data: every number downstream is compared
                # against every other under one FDR, and a table holding
                # corrected and uncorrected values at once cannot be read.
                corrected_gene_expr = {}
                for gene in sorted(gene_rows):
                    if gene not in corrected_matrix.index:
                        continue
                    corrected_gene_expr[gene] = {}
                    offset = 0
                    for plat in platforms:
                        n = len(datasets[plat])
                        vals = corrected_matrix.loc[gene].iloc[offset:offset + n]
                        corrected_gene_expr[gene][plat] = vals.values.astype(float)
                        offset += n
            results['batch_correction_used'] = method_used
    else:
        results['batch_correction_used'] = 'none'

    results['corrected_gene_expr'] = corrected_gene_expr

    # ---- 6. per-gene comparison -----------------------------------
    progress(45, "Running per-gene statistical comparisons...")
    gene_stats: List[dict] = []
    all_deffs: List[float] = []
    total_genes = len(corrected_gene_expr)

    for gi, (gene, plat_data) in enumerate(corrected_gene_expr.items()):
        if gi % 500 == 0 and gi > 0:
            progress(45 + 45 * gi / max(total_genes, 1),
                     f"Analyzing gene {gi:,}/{total_genes:,}...")
        if len(plat_data) < 2:
            continue
        active = [p for p in platforms if p in plat_data and len(plat_data[p]) > 2]
        if len(active) < 2:
            continue

        ref_vals = plat_data.get(ref)
        if ref_vals is None or len(ref_vals) < 3:
            ref_vals = plat_data[active[0]]
            local_ref = active[0]
        else:
            local_ref = ref
        ref_mean = float(np.nanmean(ref_vals))
        ref_std = float(np.nanstd(ref_vals))

        # How much of this gene's spread on each platform is between studies
        # rather than between samples.
        g_of = gene_groups.get(gene, {})
        deff = {}
        for p in active:
            grp = g_of.get(p)
            v = plat_data[p]
            if grp is None or len(grp) != len(v):
                deff[p] = 1.0
                continue
            rho_p, m_bar = icc_oneway(v, grp)
            deff[p] = design_effect(m_bar, rho_p)

        max_delta = 0.0
        details: Dict[str, dict] = {}
        all_pvals: List[float] = []

        for other in active:
            if other == local_ref:
                details[other] = {'mean': ref_mean, 'std': ref_std,
                                  'n': len(ref_vals), 'pval': 1.0,
                                  'delta_mean': 0.0, 'ks_stat': 0.0,
                                  'effect_size': 0.0}
                continue

            other_vals = plat_data[other]
            other_mean = float(np.nanmean(other_vals))
            other_std = float(np.nanstd(other_vals))
            delta_mean = other_mean - ref_mean

            try:
                _, pval_w = mannwhitneyu(ref_vals, other_vals,
                                         alternative='two-sided')
            except Exception:
                pval_w = 1.0

            # Mann-Whitney counts every sample as one independent reading. It
            # is not: the samples came in studies. Spending the two design
            # effects, weighted by how many samples each side contributes,
            # widens the test by exactly the factor by which the sample count
            # overstates the evidence.
            n_r, n_o = len(ref_vals), len(other_vals)
            pair_deff = ((deff[local_ref] * n_r + deff[other] * n_o)
                         / max(n_r + n_o, 1))
            pval_raw = pval_w
            all_deffs.append(pair_deff)
            if pair_deff > 1.0:
                pval_w = _inflate_p(pval_w, pair_deff)

            try:
                ks_stat, ks_pval = ks_2samp(ref_vals, other_vals)
                if pair_deff > 1.0:
                    ks_pval = _inflate_p(ks_pval, pair_deff)
            except Exception:
                ks_stat, ks_pval = 0.0, 1.0

            pooled_std = np.sqrt((ref_std ** 2 + other_std ** 2) / 2)
            details[other] = {
                'mean': other_mean, 'std': other_std, 'n': n_o,
                'pval': pval_w, 'pval_unclustered': pval_raw,
                'design_effect': pair_deff,
                'n_eff': (n_r + n_o) / max(pair_deff, 1e-9),
                'delta_mean': delta_mean,
                'ks_stat': ks_stat, 'ks_pval': ks_pval,
                'effect_size': abs(delta_mean) / max(pooled_std, 1e-10),
                'var_ratio': (other_std / max(ref_std, 1e-10)) ** 2,
            }
            all_pvals.append(pval_w)
            max_delta = max(max_delta, abs(delta_mean))

        # A gene on k platforms is tested k-1 times against the reference, and
        # the smallest of those p-values is not itself a p-value: under the null
        # it is the minimum of k-1 uniforms. Sidak turns it back into one, so
        # the BH step below corrects across genes only, which is what it
        # assumes it is doing.
        m = len(all_pvals)
        min_p = min(all_pvals) if all_pvals else 1.0
        gene_stats.append({
            'gene': gene, 'ref_platform': local_ref,
            'ref_mean': ref_mean, 'ref_std': ref_std,
            'n_platforms': len(active), 'n_comparisons': m,
            'max_abs_delta': max_delta, 'min_pval': min_p,
            'gene_pval': 1.0 - (1.0 - min_p) ** m if m else 1.0,
            'max_pval': max(all_pvals) if all_pvals else 1.0,
            'platform_details': details,
        })

    de_genes: List[dict] = []
    conserved_genes: List[dict] = []
    if gene_stats:
        adj = _bh(np.array([g['gene_pval'] for g in gene_stats]))
        for i, gs in enumerate(gene_stats):
            gs['adj_pval'] = float(adj[i])
            # One definition of "differentially expressed", the corrected one.
            gs['is_de'] = bool(gs['adj_pval'] < pval_threshold
                               and gs['max_abs_delta'] > delta_threshold)
        de_genes = [g for g in gene_stats if g['is_de']]
        conserved_genes = [g for g in gene_stats
                           if g['max_pval'] > 0.5
                           and g['max_abs_delta'] < delta_threshold * 0.5]

    results['gene_stats'] = gene_stats
    results['de_genes'] = sorted(de_genes, key=lambda x: x.get('adj_pval', 1))
    results['conserved_genes'] = sorted(conserved_genes,
                                        key=lambda x: -x['max_pval'])
    results['n_de'] = len(de_genes)
    results['n_conserved'] = len(conserved_genes)
    results['n_tested'] = len(gene_stats)
    results['median_design_effect'] = (float(np.median(all_deffs))
                                       if all_deffs else 1.0)
    results['has_study_ids'] = any(v is not None for v in plat_series.values())

    # ---- 7. platform similarity -----------------------------------
    progress(95, "Computing platform similarity...")
    results['plat_correlations'] = platform_similarity(
        platforms, datasets, gene_maps, common_all)

    progress(100, "Analysis complete!")
    log(f"Done: {results['n_tested']:,} genes tested, {results['n_de']:,} DE, "
        f"{results['n_conserved']:,} conserved")
    return results


# -----------------------------------------------------------------
# 5. Narration
# -----------------------------------------------------------------
def summarize_cross_platform(results: Mapping[str, Any], *, top: int = 10) -> str:
    """Markdown account of a cross-platform run, caveats first."""
    plats = results['platforms']
    ref = results['reference']
    out = [f"# {len(plats)} platforms compared against {ref}\n"]

    if results.get('is_cross_technology'):
        techs = ", ".join(sorted({t for t in results['tech_map'].values()
                                  if t not in ('unknown', 'custom')}))
        out.append(f"**These platforms do not measure the same quantity** "
                   f"({techs}). Differences below include the difference in "
                   f"units, and no correction can remove that. Read the gene "
                   f"*rankings* and the labels, not the values.\n")
    if results.get('is_cross_species'):
        sp = ", ".join(sorted({s for s in results['species_map'].values()
                               if s != 'unknown'}))
        out.append(f"**Cross-species comparison** ({sp}). Genes are matched by "
                   f"symbol, which is not the same as one-to-one orthology.\n")
    if results.get('batch_correction_caveat'):
        out.append(f"> {results['batch_correction_caveat']}\n")

    out.append("## Coverage\n")
    out.append(f"- {len(results['all_genes']):,} genes across all platforms; "
               f"{len(results['common_all']):,} measured on every one.")
    for plat in plats:
        n_uni = len(results['unique_genes'].get(plat, set()))
        out.append(f"- {plat}: {len(results['gene_sets'].get(plat, set())):,} "
                   f"genes, {n_uni:,} of them on no other platform.")

    out.append("\n## Agreement\n")
    for (a, b), info in results['plat_correlations'].items():
        rho = info['spearman']
        rho_s = f"{rho:.3f}" if rho == rho else "n/a"
        out.append(f"- {a} vs {b}: Spearman rho {rho_s} over "
                   f"{info['n_genes_compared']:,} genes.")

    deff = results.get('median_design_effect', 1.0)
    out.append("\n## Per-gene comparison\n")
    out.append(f"- {results['n_tested']:,} genes tested; "
               f"{results['n_de']:,} differ (BH q < "
               f"{results['pval_threshold']} and |delta| > "
               f"{results['delta_threshold']}), "
               f"{results['n_conserved']:,} are conserved.")
    if results.get('has_study_ids') and deff > 1.05:
        # The design effect is how many samples it takes to buy one sample's
        # worth of evidence, so the honest sample size is n/deff.
        out.append(f"- Median design effect {deff:.2f}: the samples arrive in "
                   f"study-sized clumps, so {deff:.2f} of them carry one "
                   f"sample's worth of independent evidence and every p was "
                   f"widened accordingly. Counting them as independent would "
                   f"treat the effective sample size as {deff:.0%} of its real "
                   f"value.")
    elif not results.get('has_study_ids'):
        out.append("- No study ids on these platforms, so every sample was "
                   "counted as independent evidence. Where samples share an "
                   "experiment that overstates the significance.")

    if results['de_genes']:
        out.append("\n### Most divergent genes\n")
        out.append("| gene | q | max abs delta | platforms |")
        out.append("| --- | --- | --- | --- |")
        for g in results['de_genes'][:top]:
            out.append(f"| {g['gene']} | {g['adj_pval']:.2e} | "
                       f"{g['max_abs_delta']:.3f} | {g['n_platforms']} |")

    score = results.get('batch_effect_score', 0.0)
    out.append("\n## Batch\n")
    if results.get('batch_score_is_unit_difference'):
        out.append(f"- Platform-median spread is {score:.2f} standard "
                   f"deviations, but that is the unit difference between "
                   f"technologies, not a batch effect. No correction can fix a "
                   f"difference of measurand.")
    elif results.get('batch_effect_detected'):
        out.append(f"- Batch effect score {score:.2f} (> 0.2): the platform "
                   f"medians are spread wide relative to within-platform "
                   f"variation. Correction is worth applying before comparing "
                   f"values.")
    else:
        out.append(f"- Batch effect score {score:.2f}: the platforms are "
                   f"well aligned, so direct comparison is reasonable.")
    out.append(f"- Correction applied: {results['batch_correction_used']}.")
    if results.get('batch_correction_genes_dropped'):
        out.append(f"- {results['batch_correction_genes_dropped']:,} genes were "
                   f"not measured on every sample and were left out of the "
                   f"correction rather than zero-filled.")
    return "\n".join(out) + "\n"
