"""
Enrichment analysis driven by GeneVariate sample-level Condition labels.

Two entry points:
  - run_enrichr(gene_list, gene_sets=...)       over-representation (ORA)
  - run_prerank_gsea(ranked_df, gene_sets=...)  GSEA on a ranked gene list

rank_genes_by_condition(expr_df, labels) produces a ranked list suitable for
feeding into run_prerank_gsea, using a simple mean-difference + t-statistic
(no external DE dependency required).

The canonical input DataFrame is GeneVariate's standard:
    GSM | series_id | GENE1 | GENE2 | ...  (rows = samples)
Sample labels come from the LLM condition curator (case vs control / cluster
names), passed as a dict {GSM -> label}.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    import gseapy
    _HAS_GSEAPY = True
except Exception:
    gseapy = None
    _HAS_GSEAPY = False


DEFAULT_LIBRARIES: Tuple[str, ...] = (
    "GO_Biological_Process_2023",
    "KEGG_2021_Human",
    "Reactome_2022",
    "MSigDB_Hallmark_2020",
)


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------
def benjamini_hochberg(pvalues: Sequence[float]) -> np.ndarray:
    """
    Benjamini-Hochberg FDR-adjusted p-values (q-values).

    NaN p-values are ignored in the ranking and returned as NaN. The result
    is the standard step-up BH adjustment with monotonicity enforced.
    """
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    finite = np.isfinite(p)
    m = int(finite.sum())
    if m == 0:
        return out
    idx = np.where(finite)[0]
    pv = p[idx]
    order = np.argsort(pv)
    ranked = pv[order]
    n = m
    adj = ranked * n / (np.arange(1, n + 1))
    # enforce monotonicity from the largest p-value downward
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    q = np.empty(n, dtype=float)
    q[order] = adj
    out[idx] = q
    return out


#: Columns that are numeric but describe the sample rather than measure a gene.
#: Everything else non-numeric is excluded on dtype alone, so this only has to
#: name the counts and identifiers that would otherwise pass for expression.
_NON_GENE_COLS = frozenset({"GSM", "series_id", "gse", "gpl", "n_cells"})


def _expr_from_canonical(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split canonical (GSM|series_id|gene..) into (expr genes x samples, GSM index).

    A gene column is a numeric column that is not one of the known metadata
    names. The dtype test matters for frames that carry descriptive annotation
    beside the expression - a pseudo-bulk frame brings donor_id, cell_type,
    pseudobulk_method, disease and the Classified_* labels - because
    transposing those into the matrix would add one fake gene per annotation
    and quietly contaminate every ranking computed from it.
    """
    if "GSM" not in df.columns:
        raise ValueError("Input must be in GeneVariate canonical format (GSM column required).")
    gsm = df["GSM"].astype(str).str.upper()
    gene_cols = [c for c in df.columns
                 if c not in _NON_GENE_COLS
                 and pd.api.types.is_numeric_dtype(df[c])]
    expr = df[gene_cols].T.copy()
    expr.columns = gsm.values
    return expr, gsm


def _theo_logf_winsor_moments(dfree: float, d0: float,
                              p: float) -> Tuple[float, float]:
    """
    Mean and variance of z = log F(dfree, d0), winsorized (clamped) at tail
    proportion ``p`` on each side, for unit prior scale (s0² = 1).

    Winsorizing an F-variate at its p / (1−p) quantiles is equivalent, in
    z = log F space, to clamping z at those quantiles. Computing the moments in
    probability space (u = CDF) keeps the quadrature limits bounded and the
    integrand smooth:

        E[W]  = p·log q_p + p·log q_{1−p} + ∫_p^{1−p} log F⁻¹(u) du
        E[W²] = p·log²q_p + p·log²q_{1−p} + ∫_p^{1−p} (log F⁻¹(u))² du

    Returns (E[W], Var[W]).
    """
    from scipy.stats import f as _f
    from scipy.integrate import quad
    fr = _f(dfree, d0)
    zlo = float(np.log(fr.ppf(p)))
    zhi = float(np.log(fr.ppf(1.0 - p)))
    i1 = quad(lambda u: np.log(fr.ppf(u)), p, 1.0 - p, limit=100)[0]
    i2 = quad(lambda u: np.log(fr.ppf(u)) ** 2, p, 1.0 - p, limit=100)[0]
    ew = p * zlo + p * zhi + i1
    ew2 = p * zlo ** 2 + p * zhi ** 2 + i2
    return ew, ew2 - ew ** 2


def _empirical_bayes_variance(s2: np.ndarray, dfree: float,
                               winsor: float = 0.05,
                               d0_cap: float = 1e6
                               ) -> Tuple[np.ndarray, float, float]:
    """
    limma-style empirical Bayes variance shrinkage (Smyth 2004) with the
    robust, winsorized hyperparameter estimator of Phipson et al. (2016).

    Given per-gene sample variances ``s2`` with ``dfree`` degrees of freedom
    each, the prior parameters (s0², d0) are found by matching the WINSORIZED
    moments of z = log(s²) to their theoretical values under log F(dfree, d0):
    d0 solves ``Var_winsor(d0) = observed winsorized var(z)`` and s0² follows
    from the winsorized mean. Winsorizing (clamping the extreme ``winsor``
    fraction of z-values to the tail quantiles) protects the fit against
    hypervariable / outlier genes, while matching against the *theoretical*
    winsorized moments removes the variance-deflation bias that plain quantile
    trimming would otherwise introduce (which inflates d0 and over-shrinks).
    The posterior shrunken variances are

        tilde_s² = (d0 · s0² + dfree · s²) / (d0 + dfree).

    Returns (shrunken_s2, d0, s0_squared).
    References:
      * Smyth GK (2004), Stat Appl Genet Mol Biol 3, Article 3.
      * Phipson B, Lee S, Majewski IJ, Alexander WS, Smyth GK (2016),
        Ann Appl Stat 10:946-963 (robust empirical Bayes / winsorized fitFDist).
    """
    s2 = np.asarray(s2, dtype=float)
    valid = np.isfinite(s2) & (s2 > 0)
    if valid.sum() < 30:
        # Too few genes to fit the prior, but returning s² untouched is not the
        # safe fallback it looks like: a gene that is exactly constant within
        # both groups has s² = 0, so se = 0, t = inf and p = 0.0 - a claim of
        # infinite evidence, reported top of the ranking with a q of exactly
        # zero, from a caller that asked for moderation and silently got none.
        # Shrink with a single prior degree of freedom toward the median
        # instead. It is the same posterior formula with the weakest prior
        # that still exists, so no variance can be zero and no t infinite.
        s0_2 = float(np.nanmedian(s2[valid])) if valid.any() else 1.0
        d0 = 1.0
        with np.errstate(invalid="ignore"):
            out = (d0 * s0_2 + dfree * s2) / (d0 + dfree)
        out = np.where(np.isfinite(s2), out, np.nan)
        return out, d0, s0_2

    z = np.log(s2[valid])
    zlo, zhi = np.quantile(z, [winsor, 1.0 - winsor])
    zw = np.clip(z, zlo, zhi)
    wv_obs = float(np.var(zw, ddof=1))
    wm_obs = float(np.mean(zw))

    # Theoretical winsorized variance decreases monotonically in d0 toward a
    # floor at d0 -> inf (the dfree-only spread). If the observed spread is at
    # or below that floor there is no detectable prior heterogeneity, so
    # d0 = inf: shrink every gene to the common variance s0².
    from scipy.optimize import brentq
    _, v_floor = _theo_logf_winsor_moments(dfree, d0_cap, winsor)
    if wv_obs <= v_floor:
        d0 = d0_cap
    else:
        def _gap(log_d0: float) -> float:
            _, v = _theo_logf_winsor_moments(dfree, float(np.exp(log_d0)),
                                             winsor)
            return v - wv_obs
        lo, hi = np.log(0.05), np.log(d0_cap)
        d0 = 0.05 if _gap(lo) < 0 else float(np.exp(brentq(_gap, lo, hi,
                                                           xtol=1e-6)))

    ew, _ = _theo_logf_winsor_moments(dfree, d0, winsor)
    s0_2 = max(float(np.exp(wm_obs - ew)), 1e-12)

    out = np.array(s2, dtype=float)
    valid_mask = np.isfinite(out) & (out > 0)
    out[valid_mask] = (d0 * s0_2 + dfree * out[valid_mask]) / (d0 + dfree)
    # fall back to s0² for invalid positions
    out[~valid_mask] = s0_2
    return out, float(d0), s0_2


def _study_map(df: pd.DataFrame) -> Optional[pd.Series]:
    """GSM -> study for a canonical frame, or None if it carries no study column."""
    for col in ("series_id", "gse"):
        if col not in df.columns:
            continue
        s = pd.Series(df[col].astype(str).str.strip().values,
                      index=df["GSM"].astype(str).str.upper().values)
        s = s[~s.isin(("", "nan", "None"))]
        s = s[~s.index.duplicated(keep="first")]
        if len(s):
            return s
    return None


def _collapse_by_study(expr: pd.DataFrame, samples: Sequence[str],
                       study_of: pd.Series) -> pd.DataFrame:
    """One column per study: the mean of that study's samples in this group.

    Averaging within a series first is what makes the later test a comparison of
    independent studies. Pooled, a 400-sample series and a 6-sample series carry
    equal weight per sample, so the large one decides the contrast, and the
    degrees of freedom count samples that share protocol, batch and often
    subject as though each were independent evidence.
    """
    keep = [s for s in samples if s in study_of.index]
    if not keep:
        return pd.DataFrame(index=expr.index)
    sub = expr[keep].astype(float)
    return sub.T.groupby(study_of.reindex(keep).values).mean().T


def rank_genes_by_condition(df: pd.DataFrame,
                            labels: Dict[str, str],
                            case_label: str,
                            control_label: str,
                            moderated: bool = False) -> pd.DataFrame:
    """
    Rank genes by case-vs-control mean difference, with optional limma-style
    empirical Bayes variance shrinkage.

    Parameters
    ----------
    moderated : bool
        If True, replace per-gene sample variances with empirical Bayes
        shrunken variances (Smyth 2004). This is the recommended option
        for microarray / log-transformed RNA-seq data with few replicates.
        For raw RNA-seq counts, prefer `pydeseq2` (see README).

    The unit of replication is the **study**, not the sample, whenever the frame
    carries ``series_id`` / ``gse``: each series is collapsed to one value per
    group before testing. A GEO platform frame is a pile of separate
    experiments, and treating its samples as independent replicates inflates the
    degrees of freedom by roughly the average series size - enough to make
    almost every gene significant. ``result.attrs`` reports ``unit``,
    ``n_case``, ``n_control`` and a ``note`` describing what was compared.

    Returns a DataFrame indexed by gene with columns:
        mean_case, mean_control, logFC, t_stat, p_value, padj, rank
    `padj` is the Benjamini-Hochberg FDR-adjusted p-value. Ranking uses the
    (moderated) t-stat, which is monotonic in both magnitude and direction -
    suitable for gseapy prerank.
    """
    expr, gsm = _expr_from_canonical(df)
    labels = {str(k).upper(): v for k, v in labels.items()}

    case_samples = [g for g in gsm if labels.get(g) == case_label]
    ctrl_samples = [g for g in gsm if labels.get(g) == control_label]
    if len(case_samples) < 2 or len(ctrl_samples) < 2:
        raise ValueError(
            f"Need at least 2 samples per group - got "
            f"case={len(case_samples)}, control={len(ctrl_samples)}"
        )

    # Samples from one GEO series are not independent replicates. When the frame
    # says which series each sample came from, each series is reduced to one
    # value per group first, so the test compares studies and the degrees of
    # freedom count studies. Frames without that column keep the pooled
    # behaviour, and say so in ``note``.
    study_of = _study_map(df)
    unit, note = "samples", ""
    case = ctrl = None
    if study_of is not None:
        c = _collapse_by_study(expr, case_samples, study_of)
        k = _collapse_by_study(expr, ctrl_samples, study_of)
        if c.shape[1] >= 2 and k.shape[1] >= 2:
            case, ctrl, unit = c, k, "studies"
            shared = sorted(set(c.columns) & set(k.columns))
            note = (
                f"Tested across studies: {c.shape[1]} case studies collapsed "
                f"from {len(case_samples)} samples, {k.shape[1]} control "
                f"studies from {len(ctrl_samples)}. "
                + (f"{len(shared)} studies contribute to both groups. "
                   if shared else
                   "No study contributes to both groups, so every difference "
                   "here is also a between-study difference. ")
                + "Samples within a series share protocol and batch, so testing "
                  "them as independent would inflate the degrees of freedom.")
        else:
            note = (
                f"Only {c.shape[1]} case / {k.shape[1]} control studies, too "
                f"few to test across studies, so samples were pooled. These "
                f"p-values assume an independence the design does not provide.")
    if case is None:
        case = expr[case_samples].astype(float)
        ctrl = expr[ctrl_samples].astype(float)
        if not note:
            note = ("No series_id/gse column on this frame, so samples were "
                    "pooled and assumed independent.")

    nc, nk = case.shape[1], ctrl.shape[1]
    mc = case.mean(axis=1)
    mk = ctrl.mean(axis=1)
    logfc = mc - mk

    # Pooled variance with (nc+nk-2) df - the natural input for limma EB
    var_c = case.var(axis=1, ddof=1)
    var_k = ctrl.var(axis=1, ddof=1)
    dfree = nc + nk - 2
    pooled = ((nc - 1) * var_c + (nk - 1) * var_k) / dfree

    if moderated:
        shrunken, d0, s0_2 = _empirical_bayes_variance(pooled.values, dfree)
        sigma2 = pd.Series(shrunken, index=pooled.index)
        se = np.sqrt(sigma2 * (1.0 / nc + 1.0 / nk))
        df_total = dfree + d0  # posterior df
    else:
        # Welch-style separate-variance
        vc = var_c.replace(0, np.nan)
        vk = var_k.replace(0, np.nan)
        se = np.sqrt(vc / nc + vk / nk)
        # Welch-Satterthwaite df (per-gene)
        df_total = ((vc / nc + vk / nk) ** 2) / (
            (vc / nc) ** 2 / max(nc - 1, 1) + (vk / nk) ** 2 / max(nk - 1, 1)
        )

    t = (logfc / se)
    # p-value (two-sided). For moderated t, use scalar df; for Welch, use per-gene.
    try:
        from scipy.stats import t as t_dist
        # Survival function (sf = 1 - cdf) keeps precision in the far tail;
        # 1 - cdf(|t|) underflows to exactly 0.0 for large |t| and would
        # corrupt the downstream BH-FDR ranking.
        # A gene with no usable t - never measured on this platform, or with a
        # logFC that could not be formed - has no p-value. Substituting t = 0
        # gives it p = 1.0, which is not a null result but a fabricated one,
        # and BH then counts it as a test that was performed: with a scalar df
        # every such gene entered the denominator and inflated every q in the
        # moderated path. Leaving it NaN lets `benjamini_hochberg` skip it, as
        # the Welch branch already did by accident through its NaN df.
        tv = np.abs(t.values.astype(float))
        if np.isscalar(df_total) or isinstance(df_total, (int, float)):
            p = 2.0 * t_dist.sf(tv, df_total)
        else:
            p = 2.0 * t_dist.sf(tv, np.asarray(df_total).clip(min=1.0))
    except Exception:
        p = np.full_like(t.values, np.nan, dtype=float)

    out = pd.DataFrame({
        "mean_case": mc,
        "mean_control": mk,
        "logFC": logfc,
        "t_stat": t,
        "p_value": pd.Series(p, index=t.index),
    })
    out["padj"] = benjamini_hochberg(out["p_value"].values)
    # A gene with no t-statistic keeps no rank. Filling it with 0.0 places it
    # in the middle of the prerank walk as a measured gene of neutral effect,
    # and it disarmed the `dropna()` that `run_prerank_gsea` applies to this
    # column for that very reason.
    out["rank"] = out["t_stat"].astype(float)
    out = out.sort_values("rank", ascending=False, na_position="last")
    # What n counts is the difference between a defensible p-value and an
    # inflated one, so it travels with the result instead of being inferred.
    out.attrs["unit"] = unit
    out.attrs["n_case"] = int(nc)
    out.attrs["n_control"] = int(nk)
    out.attrs["note"] = note
    return out


# -----------------------------------------------------------------
# Over-representation (Enrichr)
# -----------------------------------------------------------------
def run_enrichr(gene_list: Sequence[str],
                gene_sets: Sequence[str] = DEFAULT_LIBRARIES,
                organism: str = "human",
                outdir: Optional[str] = None,
                cutoff: float = 0.05) -> pd.DataFrame:
    """
    Run Enrichr over-representation on `gene_list`. Returns a DataFrame
    of significant terms (Adjusted P-value < cutoff) across all libraries.
    """
    if not _HAS_GSEAPY:
        raise RuntimeError("gseapy is not installed. `pip install gseapy`")
    gene_list = [str(g).strip() for g in gene_list if str(g).strip()]
    if not gene_list:
        raise ValueError("gene_list is empty")

    enr = gseapy.enrichr(
        gene_list=list(gene_list),
        gene_sets=list(gene_sets),
        organism=organism,
        outdir=outdir,
        cutoff=cutoff,
        no_plot=True,
    )
    res = enr.results if enr is not None else pd.DataFrame()
    if res is None or res.empty:
        return pd.DataFrame()
    sig = res[res["Adjusted P-value"] < cutoff].copy()
    sig = sig.sort_values("Adjusted P-value").reset_index(drop=True)
    return sig


# -----------------------------------------------------------------
# GSEA prerank
# -----------------------------------------------------------------
def run_prerank_gsea(ranked: pd.DataFrame,
                     gene_sets: Sequence[str] = DEFAULT_LIBRARIES,
                     outdir: Optional[str] = None,
                     permutation_num: int = 1000,
                     seed: int = 42) -> pd.DataFrame:
    """
    GSEA prerank on the output of `rank_genes_by_condition`.
    `ranked` must have a 'rank' column and a gene index.
    Returns merged gseapy results for each library.
    """
    if not _HAS_GSEAPY:
        raise RuntimeError("gseapy is not installed. `pip install gseapy`")
    if "rank" not in ranked.columns:
        raise ValueError("ranked DataFrame must contain a 'rank' column")

    rnk = ranked["rank"].dropna().sort_values(ascending=False)
    rnk.index = rnk.index.astype(str).str.upper()
    rnk = rnk[~rnk.index.duplicated(keep="first")]

    frames: List[pd.DataFrame] = []
    for lib in gene_sets:
        try:
            res = gseapy.prerank(
                rnk=rnk.reset_index().rename(columns={"index": "gene", "rank": "score"}),
                gene_sets=lib,
                outdir=outdir,
                permutation_num=permutation_num,
                seed=seed,
                no_plot=True,
            )
            if res is not None and res.res2d is not None:
                d = res.res2d.copy()
                d["library"] = lib
                frames.append(d)
        except Exception as e:
            frames.append(pd.DataFrame([{"library": lib, "error": str(e)}]))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


# -----------------------------------------------------------------
# Markdown report
# -----------------------------------------------------------------
def _df_to_md(frame: pd.DataFrame) -> str:
    """Render a small DataFrame as a GitHub-flavored markdown table.
    Falls back to a hand-rolled implementation when `tabulate` isn't installed."""
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        pass
    cols = [str(c) for c in frame.columns]
    def _fmt(x):
        if isinstance(x, float):
            return f"{x:.4g}"
        return str(x)
    rows = [[_fmt(v) for v in row] for row in frame.itertuples(index=False, name=None)]
    head = "| " + " | ".join(cols) + " |"
    sep  = "|" + "|".join("---" for _ in cols) + "|"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join([head, sep, body])


def enrichment_report_markdown(ora: pd.DataFrame,
                               gsea: pd.DataFrame,
                               comparison: str,
                               top_n: int = 15,
                               out_path: Optional[str] = None,
                               ranked: Optional[pd.DataFrame] = None) -> str:
    """
    Build a short markdown report summarising ORA + GSEA results.
    If out_path is given, the report is written to disk and the path returned;
    otherwise the markdown string itself is returned.

    Pass ``ranked`` (the output of :func:`rank_genes_by_condition`) to record
    what was compared. Enrichment inherits every property of the ranking it was
    given, so a report that omits whether n counted studies or samples cannot be
    checked by the person reading it.
    """
    lines: List[str] = []
    lines.append(f"# Enrichment report - {comparison}\n")
    if ranked is not None and ranked.attrs.get("note"):
        lines.append("## What was compared")
        lines.append(f"- **Unit of replication**: {ranked.attrs.get('unit')} "
                     f"(n_case={ranked.attrs.get('n_case')}, "
                     f"n_control={ranked.attrs.get('n_control')})")
        lines.append(f"- {ranked.attrs['note']}\n")
    lines.append("## Over-representation (Enrichr)")
    if ora is None or ora.empty:
        lines.append("_No significant terms._\n")
    else:
        cols = [c for c in ("Gene_set", "Term", "Adjusted P-value",
                            "Combined Score", "Genes") if c in ora.columns]
        lines.append(_df_to_md(ora[cols].head(top_n)))
        lines.append("")

    lines.append("## GSEA prerank")
    if gsea is None or gsea.empty:
        lines.append("_No results._\n")
    else:
        cols = [c for c in ("library", "Term", "NES", "FDR q-val", "NOM p-val",
                            "Lead_genes") if c in gsea.columns]
        available = [c for c in cols if c in gsea.columns]
        sort_col = "FDR q-val" if "FDR q-val" in gsea.columns else available[0]
        lines.append(_df_to_md(gsea[available].sort_values(sort_col).head(top_n)))
        lines.append("")

    md = "\n".join(lines)
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(md)
        return out_path
    return md
