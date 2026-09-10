"""
Cross-modality gene analysis for GeneVariate.

Two questions this module answers, both Tk-free and headless:

1. **Comparison of the same gene across data modalities.** A gene measured on a
   microarray (log2 intensity), by bulk RNA-seq (log-CPM), and in single-cell
   pseudo-bulk lives on three different scales, so a naive test on the raw values
   is meaningless. :func:`compare_gene_across_modalities` harmonises each source
   to a common scale (z-score or rank) first, then compares the *shape* of the
   distribution across modalities and reports whether they agree.

2. **Connections between genes.** :func:`gene_coexpression` finds the genes whose
   expression tracks a query gene within one source (Pearson/Spearman), and
   :func:`coexpression_consensus` keeps only the partners whose connection holds
   - with a consistent sign - across several sources/modalities. That surfaces
   co-expression links that are reproducible rather than platform artefacts.

The canonical input is GeneVariate's "platform DataFrame" (rows = samples, a
``GSM`` column, optional ``series_id``/``Classified_*`` metadata, the remaining
columns numeric genes) - exactly what every loader and the NGS bridge produce.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_META_PREFIXES = ("GSM", "series_id")


# -----------------------------------------------------------------
# Small shared helpers (kept local to avoid importing the registry)
# -----------------------------------------------------------------
def _is_meta(col) -> bool:
    return col in _META_PREFIXES or str(col).startswith("Classified_")


def _gene_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if not _is_meta(c)]


def _find_gene_column(df: pd.DataFrame, gene: str) -> Optional[str]:
    """Case-insensitive match of a gene symbol to a platform gene column."""
    if not gene:
        return None
    g = str(gene).strip().upper()
    for c in df.columns:
        if _is_meta(c):
            continue
        if str(c).upper() == g:
            return c
    return None


def _gene_vector(df: pd.DataFrame, gene: str) -> Optional[np.ndarray]:
    col = _find_gene_column(df, gene)
    if col is None:
        return None
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


# gpl_downloader.classify_technology speaks in ingestion categories; this
# module reports in modalities. One map, so the two vocabularies cannot drift.
_CATEGORY_MODALITY = {
    "microarray": "microarray",
    "bulk-rna-seq": "rna-seq",
    "single-cell": "single-cell",
    "methylation": "methylation",
    "sequencing-other": "sequencing",
}


def modality_from_category(category: Optional[str]) -> Optional[str]:
    """Translate a ``gpl_downloader`` technology category into a modality.

    Returns ``None`` for an unknown/absent category so the caller can fall back
    to :func:`infer_modality`.
    """
    return _CATEGORY_MODALITY.get(str(category or "").strip().lower())


def infer_modality(name: str) -> str:
    """Guess a source's data modality from its platform key/name.

    Returns one of ``microarray``, ``rna-seq``, ``single-cell`` or the generic
    ``expression`` when nothing matches. This is only a label for the report -
    the harmonisation does the real cross-scale work.

    A bare GEO accession is deliberately *not* resolved here. "GPL" is GEO's
    platform namespace, not a technology: GPL570 is an Affymetrix array and
    GPL18573 is an Illumina NextSeq, and the accession alone cannot tell them
    apart. Only GEOmetadb can, so an opaque accession returns ``expression``
    and callers that hold the real technology should pass it via the
    ``modalities`` argument of :func:`compare_gene_across_modalities`.
    """
    n = str(name).lower()
    if "microarray" in n or "array" in n:
        return "microarray"
    if any(k in n for k in ("scrna", "single", "cellxgene", "census", "_sc", "sc_")):
        return "single-cell"
    if any(k in n for k in ("ngs", "rnaseq", "rna-seq", "rna_seq", "counts",
                            "deseq", "cpm", "salmon", "star", "kallisto")):
        return "rna-seq"
    return "expression"


# -----------------------------------------------------------------
# Scale harmonisation
# -----------------------------------------------------------------
def harmonize_vectors(vectors: Dict[str, np.ndarray],
                      method: str = "zscore") -> Dict[str, np.ndarray]:
    """Put per-source value vectors on a common scale for cross-modality tests.

    ``zscore`` - ``(x - mean) / std`` (centres + unit-variance each source).
    ``rank``   - percentile rank in ``[0, 1]`` (robust to scale *and* shape).
    ``none``   - return finite-filtered copies unchanged.
    NaNs are dropped per source. A degenerate source (std==0 or empty) yields an
    empty array so downstream code can skip it.
    """
    out: Dict[str, np.ndarray] = {}
    for key, vec in vectors.items():
        v = np.asarray(vec, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            out[key] = v
            continue
        if method == "none":
            out[key] = v
        elif method == "rank":
            order = v.argsort().argsort().astype(float)
            out[key] = order / max(v.size - 1, 1)
        else:  # zscore (default)
            sd = v.std(ddof=1) if v.size > 1 else 0.0
            out[key] = (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)
    return out


def harmonised_ks(a: np.ndarray, b: np.ndarray, method: str = "zscore",
                  n_perm: int = 999,
                  rng: Optional[np.random.Generator] = None
                  ) -> Tuple[float, float, float]:
    """Two-sample KS on harmonised vectors, with a *calibrated* null.

    Harmonisation estimates and removes each source's location and scale, which
    makes the two empirical CDFs systematically closer than the textbook
    two-sample KS null assumes. The asymptotic ``ks_2samp`` p-value is therefore
    far too conservative on harmonised data (measured type-I ~0.000 at a nominal
    0.05), and with it the test has almost no power against a genuine shape
    difference.

    The null is instead built by permutation: pool the harmonised values,
    re-split them at the original sizes, re-harmonise each half and recompute D.
    That reproduces the parameter-estimation step under H0 and restores nominal
    type-I error. Returns ``(D, p_permutation, p_asymptotic)``. ``n_perm=0``
    falls back to the raw asymptotic p-value.
    """
    from scipy.stats import ks_2samp
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    res = ks_2samp(a, b)
    d_obs, p_asym = float(res.statistic), float(res.pvalue)
    if not n_perm or n_perm <= 0:
        return d_obs, p_asym, p_asym
    if d_obs <= 0.0:
        # e.g. rank harmonisation with equal n -> identical harmonised vectors
        return d_obs, 1.0, p_asym
    rng = rng if rng is not None else np.random.default_rng(0)
    pool = np.concatenate([a, b])
    na = a.size
    ge = 1
    for _ in range(int(n_perm)):
        s = rng.permutation(pool)
        h = harmonize_vectors({"a": s[:na], "b": s[na:]}, method=method)
        if h["a"].size < 2 or h["b"].size < 2:
            continue
        if float(ks_2samp(h["a"], h["b"]).statistic) >= d_obs - 1e-12:
            ge += 1
    return d_obs, ge / (float(n_perm) + 1.0), p_asym


def _dist_class(v: np.ndarray) -> str:
    from genevariate.core.analysis.bimodality import classify_gene_distribution
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "n/a"
    return classify_gene_distribution(v)


def _source_stats(name: str, vec: np.ndarray,
                  modality: Optional[str] = None) -> Dict[str, object]:
    modality = modality or infer_modality(name)
    v = np.asarray(vec, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"source": name, "modality": modality, "n": 0}
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if v.size > 1 else 0.0
    return {
        "source": name,
        "modality": modality,
        "n": int(v.size),
        "mean": mean,
        "median": float(np.median(v)),
        "std": std,
        "cv": float(std / mean) if mean else float("nan"),
        "distribution": _dist_class(v),
    }


# -----------------------------------------------------------------
# 1) Same gene across modalities
# -----------------------------------------------------------------
def compare_gene_across_modalities(sources: Dict[str, pd.DataFrame],
                                   gene: str,
                                   method: str = "zscore",
                                   n_perm: int = 999,
                                   random_state: int = 0,
                                   modalities: Optional[Dict[str, str]] = None,
                                   ) -> Dict[str, object]:
    """Compare one gene's distribution across data modalities on a common scale.

    ``sources`` maps a source/platform name to its platform DataFrame. ``method``
    is the scale-harmonisation strategy: ``zscore``/``rank``/``none`` operate on
    the gene vector alone, while ``combat`` first runs ComBat batch correction on
    the matrix of genes shared across sources (real batch-effect removal, not
    mere rescaling) and then compares the corrected values directly. ``combat``
    silently falls back to ``zscore`` if no ComBat implementation is installed.

    The pairwise KS p-value is calibrated by permutation (``n_perm``
    re-splits of the pooled harmonised values, re-harmonised each time; set
    ``n_perm=0`` for the raw asymptotic p-value) - see :func:`harmonised_ks`.
    ``random_state`` keeps that permutation null deterministic. ``modalities``
    maps a source name to its known modality; supply it when the caller can
    look the technology up (the GUI does, from GEOmetadb), because a GEO
    accession on its own does not reveal one - see :func:`infer_modality`.

    Returns a dict with:
      ``table``       per-source stats (modality, n, mean, std, cv, class),
      ``pairwise``    harmonised pairwise KS D + calibrated p (and the raw
                      asymptotic p as ``p_asymptotic``) for every source pair,
      ``harmonized``  the harmonised vectors (for plotting),
      ``concordant``  whether every source shares the same distribution class,
      ``summary``     one-line text answer,
      ``report``      markdown description + analysis.
    """
    gene = str(gene).strip()

    # Batch correction happens on the full shared-gene matrix, not one gene, so
    # do it up front and then compare the corrected values without re-scaling.
    if method == "combat":
        try:
            from genevariate.core.analysis.integration import combat_correct
            sources = combat_correct(sources)
            method = "none"
        except Exception:
            method = "zscore"

    known = dict(modalities or {})
    raw: Dict[str, np.ndarray] = {}
    rows: List[Dict[str, object]] = []
    for name, df in sources.items():
        mod = known.get(name) or infer_modality(name)
        vec = _gene_vector(df, gene)
        if vec is None:
            rows.append({"source": name, "modality": mod,
                         "n": 0, "note": "gene not found"})
            continue
        raw[name] = vec
        rows.append(_source_stats(name, vec, modality=mod))
    table = pd.DataFrame(rows)

    harmonized = harmonize_vectors(raw, method=method)
    usable = [k for k in raw if harmonized.get(k) is not None
              and harmonized[k].size > 1]

    # Harmonised pairwise KS: are the *shapes* the same once scale is removed?
    pair_rows: List[Dict[str, object]] = []
    if len(usable) >= 2:
        rng = np.random.default_rng(random_state)
        for i in range(len(usable)):
            for j in range(i + 1, len(usable)):
                a, b = usable[i], usable[j]
                d, p, p_asym = harmonised_ks(harmonized[a], harmonized[b],
                                             method=method, n_perm=n_perm,
                                             rng=rng)
                if method == "rank":
                    # percentile-rank harmonisation maps every continuous
                    # source onto the same uniform, so a KS shape test on it
                    # can never reject. Say so instead of implying agreement.
                    verdict = "not testable (rank harmonisation removes shape)"
                else:
                    # A non-significant KS means we could not detect a shape
                    # difference - NOT proof the shapes are identical. Label it
                    # honestly rather than asserting the null.
                    verdict = "no sig. difference" if p >= 0.05 else "differs"
                pair_rows.append({
                    "source_a": a, "source_b": b,
                    "ks_D": float(d), "p_value": float(p),
                    "p_asymptotic": float(p_asym),
                    "verdict": verdict,
                })
    pairwise = pd.DataFrame(pair_rows)

    classes = {r["distribution"] for r in rows if r.get("n")}
    concordant = len(classes) == 1 and bool(classes)

    n_found = len([r for r in rows if r.get("n")])
    if n_found == 0:
        summary = f"{gene}: not found in any of the {len(sources)} source(s)."
    else:
        mods = ", ".join(sorted({str(r["modality"]) for r in rows if r.get("n")}))
        if pairwise.empty:
            shape = ""
        elif method == "rank":
            shape = (" Rank harmonisation removes distribution shape, so the "
                     "pairwise KS comparison is not informative here.")
        else:
            agree = int((pairwise["verdict"] == "no sig. difference").sum())
            shape = (f" After {method} harmonisation, {agree}/{len(pairwise)} "
                     f"source pair(s) show no detectable shape difference.")
        cls = (f" Distribution class is consistent ({classes.pop()})"
               if concordant else
               f" Distribution class varies across modalities "
               f"({', '.join(sorted(classes))})")
        summary = (f"{gene} compared across {n_found} source(s) "
                   f"[{mods}].{shape}{cls}.")

    report = _cross_modality_report(gene, table, pairwise, method, concordant)
    return {"gene": gene, "table": table, "pairwise": pairwise,
            "harmonized": harmonized, "concordant": concordant,
            "summary": summary, "report": report, "method": method}


def _cross_modality_report(gene, table, pairwise, method, concordant) -> str:
    lines = [f"# {gene} across modalities\n",
             f"Harmonisation: **{method}**.\n"]
    for _, r in table.iterrows():
        if r.get("n"):
            lines.append(
                f"- **{r['source']}** ({r['modality']}): {r.get('distribution', '?')}, "
                f"mean={r.get('mean', float('nan')):.3g}, "
                f"std={r.get('std', float('nan')):.3g}, n={int(r.get('n', 0)):,}")
        else:
            lines.append(f"- **{r['source']}** ({r.get('modality','?')}): "
                         f"{r.get('note', 'no data')}")
    if not pairwise.empty:
        lines.append("\n**Harmonised pairwise shape test (KS):**")
        for _, r in pairwise.iterrows():
            lines.append(
                f"- {r['source_a']} vs {r['source_b']}: D={r['ks_D']:.3f}, "
                f"p={r['p_value']:.3g} - {r['verdict']}")
    lines.append("\n**Verdict:** "
                 + ("the gene's expression profile is consistent across the "
                    "compared modalities." if concordant else
                    "the gene behaves differently across modalities - interpret "
                    "cross-platform claims with care."))
    return "\n".join(lines)


# -----------------------------------------------------------------
# 2) Connections between genes (co-expression)
# -----------------------------------------------------------------
def _corr_against(expr: pd.DataFrame, x: np.ndarray,
                  method: str = "pearson") -> pd.Series:
    """Correlate every column of ``expr`` (samples x genes) with vector ``x``.

    ``method`` is one of ``pearson``, ``spearman`` or ``rho``. ``rho`` is Lovell's
    proportionality metric (2015), which is robust to the compositional bias that
    inflates Pearson/Spearman correlations on normalised (CPM/log-CPM) data. It
    treats the values as log-expression and returns
    ``1 - var(a - b) / (var(a) + var(b))``.
    """
    X = expr.to_numpy(dtype=float)
    xv = np.asarray(x, dtype=float)
    # keep samples where the query gene is finite
    good = np.isfinite(xv)
    X = X[good]
    xv = xv[good]
    if xv.size < 3:
        return pd.Series(dtype=float)
    if method == "rho":
        return _rho_against(pd.DataFrame(X, columns=expr.columns), xv)
    if method == "spearman":
        xv = pd.Series(xv).rank().to_numpy()
        X = pd.DataFrame(X).rank().to_numpy()
    # column-wise Pearson via standardisation, NaN-safe per column
    xc = xv - xv.mean()
    xden = np.sqrt(np.nansum(xc * xc))
    with np.errstate(invalid="ignore", divide="ignore"):
        Xc = X - np.nanmean(X, axis=0)
        num = np.nansum(Xc * xc[:, None], axis=0)
        den = np.sqrt(np.nansum(Xc * Xc, axis=0)) * xden
        r = np.where(den > 0, num / den, np.nan)
    return pd.Series(r, index=expr.columns)


def _rho_against(expr: pd.DataFrame, x: np.ndarray) -> pd.Series:
    """Lovell's proportionality rho of every column of ``expr`` against ``x``.

    ``rho = 1 - var(x - g) / (var(x) + var(g))``, computed per gene column,
    NaN-safe. Values near 1 indicate proportional (co-expressed) log profiles;
    near 0 unrelated; negative anti-proportional.
    """
    X = expr.to_numpy(dtype=float)
    xv = np.asarray(x, dtype=float)
    n = xv.size
    if n < 3:
        return pd.Series(dtype=float)
    var_x = float(np.var(xv, ddof=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        diff = X - xv[:, None]
        # per-column variance ignoring NaNs (ddof=1)
        cnt = np.sum(np.isfinite(X), axis=0)
        mean_g = np.nanmean(X, axis=0)
        var_g = np.nansum((X - mean_g) ** 2, axis=0) / np.maximum(cnt - 1, 1)
        mean_d = np.nanmean(diff, axis=0)
        var_d = np.nansum((diff - mean_d) ** 2, axis=0) / np.maximum(cnt - 1, 1)
        denom = var_x + var_g
        rho = np.where(denom > 0, 1.0 - var_d / denom, np.nan)
    return pd.Series(rho, index=expr.columns)


def gene_coexpression(df: pd.DataFrame, gene: str,
                      method: str = "pearson",
                      top_n: int = 25,
                      min_abs: float = 0.0) -> pd.DataFrame:
    """Genes whose expression is connected to ``gene`` within one source.

    ``method`` is ``pearson``, ``spearman`` or ``rho`` (Lovell's proportionality,
    which avoids the compositional bias of correlation on normalised data).
    Returns a gene-indexed DataFrame with columns ``r`` and ``abs_r`` (``r`` holds
    the proportionality statistic when ``method='rho'``), sorted by ``abs_r``
    descending, filtered to ``|r| >= min_abs`` and the top ``top_n``. The query
    gene itself is excluded.
    """
    x = _gene_vector(df, gene)
    if x is None:
        raise ValueError(f"gene {gene!r} not found")
    qcol = _find_gene_column(df, gene)
    gene_cols = [c for c in _gene_columns(df) if c != qcol]
    if not gene_cols:
        return pd.DataFrame(columns=["r", "abs_r"])
    expr = df[gene_cols].apply(pd.to_numeric, errors="coerce")
    r = _corr_against(expr, x, method=method).dropna()
    out = pd.DataFrame({"r": r})
    out["abs_r"] = out["r"].abs()
    out = out[out["abs_r"] >= float(min_abs)]
    out = out.sort_values("abs_r", ascending=False)
    if top_n and top_n > 0:
        out = out.head(int(top_n))
    return out


def coexpression_consensus(sources: Dict[str, pd.DataFrame], gene: str,
                           method: str = "pearson",
                           top_n: int = 25,
                           min_abs: float = 0.3) -> Dict[str, object]:
    """Co-expression partners of ``gene`` that hold across several sources.

    A partner is kept only when it correlates with the query gene in at least
    two sources with the *same sign*. Ranked by mean ``|r|`` across the sources
    where it is present. Returns a dict with a ``table`` (gene x per-source r +
    ``mean_r``/``n_sources``), a ``summary`` and a markdown ``report``.
    """
    per: Dict[str, pd.Series] = {}
    used: List[str] = []
    for name, df in sources.items():
        try:
            full = gene_coexpression(df, gene, method=method, top_n=0,
                                     min_abs=0.0)
        except ValueError:
            continue
        if not full.empty:
            per[name] = full["r"]
            used.append(name)
    if len(per) < 2:
        return {"gene": gene, "table": pd.DataFrame(), "sources": used,
                "summary": f"Need {gene!r} in at least two sources for a "
                           "consensus (found in "
                           f"{len(per)}).",
                "report": ""}

    merged = pd.DataFrame(per)  # rows = genes, cols = sources
    signs = np.sign(merged)
    present = merged.notna().sum(axis=1)
    # consistent sign among the sources where present (ignore NaN)
    sign_consistent = signs.apply(
        lambda row: row.dropna().nunique() == 1 and len(row.dropna()) >= 2,
        axis=1)
    strong = (merged.abs() >= float(min_abs)).sum(axis=1) >= 2
    keep = merged[(present >= 2) & sign_consistent & strong].copy()
    keep["mean_r"] = keep[used].mean(axis=1, skipna=True)
    keep["n_sources"] = keep[used].notna().sum(axis=1)
    keep["abs_mean_r"] = keep["mean_r"].abs()
    keep = keep.sort_values("abs_mean_r", ascending=False)
    if top_n and top_n > 0:
        keep = keep.head(int(top_n))
    table = keep.drop(columns=["abs_mean_r"])

    n = len(table)
    top_names = ", ".join(list(table.index[:8])) if n else "none"
    summary = (f"{gene}: {n} co-expression partner(s) consistent across "
               f"{len(used)} source(s) [{', '.join(used)}]. Top: {top_names}.")
    rlines = [f"# {gene} connections across {len(used)} source(s)\n",
              f"Method: **{method}**, |r| >= {min_abs} in >=2 sources, "
              "consistent sign.\n"]
    for g, row in table.iterrows():
        rlines.append(f"- **{g}**: mean r={row['mean_r']:.3f} "
                      f"(in {int(row['n_sources'])} source(s))")
    report = "\n".join(rlines)
    return {"gene": gene, "table": table, "sources": used,
            "summary": summary, "report": report}
