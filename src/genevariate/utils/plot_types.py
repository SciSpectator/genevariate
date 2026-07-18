"""
GeneVariate — Extra plot types for gene expression & label analyses.

All helpers assume ``apply_genevariate_style()`` has been called, so they
keep per-axes tweaks to a minimum and rely on the global rcParams.

Plot catalogue
--------------
* plot_ma            — MA (log ratio vs log mean)
* plot_bland_altman  — agreement / bias between two measurements
* plot_ecdf          — empirical cumulative distribution
* plot_hexbin        — density-safe scatter for large samples
* plot_qq            — quantile-quantile against normal
* plot_enrichment_bar   — enrichment fold-change with significance bars
* plot_enrichment_dot   — dot plot (gene set × group)  — standard enrichment view
* plot_enrichment_volcano — log2(OR) vs -log10(q)

Enrichment utilities
--------------------
* enrichment_fisher(a, b, c, d)  — 2x2 Fisher exact + OR
* label_enrichment(labels, group_mask, background_mask=None)  — per-value
  enrichment test (with Bonferroni + Benjamini–Hochberg)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .viz_style import (
    AERO, apply_plot_polish, apply_aero_background,
    cmap_for, legend_outside, palette_for, style_axis, TYPOGRAPHY,
)


# =====================================================================
# Basic statistical plots
# =====================================================================

def plot_ma(ax, sample: np.ndarray, reference: np.ndarray,
            title: str = "MA plot",
            highlight_mask: Optional[np.ndarray] = None) -> None:
    """MA plot — M = log2(sample/reference), A = 0.5*log2(sample*reference).

    Accepts raw intensities (will clip non-positive to avoid log warnings).
    """
    sample = np.asarray(sample, dtype=float)
    reference = np.asarray(reference, dtype=float)
    ok = (sample > 0) & (reference > 0) & np.isfinite(sample) & np.isfinite(reference)
    s, r = sample[ok], reference[ok]
    if len(s) == 0:
        ax.text(0.5, 0.5, "No finite data", ha="center", va="center",
                transform=ax.transAxes, color=AERO["muted"])
        return
    M = np.log2(s / r)
    A = 0.5 * np.log2(s * r)

    ax.scatter(A, M, s=10, alpha=0.35, c=AERO["accent_dark"],
               edgecolors="none", rasterized=True)
    if highlight_mask is not None and highlight_mask.any():
        hm = highlight_mask[ok]
        ax.scatter(A[hm], M[hm], s=14, alpha=0.8, c=AERO["danger"],
                   edgecolors="none", label="highlighted", rasterized=True)
    ax.axhline(0, color=AERO["danger"], ls="--", lw=0.9, alpha=0.7)
    style_axis(ax, xlabel="A = ½·log2(sample·ref)",
               ylabel="M = log2(sample/ref)", title=title)


def plot_bland_altman(ax, a: np.ndarray, b: np.ndarray,
                      title: str = "Bland-Altman") -> None:
    """Agreement plot: (a-b) vs mean(a,b), with ±1.96σ limits."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) == 0:
        ax.text(0.5, 0.5, "No finite data", ha="center", va="center",
                transform=ax.transAxes, color=AERO["muted"])
        return
    mean = (a + b) / 2.0
    diff = a - b
    mu = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    lo, hi = mu - 1.96 * sd, mu + 1.96 * sd

    ax.scatter(mean, diff, s=14, alpha=0.55, c=AERO["accent"],
               edgecolors="none", rasterized=True)
    ax.axhline(mu, color=AERO["danger"], ls="--", lw=1.0,
               label=f"mean = {mu:+.3f}")
    ax.axhline(hi, color=AERO["muted"], ls=":", lw=0.9,
               label=f"+1.96σ = {hi:+.3f}")
    ax.axhline(lo, color=AERO["muted"], ls=":", lw=0.9,
               label=f"-1.96σ = {lo:+.3f}")
    style_axis(ax, xlabel="mean(a, b)", ylabel="a − b", title=title)
    ax.legend(**TYPOGRAPHY["legend"])


def plot_ecdf(ax, groups: Dict[str, np.ndarray],
              title: str = "ECDF", xlabel: str = "value") -> None:
    """Empirical cumulative distribution for one or several groups."""
    colors = palette_for(len(groups), "discrete")
    for (name, values), clr in zip(groups.items(), colors):
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        xs = np.sort(v)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.step(xs, ys, where="post", color=clr, lw=1.8, label=f"{name} (n={len(xs)})")
    style_axis(ax, xlabel=xlabel, ylabel="F(x)", title=title)
    ax.set_ylim(0, 1.02)
    ax.legend(**TYPOGRAPHY["legend"])


def plot_hexbin(ax, x: np.ndarray, y: np.ndarray, gridsize: int = 40,
                title: str = "Density", xlabel: str = "x",
                ylabel: str = "y", log_count: bool = True) -> None:
    """Hexbin for dense scatter; ``log_count`` gives a log-scaled count cmap."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) == 0:
        ax.text(0.5, 0.5, "No finite data", ha="center", va="center",
                transform=ax.transAxes, color=AERO["muted"])
        return
    hb = ax.hexbin(x, y, gridsize=gridsize, cmap=cmap_for("intensity"),
                   mincnt=1, bins="log" if log_count else None,
                   linewidths=0.1, edgecolors=AERO["plot_bg"])
    cbar = ax.figure.colorbar(hb, ax=ax, pad=0.02)
    cbar.set_label("log10(count)" if log_count else "count",
                   fontsize=TYPOGRAPHY["annot"]["fontsize"])
    style_axis(ax, xlabel=xlabel, ylabel=ylabel, title=title)


def plot_qq(ax, values: np.ndarray, title: str = "Q-Q (normal)") -> None:
    """Quantile-quantile plot of ``values`` against a standard normal."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        ax.text(0.5, 0.5, "Need ≥2 points", ha="center", va="center",
                transform=ax.transAxes, color=AERO["muted"])
        return
    try:
        from scipy.stats import probplot
        probplot(v, dist="norm", plot=ax)
    except Exception:
        # Manual fallback
        n = len(v)
        theoretical = np.sort(np.random.normal(size=n))
        empirical = np.sort(v)
        ax.scatter(theoretical, empirical, s=10, alpha=0.6,
                   c=AERO["accent"], edgecolors="none")
    # Re-skin what probplot drew
    for ln in ax.get_lines():
        if ln.get_linestyle() in ("-", "--"):
            ln.set_color(AERO["danger"])
            ln.set_linewidth(1.0)
    style_axis(ax, xlabel="theoretical quantiles",
               ylabel="sample quantiles", title=title)


# =====================================================================
# Enrichment analysis
# =====================================================================

@dataclass
class EnrichmentResult:
    """One row of an enrichment table."""
    term: str
    k: int                # group ∩ term
    K: int                # total term
    n: int                # group size
    N: int                # background size
    odds_ratio: float
    p_value: float
    q_value: float = 1.0
    fold_change: float = 1.0

    def as_dict(self) -> dict:
        return dict(term=self.term, k=self.k, K=self.K, n=self.n, N=self.N,
                    odds_ratio=self.odds_ratio, p_value=self.p_value,
                    q_value=self.q_value, fold_change=self.fold_change)


def _bh_fdr(pvals: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR. Returns q-values (same shape as input)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n, dtype=float)
    out[order] = q
    return out


def enrichment_fisher(k: int, K: int, n: int, N: int
                      ) -> Tuple[float, float]:
    """Fisher exact (two-sided) on the 2×2:

        [[ k,      n - k      ],
         [ K - k,  N - n - K + k ]]

    Returns (odds_ratio, p_value). Safe against zeros.
    """
    try:
        from scipy.stats import fisher_exact
    except Exception:
        return (1.0, 1.0)
    a = max(int(k), 0)
    b = max(int(n - k), 0)
    c = max(int(K - k), 0)
    d = max(int(N - n - K + k), 0)
    try:
        odds, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
    except Exception:
        return (1.0, 1.0)
    return (float(odds), float(p))


def label_enrichment(labels: pd.Series,
                     group_mask: pd.Series,
                     background_mask: Optional[pd.Series] = None,
                     min_count: int = 3,
                     alpha: float = 0.05
                     ) -> pd.DataFrame:
    """Test each label value for over/under-representation in a group.

    Parameters
    ----------
    labels          series of categorical label values (e.g. tissues)
    group_mask      boolean series — foreground (e.g. samples in a gene region)
    background_mask optional boolean — restrict universe; defaults to all non-NA
    min_count       drop label values observed <min_count times in the universe
    alpha           significance threshold (used only for ``.significant`` flag)

    Returns a dataframe sorted by p-value, columns:
        term, k, K, n, N, odds_ratio, fold_change, p_value, q_value, significant
    """
    labels = pd.Series(labels).astype(object)
    group_mask = pd.Series(group_mask).fillna(False).astype(bool)
    if background_mask is None:
        background_mask = labels.notna() & (labels.astype(str).str.len() > 0)
    else:
        background_mask = pd.Series(background_mask).fillna(False).astype(bool)
    # align indices
    idx = labels.index.intersection(group_mask.index).intersection(background_mask.index)
    labels = labels.loc[idx]
    group_mask = group_mask.loc[idx]
    background_mask = background_mask.loc[idx]

    universe = labels[background_mask]
    if universe.empty:
        return pd.DataFrame(columns=["term", "k", "K", "n", "N",
                                     "odds_ratio", "fold_change",
                                     "p_value", "q_value", "significant"])

    N = int(background_mask.sum())
    n = int((group_mask & background_mask).sum())
    counts = universe.value_counts()

    rows: List[EnrichmentResult] = []
    for term, K in counts.items():
        if K < min_count:
            continue
        in_term = (labels == term) & background_mask
        k = int((group_mask & in_term).sum())
        odds, p = enrichment_fisher(k, int(K), n, N)
        # fold change = (k/n) / (K/N)  — ratio of observed to expected fraction
        obs_frac = (k / n) if n > 0 else 0.0
        exp_frac = (K / N) if N > 0 else 0.0
        fc = (obs_frac / exp_frac) if exp_frac > 0 else (float("inf") if obs_frac > 0 else 1.0)
        rows.append(EnrichmentResult(
            term=str(term), k=k, K=int(K), n=n, N=N,
            odds_ratio=odds, p_value=p, fold_change=fc))

    if not rows:
        return pd.DataFrame(columns=["term", "k", "K", "n", "N",
                                     "odds_ratio", "fold_change",
                                     "p_value", "q_value", "significant"])

    df = pd.DataFrame([r.as_dict() for r in rows])
    df["q_value"] = _bh_fdr(df["p_value"].values)
    df["significant"] = df["q_value"] < alpha
    df = df.sort_values("p_value").reset_index(drop=True)
    return df


# =====================================================================
# Enrichment plots
# =====================================================================

def plot_enrichment_bar(ax, enrichment_df: pd.DataFrame, *,
                        top_n: int = 20,
                        metric: str = "fold_change",
                        title: str = "Label enrichment") -> None:
    """Horizontal bar chart of top-N terms by p-value.

    Bar color encodes log2(fold_change); bar length = metric.
    Adds small asterisks for significance (q<0.05 / 0.01 / 0.001).
    """
    if enrichment_df is None or enrichment_df.empty:
        ax.text(0.5, 0.5, "No enrichment results", ha="center", va="center",
                transform=ax.transAxes, color=AERO["muted"])
        return

    df = enrichment_df.head(top_n).iloc[::-1].copy()  # reverse for top-at-top
    if metric == "odds_ratio":
        vals = np.log2(np.clip(df["odds_ratio"].values, 1e-6, 1e6))
        xlabel = "log2(odds ratio)"
    elif metric == "minus_log10_p":
        vals = -np.log10(np.clip(df["p_value"].values, 1e-300, 1.0))
        xlabel = "-log10(p)"
    else:  # fold_change
        vals = np.log2(np.clip(df["fold_change"].values, 1e-6, 1e6))
        xlabel = "log2(fold change)"

    # color by sign (up=green, down=danger) with saturation via magnitude
    max_abs = float(np.max(np.abs(vals))) if len(vals) else 1.0
    max_abs = max_abs if max_abs > 0 else 1.0
    colors = []
    for v in vals:
        base = AERO["green_dark"] if v >= 0 else AERO["danger"]
        colors.append(base)

    y_pos = np.arange(len(df))
    ax.barh(y_pos, vals, color=colors, edgecolor=AERO["text"], lw=0.4, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["term"].tolist())
    ax.axvline(0, color=AERO["muted"], lw=0.8, ls="-")

    # annotate with stars
    for i, (v, q) in enumerate(zip(vals, df["q_value"].values)):
        stars = ""
        if q < 0.001: stars = "***"
        elif q < 0.01: stars = "**"
        elif q < 0.05: stars = "*"
        if stars:
            ha = "left" if v >= 0 else "right"
            ax.text(v + (0.02 * max_abs) * (1 if v >= 0 else -1),
                    i, stars, va="center", ha=ha,
                    fontsize=10, fontweight="bold", color=AERO["text"])
    style_axis(ax, xlabel=xlabel, ylabel="", title=title)


def plot_enrichment_dot(ax, enrichment_df: pd.DataFrame, *,
                        top_n: int = 25,
                        title: str = "Enrichment dot plot") -> None:
    """Single-column dot plot: dot size = k, color = -log10(q)."""
    if enrichment_df is None or enrichment_df.empty:
        ax.text(0.5, 0.5, "No enrichment results", ha="center", va="center",
                transform=ax.transAxes, color=AERO["muted"])
        return
    df = enrichment_df.head(top_n).iloc[::-1].copy()
    fc = np.log2(np.clip(df["fold_change"].values, 1e-6, 1e6))
    neglogq = -np.log10(np.clip(df["q_value"].values, 1e-300, 1.0))
    sizes = 40 + (df["k"].values / max(1, df["k"].max())) * 260

    sc = ax.scatter(fc, np.arange(len(df)),
                    s=sizes, c=neglogq, cmap=cmap_for("pvalue"),
                    edgecolors=AERO["text"], linewidths=0.4)
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["term"].tolist())
    ax.axvline(0, color=AERO["muted"], lw=0.8)
    cb = ax.figure.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("-log10(q)", fontsize=TYPOGRAPHY["annot"]["fontsize"])
    style_axis(ax, xlabel="log2(fold change)", ylabel="", title=title)


def plot_enrichment_volcano(ax, enrichment_df: pd.DataFrame, *,
                            title: str = "Enrichment volcano",
                            q_threshold: float = 0.05,
                            fc_threshold: float = 1.5,
                            annotate_top: int = 8) -> None:
    """Volcano of log2(fold_change) vs -log10(q)."""
    if enrichment_df is None or enrichment_df.empty:
        ax.text(0.5, 0.5, "No enrichment results", ha="center", va="center",
                transform=ax.transAxes, color=AERO["muted"])
        return
    df = enrichment_df.copy()
    x = np.log2(np.clip(df["fold_change"].values, 1e-6, 1e6))
    y = -np.log10(np.clip(df["q_value"].values, 1e-300, 1.0))
    sig_up   = (df["q_value"].values < q_threshold) & (df["fold_change"].values >= fc_threshold)
    sig_down = (df["q_value"].values < q_threshold) & (df["fold_change"].values <= 1.0 / fc_threshold)
    other    = ~(sig_up | sig_down)

    ax.scatter(x[other], y[other], s=18, c=AERO["muted"],
               alpha=0.45, edgecolors="none", label="ns")
    ax.scatter(x[sig_up], y[sig_up], s=26, c=AERO["green_dark"],
               alpha=0.85, edgecolors="none", label="enriched")
    ax.scatter(x[sig_down], y[sig_down], s=26, c=AERO["danger"],
               alpha=0.85, edgecolors="none", label="depleted")

    ax.axvline(np.log2(fc_threshold), color=AERO["muted"], ls=":", lw=0.7)
    ax.axvline(-np.log2(fc_threshold), color=AERO["muted"], ls=":", lw=0.7)
    ax.axhline(-np.log10(q_threshold), color=AERO["muted"], ls=":", lw=0.7)

    # annotate top significant terms
    sig_df = df.loc[sig_up | sig_down].head(annotate_top)
    for _, row in sig_df.iterrows():
        xv = np.log2(max(row["fold_change"], 1e-6))
        yv = -np.log10(max(row["q_value"], 1e-300))
        ax.annotate(str(row["term"]), (xv, yv),
                    fontsize=TYPOGRAPHY["annot"]["fontsize"],
                    xytext=(4, 4), textcoords="offset points",
                    color=AERO["text"])
    style_axis(ax, xlabel="log2(fold change)", ylabel="-log10(q)",
               title=title)
    ax.legend(**TYPOGRAPHY["legend"])


# =====================================================================
# High-level: multi-group label enrichment (heatmap)
# =====================================================================

def multi_group_enrichment(labels: pd.Series,
                           groups: Dict[str, pd.Series],
                           *,
                           min_count: int = 3,
                           alpha: float = 0.05) -> pd.DataFrame:
    """Run label_enrichment() once per group, return a long-form frame.

    ``groups``: mapping of group name -> boolean mask (foreground).
    """
    frames = []
    for name, mask in groups.items():
        df = label_enrichment(labels, mask, min_count=min_count, alpha=alpha)
        if df.empty:
            continue
        df.insert(0, "group", name)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_enrichment_heatmap(ax, long_df: pd.DataFrame, *,
                            metric: str = "log2_fc",
                            top_n_per_group: int = 8,
                            title: str = "Label enrichment (groups × terms)"
                            ) -> None:
    """Heatmap of groups × terms, colored by metric; stars = significance.

    Uses the long-form frame from ``multi_group_enrichment()``.
    """
    if long_df is None or long_df.empty:
        ax.text(0.5, 0.5, "No enrichment results", ha="center", va="center",
                transform=ax.transAxes, color=AERO["muted"])
        return

    df = long_df.copy()
    df["log2_fc"] = np.log2(np.clip(df["fold_change"].values, 1e-6, 1e6))
    df["neglog10_q"] = -np.log10(np.clip(df["q_value"].values, 1e-300, 1.0))

    # pick top-n terms per group by p-value, then union
    picks = (df.sort_values(["group", "p_value"])
               .groupby("group").head(top_n_per_group)["term"].unique())
    df = df[df["term"].isin(picks)]

    pivot = df.pivot_table(index="term", columns="group", values=metric, aggfunc="first")
    qpivot = df.pivot_table(index="term", columns="group", values="q_value", aggfunc="first")

    # order rows by overall strength
    pivot = pivot.loc[pivot.abs().max(axis=1).sort_values(ascending=False).index]
    qpivot = qpivot.loc[pivot.index, pivot.columns]

    vmax = float(np.nanmax(np.abs(pivot.values))) if pivot.size else 1.0
    vmax = vmax if vmax > 0 else 1.0
    im = ax.imshow(pivot.values, aspect="auto",
                   cmap=cmap_for("divergent"),
                   vmin=-vmax, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=30, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index.tolist())
    cb = ax.figure.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("log2(fold change)",
                 fontsize=TYPOGRAPHY["annot"]["fontsize"])

    # star significance on top of each cell
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            q = qpivot.values[i, j]
            if np.isnan(q):
                continue
            if q < 0.001:
                s = "***"
            elif q < 0.01:
                s = "**"
            elif q < 0.05:
                s = "*"
            else:
                s = ""
            if s:
                ax.text(j, i, s, ha="center", va="center",
                        color=AERO["text"], fontsize=9, fontweight="bold")
    style_axis(ax, xlabel="group", ylabel="term", title=title)
