"""
GeneVariate - Extra plot types for gene expression & label analyses.

All helpers assume ``apply_genevariate_style()`` has been called, so they
keep per-axes tweaks to a minimum and rely on the global rcParams.

Plot catalogue
--------------
* plot_ma            - MA (log ratio vs log mean)
* plot_bland_altman  - agreement / bias between two measurements
* plot_ecdf          - empirical cumulative distribution
* plot_hexbin        - density-safe scatter for large samples
* plot_qq            - quantile-quantile against normal
* plot_enrichment_bar   - enrichment fold-change with significance bars
* plot_enrichment_dot   - dot plot (gene set × group)  - standard enrichment view
* plot_enrichment_volcano - log2(OR) vs -log10(q)

Enrichment utilities
--------------------
* enrichment_fisher(a, b, c, d)  - 2x2 Fisher exact + OR
* label_enrichment(labels, group_mask, background_mask=None)  - per-value
  enrichment test (with Bonferroni + Benjamini-Hochberg)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from .viz_style import (
    AERO, apply_plot_polish, apply_aero_background, attach_point_labels,
    cmap_for, grid_matrix_cells, legend_outside, palette_for, style_axis,
    TYPOGRAPHY,
)


# =====================================================================
# Basic statistical plots
# =====================================================================

def plot_ma(ax, sample: np.ndarray, reference: np.ndarray,
            title: str = "MA plot",
            highlight_mask: Optional[np.ndarray] = None) -> None:
    """MA plot - M = log2(sample/reference), A = 0.5*log2(sample*reference).

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
    group_mask      boolean series - foreground (e.g. samples in a gene region)
    background_mask optional boolean - restrict universe; defaults to all non-NA
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
        # fold change = (k/n) / (K/N)  - ratio of observed to expected fraction
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

def _ratio_log2(values) -> np.ndarray:
    """log2 of a ratio, keeping 0 and inf as ∓inf instead of clipping them.

    A term absent from one group has fold change 0 (or an infinite odds ratio).
    Clipping that to a constant floor such as 1e-6 turns every one of them into
    the same arbitrary -19.9, which then plots as a wall of identical maximal
    bars. Keep the infinity here; :func:`_place_unbounded` decides where to draw
    it, from the data.
    """
    v = np.asarray(values, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(v > 0, np.log2(np.where(v > 0, v, 1.0)), -np.inf)
    return np.where(np.isnan(v), np.nan, out)


def _place_unbounded(vals: np.ndarray):
    """Drawable positions for *vals* plus the mask of the unbounded ones.

    Infinite values are placed just beyond the largest finite magnitude in the
    same plot - a margin taken from the data, never a fixed constant - so the
    reader sees they are off-scale rather than seeing them silently dominate
    the axis. Returns ``(positions, unbounded_mask)``.
    """
    vals = np.asarray(vals, dtype=float)
    finite = np.isfinite(vals)
    span = float(np.max(np.abs(vals[finite]))) if finite.any() else 1.0
    edge = (span if span > 0 else 1.0) * 1.25
    pos = np.where(np.isposinf(vals), edge,
                   np.where(np.isneginf(vals), -edge, vals))
    return pos, np.isinf(vals)


def _term_fontsize(n: int) -> float:
    """Tick-label size that keeps *n* term names legible and non-overlapping."""
    base = float(TYPOGRAPHY["tick"]["labelsize"])
    if n <= 25:
        return base
    return max(4.5, base * (25.0 / n) ** 0.5)


def plot_enrichment_bar(ax, enrichment_df: pd.DataFrame, *,
                        top_n: int | None = 20,
                        metric: str = "fold_change",
                        title: str = "Label enrichment") -> None:
    """Horizontal bar chart of the enrichment terms, ranked as given.

    *top_n* limits how many terms are drawn; ``None`` draws every one of them.
    Bar length = metric, colour = direction, asterisks = significance
    (q<0.05 / 0.01 / 0.001). Terms with a count of zero in one group are
    unbounded on a log scale and are drawn hatched, past the last finite bar.
    """
    if enrichment_df is None or enrichment_df.empty:
        ax.text(0.5, 0.5, "No enrichment results", ha="center", va="center",
                transform=ax.transAxes, color=AERO["muted"])
        return

    df = enrichment_df if top_n is None else enrichment_df.head(top_n)
    df = df.iloc[::-1].copy()  # reverse for top-at-top
    if metric == "odds_ratio":
        vals = _ratio_log2(df["odds_ratio"].values)
        xlabel = "log2(odds ratio)"
    elif metric == "minus_log10_p":
        vals = -np.log10(np.clip(df["p_value"].values, 1e-300, 1.0))
        xlabel = "-log10(p)"
    else:  # fold_change
        vals = _ratio_log2(df["fold_change"].values)
        xlabel = "log2(fold change)"

    vals, unbounded = _place_unbounded(vals)

    # color by sign (up=green, down=danger)
    max_abs = float(np.max(np.abs(vals[np.isfinite(vals)]))) if len(vals) else 1.0
    max_abs = max_abs if max_abs > 0 else 1.0
    colors = [AERO["green_dark"] if v >= 0 else AERO["danger"] for v in vals]

    y_pos = np.arange(len(df))
    bars = ax.barh(y_pos, vals, color=colors, edgecolor=AERO["text"],
                   lw=0.4, alpha=0.85)
    for bar, off in zip(bars, unbounded):
        if off:
            bar.set_hatch("//")
            bar.set_alpha(0.45)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["term"].tolist(),
                       fontsize=_term_fontsize(len(df)))
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
    if unbounded.any():
        ax.text(0.99, -0.09,
                "hatched = count 0 in one group (off-scale, not a measured size)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=TYPOGRAPHY["annot"]["fontsize"], color=AERO["muted"])
    style_axis(ax, xlabel=xlabel, ylabel="", title=title)


def plot_enrichment_dot(ax, enrichment_df: pd.DataFrame, *,
                        top_n: int | None = 25,
                        title: str = "Enrichment dot plot") -> None:
    """Single-column dot plot: dot size = k, color = -log10(q).

    *top_n* limits how many terms are drawn; ``None`` draws every one.
    """
    if enrichment_df is None or enrichment_df.empty:
        ax.text(0.5, 0.5, "No enrichment results", ha="center", va="center",
                transform=ax.transAxes, color=AERO["muted"])
        return
    df = enrichment_df if top_n is None else enrichment_df.head(top_n)
    df = df.iloc[::-1].copy()
    fc, fc_unbounded = _place_unbounded(_ratio_log2(df["fold_change"].values))
    neglogq = -np.log10(np.clip(df["q_value"].values, 1e-300, 1.0))
    sizes = 40 + (df["k"].values / max(1, df["k"].max())) * 260

    sc = ax.scatter(fc, np.arange(len(df)),
                    s=sizes, c=neglogq, cmap=cmap_for("pvalue"),
                    edgecolors=AERO["text"], linewidths=0.4)
    if fc_unbounded.any():
        ax.scatter(fc[fc_unbounded], np.arange(len(df))[fc_unbounded],
                   s=sizes[fc_unbounded] * 0.6, facecolors="none",
                   edgecolors=AERO["text"], linewidths=1.2, zorder=3)
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["term"].tolist(),
                       fontsize=_term_fontsize(len(df)))
    ax.axvline(0, color=AERO["muted"], lw=0.8)
    cb = ax.figure.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("-log10(q)", fontsize=TYPOGRAPHY["annot"]["fontsize"])
    if fc_unbounded.any():
        # Footnote rather than a legend box: with the terms ranked down the
        # y-axis the box lands inside the plotting area, on top of the top
        # term's row. Same wording and placement as plot_enrichment_bar.
        ax.text(0.99, -0.09,
                "ringed = count 0 in one group (off-scale, not a measured size)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=TYPOGRAPHY["annot"]["fontsize"], color=AERO["muted"])
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
    x, x_unbounded = _place_unbounded(_ratio_log2(df["fold_change"].values))
    y = -np.log10(np.clip(df["q_value"].values, 1e-300, 1.0))
    sig_up   = (df["q_value"].values < q_threshold) & (df["fold_change"].values >= fc_threshold)
    sig_down = (df["q_value"].values < q_threshold) & (df["fold_change"].values <= 1.0 / fc_threshold)
    other    = ~(sig_up | sig_down)

    terms = df["term"].astype(str).values
    for mask, size, colour, name in (
            (other,    18, AERO["muted"],      "ns"),
            (sig_up,   26, AERO["green_dark"], "enriched"),
            (sig_down, 26, AERO["danger"],     "depleted")):
        sc = ax.scatter(x[mask], y[mask], s=size, c=colour,
                        alpha=0.45 if name == "ns" else 0.85,
                        edgecolors="none", label=name)
        # Each scatter holds only its own subset, so the terms have to be
        # masked the same way or hovering would report the wrong one.
        attach_point_labels(sc, terms[mask])

    if x_unbounded.any():
        ax.scatter(x[x_unbounded], y[x_unbounded], s=46, facecolors="none",
                   edgecolors=AERO["text"], linewidths=1.0, zorder=3,
                   label="count 0 in one group (off-scale)")

    ax.axvline(np.log2(fc_threshold), color=AERO["muted"], ls=":", lw=0.7)
    ax.axvline(-np.log2(fc_threshold), color=AERO["muted"], ls=":", lw=0.7)
    ax.axhline(-np.log10(q_threshold), color=AERO["muted"], ls=":", lw=0.7)

    # annotate top significant terms
    sig_idx = np.flatnonzero(sig_up | sig_down)[:annotate_top]
    for i in sig_idx:
        row = df.iloc[i]
        xv, yv = x[i], y[i]
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


