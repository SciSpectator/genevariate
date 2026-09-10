"""The program's drawing routines, in one place and free of Tkinter.

A figure the assistant returns and a figure a window draws have to be the same
figure. They were not: the Explorer built its histogram inside a widget method
and the assistant's chart module built a second one beside it, and while both
took their bins, their KDE and their mode rule from the same helpers, the
drawing itself was written twice. Two routines that agree today are two
routines that can disagree tomorrow, and a reader who is shown a chart no
button produces cannot check it against anything.

So the drawing lives here, on a bare Matplotlib ``Axes``: a window passes the
axis it already owns and keeps its interactivity, an executor makes a figure
off the main thread and passes that. Nothing here imports Tkinter, which is
what lets the chatbot use it at all.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = ["describe_values", "draw_gene_histogram", "histogram_title"]

#: Region shading, kept here so the assistant and the windows shade alike.
_REGION_FILL = "#1E90E0"
_REGION_EDGE = "#0A5B9A"
_MODE_LINE = "#7D3C98"


def describe_values(values: Sequence[float],
                    dist_class: str = "") -> Dict[str, Any]:
    """The numbers a reader would take off a distribution chart."""
    v = np.asarray(list(values), dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    q1, med, q3 = (float(x) for x in np.percentile(v, [25, 50, 75]))
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(np.count_nonzero((v < lo) | (v > hi)))
    mean = float(np.mean(v))
    std = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
    try:
        from scipy.stats import skew as _skew
        skew = float(_skew(v)) if v.size > 2 else 0.0
    except Exception:
        skew = 0.0
    cls = (dist_class or "").strip()
    n_modes = 2 if cls == "Bimodal" else (3 if cls == "Multimodal" else 1)
    return {
        "n": int(v.size), "mean": mean, "median": med, "std": std,
        "cv": (std / mean) if mean else float("nan"),
        "iqr": iqr, "skew": skew, "n_modes": n_modes, "n_outliers": n_out,
        "min": float(v.min()), "max": float(v.max()), "class": cls,
    }


def histogram_title(head: str, dist_class: str, desc: Mapping[str, Any],
                    modes: Sequence[Tuple[float, float]]) -> str:
    """The Explorer's title line, so both routes name a chart the same way."""
    n = int(desc.get("n", 0) or 0)
    if len(modes) > 1:
        shape = "  ".join(f"{x:.1f}" for x, _ in modes)
        return f"{head} | {dist_class} | n={n:,} | modes at {shape}"
    return (f"{head} | {dist_class or '?'} | n={n:,} | "
            f"u={desc.get('mean', float('nan')):.2f} "
            f"SD={desc.get('std', float('nan')):.2f}")


def draw_gene_histogram(ax, values: Sequence[float], *,
                        gene: str = "",
                        label: str = "",
                        dist_class: str = "",
                        x_label: str = "Expression",
                        bounds: Optional[Tuple[float, float]] = None,
                        title: bool = True) -> Dict[str, Any]:
    """Draw one gene's distribution on *ax*, the way the Explorer draws it.

    Density-normalised histogram, the classifier's own KDE over it, a rule at
    each counted mode, and - when ``bounds`` is given - the region shaded
    against the whole distribution rather than plotted on its own.

    The y-axis is a single Density axis on purpose. An earlier design put the
    KDE on ``ax.twinx()``, and the twin rendered a stale ghost axis whenever
    the canvas resized; one axis cannot ghost.

    Returns the chart's description, with ``bins`` and ``patches`` for a caller
    that wants to make the bars interactive, and a ``region`` block when a
    region was shaded.
    """
    from genevariate.config import CONFIG
    from genevariate.core.analysis.bimodality import (
        BIMODAL_TAGS, density_modes, optimal_bins, robust_kde,
    )

    cfg = CONFIG['plotting']['histogram']
    v = np.asarray(list(values), dtype=float)
    v = v[np.isfinite(v)]
    desc = describe_values(v, dist_class)
    desc["bins"], desc["patches"], desc["modes"] = None, None, []
    if not v.size:
        return desc

    modes: list = []
    if dist_class in BIMODAL_TAGS:
        try:
            modes = density_modes(v)
        except Exception:
            modes = []

    counts, bins, patches = ax.hist(
        v, bins=optimal_bins(v, method='auto'), density=True,
        edgecolor=cfg['edge_color'], alpha=cfg['alpha'],
        color=cfg['default_color'], linewidth=0.5)
    y_max = max(counts) if len(counts) else 0

    if (v.size >= cfg['min_samples_for_kde']
            and np.var(v) > cfg['min_variance_for_kde']):
        try:
            # The classifier's own density, so the curve drawn and the shape
            # named in the title are the same object.
            x_range = np.linspace(v.min(), v.max(), 200)
            kde_vals = robust_kde(v)(x_range)
            ax.plot(x_range, kde_vals, 'r-', alpha=0.6, linewidth=2,
                    label='KDE')
            ax.legend(loc='upper right', fontsize=7, framealpha=0.7)
            y_max = max(y_max, kde_vals.max())
        except Exception:
            pass
    if y_max > 0:
        ax.set_ylim(0, y_max * 1.15)

    # A minor mode can be an order of magnitude below the major one, so it is
    # drawn as a full-height rule rather than scaled by its own density.
    for x_mode, _ in (modes if len(modes) > 1 else []):
        ax.axvline(x_mode, color=_MODE_LINE, linestyle='--',
                   linewidth=1.2, alpha=0.85, zorder=3)

    if bounds is not None:
        lo, hi = float(bounds[0]), float(bounds[1])
        n_in = int(((v >= lo) & (v <= hi)).sum())
        ax.axvspan(lo, hi, color=_REGION_FILL, alpha=0.18, zorder=0,
                   label=f"region {lo:.3g}-{hi:.3g} (n={n_in:,})")
        ax.axvline(lo, color=_REGION_EDGE, lw=1.4, zorder=2)
        ax.legend(loc='upper right', fontsize=7, framealpha=0.7)
        desc["region"] = {"low": lo, "high": hi, "n_in": n_in,
                          "pct_in": round(100.0 * n_in / max(1, v.size), 2)}

    if title:
        head = f"{label} - {gene}" if label else gene
        ax.set_title(histogram_title(head, dist_class, desc, modes),
                     fontsize=10, weight='bold', pad=15)
    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2, linestyle='--')

    desc["bins"], desc["patches"], desc["modes"] = bins, patches, modes
    return desc


# ── Region enrichment ──────────────────────────────────────────────────────
#: Significance -> bar colour, shared so the window and the assistant grade
#: a hit the same way.
_SIG_COLOR = {"***": "#C62828", "**": "#E53935", "*": "#EF9A9A"}
_SIG_NS = "#BDBDBD"
_SEL_COLOR = "#C62828"
_BG_COLOR = "#78909C"
#: Fold values are capped for display only; the annotation prints the real one.


def _trunc(s, m=28):
    s = "".join(ch if ord(ch) >= 32 else " " for ch in str(s)).strip()
    return (s[:m - 1] + "..") if len(s) > m else s


def enrichment_rows(table) -> list:
    """``region_label_enrichment`` output as the row dicts the plot reads.

    ``n_gse`` arrives as a float because most rows have none; it is handed
    back as an int or nothing rather than "12.0" or "nan".
    """
    out = []
    for rec in table.to_dict("records"):
        n_gse = rec.get("n_gse")
        n_gse = (None if n_gse is None or not np.isfinite(n_gse)
                 else int(n_gse))
        out.append({
            "Region": rec["region"], "Label Column": rec["label_column"],
            "Value": rec["value"],
            "a": rec["a"], "n_sel": rec["n_region"],
            "c": rec["c"], "n_non": rec["n_background"],
            "Sel%": rec["region_pct"], "BG%": rec["background_pct"],
            "Enrichment": rec["enrichment"], "p-value": rec["p_value"],
            "padj": rec["q_value"], "Sig": rec["significance"],
            "n_gse": n_gse, "rho": rec.get("rho"), "n_eff": rec.get("n_eff"),
            "ci_low": rec.get("ci_low"), "ci_high": rec.get("ci_high"),
        })
    return out


def draw_enrichment(ax_freq, ax_fold, rows: Sequence[Mapping[str, Any]], *,
                    label_column: str) -> Dict[str, Any]:
    """Draw one region x label-column enrichment onto two axes.

    ``ax_freq`` gets the paired frequency bars, ``ax_fold`` the enrichment
    ratio with the study-bootstrap interval as a whisker. Returns the numbers
    a reader would take off the chart, so a caller that cannot see it can
    still say what it shows.
    """
    rows = list(rows)
    if not rows:
        return {"label_column": label_column, "n": 0, "top": []}
    n_sel = rows[0]["n_sel"]
    n_non = rows[0]["n_non"]

    labels = [_trunc(r["Value"]) for r in rows]
    sel_pcts = [r["Sel%"] for r in rows]
    bg_pcts = [r["BG%"] for r in rows]
    sigs = [r["Sig"] for r in rows]
    y_pos = np.arange(len(rows))
    bar_h = 0.35

    ax_freq.barh(y_pos - bar_h / 2, sel_pcts, bar_h,
                 label=f"Selected Region (n={n_sel})",
                 color=_SEL_COLOR, edgecolor="black", lw=0.4, alpha=0.85)
    ax_freq.barh(y_pos + bar_h / 2, bg_pcts, bar_h,
                 label=f"Rest of Loaded Samples (n={n_non:,})",
                 color=_BG_COLOR, edgecolor="black", lw=0.4, alpha=0.65)
    for i, r in enumerate(rows):
        ax_freq.text(max(sel_pcts[i] + 0.5, 1), y_pos[i] - bar_h / 2,
                     f" {r['a']}/{n_sel}  {r['Sig']}", va="center", fontsize=7,
                     fontweight="bold" if r["Sig"] != "ns" else "normal",
                     color=_SEL_COLOR if r["Sig"] != "ns" else "#999")
        ax_freq.text(max(bg_pcts[i] + 0.5, 1), y_pos[i] + bar_h / 2,
                     f" {r['c']}/{n_non:,}", va="center", fontsize=6.5,
                     color="#546E7A")
    # The annotations are drawn past the end of each bar in data coordinates,
    # so the axis needs room for them or the chart is cut off.
    ax_freq.set_xlim(0, max(sel_pcts + bg_pcts + [1.0]) * 1.38)
    ax_freq.set_yticks(y_pos)
    ax_freq.set_yticklabels(labels, fontsize=7.5)
    ax_freq.set_xlabel("Frequency (%)", fontsize=9)
    ax_freq.set_title(f"Selected vs Rest - {label_column}", fontsize=10,
                      weight="bold")
    ax_freq.legend(fontsize=9, loc="lower right", framealpha=0.9)
    ax_freq.invert_yaxis()
    ax_freq.grid(axis="x", alpha=0.2)

    # Every bar is drawn at the fold the test returned. A ratio of several
    # hundred beside a ratio of two is unreadable on a linear axis, so the axis
    # is logarithmic; shortening the bar instead would put a number on the
    # chart that the table contradicts.
    finite = [r["Enrichment"] for r in rows
              if np.isfinite(r["Enrichment"]) and r["Enrichment"] > 0]
    top = max(finite) if finite else 1.0
    # An undefined ratio - the value occurs in the region and nowhere outside -
    # has no length. It is drawn to the end of the axis and named there.
    inf_at = top * 2.2
    fold = [r["Enrichment"] if np.isfinite(r["Enrichment"]) and r["Enrichment"] > 0
            else inf_at for r in rows]
    colors = [_SIG_COLOR.get(s, _SIG_NS) for s in sigs]
    # The dashed null sits at 1.0, so the axis starts below it whatever the
    # bars do: an axis that excludes the null gives a bar nothing to be
    # read against and it fills the panel.
    base = min([f for f in fold if f > 0] + [1.0]) * 0.5
    base = min(base, 0.5)
    ax_fold.barh(y_pos, [f - base for f in fold], 0.55, left=base,
                 color=colors, edgecolor="black", lw=0.4, alpha=0.85)
    ax_fold.set_xscale("log")
    ax_fold.axvline(1.0, color="black", ls="--", lw=1, alpha=0.5)
    for i, r in enumerate(rows):
        q = r.get("padj", float("nan"))
        q_str = (f"q={q:.1e}" if np.isfinite(q) and q < 0.01
                 else f"q={q:.3f}" if np.isfinite(q) else "q=n/a")
        gse = r.get("n_gse")
        gse_str = "" if gse is None else f"  [{gse} GSE]"
        shown = (f"{r['Enrichment']:.1f}x" if np.isfinite(r["Enrichment"])
                 else "infinite")
        ax_fold.text(fold[i] * 1.06, y_pos[i],
                     f" {shown}  {q_str}{gse_str}",
                     va="center", fontsize=6.5,
                     fontweight="bold" if r["Sig"] != "ns" else "normal",
                     color=colors[i])
        lo, hi = r.get("ci_low", float("nan")), r.get("ci_high", float("nan"))
        if np.isfinite(lo) and np.isfinite(hi) and lo > 0:
            ax_fold.plot([lo, hi], [y_pos[i], y_pos[i]], color="#37474F",
                         lw=1.1, alpha=0.8, zorder=4, solid_capstyle="butt")
    hi_ci = [r.get("ci_high", 0.0) for r in rows
             if np.isfinite(r.get("ci_high", float("nan")))]
    ax_fold.set_xlim(base, max(fold + hi_ci + [1.0]) * 6.0)
    ax_fold.set_yticks(y_pos)
    ax_fold.set_yticklabels(["" for _ in rows])
    ax_fold.set_xlabel(
        "Enrichment ratio (fold, log axis; whiskers = 95% CI by study)",
        fontsize=9)
    ax_fold.set_title(f"Enrichment - {label_column}", fontsize=10, weight="bold")
    ax_fold.invert_yaxis()
    ax_fold.grid(axis="x", alpha=0.2)
    try:
        from matplotlib.patches import Patch
        ax_fold.legend(handles=[
            Patch(facecolor=_SIG_COLOR["***"], label="q<0.001 ***"),
            Patch(facecolor=_SIG_COLOR["**"], label="q<0.01 **"),
            Patch(facecolor=_SIG_COLOR["*"], label="q<0.05 *"),
            Patch(facecolor=_SIG_NS, label="ns"),
        ], fontsize=6, loc="lower right", framealpha=0.9)
    except Exception:
        pass

    single_study = sum(1 for r in rows
                       if r.get("n_gse") is not None and r["n_gse"] < 3)
    return {
        "label_column": label_column, "n": len(rows),
        "n_selected": n_sel, "n_background": n_non,
        "single_study": single_study,
        "top": [{"value": str(r["Value"]), "fold": round(float(r["Enrichment"]), 2),
                 "q": float(r["padj"]), "n_gse": r.get("n_gse"),
                 "ci": [r.get("ci_low"), r.get("ci_high")]}
                for r in rows[:10]],
    }


# ── Pooled effect forest ───────────────────────────────────────────
def draw_pooled_forest(ax, rows, *, gene: str, label_column: str,
                       palette: Optional[Mapping[str, str]] = None,
                       ) -> Dict[str, Any]:
    """Pooled log odds ratio per label value, with each platform marked.

    The single platforms are drawn as ticks beside the pooled point, so the
    pooled estimate is never the only thing on the plot: a value two platforms
    disagree about looks different from one they agree about, whatever the
    combined interval says.
    """
    p = dict(palette or {})
    c_conc = p.get("green_dark", "#2E7D32")
    c_sig = p.get("accent", "#1E90E0")
    c_muted = p.get("muted", "#5F7D95")

    rows = [r for r in rows if r.get("k", 0) >= 2][:20]
    if not rows:
        return {"gene": gene, "label_column": label_column, "n": 0, "top": []}
    ys = np.arange(len(rows))[::-1]
    for y, r in zip(ys, rows):
        lo = np.log(max(r["ci_low"], 1e-6))
        hi = np.log(max(r["ci_high"], 1e-6))
        sig = np.isfinite(r["q"]) and r["q"] < 0.05
        col = (c_conc if sig and r.get("concordant") else
               c_sig if sig else c_muted)
        ax.plot([lo, hi], [y, y], color=col, lw=1.6, zorder=2)
        ax.plot([r["pooled_log_or"]], [y], "o", color=col, ms=5, zorder=3)
        for pv in (r.get("per_platform") or {}).values():
            ax.plot([pv["log_or"]], [y], "|", color="#5A6B7A", ms=7, mew=1.0,
                    zorder=4)
    ax.axvline(0.0, color="#888", lw=1.0, ls="--", zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([_trunc(r["value"], 26) for r in rows], fontsize=8)
    ax.set_xlabel("log odds ratio, region vs rest of its own platform",
                  fontsize=9)
    ax.set_title(f"{gene} - {label_column}  (bars 95% CI; "
                 f"ticks = single platforms)", fontsize=9)

    n_sig = sum(1 for r in rows if np.isfinite(r["q"]) and r["q"] < 0.05)
    return {
        "gene": gene, "label_column": label_column, "n": len(rows),
        "n_sig": n_sig,
        "n_concordant": sum(1 for r in rows if r.get("concordant")),
        "top": [{"value": str(r["value"]), "k": r.get("k"),
                 "pooled_or": r.get("pooled_or"), "q": r.get("q"),
                 "concordant": r.get("concordant")} for r in rows[:10]],
    }
