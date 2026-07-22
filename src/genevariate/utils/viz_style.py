"""
GeneVariate — Unified plotting stylesheet (Frutiger Aero).

Central source of truth for matplotlib/seaborn appearance across every
plot window.  Call ``apply_genevariate_style()`` once on module load
(see ``gui/app.py``), and use the helpers below per-axes.

Design goals
------------
* Consistent type scale (title 13pt, labels 11pt, ticks 9pt, legend 9pt)
* Colorblind-safe discrete palettes (tab10 → tab20 → husl), explicit
  diverging/sequential cmaps for logFC / intensity / p-value
* Sky-blue Frutiger Aero plot chrome to visually tie plots to the GUI
* Safe caps on figure size so no plot blows past the screen
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib as _mpl
import matplotlib.colors as _mcolors
import matplotlib.pyplot as _plt
import numpy as _np
import seaborn as _sns


# ─────────────────────────────────────────────────────────────────────
# AERO palette  (mirrors gui/app.py AERO dict — keep these in sync)
# ─────────────────────────────────────────────────────────────────────
AERO = {
    "bg":           "#FFFFFF",
    "bg_top":       "#EAF6FE",
    "panel":        "#FFFFFF",
    "panel_top":    "#FCFEFF",
    "panel_bot":    "#EDF7FF",
    "border":       "#C5DAEA",
    "border_soft":  "#E0EEF7",
    "text":         "#0E2A45",
    "muted":        "#5F7D95",
    "accent":       "#1E90E0",
    "accent_dark":  "#0A5B9A",
    "accent_light": "#B9E3FA",
    "green":        "#4CAF50",
    "green_dark":   "#2E7D32",
    "green_light":  "#C9EFC7",
    "danger":       "#C0392B",
    "warn":         "#E67E22",
    "plot_bg":      "#FBFDFF",   # very subtle sky tint
    "spine":        "#6EA4C8",
    "grid":         "#CFE0EE",
}

# ─────────────────────────────────────────────────────────────────────
# Typography
# ─────────────────────────────────────────────────────────────────────
TYPOGRAPHY = {
    "title":       {"fontsize": 13, "fontweight": "bold"},
    "subtitle":    {"fontsize": 11, "fontweight": "bold"},
    "axis_label":  {"fontsize": 11, "fontweight": "bold"},
    "tick":        {"labelsize": 9},
    "legend":      {"fontsize": 9, "framealpha": 0.92},
    "legend_dense": {"fontsize": 8, "framealpha": 0.92, "ncol": 2},
    "annot":       {"fontsize": 9},
    "stats":       {"fontsize": 9, "family": "monospace"},
}

# ─────────────────────────────────────────────────────────────────────
# DPI / size caps
# ─────────────────────────────────────────────────────────────────────
SCREEN_DPI = 100
EXPORT_DPI = 300
MAX_FIG_W = 16.0
MAX_FIG_H = 10.0


# ─────────────────────────────────────────────────────────────────────
# One-shot stylesheet
# ─────────────────────────────────────────────────────────────────────
_APPLIED = False


def apply_genevariate_style() -> None:
    """Install the global matplotlib rcParams. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True

    rc = {
        # Figure
        "figure.dpi":          SCREEN_DPI,
        "figure.facecolor":    AERO["bg"],
        "figure.edgecolor":    AERO["bg"],
        "figure.autolayout":   False,
        "savefig.dpi":         EXPORT_DPI,
        "savefig.bbox":        "tight",
        "savefig.facecolor":   AERO["bg"],
        "savefig.edgecolor":   AERO["bg"],
        "figure.max_open_warning": 50,

        # Axes
        "axes.facecolor":      AERO["plot_bg"],
        "axes.edgecolor":      AERO["spine"],
        "axes.linewidth":      0.9,
        "axes.labelcolor":     AERO["text"],
        "axes.labelsize":      TYPOGRAPHY["axis_label"]["fontsize"],
        "axes.labelweight":    "bold",
        "axes.titlesize":      TYPOGRAPHY["title"]["fontsize"],
        "axes.titleweight":    "bold",
        "axes.titlepad":       8,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.grid":           True,
        "axes.grid.axis":      "both",
        "axes.grid.which":     "major",
        "axes.axisbelow":      True,

        # Grid
        "grid.color":          AERO["grid"],
        "grid.alpha":          0.45,
        "grid.linestyle":      "--",
        "grid.linewidth":      0.5,

        # Ticks
        "xtick.color":         AERO["text"],
        "ytick.color":         AERO["text"],
        "xtick.labelsize":     TYPOGRAPHY["tick"]["labelsize"],
        "ytick.labelsize":     TYPOGRAPHY["tick"]["labelsize"],
        "xtick.direction":     "out",
        "ytick.direction":     "out",
        "xtick.major.size":    4,
        "ytick.major.size":    4,
        "xtick.major.width":   0.8,
        "ytick.major.width":   0.8,
        "xtick.minor.size":    2,
        "ytick.minor.size":    2,
        "xtick.minor.width":   0.5,
        "ytick.minor.width":   0.5,

        # Legend
        "legend.fontsize":     TYPOGRAPHY["legend"]["fontsize"],
        "legend.framealpha":   TYPOGRAPHY["legend"]["framealpha"],
        "legend.edgecolor":    AERO["border"],
        "legend.facecolor":    "#FFFFFF",
        "legend.fancybox":     True,
        "legend.borderpad":    0.45,

        # Lines / markers
        "lines.linewidth":     2.0,
        "lines.markersize":    5,
        "patch.linewidth":     0.8,
        "patch.edgecolor":     AERO["text"],

        # Fonts — prefer the same family the GUI uses
        "font.family":         ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size":           10,

        # Image / cmap defaults
        "image.cmap":          "viridis",
    }
    _mpl.rcParams.update(rc)

    # Seaborn: align with rcParams (don't let it override our grid/spine)
    try:
        _sns.set_theme(style="whitegrid", rc=rc)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────
# Palettes
# ─────────────────────────────────────────────────────────────────────
_USE_CASES = {"discrete", "logfc", "intensity", "pvalue", "divergent", "aero"}


def palette_for(n: int, use_case: str = "discrete") -> List[str]:
    """Return ``n`` hex colors appropriate for ``use_case``.

    use_case values:
        * discrete  — categorical groups (tab10 / tab20 / husl)
        * aero      — sky/green GeneVariate accent palette
        * logfc     — ignored for discrete (use cmap_for instead)
    """
    if use_case == "aero":
        # Blue/green Frutiger-Aero series colors — never orange (warn is a
        # UI-only warning hue, not a data-series color). Teal fills the slot.
        base = [AERO["accent"], AERO["green"], AERO["accent_dark"],
                AERO["green_dark"], "#00838F", AERO["danger"],
                AERO["accent_light"], AERO["green_light"]]
        if n <= len(base):
            return base[:n]
        # extend with tab20 if we need more
        extra = _sns.color_palette("tab20", n - len(base))
        return base + [_mcolors.to_hex(c) for c in extra]

    # discrete (default)
    if n <= 10:
        pal = _sns.color_palette("tab10", n)
    elif n <= 20:
        pal = _sns.color_palette("tab20", n)
    else:
        pal = _sns.color_palette("husl", n)
    return [_mcolors.to_hex(c) for c in pal]


def cmap_for(kind: str = "intensity"):
    """Return a matplotlib colormap name for a given data kind.

    kind:
        * intensity  — sequential low→high (viridis)
        * logfc      — diverging around 0 (RdBu_r)
        * divergent  — alias for logfc
        * pvalue / qvalue — sequential (YlGnBu, high = significant)
        * correlation    — diverging (coolwarm)
    """
    mapping = {
        "intensity":   "viridis",
        "expression":  "viridis",
        "logfc":       "RdBu_r",
        "divergent":   "RdBu_r",
        "pvalue":      "YlGnBu",
        "qvalue":      "YlGnBu",
        "correlation": "coolwarm",
        "heat":        "magma",
    }
    return mapping.get(kind, "viridis")


# ─────────────────────────────────────────────────────────────────────
# Per-axes helpers
# ─────────────────────────────────────────────────────────────────────
def apply_aero_background(ax) -> None:
    """Apply soft sky-tinted facecolor + sky-blue spines to an axes."""
    ax.set_facecolor(AERO["plot_bg"])
    for sp in ("bottom", "left"):
        ax.spines[sp].set_color(AERO["spine"])
        ax.spines[sp].set_linewidth(1.0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=AERO["text"], which="major")


def apply_plot_polish(ax, *, grid: bool = True, minor: bool = False,
                      log: Optional[str] = None) -> None:
    """Final polish: grid, spines, ticks, optional log scale.

    log: ``'x'``, ``'y'``, ``'xy'`` or ``None``.
    """
    if grid:
        ax.grid(True, alpha=0.45, linestyle="--", linewidth=0.5,
                which="major", color=AERO["grid"])
        if minor:
            ax.minorticks_on()
            ax.grid(True, alpha=0.20, linestyle=":", linewidth=0.4,
                    which="minor", color=AERO["grid"])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("bottom", "left"):
        ax.spines[sp].set_linewidth(0.9)
        ax.spines[sp].set_color(AERO["spine"])
    if log:
        if "x" in log:
            ax.set_xscale("log")
        if "y" in log:
            ax.set_yscale("log")


def style_axis(ax, xlabel: Optional[str] = None,
               ylabel: Optional[str] = None,
               title: Optional[str] = None,
               grid: bool = True) -> None:
    """Apply labels + unified typography + polish in one call."""
    if xlabel:
        ax.set_xlabel(xlabel, **TYPOGRAPHY["axis_label"])
    if ylabel:
        ax.set_ylabel(ylabel, **TYPOGRAPHY["axis_label"])
    if title:
        ax.set_title(title, **TYPOGRAPHY["title"], pad=8)
    ax.tick_params(**TYPOGRAPHY["tick"])
    apply_plot_polish(ax, grid=grid)


def legend_outside(ax, *, dense: bool = False, title: Optional[str] = None):
    """Place a legend outside the right edge (caller already made space)."""
    kw = dict(TYPOGRAPHY["legend_dense"] if dense else TYPOGRAPHY["legend"])
    kw.update(dict(loc="upper left", bbox_to_anchor=(1.02, 1.0)))
    if title:
        kw["title"] = title
    return ax.legend(**kw)


# ─────────────────────────────────────────────────────────────────────
# Figure sizing
# ─────────────────────────────────────────────────────────────────────
def smart_figsize(kind: str = "default", n_plots: int = 1,
                  n_rows: int = 1) -> Tuple[float, float]:
    """Return a sensible (w, h) figsize, never exceeding the screen caps."""
    if kind == "heatmap":
        w = min(MAX_FIG_W, max(6, 1.0 + n_plots * 1.2))
        h = min(MAX_FIG_H, max(4, 0.8 + n_plots * 0.8))
    elif kind == "scatter":
        w, h = 10.0, 7.0
    elif kind == "histogram":
        w = min(MAX_FIG_W, 3 + n_plots * 4)
        h = 6.0
    elif kind == "side_by_side":
        w = min(MAX_FIG_W, 1.5 + n_plots * 4.5)
        h = 6.0
    elif kind == "grid":
        w = min(MAX_FIG_W, 4 + n_plots * 3.5)
        h = min(MAX_FIG_H, 3 + n_rows * 3.5)
    else:
        w, h = 10.0, 6.0
    return (min(w, MAX_FIG_W), min(h, MAX_FIG_H))


def cap_figsize(w: float, h: float) -> Tuple[float, float]:
    """Cap a raw (w, h) to the screen limits."""
    return (min(float(w), MAX_FIG_W), min(float(h), MAX_FIG_H))


# ─────────────────────────────────────────────────────────────────────
# Hover tooltips (optional — silently no-op if mplcursors unavailable)
# ─────────────────────────────────────────────────────────────────────
def enable_hover(artists=None, fig=None, formatter=None):
    """Attach mplcursors hover tooltips to scatter/line artists.

    Silently does nothing if ``mplcursors`` isn't installed — always safe
    to call from any plot.
    """
    try:
        import mplcursors
    except Exception:
        return None

    try:
        if artists is None and fig is not None:
            # default: hover on every Axes in the figure
            target = fig.axes
        else:
            target = artists
        cursor = mplcursors.cursor(target, hover=True)
        if formatter is not None:
            @cursor.connect("add")
            def _on_add(sel):
                try:
                    sel.annotation.set_text(formatter(sel))
                except Exception:
                    pass
        return cursor
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────
# Back-compat shim: let the old Plotter API keep working
# ─────────────────────────────────────────────────────────────────────
def distinct_colors(n: int) -> List[str]:
    """Thin alias — use palette_for() in new code."""
    return palette_for(n, "discrete")
