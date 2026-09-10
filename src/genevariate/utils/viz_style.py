"""
GeneVariate - Unified plotting stylesheet (Frutiger Aero).

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
import matplotlib.collections as _mcoll
import matplotlib.colors as _mcolors
import matplotlib.pyplot as _plt
import numpy as _np
import seaborn as _sns


# ─────────────────────────────────────────────────────────────────────
# AERO palette  (mirrors gui/app.py AERO dict - keep these in sync)
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


def _available_fonts(preferred: Sequence[str]) -> List[str]:
    """Keep only the font families this machine can actually render."""
    try:
        from matplotlib import font_manager
        installed = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        return ["sans-serif"]
    keep = [f for f in preferred if f in installed or f.endswith("-serif")]
    return keep or ["sans-serif"]


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

        # Fonts - prefer the same family the GUI uses, but only name families
        # that are actually installed. Listing Segoe UI unconditionally made
        # matplotlib emit a "font family not found" warning for every single
        # text object drawn on Linux, which buried real warnings in the log.
        "font.family":         _available_fonts(
            ["Segoe UI", "Selawik", "DejaVu Sans", "sans-serif"]),
        "font.size":           10,

        # Image / cmap defaults
        "image.cmap":          "viridis",

        # Redraw cost. Pan and zoom cannot be blitted - the data transform
        # itself changes - so every frame of a drag re-rasterizes each curve,
        # and the app routinely draws 20 KDE curves of 200+ vertices over a
        # background histogram. Simplification drops vertices that land on
        # the same pixel (invisible at screen dpi, and turned off again by
        # savefig's higher dpi), and chunking lets Agg render a long path in
        # pieces instead of building one enormous one.
        "path.simplify":           True,
        "path.simplify_threshold": 1.0,
        "agg.path.chunksize":      10_000,
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
        * discrete  - categorical groups (tab10 / tab20 / husl)
        * aero      - sky/green GeneVariate accent palette
        * logfc     - ignored for discrete (use cmap_for instead)
    """
    if use_case == "aero":
        # Blue/green Frutiger-Aero series colors - never orange (warn is a
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
        * intensity  - sequential low→high (viridis)
        * logfc      - diverging around 0 (RdBu_r)
        * divergent  - alias for logfc
        * pvalue / qvalue - sequential (YlGnBu, high = significant)
        * correlation    - diverging (coolwarm)
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


def grid_matrix_cells(ax, nrows: int, ncols: int, *,
                      color: str = "black", lw: float = 0.6) -> None:
    """Draw black separators between the cells of an ``imshow`` matrix.

    A heatmap without cell borders reads as a continuous field, so neighbouring
    cells of similar colour blur into one block and it stops being obvious which
    row a value belongs to. ``imshow`` centres cell *i* on *i*, which puts the
    boundaries on the half-integers - hence the minor ticks at -0.5, 0.5, ...

    Skipped once the grid is dense enough that the lines would cover more area
    than the cells they separate.
    """
    if nrows <= 0 or ncols <= 0 or nrows > 60 or ncols > 60:
        return
    ax.set_xticks(_np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(_np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which="minor", color=color, linestyle="-", linewidth=lw)
    # Minor ticks would otherwise sprout marks between every label.
    ax.tick_params(which="minor", length=0)


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
# Interactivity
# ─────────────────────────────────────────────────────────────────────
#
# Every plot in the app is meant to be explorable: hover to read a point,
# wheel to zoom, right-drag to pan, double-click to go back.  This used to
# be delegated to ``mplcursors``, which is not a declared dependency, so on
# any machine without it every hover call was a silent no-op and the plots
# were in practice static pictures.  The implementation below is plain
# matplotlib event handling, so it works wherever the app runs.


def _fmt_num(v) -> str:
    """Format a number for a tooltip without scientific-notation noise."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not _np.isfinite(f):
        return str(v)
    if f != 0 and (abs(f) < 1e-3 or abs(f) >= 1e5):
        return f"{f:.3g}"
    return f"{f:.4g}"


def attach_point_labels(artist, labels) -> None:
    """Name the points of ``artist`` so hover can report them.

    Without this a tooltip can only say where a point is; with it the
    tooltip can say what the point *is* (a sample ID, a gene, a term),
    which is the part the user actually needs.
    """
    try:
        artist._gv_labels = list(labels)
    except Exception:
        pass


def attach_sample_table(fig, table, key_col: str = "GSM") -> None:
    """Register the full record behind the named points of ``fig``.

    A plot shows two numbers about a sample. Once a user has lassoed a
    cluster the question is always the wider one - which studies are these,
    which tissues, which treatments - and answering it needs the whole row,
    not the two coordinates that happened to be plotted. This is where the
    selection window gets those rows from; without it a selection can still
    be made, it just reports accessions.

    ``table`` may be a callable returning the frame. Assembling one can mean
    merging the expression matrix with the study index and the label table,
    which is not work to do while drawing a plot that may never be selected
    on - pass the builder and it is called on the first selection instead.
    """
    try:
        if table is None or (not callable(table)
                             and getattr(table, "empty", False)):
            return
        fig._gv_sample_table = (table, key_col)
    except Exception:
        pass


def sample_table_of(fig):
    """Return ``(table, key_col)`` registered on ``fig``, or ``(None, None)``."""
    got = getattr(fig, "_gv_sample_table", None)
    return got if got else (None, None)


class PointPick:
    """One plotted point the user clicked.

    ``label`` is whatever :func:`attach_point_labels` named the point - for
    sample-level plots that is the GSM accession, which is what lets a click
    be turned into a metadata lookup.
    """

    __slots__ = ("label", "index", "x", "y", "artist", "axes")

    def __init__(self, label, index, x, y, artist, axes):
        self.label = label
        self.index = index
        self.x = x
        self.y = y
        self.artist = artist
        self.axes = axes

    def __repr__(self):
        return f"PointPick({self.label!r}, x={self.x:.4g}, y={self.y:.4g})"


class PlotInteractor:
    """Hover, zoom, pan, reset and sample selection for a figure.

    Deliberately conservative about which mouse buttons it claims. A plain
    button 1 is left entirely alone: the toolbar's pan/zoom modes and the
    brushing selectors in the region and subset windows all use it, and
    stealing it would break them. Panning is therefore on the right button,
    selection is on button 1 *with a modifier*, and every handler stands down
    while a toolbar mode is active.

    Gestures
    --------
    ==========================  ====================================
    wheel                       zoom both axes about the cursor
    shift + wheel               zoom x only
    ctrl + wheel                zoom y only  (spreads a stacked strip)
    right-drag                  pan
    double-click                back to the starting view
    shift + left-drag           rectangle-select samples
    ctrl + left-drag            lasso-select samples
    ==========================  ====================================
    """

    # Hover walks the artists under the cursor, so a figure carrying tens of
    # thousands of them (a dense heatmap of individual patches) would stutter.
    # Past this count hover is dropped and zoom/pan are kept.
    _MAX_HOVER_ARTISTS = 600

    def __init__(self, fig, *, hover=True, zoom=True, pan=True, on_pick=None,
                 select=True, on_select=None):
        self.fig = fig
        self.on_pick = on_pick
        self.on_select = on_select
        self._annots = {}
        self._home = {}
        self._drag = None
        self._last_key = None
        self._moved = set()
        self._cids = []
        self._sel = None          # in-flight selection gesture
        self._band = None         # rubber band being dragged
        self._marks = []          # rings left on the last selected points
        # Cached pixels of everything except the tooltip. Moving a tooltip
        # changes one artist, but a full redraw repaints every curve, fill and
        # rug point behind it -- measured at 117 ms on a 20-group distribution
        # plot, i.e. ~8 fps while the pointer is moving. Blitting repaints only
        # the tooltip over these cached pixels, measured at 1.8 ms (64x).
        self._bg = None
        self._blit_ok = True
        canvas = fig.canvas
        if canvas is None:
            return
        # "Home" is recharged on every draw the user did not cause, so a panel
        # that replots into the same figure - the single-cell views do this
        # repeatedly - resets to the new plot rather than to a stale view of
        # the old one. Axes the user has zoomed are left alone so their home
        # stays the view they started from.
        self._cids.append(canvas.mpl_connect("draw_event", self._on_draw))
        if hover and self._artist_count() <= self._MAX_HOVER_ARTISTS:
            self._cids.append(
                canvas.mpl_connect("motion_notify_event", self._on_hover))
        if zoom:
            self._cids.append(
                canvas.mpl_connect("scroll_event", self._on_scroll))
        if pan:
            self._cids.append(
                canvas.mpl_connect("button_press_event", self._on_press))
            self._cids.append(
                canvas.mpl_connect("motion_notify_event", self._on_drag))
            self._cids.append(
                canvas.mpl_connect("button_release_event", self._on_release))
        self._cids.append(
            canvas.mpl_connect("button_press_event", self._on_reset))
        if on_pick is not None:
            self._cids.append(
                canvas.mpl_connect("button_press_event", self._on_click))
        if select:
            self._cids.append(
                canvas.mpl_connect("button_press_event", self._sel_press))
            self._cids.append(
                canvas.mpl_connect("motion_notify_event", self._sel_move))
            self._cids.append(
                canvas.mpl_connect("button_release_event", self._sel_release))
        # Keep the interactor alive for as long as the figure is: nothing
        # else holds a reference to it, and a garbage-collected interactor
        # is a plot that silently stops responding.
        fig._gv_interactor = self

    # ── plumbing ─────────────────────────────────────────────────────
    def _artist_count(self) -> int:
        n = 0
        for ax in self.fig.axes:
            n += (len(ax.collections) + len(ax.lines) + len(ax.patches)
                  + len(ax.images) + len(ax.texts))
        return n

    def _on_draw(self, _event) -> None:
        for ax in self.fig.axes:
            if ax not in self._moved:
                self._home[ax] = (ax.get_xlim(), ax.get_ylim())
        # The figure has just been rendered without the tooltips (they are
        # animated), so these pixels are exactly the background to blit onto.
        # Grabbing it here rather than on first hover means the cache is always
        # in step with what is on screen, including after a pan or zoom.
        self._bg = None
        if not self._blit_ok:
            return
        try:
            self._bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        except Exception:
            # Backend without a pixel buffer: fall back to plain redraws.
            self._blit_ok = False

    def _blit(self, ax) -> None:
        """Repaint just the tooltips and rubber band over the cached pixels."""
        canvas = self.fig.canvas
        if not self._blit_ok or self._bg is None:
            canvas.draw_idle()
            return
        try:
            canvas.restore_region(self._bg)
            overlay = list(self._annots.values())
            if self._band is not None:
                overlay.append(self._band)
            for a in overlay:
                if a.get_visible() and a.axes is not None:
                    a.axes.draw_artist(a)
            canvas.blit(self.fig.bbox)
        except Exception:
            self._blit_ok = False
            canvas.draw_idle()

    def _remember_home(self, ax) -> None:
        """Record ``ax`` as user-moved, pinning whatever home it has now."""
        self._home.setdefault(ax, (ax.get_xlim(), ax.get_ylim()))
        self._moved.add(ax)

    def _busy(self) -> bool:
        """True while the toolbar owns the mouse (pan or zoom-rect active)."""
        tb = getattr(self.fig.canvas, "toolbar", None)
        return bool(getattr(tb, "mode", "") or "")

    def _annotation(self, ax):
        annot = self._annots.get(ax)
        if annot is None or annot.axes is not ax:
            annot = ax.annotate(
                "", xy=(0, 0), xytext=(14, 14), textcoords="offset points",
                ha="left", va="bottom", zorder=10_000, annotation_clip=False,
                fontsize=TYPOGRAPHY["annot"]["fontsize"],
                color=AERO["text"],
                bbox=dict(boxstyle="round,pad=0.45", fc="#FFFFFF",
                          ec=AERO["accent"], lw=1.0, alpha=0.96),
                arrowprops=dict(arrowstyle="-", color=AERO["accent"], lw=0.9))
            annot.set_visible(False)
            # Animated artists are skipped by a normal draw, which is what keeps
            # them out of the cached background and lets them be blitted on top.
            annot.set_animated(True)
            self._annots[ax] = annot
        return annot

    # ── hover ────────────────────────────────────────────────────────
    def _on_hover(self, event):
        ax = event.inaxes
        if ax is None or self._drag is not None:
            self._hide()
            return
        hit = self._hit_test(ax, event)
        if hit is None:
            self._hide()
            return
        key, text, (x, y), _artist, _index = hit
        if key == self._last_key:
            return
        self._last_key = key
        annot = self._annotation(ax)
        annot.xy = (x, y)
        annot.set_text(text)
        annot.set_visible(True)
        self._blit(ax)

    def _hide(self) -> None:
        if self._last_key is None:
            return
        self._last_key = None
        for annot in self._annots.values():
            annot.set_visible(False)
        self._blit(None)

    def _hit_test(self, ax, event):
        """Return (key, text, (x, y), artist, index) for the best hit.

        An artist that names its points always wins over one that does not,
        whatever the draw order. Purely topmost-wins was wrong: ``axvspan``
        region shading and the sample-strip band are full-plot rectangles
        drawn after the points, and patches are tested before collections,
        so every hover in a highlighted region reported the shading ("value:
        1") and every sample underneath became unclickable - the pick needs
        a named point and the shading has none.
        """
        # Reverse order so the artist drawn last - the one visually on top -
        # is the one preferred within each class.
        fallback = None
        for artist in reversed(list(ax.collections) + list(ax.lines)
                               + list(ax.patches) + list(ax.images)):
            # The rings marking a selection sit exactly on top of the points
            # they mark, so hit-testing them would report the overlay instead
            # of the sample underneath and make selected points un-clickable.
            if not artist.get_visible() or getattr(artist, "_gv_overlay", False):
                continue
            try:
                found, info = artist.contains(event)
            except Exception:
                continue
            if not found:
                continue
            try:
                out = self._describe(ax, artist, info, event)
            except Exception:
                continue
            if out is None:
                continue
            key, text, xy, index = out
            if getattr(artist, "_gv_labels", None):
                return key, text, xy, artist, index
            if fallback is None:
                fallback = (key, text, xy, artist, index)
        return fallback

    def _describe(self, ax, artist, info, event):
        name = getattr(artist, "get_label", lambda: "")() or ""
        if name.startswith("_"):
            name = ""
        labels = getattr(artist, "_gv_labels", None)
        xlab = ax.get_xlabel() or "x"
        ylab = ax.get_ylabel() or "y"
        ind = info.get("ind") if isinstance(info, dict) else None

        # QuadMesh (pcolormesh, seaborn heatmap) has to be identified before
        # the offsets branch: it inherits a single default offset from
        # Collection and would otherwise be read as a one-point scatter.
        if isinstance(artist, _mcoll.QuadMesh):
            val = artist.get_cursor_data(event)
            if val is None:
                return None
            row, col, (cx, cy), (tx, ty) = _mesh_cell(artist, event)
            rname = _tick_text(ax.get_yticklabels(), ax.get_yticks(), cy, ty)
            cname = _tick_text(ax.get_xticklabels(), ax.get_xticks(), cx, tx)
            head = " / ".join(p for p in (rname, cname) if p)
            body = _fmt_num(_np.ravel(val)[0] if _np.ndim(val) else val)
            return (("mesh", id(artist), row, col),
                    f"{head}\n{body}" if head else body,
                    (event.xdata, event.ydata), None)

        # Scatter / any offset-based collection
        offsets = getattr(artist, "get_offsets", None)
        if offsets is not None and ind is not None and len(ind):
            pts = _np.asarray(offsets())
            if pts.ndim == 2 and len(pts):
                i = int(ind[0]) % len(pts)
                x, y = float(pts[i][0]), float(pts[i][1])
                head = (str(labels[i]) if labels is not None and i < len(labels)
                        else name)
                body = f"{xlab}: {_fmt_num(x)}\n{ylab}: {_fmt_num(y)}"
                return (("pt", id(artist), i),
                        f"{head}\n{body}" if head else body, (x, y), i)

        # Line / curve
        if hasattr(artist, "get_xydata") and ind is not None and len(ind):
            pts = _np.asarray(artist.get_xydata())
            if pts.ndim == 2 and len(pts):
                i = int(ind[0]) % len(pts)
                x, y = float(pts[i][0]), float(pts[i][1])
                head = (str(labels[i]) if labels is not None and i < len(labels)
                        else name)
                body = f"{xlab}: {_fmt_num(x)}\n{ylab}: {_fmt_num(y)}"
                return (("ln", id(artist), i),
                        f"{head}\n{body}" if head else body, (x, y), i)

        # Bar / box / any rectangle patch
        if hasattr(artist, "get_width") and hasattr(artist, "get_height"):
            w, h = artist.get_width(), artist.get_height()
            x0, y0 = artist.get_xy()
            # A horizontal bar's magnitude is its width, a vertical bar's is
            # its height; pick whichever axis the bar actually extends along.
            vertical = abs(h) >= abs(w)
            value = h if vertical else w
            anchor = (x0 + w / 2, y0 + h) if vertical else (x0 + w, y0 + h / 2)
            head = name or "value"
            return (("bar", id(artist)),
                    f"{head}: {_fmt_num(value)}", anchor, None)

        # Image / heatmap / mesh
        if hasattr(artist, "get_array") and artist.get_array() is not None:
            arr = _np.asarray(artist.get_array())
            if hasattr(artist, "get_extent") and arr.ndim >= 2:
                x0, x1, y0, y1 = artist.get_extent()
                nr, nc = arr.shape[0], arr.shape[1]
                # extent is (left, right, bottom, top). Under imshow's default
                # origin='upper' the bottom coordinate is the larger one and
                # row 0 sits at the *top* edge, so the fraction along the axis
                # has to be measured from the far end or every row is mirrored.
                tx = (event.xdata - x0) / ((x1 - x0) or 1)
                ty = (event.ydata - y0) / ((y1 - y0) or 1)
                if x0 > x1:
                    tx = 1.0 - tx
                if y0 > y1:
                    ty = 1.0 - ty
                col = max(0, min(nc - 1, int(tx * nc)))
                row = max(0, min(nr - 1, int(ty * nr)))
                val = arr[row, col]
                # Map the cell back to a data coordinate so its tick label can
                # be found, undoing the origin flip applied above.
                fx = (col + 0.5) / nc
                fy = (row + 0.5) / nr
                cx = x0 + (1 - fx if x0 > x1 else fx) * (x1 - x0)
                cy = y0 + (1 - fy if y0 > y1 else fy) * (y1 - y0)
                rname = _tick_text(ax.get_yticklabels(), ax.get_yticks(), cy,
                                   abs(y1 - y0) / (2 * nr))
                cname = _tick_text(ax.get_xticklabels(), ax.get_xticks(), cx,
                                   abs(x1 - x0) / (2 * nc))
                head = " / ".join(p for p in (rname, cname) if p)
                body = _fmt_num(_np.ravel(val)[0] if _np.ndim(val) else val)
                return (("img", id(artist), row, col),
                        f"{head}\n{body}" if head else body,
                        (event.xdata, event.ydata), None)
        return None

    # ── click to inspect ─────────────────────────────────────────────
    def _on_click(self, event):
        """Report the point the user clicked, so the GUI can open its record.

        Only a plain left click on an actual point counts. The event is not
        consumed: the brushing selectors in the region and subset windows are
        also on button 1, and they have to keep receiving it.
        """
        if (event.button != 1 or event.dblclick or event.inaxes is None
                or self._busy()):
            return
        hit = self._hit_test(event.inaxes, event)
        if hit is None:
            return
        _key, _text, (x, y), artist, index = hit
        if index is None:
            return
        labels = getattr(artist, "_gv_labels", None)
        label = (str(labels[index])
                 if labels is not None and index < len(labels) else None)
        if label is None:
            return
        try:
            self.on_pick(PointPick(label, index, x, y, artist, event.inaxes))
        except Exception:
            pass

    # ── zoom ─────────────────────────────────────────────────────────
    def _on_scroll(self, event):
        ax = event.inaxes
        if ax is None or self._busy():
            return
        self._remember_home(ax)
        # Zoom about the cursor rather than the centre, so the feature the
        # user is pointing at stays under the pointer as the view tightens.
        scale = 1 / 1.2 if event.button == "up" else 1.2
        # Zooming both axes at once cannot separate samples that are stacked
        # in y at one shared x - the gap grows in x just as fast. Holding
        # ctrl stretches y alone, which is what pulls a tie pile apart;
        # shift does the same for x.
        key = event.key or ""
        axes_wanted = ("x", "y")
        if "control" in key or "ctrl" in key:
            axes_wanted = ("y",)
        elif "shift" in key:
            axes_wanted = ("x",)
        for which, get, set_, anchor in (
                ("x", ax.get_xlim, ax.set_xlim, event.xdata),
                ("y", ax.get_ylim, ax.set_ylim, event.ydata)):
            if which not in axes_wanted:
                continue
            lo, hi = get()
            if anchor is None or not _np.isfinite([lo, hi]).all():
                continue
            set_(anchor + (lo - anchor) * scale,
                 anchor + (hi - anchor) * scale)
        self.fig.canvas.draw_idle()

    # ── pan ──────────────────────────────────────────────────────────
    def _on_press(self, event):
        if event.button != 3 or event.inaxes is None or self._busy():
            return
        self._remember_home(event.inaxes)
        self._drag = (event.inaxes, event.xdata, event.ydata,
                      event.inaxes.get_xlim(), event.inaxes.get_ylim())

    def _on_drag(self, event):
        if self._drag is None or event.x is None:
            return
        ax, x0, y0, xlim, ylim = self._drag
        # Work in pixels: converting the press point back through the *current*
        # limits each time would chase its own tail and make the pan accelerate.
        try:
            px0, py0 = ax.transData.transform((x0, y0))
        except Exception:
            return
        dx, dy = event.x - px0, event.y - py0
        inv = ax.transData.inverted()
        (ax0, ay0), (ax1, ay1) = inv.transform((0, 0)), inv.transform((dx, dy))
        ax.set_xlim(xlim[0] - (ax1 - ax0), xlim[1] - (ax1 - ax0))
        ax.set_ylim(ylim[0] - (ay1 - ay0), ylim[1] - (ay1 - ay0))
        self.fig.canvas.draw_idle()

    def _on_release(self, event):
        if event.button == 3:
            self._drag = None

    # ── reset ────────────────────────────────────────────────────────
    def _on_reset(self, event):
        if not event.dblclick or event.inaxes is None or self._busy():
            return
        home = self._home.get(event.inaxes)
        if home is None:
            return
        event.inaxes.set_xlim(home[0])
        event.inaxes.set_ylim(home[1])
        # Back under the draw handler's care: the next replot should set a
        # fresh home rather than keep restoring this one.
        self._moved.discard(event.inaxes)
        self.fig.canvas.draw_idle()

    # ── select ───────────────────────────────────────────────────────
    def _sel_press(self, event):
        ax = event.inaxes
        if event.button != 1 or ax is None or self._busy() or event.dblclick:
            return
        key = event.key or ""
        if "control" in key or "ctrl" in key:
            mode = "lasso"
        elif "shift" in key:
            mode = "rect"
        else:
            return
        # Some windows already run their own RectangleSelector on button 1;
        # firing here too would open two selection windows for one drag.
        if getattr(ax, "_gv_external_brush", False):
            return
        self._hide()
        self._sel = {"ax": ax, "mode": mode,
                     "x0": event.xdata, "y0": event.ydata,
                     "verts": [(event.xdata, event.ydata)]}
        if mode == "rect":
            from matplotlib.patches import Rectangle
            self._band = Rectangle((event.xdata, event.ydata), 0, 0,
                                   facecolor=AERO["accent"], alpha=0.18,
                                   edgecolor=AERO["danger"], lw=1.4, zorder=9_999)
            ax.add_patch(self._band)
        else:
            from matplotlib.lines import Line2D
            self._band = Line2D([event.xdata], [event.ydata],
                                color=AERO["danger"], lw=1.6, zorder=9_999)
            ax.add_line(self._band)
        self._band.set_animated(True)
        self._band._gv_overlay = True

    def _sel_move(self, event):
        if self._sel is None or event.xdata is None or event.ydata is None:
            return
        if event.inaxes is not self._sel["ax"]:
            return
        if self._sel["mode"] == "rect":
            x0, y0 = self._sel["x0"], self._sel["y0"]
            self._band.set_bounds(min(x0, event.xdata), min(y0, event.ydata),
                                  abs(event.xdata - x0), abs(event.ydata - y0))
        else:
            self._sel["verts"].append((event.xdata, event.ydata))
            xs, ys = zip(*self._sel["verts"])
            self._band.set_data(xs, ys)
        self._blit(self._sel["ax"])

    def _sel_release(self, event):
        sel, self._sel = self._sel, None
        if sel is None:
            return
        band, self._band = self._band, None
        try:
            band.remove()
        except Exception:
            pass
        ax = sel["ax"]
        if sel["mode"] == "rect":
            x1, y1 = event.xdata, event.ydata
            if x1 is None or y1 is None:
                self.fig.canvas.draw_idle()
                return
            lo = (min(sel["x0"], x1), min(sel["y0"], y1))
            hi = (max(sel["x0"], x1), max(sel["y0"], y1))
            inside = lambda p: (lo[0] <= p[0] <= hi[0]) and (lo[1] <= p[1] <= hi[1])
        else:
            verts = sel["verts"]
            if len(verts) < 3:
                self.fig.canvas.draw_idle()
                return
            from matplotlib.path import Path
            path = Path(verts)
            inside = path.contains_point
        picked = self._points_in(ax, inside)
        self._mark(ax, picked)
        self.fig.canvas.draw_idle()
        if picked:
            self._deliver(ax, picked)

    def _points_in(self, ax, inside):
        """Every named point of ``ax`` the predicate accepts, de-duplicated.

        Named is the operative word: a selection is only meaningful over
        artists whose points carry identities (see
        :func:`attach_point_labels`). Ordinary curve vertices are geometry,
        not samples, and pulling them in would fill the window with noise.
        """
        seen, out = set(), []
        for artist in list(ax.collections) + list(ax.lines):
            labels = getattr(artist, "_gv_labels", None)
            if not labels or getattr(artist, "_gv_overlay", False):
                continue
            if not artist.get_visible():
                continue
            try:
                if hasattr(artist, "get_offsets"):
                    pts = _np.asarray(artist.get_offsets())
                else:
                    pts = _np.asarray(artist.get_xydata())
            except Exception:
                continue
            if pts.ndim != 2 or not len(pts):
                continue
            for i, p in enumerate(pts):
                if i >= len(labels) or not _np.isfinite(p).all():
                    continue
                if not inside((float(p[0]), float(p[1]))):
                    continue
                name = str(labels[i])
                if name in seen:
                    continue
                seen.add(name)
                out.append((name, float(p[0]), float(p[1])))
        return out

    def _mark(self, ax, picked):
        """Ring the selected points so the selection stays visible."""
        for m in self._marks:
            try:
                m.remove()
            except Exception:
                pass
        self._marks = []
        if not picked:
            return
        xs = [p[1] for p in picked]
        ys = [p[2] for p in picked]
        ring = ax.scatter(xs, ys, s=110, facecolors="none",
                          edgecolors=AERO["danger"], linewidths=1.6,
                          zorder=9_000, label="_nolegend_")
        ring._gv_overlay = True
        self._marks.append(ring)

    def _deliver(self, ax, picked):
        names = [p[0] for p in picked]
        if self.on_select is not None:
            try:
                self.on_select(names, ax, self.fig)
            except Exception:
                pass
            return
        # Imported here, not at module scope: this is a plotting utility and
        # must stay importable in a headless context with no GUI toolkit.
        try:
            from genevariate.gui.windows.sample_selection import show_selection
        except Exception:
            return
        try:
            show_selection(self.fig, names, ax)
        except Exception:
            pass

    def disconnect(self) -> None:
        canvas = self.fig.canvas
        for cid in self._cids:
            try:
                canvas.mpl_disconnect(cid)
            except Exception:
                pass
        self._cids = []


def _mesh_cell(mesh, event):
    """Locate the QuadMesh cell under the cursor on a regular grid.

    Returns ``(row, col, centre_xy, half_cell_xy)``; the centre and cell size
    are what let the tick labels be matched back to the cell.
    """
    try:
        corners = _np.asarray(mesh.get_coordinates())
        xs, ys = corners[0, :, 0], corners[:, 0, 1]
        col = int(_np.searchsorted(xs, event.xdata) - 1)
        row = int(_np.searchsorted(ys, event.ydata) - 1)
        if 0 <= row < len(ys) - 1 and 0 <= col < len(xs) - 1:
            return (row, col,
                    ((xs[col] + xs[col + 1]) / 2, (ys[row] + ys[row + 1]) / 2),
                    (abs(xs[col + 1] - xs[col]) / 2,
                     abs(ys[row + 1] - ys[row]) / 2))
    except Exception:
        pass
    return None, None, (None, None), (0.5, 0.5)


def _tick_text(labels, ticks, centre, tol=0.5) -> str:
    """Name the heatmap row/column centred on ``centre`` from its tick labels.

    Matching is on position rather than index because the two heatmap kinds
    disagree about where a cell sits: imshow puts cell *i* at *i*, pcolormesh
    puts it at *i + 0.5*.

    Only categorical names are returned. A tick whose text is merely its own
    numeric position belongs to an auto-generated axis, and reporting "0.75"
    as the name of a row is worse than saying nothing at all.
    """
    if centre is None:
        return ""
    best, best_d = "", float(tol)
    for lbl, t in zip(labels, ticks):
        try:
            d = abs(float(t) - float(centre))
        except (TypeError, ValueError):
            continue
        if d >= best_d:
            continue
        text = lbl.get_text().strip()
        if not text:
            continue
        try:
            if abs(float(text) - float(t)) < 1e-9:
                continue
        except ValueError:
            pass
        best, best_d = text, d
    return best


def make_interactive(fig, *, hover=True, zoom=True, pan=True, on_pick=None,
                     select=True, on_select=None):
    """Make every axes in ``fig`` hoverable, zoomable, pannable, selectable.

    Pass ``on_pick`` to also make named points clickable; it receives a
    :class:`PointPick` describing the point, which for sample-level plots is
    enough to look the sample's metadata up.

    Selection (shift/ctrl + left-drag) is on by default and needs no wiring:
    it collects whatever :func:`attach_point_labels` named and opens the
    selected-samples window over the table :func:`attach_sample_table`
    registered. Pass ``on_select(names, ax, fig)`` to handle it instead.

    Safe to call on any figure, including ones already carrying a toolbar;
    returns ``None`` rather than raising if the figure cannot support it.
    """
    if fig is None or getattr(fig, "canvas", None) is None:
        return None
    old = getattr(fig, "_gv_interactor", None)
    if old is not None:
        old.disconnect()
    try:
        return PlotInteractor(fig, hover=hover, zoom=zoom, pan=pan,
                              on_pick=on_pick, select=select,
                              on_select=on_select)
    except Exception:
        return None
