"""
GeneVariate - Region Analysis Window v5
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, colorchooser
import pandas as pd
import numpy as np
import itertools
import threading
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.stats import ranksums, wasserstein_distance

from genevariate.core import label_entities
from genevariate.core.analysis.bimodality import robust_kde
from genevariate.core.analysis import (
    build_enrichment_cells,
    region_label_enrichment,
)
from genevariate.gui.theme import (
    AERO, MONO_FONT, UI_FONT, labelframe, ensure_theme, style_toolbar, style_window,
    wrap_to_parent as _wrap_to_parent,
)

plt.rcParams['figure.max_open_warning'] = 50

# Bootstrap resamples (of studies, not samples) behind every enrichment CI
_ENRICH_BOOT = 500

# Unified GeneVariate plot stylesheet (graceful fallback if utils missing)
try:
    from genevariate.utils.viz_style import (
        apply_genevariate_style as _apply_gv_style,
        palette_for as _palette_for,
        cmap_for as _cmap_for,
        style_axis as _style_axis,
        smart_figsize as _smart_figsize,
        cap_figsize as _cap_figsize,
        make_interactive as _make_interactive,
        attach_point_labels as _point_labels,
        attach_sample_table as _sample_table,
        grid_matrix_cells as _cell_grid,
    )
    _apply_gv_style()
except Exception:
    def _palette_for(n, use_case="discrete"):
        if n <= 10: p = sns.color_palette("tab10", n)
        elif n <= 20: p = sns.color_palette("tab20", n)
        else: p = sns.color_palette("husl", n)
        return [mcolors.to_hex(c) for c in p]
    def _cmap_for(kind="sequential"):
        return "viridis" if kind != "diverging" else "RdBu_r"
    def _style_axis(ax, xlabel=None, ylabel=None, title=None):
        if xlabel is not None: ax.set_xlabel(xlabel)
        if ylabel is not None: ax.set_ylabel(ylabel)
        if title is not None: ax.set_title(title)
    def _smart_figsize(kind="default"): return (10, 6)
    def _cap_figsize(w, h, max_w=16.0, max_h=10.0):
        return (min(w, max_w), min(h, max_h))
    def _make_interactive(fig, **kw): return None
    def _point_labels(artist, labels): return None
    def _sample_table(fig, table, key_col="GSM"): return None
    def _cell_grid(ax, nrows, ncols, **kw): return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# How much of a result each chart and table shows is the user's setting, not a
# constant: see gui/display_limits.py and the Display limits button below.
def _lim(key):
    from genevariate.gui import display_limits
    n = display_limits.get(key)
    # An unlimited setting still has to be a number where one is indexed with.
    return 10 ** 9 if n is None else n

# Treeview row tints, tied to the palette rather than to ad-hoc hexes
_ROW_GOOD = AERO['green_light']
_ROW_BAD = '#F6D5D0'    # AERO danger, lightened to stay readable behind text
_ROW_MUTED = AERO['border_soft']
# The model is given the computed summary and asked to read it. It is never
# given the data, so it has no numbers of its own to offer and cannot quietly
# replace a measured one with a plausible one.
_CMP_SYSTEM_PROMPT = (
    "You are reading the output of a region-comparison analysis of gene "
    "expression data from GEO. Every number below has already been computed. "
    "Explain what they mean for a biologist in at most six short paragraphs.\n"
    "Rules you must follow:\n"
    "- Never state a number that is not in the text you were given, and never "
    "round one into a different claim.\n"
    "- A region is a range brushed on one gene's expression; samples arrive in "
    "study-sized clumps, so a result carried by few studies is weak evidence "
    "no matter how small its q-value.\n"
    "- If two regions have a high Jaccard overlap, say plainly that they are "
    "largely the same samples and not independent evidence.\n"
    "- If a cross-fitted AUC is near 0.5, say the region is not distinguishable "
    "once whole studies are held out, and that this outranks the raw counts.\n"
    "- Enrichment is association, never causation. Do not suggest mechanism.\n"
    "- If the numbers do not support a clear conclusion, say so instead of "
    "manufacturing one."
)

_BG_CLR = '#8888AA'
_BG_ALP = 0.50
_BG_EDGE = '#666688'
_LW = 2.8
_LA = 0.88


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Flat AERO controls for the toolbars
#
#  These stay tk.Button rather than ttk because the toolbars need per-button
#  fills and a latched on/off state; ttk's pill styles carry neither. Flat with
#  a 1px border and a hover fill, never the beveled relief=RAISED look.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _flat_button(parent, text, command, *, fill=None, fg=None, hover=None,
                 font=(UI_FONT, 10, 'bold'), padx=14, pady=5):
    """A flat, filled toolbar button. ``fill=None`` gives the quiet variant."""
    fill = fill or AERO['panel']
    fg = fg or (AERO['accent_dark'] if fill == AERO['panel'] else 'white')
    hover = hover or (AERO['hover_sky'] if fill == AERO['panel'] else fill)
    b = tk.Button(parent, text=text, command=command, font=font,
                  padx=padx, pady=pady, cursor='hand2',
                  relief=tk.FLAT, bd=0, highlightthickness=1,
                  highlightbackground=AERO['border'],
                  highlightcolor=AERO['border'],
                  bg=fill, fg=fg,
                  activebackground=hover, activeforeground=fg)
    return b


def _paint_toggle(btn, active, accent=None):
    """Latch a toolbar button on or off without the beveled sunken relief."""
    accent = accent or AERO['accent']
    if active:
        btn.config(bg=accent, fg='white', activebackground=accent,
                   highlightbackground=accent, highlightcolor=accent)
    else:
        btn.config(bg=AERO['panel'], fg=AERO['accent_dark'],
                   activebackground=AERO['hover_sky'],
                   highlightbackground=AERO['border'],
                   highlightcolor=AERO['border'])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_ID_LIKE_COLS = {'gsm', 'gse', 'gpl', 'series_id', 'sample', 'sample_id',
                 'geo_accession', 'platform_id'}


def _is_identifier_col(name):
    """True for a column that names which sample, study or subject a row is."""
    s = str(name).strip().lower()
    return s in _ID_LIKE_COLS or s.endswith('_id')


def _col_signature(s):
    """A hashable stand-in for a column's values, for spotting aliases."""
    return hash(tuple(s.astype(object).fillna('').astype(str)))


def platform_label_cols(plat_labels):
    """The label-value columns of a label file, one per extracted field.

    A label file is not three columns wide. The extractor writes every field
    once per pass (``phase1_``/``phase1b_``/``phase2_``/``final_``) plus an
    ontology id, a mesh id and a provenance flag for each, and it carries the
    raw ``title``/``characteristics``/``description`` it read. Testing all of
    them asks the same question four times over with identical counts, which
    inflates the hit count and breaks the FDR correction this window depends
    on, and it turns ``gse`` - pure batch structure - into a biological
    finding.

    So prefer the one value column per field that ``semantic_label_columns``
    identifies from the extractor's own field list. Files that do not follow
    that naming - a hand-made ``Tissue``/``Sex`` sheet, a curator's
    ``raw_tissue`` - fall back to every column that is neither an identifier
    nor near-unique per sample.
    """
    from genevariate.core.label_entities import label_value_columns
    return label_value_columns(plat_labels)


def _kde(vals, n=300, x_range=None):
    """KDE with tails that reach y≈0.
    
    x_range: optional (xmin, xmax) to evaluate over (e.g. full platform range).
             If None, extends by 3x bandwidth on each side so tails touch x-axis.
    """
    v = np.asarray(vals, dtype=float); v = v[np.isfinite(v)]
    if len(v) < 2 or np.ptp(v) == 0: return None
    try:
        # The estimator the Distribution Classifier counts modes on, so a curve
        # drawn anywhere in the program shows the shape the program names.
        k = robust_kde(v)
        if x_range is not None:
            # The grid has to resolve the *bandwidth*, not the axis. A selected
            # region is a narrow band of a wide platform range, so a fixed 300
            # points over that range put only a handful inside the band and the
            # curve aliased into a couple of razor spikes -- which reads as
            # "185 samples took two values" when they took 183 distinct ones.
            lo, hi = float(x_range[0]), float(x_range[1])
            bw = k.factor * v.std(ddof=1)
            if bw > 0 and hi > lo:
                # >= 8 grid points per bandwidth resolves every real feature.
                n = int(min(20000, max(n, 8.0 * (hi - lo) / bw)))
            xs = np.linspace(lo, hi, n)
        else:
            # pad by 3x the KDE bandwidth - guarantees tails drop to ≈0
            bw = k.factor * v.std(ddof=1)
            pad = max(3.0 * bw, 0.05 * np.ptp(v), 0.01)
            xs = np.linspace(v.min() - pad, v.max() + pad, n)
        ys = k(xs)
        ys = np.maximum(ys, 0)
        return xs, ys
    except: return None

def _clrs(n):
    # Delegate to unified stylesheet for a colorblind-safe, consistent palette.
    return _palette_for(n, use_case="discrete")

def _tr(s, m=28):
    s = str(s)
    # Remove control characters (tab, newline, etc.) that cause glyph warnings
    s = ''.join(ch if ord(ch) >= 32 else ' ' for ch in s)
    s = s.strip()
    return (s[:m-1] + '..') if len(s) > m else s


def _fmt_ci(row):
    """Render the study-bootstrap CI on the enrichment ratio."""
    lo, hi = row.get('ci_low', float('nan')), row.get('ci_high', float('nan'))
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "n/a"
    return f"{lo:.1f} - {hi:.1f}"


def _is_thin(row):
    """True when a hit is not backed by replicated, study-independent evidence.

    Either it rests on fewer than three studies, or the study-bootstrap CI on
    the enrichment ratio still covers 1.0 - both mean the p-value is far more
    confident than the data warrant.
    """
    n_gse = row.get('n_gse')
    if n_gse is not None and n_gse < 3:
        return True
    lo = row.get('ci_low', float('nan'))
    return bool(np.isfinite(lo) and lo <= 1.0)


def _fmt_neff(row):
    """Effective sample size, flagged when study clumping bites hard."""
    n_eff = row.get('n_eff')
    n_sel = row.get('n_sel') or 0
    if n_eff is None or not np.isfinite(n_eff):
        return "n/a"
    txt = f"{n_eff:,.0f}"
    if n_sel and n_eff < n_sel * 0.5:
        txt += f" ({n_sel / n_eff:.0f}x)"
    return txt


def _smart_series(series, max_cats=None):
    """Prepare a label series for plotting at scale.
    Auto-bins numeric columns (Age, dosage, time) into ranges.
    Collapses high-cardinality text columns into top-N + 'Other'.
    Returns (cleaned_series, was_binned_flag).
    ``max_cats=None`` reads the user's Display limits setting."""
    if max_cats is None:
        max_cats = _lim('bars')
    s = series.fillna("N/A").astype(str)
    nuniq = s.nunique()
    if nuniq <= max_cats:
        return s, False
    # Try numeric binning for columns like Age, dosage, etc.
    numeric = pd.to_numeric(series, errors='coerce')
    valid_frac = numeric.notna().sum() / max(1, len(series))
    if valid_frac > 0.5:
        try:
            n_bins = min(12, max(5, nuniq // 5))
            binned = pd.cut(numeric, bins=n_bins, duplicates='drop')
            labels = binned.astype(str).fillna("N/A")
            return labels, True
        except Exception:
            pass
    # Non-numeric high cardinality: keep top N, rest = "Other"
    top = s.value_counts().head(max_cats - 1).index
    result = s.where(s.isin(top), "Other")
    return result, False

def _bg_range(bg_df, col):
    """Get the full (min, max) range from the platform background data."""
    if bg_df is None or col not in bg_df.columns: return None
    v = pd.to_numeric(bg_df[col], errors='coerce').dropna()
    if len(v) == 0: return None
    pad = (v.max() - v.min()) * 0.03
    return (v.min() - pad, v.max() + pad)

def _draw_bg(ax, bg_df, col):
    """Draw platform background histogram, peak-normalized to max=1."""
    if bg_df is None or col not in bg_df.columns: return
    v = pd.to_numeric(bg_df[col], errors='coerce').dropna()
    if len(v) == 0: return
    counts, bin_edges = np.histogram(v, bins=min(200, max(60, int(np.sqrt(len(v))))))
    if counts.max() > 0:
        heights = counts / counts.max()  # peak-normalize to 1.0
    else:
        heights = counts.astype(float)
    widths = np.diff(bin_edges)
    ax.bar(bin_edges[:-1], heights, width=widths, align='edge',
           color=_BG_CLR, alpha=_BG_ALP, edgecolor=_BG_EDGE, linewidth=0.3, zorder=1)

def _sample_ids(frame, vals):
    """GSM accessions for the rows that survived into ``vals``, or ``None``.

    ``vals`` has been through ``dropna()``, so its index - not its position -
    is what still ties a plotted value back to the sample it came from.
    """
    try:
        if frame is None or "GSM" not in frame.columns:
            return None
        return frame.loc[vals.index, "GSM"].astype(str).tolist()
    except Exception:
        return None

#: Depth of one stacked sample, in the same units as the peak-normalized
#: density (which tops out at 1.0). Rescaled by :func:`_finish_sample_strip`
#: if a pile turns out deep enough to crowd the curves.
_STRIP_STEP = 0.02
#: Most of the axes the sample strip is ever allowed to take.
_STRIP_MAX_FRAC = 0.34


def _strip_offsets(ax, vv, x_range=None, step=_STRIP_STEP):
    """Stack samples that share an x, in a band BELOW the density baseline.

    A GSE is not one measurement: it is many GSMs, and the curve drawn for it
    is a density over those samples, never an average. The per-sample layer is
    what lets a reader see that -- but ties hide it. After quantile
    normalization the top-ranked gene of every sample lands on one shared
    ceiling, so dozens of GSMs take the identical x and collapse into a mark.

    Stacking them is therefore necessary, but the stack height is a tie COUNT,
    not a density, and drawing it inside the density axes invited exactly that
    misreading -- piles climbed to y~0.46 against an axis labelled "Density".
    So the strip hangs below zero, outside the range any density can occupy,
    and :func:`_finish_sample_strip` shades and captions it.

    Columns are tracked on the axes, not per call, so samples from *different*
    groups at the same expression stack past each other instead of the
    last-drawn group hiding the ones beneath it; because groups are drawn one
    at a time, each group's samples also stay contiguous within a column
    instead of interleaving into confetti. Returned y values are DATA
    coordinates.
    """
    yy = np.full(vv.shape, -0.8 * step, dtype=float)
    if not vv.size:
        return yy
    span = None
    if x_range is not None:
        span = float(x_range[1]) - float(x_range[0])
    if not span or not np.isfinite(span) or span <= 0:
        finite = vv[np.isfinite(vv)]
        span = float(np.ptp(finite)) if finite.size > 1 else 0.0
    # Closer than this and two marks would overlap on screen anyway, so that
    # is exactly the set that has to be stacked.
    tol = (span / 200.0) if span > 0 else 0.0

    cols = getattr(ax, "_gv_strip_cols", None)
    if cols is None:
        cols = ax._gv_strip_cols = []
    for idx in np.argsort(vv, kind="stable"):
        x = vv[idx]
        if not np.isfinite(x):
            continue
        slot = next((c for c in cols if abs(c[0] - x) <= tol), None)
        if slot is None:
            slot = [float(x), 0]
            cols.append(slot)
        # No wrap: a pile of 35 that restarted at the bottom every 30 was
        # indistinguishable from a pile of 5, which is a plain misreading.
        yy[idx] = -(slot[1] + 0.8) * step
        slot[1] += 1
    return yy


def _finish_sample_strip(ax, top=None):
    """Shade, caption and scale the per-sample band once every group is drawn.

    Called after the last :func:`_plot_grp` because only then is the deepest
    pile known. Keeps the strip to at most ``_STRIP_MAX_FRAC`` of the axes so a
    heavily tied gene cannot squash the curves, and suppresses the negative y
    tick labels -- the strip is a sample count, not a negative density.
    """
    rugs = [c for c in ax.collections if getattr(c, "_gv_strip", False)]
    if top is None:
        top = max([1.0] + [float(np.max(l.get_ydata())) for l in ax.lines
                           if l.get_ydata() is not None and len(l.get_ydata())
                           and np.isfinite(l.get_ydata()).any()])
    if not rugs:
        ax.set_ylim(0, top * 1.05)
        return
    deepest = min(float(np.min(c.get_offsets()[:, 1])) for c in rugs
                  if len(c.get_offsets()))
    budget = _STRIP_MAX_FRAC / (1.0 - _STRIP_MAX_FRAC) * top
    if deepest < -budget:
        scale = budget / abs(deepest)
        for c in rugs:
            off = np.array(c.get_offsets())
            off[:, 1] *= scale
            c.set_offsets(off)
        deepest = -budget
    floor = deepest - 0.8 * _STRIP_STEP
    ax.set_ylim(floor, top * 1.05)
    # Under the region's red highlight (zorder 0) so the selected band stays
    # readable across the strip as well as across the curves.
    # Chrome, not data: both span the whole plot, so leaving them hoverable
    # makes every sample dot underneath unhittable.
    band = ax.axhspan(floor, 0, facecolor="#F2F6FA", edgecolor="none", zorder=-1)
    base = ax.axhline(0, color="#8FA6BC", lw=1.0, zorder=6)
    band._gv_overlay = True
    base._gv_overlay = True
    n = sum(len(c.get_offsets()) for c in rugs)
    ax.text(0.004, 0.012, f"individual samples (n={n}) \u2014 stacked where equal",
            transform=ax.transAxes, fontsize=8, style="italic",
            color=AERO['muted'], va="bottom", zorder=8)
    # Below zero the y value counts samples, so a density tick there would lie.
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _p: "" if v < -1e-9 else f"{v:g}"))


def _plot_grp(ax, vals, clr, mode, lw=_LW, x_range=None, ids=None):
    """KDE density with FILLED area + rug. Peak-normalized (max=1).
    Filled area makes conditions clearly visible over background histogram.

    ``ids`` names the sample behind each value, which is what makes the rug
    ticks hoverable and clickable rather than decorative.
    """
    arts = []
    if mode in ("density", "both"):
        kd = _kde(vals, x_range=x_range)
        if kd:
            xs, ys = kd
            # Peak-normalize: tallest peak = 1.0
            peak = ys.max()
            if peak > 0:
                ys = ys / peak
            # Filled area (semi-transparent) + bold line on top
            fill = ax.fill_between(xs, ys, alpha=0.25, color=clr, zorder=4)
            ln, = ax.plot(xs, ys, color=clr, lw=lw + 0.5, alpha=_LA, zorder=5)
            arts.append(ln)
            arts.append(fill)
        else:
            # No KDE is possible when the group has no spread: one sample, or
            # many samples carrying the identical value. The second case is
            # ordinary after quantile normalization, which gives every sample
            # the same sorted value distribution -- so the gene ranked top in
            # a sample always lands on the one shared ceiling, and a group of
            # such samples is exactly constant. Drawing the spike says
            # "all here"; drawing nothing reads as "no data".
            v = np.asarray(vals, dtype=float)
            v = v[np.isfinite(v)]
            if v.size and np.ptp(v) == 0:
                vl = ax.axvline(float(v[0]), color=clr, ls=':', lw=lw,
                                alpha=0.7, zorder=4)
                arts.append(vl)
    if mode in ("rug", "both"):
        # One mark per GSM, drawn as a scatter rather than with sns.rugplot:
        # a rugplot is a LineCollection with no per-point offsets, so its
        # ticks cannot be hit-tested, and each one is a sample the user should
        # be able to hover and click.
        vv = np.asarray(vals, dtype=float)
        # Below the baseline, in data coordinates, so the strip can be shaded
        # and scaled as a unit once every group has contributed to it. Height
        # here counts tied samples and must never be read off the density axis.
        rug = ax.scatter(vv, _strip_offsets(ax, vv, x_range), marker='o', s=18,
                         facecolor=clr, edgecolor='white', alpha=0.95,
                         linewidths=0.5, zorder=7)
        rug._gv_strip = True
        if ids is not None:
            _point_labels(rug, [str(i) for i in ids])
        arts.append(rug)
    return arts

def _interactive_legend(fig, legend, artist_map):
    """Click legend entry -> color picker -> recolor handle + artists."""
    if not legend: return
    hmap = {}
    for lh, lt in zip(legend.legend_handles, legend.get_texts()):
        lb = lt.get_text()
        if lb in artist_map: hmap[lh] = (lb, artist_map[lb]); lh.set_picker(8)
    def _pick(ev):
        h = ev.artist
        if h not in hmap: return
        lb, arts = hmap[h]
        try: cur = mcolors.to_hex(h.get_color())
        except:
            try: cur = mcolors.to_hex(h.get_facecolor())
            except: cur = '#FF0000'
        r = colorchooser.askcolor(color=cur, title=f"Color: {lb}")
        if r and r[1]:
            for s in ('set_color','set_facecolor','set_edgecolor'):
                try: getattr(h, s)(r[1])
                except: pass
            for a in arts:
                for s in ('set_color','set_facecolor','set_edgecolor'):
                    try: getattr(a, s)(r[1])
                    except: pass
            fig.canvas.draw_idle()
    fig.canvas.mpl_connect('pick_event', _pick)


class ScrollableCanvasFrame(ttk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        # A bare tk.Canvas defaults to platform grey and the ttk theme cannot
        # reach it, so any tab whose content did not fill the frame showed a
        # grey slab behind it.
        self.canvas = tk.Canvas(self, highlightthickness=0,
                                bg=AERO["bg_top"])
        self.vs = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.hs = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.sf = ttk.Frame(self.canvas)
        self.sf.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self._fid = self.canvas.create_window((0, 0), window=self.sf, anchor="nw")
        self.canvas.configure(yscrollcommand=self.vs.set, xscrollcommand=self.hs.set)
        self.vs.pack(side=tk.RIGHT, fill=tk.Y)
        self.hs.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(self._fid, width=e.width))
        self.canvas.bind_all("<Button-4>", _wheel_to_frame_under_pointer)
        self.canvas.bind_all("<Button-5>", _wheel_to_frame_under_pointer)
    @property
    def scrollable_frame(self): return self.sf
    def clear(self):
        for w in self.sf.winfo_children(): w.destroy()


def _wheel_to_frame_under_pointer(event):
    """Scroll the frame the pointer is actually over.

    bind_all is application-global and there is one of these frames per tab,
    so a binding that closes over a particular canvas points at whichever
    frame was built last, and at a destroyed widget once a tab is re-rendered.
    Walking up from the event widget keeps the binding free of any canvas
    reference, so re-registering it is a no-op rather than a leak.
    """
    w = getattr(event, "widget", None)
    for _ in range(64):
        if w is None or isinstance(w, str):
            return
        if isinstance(w, ScrollableCanvasFrame):
            try:
                w.canvas.yview_scroll(-3 if event.num == 4 else 3, "units")
            except tk.TclError:
                pass
            return
        w = getattr(w, "master", None)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main Window
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RegionAnalysisWindow(tk.Toplevel):
    AI_COLS = ['Condition', 'Tissue', 'Treatment', 'Age']
    # Known non-label columns to skip
    _SKIP_COLS = {'GSM', 'gsm', '_platform', 'series_id', 'title', 'source_name_ch1',
                  'organism_ch1', 'characteristics_ch1'}

    def __init__(self, parent, app_ref, regions_data, mode="analyze", platform_labels_df=None):
        super().__init__(parent)
        self.app = app_ref
        self.regions = regions_data
        self.mode = mode
        self.platform_labels_df = platform_labels_df  # full platform labels for enrichment
        self.figs = {}          # key -> fig
        self.canvases = {}      # key -> FigureCanvasTkAgg
        self.toolbars = {}      # key -> NavigationToolbar2Tk
        self._stale = set()     # tab names awaiting a re-render
        self._timers = set()

        # state
        self.plot_mode = tk.StringVar(value="both")
        self.gse_scope = tk.StringVar(value="selected")
        self.color_column = tk.StringVar(value="")
        self.ai_label_col = tk.StringVar(value="")
        self.cmp_col = tk.StringVar(value="")     # region-comparison label column
        self.cmp_gene = tk.StringVar(value="")    # gene pooled across platforms
        self._cmp_result = None                   # last comparison, for the AI read
        self._enrich_rows = []                    # last enrichment rows, for Summary
        self.ml_gene = tk.StringVar(value="")       # label-ML stratify gene
        self.ml_color_by = tk.StringVar(value="cluster")  # cluster embedding colour
        self.merge_regions = tk.BooleanVar(value=False)
        self.overlay = tk.BooleanVar(value=False)
        self.filter_values = set()  # which values are selected in the filter listbox

        n = len(regions_data)
        t = (f"Region Analysis ({n} region{'s' if n > 1 else ''})" if mode == "analyze"
             else f"Region Comparison ({n} regions)")
        self.title(t)
        self.geometry("1700x1050")
        try:
            _sw, _sh = self.winfo_screenwidth(), self.winfo_screenheight()
            _w, _h = min(1700, int(_sw * 0.92)), min(1050, int(_sh * 0.92))
            self.geometry(f"{_w}x{_h}+{(_sw-_w)//2}+{(_sh-_h)//2}")
            self.minsize(900, 600)
        except Exception: pass

        try:
            self._install_styles()

            self._mc = {}
            self._mc_total = {}
            self._log("Precomputing data...")
            self._precompute()
            self._log("Building UI...")
            self._build_ui()
            self._log("Scheduling render...")
            self.after(200, self._render_all)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self._log(f"INIT ERROR: {e}")
            print(f"[RegionAnalysis INIT ERROR]\n{tb}")
            ttk.Label(self, text=f"Initialization Error:\n\n{e}\n\nCheck terminal for full traceback.",
                      foreground="red", font=("Consolas", 10), wraplength=800).pack(pady=40)

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        # Force window to front
        self.lift()
        self.focus_force()
        self.attributes('-topmost', True)
        self.after(500, lambda: self.attributes('-topmost', False))

    def after(self, ms, func=None, *args):
        """Record the timer so ``_on_close`` can disarm it."""
        tid = super().after(ms, func, *args)
        if func is not None:
            self._timers.add(tid)
        return tid

    def after_cancel(self, id):
        self._timers.discard(id)
        return super().after_cancel(id)

    def _install_styles(self):
        """Named ttk styles for this window, all drawn from the AERO palette.

        ``ensure_theme`` installs the shared app theme first, so this window
        looks like the rest of the app even when it is the only one open -
        relying on the main window having run left it on bare clam. The rest
        are the extra roles this window needs (section headings, explanatory
        hints, metric readouts). Keeping them as named styles rather than
        per-widget colours is what stops this window drifting away.
        """
        ensure_theme(self)
        style_window(self)
        s = ttk.Style(self)
        s.configure('Section.TLabel', foreground=AERO['accent_dark'],
                    font=(UI_FONT, 11, 'bold'))
        s.configure('Sub.TLabel', foreground=AERO['text'],
                    font=(UI_FONT, 10, 'bold'))
        s.configure('Field.TLabel', foreground=AERO['text'],
                    font=(UI_FONT, 9, 'bold'))
        s.configure('Hint.TLabel', foreground=AERO['muted'],
                    font=(UI_FONT, 9, 'italic'))
        s.configure('Footnote.TLabel', foreground=AERO['muted'],
                    font=(UI_FONT, 8, 'italic'))
        s.configure('Empty.TLabel', foreground=AERO['muted'],
                    font=(UI_FONT, 11))
        s.configure('Metric.TLabel', foreground=AERO['text'],
                    font=(MONO_FONT, 9))
        s.configure('MetricStrong.TLabel', foreground=AERO['accent_dark'],
                    font=(MONO_FONT, 9, 'bold'))
        s.configure('Caution.TLabel', foreground=AERO['warn'],
                    font=(UI_FONT, 9, 'bold'))
        s.configure('Error.TLabel', foreground=AERO['danger'],
                    font=(UI_FONT, 10))
        s.configure('Card.TLabelframe', background=AERO['panel'],
                    bordercolor=AERO['border'], relief='solid', borderwidth=1)
        s.configure('Card.TLabelframe.Label', background=AERO['panel'],
                    foreground=AERO['accent_dark'], font=(UI_FONT, 9, 'bold'))

    def _log(self, msg):
        """Log to both terminal and GUI log."""
        full = f"[Region Analysis] {msg}"
        print(full)
        try: self.app.enqueue_log(full)
        except: pass

    def _value_axis_label(self, region=None):
        """Axis label for the regions' platform, generic when they disagree."""
        get = getattr(self.app, 'platform_measurement_label', None)
        if get is None:
            return "expression"
        plats = ({str(region.get('platform', ''))} if region is not None
                 else {str(r.get('platform', '')) for r in self.regions})
        try:
            labels = {get(p) for p in plats if p}
        except Exception:
            return "expression"
        return labels.pop() if len(labels) == 1 else "expression"

    # ── Precompute merged dataframes ────────────────────────────────
    def _precompute(self):
        # Platform-wide labels (from loaded file) - covers ALL GSMs
        # IMPORTANT: use reference only - no .copy() to save memory
        plat_lbl = self.platform_labels_df
        plat_lbl_slim = None
        if plat_lbl is not None and not plat_lbl.empty:
            if 'GSM' not in plat_lbl.columns:
                for c in plat_lbl.columns:
                    if c.lower() == 'gsm':
                        plat_lbl = plat_lbl.rename(columns={c: 'GSM'})
                        break
            if 'GSM' in plat_lbl.columns:
                # Normalize GSMs for reliable matching
                plat_lbl = plat_lbl.copy()
                plat_lbl['GSM'] = plat_lbl['GSM'].astype(str).str.strip().str.upper()
                # Backward compat: strip Classified_ prefix from old label files
                strip_rename = {c: c.replace('Classified_', '', 1)
                                for c in plat_lbl.columns if c.startswith('Classified_')}
                if strip_rename:
                    plat_lbl = plat_lbl.rename(columns=strip_rename)
                # Derive the kind column here rather than trusting that it was
                # derived upstream. ``semantic_label_columns`` counts it among
                # the labels - whether a Tissue resolved to anatomy or to a
                # catalogued cell line is a fact of its own - and the
                # assistant's label resolver derives it before selecting
                # columns. A window that selects first and never derives tests
                # one column fewer than the assistant does on the same file,
                # which is enough to change the size of the correction grid and
                # so every q value in the table. It is idempotent, so calling
                # it on a frame that already carries the column costs nothing.
                label_entities.add_kind_columns(plat_lbl)
                # One value column per extracted field, so the identifiers and
                # the earlier extraction passes never reach a test.
                lbl_only = ['GSM'] + [c for c in platform_label_cols(plat_lbl)
                                      if c != 'GSM']
                plat_lbl_slim = plat_lbl[lbl_only].drop_duplicates('GSM')
                self._log(f"Platform labels: {len(plat_lbl_slim):,} GSMs, "
                          f"cols={[c for c in lbl_only if c != 'GSM']}")
            else:
                plat_lbl = None

        # Kept as the window's own label namespace. The region frames carry
        # these semantic names ("Tissue", not "final_Tissue"), so anything
        # asked to work on a label name the window offers has to be handed
        # this frame rather than the raw file it was slimmed from.
        self._plat_lbl_slim = plat_lbl_slim

        for r in self.regions:
            col = r['column']
            bg = r.get('platform_df', pd.DataFrame())
            meta = r.get('meta_df', pd.DataFrame())
            ai = r.get('ai_labels_df', pd.DataFrame())
            gsms = set(str(g).strip().upper() for g in r['gsm_list'])

            # Normalize GSMs in expression data for reliable label matching
            if not bg.empty and 'GSM' in bg.columns:
                bg = bg.copy()
                bg['GSM'] = bg['GSM'].astype(str).str.strip().str.upper()

            bg_shape = f"{bg.shape}" if not bg.empty else "EMPTY"
            has_gsm = 'GSM' in bg.columns if not bg.empty else False
            self._log(f"Region '{r['label']}': col={col}, bg={bg_shape}, "
                      f"gsms={len(gsms)}, has_GSM={has_gsm}")

            if bg.empty or 'GSM' not in bg.columns:
                self._mc[r['label']] = pd.DataFrame()
                self._mc_total[r['label']] = pd.DataFrame()
                self._log(f"[!] Region '{r['label']}': no platform data or no GSM column!")
                continue

            try:
                # SELECTED: only GSMs in this region, carrying the same
                # columns as TOTAL. Cut down to GSM + expression it could not
                # be grouped or filtered by study, so the picker moved the
                # whole-platform view and left the selected one alone.
                sub = bg[bg['GSM'].isin(gsms)].copy()
                sub = self._merge_meta_ai(sub, meta, ai)
                if plat_lbl_slim is not None:
                    merge_cols = [c for c in plat_lbl_slim.columns if c != 'GSM']
                    if merge_cols:
                        # Drop any label columns that already exist (prevent _x/_y)
                        drop_existing = [c for c in merge_cols if c in sub.columns]
                        if drop_existing:
                            sub = sub.drop(columns=drop_existing)
                        sub = sub.merge(plat_lbl_slim[['GSM'] + merge_cols].drop_duplicates('GSM'),
                                        on='GSM', how='left')
                        n_sel_matched = sub[merge_cols[0]].notna().sum()
                        self._log(f"Selected labels: {n_sel_matched}/{len(sub)} GSMs matched")
                self._mc[r['label']] = sub
                self._log(f"OK Selected: {sub.shape[0]} rows, "
                          f"labels={[c for c in sub.columns if c not in self._SKIP_COLS and c != 'GSM' and c != col]}")

                # TOTAL: ALL GSMs - keep all available metadata columns
                total = bg.copy()

                # ALWAYS merge platform-wide labels with simple left join
                if plat_lbl_slim is not None:
                    lbl_cols = [c for c in plat_lbl_slim.columns if c != 'GSM']
                    if lbl_cols:
                        # Drop any label columns that already exist (prevent _x/_y)
                        drop_existing = [c for c in lbl_cols if c in total.columns]
                        if drop_existing:
                            total = total.drop(columns=drop_existing)
                            self._log(f"Dropped existing label cols for re-merge: {drop_existing}")
                        total = total.merge(
                            plat_lbl_slim[['GSM'] + lbl_cols].drop_duplicates('GSM'),
                            on='GSM', how='left')
                        first = total[lbl_cols[0]]
                        if getattr(first, "ndim", 1) > 1:
                            first = first.iloc[:, 0]
                        n_matched = int(first.notna().sum())
                        self._log(f"Total labels merged: {n_matched:,}/{len(total):,} "
                                  f"GSMs matched, cols={lbl_cols}")

                total = self._merge_meta_ai(total, meta, ai)
                self._mc_total[r['label']] = total

                n_lbl = sum(1 for c in total.columns
                            if c not in self._SKIP_COLS and c != 'GSM' and c != col
                            and total[c].dtype == 'object' and total[c].notna().sum() > 0)
                lbl_detail = [(c, total[c].notna().sum(), total[c].nunique())
                              for c in total.columns
                              if c not in self._SKIP_COLS and c != 'GSM' and c != col
                              and total[c].dtype == 'object']
                self._log(f"OK Total: {total.shape[0]:,} rows, {n_lbl} label cols with data")
                self._log(f"   Total columns: {list(total.columns)}")
                for lc, nn, nu in lbl_detail:
                    self._log(f"   Label '{lc}': {nn:,} non-null, {nu} unique")
            except Exception as e:
                import traceback
                self._log(f"ERROR in precompute: {e}")
                print(traceback.format_exc())
                self._mc[r['label']] = pd.DataFrame()
                self._mc_total[r['label']] = pd.DataFrame()

    @staticmethod
    def _merge_meta_ai(df, meta, ai):
        """Merge metadata + AI labels onto a GSM+expression df.
        Only adds columns not already present to avoid _x/_y duplicates."""
        if not meta.empty:
            mc = 'gsm' if ('gsm' in meta.columns and 'GSM' not in meta.columns) else 'GSM'
            ms = meta.rename(columns={mc: 'GSM'}) if mc != 'GSM' else meta
            kp = ['GSM'] + [c for c in ms.columns if c != 'GSM' and c not in df.columns]
            if len(kp) > 1:
                df = df.merge(ms[kp].drop_duplicates('GSM'), on='GSM', how='left')
        if not ai.empty:
            ac = 'GSM' if 'GSM' in ai.columns else 'gsm'
            ais = ai.rename(columns={ac: 'GSM'}) if ac != 'GSM' else ai
            cls = ['GSM'] + [c for c in ais.columns
                             if c not in ('GSM', 'gsm') and c not in df.columns]
            if len(cls) > 1:
                df = df.merge(ais[cls].drop_duplicates('GSM'), on='GSM', how='left')
        return df

    def _gse_map(self):
        """GSM -> study id, pooled over every region's platform frame.

        Returns None when no study column exists anywhere, so the enrichment
        tab degrades to "unknown" rather than pretending samples are
        independent draws.
        """
        cached = getattr(self, '_gse_map_cache', '__unset__')
        if cached != '__unset__':
            return cached
        out = {}
        for src in (self._mc_total, self._mc):
            for df in src.values():
                if df is None or df.empty or 'GSM' not in df.columns:
                    continue
                col = next((c for c in ('series_id', 'gse', 'GSE', 'series')
                            if c in df.columns), None)
                if col is None:
                    continue
                sub = df[['GSM', col]].dropna()
                for g, s in zip(sub['GSM'].astype(str).str.upper(),
                                sub[col].astype(str)):
                    out.setdefault(g, s)
        self._gse_map_cache = out or None
        return self._gse_map_cache

    # ── UI Layout ───────────────────────────────────────────────────
    def _build_ui(self):
        # ── TOP CONTROLS BAR - ROW 1: Scope buttons (large & prominent) ──
        bar_bg = AERO['bg_top']
        scope_bar = tk.Frame(self, bg=bar_bg, pady=6)
        scope_bar.pack(fill=tk.X)

        tk.Label(scope_bar, text="View", font=(UI_FONT, 10, 'bold'),
                 fg=AERO['muted'], bg=bar_bg).pack(side=tk.LEFT, padx=(12, 6))

        self._scope_btn_selected = _flat_button(
            scope_bar, "Selected Region", lambda: self._set_scope("selected"),
            fill=AERO['accent'], font=(UI_FONT, 10, 'bold'), padx=16, pady=6)
        self._scope_btn_selected.pack(side=tk.LEFT, padx=3)

        self._scope_btn_total = _flat_button(
            scope_bar, "All Loaded Samples", lambda: self._set_scope("total"),
            font=(UI_FONT, 10, 'bold'), padx=16, pady=6)
        self._scope_btn_total.pack(side=tk.LEFT, padx=3)

        ttk.Separator(scope_bar, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=10)

        self._overlay_btn = _flat_button(
            scope_bar, "+ Loaded Background", self._toggle_overlay,
            font=(UI_FONT, 9, 'bold'), padx=12, pady=5)
        self._overlay_btn.pack(side=tk.LEFT, padx=3)

        if len(self.regions) > 1:
            self._merge_btn = _flat_button(
                scope_bar, "Merge All Regions", self._toggle_merge,
                font=(UI_FONT, 9, 'bold'), padx=12, pady=5)
            self._merge_btn.pack(side=tk.LEFT, padx=3)

        # Color By (right side of scope bar)
        ttk.Separator(scope_bar, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=10)
        tk.Label(scope_bar, text="Color by", font=(UI_FONT, 9, 'bold'),
                 fg=AERO['muted'], bg=bar_bg).pack(side=tk.LEFT, padx=(4, 4))
        opts = self._get_color_cols()
        self.color_column.set(opts[0] if opts else "(none)")
        self.cc = ttk.Combobox(scope_bar, textvariable=self.color_column,
                               values=opts, width=22, state='readonly',
                               font=(UI_FONT, 10))
        self.cc.pack(side=tk.LEFT, padx=4)
        self.cc.bind("<<ComboboxSelected>>", lambda e: self._on_color_col_changed())

        ttk.Separator(scope_bar, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=10)

        for text, cmd, fill in [
            ("Multi-Label Query", self._open_multi_label_query, AERO['accent_dark']),
            ("Refresh", self._refresh_labels_from_app, None),
            ("Curate Labels (LLM)", lambda: self.app._open_llm_curator(), AERO['warn']),
        ]:
            _flat_button(scope_bar, text, cmd, fill=fill,
                         font=(UI_FONT, 9, 'bold'), padx=10, pady=5
                         ).pack(side=tk.LEFT, padx=3)

        # Button descriptions (tooltip-like, below scope bar)
        phase_info = tk.Frame(self, bg=bar_bg)
        phase_info.pack(fill=tk.X)
        tk.Label(phase_info,
            text="Refresh = reload labels after background processing",
            font=(UI_FONT, 8), fg=AERO['muted'], bg=bar_bg).pack(anchor=tk.W, padx=12,
                                                                 pady=(0, 4))

        # ── ROW 2: Plot Mode buttons ──
        mode_bg = AERO['panel_bot']
        mode_bar = tk.Frame(self, bg=mode_bg, pady=4)
        mode_bar.pack(fill=tk.X)

        tk.Label(mode_bar, text="Plot", font=(UI_FONT, 9, 'bold'),
                 fg=AERO['muted'], bg=mode_bg).pack(side=tk.LEFT, padx=(12, 6))

        self._mode_btns = {}
        for val, label in [("density", "Density"), ("rug", "Rug"), ("both", "Both")]:
            btn = _flat_button(mode_bar, label, lambda v=val: self._set_plot_mode(v),
                               font=(UI_FONT, 9, 'bold'), padx=14, pady=4)
            _paint_toggle(btn, val == "both")
            btn.pack(side=tk.LEFT, padx=2)
            self._mode_btns[val] = (btn, AERO['accent'])

        # Region info
        ttk.Separator(mode_bar, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=10)
        parts = [f"{r['label']} ({self._tech_label(r.get('platform')) or 'unknown'}) "
                 f"[{r['range'][0]:.2f}-{r['range'][1]:.2f}] n={len(r['gsm_list'])}"
                 for r in self.regions]
        tk.Label(mode_bar, text="   ".join(parts), font=(UI_FONT, 9),
                 fg=AERO['text'], bg=mode_bg,
                 wraplength=700).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # ── PROGRESS BAR (for rendering / extraction operations) ──
        self._prog_frame = tk.Frame(self, bg=mode_bg)
        self._prog_frame.pack(fill=tk.X, padx=6, pady=(2, 0))

        self._prog_bar = ttk.Progressbar(
            self._prog_frame, mode='determinate', length=400,
            style='Accent.Horizontal.TProgressbar')
        self._prog_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        self._prog_label = tk.Label(
            self._prog_frame, text="Ready", font=(MONO_FONT, 9),
            fg=AERO['muted'], bg=mode_bg, anchor='w', width=50)
        self._prog_label.pack(side=tk.LEFT)

        self._prog_pct = tk.Label(
            self._prog_frame, text="", font=(MONO_FONT, 9, 'bold'),
            fg=AERO['accent_dark'], bg=mode_bg, width=6, anchor='e')
        self._prog_pct.pack(side=tk.RIGHT, padx=(0, 4))

        # ── MAIN SPLIT: left panel (filter) + right (notebook) ──
        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

        # ╔══════════════════════════════════════════════════════════╗
        # ║  LEFT: Fancy Filter Panel                                ║
        # ╚══════════════════════════════════════════════════════════╝
        left = ttk.Frame(main_pane, width=300)
        main_pane.add(left, weight=0)

        # Header
        hdr = ttk.Frame(left)
        hdr.pack(fill=tk.X, padx=4, pady=(6, 2))
        ttk.Label(hdr, text="Filter Values", style='Section.TLabel').pack(side=tk.LEFT)
        self.filter_count_lbl = ttk.Label(hdr, text="", font=(UI_FONT, 8),
                                          foreground=AERO['accent'])
        self.filter_count_lbl.pack(side=tk.RIGHT)

        # Search box
        search_frame = ttk.Frame(left)
        search_frame.pack(fill=tk.X, padx=4, pady=(2, 4))
        ttk.Label(search_frame, text="Search", font=(UI_FONT, 8),
                  foreground=AERO['muted']).pack(side=tk.LEFT)
        self.filter_search_var = tk.StringVar()
        self.filter_search_var.trace_add('write', lambda *a: self._filter_search_changed())
        search_entry = ttk.Entry(search_frame, textvariable=self.filter_search_var, width=18)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(search_frame, text="\u2715", width=2, style="Secondary.TButton",
                   command=lambda: self.filter_search_var.set("")).pack(side=tk.RIGHT)

        # Buttons
        btn_frame = ttk.Frame(left); btn_frame.pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(btn_frame, text="All", width=6, style="Secondary.TButton",
                   command=self._select_all_filter).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="None", width=6, style="Secondary.TButton",
                   command=self._select_none_filter).pack(side=tk.LEFT, padx=2)
        ttk.Label(btn_frame, text="Top", font=(UI_FONT, 9),
                  foreground=AERO['muted']).pack(side=tk.LEFT, padx=(6, 1))
        self._top_n_var = tk.StringVar(value="10")
        top_n_entry = ttk.Entry(btn_frame, textvariable=self._top_n_var, width=4,
                                font=(UI_FONT, 9))
        top_n_entry.pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_frame, text="\u25b8", width=3, style="Secondary.TButton",
                   command=self._select_topN_filter).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Apply", width=7, style="Primary.TButton",
                   command=self._refresh_plots).pack(side=tk.RIGHT, padx=2)

        # Treeview with checkboxes
        tree_frame = ttk.Frame(left)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(2, 4))

        self.filter_tree = ttk.Treeview(
            tree_frame, columns=("check", "value", "platform", "count"),
            show="headings", selectmode="extended", height=22)
        self.filter_tree.heading("check", text="\u2713")
        self.filter_tree.heading("value", text="Value")
        self.filter_tree.heading("platform", text="Platform")
        self.filter_tree.heading("count", text="n")
        # Match the app-wide table convention: centre-aligned columns (every
        # other Treeview uses anchor='center'). Right-aligning the count made
        # its digits overflow past the column separator line.
        self.filter_tree.column("check", width=30, anchor="center", stretch=False)
        self.filter_tree.column("value", width=118, minwidth=80, anchor="center")
        self.filter_tree.column("platform", width=86, minwidth=60, anchor="center")
        self.filter_tree.column("count", width=46, minwidth=40, anchor="center", stretch=False)
        ftv_sb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.filter_tree.yview)
        self.filter_tree.configure(yscrollcommand=ftv_sb.set)
        self.filter_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ftv_sb.pack(side=tk.RIGHT, fill=tk.Y)

        # Click row -> toggle check
        self.filter_tree.bind("<ButtonRelease-1>", self._on_filter_tree_click)
        # Store check state: {value_str: bool}
        self._filter_checks = {}
        # Store all filter data: [(value, count)]
        self._filter_all_items = []
        # {value: {platform, ...}} - which platform each value was seen on
        self._filter_value_plats = {}

        # Info label
        self.filter_info = ttk.Label(left, text="Click rows to toggle, then Apply",
                                     style='Footnote.TLabel')
        self.filter_info.pack(pady=(0, 4))

        # RIGHT: Notebook
        right = ttk.Frame(main_pane)
        main_pane.add(right, weight=1)

        self.nb = ttk.Notebook(right, padding=2); self.nb.pack(fill=tk.BOTH, expand=True)

        # Tab 1: GSE Distributions (was Tab 2, Expression tab removed as redundant)
        self.t_gse = ttk.Frame(self.nb); self.nb.add(self.t_gse, text=" Distributions ")

        # Tab 3: AI Labels
        self.t_ai = ttk.Frame(self.nb); self.nb.add(self.t_ai, text=" Labels ")
        # AI label selector inside tab
        ai_ctrl = ttk.Frame(self.t_ai); ai_ctrl.pack(fill=tk.X, padx=5, pady=3)
        ttk.Label(ai_ctrl, text="Label column:", style='Field.TLabel').pack(side=tk.LEFT)
        ai_opts = self._get_ai_cols()
        self.ai_label_col.set(self._default_label_col(ai_opts))
        self.ai_combo = ttk.Combobox(ai_ctrl, textvariable=self.ai_label_col,
                                      values=ai_opts, width=24, state='readonly')
        self.ai_combo.pack(side=tk.LEFT, padx=6)
        self.ai_combo.bind("<<ComboboxSelected>>", lambda e: self._render_ai_tab())
        self.ai_scroll = ScrollableCanvasFrame(self.t_ai)
        self.ai_scroll.pack(fill=tk.BOTH, expand=True)

        # Tab 4: Frequency Analysis
        self.t_freq = ttk.Frame(self.nb); self.nb.add(self.t_freq, text=" Frequency ")

        # Tab 5: Fisher Enrichment
        self.t_enrich = ttk.Frame(self.nb); self.nb.add(self.t_enrich, text=" Enrichment ")

        # Tab 5b: Region Comparison - the regions against each other, on demand
        self.t_cmp = ttk.Frame(self.nb); self.nb.add(self.t_cmp, text=" Comparison ")

        # Tab 6: Statistics
        self.t_stats = ttk.Frame(self.nb); self.nb.add(self.t_stats, text=" Statistics ")
        self.st = None

        # Tab 7: Samples
        self.t_table = ttk.Frame(self.nb); self.nb.add(self.t_table, text=" Samples ")

        # Tab 8: Summary - one page per region, its results around its histogram
        self.t_summary = ttk.Frame(self.nb); self.nb.add(self.t_summary, text=" Summary ")

        # A toolbar change invalidates every tab but only redraws the visible
        # one, so opening a tab is where the rest catch up.
        self.nb.bind("<<NotebookTabChanged>>", self._render_current_tab, add="+")

        # Bottom
        bot = ttk.Frame(self, padding=5); bot.pack(fill=tk.X)
        ttk.Button(bot, text="Export All", style="Action.TButton", command=self._export).pack(side=tk.RIGHT, padx=5)
        ttk.Button(bot, text="Close", style="Secondary.TButton", command=self._on_close).pack(side=tk.RIGHT, padx=5)
        ttk.Button(bot, text="Display limits…", style="Secondary.TButton",
                   command=self._open_display_limits).pack(side=tk.LEFT, padx=5)

    def _open_display_limits(self):
        """How much of each result this window draws, set by the user.

        Every tab is marked stale afterwards rather than redrawn on the spot:
        the setting changes what a chart contains, and a tab the user is not
        looking at costs nothing until they open it.
        """
        from genevariate.gui import display_limits

        def _redraw():
            self._stale = {spec[0] for spec in self._tab_specs()}
            self._render_current_tab()
            self._log("[Display limits] changed; tabs will redraw")

        display_limits.open_dialog(self, on_change=_redraw)

    # ── Column detection ────────────────────────────────────────────
    def _get_color_cols(self):
        cs = set()
        for r in self.regions:
            # Check BOTH selected and total data for available columns
            for source in [self._mc, self._mc_total]:
                m = source.get(r['label'], pd.DataFrame())
                if m.empty: continue
                for c in m.columns:
                    if c in ('GSM', r['column']): continue
                    if m[c].dtype == 'object' or c == 'series_id':
                        nuniq = m[c].nunique()
                        if 1 < nuniq <= 500: cs.add(c)
        # Priority: series_id first, then known label columns
        known_labels = ['Condition', 'Tissue', 'Treatment', 'Age']
        lbl_found = sorted(c for c in cs if c in known_labels)
        pri = ['series_id'] + lbl_found
        o = [c for c in pri if c in cs] + sorted(c for c in cs if c not in pri)
        return o if o else ["(none)"]

    def _get_ai_cols(self):
        found = set()
        skip = self._SKIP_COLS | {'GSM'}
        for r in self.regions:
            m = self._mc.get(r['label'], pd.DataFrame())
            if not m.empty:
                for c in m.columns:
                    if c not in skip and c != r.get('column', '') and m[c].dtype == 'object' and m[c].notna().sum() > 0:
                        found.add(c)
            # Also check _mc_total for more columns
            t = self._mc_total.get(r['label'], pd.DataFrame())
            if not t.empty:
                for c in t.columns:
                    if c not in skip and c != r.get('column', '') and t[c].dtype == 'object' and t[c].notna().sum() > 0:
                        found.add(c)
        return sorted(found) if found else ["(none)"]

    def _label_col_coverage(self, col):
        """How much of the analysed regions *col* actually covers.

        Returns ``(regions_carrying_it, non_null_rows, distinct_values)``.
        """
        regions = rows = uniq = 0
        for r in self.regions:
            best = 0
            for source in (self._mc, self._mc_total):
                m = source.get(r['label'], pd.DataFrame())
                if m.empty or col not in m.columns:
                    continue
                nn = int(m[col].notna().sum())
                if nn > best:
                    best = nn
                    uniq = max(uniq, int(m[col].nunique()))
            if best:
                regions += 1
                rows += best
        return regions, rows, uniq

    def _default_label_col(self, cols):
        """The column to open a label tab on: the one the data best supports.

        A region set can span platforms that were annotated separately, so the
        picker lists the union of their columns and some of them exist for only
        one platform. Opening on whichever name sorts first leaves the other
        panels with nothing to draw, which reads as a broken tab rather than as
        a column that does not apply there. Ranking by measured coverage instead
        keeps the choice with the data and works for any label vocabulary.
        """
        real = [c for c in (cols or []) if c and c != "(none)"]
        if not real:
            return ""
        scored = [(self._label_col_coverage(c), c) for c in real]
        scored.sort(key=lambda t: (-t[0][0], -t[0][1], t[1]))
        return scored[0][1]

    # ── Filter Treeview management ──────────────────────────────────
    def _populate_filter_listbox(self):
        """Populate filter treeview with unique values from the current Color By column."""
        self.filter_tree.delete(*self.filter_tree.get_children())
        self._filter_checks.clear()
        self._filter_all_items.clear()
        ccol = self.color_column.get()
        self._filter_column = None if (not ccol or ccol == "(none)") else ccol
        if not ccol or ccol == "(none)":
            self.filter_count_lbl.config(text="")
            return

        # Use TOTAL data if scope is total, else selected
        scope = self.gse_scope.get()
        source = self._mc_total if scope == "total" else self._mc

        # Gather all values + counts across regions, and the platform each value
        # was seen on. Regions can span platforms, and the identifier namespaces
        # do not: a GEO platform contributes GSE accessions while a CELLxGENE
        # arm contributes dataset UUIDs. Pooled into one list with no attribution
        # the UUIDs read as corrupt data, and there is no way to tell which study
        # belongs to which platform before choosing what to analyse.
        freq = {}
        plats = {}
        for r in self.regions:
            m = source.get(r['label'], pd.DataFrame())
            if not m.empty and ccol in m.columns:
                plat = str(r.get('platform') or "?")
                for v, c in m[ccol].fillna("N/A").astype(str).value_counts().items():
                    freq[v] = freq.get(v, 0) + c
                    plats.setdefault(v, set()).add(plat)
        self._filter_value_plats = plats

        sorted_items = sorted(freq.items(), key=lambda x: -x[1])
        self._filter_all_items = sorted_items

        # All checked by default
        for val, cnt in sorted_items:
            self._filter_checks[val] = True

        self._render_filter_tree()
        total_checked = sum(1 for v in self._filter_checks.values() if v)
        self.filter_count_lbl.config(text=f"{total_checked}/{len(sorted_items)} selected")

    def _filter_value_colors(self):
        """Colour each value as the Distributions plot does, so the picker doubles as a legend."""
        items = [v for v, _ in self._filter_all_items]
        tops = [v for v in items if self._filter_checks.get(v, False)][:_lim('groups')]
        colors = _clrs(max(1, len(tops)))
        return {v: colors[i] for i, v in enumerate(tops) if i < len(colors)}

    def _cell_line_values(self, column):
        """Values of *column* that name a catalogued cell line, not a tissue.

        Only ever non-empty for the Tissue column of a normalized label file:
        the extractor refuses a Cellosaurus identifier in any other field, so
        there is nowhere else the distinction could come from.
        """
        df = self.platform_labels_df
        if not column or df is None or getattr(df, 'empty', True):
            return set()
        cols = label_entities.field_columns(df.columns,
                                            label_entities.CELL_LINE_FIELD)
        if cols.get('value') != column:
            return set()
        cached = getattr(self, '_cell_line_cache', None)
        if cached is None or cached[0] != column:
            self._cell_line_cache = (column,
                                     label_entities.cell_line_values(df))
        return self._cell_line_cache[1]

    def _render_filter_tree(self):
        """Render the filter treeview items (respecting search filter)."""
        # Closing the window clears the search box, and clearing it fires the
        # trace that lands here - by which time the table it would redraw has
        # already been destroyed.
        if not self.filter_tree.winfo_exists():
            return
        self.filter_tree.delete(*self.filter_tree.get_children())
        search = self.filter_search_var.get().strip().lower() if hasattr(self, 'filter_search_var') else ""

        cmap = self._filter_value_colors()
        used_tags = set()
        # Rows show a truncated value, so the row itself has to remember which
        # value it stands for -- matching a click back by display text made two
        # values sharing a 40-character prefix toggle each other.
        self._filter_row_val = {}
        # A value that resolved to a catalogued cell line is said so outright.
        # It reads as a tissue and is not one, and this list is where the user
        # decides which values an analysis will be run on.
        cells = self._cell_line_values(getattr(self, '_filter_column', None))
        for val, cnt in self._filter_all_items:
            if search and search not in val.lower():
                continue
            checked = self._filter_checks.get(val, False)
            chk = "\u2611" if checked else "\u2610"
            shown = (f"{_tr(val, 20)}  \u00b7 cell line" if val in cells
                     else _tr(val, 30))
            src = sorted(getattr(self, '_filter_value_plats', {}).get(val, ()))
            plat = (_tr(src[0], 14) if len(src) == 1
                    else f"{len(src)} platforms" if src else "")
            iid = self.filter_tree.insert("", tk.END,
                                          values=(chk, shown, plat, cnt))
            self._filter_row_val[iid] = val
            # Checked values that appear in the plot get that plot's exact
            # colour (picker acts as the legend); anything else is muted grey.
            clr = cmap.get(val) if checked else None
            if clr:
                tag = "clr_" + clr.lstrip('#')
                if tag not in used_tags:
                    self.filter_tree.tag_configure(tag, foreground=clr)
                    used_tags.add(tag)
                self.filter_tree.item(iid, tags=(tag,))
            elif checked:
                self.filter_tree.item(iid, tags=("checked",))
            else:
                self.filter_tree.item(iid, tags=("unchecked",))

        # Checked-but-not-plotted rows use the standard navy text; deselected
        # rows are dimmed to the app's standard muted grey.
        self.filter_tree.tag_configure("checked", foreground=AERO['text'])
        self.filter_tree.tag_configure("unchecked", foreground=AERO['muted'])

    def _on_filter_tree_click(self, event):
        """Toggle checkbox on row click."""
        item = self.filter_tree.identify_row(event.y)
        if not item:
            return
        val = getattr(self, "_filter_row_val", {}).get(item)
        if val is None:
            return
        # Default False, matching what the row was drawn as -- defaulting to
        # True here made a click on an unknown row leave it unchecked, i.e.
        # do nothing.
        self._filter_checks[val] = not self._filter_checks.get(val, False)

        self._render_filter_tree()
        total_checked = sum(1 for v in self._filter_checks.values() if v)
        self.filter_count_lbl.config(text=f"{total_checked}/{len(self._filter_all_items)} selected")

    def _filter_search_changed(self):
        """Re-render treeview when search text changes."""
        self._render_filter_tree()

    def _get_selected_filter_values(self):
        """Get the set of values that are checked in the filter."""
        return {val for val, checked in self._filter_checks.items() if checked}

    def _apply_filter(self, df):
        """Rows of *df* whose filter-column value is still ticked.

        The picker chooses samples, not plotted categories: unticking a series
        has to remove it from every figure and every table alike, or an export
        holds a plot and a statistic computed over different samples.
        """
        col = getattr(self, "_filter_column", None)
        if (df is None or df.empty or not col or col not in df.columns
                or not self._filter_checks):
            return df
        if self.filter_values == set(self._filter_checks):
            return df
        return df[df[col].fillna("N/A").astype(str).isin(self.filter_values)]

    def _sel(self, label):
        """Selected-region samples for *label*, after the filter."""
        return self._apply_filter(self._mc.get(label, pd.DataFrame()))

    def _tot(self, label):
        """All loaded samples behind *label*'s region, after the filter."""
        return self._apply_filter(self._mc_total.get(label, pd.DataFrame()))

    _TECH_NAMES = {"microarray": "Microarray", "bulk-rna-seq": "RNA-seq",
                   "single-cell": "Single-cell", "methylation": "Methylation",
                   "sequencing-other": "Sequencing", "custom": "Custom"}

    def _tech_label(self, plat):
        """Human name for what *plat* measures, so merged views stay separable."""
        facts = getattr(self.app, '_platform_facts', None)
        if facts is None or not plat:
            return ""
        try:
            cat = facts(plat).get("category", "")
        except Exception:
            return ""
        return self._TECH_NAMES.get(cat, str(cat or "").title())

    def _sel_gsms(self, region):
        """GSMs of *region*'s selection that survive the filter."""
        df = self._sel(region.get('label'))
        if df is None or 'GSM' not in getattr(df, 'columns', []):
            return {str(g).strip().upper() for g in region.get('gsm_list', [])}
        return set(df['GSM'].astype(str).str.strip().str.upper())

    def _select_all_filter(self):
        for val in self._filter_checks:
            self._filter_checks[val] = True
        self._render_filter_tree()
        self.filter_count_lbl.config(text=f"{len(self._filter_checks)}/{len(self._filter_all_items)} selected")

    def _select_none_filter(self):
        for val in self._filter_checks:
            self._filter_checks[val] = False
        self._render_filter_tree()
        self.filter_count_lbl.config(text=f"0/{len(self._filter_all_items)} selected")

    def _select_topN_filter(self):
        """Select only top N by count (user-specified)."""
        try:
            n = int(self._top_n_var.get())
        except (ValueError, AttributeError):
            n = 10
        n = max(1, min(n, len(self._filter_all_items)))
        for val in self._filter_checks:
            self._filter_checks[val] = False
        for i, (val, _) in enumerate(self._filter_all_items):
            if i < n:
                self._filter_checks[val] = True
        self._render_filter_tree()
        total_checked = sum(1 for v in self._filter_checks.values() if v)
        self.filter_count_lbl.config(text=f"{total_checked}/{len(self._filter_all_items)} selected")

    def _open_multi_label_query(self):
        """Open a multi-label query builder to create a compound color filter.
        E.g., Tissue=Liver AND Condition=Cancer AND Age=50.
        Only matching samples get colored as a single group.
        """
        # Use the merged data from all regions
        all_dfs = []
        for lbl in self.regions:
            mc = self._mc_total.get(lbl['label'], pd.DataFrame())
            if not mc.empty:
                all_dfs.append(mc)
        if not all_dfs:
            messagebox.showinfo(
                "Multi-Label Query",
                "No samples are loaded for these regions yet.", parent=self)
            return

        df = pd.concat(all_dfs, ignore_index=True)

        # Available label columns. The grouping dialog and the assistant ask
        # this same question, so what counts as a label column is decided in
        # one place; only the extra GEO metadata this window shows elsewhere
        # is window-specific.
        from genevariate.core.analysis import queryable_columns
        label_cols = queryable_columns(df, skip=self._SKIP_COLS)
        if not label_cols:
            messagebox.showinfo(
                "Multi-Label Query",
                "There are no categorical label columns to query.\n\n"
                "Load a label file for this platform, or run label "
                "extraction, and try again.", parent=self)
            return

        dlg = tk.Toplevel(self)
        style_window(dlg)
        dlg.title("Multi-Label Query - Compound Color Filter")
        dlg.transient(self)
        dlg.grab_set()

        ttk.Label(dlg, text="Build a compound query - matching samples will be highlighted as one group",
                  font=('Segoe UI', 10, 'bold')).pack(padx=15, pady=(15, 5))
        ttk.Label(dlg, text="Example: Tissue=Brain AND Condition=Alzheimer Disease",
                  font=('Segoe UI', 9, 'italic'), foreground=AERO['muted']).pack(padx=15, pady=(0, 10))

        rows_frame = ttk.Frame(dlg)
        rows_frame.pack(fill=tk.X, padx=15, pady=5)
        query_rows = []

        def _add_row():
            row_frame = ttk.Frame(rows_frame)
            row_frame.pack(fill=tk.X, pady=3)
            if query_rows:
                ttk.Label(row_frame, text="AND", font=('Segoe UI', 9, 'bold'),
                          foreground=AERO['danger']).pack(side=tk.LEFT, padx=5)
            col_var = tk.StringVar(value=label_cols[0])
            col_cb = ttk.Combobox(row_frame, textvariable=col_var,
                                   values=label_cols, state='readonly', width=15)
            col_cb.pack(side=tk.LEFT, padx=5)
            ttk.Label(row_frame, text="=", font=('Segoe UI', 11, 'bold')).pack(side=tk.LEFT, padx=3)
            val_var = tk.StringVar()
            val_cb = ttk.Combobox(row_frame, textvariable=val_var, width=25)
            val_cb.pack(side=tk.LEFT, padx=5)

            def _on_col(e=None):
                c = col_var.get()
                if c and c in df.columns:
                    # Every value: anything past a cut is not selectable at
                    # all, and a label column may well have hundreds.
                    from genevariate.core.analysis import column_values
                    vals = column_values(df, c)
                    val_cb['values'] = vals
                    if vals: val_var.set(vals[0])
                _preview()

            col_cb.bind('<<ComboboxSelected>>', _on_col)
            val_cb.bind('<<ComboboxSelected>>', lambda e: _preview())
            _on_col()

            def _remove():
                query_rows.remove((col_var, val_var, row_frame))
                row_frame.destroy()
                _preview()

            _flat_button(row_frame, "\u2715", _remove, fill=AERO['panel'],
                         fg=AERO['danger'], font=(UI_FONT, 8, 'bold'),
                         padx=6, pady=2).pack(side=tk.LEFT, padx=5)
            query_rows.append((col_var, val_var, row_frame))

        preview_lbl = ttk.Label(dlg, text="", font=('Segoe UI', 9), foreground=AERO['accent_dark'])
        preview_lbl.pack(padx=15, pady=5)

        def _current_criteria():
            return [(cv.get(), vv.get()) for cv, vv, _ in query_rows]

        def _preview(*a):
            from genevariate.core.analysis import query_mask, describe_criteria
            crit = _current_criteria()
            n = int(query_mask(df, crit).sum())
            shown = describe_criteria(crit).replace(" AND ", "  AND  ")
            preview_lbl.config(text=f"{shown}  →  {n:,} samples")

        _flat_button(dlg, "+ Add Criterion", lambda: [_add_row(), _preview()],
                     fill=AERO['green_dark'], font=(UI_FONT, 9, 'bold'),
                     padx=12, pady=4).pack(anchor=tk.W, padx=15, pady=3)

        name_frame = ttk.Frame(dlg)
        name_frame.pack(fill=tk.X, padx=15, pady=5)
        ttk.Label(name_frame, text="Label:", font=('Segoe UI', 9, 'bold')).pack(side=tk.LEFT)
        name_var = tk.StringVar(value="")
        ttk.Entry(name_frame, textvariable=name_var, width=30).pack(side=tk.LEFT, padx=5)
        ttk.Label(name_frame, text="(leave empty = auto-fill with values)",
                  font=('Segoe UI', 8), foreground='gray').pack(side=tk.LEFT, padx=3)

        btn_frame = ttk.Frame(dlg)
        btn_frame.pack(fill=tk.X, padx=15, pady=(5, 15))

        def _apply():
            from genevariate.core.analysis import run_query
            res = run_query(df, _current_criteria(), skip=self._SKIP_COLS)
            mask = res.mask
            parts = [f"{c}={v}" for c, v in res.criteria]
            if not parts:
                messagebox.showinfo("Multi-Label Query",
                                    "Add at least one criterion first.",
                                    parent=dlg)
                return
            n = mask.sum()
            if n == 0:
                messagebox.showwarning("No Matches", "No samples match.", parent=dlg)
                return

            # Create a synthetic column with descriptive query name
            # Label = actual values joined: "Alzheimer Disease + Brain" (not "Query Match")
            query_values = " + ".join(c.split("=")[1].strip() if "=" in c else c for c in parts)
            query_col = " + ".join(c.split("=")[0].strip() for c in parts if "=" in c)
            if not query_col:
                query_col = "Query"
            # Use custom name if user typed one, otherwise use the values
            display_name = name_var.get().strip()
            if not display_name or display_name.lower() in ('query match', 'query', 'match'):
                display_name = query_values
            matched_gsms = (set(df.loc[mask, 'GSM'].astype(str).str.upper())
                            if 'GSM' in df.columns else set())
            for lbl in self.regions:
                for scope_df_key in [self._mc, self._mc_total]:
                    mc = scope_df_key.get(lbl['label'], pd.DataFrame())
                    if not mc.empty and 'GSM' in mc.columns and matched_gsms:
                        mc[query_col] = np.where(
                            mc['GSM'].astype(str).str.upper().isin(matched_gsms),
                            display_name, 'Other')
                        scope_df_key[lbl['label']] = mc

            # Switch color to the new column
            if query_col not in self.cc['values']:
                current_vals = list(self.cc['values']) + [query_col]
                self.cc['values'] = current_vals
            self.color_column.set(query_col)
            self._on_color_col_changed()
            dlg.destroy()

        _flat_button(btn_frame, "Apply & Color", _apply, fill=AERO['accent'],
                     font=(UI_FONT, 10, 'bold'), padx=20,
                     pady=6).pack(side=tk.LEFT, padx=5)
        _flat_button(btn_frame, "Cancel", dlg.destroy, font=(UI_FONT, 10),
                     padx=16, pady=6).pack(side=tk.RIGHT, padx=5)

        _add_row()
        _preview()   # the first row is appended after _add_row's own preview
        dlg.update_idletasks()
        w = max(600, dlg.winfo_reqwidth())
        h = dlg.winfo_reqheight()
        try:
            x = self.winfo_x() + (self.winfo_width() - w) // 2
            y = self.winfo_y() + (self.winfo_height() - h) // 2
            dlg.geometry(f"{w}x{h}+{max(0,x)}+{max(0,y)}")
        except: pass

    def _live_labels_from_app(self):
        """The app's current label table for the platforms these regions use.

        ``app.platform_labels`` is the live per-platform store the extraction
        writes into, so it is what a refresh must read. Only the platforms
        actually on screen are taken -- the app-wide concatenation carries
        every other platform's GSMs, which do not belong in these plots.
        """
        store = getattr(self.app, 'platform_labels', None) or {}
        plats, seen = [], set()
        for r in self.regions:
            p = r.get('platform', '')
            if p and p not in seen:
                seen.add(p); plats.append(p)
        frames = [store[p] for p in plats
                  if p in store and store[p] is not None and not store[p].empty]
        if not frames:
            return None
        if len(frames) == 1:
            return frames[0]
        return pd.concat(frames, ignore_index=True)

    def _refresh_labels_from_app(self):
        """Reload labels from app's platform_labels (e.g., after background processing finishes)."""
        try:
            merged = self._live_labels_from_app()
            if merged is not None and not merged.empty:
                self.platform_labels_df = merged.copy()
                self._log("Refreshing labels from app...")

                # Re-merge with expression data
                self._precompute()

                # Update color column options
                opts = self._get_color_cols()
                current = self.color_column.get()
                self.cc['values'] = opts
                if current in opts:
                    self.color_column.set(current)
                elif opts:
                    self.color_column.set(opts[0])

                # Refresh plots
                self._populate_filter_listbox()
                self._refresh_plots()

                new_count = len([c for c in self.platform_labels_df.columns
                                 if c not in self._SKIP_COLS and c != 'GSM'])
                self._log(f"Labels refreshed: {len(self.platform_labels_df):,} GSMs, "
                          f"{new_count} label columns")
                import tkinter.messagebox as mb
                mb.showinfo("Labels Refreshed",
                            f"Labels reloaded from latest extraction.\n"
                            f"{len(self.platform_labels_df):,} GSMs, {new_count} label columns.\n\n"
                            f"Plots updated.", parent=self)
            else:
                import tkinter.messagebox as mb
                mb.showinfo("No Labels", "No updated labels available yet.", parent=self)
        except Exception as e:
            # Logging alone made a failed refresh indistinguishable from a
            # button that does nothing, which is how this reads to the user.
            self._log(f"Label refresh error: {e}")
            messagebox.showerror("Refresh Failed", str(e), parent=self)

    def _on_color_col_changed(self):
        self._populate_filter_listbox()
        self._refresh_plots()

    def _set_plot_mode(self, mode):
        """Toggle plot mode with button appearance."""
        self.plot_mode.set(mode)
        for val, (btn, active_bg) in self._mode_btns.items():
            _paint_toggle(btn, val == mode, active_bg)
        self._refresh_plots()

    def _set_scope(self, scope):
        """Toggle scope: 'selected' or 'total' (whole platform)."""
        self.gse_scope.set(scope)
        _paint_toggle(self._scope_btn_selected, scope == "selected", AERO['accent'])
        _paint_toggle(self._scope_btn_total, scope != "selected", AERO['green_dark'])
        self._on_scope_changed()

    def _toggle_overlay(self):
        """Toggle overlay: show gene distribution on top of platform distribution."""
        val = not self.overlay.get()
        self.overlay.set(val)
        _paint_toggle(self._overlay_btn, val, AERO['accent_dark'])
        self._refresh_plots()

    def _toggle_merge(self):
        """Toggle merge regions on/off."""
        val = not self.merge_regions.get()
        self.merge_regions.set(val)
        _paint_toggle(self._merge_btn, val, AERO['accent_dark'])
        self._refresh_plots()

    def _on_scope_changed(self):
        """Scope changed -> repopulate filter + refresh."""
        self._log(f"[SCOPE] Changed to: {self.gse_scope.get()}")
        self._populate_filter_listbox()
        n_items = len(self._filter_all_items)
        self._log(f"[SCOPE] Filter repopulated: {n_items} items (all checked)")
        self._refresh_plots()

    # ── Refresh ─────────────────────────────────────────────────────
    def _tab_specs(self):
        """(name, render fn, notebook page, frame to report errors in).

        Each render function clears its own frame and closes its own figures,
        so any one tab can be redrawn on its own. Labels reports into its
        scroll body rather than its page, because the page also holds the
        column selector, which its render must not disturb.
        """
        return [
            ("Grouped",    self._render_gse_tab,        self.t_gse,    self.t_gse),
            ("Labels",     self._render_ai_tab,         self.t_ai,
             self.ai_scroll.scrollable_frame),
            ("Frequency",  self._render_freq_tab,       self.t_freq,   self.t_freq),
            ("Enrichment", self._render_enrichment_tab, self.t_enrich, self.t_enrich),
            ("Comparison", self._render_comparison_tab, self.t_cmp,    self.t_cmp),
            ("Statistics", self._render_stats_tab,      self.t_stats,  self.t_stats),
            ("Samples",    self._render_table_tab,      self.t_table,  self.t_table),
            # Last on purpose: it draws what the tabs above have already
            # computed, so it must run after them.
            ("Summary",    self._render_summary_tab,    self.t_summary, self.t_summary),
        ]

    def _render_tab(self, name, fn, report_to):
        """Draw one tab, leaving the reason on screen if it cannot be drawn."""
        try:
            fn()
            self._log(f"OK {name} tab rendered")
            return True
        except Exception as e:
            import traceback
            print(f"[RegionAnalysis] {name} traceback:\n{traceback.format_exc()}")
            self._log(f"X {name} tab FAILED: {e}")
            try:
                ttk.Label(report_to, text=f"Error rendering {name}:\n\n{e}",
                          foreground="red", font=("Consolas", 9),
                          wraplength=600).pack(pady=20)
            except Exception:
                pass
            return False

    def _age_summary(self):
        """The Summary page draws other tabs' results, so it ages when they do.

        Its inputs arrive when a button is pressed on another tab, long after
        it was last drawn. Without this, pressing 'Fit model' and then opening
        Summary shows the page as it stood before the model existed.
        """
        self._stale.add("Summary")

    def _toolbar_signature(self):
        """Everything the tabs are drawn from, as one comparable value."""
        return (self.plot_mode.get(), self.gse_scope.get(),
                self.color_column.get(), self.ai_label_col.get(),
                bool(self.merge_regions.get()), bool(self.overlay.get()),
                tuple(sorted(self.filter_values)))

    def _refresh_plots(self):
        """Re-render for the toolbar's current settings.

        Scope, colour column, overlay and merge feed every tab, so every tab
        is invalidated -- but only the one on screen is redrawn now, the rest
        when they are opened. Redrawing all ten on each toolbar click costs
        seconds; redrawing three (what this used to do) left the other seven
        blank, because their canvases had already been destroyed here.

        A control that has not actually moved invalidates nothing. The Refresh
        button, a combobox that re-fires its selection event and a filter list
        rebuilt with the same values all land here, and each one used to throw
        away eleven drawn tabs and pay to compute them again. Comparing the
        settings against the ones the tabs were last drawn under makes the
        repeat free, which matters most on the machines that can least afford
        the recomputation.
        """
        self.filter_values = self._get_selected_filter_values()
        sig = self._toolbar_signature()
        if sig == getattr(self, "_drawn_signature", None) and not self._stale:
            return
        self._drawn_signature = sig
        self._stale = {spec[0] for spec in self._tab_specs()}
        self._render_current_tab()

    def _render_current_tab(self, _event=None):
        """Draw the visible tab if the toolbar moved on since it was drawn."""
        try:
            current = self.nb.nametowidget(self.nb.select())
        except Exception:
            return
        for name, fn, page, report_to in self._tab_specs():
            if page is current and name in self._stale:
                self._stale.discard(name)
                self._render_tab(name, fn, report_to)
                return

    def _flush_stale(self):
        """Draw every tab the toolbar invalidated but the user never opened.

        Export writes what the window is showing, so it has to make the
        unopened tabs real first.
        """
        for name, fn, _page, report_to in self._tab_specs():
            if name in self._stale:
                self._stale.discard(name)
                self._render_tab(name, fn, report_to)

    def _update_progress(self, step, total, label=""):
        """Update the progress bar and label (safe to call from main thread)."""
        try:
            pct = 100 * step / total if total > 0 else 0
            self._prog_bar['value'] = pct
            self._prog_label.configure(text=label or f"Step {step}/{total}")
            self._prog_pct.configure(text=f"{pct:.0f}%")
            self.update_idletasks()
        except Exception:
            pass

    def _render_all(self):
        self._log("Starting render pipeline...")
        try:
            self._populate_filter_listbox()
            self.filter_values = self._get_selected_filter_values()
        except Exception as e:
            self._log(f"Filter init error: {e}")

        tabs = self._tab_specs()
        n_tabs = len(tabs)

        self._update_progress(0, n_tabs, "Rendering tabs...")

        for i, (name, fn, _page, report_to) in enumerate(tabs):
            self._update_progress(i, n_tabs, f"Rendering {name}...")
            self._render_tab(name, fn, report_to)

        self._stale.clear()
        self._drawn_signature = self._toolbar_signature()
        self._update_progress(n_tabs, n_tabs, "Render complete")
        self._log("Render pipeline complete.")

    # ═══════════════════════════════════════════════════════════════════
    #  TAB 1 - Grouped Distributions
    #
    #  Groups by the current Color By column (series_id, title, tissue...)
    #  SELECTED scope: only samples within the selected expression range
    #  TOTAL scope:    ALL samples from the entire platform (pre-merged)
    #  Merge: combine all regions into one plot with source-prefixed labels
    #  Filter: STRICTLY controls which groups appear - no "Other" if few selected
    # ═══════════════════════════════════════════════════════════════════
    def _render_gse_tab(self):
        for w in self.t_gse.winfo_children(): w.destroy()
        for k in [k for k in self.figs if k.startswith("gse_") or k == "gse"]:
            try: plt.close(self.figs.pop(k))
            except: self.figs.pop(k, None)
            self.canvases.pop(k, None); self.toolbars.pop(k, None)
        mode = self.plot_mode.get()
        scope = self.gse_scope.get()
        ccol = self.color_column.get()
        do_merge = self.merge_regions.get() and len(self.regions) > 1
        do_overlay = self.overlay.get()
        n = len(self.regions)

        scope_names = {"selected": "SELECTED REGION", "total": "ALL LOADED SAMPLES"}
        scope_label = scope_names.get(scope, scope.upper())
        overlay_label = " [+BG]" if do_overlay else ""
        cb_title = f" by {ccol}" if ccol and ccol != "(none)" else ""
        merge_label = " [MERGED]" if do_merge else ""
        ttk.Label(self.t_gse,
                  text=f"{scope_label}{overlay_label} ({mode}){cb_title}{merge_label}  |  "
                       f"{n} region(s)",
                  font=("Segoe UI", 12, "bold")).pack(fill=tk.X, padx=8, pady=(6, 2))

        gse_scroll = ScrollableCanvasFrame(self.t_gse)
        gse_scroll.pack(fill=tk.BOTH, expand=True)

        if do_merge:
            self._render_gse_merged(gse_scroll.scrollable_frame, mode, scope, ccol)
        else:
            self._render_gse_separate(gse_scroll.scrollable_frame, mode, scope, ccol)

    def _render_gse_separate(self, parent, mode, scope, ccol):
        """Render one figure per region.
        Scopes:
          selected  - only samples in highlighted range, colored by condition
          total     - ALL samples on the platform for this gene, colored by condition
        Overlay: when ON, shows platform background histogram underneath.
        """
        do_overlay = self.overlay.get()

        for ri, region in enumerate(self.regions):
            col = region['column']; lo, hi = region['range']
            bg_df = region.get('platform_df', pd.DataFrame())

            fig = Figure(figsize=(16, 7))
            ax = fig.subplots()

            # Gene distribution background (gray histogram)
            # ALWAYS show when Whole Platform - this IS the gene distribution
            # For Selected Region - only show if overlay is ON
            show_bg = (scope == "total") or do_overlay
            if show_bg:
                _draw_bg(ax, bg_df, col)
            xr = _bg_range(bg_df, col)

            # Pick data source based on scope
            if scope == "selected":
                mg = self._sel(region['label'])
                slbl = f"SELECTED [{lo:.2f}-{hi:.2f}] n={len(mg)}"
            else:
                mg = self._tot(region['label'])
                slbl = f"ALL LOADED SAMPLES n={len(mg)}" if not mg.empty else "TOTAL n=0"

            amap = {}; handles = []

            # Add background entry to legend
            if show_bg and not bg_df.empty and col in bg_df.columns:
                n_plat = len(pd.to_numeric(bg_df[col], errors='coerce').dropna())
                handles.append(mlines.Line2D([], [], color=_BG_CLR, lw=6,
                               alpha=0.5, label=f"Gene Distribution (n={n_plat:,})"))

            # ── DEBUG: trace exactly what data the render has ──
            if not mg.empty:
                if ccol in mg.columns:
                    uniq = mg[ccol].fillna("N/A").astype(str).nunique()

            # Color by condition labels
            if not mg.empty and ccol and ccol != "(none)" and ccol in mg.columns:
                grps = mg[ccol].fillna("N/A").astype(str)
                n_total = int(grps.nunique())
                tops = list(grps.value_counts().head(_lim('groups')).index)
                self._log(f"[RENDER] tops={tops[:5]}... (total={len(tops)})")
                colors = _clrs(max(1, len(tops)))

                for i, val in enumerate(tops):
                    clr = colors[i] if i < len(colors) else '#888888'
                    vs = pd.to_numeric(mg.loc[grps == val, col], errors="coerce").dropna()
                    self._log(f"[RENDER] Plotting '{val}': {len(vs)} numeric samples, "
                              f"col='{col}', col_in_mg={col in mg.columns}")
                    if vs.empty:
                        # Diagnose WHY it's empty
                        raw = mg.loc[grps == val, col] if col in mg.columns else pd.Series()
                        self._log(f"[RENDER]   EMPTY! raw_count={len(raw)}, "
                                  f"null_count={raw.isna().sum() if len(raw) > 0 else 'N/A'}, "
                                  f"sample_values={raw.head(3).tolist() if len(raw) > 0 else 'N/A'}")
                        continue
                    lb = f"{_tr(val)} (n={len(vs)})"
                    amap[lb] = _plot_grp(ax, vs, clr, mode, lw=2.0, x_range=xr,
                                         ids=_sample_ids(mg, vs))
                    handles.append(mlines.Line2D([], [], color=clr, lw=2, label=lb))

                if len(tops) < n_total:
                    other_mask = ~grps.isin(tops)
                    if other_mask.any():
                        vs = pd.to_numeric(mg.loc[other_mask, col], errors="coerce").dropna()
                        if not vs.empty and len(vs) > 1:
                            lb = f"Other ({len(vs)})"; arts = []
                            if mode in ("density", "both"):
                                kd = _kde(vs, x_range=xr)
                                if kd:
                                    ln, = ax.plot(kd[0], kd[1], color='gray', lw=1.2,
                                                  ls='--', alpha=0.6, zorder=3)
                                    arts.append(ln)
                            if mode in ("rug", "both"):
                                # Into the same stacked strip as the named
                                # groups: a rugplot draws in axes fractions and
                                # would land on top of the strip rather than in
                                # it, and its ticks carry no per-sample offsets
                                # so they could not be hovered either.
                                ov = np.asarray(vs, dtype=float)
                                orug = ax.scatter(
                                    ov, _strip_offsets(ax, ov, xr), marker='o',
                                    s=14, facecolor='gray', edgecolor='white',
                                    alpha=0.55, linewidths=0.4, zorder=6)
                                orug._gv_strip = True
                                _point_labels(orug, [str(i) for i in
                                                     _sample_ids(mg, vs)])
                                arts.append(orug)
                            handles.append(mlines.Line2D([], [], color='gray', lw=1.2,
                                                          ls='--', label=lb))
                            amap[lb] = arts

            elif not mg.empty:
                self._log(f"[RENDER] FALLBACK: ccol '{ccol}' NOT in mg.columns → showing All")
                vs = pd.to_numeric(mg[col], errors="coerce").dropna()
                lb = f"All (n={len(vs)})"
                amap[lb] = _plot_grp(ax, vs, 'steelblue', mode, lw=2, x_range=xr,
                                     ids=_sample_ids(mg, vs))
                handles.append(mlines.Line2D([], [], color='steelblue', lw=2, label=lb))
            else:
                ax.text(0.5, 0.5, "No data", ha='center', va='center',
                        transform=ax.transAxes, color='gray')

            # Region boundary markers
            ax.axvline(lo, color='red', ls='--', lw=1.2, alpha=0.7, zorder=6)
            ax.axvline(hi, color='red', ls='--', lw=1.2, alpha=0.7, zorder=6)
            ax.axvspan(lo, hi, alpha=0.08, color='red', zorder=0)._gv_overlay = True
            cb = f" by {ccol}" if ccol and ccol != "(none)" else ""
            overlay_t = " [+BG]" if do_overlay else ""
            ax.set_title(f"{region['label']} - {slbl}{cb}{overlay_t}", fontsize=12, weight='bold')
            ax.set_xlabel(self._value_axis_label(region))
            ax.set_ylabel("Normalized Density")
            _finish_sample_strip(ax)
            if handles:
                leg = ax.legend(handles=handles, fontsize=9, loc='upper left',
                                bbox_to_anchor=(1.01, 1.0), framealpha=0.92, fancybox=True)
                _interactive_legend(fig, leg, amap)
            fig.subplots_adjust(left=0.06, right=0.76, top=0.92, bottom=0.10)
            self._embed(fig, parent, f"gse_{ri}")

    def _render_gse_merged(self, parent, mode, scope, ccol):
        """Render ALL regions combined into ONE figure.
        Each condition gets a source tag (gene/platform) so the user can
        distinguish identical condition labels coming from different sources.
        """
        n_regions = len(self.regions)

        # Distinct color palettes per region (so same condition from different
        # regions gets a different hue)
        region_palettes = [
            ['#1565C0', '#1E88E5', '#42A5F5', '#90CAF9', '#BBDEFB',
             '#0D47A1', '#1976D2', '#2196F3', '#64B5F6'],
            ['#C62828', '#E53935', '#EF5350', '#EF9A9A', '#FFCDD2',
             '#B71C1C', '#D32F2F', '#F44336', '#E57373'],
            ['#2E7D32', '#388E3C', '#43A047', '#66BB6A', '#A5D6A7',
             '#1B5E20', '#4CAF50', '#81C784', '#C8E6C9'],
            ['#4E342E', '#6D4C41', '#8D6E63', '#A1887F', '#D7CCC8',
             '#3E2723', '#5D4037', '#795548', '#BCAAA4'],
            ['#6A1B9A', '#7B1FA2', '#8E24AA', '#AB47BC', '#CE93D8',
             '#4A148C', '#9C27B0', '#BA68C8', '#E1BEE7'],
            ['#00838F', '#00ACC1', '#00BCD4', '#26C6DA', '#80DEEA',
             '#006064', '#0097A7', '#4DD0E1', '#B2EBF2'],
        ]

        fig = Figure(figsize=_cap_figsize(18, 8))
        ax = fig.subplots()
        amap = {}; handles = []

        # Draw background (always for Whole Platform, optional for Selected)
        bg0 = self.regions[0].get('platform_df', pd.DataFrame())
        col0 = self.regions[0]['column']
        if scope == "total" or self.overlay.get():
            _draw_bg(ax, bg0, col0)
        xr = _bg_range(bg0, col0)

        # Add a section separator in legend
        color_idx = 0
        for ri, region in enumerate(self.regions):
            col = region['column']; lo, hi = region['range']
            palette = region_palettes[ri % len(region_palettes)]
            gene = region.get('gene', f'R{ri+1}')
            plat = region.get('platform', '')
            src_tag = f"{gene}/{plat}" if plat else gene

            if scope == "selected":
                mg = self._sel(region['label'])
            else:
                mg = self._tot(region['label'])

            if mg.empty:
                continue

            # Region boundary markers with region-specific color
            base_clr = palette[0]
            ax.axvline(lo, color=base_clr, ls='--', lw=1.2, alpha=0.5, zorder=6)
            ax.axvline(hi, color=base_clr, ls='--', lw=1.2, alpha=0.5, zorder=6)
            ax.axvspan(lo, hi, alpha=0.04, color=base_clr, zorder=0)._gv_overlay = True

            # Add region header in legend
            handles.append(mlines.Line2D([], [], color='none',
                           label=f"── {src_tag} [{lo:.1f}-{hi:.1f}] ──"))

            if ccol and ccol != "(none)" and ccol in mg.columns:
                grps = mg[ccol].fillna("N/A").astype(str)
                tops = list(grps.value_counts().head(_lim('groups')).index)

                for i, val in enumerate(tops):
                    clr = palette[i % len(palette)]
                    vs = pd.to_numeric(mg.loc[grps == val, col], errors="coerce").dropna()
                    if vs.empty: continue
                    lb = f"[{src_tag}] {_tr(val)} (n={len(vs)})"
                    # Use different line styles per region for additional distinction
                    ls_list = ['-', '--', '-.', ':']
                    lw_base = 2.0
                    art = _plot_grp(ax, vs, clr, mode, lw=lw_base, x_range=xr,
                                    ids=_sample_ids(mg, vs))
                    # Apply linestyle to density lines
                    ls_style = ls_list[ri % len(ls_list)]
                    for a in art:
                        if hasattr(a, 'set_linestyle'):
                            a.set_linestyle(ls_style)
                    amap[lb] = art
                    handles.append(mlines.Line2D([], [], color=clr, lw=lw_base,
                                   ls=ls_style, label=lb))
            else:
                vs = pd.to_numeric(mg[col], errors="coerce").dropna()
                lb = f"[{src_tag}] All (n={len(vs)})"
                ls_style = ['-', '--', '-.', ':'][ri % 4]
                art = _plot_grp(ax, vs, base_clr, mode, lw=2.0, x_range=xr,
                                ids=_sample_ids(mg, vs))
                for a in art:
                    if hasattr(a, 'set_linestyle'):
                        a.set_linestyle(ls_style)
                amap[lb] = art
                handles.append(mlines.Line2D([], [], color=base_clr, lw=2,
                               ls=ls_style, label=lb))

        scope_names = {"selected": "SELECTED", "total": "ALL LOADED SAMPLES"}
        scope_lbl = scope_names.get(scope, scope.upper())
        overlay_t = " [+BG]" if self.overlay.get() else ""
        cb = f" by {ccol}" if ccol and ccol != "(none)" else ""
        ax.set_title(f"MERGED: {n_regions} regions - {scope_lbl}{overlay_t}{cb}", fontsize=13, weight='bold')
        ax.set_xlabel(self._value_axis_label()); ax.set_ylabel("Normalized Density")
        _finish_sample_strip(ax)
        if handles:
            leg = ax.legend(handles=handles, fontsize=9, loc='upper left',
                            bbox_to_anchor=(1.01, 1.0), framealpha=0.92, fancybox=True)
            _interactive_legend(fig, leg, amap)
        fig.subplots_adjust(left=0.06, right=0.68, top=0.92, bottom=0.10)
        self._embed(fig, parent, "gse_merged")

    # ═══════════════════════════════════════════════════════════════════
    #  TAB 3 - AI Labels (per-label selector + density + frequency)
    # ═══════════════════════════════════════════════════════════════════
    # ── Inline label extraction (runs inside Region Analysis) ──

    def _start_inline_extraction(self):
        """Launch LLM label extraction for samples in current regions, with live progress."""
        if getattr(self, '_extraction_running', False):
            return

        # Gather all GSMs across all regions
        all_gsms = set()
        for r in self.regions:
            all_gsms.update(str(g).strip() for g in r['gsm_list'])
        n_total = len(all_gsms)
        if n_total == 0:
            messagebox.showinfo("Extract Labels",
                                "These regions contain no samples to extract.",
                                parent=self)
            return

        self._extraction_running = True
        self._update_progress(0, n_total, f"Extracting labels for {n_total:,} samples...")
        self._log(f"Starting inline extraction for {n_total:,} samples...")

        # Update button state
        if hasattr(self, '_btn_extract'):
            self._btn_extract.configure(state='disabled', text='Extracting...')

        import time as _time

        def _run_extraction():
            try:
                # Get the classification agent from the main app
                agent = getattr(self.app, 'ai_agent', None)
                if agent is None:
                    self._log("[WARN] No AI agent available - creating one")
                    # Extraction flows through the vendored geo_label_extractor
                    # inside SampleClassificationAgent.process_samples; no tools list
                    # is needed (it is ignored).
                    from genevariate.utils.workers import SampleClassificationAgent
                    agent = SampleClassificationAgent([], self._log, max_workers=4)

                # Build a DataFrame of samples to classify
                import pandas as _pd
                rows, seen = [], set()
                for r in self.regions:
                    mg = self._mc.get(r['label'], _pd.DataFrame())
                    if mg.empty:
                        continue
                    for _, row in mg.iterrows():
                        gsm = str(row.get('GSM', row.get('gsm', ''))).strip()
                        # Was re-derived from `rows` on every row, making this
                        # quadratic in the sample count.
                        if gsm and gsm not in seen:
                            seen.add(gsm)
                            rows.append(row.to_dict())

                if not rows:
                    self._extraction_running = False
                    return

                samples_df = _pd.DataFrame(rows)
                if 'GSM' not in samples_df.columns and 'gsm' in samples_df.columns:
                    samples_df = samples_df.rename(columns={'gsm': 'GSM'})

                n = len(samples_df)
                self._log(f"Classifying {n:,} samples with LLM agent...")

                # Track progress via polling
                t0 = _time.time()
                _done = [0]
                _log_attr = 'log_func' if hasattr(agent, 'log_func') else 'log'
                _orig_log = getattr(agent, _log_attr)

                def _progress_log(msg):
                    _orig_log(msg)
                    if "Progress:" in msg:
                        try:
                            parts = msg.split("Progress:")[1].strip().split("/")
                            _done[0] = int(parts[0].strip())
                        except Exception:
                            pass
                    # Update progress bar from any thread via after()
                    elapsed = _time.time() - t0
                    spd = _done[0] / elapsed if elapsed > 0 else 0
                    eta_s = int((n - _done[0]) / spd) if spd > 0 else 0
                    eta_str = f"{eta_s // 60}m {eta_s % 60}s" if eta_s > 0 else ""
                    lat = f"{elapsed / max(_done[0], 1) * 1000:.0f}ms/sample"
                    try:
                        self.after(0, lambda d=_done[0], lt=lat, et=eta_str: (
                            self._update_progress(d, n,
                                f"Extracting: {d}/{n}  {lt}  ETA: {et}"),
                        ))
                    except Exception:
                        pass

                setattr(agent, _log_attr, _progress_log)

                _facts = getattr(self.app, '_platform_facts', None)
                _cats = {_facts(r.get('platform', '')).get('category', '')
                         for r in self.regions} if _facts else set()
                _enrich = (bool(getattr(self.app, '_extraction_enrich', False))
                           and len(_cats) == 1)

                # Run extraction
                result_df = agent.process_samples(
                    samples_df, enrich=_enrich,
                    category=next(iter(_cats)) if _cats else "")

                setattr(agent, _log_attr, _orig_log)

                if result_df is not None and not result_df.empty:
                    # Merge results into platform_labels_df
                    self._log(f"Extraction complete: {len(result_df):,} samples classified")

                    # Store as platform labels
                    self.platform_labels_df = result_df
                    # app.platform_labels maps platform id -> table. Assigning
                    # the frame itself here replaced that dict and broke every
                    # reader of it, so write into the platforms these regions
                    # came from instead.
                    try:
                        store = getattr(self.app, 'platform_labels', None)
                        if isinstance(store, dict):
                            for p in {r.get('platform', '') for r in self.regions}:
                                if p:
                                    store[p] = result_df
                    except Exception:
                        pass

                    # Recompute merged data with new labels
                    self.after(0, self._post_extraction_refresh)
                else:
                    self._log("[WARN] Extraction returned no results")
                    self.after(0, lambda: self._update_progress(
                        n, n, "Extraction returned no labels - check the "
                              "extraction backend"))

            except Exception as exc:
                import traceback
                self._log(f"[ERROR] Extraction failed: {exc}")
                print(traceback.format_exc())
                # `exc` no longer exists once the except block ends, so the
                # message is bound here rather than looked up on the Tk thread.
                self.after(0, lambda msg=str(exc): self._update_progress(
                    0, 1, f"Error: {msg}"))
            finally:
                self._extraction_running = False
                try:
                    self.after(0, lambda: (
                        hasattr(self, '_btn_extract') and
                        self._btn_extract.configure(
                            state='normal',
                            text='Extract Labels (LLM)')))
                except Exception:
                    pass

        threading.Thread(target=_run_extraction, daemon=True).start()

    def _post_extraction_refresh(self):
        """Refresh all data and tabs after extraction completes."""
        self._log("Refreshing views with new labels...")
        self._update_progress(1, 1, "Labels extracted - refreshing views...")

        try:
            self._precompute()
            # Update AI combo with new columns
            new_opts = self._get_ai_cols()
            self.ai_combo['values'] = new_opts
            best = self._default_label_col(new_opts)
            if best:
                self.ai_label_col.set(best)
            self._render_all()
            self._update_progress(1, 1, "Labels ready")
        except Exception as e:
            self._log(f"Refresh error: {e}")
            self._update_progress(1, 1, f"Refresh error: {e}")

    def _render_ai_tab(self):
        self.ai_scroll.clear()
        # Clean up old per-region figures
        for k in [k for k in self.figs if k.startswith("ai_") or k == "ai"]:
            try: plt.close(self.figs.pop(k))
            except: self.figs.pop(k, None)
            self.canvases.pop(k, None); self.toolbars.pop(k, None)
        lc = self.ai_label_col.get()
        mode = self.plot_mode.get()

        if not lc or lc == "(none)":
            sf = self.ai_scroll.scrollable_frame

            # Count total samples across all regions
            total_gsms = set()
            for r in self.regions:
                total_gsms.update(str(g).strip() for g in r['gsm_list'])
            n_gsms = len(total_gsms)

            # ── No labels: show extraction UI ──
            header = ttk.Label(sf,
                text="Labels Not Yet Extracted",
                font=(UI_FONT, 14, "bold"), foreground=AERO['accent_dark'])
            header.pack(pady=(30, 8))

            ttk.Label(sf,
                text=f"{n_gsms:,} samples in selected region(s) need label extraction.\n"
                     f"The LLM agent will classify Tissue, Condition, and Treatment\n"
                     f"for each sample using its GEO metadata.",
                font=(UI_FONT, 10), foreground=AERO['muted'],
                justify="center").pack(pady=(0, 12))

            # Extraction progress frame
            prog_frame = ttk.Frame(sf)
            prog_frame.pack(fill='x', padx=60, pady=(0, 8))

            self._ext_bar = ttk.Progressbar(
                prog_frame, mode='determinate',
                style='Accent.Horizontal.TProgressbar', length=400)
            self._ext_bar.pack(fill='x', pady=(0, 4))

            self._ext_status = ttk.Label(prog_frame,
                text=f"Ready to extract {n_gsms:,} samples",
                style='Metric.TLabel')
            self._ext_status.pack(anchor='w')

            # Extract button
            self._btn_extract = _flat_button(sf,
                f"Extract Labels (LLM)  \u2014  {n_gsms:,} samples",
                self._start_inline_extraction,
                fill=AERO['accent'], hover=AERO['sky_bot'],
                font=(UI_FONT, 11, "bold"), padx=30, pady=10)
            self._btn_extract.pack(pady=(4, 16))

            # Info text
            info_frame = ttk.Frame(sf)
            info_frame.pack(padx=60, fill='x')
            for step, text in [
                (True, "Raw LLM extraction (Tissue, Condition, Treatment)"),
                (True, "Per-GSE label collapsing (abbreviation matching)"),
                (True, "Results shown automatically when complete"),
                (False, "Requires a reachable extraction backend"),
            ]:
                row = ttk.Frame(info_frame)
                row.pack(fill='x', pady=1)
                ttk.Label(row, text="\u25b8" if step else "\u2022",
                          font=(UI_FONT, 9),
                          foreground=AERO['accent'] if step else AERO['muted']
                          ).pack(side='left', padx=(0, 6))
                ttk.Label(row, text=text, font=(UI_FONT, 9),
                          foreground=AERO['text'] if step else AERO['muted']
                          ).pack(side='left')

            return

        nice = lc.replace('_', ' ')
        sf = self.ai_scroll.scrollable_frame

        for ci, region in enumerate(self.regions):
            scope = self.gse_scope.get()
            if scope == "selected":
                mg = self._sel(region['label'])
            else:
                mg = self._tot(region['label'])
            ecol = region['column']

            # ── Figure 1: Density plot (full width) ──
            fig_d = Figure(figsize=(16, 7))
            ax_d = fig_d.subplots()
            if mg.empty or lc not in mg.columns:
                # An empty panel has two very different causes and a bare "N/A"
                # hides which one it is: the region can hold no samples, or the
                # chosen field can be one the platform behind this region was
                # never annotated with. Naming the field and the platform is
                # what separates "nothing applies here" from "this is broken".
                if mg.empty:
                    why = "no samples in this region"
                else:
                    why = (f"'{lc}' is not among the labels loaded for "
                           f"{region.get('platform') or 'this platform'}")
                ax_d.text(0.5, 0.5, why, ha='center', va='center',
                          transform=ax_d.transAxes, color='gray')
                ax_d.set_title(f"{region['label']} - {nice} Density", fontsize=11)
            else:
                bg = region.get('platform_df')
                show_bg = (scope == "total") or self.overlay.get()
                if show_bg:
                    _draw_bg(ax_d, bg, ecol)
                xr = _bg_range(bg, ecol)
                smart, binned = _smart_series(mg[lc], max_cats=_lim('groups'))

                tops = list(smart.value_counts().head(_lim('groups')).index)

                colors = _clrs(max(1, len(tops)))
                handles = []; amap = {}
                for val, clr in zip(tops, colors):
                    sub = pd.to_numeric(mg.loc[smart == val, ecol], errors="coerce").dropna()
                    if sub.empty: continue
                    lb = f"{_tr(val)} ({len(sub)})"
                    amap[lb] = _plot_grp(ax_d, sub, clr, mode, lw=2.0, x_range=xr,
                                         ids=_sample_ids(mg, sub))
                    handles.append(mlines.Line2D([], [], color=clr, lw=2, label=lb))
                suffix = " (binned)" if binned else ""
                scope_lbl = "SELECTED" if scope == "selected" else "WHOLE PLATFORM"
                ax_d.set_title(f"{region['label']} - {nice}{suffix} {scope_lbl}",
                               fontsize=12, weight='bold')
                ax_d.set_xlabel(self._value_axis_label(region), fontsize=10)
                ax_d.set_ylabel("Normalized Density", fontsize=10)
                _finish_sample_strip(ax_d)
                if handles:
                    leg = ax_d.legend(handles=handles, fontsize=9, loc='upper left',
                                      bbox_to_anchor=(1.01, 1.0), ncol=max(1, len(handles) // 12),
                                      framealpha=0.92)
                    _interactive_legend(fig_d, leg, amap)
            fig_d.subplots_adjust(left=0.06, right=0.75, top=0.92, bottom=0.10)
            self._embed(fig_d, sf, f"ai_d_{ci}")

            # ── Figure 2: Frequency bar chart (full width) ──
            if not mg.empty and lc in mg.columns:
                smart_f, binned_f = _smart_series(mg[lc], max_cats=_lim('bars'))
                all_counts_f = smart_f.value_counts()
                counts = all_counts_f.head(_lim('bars'))
                total_all = len(smart_f)
                n_other = total_all - counts.sum()

                if not counts.empty:
                    n_bars = len(counts)
                    fig_h = max(4, 0.4 * n_bars + 1.5)
                    fig_f = Figure(figsize=(16, fig_h))
                    ax_f = fig_f.subplots()

                    trunc_idx = [_tr(s, 35) for s in counts.index]
                    colors = _clrs(len(counts))
                    bars = ax_f.barh(trunc_idx[::-1], counts.values[::-1],
                                     color=colors[::-1], edgecolor='black', lw=0.4)
                    for bar, cnt in zip(bars, counts.values[::-1]):
                        pct = cnt / total_all * 100
                        ax_f.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                                  f"{cnt} ({pct:.1f}%)", va='center', fontsize=8, color='#333')
                    suffix_f = " (binned)" if binned_f else ""
                    xlabel = "Count"
                    if n_other > 0:
                        xlabel += f"  (+ {n_other:,} other)"
                    ax_f.set_xlabel(xlabel, fontsize=10)
                    from genevariate.utils import display_limits
                    note_f = display_limits.cap_note(len(counts),
                                                     len(all_counts_f),
                                                     noun="values")
                    ax_f.set_title(
                        f"{region['label']} - {nice}{suffix_f} Frequency"
                        + (f"  ({note_f})" if note_f else ""),
                        fontsize=12, weight='bold')
                    ax_f.tick_params(labelsize=8)
                    fig_f.subplots_adjust(left=0.22, right=0.92, top=0.92, bottom=0.10)
                    self._embed(fig_f, sf, f"ai_f_{ci}")

    # ═══════════════════════════════════════════════════════════════════
    #  TAB 4 - Frequency Analysis (for ANY Color By column)
    # ═══════════════════════════════════════════════════════════════════
    def _render_freq_tab(self):
        for w in self.t_freq.winfo_children(): w.destroy()
        # Clean up old per-region figures
        for k in [k for k in self.figs if k.startswith("freq_") or k == "freq"]:
            try: plt.close(self.figs.pop(k))
            except: self.figs.pop(k, None)
            self.canvases.pop(k, None); self.toolbars.pop(k, None)
        ccol = self.color_column.get()
        if not ccol or ccol == "(none)":
            ttk.Label(self.t_freq, text="Select a Color By column to see frequency analysis.",
                      font=("Segoe UI", 11), foreground="gray").pack(pady=40)
            return

        # ── Scrollable area for per-region frequency charts ──
        freq_scroll = ScrollableCanvasFrame(self.t_freq)
        freq_scroll.pack(fill=tk.BOTH, expand=True)
        sf = freq_scroll.scrollable_frame

        all_freq_data = []

        for idx, region in enumerate(self.regions):
            # SELECTED region samples
            mg = self._sel(region['label'])
            # ALL platform samples (with labels merged)
            mg_total = self._tot(region['label'])

            if mg_total.empty or ccol not in mg_total.columns:
                continue

            # Platform-wide label counts (ALL samples)
            plat_ser = mg_total[ccol].fillna("N/A").astype(str)
            plat_smart, binned = _smart_series(plat_ser, max_cats=_lim('bars'))
            plat_counts = plat_smart.value_counts()
            plat_total = len(plat_smart)

            # Selected region label counts
            sel_gsms = self._sel_gsms(region)
            if not mg.empty and ccol in mg.columns:
                sel_ser = mg[ccol].fillna("N/A").astype(str)
                if binned:
                    # Apply same binning to selected data
                    sel_smart, _ = _smart_series(sel_ser, max_cats=_lim('bars'))
                else:
                    sel_smart = sel_ser
                sel_counts = sel_smart.value_counts()
                sel_total = len(sel_smart)
            else:
                sel_counts = pd.Series(dtype=int)
                sel_total = 0

            if plat_counts.empty:
                continue

            # Build rows for ALL label values on the platform
            rows = []
            for val in plat_counts.head(_lim('bars')).index:
                plat_cnt = int(plat_counts.get(val, 0))
                sel_cnt = int(sel_counts.get(val, 0))
                rest_cnt = max(0, plat_cnt - sel_cnt)
                sel_frac = sel_cnt / max(1, sel_total)
                plat_frac = plat_cnt / max(1, plat_total)
                enr = sel_frac / plat_frac if plat_frac > 0 else 0.0
                rows.append({
                    'Value': val, 'Selected': sel_cnt, 'Rest': rest_cnt,
                    'Total': plat_cnt, 'Sel%': sel_frac * 100,
                    'Plat%': plat_frac * 100, 'Enrichment': enr,
                    'Region': region['label']
                })

            rdf = pd.DataFrame(rows).sort_values('Enrichment', ascending=True)
            all_freq_data.extend(rows)

            # One full-size figure per region
            n_bars = len(rdf)
            fig_h = max(5, 0.4 * n_bars + 1.5)
            fig = Figure(figsize=(16, fig_h))
            ax = fig.subplots()

            bc = ['#C62828' if r >= 3 else '#E53935' if r >= 2 else '#EF9A9A' if r >= 1.5
                  else '#43A047' if r >= 1 else '#78909C' for r in rdf['Enrichment']]
            trunc_vals = [_tr(v, 35) for v in rdf['Value']]
            bars = ax.barh(trunc_vals, rdf['Enrichment'], color=bc, edgecolor='black', lw=0.4)
            ax.axvline(1.0, color='black', ls='--', lw=1, alpha=0.5)
            # The counts are written to the right of each bar in data units, so
            # the axis has to be widened to hold them. Without the floor, a
            # region that enriches nothing collapses the axis to near-zero width
            # and both the annotations and the 1.0 reference line fall outside.
            ax.set_xlim(0, max(1.15, float(rdf['Enrichment'].max())) * 1.45)

            for bar, row in zip(bars, rdf.itertuples()):
                ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                        f"{row.Selected}sel / {row.Rest}rest / {row.Total}tot  ({row._5:.1f}%)",
                        va='center', fontsize=7, color='#333')

            suffix = " (binned)" if binned else ""
            ax.set_xlabel("Enrichment Ratio (selected vs platform)", fontsize=10)
            ax.set_title(f"{region['label']} - {ccol}{suffix}\n"
                         f"Selected: {sel_total:,} samples | Platform: {plat_total:,} samples",
                         fontsize=12, weight='bold')
            ax.tick_params(labelsize=8)
            # Sized to the labels that are actually there rather than a fixed
            # 22% gutter, which left a hand's width of white space whenever the
            # category names were short.
            fig.tight_layout()
            self._embed(fig, sf, f"freq_{idx}")

        # ── Summary frequency table ──
        if all_freq_data:
            tbl_frame = labelframe(sf, text="Frequency Table (All Platform Labels)", padding=5)
            tbl_frame.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)

            cols = ("Region", "Value", "Selected", "Rest", "Total", "Sel%", "Plat%", "Enrichment")
            tree = ttk.Treeview(tbl_frame, columns=cols, show="headings", height=16)
            for c in cols:
                tree.heading(c, text=c)
                tree.column(c, width=100 if c != "Value" else 200, anchor='center')
            vsb = ttk.Scrollbar(tbl_frame, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=vsb.set)
            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            vsb.pack(side=tk.RIGHT, fill=tk.Y)

            for row in sorted(all_freq_data, key=lambda x: -x['Enrichment']):
                tree.insert("", tk.END, values=(
                    row['Region'], _tr(row['Value'], 40),
                    row['Selected'], row['Rest'], row['Total'],
                    f"{row['Sel%']:.1f}%", f"{row['Plat%']:.1f}%",
                    f"{row['Enrichment']:.2f}"
                ))

    # ═══════════════════════════════════════════════════════════════════
    #  TAB 5 - Fisher Enrichment Analysis
    #
    #  For each region x each label column:
    #    - Selected region GSMs vs rest of platform
    #    - 2x2 contingency table -> Fisher exact test (one-sided, greater)
    #    - Enrichment ratio, p-value, significance stars
    #  Uses platform_labels_df (default labels) or AI labels from _mc
    # ═══════════════════════════════════════════════════════════════════
    def _enrich_label_cols(self):
        """Label columns worth testing: AI labels plus loaded platform labels.

        A region frame is a merge of three provenances: the labels a file or
        the extractor assigned, and GEO's own sample record, which is joined in
        so the Samples tab can show what a sample actually says. That record is
        not a set of labels. ``library_strategy`` and ``molecule_ch1`` are how
        the library was built, ``supplementary_file`` is near-unique per
        sample, and ``description`` is free text -- testing them asks 30
        questions no one posed and spends the FDR budget that the real fields
        need. So every column GEO supplied is excluded, and what remains is
        what someone assigned as a label.

        Two things are still not labels after that. A column naming the donor,
        the dataset or the ontology term the row came from answers "which batch
        is this" rather than "what is this", and single-cell frames carry every
        field twice, once under its own name and once under a ``Classified_``
        alias, so a field tested through both is counted twice by the FDR
        correction. Identifier columns are dropped by name, and a column whose
        values are identical to one already accepted is dropped as an alias.
        """
        label_cols = []
        for r in self.regions:
            m = self._sel(r['label'])
            if m.empty:
                continue
            geo_record = {str(c) for c in r.get('meta_df', pd.DataFrame()).columns}
            seen = {_col_signature(m[c]): c for c in label_cols if c in m.columns}
            for c in m.columns:
                if c in self._SKIP_COLS or c == 'GSM' or c in label_cols:
                    continue
                if c in geo_record or _is_identifier_col(c):
                    continue
                if m[c].dtype != 'object' or m[c].nunique() <= 1:
                    continue
                sig = _col_signature(m[c])
                if sig in seen:
                    continue
                seen[sig] = c
                label_cols.append(c)

        plat_labels = self.platform_labels_df
        if plat_labels is not None and not plat_labels.empty:
            for c in platform_label_cols(plat_labels):
                if c not in label_cols:
                    label_cols.append(c)
        return label_cols

    def _render_enrichment_tab(self):
        for w in self.t_enrich.winfo_children(): w.destroy()

        # ── Detect available label columns ──
        label_cols = self._enrich_label_cols()
        plat_labels = self.platform_labels_df

        if not label_cols:
            ttk.Label(self.t_enrich,
                      text="No label columns available for enrichment analysis.\n\n"
                           "Load labels from file or run classification first.",
                      style='Empty.TLabel').pack(pady=40)
            return

        # ── Header ──
        ttk.Label(self.t_enrich,
                  text="Fisher's Exact Test - Significant Enrichments Only (FDR q<0.05)",
                  style='Section.TLabel').pack(fill=tk.X, padx=10, pady=(8, 2))
        _hint = ttk.Label(self.t_enrich,
                  text="Shows ONLY label values significantly enriched in the selected region vs rest of platform. "
                       "For ALL labels, see the Frequency Analysis tab.\n"
                       "GEO samples arrive in study-sized clumps, so every hit also reports the number of "
                       "contributing studies (n_GSE), the effective sample size after that clumping (n_eff) "
                       "and a 95% CI bootstrapped over studies rather than samples.",
                  style='Hint.TLabel', justify=tk.LEFT)
        _wrap_to_parent(_hint)
        _hint.pack(fill=tk.X, padx=10, pady=(0, 4))

        # ── Compute enrichment for each region x label column ──
        # Store structured data for both table and plots
        # The test itself lives in analysis.region_enrichment so that the
        # assistant can report the same q values this tab displays instead of
        # a second implementation of them. What stays here is the part that is
        # genuinely about this window: deciding which frame each label column's
        # values come from, and which samples count as the background.
        all_rows = []           # flat list for table
        self._enrich_rows = all_rows   # the Summary tab reads these, uncomputed
        self._age_summary()
        plot_groups = {}        # (region_label, lcol) -> list of row dicts
        cells = {}              # (region_label, lcol) -> (inside, outside)
        cell_groups = {}        # (region_label, lcol) -> study id per sample
        gse_map = self._gse_map()

        for region in self.regions:
            sel_gsms = self._sel_gsms(region)
            mg = self._sel(region['label'])

            for lcol in label_cols:
                # Build label series for selected GSMs
                sel_labels = None
                if not mg.empty and lcol in mg.columns:
                    sel_labels = mg.set_index('GSM')[lcol].dropna().astype(str)

                # The background is the loaded samples the region was drawn
                # from -- not the platform, which is only fully loaded when
                # every one of its series has been downloaded.
                # A label file can describe far more samples than the platform
                # measured, and a sample with no expression value could never
                # have entered or left the region, so counting it as "not
                # selected" inflates every enrichment and leaves the selection
                # with no study to bootstrap over. The file is consulted only
                # for columns the platform frame does not carry.
                all_labels = None
                total = self._tot(region['label'])
                if not total.empty and 'GSM' in total.columns and lcol in total.columns:
                    all_labels = total.set_index(
                        total['GSM'].astype(str).str.upper())[lcol].dropna().astype(str)
                if ((all_labels is None or all_labels.empty)
                        and plat_labels is not None and not plat_labels.empty
                        and lcol in plat_labels.columns and 'GSM' in plat_labels.columns):
                    all_labels = plat_labels.set_index(
                        plat_labels['GSM'].astype(str).str.upper())[lcol].dropna().astype(str)

                if all_labels is None or all_labels.empty:
                    continue

                # Resolving the labels above is this window's job - it knows
                # which frame a column lives in and when to fall back to the
                # file. Splitting them into the region and its background is
                # not: `build_enrichment_cells` does that for the assistant's
                # tool as well, so the two cannot drift the way they did when
                # each kept its own copy.
                c_cells, c_groups = build_enrichment_cells(
                    {lcol: all_labels}, sel_gsms,
                    region_name=region['label'],
                    study_of={str(k).upper(): v
                              for k, v in gse_map.items()} if gse_map else None)
                if not c_cells:
                    continue
                cells.update(c_cells)
                cell_groups.update(c_groups)
                if gse_map and not c_groups:
                    self._log(f"[!] '{lcol}': too few GSMs carry a study id - "
                              f"clumping stats unavailable")

        # One Fisher grid, one BH correction over all of it, diagnostics for the
        # survivors only. The table comes back sorted by p.
        table = region_label_enrichment(
            cells, cell_groups, max_values=_lim('enrichment_rows'),
            n_boot=_ENRICH_BOOT)

        from genevariate.core.analysis.figures import enrichment_rows
        for row in enrichment_rows(table):
            all_rows.append(row)
            plot_groups.setdefault((row['Region'], row['Label Column']),
                                   []).append(row)

        if not all_rows:
            ttk.Label(self.t_enrich,
                      text="No enrichment data could be computed.\n"
                           "Ensure labels have matching GSMs with the platform.",
                      font=("Segoe UI", 11), foreground="orange").pack(pady=30)
            return

        n_sig = sum(1 for r in all_rows if r['Sig'] != 'ns')
        n_thin = sum(1 for r in all_rows if r['Sig'] != 'ns' and _is_thin(r))
        head = (f"OK {n_sig} significantly enriched (FDR q<0.05) / {len(all_rows)} tested  |  "
                f"{len(plot_groups)} group(s) across {len(self.regions)} region(s)")
        if n_thin:
            head += (f"  |  WARNING {n_thin} of them are not replicated across studies "
                     f"(<3 GSEs or CI covers 1.0)")
        ttk.Label(self.t_enrich, text=head,
                  font=("Segoe UI", 9, "bold"),
                  foreground=AERO['danger'] if n_sig > 0 else AERO['muted']).pack(fill=tk.X, padx=10, pady=2)

        # ═══════════════════════════════════════════════════════════════
        #  SCROLLABLE AREA for all plots + table
        # ═══════════════════════════════════════════════════════════════
        enrich_scroll = ScrollableCanvasFrame(self.t_enrich)
        enrich_scroll.pack(fill=tk.BOTH, expand=True)
        sf = enrich_scroll.scrollable_frame

        plot_idx = 0
        # Only plot groups that have at least one significant result (p<0.05)
        # This prevents creating 24+ figures when most groups are non-significant
        sig_groups = {k: v for k, v in plot_groups.items()
                      if any(r['Sig'] != 'ns' for r in v)}
        skip_groups = {k: v for k, v in plot_groups.items() if k not in sig_groups}
        if skip_groups:
            ttk.Label(sf,
                      text=f"({len(skip_groups)} group(s) with no significant enrichment - plots omitted)",
                      font=("Segoe UI", 8, "italic"), foreground="gray").pack(padx=10, pady=2)

        for (reg_label, lcol), rows in sig_groups.items():
            nice_col = lcol
            n_sel = rows[0]['n_sel']
            n_non = rows[0]['n_non']

            # ── Section header ──
            ttk.Separator(sf, orient='horizontal').pack(fill=tk.X, pady=6)
            ttk.Label(sf,
                      text=f"> {reg_label}  x  {nice_col}   "
                           f"(selected: {n_sel}  |  rest: {n_non:,})",
                      font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, padx=10, pady=(4, 2))

            # Only show significant rows in the enrichment plots
            sig_only = [r for r in rows if r['Sig'] != 'ns']
            top_rows = sig_only[:_lim('enrichment_rows')]
            n_omitted = len(sig_only) - len(top_rows)

            # Same routine the assistant calls, so the chart a button draws
            # and the chart a tool returns cannot drift apart.
            from genevariate.core.analysis.figures import draw_enrichment
            if not top_rows:
                continue
            fig_h = max(4, min(20, 0.45 * len(top_rows) + 1.5))
            fig = Figure(figsize=(16, fig_h))
            ax1, ax2 = fig.subplots(1, 2, gridspec_kw={'width_ratios': [3, 2]})
            draw_enrichment(ax1, ax2, top_rows, label_column=nice_col)
            # An exported figure has to carry its own caveat: the note below
            # is a widget and does not travel with the PNG.
            from genevariate.utils import display_limits
            note = display_limits.cap_note(len(top_rows), len(sig_only),
                                           noun="values that survived")
            fig.suptitle(f"{reg_label} - Fisher Enrichment: {nice_col}"
                         + (f"  ({note})" if note else ""),
                         fontsize=11, weight='bold', y=0.995)
            try:
                fig.tight_layout(rect=(0, 0, 1, 0.96))
            except Exception:
                pass
            self._embed(fig, sf, f"enrich_{plot_idx}")
            plot_idx += 1
            if n_omitted > 0:
                ttk.Label(sf, text=f"  ({n_omitted} additional label values not shown in plot)",
                          font=("Segoe UI", 8, "italic"), foreground="gray").pack(padx=10, pady=1)

        # ════════════════════════════════════════════════════════════
        #  PLOT 3: Volcano plot (all groups combined)
        # ════════════════════════════════════════════════════════════
        # A volcano needs a finite x. Values with an undefined or unbounded
        # fold have no position on it, so they are left out and counted, never
        # dropped in silence.
        volcano_rows = [r for r in all_rows if r['p-value'] < 1.0
                        and 0 < r['Enrichment'] < 100]
        volcano_off = len([r for r in all_rows if r['p-value'] < 1.0
                           and not (0 < r['Enrichment'] < 100)])
        if len(volcano_rows) >= 3:
            ttk.Separator(sf, orient='horizontal').pack(fill=tk.X, pady=6)
            ttk.Label(sf, text="> Volcano Plot - All Regions x All Label Columns",
                      font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, padx=10, pady=(4, 2))

            fig = Figure(figsize=(14, 7))
            ax = fig.subplots()
            x_vals = [np.log2(max(r['Enrichment'], 0.01)) for r in volcano_rows]
            y_vals = [-np.log10(max(r['p-value'], 1e-50)) for r in volcano_rows]
            v_sigs = [r['Sig'] for r in volcano_rows]
            v_colors = ['#C62828' if s == '***' else '#E53935' if s == '**'
                        else '#EF9A9A' if s == '*' else '#BDBDBD' for s in v_sigs]

            ax.scatter(x_vals, y_vals, c=v_colors, s=45, alpha=0.75,
                       edgecolor='black', lw=0.3, zorder=3,
                       picker=True, pickradius=5)
            # threshold line sits at the FDR cutoff, not the raw p cutoff
            _sig_p = [r['p-value'] for r in volcano_rows if r['Sig'] != 'ns']
            if _sig_p:
                ax.axhline(-np.log10(max(max(_sig_p), 1e-50)),
                           color='gray', ls='--', lw=1, alpha=0.5)
            ax.axvline(0, color='gray', ls='--', lw=1, alpha=0.5)

            # NO auto-labels - click to inspect instead. The hint goes at the
            # foot of the figure, not just under the axes, where it would be
            # drawn on top of the x-axis label.
            ax.figure.text(
                0.5, 0.005,
                'Click points to inspect  •  Shift+click to multi-select  •  Double-click to clear',
                fontsize=7.5, ha='center', color='#777777', style='italic')

            ax.set_xlabel("log2(Enrichment Ratio)", fontsize=10)
            ax.set_ylabel("-log10(p-value)", fontsize=10)
            ax.set_title(
                "Enrichment Volcano - Selected Region vs Rest of Loaded Samples"
                + (f"  ({volcano_off} value(s) off scale: fold undefined or "
                   f"above 100x)" if volcano_off else ""),
                fontsize=12, weight='bold')
            from matplotlib.patches import Patch as _Patch2
            ax.legend(handles=[
                _Patch2(facecolor='#C62828', label='q<0.001 ***'),
                _Patch2(facecolor='#E53935', label='q<0.01 **'),
                _Patch2(facecolor='#EF9A9A', label='q<0.05 *'),
                _Patch2(facecolor='#BDBDBD', label='ns'),
            ], fontsize=9, loc='upper left', framealpha=0.9)
            ax.grid(alpha=0.15)
            try:
                fig.tight_layout()
            except Exception:
                pass
            self._embed(fig, sf, f"enrich_volcano")

            # ── Click-to-inspect for enrichment volcano ──
            x_arr = np.array(x_vals)
            y_arr = np.array(y_vals)
            _v_sel_anns = {}  # idx -> annotation

            # Info table below volcano
            _v_info_frame = labelframe(sf,
                text="Selected Enrichments (click points above)")
            _v_info_frame.pack(fill=tk.X, padx=5, pady=(0, 4))

            _v_info_cols = ('Region', 'Label Column', 'Value', 'Sel', 'Sel%',
                            'BG%', 'Enrichment', '95% CI (by study)', 'n_GSE',
                            'n_eff', 'q-value', 'Sig')
            _v_tree = ttk.Treeview(_v_info_frame, columns=_v_info_cols,
                                    show='headings', height=5)
            for _vc in _v_info_cols:
                _v_tree.heading(_vc, text=_vc)
                _v_tree.column(_vc, width=85, anchor=tk.CENTER)
            _v_sb = ttk.Scrollbar(_v_info_frame, orient='vertical',
                                   command=_v_tree.yview)
            _v_tree.config(yscrollcommand=_v_sb.set)
            _v_sb.pack(side=tk.RIGHT, fill=tk.Y)
            _v_tree.pack(fill=tk.BOTH, expand=True)
            _v_tree.tag_configure("sig3", background="#FFCDD2")
            _v_tree.tag_configure("sig2", background="#FFE0B2")
            _v_tree.tag_configure("sig1", background="#FFF9C4")

            def _vpick(event):
                if event.mouseevent.dblclick:
                    for _a in _v_sel_anns.values():
                        _a.remove()
                    _v_sel_anns.clear()
                    for _it in _v_tree.get_children():
                        _v_tree.delete(_it)
                    fig.canvas.draw_idle()
                    return

                mx, my = event.mouseevent.xdata, event.mouseevent.ydata
                if mx is None or my is None:
                    return

                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                xr = (xlim[1] - xlim[0]) or 1
                yr = (ylim[1] - ylim[0]) or 1
                d = ((x_arr - mx)/xr)**2 + ((y_arr - my)/yr)**2
                ci = int(np.argmin(d))

                shift = bool(event.mouseevent.key == 'shift')

                if ci in _v_sel_anns:
                    _v_sel_anns[ci].remove()
                    del _v_sel_anns[ci]
                    for _it in _v_tree.get_children():
                        if _v_tree.set(_it, 'Value') == _tr(volcano_rows[ci]['Value'], 18):
                            _v_tree.delete(_it)
                            break
                else:
                    if not shift:
                        for _a in _v_sel_anns.values():
                            _a.remove()
                        _v_sel_anns.clear()
                        for _it in _v_tree.get_children():
                            _v_tree.delete(_it)

                    r = volcano_rows[ci]
                    label = f"{_tr(r['Value'], 22)}\n{r['Label Column']}"
                    _ann = ax.annotate(
                        label, (x_arr[ci], y_arr[ci]),
                        fontsize=7, fontweight='bold',
                        ha='center', va='bottom',
                        xytext=(0, 8), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='#FFEB3B', edgecolor='#333',
                                  alpha=0.9),
                        arrowprops=dict(arrowstyle='->',
                                        color='#333', lw=0.8))
                    _v_sel_anns[ci] = _ann

                    sig_tag = ("sig3" if r['Sig'] == '***' else
                               "sig2" if r['Sig'] == '**' else
                               "sig1" if r['Sig'] == '*' else '')
                    _v_gse = r.get('n_gse')
                    _v_tree.insert('', tk.END, values=(
                        r.get('Region', ''),
                        r.get('Label Column', ''),
                        _tr(r.get('Value', ''), 22),
                        f"{r.get('a', 0)}/{r.get('n_sel', 0)}",
                        f"{r.get('Sel%', 0):.1f}%",
                        f"{r.get('BG%', 0):.1f}%",
                        f"{r.get('Enrichment', 0):.2f}",
                        _fmt_ci(r),
                        "?" if _v_gse is None else _v_gse,
                        _fmt_neff(r),
                        f"{r.get('padj', float('nan')):.2e}",
                        r.get('Sig', ''),
                    ), tags=(sig_tag,))

                fig.canvas.draw_idle()

            fig.canvas.mpl_connect('pick_event', _vpick)

        # ════════════════════════════════════════════════════════════
        #  TABLE: Full results (sortable by p-value)
        # ════════════════════════════════════════════════════════════
        ttk.Separator(sf, orient='horizontal').pack(fill=tk.X, pady=6)
        sig_rows = [r for r in all_rows if r['Sig'] != 'ns']
        n_ns = len(all_rows) - len(sig_rows)
        ttk.Label(sf, text=f"> Significant Enrichments Table  ({len(sig_rows)} significant, "
                            f"{n_ns} non-significant hidden)",
                  font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, padx=10, pady=(4, 2))

        tbl_frame = ttk.Frame(sf)
        tbl_frame.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)

        tcols = ("Region", "Column", "Value", "Sel", "Sel%", "BG%",
                 "Enrichment", "95% CI (by study)", "n_GSE", "n_eff",
                 "p-value", "q-value", "Sig")
        tree = ttk.Treeview(tbl_frame, columns=tcols, show="headings", height=20)
        widths = {"Region": 140, "Column": 110, "Value": 170, "Sel": 80,
                  "Sel%": 60, "BG%": 60, "Enrichment": 85,
                  "95% CI (by study)": 120, "n_GSE": 55, "n_eff": 70,
                  "p-value": 90, "q-value": 90, "Sig": 45}
        for c in tcols:
            tree.heading(c, text=c)
            tree.column(c, width=widths.get(c, 90), anchor='center')

        vsb = ttk.Scrollbar(tbl_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tree.tag_configure("sig3", background="#FFCDD2")
        tree.tag_configure("sig2", background="#FFE0B2")
        tree.tag_configure("sig1", background="#FFF9C4")
        # a hit carried by one or two studies is not replicated evidence
        tree.tag_configure("thin", background=_ROW_MUTED, foreground=AERO['muted'])

        for row in sig_rows:
            enr_str = f"{row['Enrichment']:.2f}" if row['Enrichment'] != float('inf') else "INF"
            p_str = f"{row['p-value']:.2e}" if row['p-value'] < 0.001 else f"{row['p-value']:.4f}"
            q = row.get('padj', float('nan'))
            q_str = ("n/a" if not np.isfinite(q) else
                     f"{q:.2e}" if q < 0.001 else f"{q:.4f}")
            tag = "sig3" if row['Sig'] == "***" else "sig2" if row['Sig'] == "**" else "sig1"
            if _is_thin(row):
                tag = "thin"
            n_gse = row.get('n_gse')
            tree.insert("", tk.END, values=(
                row['Region'], row['Label Column'], _tr(row['Value'], 35),
                f"{row['a']}/{row['n_sel']}",
                f"{row['Sel%']:.1f}%", f"{row['BG%']:.1f}%",
                enr_str, _fmt_ci(row), "?" if n_gse is None else n_gse,
                _fmt_neff(row), p_str, q_str, row['Sig']
            ), tags=(tag,))

        # The tree shortens a long label so the column fits; the export must
        # not inherit that. A CSV is data, and a value cut to "Non-alcoholic
        # Fatty Liver Disease;.." cannot be joined back to the labels it came
        # from or told apart from its neighbours. Attach the full frame, which
        # `export_window` prefers over scraping the widget.
        if sig_rows:
            tree._export_frame = pd.DataFrame([{
                "Region": r['Region'], "Column": r['Label Column'],
                "Value": r['Value'],
                "Sel": f"{r['a']}/{r['n_sel']}",
                "Sel%": round(float(r['Sel%']), 1),
                "BG%": round(float(r['BG%']), 1),
                "Enrichment": r['Enrichment'],
                "95% CI (by study)": _fmt_ci(r),
                "n_GSE": r.get('n_gse'),
                "n_eff": _fmt_neff(r),
                "p-value": r['p-value'],
                "q-value": r.get('padj', float('nan')),
                "Sig": r['Sig'],
            } for r in sig_rows])

        if sig_rows:
            _foot = ttk.Label(sf,
                      text="Greyed rows are not replicated across studies (<3 contributing GSEs, "
                           "or the study-bootstrap CI still covers 1.0) - treat them as "
                           "hypothesis-generating regardless of their q-value.  "
                           "n_eff is the raw selection size after correcting for study clumping; "
                           "'(30x)' means the count is worth 30x fewer independent samples.",
                      style='Footnote.TLabel', justify=tk.LEFT)
            _wrap_to_parent(_foot)
            _foot.pack(anchor=tk.W, padx=10, pady=(2, 6))
        else:
            ttk.Label(sf, text="No significantly enriched labels found (FDR q<0.05).\n"
                               "Check the Frequency Analysis tab for full label breakdown.",
                      style='Empty.TLabel').pack(pady=10)

    def _platform_label_series(self, lcol):
        """Platform-wide GSM -> label value for one column (upper-cased index).

        The platform's own frame comes first: a label file can describe samples
        the platform never measured, and those samples cannot be inside or
        outside a box drawn on expression, so admitting them would make the
        box look rarer than it is.
        """
        src = None
        for df in (self._tot(r['label']) for r in self.regions):
            if df is not None and not df.empty and 'GSM' in df.columns and lcol in df.columns:
                src = df
                break
        if src is None:
            plat = self.platform_labels_df
            if (plat is not None and not plat.empty
                    and lcol in plat.columns and 'GSM' in plat.columns):
                src = plat
        if src is None:
            return None
        s = src[['GSM', lcol]].dropna()
        s = pd.Series(s[lcol].astype(str).values,
                      index=s['GSM'].astype(str).str.upper())
        return s[~s.index.duplicated()]

    _CMP_BOOT = 300

    def _render_comparison_tab(self):
        for w in self.t_cmp.winfo_children():
            w.destroy()
        for k in [k for k in self.figs if k.startswith("cmp_")]:
            try:
                plt.close(self.figs.pop(k))
            except Exception:
                self.figs.pop(k, None)
            self.canvases.pop(k, None)
            self.toolbars.pop(k, None)
        self._cmp_result = None

        label_cols = self._enrich_label_cols()
        if len(self.regions) < 2:
            ttk.Label(self.t_cmp,
                      text="Comparing regions needs at least two of them.\n\n"
                           "Brush a second gene range and re-open this window.",
                      style='Empty.TLabel', justify=tk.LEFT).pack(pady=40)
            return
        if not label_cols:
            ttk.Label(self.t_cmp,
                      text="No label columns available to compare regions on.\n\n"
                           "Load labels from file or run classification first.",
                      style='Empty.TLabel', justify=tk.LEFT).pack(pady=40)
            return

        ttk.Label(self.t_cmp,
                  text=f"Region Comparison - {len(self.regions)} regions against each other",
                  style='Section.TLabel').pack(fill=tk.X, padx=10, pady=(8, 2))
        _hint = ttk.Label(self.t_cmp,
                  text="The Enrichment tab tests each region against the rest of the platform, so "
                       "two regions can both be 'enriched for Brain' and still be indistinguishable. "
                       "This tab compares them directly: one FDR correction over the whole "
                       "regions x values grid, a pairwise test that carries the Jaccard overlap on "
                       "every row, and a heterogeneity test whose variance is inflated by each "
                       "region's design effect so a label confined to four studies is not reported "
                       "as region-specific.\n"
                       "The AUC column is cross-fitted with folds split by STUDY - it is the answer "
                       "to 'would this region still look different in studies the model never saw?'.",
                  style='Hint.TLabel', justify=tk.LEFT)
        _wrap_to_parent(_hint)
        _hint.pack(fill=tk.X, padx=10, pady=(0, 4))

        ctrl = ttk.Frame(self.t_cmp)
        ctrl.pack(fill=tk.X, padx=10, pady=4)
        ttk.Label(ctrl, text="Label column:", style='Field.TLabel').pack(side=tk.LEFT)
        col_cb = ttk.Combobox(ctrl, textvariable=self.cmp_col, values=label_cols,
                              width=22, state='readonly')
        col_cb.pack(side=tk.LEFT, padx=(4, 12))
        if self.cmp_col.get() not in label_cols:
            self.cmp_col.set(label_cols[0])

        body = ScrollableCanvasFrame(self.t_cmp)
        body.pack(fill=tk.BOTH, expand=True)
        self._cmp_body = body

        # Regions on the same platform share a background and can be put in one
        # grid. Regions on different platforms cannot: each has its own
        # background, and a grid over them would be a table of one platform's
        # samples with the others' regions drawn on it. Those two cases get
        # different buttons rather than one button that quietly does the wrong
        # thing on the second.
        pooled = self._pooled_groups()
        same_plat = len({str(r.get('platform') or "") for r in self.regions}) == 1

        if same_plat:
            ttk.Button(ctrl, text="Compare regions", style='Action.TButton',
                       command=lambda: self._run_comparison(body)).pack(side=tk.LEFT)
        if pooled:
            genes = sorted(pooled)
            ttk.Label(ctrl, text="Gene:", style='Field.TLabel').pack(side=tk.LEFT)
            gene_cb = ttk.Combobox(ctrl, textvariable=self.cmp_gene, values=genes,
                                   width=14, state='readonly')
            gene_cb.pack(side=tk.LEFT, padx=(4, 12))
            if self.cmp_gene.get() not in genes:
                self.cmp_gene.set(genes[0])
            ttk.Button(ctrl, text="Pool across platforms", style='Action.TButton',
                       command=lambda: self._run_pooled(body)).pack(side=tk.LEFT)

        self._cmp_ai_btn = ttk.Button(
            ctrl, text="Interpret with AI", style='Secondary.TButton',
            state=tk.DISABLED, command=lambda: self._interpret_comparison(body))
        self._cmp_ai_btn.pack(side=tk.LEFT, padx=(8, 0))

        if pooled:
            msg = ("Pick a label column and a gene, then press 'Pool across "
                   "platforms'.\n\n"
                   "The regions here were brushed on more than one platform, so each "
                   "platform is tested against its own background and only the effect "
                   "sizes are combined. Pooling the samples instead would build a "
                   "corpus nobody assembled, in which a label can be enriched on every "
                   "platform and depleted overall.")
            if not same_plat:
                msg += ("\n\nThe regions x values grid is not offered: it needs one "
                        "shared background, and these regions do not have one.")
        elif same_plat:
            msg = ("Pick a label column and press 'Compare regions'.\n\n"
                   "Running is not automatic: every cell and every pair carries a "
                   "confidence interval bootstrapped over studies, which is the slow step.")
        else:
            msg = ("These regions span several platforms and no gene was brushed on "
                   "more than one of them, so there is nothing to compare.\n\n"
                   "Regions on different platforms have different backgrounds, which "
                   "rules out the shared grid, and regions on different genes are "
                   "different questions, which rules out pooling. Brush the same gene "
                   "on a second platform to compare, or open this window from a single "
                   "platform.")
        ttk.Label(body.scrollable_frame, text=msg,
                  style='Empty.TLabel', justify=tk.LEFT).pack(pady=30, padx=10)

    def _pooled_groups(self):
        """{gene: {platform: region}} for genes brushed on two or more platforms.

        This is the unit the cross-platform comparison is defined on: one gene,
        one brush rule, each platform answering separately. A gene brushed twice
        on the same platform is two different questions and cannot be a stratum
        pair, so the later region wins and the platform still contributes once.
        """
        by_gene = {}
        for r in self.regions:
            plat = str(r.get('platform') or "")
            gene = str(r.get('gene') or r.get('label') or "")
            if not plat or not gene:
                continue
            by_gene.setdefault(gene, {})[plat] = r
        return {g: p for g, p in by_gene.items() if len(p) >= 2}

    def _pooled_inputs(self, gene, lcol):
        """{platform: stratum} for ``pooled_label_enrichment``.

        Each platform brings its own samples, its own background and its own
        study ids. Nothing is aligned across platforms and nothing is
        concatenated: the samples of a microarray corpus and of a single-cell
        census are not rows of one table, and the whole point of stratifying is
        that they are never treated as if they were.
        """
        groups_of = self._gse_map() or {}
        strata = {}
        for plat, region in (self._pooled_groups().get(gene) or {}).items():
            total = self._tot(region['label'])
            if total is None or total.empty or 'GSM' not in total.columns:
                continue
            if lcol not in total.columns:
                continue
            frame = total[['GSM', lcol]].dropna()
            if frame.empty:
                continue
            gsms = frame['GSM'].astype(str).str.strip().str.upper()
            keep = ~gsms.duplicated()
            gsms = gsms[keep]
            sel = self._sel_gsms(region)
            in_region = gsms.isin(sel).to_numpy()
            if not in_region.any() or in_region.all():
                continue
            grp = [groups_of.get(g) for g in gsms]
            n_known = sum(1 for x in grp if x)
            strata[plat] = {
                "in_region": in_region,
                "labels": frame[lcol][keep].astype(str).to_numpy(dtype=object),
                "groups": ([x if x else "_unknown" for x in grp]
                           if n_known >= len(grp) * 0.5 else None),
                "technology": self._tech_label(plat).lower() or "unknown",
            }
        return strata if len(strata) >= 2 else None

    def _comparison_inputs(self, lcol):
        """(masks, labels, groups, label_frame) aligned on one platform universe.

        Returns ``None`` when fewer than two regions survive - comparing one
        region against itself is not a question.
        """
        series = self._platform_label_series(lcol)
        if series is None or series.empty:
            return None
        universe = series.index
        gse_map = self._gse_map() or {}

        masks = {}
        for r in self.regions:
            sel = self._sel_gsms(r)
            m = universe.isin(sel).astype(bool)
            if m.sum() >= 10:
                masks[r['label']] = m
        if len(masks) < 2:
            return None

        groups = None
        if gse_map:
            g = [gse_map.get(k) for k in universe]
            if sum(1 for x in g if x) >= len(g) * 0.5:
                groups = np.array([x if x else "_unknown" for x in g], dtype=object)

        frame = pd.DataFrame(index=range(len(universe)))
        for c in self._enrich_label_cols():
            s = self._platform_label_series(c)
            if s is None or s.empty:
                continue
            frame[c] = s.reindex(universe).values
        if frame.empty:
            frame[lcol] = series.values

        return masks, series.values, groups, frame

    def _run_pooled(self, body):
        from genevariate.core.analysis import pooled_label_enrichment

        body.clear()
        sf = body.scrollable_frame
        lcol, gene = self.cmp_col.get(), self.cmp_gene.get()
        self._cmp_result = None
        try:
            self._cmp_ai_btn.configure(state=tk.DISABLED)
        except Exception:
            pass

        strata = self._pooled_inputs(gene, lcol)
        if strata is None:
            ttk.Label(sf,
                      text=f"'{lcol}' is not available on at least two of the "
                           f"platforms {gene} was brushed on.\n\nPooling needs the "
                           f"same label column on both sides; a value that exists on "
                           f"one platform only is a single-platform result and is "
                           f"reported as one on the Enrichment tab.",
                      style='Empty.TLabel', justify=tk.LEFT).pack(pady=30, padx=10)
            return

        self.configure(cursor="watch")
        self.update_idletasks()
        try:
            res = pooled_label_enrichment(strata,
                                          max_values=_lim('comparison_values'))
        except Exception as e:
            ttk.Label(sf, text=f"Could not pool the platforms:\n\n{e}",
                      style='Error.TLabel', justify=tk.LEFT).pack(pady=20, padx=10)
            return
        finally:
            self.configure(cursor="")

        res["lcol"], res["gene"] = lcol, gene
        self._cmp_result = {"lcol": lcol, "pooled": res}
        try:
            self._cmp_ai_btn.configure(state=tk.NORMAL)
        except Exception:
            pass
        self._draw_pooled(sf, res)
        self._log(f"[Pooled] {gene} / {lcol}: {len(res['platforms'])} platforms, "
                  f"{len(res['rows'])} values, "
                  f"{sum(1 for r in res['rows'] if np.isfinite(r['q']) and r['q'] < 0.05)}"
                  f" pass FDR")

    def _draw_pooled(self, sf, res):
        rows, plats = res["rows"], res["platforms"]
        techs = res["technologies"]
        cross = res["cross_technology"]

        n_sig = sum(1 for r in rows if np.isfinite(r["q"]) and r["q"] < 0.05)
        ttk.Label(sf,
                  text=f"{res['gene']} - {res['lcol']}  |  {len(plats)} platforms, "
                       f"{len(rows)} label values, {n_sig} pass BH-FDR on the "
                       f"pooled effect",
                  font=("Segoe UI", 9, "bold"),
                  foreground=AERO['green_dark']).pack(fill=tk.X, padx=10, pady=(6, 2))
        ttk.Label(sf,
                  text="Strata: " + ",  ".join(
                      f"{p} ({techs.get(p, 'unknown')})" for p in plats),
                  style='Footnote.TLabel').pack(anchor=tk.W, padx=10)

        _hd = ttk.Label(sf,
                text="Each platform is tested against its own background; only the log odds "
                     "ratios are combined, by DerSimonian-Laird random effects, with each "
                     "platform's variance charged its design effect first. The samples are "
                     "never concatenated - three corpora differ in label composition by "
                     "construction, and a label can be enriched on every platform and "
                     "depleted in the pool of all three.",
                style='Hint.TLabel', justify=tk.LEFT)
        _wrap_to_parent(_hd)
        _hd.pack(anchor=tk.W, padx=10, pady=(2, 6))

        if cross:
            _ct = ttk.Label(sf,
                    text="These platforms use different technologies. Read agreement and "
                         "disagreement differently: if every technology puts the label the "
                         "same way, that is evidence, because three unrelated measurement "
                         "scales are unlikely to agree by accident. If they disagree, that "
                         "is not evidence of biology - a monotone rescaling preserves a "
                         "quantile but not which samples sit inside it, so the same brush "
                         "rule selects different samples on different technologies.",
                    style='Caution.TLabel', justify=tk.LEFT)
            _wrap_to_parent(_ct)
            _ct.pack(anchor=tk.W, padx=10, pady=(0, 6))

        # ── forest of the pooled effects ──────────────────────────────
        shown = [r for r in rows if r["k"] >= 2][:20]
        if shown:
            ttk.Separator(sf, orient='horizontal').pack(fill=tk.X, pady=6)
            ttk.Label(sf, text="Pooled effect per label value",
                      font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, padx=10,
                                                          pady=(4, 2))
            fig = Figure(figsize=(7.5, max(2.4, 0.34 * len(shown) + 1.2)), dpi=100)
            ax = fig.add_subplot(111)
            # Same routine the assistant calls.
            from genevariate.core.analysis.figures import draw_pooled_forest
            draw_pooled_forest(ax, shown, gene=res['gene'],
                               label_column=res['lcol'], palette=AERO)
            fig.tight_layout()
            self._embed(fig, sf, "cmp_pooled")

        # ── the numbers ───────────────────────────────────────────────
        ttk.Separator(sf, orient='horizontal').pack(fill=tk.X, pady=6)
        ttk.Label(sf, text="Do the platforms agree?",
                  font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, padx=10,
                                                      pady=(4, 2))
        tframe = ttk.Frame(sf)
        tframe.pack(fill=tk.X, padx=10, pady=(0, 4))
        cols = ("Value", "k", "Pooled OR", "95% CI", "q", "I2", "q het", "Read")
        tree = ttk.Treeview(tframe, columns=cols, show="headings",
                            height=min(14, max(3, len(rows))))
        for c, w in zip(cols, (170, 40, 90, 130, 90, 65, 90, 280)):
            tree.heading(c, text=c)
            tree.column(c, width=w, minwidth=40,
                        anchor='center' if c != "Read" else 'w')
        tree.tag_configure("agree", background="#E3F2E1")
        tree.tag_configure("split", background="#FFE0B2")
        tree.tag_configure("thin", background=_ROW_MUTED,
                           foreground=AERO['muted'])
        tree.pack(fill=tk.X)

        for r in rows:
            if r["k"] < 2:
                read = ("only " + ", ".join(r["per_platform"])
                        + " could test this; "
                        + "; ".join(f"{p}: {why}"
                                    for p, why in r["dropped"].items()))
                tag = "thin"
            elif np.isfinite(r["q_het"]) and r["q_het"] < 0.05 and r["i2"] >= 50:
                read = ("the platforms disagree"
                        + (" - not interpretable across technologies"
                           if cross else " about this label"))
                tag = "split"
            elif r["concordant"]:
                read = f"all {r['k']} platforms point the same way"
                tag = "agree"
            else:
                read = "spread is within sampling noise"
                tag = "agree"
            tree.insert("", tk.END, values=(
                _tr(r["value"], 28), r["k"],
                f"{r['pooled_or']:.2f}",
                f"{r['ci_low']:.2f} - {r['ci_high']:.2f}",
                "n/a" if not np.isfinite(r["q"]) else
                (f"{r['q']:.2e}" if r["q"] < 0.001 else f"{r['q']:.4f}"),
                "n/a" if not np.isfinite(r["i2"]) else f"{r['i2']:.0f}%",
                "n/a" if not np.isfinite(r["q_het"]) else
                (f"{r['q_het']:.2e}" if r["q_het"] < 0.001 else f"{r['q_het']:.4f}"),
                read,
            ), tags=(tag,))

        _tf = ttk.Label(sf,
                text="k is how many platforms could answer. A value carried by every sample "
                     "of a platform has no odds ratio there and is dropped rather than "
                     "counted as disagreement - a liver-only single-cell census cannot be "
                     "asked about tissue, and saying so is not the same as saying it "
                     "contradicts the arrays. q and q het are corrected separately: "
                     "'is it enriched' and 'do the platforms agree' are two questions, "
                     "and one FDR budget over both would answer neither.",
                style='Footnote.TLabel', justify=tk.LEFT)
        _wrap_to_parent(_tf)
        _tf.pack(anchor=tk.W, padx=10, pady=(2, 8))

        ttk.Separator(sf, orient='horizontal').pack(fill=tk.X, pady=6)
        ttk.Label(sf, text="Interpretation",
                  font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, padx=10,
                                                      pady=(4, 2))
        self._cmp_text = tk.Text(sf, height=14, wrap=tk.WORD, font=MONO_FONT,
                                 bg="#FFFFFF", fg=AERO['text'], relief=tk.FLAT,
                                 highlightthickness=1,
                                 highlightbackground=AERO['border_soft'],
                                 padx=10, pady=8)
        self._cmp_text.pack(fill=tk.X, padx=10, pady=(0, 10))
        self._cmp_text.insert("1.0", self._cmp_facts(self._cmp_result))
        self._cmp_text.configure(state=tk.DISABLED)

    def _run_comparison(self, body):
        from genevariate.core.analysis import (
            cluster_regions, enrichment_matrix, heterogeneity,
            pairwise_differential, region_separability,
        )

        body.clear()
        sf = body.scrollable_frame
        lcol = self.cmp_col.get()
        self._cmp_result = None
        try:
            self._cmp_ai_btn.configure(state=tk.DISABLED)
        except Exception:
            pass

        prepared = self._comparison_inputs(lcol)
        if prepared is None:
            ttk.Label(sf,
                      text=f"Not enough labelled samples in at least two regions "
                           f"for '{lcol}'.\n\nEach region needs 10 samples carrying "
                           f"that label before a comparison means anything.",
                      style='Empty.TLabel', justify=tk.LEFT).pack(pady=30, padx=10)
            return
        masks, labels, groups, frame = prepared

        self.configure(cursor="watch")
        self.update_idletasks()
        try:
            matrix = enrichment_matrix(masks, labels, groups,
                                       max_values=_lim('comparison_values'),
                                       n_boot=self._CMP_BOOT)
            order = cluster_regions(matrix)["order"]
            het = heterogeneity(masks, labels, groups,
                                values=matrix["values"])
            pairs = pairwise_differential(masks, labels, groups,
                                          values=matrix["values"],
                                          n_boot=self._CMP_BOOT)
            try:
                sep = region_separability(masks, frame, groups)
            except Exception as e:
                self._log(f"[!] region separability unavailable: {e}")
                sep = None
        except Exception as e:
            self.configure(cursor="")
            ttk.Label(sf, text=f"Could not compare the regions:\n\n{e}",
                      style='Error.TLabel', justify=tk.LEFT).pack(pady=20, padx=10)
            return
        finally:
            self.configure(cursor="")

        self._cmp_result = {"lcol": lcol, "matrix": matrix, "order": order,
                            "hetero": het, "pairs": pairs, "sep": sep}
        try:
            self._cmp_ai_btn.configure(state=tk.NORMAL)
        except Exception:
            pass

        self._draw_comparison(sf, self._cmp_result)
        self._log(f"[Comparison] {lcol}: {len(matrix['regions'])} regions x "
                  f"{len(matrix['values'])} values, "
                  f"{int(np.nansum(matrix['q'] < 0.05))} cells pass FDR")

    def _draw_comparison(self, sf, res):
        matrix, order = res["matrix"], res["order"]
        regions, values = matrix["regions"], matrix["values"]
        lift, q, a = matrix["log2_lift"], matrix["q"], matrix["a"]

        n_sig = int(np.nansum((q < 0.05) & (a >= matrix["min_count"])))
        head = (f"OK {n_sig} of {int(np.isfinite(q).sum())} cells pass BH-FDR "
                f"over the whole grid  |  {matrix['n_labelled']:,} labelled samples")
        if groups_missing := (not np.isfinite(matrix["rho"]).any()):
            head += "  |  WARNING no study column - clumping cannot be corrected"
        ttk.Label(sf, text=head, font=("Segoe UI", 9, "bold"),
                  foreground=AERO['danger'] if groups_missing else AERO['green_dark']
                  ).pack(fill=tk.X, padx=10, pady=(6, 2))

        # ── the grid ──────────────────────────────────────────────────
        ttk.Separator(sf, orient='horizontal').pack(fill=tk.X, pady=6)
        ttk.Label(sf, text="Which region does each label belong to?",
                  font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, padx=10, pady=(4, 2))

        rows_o = [regions[i] for i in order]
        L = lift[order][:, :]
        Q = q[order][:, :]
        A = a[order][:, :]
        cap = float(np.nanpercentile(np.abs(L[np.isfinite(L)]), 98)) if np.isfinite(L).any() else 1.0
        cap = max(cap, 0.5)

        fig_w = max(7.0, 0.55 * len(values) + 3.0)
        fig_h = max(2.6, 0.45 * len(rows_o) + 1.8)
        fig = Figure(figsize=(fig_w, fig_h), dpi=100)
        ax = fig.add_subplot(111)
        im = ax.imshow(L, cmap="RdBu_r", vmin=-cap, vmax=cap, aspect="auto")
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([_tr(v, 18) for v in values], rotation=45,
                           ha="right", fontsize=8)
        ax.set_yticks(range(len(rows_o)))
        ax.set_yticklabels([_tr(r, 26) for r in rows_o], fontsize=8)
        _cell_grid(ax, L.shape[0], L.shape[1])
        # a cell only earns a marker if it also has samples behind it
        for i in range(L.shape[0]):
            for j in range(L.shape[1]):
                if np.isfinite(Q[i, j]) and Q[i, j] < 0.05 and A[i, j] >= matrix["min_count"]:
                    ax.text(j, i, "*", ha="center", va="center",
                            fontsize=11, color="#111", fontweight="bold")
        ax.set_title(f"log2 lift vs platform - {res['lcol']}  "
                     f"(rows clustered; * = FDR q<0.05)", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="log2 lift")
        fig.tight_layout()
        self._embed(fig, sf, "cmp_matrix")

        _cap = ttk.Label(sf,
                text="Rows are ordered by clustering their lift profiles, so regions that say the "
                     "same thing sit together - adjacent rows are redundant brushings, not "
                     "independent evidence. Blank-looking cells are labels the region simply "
                     "does not contain; they are Haldane-corrected rather than infinite.",
                style='Footnote.TLabel', justify=tk.LEFT)
        _wrap_to_parent(_cap)
        _cap.pack(anchor=tk.W, padx=10, pady=(2, 6))

        # ── heterogeneity ─────────────────────────────────────────────
        ttk.Separator(sf, orient='horizontal').pack(fill=tk.X, pady=6)
        ttk.Label(sf, text="Is the label region-specific, or the same everywhere?",
                  font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, padx=10, pady=(4, 2))
        hframe = ttk.Frame(sf); hframe.pack(fill=tk.X, padx=10, pady=(0, 4))
        hcols = ("Value", "I2", "Q", "df", "q", "Pooled OR", "Max deff", "Read")
        htree = ttk.Treeview(hframe, columns=hcols, show="headings",
                             height=min(10, max(3, len(res["hetero"]))))
        for c, w in zip(hcols, (180, 70, 90, 45, 95, 90, 80, 240)):
            htree.heading(c, text=c)
            htree.column(c, width=w, anchor='center' if c != "Read" else 'w')
        htree.tag_configure("specific", background="#FFE0B2")
        htree.tag_configure("flat", background=_ROW_MUTED, foreground=AERO['muted'])
        htree.pack(fill=tk.X)
        for h in res["hetero"]:
            deffs = [v.get("deff", 1.0) for v in h["per_region"].values()]
            mx = max(deffs) if deffs else 1.0
            # One Q test per label, so gate on the BH-adjusted q like every
            # other table here - on the raw p, two of forty labels would be
            # called region-specific by chance.
            specific = h["i2"] > 50 and np.isfinite(h["q"]) and h["q"] < 0.05
            if specific:
                read = "belongs to some regions and not others"
            elif mx > 3:
                read = f"flat once study clumping is charged for ({mx:.0f}x)"
            else:
                read = "no region tells a different story"
            htree.insert("", tk.END, values=(
                _tr(h["value"], 30), f"{h['i2']:.1f}%", f"{h['q_stat']:.1f}",
                h["df"],
                "n/a" if not np.isfinite(h["q"]) else
                (f"{h['q']:.2e}" if h["q"] < 0.001 else f"{h['q']:.4f}"),
                f"{h['pooled_or']:.2f}", f"{mx:.1f}x", read,
            ), tags=("specific" if specific else "flat",))

        _hf = ttk.Label(sf,
                text="I2 is the share of the spread between regions that is not sampling noise. "
                     "Each region's variance is multiplied by its design effect first, so a label "
                     "that lives in four studies cannot borrow significance from the samples those "
                     "studies happen to contain - that correction is the difference between "
                     "reading biology and reading a batch.",
                style='Footnote.TLabel', justify=tk.LEFT)
        _wrap_to_parent(_hf)
        _hf.pack(anchor=tk.W, padx=10, pady=(2, 6))

        # ── pairwise ──────────────────────────────────────────────────
        ttk.Separator(sf, orient='horizontal').pack(fill=tk.X, pady=6)
        sig_pairs = [r for r in res["pairs"]
                     if np.isfinite(r["q"]) and r["q"] < 0.05]
        ttk.Label(sf, text=f"Region against region ({len(sig_pairs)} of "
                           f"{len(res['pairs'])} pairs separate at q<0.05)",
                  font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, padx=10, pady=(4, 2))
        pframe = ttk.Frame(sf); pframe.pack(fill=tk.X, padx=10, pady=(0, 4))
        pcols = ("Value", "Region A", "Rate A", "Region B", "Rate B",
                 "OR", "95% CI (by study)", "q", "Overlap J")
        ptree = ttk.Treeview(pframe, columns=pcols, show="headings",
                             height=min(16, max(3, len(sig_pairs) or 3)))
        for c, w in zip(pcols, (150, 150, 70, 150, 70, 70, 130, 90, 80)):
            ptree.heading(c, text=c)
            ptree.column(c, width=w, anchor='center')
        pvsb = ttk.Scrollbar(pframe, orient="vertical", command=ptree.yview)
        ptree.configure(yscrollcommand=pvsb.set)
        pvsb.pack(side=tk.RIGHT, fill=tk.Y)
        ptree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ptree.tag_configure("clean", background="#C8E6C9")
        # regions that share most of their samples are not two groups
        ptree.tag_configure("overlapping", background=_ROW_MUTED,
                            foreground=AERO['muted'])
        for r in sig_pairs[:200]:
            ci = ("n/a" if not (np.isfinite(r["ci_low"]) and np.isfinite(r["ci_high"]))
                  else f"{np.exp(r['ci_low']):.2f} - {np.exp(r['ci_high']):.2f}")
            ptree.insert("", tk.END, values=(
                _tr(r["value"], 24), _tr(r["region_a"], 24), f"{100*r['rate_a']:.1f}%",
                _tr(r["region_b"], 24), f"{100*r['rate_b']:.1f}%",
                f"{np.exp(r['log_or']):.2f}", ci,
                f"{r['q']:.2e}" if r["q"] < 0.001 else f"{r['q']:.4f}",
                f"{r['jaccard']:.2f}",
            ), tags=("overlapping" if r["jaccard"] > 0.5 else "clean",))
        if not sig_pairs:
            ttk.Label(sf, text="No two regions differ from each other at q<0.05 - "
                              "whatever the platform-wide enrichments say, these "
                              "regions are telling the same story.",
                      style='Empty.TLabel', justify=tk.LEFT).pack(padx=10, pady=8)
        else:
            if len(sig_pairs) > 200:
                ttk.Label(sf,
                          text=f"Showing the 200 strongest of {len(sig_pairs):,} "
                               f"significant pairs.",
                          style='Caution.TLabel',
                          justify=tk.LEFT).pack(anchor=tk.W, padx=10, pady=(2, 0))
            _pf = ttk.Label(sf,
                    text="Greyed rows are region pairs sharing more than half their samples "
                         "(Jaccard > 0.5) - the test still runs, but it is comparing a set with "
                         "itself and the p-value is not evidence of two distinct populations.",
                    style='Footnote.TLabel', justify=tk.LEFT)
            _wrap_to_parent(_pf)
            _pf.pack(anchor=tk.W, padx=10, pady=(2, 6))

        # ── separability ──────────────────────────────────────────────
        sep = res.get("sep")
        if sep and sep.get("regions"):
            ttk.Separator(sf, orient='horizontal').pack(fill=tk.X, pady=6)
            ttk.Label(sf, text="Would the region still look different in unseen studies?",
                      font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, padx=10, pady=(4, 2))
            sframe = ttk.Frame(sf); sframe.pack(fill=tk.X, padx=10, pady=(0, 4))
            scols = ("Region", "AUC", "Folds by GSE", "n in region",
                     "Label that separates it", "Gain")
            stree = ttk.Treeview(sframe, columns=scols, show="headings",
                                 height=min(10, max(3, len(sep["regions"]))))
            for c, w in zip(scols, (200, 70, 100, 100, 220, 80)):
                stree.heading(c, text=c)
                stree.column(c, width=w, anchor='center' if c != scols[4] else 'w')
            stree.tag_configure("real", background="#C8E6C9")
            stree.tag_configure("chance", background=_ROW_MUTED,
                                foreground=AERO['muted'])
            stree.pack(fill=tk.X)
            for name, v in sep["regions"].items():
                if "skipped" in v:
                    stree.insert("", tk.END, values=(
                        _tr(name, 32), "-", "-", v.get("n_pos", "-"),
                        f"skipped: {v['skipped']}", "-"), tags=("chance",))
                    continue
                imp = v.get("importance") or {}
                best = max(imp.items(), key=lambda kv: kv[1]) if imp else ("-", 0.0)
                stree.insert("", tk.END, values=(
                    _tr(name, 32), f"{v['auc']:.3f}",
                    "yes" if v.get("grouped") else "no (samples)",
                    v.get("n_pos", "-"), _tr(str(best[0]), 30),
                    f"+{best[1]:.3f}" if best[1] > 0 else f"{best[1]:.3f}",
                ), tags=("real" if v["auc"] >= 0.65 else "chance",))
            _sf2 = ttk.Label(sf,
                    text="Greyed rows sit near chance: once whole studies are held out, nothing in "
                         "the label profile distinguishes those samples, however significant their "
                         "counts were. 'Gain' is the permutation importance of the named column - "
                         "how much AUC is lost when that column alone is shuffled.",
                    style='Footnote.TLabel', justify=tk.LEFT)
            _wrap_to_parent(_sf2)
            _sf2.pack(anchor=tk.W, padx=10, pady=(2, 6))

        # ── the written read ──────────────────────────────────────────
        ttk.Separator(sf, orient='horizontal').pack(fill=tk.X, pady=6)
        ttk.Label(sf, text="Interpretation",
                  font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, padx=10, pady=(4, 2))
        self._cmp_text = tk.Text(sf, height=14, wrap=tk.WORD, font=MONO_FONT,
                                 bg="#FFFFFF", fg=AERO['text'], relief=tk.FLAT,
                                 highlightthickness=1,
                                 highlightbackground=AERO['border_soft'],
                                 padx=10, pady=8)
        self._cmp_text.pack(fill=tk.X, padx=10, pady=(0, 10))
        self._cmp_text.insert("1.0", self._cmp_facts(self._cmp_result))
        self._cmp_text.configure(state=tk.DISABLED)

    def _cmp_facts(self, res):
        """The computed summary of whichever comparison last ran."""
        if res.get("pooled") is not None:
            from genevariate.core.analysis import summarize_pooled_enrichment
            return summarize_pooled_enrichment(res["pooled"], res["lcol"])
        from genevariate.core.analysis import summarize_comparison
        return summarize_comparison(res["matrix"], res["hetero"],
                                    res["pairs"], res.get("sep"))

    def _interpret_comparison(self, body):
        """Hand the computed summary to gemma4 and ask it to explain, not invent.

        The model never sees the data, only the numbers this tab already
        computed, so the worst it can do is phrase them badly. When no model is
        running the deterministic summary stays on screen unchanged - that text
        is the fallback, not a placeholder for one.
        """
        res, box = self._cmp_result, getattr(self, '_cmp_text', None)
        if not res or box is None:
            return
        facts = self._cmp_facts(res)

        def _set(text):
            try:
                box.configure(state=tk.NORMAL)
                box.delete("1.0", tk.END)
                box.insert("1.0", text)
                box.configure(state=tk.DISABLED)
            except Exception:
                pass

        _set(facts + "\n\n--- asking the model to read this ---\n")
        self._cmp_ai_btn.configure(state=tk.DISABLED, text="Thinking...")

        def _work():
            out, err, model = None, None, "gemma4:e2b"
            try:
                from genevariate.core import llm_client as llm_backend
                model = llm_backend.default_model()
                # Probe whatever backend chat() will use - a served
                # OpenAI-compatible endpoint as readily as a local Ollama.
                ready, why = llm_backend.available(model)
                if not ready:
                    err = f"No model to ask: {why}"
                else:
                    out = llm_backend.chat([
                        {"role": "system", "content": _CMP_SYSTEM_PROMPT},
                        {"role": "user", "content":
                            f"Label column: {res['lcol']}\n\n{facts}"},
                    ], model=model, temperature=0.1, num_predict=700,
                        think=False, timeout=120)
            except Exception as e:
                err = str(e)

            def _done():
                if out and out.strip():
                    _set(f"{facts}\n\n--- {model} reads it as ---\n\n{out.strip()}")
                else:
                    _set(f"{facts}\n\n--- no model available ---\n"
                         f"{err or 'The model returned nothing.'}\n"
                         f"The numbers above are computed, not generated, so nothing is missing.")
                try:
                    self._cmp_ai_btn.configure(state=tk.NORMAL,
                                               text="Interpret with AI")
                except Exception:
                    pass
            try:
                self.after(0, _done)
            except Exception:
                pass

        threading.Thread(target=_work, daemon=True).start()

    # ═══════════════════════════════════════════════════════════════════
    #  TAB 6 - Statistics
    # ═══════════════════════════════════════════════════════════════════
    def _render_stats_tab(self):
        if self.st: self.st.destroy()
        cols = ("Region A", "Region B", "Metric", "Value", "Sig")
        self.st = ttk.Treeview(self.t_stats, columns=cols, show="headings")
        for c, w in zip(cols, [220, 220, 140, 120, 100]):
            self.st.heading(c, text=c); self.st.column(c, width=w)
        sb = ttk.Scrollbar(self.t_stats, orient="vertical", command=self.st.yview)
        self.st.configure(yscrollcommand=sb.set)
        self.st.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        # Read off the filtered frame, not the region's stored values: the
        # picker chooses samples, so a statistic computed over the unfiltered
        # selection would disagree with every plot beside it.
        vals = {}
        for r in self.regions:
            mg = self._sel(r['label'])
            src = (mg[r['column']] if not mg.empty and r['column'] in mg.columns
                   else pd.Series(r['expression_values']))
            vals[r['label']] = pd.to_numeric(pd.Series(src),
                                             errors='coerce').dropna()
        for r in self.regions:
            v = vals[r['label']]
            for m, fn in [("N", lambda x: str(len(x))), ("Mean", lambda x: f"{x.mean():.4f}"),
                          ("Median", lambda x: f"{x.median():.4f}"), ("Std", lambda x: f"{x.std():.4f}"),
                          ("IQR", lambda x: f"{x.quantile(.75)-x.quantile(.25):.4f}")]:
                self.st.insert("", tk.END, values=(r['label'], "-", m, fn(v), ""))

        keys = list(vals.keys())
        if len(keys) >= 2:
            self.st.insert("", tk.END, values=("-" * 18, "-" * 18, "PAIRWISE", "-" * 10, ""))
            for k1, k2 in itertools.combinations(keys, 2):
                d1, d2 = vals[k1], vals[k2]
                try:
                    s, p = ranksums(d1, d2)
                    sig = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"
                    self.st.insert("", tk.END, values=(k1, k2, "Wilcoxon Z", f"{s:.4f}", sig))
                    self.st.insert("", tk.END, values=(k1, k2, "p-value", f"{p:.2e}", sig))
                except: pass
                try:
                    wd = wasserstein_distance(d1, d2)
                    self.st.insert("", tk.END, values=(k1, k2, "Wasserstein", f"{wd:.4f}",
                        "High" if wd > 1 else "Mod" if wd > .5 else "Low"))
                except: pass
                self.st.insert("", tk.END, values=(k1, k2, "Delta-Mean",
                    f"{abs(d1.mean()-d2.mean()):.4f}", ""))

    # ═══════════════════════════════════════════════════════════════════
    #  TAB 6 - Samples (highlight + compare selected)
    #
    #  - Click column header -> set Color By + re-color all plots
    #  - Select rows -> highlights with dynamic colors (by Color By group)
    #  - "Compare Selected" -> opens CompareDistributionsWindow
    # ═══════════════════════════════════════════════════════════════════
    def _render_table_tab(self):
        for w in self.t_table.winfo_children(): w.destroy()
        frames = []
        for r in self.regions:
            m = self._sel(r['label']).copy()
            if m.empty:
                continue
            m['Region'] = r['label']
            m['Platform'] = r.get('platform', '')
            m['Technology'] = self._tech_label(r.get('platform'))
            m['Gene'] = r.get('gene', '')
            frames.append(m)
        if not frames:
            ttk.Label(self.t_table, text="No metadata.", font=("Segoe UI", 11),
                      foreground="gray").pack(pady=30)
            return

        combined = pd.concat(frames, ignore_index=True)
        self._table_df = combined  # keep ref for compare

        # Priority columns
        fixed = ('GSM', 'Region', 'Platform', 'Technology', 'Gene', 'Expression')
        cls_cols = sorted([c for c in combined.columns
                           if c not in fixed and combined[c].dtype == 'object'])
        pri = ['Region', 'Platform', 'Technology', 'Gene', 'GSM', 'series_id',
               'title', 'source_name_ch1'] + cls_cols
        drop = {'contact', 'supplementary_file', 'data_row_count', 'channel_count',
                'status', 'submission_date', 'last_update_date'}
        # dict.fromkeys, not a plain list: 'series_id' is both a priority
        # column and an object column, and a repeated column name gives the
        # table two identical headings and the export a duplicated field.
        ordered = list(dict.fromkeys(c for c in pri if c in combined.columns))
        ordered += [c for c in combined.columns if c not in ordered
                    and c not in drop]
        # The display keeps 25 columns wide enough to read; the export keeps
        # every column, including the ones dropped from the view, because the
        # notice below tells the user to export to get all the data.
        export_cols = ordered + [c for c in combined.columns if c not in ordered]
        cols = ordered[:25]
        self._table_cols = cols
        self._table_export_cols = export_cols

        # ── Top controls bar ──
        ctrl = ttk.Frame(self.t_table)
        ctrl.pack(fill=tk.X, padx=5, pady=(4, 2))

        ttk.Label(ctrl,
                  text="Click column header \u2192 Color By  |  Select rows \u2192 highlight + compare",
                  style='Hint.TLabel').pack(side=tk.LEFT)

        self._sel_count_lbl = ttk.Label(ctrl, text="0 selected", style='Field.TLabel')
        self._sel_count_lbl.pack(side=tk.RIGHT, padx=8)

        btn_bar = ttk.Frame(self.t_table)
        btn_bar.pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(btn_bar, text="Select All", width=11, style="Secondary.TButton",
                   command=self._table_select_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_bar, text="Clear", width=8, style="Secondary.TButton",
                   command=self._table_clear_sel).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_bar, text="Compare Selected", style="Action.TButton",
                   command=self._compare_selected_samples).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btn_bar, text="Highlight Rows", style="Secondary.TButton",
                   command=self._highlight_selection).pack(side=tk.RIGHT, padx=4)

        # ── Treeview ──
        tree_frame = ttk.Frame(self.t_table)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)

        self._sample_tree = ttk.Treeview(tree_frame, columns=cols, show="headings",
                                          selectmode="extended")
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self._sample_tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self._sample_tree.xview)
        self._sample_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        # Stretch columns to fill the panel width when few are shown (removes the
        # dead space on the right); a horizontal scrollbar still handles the case
        # where many gene columns overflow the viewport.
        for c in cols:
            self._sample_tree.heading(c, text=c.replace('_', ' '))
            self._sample_tree.column(c, width=130, minwidth=90, stretch=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self._sample_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._sample_tree._export_frame = combined[export_cols]

        # Store row data for fast lookup
        max_table_rows = 2000
        self._table_iid_to_row = {}
        for row_idx, (_, row) in enumerate(combined.head(max_table_rows).iterrows()):
            vals = [str(row.get(c, ''))[:90] for c in cols]
            iid = self._sample_tree.insert("", tk.END, values=vals)
            self._table_iid_to_row[iid] = row_idx

        if len(combined) > max_table_rows:
            ttk.Label(self.t_table,
                      text=f"Showing {max_table_rows:,} / {len(combined):,} rows. "
                           f"Use Export to get all data.",
                      font=("Segoe UI", 8, "italic"), foreground="gray").pack(pady=2)

        # Click header -> set Color By
        self._sample_tree.bind("<Button-1>", self._on_table_click)
        # Selection change -> update count
        self._sample_tree.bind("<<TreeviewSelect>>", self._on_table_select)

    def _on_table_click(self, event):
        """Click column header -> set as Color By, re-render."""
        region = self._sample_tree.identify_region(event.x, event.y)
        if region == "heading":
            col_id = self._sample_tree.identify_column(event.x)
            col_name = self._sample_tree.column(col_id, "id")
            if col_name and col_name not in ('GSM', 'Region'):
                self.color_column.set(col_name)
                vals = list(self.cc['values'])
                if col_name not in vals: vals.insert(0, col_name); self.cc['values'] = vals
                try: self.app.enqueue_log(f"[Region Analysis] Color column -> {col_name}")
                except: pass
                self._on_color_col_changed()

    def _on_table_select(self, event=None):
        """Update selection count label."""
        sel = self._sample_tree.selection()
        n = len(sel)
        self._sel_count_lbl.config(
            text=f"{n} selected" if n > 0 else "0 selected",
            foreground=AERO['danger'] if n > 0 else AERO['muted']
        )

    def _table_select_all(self):
        self._sample_tree.selection_set(self._sample_tree.get_children())
        self._on_table_select()

    def _table_clear_sel(self):
        self._sample_tree.selection_remove(*self._sample_tree.selection())
        # Remove highlight tags
        for iid in self._sample_tree.get_children():
            self._sample_tree.item(iid, tags=())
        self._on_table_select()

    def _highlight_selection(self):
        """Color-code selected rows by their Color By group value."""
        sel = self._sample_tree.selection()
        if not sel:
            messagebox.showinfo("No Selection", "Select rows first, then highlight.", parent=self)
            return

        ccol = self.color_column.get()
        if not hasattr(self, '_table_df') or self._table_df.empty:
            return

        # Clear all tags first
        for iid in self._sample_tree.get_children():
            self._sample_tree.item(iid, tags=())

        # Find Color By column index
        cols = self._table_cols
        ccol_idx = None
        if ccol and ccol in cols:
            ccol_idx = cols.index(ccol)

        # Group selected rows by Color By value
        groups = {}
        for iid in sel:
            if ccol_idx is not None:
                vals = self._sample_tree.item(iid, 'values')
                grp = vals[ccol_idx] if ccol_idx < len(vals) else 'Unknown'
            else:
                grp = 'Selected'
            groups.setdefault(grp, []).append(iid)

        # Assign colors per group
        uniq_grps = sorted(groups.keys())
        colors = _clrs(max(1, len(uniq_grps)))
        grp_color = {g: colors[i] for i, g in enumerate(uniq_grps)}

        # Apply tags
        for grp, iids in groups.items():
            tag = f"hl_{grp}"
            for iid in iids:
                self._sample_tree.item(iid, tags=(tag,))
            self._sample_tree.tag_configure(tag, background=grp_color[grp],
                                             foreground='white' if grp_color[grp][1:3] < '88' else 'black')

        n_grps = len(uniq_grps)
        self._sel_count_lbl.config(
            text=f"{len(sel)} selected in {n_grps} group{'s' if n_grps > 1 else ''}",
            foreground=AERO['green_dark']
        )

    def _compare_selected_samples(self):
        """Send selected samples to CompareDistributionsWindow grouped by Color By."""
        sel = self._sample_tree.selection()
        if not sel or len(sel) < 2:
            messagebox.showwarning("Selection Needed",
                "Select >=2 samples to compare.\n\n"
                "Tip: Select rows, then click 'Compare Selected'.\n"
                "Samples will be grouped by the current Color By column.",
                parent=self)
            return

        if not hasattr(self, '_table_df') or self._table_df.empty:
            return

        ccol = self.color_column.get()
        cols = self._table_cols

        # Get selected row indices
        sel_indices = [self._table_iid_to_row[iid] for iid in sel if iid in self._table_iid_to_row]
        sel_df = self._table_df.iloc[sel_indices].copy()

        if sel_df.empty:
            return

        # Get the expression column (same for all regions)
        expr_col = self.regions[0]['column']

        # Determine grouping column
        if ccol and ccol != "(none)" and ccol in sel_df.columns:
            grp_col = ccol
        elif 'series_id' in sel_df.columns:
            grp_col = 'series_id'
        elif 'Region' in sel_df.columns:
            grp_col = 'Region'
        else:
            grp_col = None

        # Build data_map for CompareDistributionsWindow
        data_map = {}
        group_gsm_map = {}
        if grp_col:
            sel_df['_group'] = sel_df[grp_col].fillna('N/A').astype(str)
        else:
            sel_df['_group'] = 'Selected'

        label_by_group = {}
        for grp_name, grp_df in sel_df.groupby('_group'):
            expr = pd.to_numeric(grp_df[expr_col], errors="coerce").dropna()
            if expr.empty: continue
            label = f"{_tr(grp_name, 35)} (n={len(expr)})"
            label_by_group[grp_name] = label
            data_map[label] = expr
            group_gsm_map[label] = grp_df['GSM'].tolist() if 'GSM' in grp_df.columns else []

        if len(data_map) < 1:
            messagebox.showinfo("No Data", "Selected samples have no expression data.", parent=self)
            return

        # Background from platform
        bg_map = {}
        bg_df = self.regions[0].get('platform_df', pd.DataFrame())
        if not bg_df.empty and expr_col in bg_df.columns:
            bg_map["Platform"] = pd.to_numeric(bg_df[expr_col], errors="coerce").dropna()

        # Build metadata
        meta_df = sel_df.copy()
        # The Group value has to be the same decorated label the data_map is
        # keyed by. Filing the raw group name here meant the compare window
        # could never match a group to its metadata, so its Color By box and
        # its label-coloured PCA both came up as "No labels available".
        meta_df['Group'] = meta_df['_group'].map(label_by_group).fillna(meta_df['_group'])
        if expr_col in meta_df.columns:
            meta_df = meta_df.rename(columns={expr_col: 'Expression'})

        # Open CompareDistributionsWindow
        try:
            from compare_analysis import CompareDistributionsWindow
        except ImportError:
            try:
                from .compare_analysis import CompareDistributionsWindow
            except ImportError:
                messagebox.showerror("Module Error",
                    "compare_analysis.py not found.", parent=self)
                return

        win = CompareDistributionsWindow(self, self.app,
              title_text=f"Compare Samples ({len(sel)} samples, {len(data_map)} groups by {grp_col or 'All'})")
        win.inject_data(
            data_map=data_map,
            bg_map=bg_map,
            metadata_df=meta_df,
            group_gsm_map=group_gsm_map,
            grouping_col=grp_col
        )
        win.after(200, win._refresh_all_plots)

        try: self.app.enqueue_log(
            f"[Compare] Sent {len(sel)} samples -> {len(data_map)} groups by '{grp_col}'")
        except: pass

    def _open_sample_card(self, pick):
        """Open the record for the sample the user clicked on a rug tick.

        A density curve says how a group is distributed but nothing about the
        one sample sitting out in its tail, which is usually the sample the
        user is squinting at. Clicking its tick answers that.
        """
        from genevariate.gui.windows.sample_card import show_sample_card

        gsm = str(pick.label).strip()
        for source in (self._mc, self._mc_total):
            for label, frame in source.items():
                if frame is None or frame.empty or "GSM" not in frame.columns:
                    continue
                if (frame["GSM"].astype(str) == gsm).any():
                    show_sample_card(self, gsm, frame, region_label=label)
                    return
        show_sample_card(self, gsm)

    def _selection_table(self):
        """Everything known about every sample, for the selection window.

        Built on the first selection rather than on every render: it merges
        the expression matrix with the study index and the label table, which
        is real work on a whole platform and is wasted on a plot the user
        never lassoes. Cached until the labels are replaced.
        """
        stamp = id(self.platform_labels_df)
        cached = getattr(self, "_sel_table_cache", None)
        if cached is not None and cached[0] == stamp:
            return cached[1]

        def _distinct(key):
            out, seen = [], set()
            for r in self.regions:
                f = r.get(key)
                if (f is None or getattr(f, "empty", True)
                        or "GSM" not in f.columns or id(f) in seen):
                    continue
                seen.add(id(f))
                out.append(f)
            return out

        table = pd.DataFrame()
        expr = _distinct("platform_df")
        if expr:
            table = (pd.concat(expr, ignore_index=True, sort=False)
                     .drop_duplicates(subset=["GSM"]))
        for extra in (_distinct("meta_df")
                      + ([self.platform_labels_df]
                         if self.platform_labels_df is not None else [])):
            if (extra is None or getattr(extra, "empty", True)
                    or "GSM" not in extra.columns):
                continue
            extra = extra.drop_duplicates(subset=["GSM"])
            if table.empty:
                table = extra.copy()
                continue
            new = [c for c in extra.columns if c == "GSM" or c not in table.columns]
            table = table.merge(extra[new], on="GSM", how="left")
        self._sel_table_cache = (stamp, table)
        return table

    # ═══════════════════════════════════════════════════════════════════
    #  TAB 8 - Summary
    #
    #  One page per region: the distribution the region was drawn on, ringed
    #  by the results the tabs above already computed for it. A panel whose
    #  result was never computed is left out and the ring re-flows, so the
    #  page never carries an empty frame.
    # ═══════════════════════════════════════════════════════════════════
    def _render_summary_tab(self):
        for w in self.t_summary.winfo_children():
            w.destroy()
        for k in [k for k in self.figs if k.startswith("summary_")]:
            try:
                plt.close(self.figs.pop(k))
            except Exception:
                self.figs.pop(k, None)
            self.canvases.pop(k, None)
            self.toolbars.pop(k, None)

        _hint = ttk.Label(self.t_summary,
                          text="One page per region: the distribution it was drawn on, "
                               "ringed by the results computed for that region. The "
                               "bottom strip carries results fitted on a wider sample "
                               "set and says so in each title. Panels appear only once "
                               "the tab behind them has run.",
                          style='Hint.TLabel', justify=tk.LEFT)
        _wrap_to_parent(_hint)
        _hint.pack(fill=tk.X, padx=10, pady=(6, 2))

        scroll = ScrollableCanvasFrame(self.t_summary)
        scroll.pack(fill=tk.BOTH, expand=True)
        drawn = 0
        for ri, region in enumerate(self.regions):
            fig = self._summary_figure(region)
            if fig is None:
                continue
            self._embed(fig, scroll.scrollable_frame, f"summary_{ri}")
            drawn += 1
        if not drawn:
            ttk.Label(scroll.scrollable_frame,
                      text="No region holds any samples after the current filter.",
                      style='Empty.TLabel').pack(pady=20)

    def _summary_figure(self, region):
        """The page for one region, sized to the panels that have results."""
        sel = self._sel(region.get('label'))
        if sel is None or sel.empty:
            return None

        sides = []
        rows = [r for r in (self._enrich_rows or [])
                if r.get('Region') == region['label'] and r.get('Sig') != 'ns']
        if rows:
            lcol = rows[0]['Label Column']
            col_rows = [r for r in rows if r['Label Column'] == lcol]
            top = col_rows[:_lim('summary_rows')]
            # The panel is one column's rows; what it leaves out is recorded
            # so the panel itself can say so.
            self._sum_hidden = {lcol: len(col_rows) - len(top)}
            sides.append(lambda ax, t=top, c=lcol: self._sum_composition(ax, t, c))
            sides.append(lambda ax, t=top, c=lcol: self._sum_enrichment(ax, t, c))
        studies = self._sum_study_counts(region)
        if studies:
            sides.append(lambda ax, s=studies: self._sum_studies(ax, s))

        # One panel per row. Side by side, a panel's title and its neighbour's
        # tick labels land on each other and the distribution's legend covers
        # whatever sits beside it, which is unreadable however wide the figure
        # is made.
        heights = [1.6] + [1.0] * len(sides)

        # Constrained layout, not tight_layout: the panels differ in how much
        # room their labels need and tight_layout cannot solve that here.
        fig = Figure(figsize=(11.0, 4.6 + 3.2 * len(sides)),
                     layout="constrained")
        gs = fig.add_gridspec(len(heights), 1, height_ratios=heights)
        self._sum_distribution(fig.add_subplot(gs[0, 0]), region)
        for i, draw in enumerate(sides):
            draw(fig.add_subplot(gs[i + 1, 0]))
        title = f"{region['label']}"
        plat = str(region.get('platform') or "")
        if plat:
            title += f"  on  {plat}"
        fig.suptitle(title, fontsize=12, weight='bold')
        return fig

    def _sum_distribution(self, ax, region):
        col = region['column']
        lo, hi = region['range']
        bg = region.get('platform_df', pd.DataFrame())
        _draw_bg(ax, bg, col)
        xr = _bg_range(bg, col)
        mg = self._sel(region['label'])
        ccol = self.color_column.get()
        mode = self.plot_mode.get()
        handles = []
        if ccol and ccol != "(none)" and ccol in mg.columns:
            grps = mg[ccol].fillna("N/A").astype(str)
            tops = list(grps.value_counts().head(_lim('summary_groups')).index)
            colors = _clrs(max(1, len(tops)))
            for i, val in enumerate(tops):
                vs = pd.to_numeric(mg.loc[grps == val, col], errors="coerce").dropna()
                if vs.empty:
                    continue
                _plot_grp(ax, vs, colors[i], mode, lw=1.8, x_range=xr,
                          ids=_sample_ids(mg, vs))
                handles.append(mlines.Line2D([], [], color=colors[i], lw=2,
                                             label=f"{_tr(val, 20)} (n={len(vs)})"))
        elif col in mg.columns:
            vs = pd.to_numeric(mg[col], errors="coerce").dropna()
            if not vs.empty:
                _plot_grp(ax, vs, AERO['accent'], mode, lw=2, x_range=xr,
                          ids=_sample_ids(mg, vs))
                handles.append(mlines.Line2D([], [], color=AERO['accent'], lw=2,
                                             label=f"selected (n={len(vs)})"))
        ax.axvline(lo, color='red', ls='--', lw=1.2, alpha=0.7, zorder=6)
        ax.axvline(hi, color='red', ls='--', lw=1.2, alpha=0.7, zorder=6)
        ax.axvspan(lo, hi, alpha=0.08, color='red', zorder=0)._gv_overlay = True
        ax.set_title(f"Distribution - selected {lo:.2f} to {hi:.2f}, n={len(mg)}",
                     fontsize=10, weight='bold')
        ax.set_xlabel(self._value_axis_label(region), fontsize=9)
        ax.set_ylabel("Normalized density", fontsize=9)
        _finish_sample_strip(ax)
        if handles:
            # Outside the axes: inside, it covers the tail of the very
            # distribution the region was drawn on.
            ax.legend(handles=handles, fontsize=7, framealpha=0.9,
                      loc='upper left', bbox_to_anchor=(1.005, 1.0),
                      borderaxespad=0.0)

    def _sum_composition(self, ax, rows, lcol):
        n_sel, n_non = rows[0]['n_sel'], rows[0]['n_non']
        y = np.arange(len(rows))
        h = 0.36
        ax.barh(y - h / 2, [r['Sel%'] for r in rows], h, color='#C62828',
                edgecolor='black', lw=0.4, alpha=0.85, label=f"selected (n={n_sel})")
        ax.barh(y + h / 2, [r['BG%'] for r in rows], h, color='#78909C',
                edgecolor='black', lw=0.4, alpha=0.65, label=f"rest (n={n_non:,})")
        ax.set_yticks(y)
        ax.set_yticklabels([_tr(r['Value'], 20) for r in rows], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Frequency (%)", fontsize=8)
        ax.set_title(f"Selected vs rest - {_tr(lcol, 22)}", fontsize=9, weight='bold')
        ax.legend(fontsize=6.5, loc='lower right', framealpha=0.9)
        ax.grid(axis='x', alpha=0.2)

    def _sum_enrichment(self, ax, rows, lcol):
        # Every bar is the fold the test returned. A ratio of several hundred
        # beside a ratio of two is unreadable on a linear axis, so the axis is
        # logarithmic; shortening a bar instead would draw a number the table
        # contradicts.
        y = np.arange(len(rows))
        finite = [r['Enrichment'] for r in rows
                  if np.isfinite(r['Enrichment']) and r['Enrichment'] > 0]
        top = max(finite) if finite else 1.0
        # An undefined ratio - the value occurs in the region and nowhere
        # outside - has no length. It is drawn to the end and named there.
        inf_at = top * 2.2
        vals = [r['Enrichment'] if np.isfinite(r['Enrichment']) and r['Enrichment'] > 0
                else inf_at for r in rows]
        # The axis has to contain the null at 1.0 whatever the bars do.
        base = min(min([v for v in vals if v > 0] + [1.0]) * 0.5, 0.5)
        colors = ['#C62828' if r['Sig'] == '***' else '#E53935' if r['Sig'] == '**'
                  else '#EF9A9A' for r in rows]
        ax.barh(y, [v - base for v in vals], 0.55, left=base, color=colors,
                edgecolor='black', lw=0.4, alpha=0.85)
        ax.set_xscale('log')
        ax.axvline(1.0, color='black', ls='--', lw=1, alpha=0.5)
        for i, r in enumerate(rows):
            lo, hi = r.get('ci_low', float('nan')), r.get('ci_high', float('nan'))
            if np.isfinite(lo) and np.isfinite(hi) and lo > 0:
                ax.plot([lo, hi], [y[i], y[i]], color='#37474F',
                        lw=1.1, alpha=0.8, zorder=4, solid_capstyle='butt')
            if not np.isfinite(r['Enrichment']):
                ax.text(inf_at * 0.97, y[i], "infinite ", ha='right', va='center',
                        fontsize=6.5, color='white', weight='bold', zorder=6)
        hi_ci = [r.get('ci_high', 0.0) for r in rows
                 if np.isfinite(r.get('ci_high', float('nan')))]
        ax.set_xlim(base, max(vals + hi_ci + [1.0]) * 1.6)
        ax.set_yticks(y)
        ax.set_yticklabels([_tr(r['Value'], 20) for r in rows], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Enrichment (fold, log axis; whiskers 95% CI by study)",
                      fontsize=8)
        hidden = getattr(self, '_sum_hidden', {}).get(lcol, 0)
        ax.set_title(f"Enrichment - {_tr(lcol, 22)}"
                     + (f"  (top {len(rows)}, {hidden} more)" if hidden else ""),
                     fontsize=9, weight='bold')
        ax.grid(axis='x', alpha=0.2)

    def _sum_study_counts(self, region):
        """Selected samples per study, so the page shows what backs the region."""
        gse_map = self._gse_map()
        if not gse_map:
            return []
        df = self._sel(region.get('label'))
        if df is None or df.empty or 'GSM' not in df.columns:
            return []
        counts = {}
        for g in df['GSM'].astype(str).str.strip().str.upper():
            s = gse_map.get(g)
            if s:
                counts[s] = counts.get(s, 0) + 1
        return sorted(counts.items(), key=lambda kv: -kv[1])

    def _sum_studies(self, ax, counts):
        top = counts[:_lim('summary_studies')]
        y = np.arange(len(top))
        ax.barh(y, [c for _, c in top], 0.6, color=AERO['accent'], alpha=0.85,
                edgecolor=AERO['text'], lw=0.4)
        ax.set_yticks(y)
        ax.set_yticklabels([_tr(s, 18) for s, _ in top], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Selected samples", fontsize=8)
        n_tot = sum(c for _, c in counts)
        share = 100 * sum(c for _, c in top) / max(1, n_tot)
        ax.set_title(f"Studies behind the selection ({len(counts)} GSE, "
                     f"top {len(top)} hold {share:.0f}%)", fontsize=9, weight='bold')
        ax.grid(axis='x', alpha=0.2)

    def _embed(self, fig, parent, key):
        """
        Embed matplotlib figure WITH full interactive toolbar
        (Home, Back, Forward, Pan, Zoom Rectangle, Save).
        Replaces previous figure with same key if exists.
        """
        # close previous if exists
        if key in self.canvases:
            try: self.canvases[key].get_tk_widget().destroy()
            except: pass
        if key in self.toolbars:
            try: self.toolbars[key].destroy()
            except: pass
        if key in self.figs:
            try: plt.close(self.figs[key])
            except: pass

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        # Hover readout, wheel zoom, right-drag pan, double-click reset.
        # Attached after the first draw so the axes limits it treats as
        # "home" are the ones the user is actually looking at.
        _make_interactive(fig, on_pick=self._open_sample_card)
        # Shift/ctrl-drag over any of these plots selects samples; this is
        # where the window that opens finds the rest of their record.
        _sample_table(fig, self._selection_table)
        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()
        style_toolbar(toolbar)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 6))

        self.figs[key] = fig
        self.canvases[key] = canvas
        self.toolbars[key] = toolbar

    def _on_demand_runs(self):
        """The analyses this window performs only when a button is pressed.

        Each is ``(name, page, figure prefix, callable)`` and the callable is
        the very handler the button calls, so an export drives the program
        rather than recomputing anything beside it.
        """
        runs = []
        cols = self._enrich_label_cols()

        # The Comparison tab draws a control panel until something is
        # pressed, so a freshly opened window has no body for the export to
        # drive. Render it here rather than exporting nothing: that is the
        # whole point of running the unclicked tabs.
        if getattr(self, '_cmp_body', None) is None and len(self.regions) >= 2:
            try:
                self._render_comparison_tab()
            except Exception as exc:
                self._log(f"[Export] could not prepare the Comparison tab: {exc}")
        body = getattr(self, '_cmp_body', None)
        if body is not None and len(self.regions) >= 2:
            # Drive whichever analysis the tab actually offers. Regions on one
            # platform get the grid; regions on several get the pooled effect,
            # once per gene brushed on more than one of them.
            if len({str(r.get('platform') or "") for r in self.regions}) == 1:
                for c in cols:
                    def _cmp(c=c, b=body):
                        self.cmp_col.set(c)
                        self._run_comparison(b)
                    runs.append((f"comparison_{c}", self.t_cmp, "cmp_", _cmp))
            for g in sorted(self._pooled_groups()):
                for c in cols:
                    def _pool(c=c, g=g, b=body):
                        self.cmp_col.set(c)
                        self.cmp_gene.set(g)
                        self._run_pooled(b)
                    runs.append((f"pooled_{g}_{c}", self.t_cmp, "cmp_", _pool))

        return runs

    def _export_on_demand(self, out):
        """Run and write every button-driven analysis, for each label column.

        Without this an export of a freshly opened window carries no region
        comparison at all: that tab draws a control panel until something is
        pressed. Each run
        replaces the one before it in its tab, so it is written out before
        the next one starts, and the tabs are put back afterwards.
        """
        from genevariate.gui.exporting import export_window, _safe
        runs = self._on_demand_runs()
        if not runs:
            return []
        keep = (self.cmp_col.get(),)
        # Each run replaces the tab's cached result too, so it is held and
        # put back: otherwise the window is left showing the last label column
        # the export happened to iterate to.
        keep_cmp = getattr(self, "_cmp_result", None)
        written = []
        for i, (name, page, prefix, run) in enumerate(runs):
            self._update_progress(i, len(runs), f"Computing {name}...")
            before = dict(self.figs)
            try:
                run()
            except Exception as e:
                self._log(f"X export could not compute {name}: {e}")
                continue
            # Only what THIS run drew. `self.figs` accumulates across runs, so
            # matching on the prefix alone re-exported every earlier figure
            # under the current run's name: a tab whose analysis produced no
            # figure still got a file, carrying the previous run's picture and
            # the failed run's title. Fourteen files came out of one session
            # that way, all of them the same clustering embedding under
            # fourteen different names. Comparing against the snapshot taken
            # before the run means a run that drew nothing exports nothing.
            figs = {k: v for k, v in self.figs.items()
                    if k.startswith(prefix) and before.get(k) is not v}
            if not figs:
                self._log(f"[Export] {name} produced no figure; nothing written")
            written += export_window(page, out, figures=figs,
                                     prefix=f"region_{_safe(name)}_")
        self.cmp_col.set(keep[0])
        self._cmp_result = keep_cmp
        self._update_progress(len(runs), len(runs), "Export complete")
        self._stale.update({"Comparison"})
        self._render_current_tab()
        return written

    def _export(self):
        d = filedialog.askdirectory(title="Export Folder", parent=self)
        if not d: return
        self._flush_stale()
        out = Path(d); out.mkdir(parents=True, exist_ok=True)
        # Everything the window is showing: every tab's figure and every tab's
        # table, not only the few frames this method was originally written
        # around. A result the user can see is a result they can keep.
        from genevariate.gui.exporting import export_window
        shown = export_window(self, out, figures=self.figs, prefix="region_")
        shown = list(shown) + self._export_on_demand(out)
        # The on-demand runs above fill tabs that draw a control panel until
        # something is pressed, so the summary pages are drawn again with
        # those results in place.
        self._stale.discard("Summary")
        self._render_tab("Summary", self._render_summary_tab, self.t_summary)
        summary = {k: v for k, v in self.figs.items() if k.startswith("summary_")}
        if summary:
            shown += export_window(self.t_summary, out, figures=summary,
                                   prefix="region_")
        shown = list(dict.fromkeys(shown))
        frames = [self._sel(r['label']).assign(Region=r['label'])
                  for r in self.regions]
        frames = [f for f in frames if not f.empty]
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(out / "region_samples.csv", index=False)
        messagebox.showinfo("Exported",
                            f"Saved to {out}\n- {len(shown)} figures and tables"
                            "\n- region_samples.csv",
                            parent=self)
        try: self.app.enqueue_log(f"[Export] Region analysis -> {out}")
        except: pass

    def _on_close(self):
        """Tear the window down on the main thread."""
        import gc

        for tid in list(self._timers):
            try: self.after_cancel(tid)
            except Exception: pass
        self._timers.clear()

        try:
            self.withdraw()
            self.update_idletasks()
        except Exception: pass

        for key in list(self.figs.keys()):
            try: plt.close(self.figs[key])
            except Exception: pass
        self.figs.clear()
        for key in list(self.canvases.keys()):
            try: self.canvases[key].get_tk_widget().destroy()
            except Exception: pass
        self.canvases.clear()
        for key in list(self.toolbars.keys()):
            try: self.toolbars[key].destroy()
            except Exception: pass
        self.toolbars.clear()

        self._mc.clear()
        self._mc_total.clear()
        self.regions = []
        self.platform_labels_df = None
        self.filter_values = set()

        self.destroy()
        gc.collect()


def build_region_spec(dfg, *, gene, platform, column, low, high, label,
                      color="#7B1FA2", log=None):
    """One entry of ``regions_data`` for :class:`RegionAnalysisWindow`.

    The window is fed the same dict whether a user brushed the region on a
    distribution or the assistant resolved it from a quantile, so a tool opens
    the window a button opens rather than a picture of it. Returns ``None``
    when the bounds select no sample.
    """
    _log = log or (lambda _m: None)
    expr_col = pd.to_numeric(dfg[column], errors='coerce')
    subset = dfg[expr_col.between(low, high)]
    if subset.empty:
        _log(f"[Analysis] Region {gene}[{low:.2f}-{high:.2f}]: empty after filter")
        return None

    if 'GSM' not in subset.columns:
        gsm_cands = [c for c in subset.columns if 'gsm' in c.lower()]
        if gsm_cands:
            subset = subset.rename(columns={gsm_cands[0]: 'GSM'})
        else:
            subset = subset.copy()
            subset['GSM'] = [f"SAMPLE_{j}" for j in range(len(subset))]

    gsms = subset['GSM'].unique().tolist()
    expr_vals = pd.to_numeric(subset[column], errors='coerce').dropna().astype(float)

    # Keep the metadata the Total Platform scope needs, minus the near-unique
    # columns that would cost memory without grouping anything.
    keep_cols = ['GSM', column]
    meta_cols = [c for c in dfg.columns
                 if c not in keep_cols
                 and (dfg[c].dtype == 'object'
                      or c in ('series_id', 'title', 'source_name_ch1',
                               'organism_ch1', 'characteristics_ch1'))]
    for mc in meta_cols:
        nuniq = dfg[mc].nunique()
        if nuniq < 10000 or nuniq < len(dfg) * 0.5:
            keep_cols.append(mc)
    platform_slim = dfg[[c for c in keep_cols if c in dfg.columns]].copy()
    if 'GSM' in platform_slim.columns:
        platform_slim['GSM'] = (platform_slim['GSM'].astype(str)
                                .str.strip().str.upper())

    return {
        'label': label, 'gene': gene, 'platform': platform, 'column': column,
        'range': (low, high),
        'color': color if isinstance(color, str) else '#7B1FA2',
        'expression_values': expr_vals,
        'gsm_list': gsms,
        'platform_df': platform_slim,
    }
