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

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.stats import gaussian_kde, ranksums, wasserstein_distance, fisher_exact

plt.rcParams['figure.max_open_warning'] = 50


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_MAX_GRP = 20
_MAX_BARS = 25          # max bars in any frequency/enrichment bar chart
_MAX_ENRICH_ROWS = 30   # max label values in enrichment plots
_BG_CLR = '#8888AA'
_BG_ALP = 0.50
_BG_EDGE = '#666688'
_LW = 2.8
_LA = 0.88


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _kde(vals, n=300, x_range=None):
    """KDE with tails that reach y≈0.
    
    x_range: optional (xmin, xmax) to evaluate over (e.g. full platform range).
             If None, extends by 3x bandwidth on each side so tails touch x-axis.
    """
    v = np.asarray(vals, dtype=float); v = v[np.isfinite(v)]
    if len(v) < 2 or np.ptp(v) == 0: return None
    try:
        k = gaussian_kde(v)
        if x_range is not None:
            xs = np.linspace(x_range[0], x_range[1], n)
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
    if n <= 10: p = sns.color_palette("tab10", n)
    elif n <= 20: p = sns.color_palette("tab20", n)
    else: p = sns.color_palette("gist_ncar", n)
    return [mcolors.to_hex(c) for c in p]

def _tr(s, m=28):
    s = str(s)
    # Remove control characters (tab, newline, etc.) that cause glyph warnings
    s = ''.join(ch if ord(ch) >= 32 else ' ' for ch in s)
    s = s.strip()
    return (s[:m-1] + '..') if len(s) > m else s


def _smart_series(series, max_cats=_MAX_BARS):
    """Prepare a label series for plotting at scale.
    Auto-bins numeric columns (Age, dosage, time) into ranges.
    Collapses high-cardinality text columns into top-N + 'Other'.
    Returns (cleaned_series, was_binned_flag)."""
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

def _plot_grp(ax, vals, clr, mode, lw=_LW, x_range=None):
    """KDE density with FILLED area + rug. Peak-normalized (max=1).
    Filled area makes conditions clearly visible over background histogram.
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
        elif len(vals) == 1:
            vl = ax.axvline(vals.iloc[0], color=clr, ls=':', lw=lw, alpha=0.7, zorder=4)
            arts.append(vl)
    if mode in ("rug", "both"):
        sns.rugplot(x=vals, ax=ax, color=clr, height=0.05, alpha=0.55, zorder=6)
        if ax.collections: arts.append(ax.collections[-1])
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
        self.canvas = tk.Canvas(self, highlightthickness=0)
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
        self.canvas.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-3, "units"))
        self.canvas.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(3, "units"))
    @property
    def scrollable_frame(self): return self.sf
    def clear(self):
        for w in self.sf.winfo_children(): w.destroy()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main Window
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RegionAnalysisWindow(tk.Toplevel):
    AI_COLS = ['Condition', 'Tissue', 'Treatment', 'Age', 'Treatment_Time']
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

        # state
        self.plot_mode = tk.StringVar(value="both")
        self.gse_scope = tk.StringVar(value="selected")
        self.color_column = tk.StringVar(value="")
        self.ai_label_col = tk.StringVar(value="")
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
            # Configure progress bar style
            _style = ttk.Style(self)
            _style.configure('Accent.Horizontal.TProgressbar',
                             troughcolor='#D0D0D0', background='#1565C0')

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

    def _log(self, msg):
        """Log to both terminal and GUI log."""
        full = f"[Region Analysis] {msg}"
        print(full)
        try: self.app.enqueue_log(full)
        except: pass

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
                # Build slim label-only view (all non-GSM, non-internal columns)
                lbl_only = ['GSM'] + [c for c in plat_lbl.columns
                                      if c not in self._SKIP_COLS and c != 'GSM']
                plat_lbl_slim = plat_lbl[lbl_only].drop_duplicates('GSM')
                self._log(f"Platform labels: {len(plat_lbl_slim):,} GSMs, "
                          f"cols={[c for c in lbl_only if c != 'GSM']}")
            else:
                plat_lbl = None

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
                # SELECTED: only GSMs in this region (small)
                sub = bg[bg['GSM'].isin(gsms)][['GSM', col]].copy()
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
                        n_matched = total[lbl_cols[0]].notna().sum()
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

    # ── UI Layout ───────────────────────────────────────────────────
    def _build_ui(self):
        # ── TOP CONTROLS BAR — ROW 1: Scope buttons (large & prominent) ──
        scope_bar = tk.Frame(self, bg='#ECEFF1', pady=5)
        scope_bar.pack(fill=tk.X)

        tk.Label(scope_bar, text="View:", font=('Segoe UI', 11, 'bold'),
                 bg='#ECEFF1').pack(side=tk.LEFT, padx=(10, 5))

        self._scope_btn_selected = tk.Button(
            scope_bar, text="  Selected Region  ",
            font=('Segoe UI', 11, 'bold'), padx=18, pady=6, cursor='hand2',
            relief=tk.SUNKEN, bd=2, bg='#1565C0', fg='white',
            command=lambda: self._set_scope("selected"))
        self._scope_btn_selected.pack(side=tk.LEFT, padx=3)

        self._scope_btn_total = tk.Button(
            scope_bar, text="  Whole Platform  ",
            font=('Segoe UI', 11, 'bold'), padx=18, pady=6, cursor='hand2',
            relief=tk.RAISED, bd=2, bg='#E0E0E0', fg='#333',
            command=lambda: self._set_scope("total"))
        self._scope_btn_total.pack(side=tk.LEFT, padx=3)

        ttk.Separator(scope_bar, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=8)

        self._overlay_btn = tk.Button(
            scope_bar, text="  + Platform Background  ",
            font=('Segoe UI', 10, 'bold'), padx=14, pady=4, cursor='hand2',
            relief=tk.RAISED, bd=2, bg='#E0E0E0', fg='#333',
            command=self._toggle_overlay)
        self._overlay_btn.pack(side=tk.LEFT, padx=3)

        if len(self.regions) > 1:
            self._merge_btn = tk.Button(
                scope_bar, text="  Merge All Regions  ",
                font=('Segoe UI', 10, 'bold'), padx=14, pady=4, cursor='hand2',
                relief=tk.RAISED, bd=2, bg='#E0E0E0', fg='#333',
                command=self._toggle_merge)
            self._merge_btn.pack(side=tk.LEFT, padx=3)

        # Color By (right side of scope bar)
        ttk.Separator(scope_bar, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=8)
        tk.Label(scope_bar, text="Color By:", font=('Segoe UI', 10, 'bold'),
                 bg='#ECEFF1').pack(side=tk.LEFT, padx=(4, 2))
        opts = self._get_color_cols()
        self.color_column.set(opts[0] if opts else "(none)")
        self.cc = ttk.Combobox(scope_bar, textvariable=self.color_column,
                               values=opts, width=22, state='readonly',
                               font=('Segoe UI', 10))
        self.cc.pack(side=tk.LEFT, padx=4)
        self.cc.bind("<<ComboboxSelected>>", lambda e: self._on_color_col_changed())

        # Multi-Label Query button
        tk.Button(scope_bar, text="Multi-Label Query",
                  command=self._open_multi_label_query,
                  bg='#5C6BC0', fg='white', font=('Segoe UI', 9, 'bold'),
                  padx=8, pady=2, cursor='hand2', relief=tk.RAISED,
                  bd=1).pack(side=tk.LEFT, padx=8)

        # Refresh button (reload labels from app after background processing)
        tk.Button(scope_bar, text="Refresh",
                  command=self._refresh_labels_from_app,
                  bg='#00897B', fg='white', font=('Segoe UI', 9, 'bold'),
                  padx=8, pady=2, cursor='hand2', relief=tk.RAISED,
                  bd=1).pack(side=tk.LEFT, padx=4)

        # Run Phase 2 (NS Recovery) button (NS curation + harmonization)
        tk.Button(scope_bar, text="Run Phase 2 (NS Recovery)",
                  command=self._run_phase2_from_region,
                  bg='#C62828', fg='white', font=('Segoe UI', 9, 'bold'),
                  padx=8, pady=2, cursor='hand2', relief=tk.RAISED,
                  bd=1).pack(side=tk.LEFT, padx=4)
        # LLM Curator button
        tk.Button(scope_bar, text="Curate Labels (LLM)",
                  command=lambda: self.app._open_llm_curator(),
                  bg='#E65100', fg='white', font=('Segoe UI', 9, 'bold'),
                  padx=8, pady=2, cursor='hand2', relief=tk.RAISED,
                  bd=1).pack(side=tk.LEFT, padx=4)

        # Phase button descriptions (tooltip-like, below scope bar)
        phase_info = tk.Frame(self, bg='#ECEFF1')
        phase_info.pack(fill=tk.X)
        tk.Label(phase_info,
            text="Refresh = reload labels after background processing  |  "
                 "Phase 2 (NS Recovery) = recover 'Not Specified' using GEO experiment context",
            font=('Segoe UI', 7), fg='#777', bg='#ECEFF1').pack(anchor=tk.W, padx=10)

        # ── ROW 2: Plot Mode buttons ──
        mode_bar = tk.Frame(self, bg='#E8EAF6', pady=3)
        mode_bar.pack(fill=tk.X)

        tk.Label(mode_bar, text="Plot:", font=('Segoe UI', 10, 'bold'),
                 bg='#E8EAF6').pack(side=tk.LEFT, padx=(10, 5))

        self._mode_btns = {}
        for val, label, active_bg in [
            ("density", "  Density  ", "#5C6BC0"),
            ("rug",     "  Rug  ",     "#7986CB"),
            ("both",    "  Both  ",    "#3F51B5"),
        ]:
            btn = tk.Button(
                mode_bar, text=label,
                font=('Segoe UI', 10, 'bold'), padx=14, pady=3, cursor='hand2',
                relief=tk.SUNKEN if val == "both" else tk.RAISED, bd=2,
                bg=active_bg if val == "both" else '#E0E0E0',
                fg='white' if val == "both" else '#333',
                command=lambda v=val: self._set_plot_mode(v))
            btn.pack(side=tk.LEFT, padx=2)
            self._mode_btns[val] = (btn, active_bg)

        # Region info
        ttk.Separator(mode_bar, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=8)
        parts = [f"{r['label']} [{r['range'][0]:.2f}-{r['range'][1]:.2f}] n={len(r['gsm_list'])}"
                 for r in self.regions]
        tk.Label(mode_bar, text=" | ".join(parts), font=("Segoe UI", 9),
                 fg="#555", bg='#E8EAF6', wraplength=700).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # ── PROGRESS BAR (for rendering / extraction operations) ──
        self._prog_frame = tk.Frame(self, bg='#E8EAF6')
        self._prog_frame.pack(fill=tk.X, padx=6, pady=(2, 0))

        self._prog_bar = ttk.Progressbar(
            self._prog_frame, mode='determinate', length=400,
            style='Accent.Horizontal.TProgressbar')
        self._prog_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        self._prog_label = tk.Label(
            self._prog_frame, text="Ready", font=('Consolas', 9),
            fg='#555', bg='#E8EAF6', anchor='w', width=50)
        self._prog_label.pack(side=tk.LEFT)

        self._prog_pct = tk.Label(
            self._prog_frame, text="", font=('Consolas', 9, 'bold'),
            fg='#1565C0', bg='#E8EAF6', width=6, anchor='e')
        self._prog_pct.pack(side=tk.RIGHT, padx=(0, 4))

        # ── MAIN SPLIT: left panel (filter) + right (notebook) ──
        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

        # ╔══════════════════════════════════════════════════════════╗
        # ║  LEFT: Fancy Filter Panel                                ║
        # ╚══════════════════════════════════════════════════════════╝
        left = ttk.Frame(main_pane, width=260)
        main_pane.add(left, weight=0)

        # Header
        hdr = ttk.Frame(left)
        hdr.pack(fill=tk.X, padx=4, pady=(6, 2))
        ttk.Label(hdr, text="* Filter Values", font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT)
        self.filter_count_lbl = ttk.Label(hdr, text="", font=("Segoe UI", 8), foreground="#1976D2")
        self.filter_count_lbl.pack(side=tk.RIGHT)

        # Search box
        search_frame = ttk.Frame(left)
        search_frame.pack(fill=tk.X, padx=4, pady=(2, 4))
        ttk.Label(search_frame, text="Search:", font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self.filter_search_var = tk.StringVar()
        self.filter_search_var.trace_add('write', lambda *a: self._filter_search_changed())
        search_entry = ttk.Entry(search_frame, textvariable=self.filter_search_var, width=18)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(search_frame, text="x", width=2,
                   command=lambda: self.filter_search_var.set("")).pack(side=tk.RIGHT)

        # Buttons
        btn_frame = ttk.Frame(left); btn_frame.pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(btn_frame, text="[v] All", width=7,
                   command=self._select_all_filter).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="[ ] None", width=7,
                   command=self._select_none_filter).pack(side=tk.LEFT, padx=2)
        ttk.Label(btn_frame, text="Top", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(6, 1))
        self._top_n_var = tk.StringVar(value="10")
        top_n_entry = ttk.Entry(btn_frame, textvariable=self._top_n_var, width=4,
                                font=("Segoe UI", 9))
        top_n_entry.pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_frame, text="<<", width=3,
                   command=self._select_topN_filter).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="> Apply", width=7,
                   command=self._refresh_plots).pack(side=tk.RIGHT, padx=2)

        # Treeview with checkboxes
        tree_frame = ttk.Frame(left)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(2, 4))

        self.filter_tree = ttk.Treeview(tree_frame, columns=("check", "value", "count"),
                                         show="headings", selectmode="extended",
                                         height=22)
        self.filter_tree.heading("check", text="OK")
        self.filter_tree.heading("value", text="Value")
        self.filter_tree.heading("count", text="n")
        self.filter_tree.column("check", width=30, anchor="center", stretch=False)
        self.filter_tree.column("value", width=160, anchor="w")
        self.filter_tree.column("count", width=50, anchor="e", stretch=False)
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

        # Info label
        self.filter_info = ttk.Label(left, text="Click rows to toggle - Apply to update plots",
                                      font=("Segoe UI", 7, "italic"), foreground="#999")
        self.filter_info.pack(pady=(0, 4))

        # RIGHT: Notebook
        right = ttk.Frame(main_pane)
        main_pane.add(right, weight=1)

        self.nb = ttk.Notebook(right, padding=2); self.nb.pack(fill=tk.BOTH, expand=True)

        # Tab 1: GSE Distributions (was Tab 2, Expression tab removed as redundant)
        self.t_gse = ttk.Frame(self.nb); self.nb.add(self.t_gse, text="* Grouped Distributions")

        # Tab 3: AI Labels
        self.t_ai = ttk.Frame(self.nb); self.nb.add(self.t_ai, text="* Labels")
        # AI label selector inside tab
        ai_ctrl = ttk.Frame(self.t_ai); ai_ctrl.pack(fill=tk.X, padx=5, pady=3)
        ttk.Label(ai_ctrl, text="Label Column:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        ai_opts = self._get_ai_cols()
        self.ai_label_col.set(ai_opts[0] if ai_opts else "")
        self.ai_combo = ttk.Combobox(ai_ctrl, textvariable=self.ai_label_col,
                                      values=ai_opts, width=24, state='readonly')
        self.ai_combo.pack(side=tk.LEFT, padx=6)
        self.ai_combo.bind("<<ComboboxSelected>>", lambda e: self._render_ai_tab())
        self.ai_scroll = ScrollableCanvasFrame(self.t_ai)
        self.ai_scroll.pack(fill=tk.BOTH, expand=True)

        # Tab 4: Frequency Analysis
        self.t_freq = ttk.Frame(self.nb); self.nb.add(self.t_freq, text="[Chart] Frequency Analysis")

        # Tab 5: Fisher Enrichment
        self.t_enrich = ttk.Frame(self.nb); self.nb.add(self.t_enrich, text="* Enrichment")

        # Tab 6: Statistics
        self.t_stats = ttk.Frame(self.nb); self.nb.add(self.t_stats, text="* Statistics")
        self.st = None

        # Tab 7: Samples
        self.t_table = ttk.Frame(self.nb); self.nb.add(self.t_table, text="* Samples")

        # Bottom
        bot = ttk.Frame(self, padding=5); bot.pack(fill=tk.X)
        ttk.Button(bot, text="* Export All", command=self._export).pack(side=tk.RIGHT, padx=5)
        ttk.Button(bot, text="Close", command=self._on_close).pack(side=tk.RIGHT, padx=5)

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
        known_labels = ['Condition', 'Tissue', 'Treatment', 'Age', 'Treatment_Time']
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

    # ── Filter Treeview management ──────────────────────────────────
    def _populate_filter_listbox(self):
        """Populate filter treeview with unique values from the current Color By column."""
        self.filter_tree.delete(*self.filter_tree.get_children())
        self._filter_checks.clear()
        self._filter_all_items.clear()
        ccol = self.color_column.get()
        if not ccol or ccol == "(none)":
            self.filter_count_lbl.config(text="")
            return

        # Use TOTAL data if scope is total, else selected
        scope = self.gse_scope.get()
        source = self._mc_total if scope == "total" else self._mc

        # Gather all values + counts across regions
        freq = {}
        for r in self.regions:
            m = source.get(r['label'], pd.DataFrame())
            if not m.empty and ccol in m.columns:
                for v, c in m[ccol].fillna("N/A").astype(str).value_counts().items():
                    freq[v] = freq.get(v, 0) + c

        sorted_items = sorted(freq.items(), key=lambda x: -x[1])
        self._filter_all_items = sorted_items

        # All checked by default
        for val, cnt in sorted_items:
            self._filter_checks[val] = True

        self._render_filter_tree()
        total_checked = sum(1 for v in self._filter_checks.values() if v)
        self.filter_count_lbl.config(text=f"{total_checked}/{len(sorted_items)} selected")

    def _render_filter_tree(self):
        """Render the filter treeview items (respecting search filter)."""
        self.filter_tree.delete(*self.filter_tree.get_children())
        search = self.filter_search_var.get().strip().lower() if hasattr(self, 'filter_search_var') else ""

        for val, cnt in self._filter_all_items:
            if search and search not in val.lower():
                continue
            chk = "[v]" if self._filter_checks.get(val, False) else "[ ]"
            iid = self.filter_tree.insert("", tk.END, values=(chk, _tr(val, 40), cnt))
            # Tag for coloring
            if self._filter_checks.get(val, False):
                self.filter_tree.item(iid, tags=("checked",))
            else:
                self.filter_tree.item(iid, tags=("unchecked",))

        # Apply tag colors
        self.filter_tree.tag_configure("checked", foreground="#1B5E20")
        self.filter_tree.tag_configure("unchecked", foreground="#999999")

    def _on_filter_tree_click(self, event):
        """Toggle checkbox on row click."""
        item = self.filter_tree.identify_row(event.y)
        if not item:
            return
        vals = self.filter_tree.item(item, 'values')
        if not vals or len(vals) < 2:
            return
        # Extract actual value name (un-truncated) from _filter_all_items
        display_val = vals[1]
        # Find matching item
        for val, cnt in self._filter_all_items:
            if _tr(val, 40) == display_val or val == display_val:
                self._filter_checks[val] = not self._filter_checks.get(val, True)
                break

        self._render_filter_tree()
        total_checked = sum(1 for v in self._filter_checks.values() if v)
        self.filter_count_lbl.config(text=f"{total_checked}/{len(self._filter_all_items)} selected")

    def _filter_search_changed(self):
        """Re-render treeview when search text changes."""
        self._render_filter_tree()

    def _get_selected_filter_values(self):
        """Get the set of values that are checked in the filter."""
        return {val for val, checked in self._filter_checks.items() if checked}

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
            return

        df = pd.concat(all_dfs, ignore_index=True)

        # Available label columns
        label_cols = [c for c in df.columns
                      if c.upper() not in ('GSM', 'GENE', '_PLATFORM', 'SERIES_ID', 'GPL')
                      and c not in self._SKIP_COLS
                      and df[c].dtype == 'object']
        if not label_cols:
            return

        dlg = tk.Toplevel(self)
        dlg.title("Multi-Label Query — Compound Color Filter")
        dlg.transient(self)
        dlg.grab_set()

        ttk.Label(dlg, text="Build a compound query — matching samples will be highlighted as one group",
                  font=('Segoe UI', 10, 'bold')).pack(padx=15, pady=(15, 5))
        ttk.Label(dlg, text="Example: Tissue=Brain AND Condition=Alzheimer Disease",
                  font=('Segoe UI', 9, 'italic'), foreground='#666').pack(padx=15, pady=(0, 10))

        rows_frame = ttk.Frame(dlg)
        rows_frame.pack(fill=tk.X, padx=15, pady=5)
        query_rows = []

        def _add_row():
            row_frame = ttk.Frame(rows_frame)
            row_frame.pack(fill=tk.X, pady=3)
            if query_rows:
                ttk.Label(row_frame, text="AND", font=('Segoe UI', 9, 'bold'),
                          foreground='#C62828').pack(side=tk.LEFT, padx=5)
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
                    vals = sorted(df[c].fillna('N/A').astype(str).unique().tolist())
                    val_cb['values'] = vals[:200]
                    if vals: val_var.set(vals[0])
                _preview()

            col_cb.bind('<<ComboboxSelected>>', _on_col)
            val_cb.bind('<<ComboboxSelected>>', lambda e: _preview())
            _on_col()

            def _remove():
                query_rows.remove((col_var, val_var, row_frame))
                row_frame.destroy()
                _preview()

            tk.Button(row_frame, text="✕", command=_remove, fg="red",
                      font=('Segoe UI', 8, 'bold'), padx=4, relief=tk.FLAT,
                      cursor="hand2").pack(side=tk.LEFT, padx=5)
            query_rows.append((col_var, val_var, row_frame))

        preview_lbl = ttk.Label(dlg, text="", font=('Segoe UI', 9), foreground='#1565C0')
        preview_lbl.pack(padx=15, pady=5)

        def _preview(*a):
            mask = pd.Series(True, index=df.index)
            parts = []
            for cv, vv, _ in query_rows:
                c, v = cv.get(), vv.get()
                if c and v and c in df.columns:
                    mask = mask & (df[c].fillna('N/A').astype(str) == v)
                    parts.append(f"{c}={v}")
            n = mask.sum()
            preview_lbl.config(text=f"{'  AND  '.join(parts) or '(no criteria)'}  →  {n:,} samples")

        tk.Button(dlg, text="+ Add Criterion", command=lambda: [_add_row(), _preview()],
                  bg="#43A047", fg="white", font=('Segoe UI', 9, 'bold'),
                  padx=10, cursor="hand2").pack(anchor=tk.W, padx=15, pady=3)

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
            mask = pd.Series(True, index=df.index)
            parts = []
            for cv, vv, _ in query_rows:
                c, v = cv.get(), vv.get()
                if c and v and c in df.columns:
                    mask = mask & (df[c].fillna('N/A').astype(str) == v)
                    parts.append(f"{c}={v}")
            if not parts:
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
            for lbl in self.regions:
                for scope_df_key in [self._mc, self._mc_total]:
                    mc = scope_df_key.get(lbl['label'], pd.DataFrame())
                    if not mc.empty and 'GSM' in mc.columns and 'GSM' in df.columns:
                        matched_gsms = set(df.loc[mask, 'GSM'].astype(str).str.upper())
                        mc[query_col] = mc['GSM'].astype(str).str.upper().apply(
                            lambda g: display_name if g in matched_gsms else 'Other')
                        scope_df_key[lbl['label']] = mc

            # Switch color to the new column
            if query_col not in self.cc['values']:
                current_vals = list(self.cc['values']) + [query_col]
                self.cc['values'] = current_vals
            self.color_column.set(query_col)
            self._on_color_col_changed()
            dlg.destroy()

        tk.Button(btn_frame, text="  Apply & Color  ", command=_apply,
                  bg="#1565C0", fg="white", font=('Segoe UI', 11, 'bold'),
                  padx=20, pady=6, cursor="hand2").pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cancel", command=dlg.destroy,
                  font=('Segoe UI', 10), padx=15, pady=6).pack(side=tk.RIGHT, padx=5)

        _add_row()
        dlg.update_idletasks()
        w = max(600, dlg.winfo_reqwidth())
        h = dlg.winfo_reqheight()
        try:
            x = self.winfo_x() + (self.winfo_width() - w) // 2
            y = self.winfo_y() + (self.winfo_height() - h) // 2
            dlg.geometry(f"{w}x{h}+{max(0,x)}+{max(0,y)}")
        except: pass

    def _run_phase2_from_region(self):
        """Let user trigger Phase 2 (NS Recovery) from Region Analysis. Runs in background, auto-refreshes."""
        import tkinter.messagebox as mb

        if self.platform_labels_df is None or self.platform_labels_df.empty:
            mb.showinfo("No Labels", "No labels loaded. Run extraction first.", parent=self)
            return

        # Count NS
        _NS_CURATE = {'Condition', 'Tissue', 'Treatment'}
        ns_count = 0
        for c in self.platform_labels_df.columns:
            if c in _NS_CURATE and self.platform_labels_df[c].dtype == 'object':
                from app import _NOT_SPECIFIED_VALUES
                ns_count += int(self.platform_labels_df[c].astype(str).str.strip().isin(
                    _NOT_SPECIFIED_VALUES).sum())

        response = mb.askyesno(
            "Run Phase 2 (NS Recovery) — Improve Labels",
            f"Current labels: {ns_count:,} 'Not Specified' in Condition/Tissue/Treatment.\n\n"
            f"{'─' * 45}\n"
            f"PHASE 2 — 'Not Specified' Recovery:\n"
            f"  • Fetches experiment descriptions from NCBI GEO\n"
            f"  • Builds consensus from sibling samples\n"
            f"  • Re-extracts {ns_count:,} missing labels using\n"
            f"    experiment context + sibling examples\n"
            f"  • Only curates: Condition, Tissue, Treatment\n\n"
            f"{'─' * 45}\n"
            f"For cross-experiment label harmonization,\n"
            f"use 'Curate Labels (LLM)' button after Phase 2.\n\n"
            f"Runs in background. Labels auto-refresh when done.\n"
            f"Continue?", parent=self)
        if not response:
            return

        # Determine platform
        plat_id = "Unknown"
        if hasattr(self.app, 'gpl_datasets'):
            for p in self.app.gpl_datasets:
                plat_id = p
                break

        from app import CONFIG
        save_dir = os.path.join(CONFIG['paths']['data'], 'labels')

        self._log("Starting Phase 2 (NS Recovery) from Region Analysis...")
        # Call app's background method — pass self as win so auto-refresh works
        self.app._run_phase2_background(
            self, self.platform_labels_df.copy(),
            plat_id, save_dir, plat_id, None, None)

    def _refresh_labels_from_app(self):
        """Reload labels from app's platform_labels (e.g., after Phase 2 (NS Recovery) finishes)."""
        try:
            # Get latest merged labels from app
            merged = getattr(self.app, 'merged_labels', None)
            if merged is not None and not merged.empty:
                old_count = 0
                if self.platform_labels_df is not None:
                    old_count = len([c for c in self.platform_labels_df.columns
                                     if c not in self._SKIP_COLS and c != 'GSM'])
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
            self._log(f"Label refresh error: {e}")

    def _on_color_col_changed(self):
        self._populate_filter_listbox()
        self._refresh_plots()

    def _set_plot_mode(self, mode):
        """Toggle plot mode with button appearance."""
        self.plot_mode.set(mode)
        for val, (btn, active_bg) in self._mode_btns.items():
            if val == mode:
                btn.config(relief=tk.SUNKEN, bg=active_bg, fg='white')
            else:
                btn.config(relief=tk.RAISED, bg='#E0E0E0', fg='#333')
        self._refresh_plots()

    def _set_scope(self, scope):
        """Toggle scope: 'selected' or 'total' (whole platform)."""
        self.gse_scope.set(scope)
        if scope == "selected":
            self._scope_btn_selected.config(relief=tk.SUNKEN, bg='#1565C0', fg='white')
            self._scope_btn_total.config(relief=tk.RAISED, bg='#E0E0E0', fg='#333')
        else:
            self._scope_btn_selected.config(relief=tk.RAISED, bg='#E0E0E0', fg='#333')
            self._scope_btn_total.config(relief=tk.SUNKEN, bg='#2E7D32', fg='white')
        self._on_scope_changed()

    def _toggle_overlay(self):
        """Toggle overlay: show gene distribution on top of platform distribution."""
        val = not self.overlay.get()
        self.overlay.set(val)
        if val:
            self._overlay_btn.config(relief=tk.SUNKEN, bg='#7B1FA2', fg='white')
        else:
            self._overlay_btn.config(relief=tk.RAISED, bg='#E0E0E0', fg='#333')
        self._refresh_plots()

    def _toggle_merge(self):
        """Toggle merge regions on/off."""
        val = not self.merge_regions.get()
        self.merge_regions.set(val)
        if val:
            self._merge_btn.config(relief=tk.SUNKEN, bg='#7986CB', fg='white')
        else:
            self._merge_btn.config(relief=tk.RAISED, bg='#E0E0E0', fg='#333')
        self._refresh_plots()

    def _on_scope_changed(self):
        """Scope changed -> repopulate filter + refresh."""
        self._log(f"[SCOPE] Changed to: {self.gse_scope.get()}")
        self._populate_filter_listbox()
        n_items = len(self._filter_all_items)
        self._log(f"[SCOPE] Filter repopulated: {n_items} items (all checked)")
        self._refresh_plots()

    # ── Refresh ─────────────────────────────────────────────────────
    def _refresh_plots(self):
        """Re-render all plot tabs."""
        self.filter_values = self._get_selected_filter_values()
        self._close_figs()
        for w in self.t_gse.winfo_children(): w.destroy()
        # AI tab: only clear the scroll, keep the combo
        self.ai_scroll.clear()
        for w in self.t_freq.winfo_children(): w.destroy()
        self._render_gse_tab()
        self._render_ai_tab()
        self._render_freq_tab()

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

        tabs = [
            ("Grouped",    self._render_gse_tab,        self.t_gse),
            ("Labels",     self._render_ai_tab,         None),
            ("Frequency",  self._render_freq_tab,       self.t_freq),
            ("Enrichment", self._render_enrichment_tab, self.t_enrich),
            ("Statistics",  self._render_stats_tab,      self.t_stats),
            ("Samples",    self._render_table_tab,      self.t_table),
        ]
        n_tabs = len(tabs)

        self._update_progress(0, n_tabs, "Rendering tabs...")

        for i, (name, fn, tab) in enumerate(tabs):
            self._update_progress(i, n_tabs, f"Rendering {name}...")
            try:
                fn()
                self._log(f"OK {name} tab rendered")
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                self._log(f"X {name} tab FAILED: {e}")
                print(f"[RegionAnalysis] {name} traceback:\n{tb}")
                if tab:
                    try:
                        ttk.Label(tab, text=f"Error rendering {name}:\n\n{e}",
                                  foreground="red", font=("Consolas", 9), wraplength=600).pack(pady=20)
                    except:
                        pass

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

        scope_names = {"selected": "SELECTED REGION", "total": "WHOLE PLATFORM"}
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

            fig, ax = plt.subplots(figsize=(16, 7))

            # Gene distribution background (gray histogram)
            # ALWAYS show when Whole Platform - this IS the gene distribution
            # For Selected Region - only show if overlay is ON
            show_bg = (scope == "total") or do_overlay
            if show_bg:
                _draw_bg(ax, bg_df, col)
            xr = _bg_range(bg_df, col)

            # Pick data source based on scope
            if scope == "selected":
                mg = self._mc.get(region['label'], pd.DataFrame())
                slbl = f"SELECTED [{lo:.2f}-{hi:.2f}] n={len(region['gsm_list'])}"
            else:
                mg = self._mc_total.get(region['label'], pd.DataFrame())
                slbl = f"WHOLE PLATFORM n={len(mg)}" if not mg.empty else "TOTAL n=0"

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
                show_vals = self.filter_values if self.filter_values else set(grps.unique())
                n_checked = len(show_vals)
                n_total = len(set(grps.unique()))
                # If user selected specific values, show ALL of them
                # Only cap at _MAX_GRP when everything is checked (prevent 500 lines)
                if n_checked < n_total:
                    tops = [v for v in grps.value_counts().index if v in show_vals]
                else:
                    tops = [v for v in grps.value_counts().head(_MAX_GRP).index if v in show_vals]
                if n_checked < n_total:
                    tops = [v for v in grps.value_counts().index if v in show_vals]
                else:
                    tops = [v for v in grps.value_counts().head(_MAX_GRP).index if v in show_vals]
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
                    amap[lb] = _plot_grp(ax, vs, clr, mode, lw=2.0, x_range=xr)
                    handles.append(mlines.Line2D([], [], color=clr, lw=2, label=lb))

                n_checked = sum(1 for v in self._filter_checks.values() if v)
                n_total = len(self._filter_checks)
                show_other = (n_checked > 10 or n_checked == n_total) and n_total > 0

                if show_other:
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
                                sns.rugplot(x=vs, ax=ax, color='gray', height=0.03,
                                            alpha=0.3, zorder=2)
                                if ax.collections: arts.append(ax.collections[-1])
                            handles.append(mlines.Line2D([], [], color='gray', lw=1.2,
                                                          ls='--', label=lb))
                            amap[lb] = arts

            elif not mg.empty:
                self._log(f"[RENDER] FALLBACK: ccol '{ccol}' NOT in mg.columns → showing All")
                vs = pd.to_numeric(mg[col], errors="coerce").dropna()
                lb = f"All (n={len(vs)})"
                amap[lb] = _plot_grp(ax, vs, 'steelblue', mode, lw=2, x_range=xr)
                handles.append(mlines.Line2D([], [], color='steelblue', lw=2, label=lb))
            else:
                ax.text(0.5, 0.5, "No data", ha='center', va='center',
                        transform=ax.transAxes, color='gray')

            # Region boundary markers
            ax.axvline(lo, color='red', ls='--', lw=1.2, alpha=0.7, zorder=6)
            ax.axvline(hi, color='red', ls='--', lw=1.2, alpha=0.7, zorder=6)
            ax.axvspan(lo, hi, alpha=0.08, color='red', zorder=0)
            cb = f" by {ccol}" if ccol and ccol != "(none)" else ""
            overlay_t = " [+BG]" if do_overlay else ""
            ax.set_title(f"{region['label']} - {slbl}{cb}{overlay_t}", fontsize=12, weight='bold')
            ax.set_xlabel("Expression"); ax.set_ylabel("Normalized Density"); ax.set_ylim(bottom=0)
            if handles:
                leg = ax.legend(handles=handles, fontsize=7, loc='upper left',
                                bbox_to_anchor=(1.01, 1.0), framealpha=0.92, fancybox=True)
                _interactive_legend(fig, leg, amap)
            plt.subplots_adjust(left=0.06, right=0.76, top=0.92, bottom=0.10)
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
            ['#F57F17', '#F9A825', '#FBC02D', '#FFD54F', '#FFE082',
             '#E65100', '#FF8F00', '#FFA000', '#FFB300'],
            ['#6A1B9A', '#7B1FA2', '#8E24AA', '#AB47BC', '#CE93D8',
             '#4A148C', '#9C27B0', '#BA68C8', '#E1BEE7'],
            ['#00838F', '#00ACC1', '#00BCD4', '#26C6DA', '#80DEEA',
             '#006064', '#0097A7', '#4DD0E1', '#B2EBF2'],
        ]

        fig, ax = plt.subplots(figsize=(18, 8))
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
                mg = self._mc.get(region['label'], pd.DataFrame())
            else:
                mg = self._mc_total.get(region['label'], pd.DataFrame())

            if mg.empty:
                continue

            # Region boundary markers with region-specific color
            base_clr = palette[0]
            ax.axvline(lo, color=base_clr, ls='--', lw=1.2, alpha=0.5, zorder=6)
            ax.axvline(hi, color=base_clr, ls='--', lw=1.2, alpha=0.5, zorder=6)
            ax.axvspan(lo, hi, alpha=0.04, color=base_clr, zorder=0)

            # Add region header in legend
            handles.append(mlines.Line2D([], [], color='none',
                           label=f"── {src_tag} [{lo:.1f}-{hi:.1f}] ──"))

            if ccol and ccol != "(none)" and ccol in mg.columns:
                grps = mg[ccol].fillna("N/A").astype(str)
                show_vals = self.filter_values if self.filter_values else set(grps.unique())
                n_checked = len(show_vals)
                n_total = len(set(grps.unique()))
                if n_checked < n_total:
                    tops = [v for v in grps.value_counts().index if v in show_vals]
                else:
                    tops = [v for v in grps.value_counts().head(_MAX_GRP).index if v in show_vals]

                for i, val in enumerate(tops):
                    clr = palette[i % len(palette)]
                    vs = pd.to_numeric(mg.loc[grps == val, col], errors="coerce").dropna()
                    if vs.empty: continue
                    lb = f"[{src_tag}] {_tr(val)} (n={len(vs)})"
                    # Use different line styles per region for additional distinction
                    ls_list = ['-', '--', '-.', ':']
                    lw_base = 2.0
                    art = _plot_grp(ax, vs, clr, mode, lw=lw_base, x_range=xr)
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
                art = _plot_grp(ax, vs, base_clr, mode, lw=2.0, x_range=xr)
                for a in art:
                    if hasattr(a, 'set_linestyle'):
                        a.set_linestyle(ls_style)
                amap[lb] = art
                handles.append(mlines.Line2D([], [], color=base_clr, lw=2,
                               ls=ls_style, label=lb))

        scope_names = {"selected": "SELECTED", "total": "WHOLE PLATFORM"}
        scope_lbl = scope_names.get(scope, scope.upper())
        overlay_t = " [+BG]" if self.overlay.get() else ""
        cb = f" by {ccol}" if ccol and ccol != "(none)" else ""
        ax.set_title(f"MERGED: {n_regions} regions - {scope_lbl}{overlay_t}{cb}", fontsize=13, weight='bold')
        ax.set_xlabel("Expression"); ax.set_ylabel("Normalized Density"); ax.set_ylim(bottom=0)
        if handles:
            leg = ax.legend(handles=handles, fontsize=7, loc='upper left',
                            bbox_to_anchor=(1.01, 1.0), framealpha=0.92, fancybox=True)
            _interactive_legend(fig, leg, amap)
        plt.subplots_adjust(left=0.06, right=0.68, top=0.92, bottom=0.10)
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
                    from genevariate.core.nlp import classify_sample
                    from genevariate.utils.workers import SampleClassificationAgent
                    agent = SampleClassificationAgent(
                        [classify_sample], self._log, max_workers=4)

                # Build a DataFrame of samples to classify
                import pandas as _pd
                rows = []
                for r in self.regions:
                    mg = self._mc.get(r['label'], _pd.DataFrame())
                    if mg.empty:
                        continue
                    for _, row in mg.iterrows():
                        gsm = str(row.get('GSM', row.get('gsm', ''))).strip()
                        if gsm and gsm not in {r.get('GSM', '') for r in rows}:
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
                _orig_log = agent.log_func

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

                agent.log_func = _progress_log

                # Run extraction
                result_df = agent.process_samples(samples_df)

                agent.log_func = _orig_log

                if result_df is not None and not result_df.empty:
                    # Merge results into platform_labels_df
                    self._log(f"Extraction complete: {len(result_df):,} samples classified")

                    # Store as platform labels
                    self.platform_labels_df = result_df
                    # Also update the main app if possible
                    try:
                        if hasattr(self.app, 'platform_labels'):
                            self.app.platform_labels = result_df
                    except Exception:
                        pass

                    # Recompute merged data with new labels
                    self.after(0, self._post_extraction_refresh)
                else:
                    self._log("[WARN] Extraction returned no results")
                    self.after(0, lambda: self._update_progress(
                        n, n, "Extraction failed - check Ollama"))

            except Exception as exc:
                import traceback
                self._log(f"[ERROR] Extraction failed: {exc}")
                print(traceback.format_exc())
                self.after(0, lambda: self._update_progress(
                    0, 1, f"Error: {exc}"))
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
            if new_opts and new_opts[0] != "(none)":
                self.ai_label_col.set(new_opts[0])
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
                font=("Segoe UI", 14, "bold"), foreground="#C62828")
            header.pack(pady=(30, 8))

            ttk.Label(sf,
                text=f"{n_gsms:,} samples in selected region(s) need label extraction.\n"
                     f"The LLM agent will classify Tissue, Condition, and Treatment\n"
                     f"for each sample using its GEO metadata.",
                font=("Segoe UI", 10), foreground="#555",
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
                font=("Consolas", 9), foreground="#666")
            self._ext_status.pack(anchor='w')

            # Extract button
            self._btn_extract = tk.Button(sf,
                text=f"Extract Labels (LLM)  --  {n_gsms:,} samples",
                font=("Segoe UI", 12, "bold"),
                bg="#1565C0", fg="white", activebackground="#0D47A1",
                activeforeground="white", cursor="hand2",
                padx=30, pady=10, relief='flat',
                command=self._start_inline_extraction)
            self._btn_extract.pack(pady=(4, 16))

            # Info text
            info_frame = ttk.Frame(sf)
            info_frame.pack(padx=60, fill='x')
            for icon, text in [
                (">>", "Phase 1: Raw LLM extraction (Tissue, Condition, Treatment)"),
                (">>", "Phase 1.5: Per-GSE label collapsing (abbreviation matching)"),
                (">>", "Results shown automatically when complete"),
                ("--", "Requires Ollama running with a model loaded"),
            ]:
                row = ttk.Frame(info_frame)
                row.pack(fill='x', pady=1)
                ttk.Label(row, text=icon, font=("Consolas", 9),
                          foreground="#1565C0" if icon == ">>" else "#999"
                          ).pack(side='left', padx=(0, 6))
                ttk.Label(row, text=text, font=("Segoe UI", 9),
                          foreground="#444").pack(side='left')

            return

        nice = lc.replace('_', ' ')
        sf = self.ai_scroll.scrollable_frame

        for ci, region in enumerate(self.regions):
            scope = self.gse_scope.get()
            if scope == "selected":
                mg = self._mc.get(region['label'], pd.DataFrame())
            else:
                mg = self._mc_total.get(region['label'], pd.DataFrame())
            ecol = region['column']

            # ── Figure 1: Density plot (full width) ──
            fig_d, ax_d = plt.subplots(figsize=(16, 7))
            if mg.empty or lc not in mg.columns:
                ax_d.text(0.5, 0.5, "N/A", ha='center', va='center',
                          transform=ax_d.transAxes, color='gray')
                ax_d.set_title(f"{region['label']} - {nice} Density", fontsize=11)
            else:
                bg = region.get('platform_df')
                show_bg = (scope == "total") or self.overlay.get()
                if show_bg:
                    _draw_bg(ax_d, bg, ecol)
                xr = _bg_range(bg, ecol)
                smart, binned = _smart_series(mg[lc], max_cats=_MAX_GRP)

                # Apply filter if values are selected
                show_vals = self.filter_values if self.filter_values else set(smart.unique())
                n_checked = len(show_vals)
                n_total = len(set(smart.unique()))
                if n_checked < n_total:
                    tops = [v for v in smart.value_counts().index if v in show_vals]
                else:
                    tops = [v for v in smart.value_counts().head(_MAX_GRP).index
                            if v in show_vals]

                colors = _clrs(max(1, len(tops)))
                handles = []; amap = {}
                for val, clr in zip(tops, colors):
                    sub = pd.to_numeric(mg.loc[smart == val, ecol], errors="coerce").dropna()
                    if sub.empty: continue
                    lb = f"{_tr(val)} ({len(sub)})"
                    amap[lb] = _plot_grp(ax_d, sub, clr, mode, lw=2.0, x_range=xr)
                    handles.append(mlines.Line2D([], [], color=clr, lw=2, label=lb))
                suffix = " (binned)" if binned else ""
                scope_lbl = "SELECTED" if scope == "selected" else "WHOLE PLATFORM"
                ax_d.set_title(f"{region['label']} - {nice}{suffix} {scope_lbl}",
                               fontsize=12, weight='bold')
                ax_d.set_xlabel("Expression", fontsize=10)
                ax_d.set_ylabel("Normalized Density", fontsize=10)
                ax_d.set_ylim(bottom=0)
                if handles:
                    leg = ax_d.legend(handles=handles, fontsize=7, loc='upper left',
                                      bbox_to_anchor=(1.01, 1.0), ncol=max(1, len(handles) // 12),
                                      framealpha=0.92)
                    _interactive_legend(fig_d, leg, amap)
            plt.subplots_adjust(left=0.06, right=0.75, top=0.92, bottom=0.10)
            self._embed(fig_d, sf, f"ai_d_{ci}")

            # ── Figure 2: Frequency bar chart (full width) ──
            if not mg.empty and lc in mg.columns:
                smart_f, binned_f = _smart_series(mg[lc], max_cats=_MAX_BARS)
                counts = smart_f.value_counts().head(_MAX_BARS)
                total_all = len(smart_f)
                n_other = total_all - counts.sum()

                if not counts.empty:
                    n_bars = len(counts)
                    fig_h = max(4, 0.4 * n_bars + 1.5)
                    fig_f, ax_f = plt.subplots(figsize=(16, fig_h))

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
                    ax_f.set_title(f"{region['label']} - {nice}{suffix_f} Frequency",
                                   fontsize=12, weight='bold')
                    ax_f.tick_params(labelsize=8)
                    plt.subplots_adjust(left=0.22, right=0.92, top=0.92, bottom=0.10)
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
            mg = self._mc.get(region['label'], pd.DataFrame())
            # ALL platform samples (with labels merged)
            mg_total = self._mc_total.get(region['label'], pd.DataFrame())

            if mg_total.empty or ccol not in mg_total.columns:
                continue

            # Platform-wide label counts (ALL samples)
            plat_ser = mg_total[ccol].fillna("N/A").astype(str)
            plat_smart, binned = _smart_series(plat_ser, max_cats=_MAX_BARS)
            plat_counts = plat_smart.value_counts()
            plat_total = len(plat_smart)

            # Selected region label counts
            sel_gsms = set(str(g).strip().upper() for g in region['gsm_list'])
            if not mg.empty and ccol in mg.columns:
                sel_ser = mg[ccol].fillna("N/A").astype(str)
                if binned:
                    # Apply same binning to selected data
                    sel_smart, _ = _smart_series(sel_ser, max_cats=_MAX_BARS)
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
            for val in plat_counts.head(_MAX_BARS).index:
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
            fig, ax = plt.subplots(figsize=(16, fig_h))

            bc = ['#C62828' if r >= 3 else '#E53935' if r >= 2 else '#FB8C00' if r >= 1.5
                  else '#43A047' if r >= 1 else '#78909C' for r in rdf['Enrichment']]
            trunc_vals = [_tr(v, 35) for v in rdf['Value']]
            bars = ax.barh(trunc_vals, rdf['Enrichment'], color=bc, edgecolor='black', lw=0.4)
            ax.axvline(1.0, color='black', ls='--', lw=1, alpha=0.5)

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
            plt.subplots_adjust(left=0.22, right=0.92, top=0.90, bottom=0.10)
            self._embed(fig, sf, f"freq_{idx}")

        # ── Summary frequency table ──
        if all_freq_data:
            tbl_frame = ttk.LabelFrame(sf, text="Frequency Table (All Platform Labels)", padding=5)
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
    def _render_enrichment_tab(self):
        for w in self.t_enrich.winfo_children(): w.destroy()

        # ── Detect available label columns ──
        label_cols = []
        for r in self.regions:
            m = self._mc.get(r['label'], pd.DataFrame())
            if not m.empty:
                for c in m.columns:
                    if c not in self._SKIP_COLS and c != 'GSM' and c not in label_cols and m[c].dtype == 'object':
                        if m[c].nunique() > 1:
                            label_cols.append(c)

        plat_labels = self.platform_labels_df
        if plat_labels is not None and not plat_labels.empty:
            for c in plat_labels.columns:
                if c != 'GSM' and plat_labels[c].nunique() > 1:
                    cc = c  # Use column name directly
                    if cc not in label_cols:
                        label_cols.append(cc)

        if not label_cols:
            ttk.Label(self.t_enrich,
                      text="No label columns available for enrichment analysis.\n\n"
                           "Load labels from file or run classification first.",
                      font=("Segoe UI", 11), foreground="gray").pack(pady=40)
            return

        # ── Header ──
        ttk.Label(self.t_enrich,
                  text="Fisher's Exact Test - Significant Enrichments Only (p<0.05)",
                  font=("Segoe UI", 11, "bold"), foreground="#1976D2").pack(fill=tk.X, padx=10, pady=(8, 2))
        ttk.Label(self.t_enrich,
                  text="Shows ONLY label values significantly enriched in the selected region vs rest of platform. "
                       "For ALL labels, see the Frequency Analysis tab.",
                  font=("Segoe UI", 9, "italic"), foreground="#666").pack(fill=tk.X, padx=10, pady=(0, 4))

        # ── Compute enrichment for each region x label column ──
        # Store structured data for both table and plots
        all_rows = []           # flat list for table
        plot_groups = {}        # (region_label, lcol) -> list of row dicts

        for region in self.regions:
            sel_gsms = set(str(g).upper() for g in region['gsm_list'])
            mg = self._mc.get(region['label'], pd.DataFrame())

            for lcol in label_cols:
                # Build label series for selected GSMs
                sel_labels = None
                if not mg.empty and lcol in mg.columns:
                    sel_labels = mg.set_index('GSM')[lcol].dropna().astype(str)

                # Build label series for ALL platform GSMs
                all_labels = None
                if plat_labels is not None and not plat_labels.empty:
                    raw_name = lcol
                    for pcol in [lcol, raw_name]:
                        if pcol in plat_labels.columns:
                            all_labels = plat_labels.set_index('GSM')[pcol].dropna().astype(str)
                            break
                if all_labels is None:
                    total = self._mc_total.get(region['label'], pd.DataFrame())
                    if not total.empty and lcol in total.columns:
                        all_labels = total.set_index('GSM')[lcol].dropna().astype(str)

                if all_labels is None or all_labels.empty:
                    continue

                if sel_labels is None or sel_labels.empty:
                    sel_labels = all_labels[all_labels.index.isin(sel_gsms)]
                if sel_labels.empty:
                    continue

                non_sel_labels = all_labels[~all_labels.index.isin(sel_gsms)]
                n_sel = len(sel_labels)
                n_non = len(non_sel_labels)
                if n_sel == 0 or n_non == 0:
                    continue

                # Smart-bin high-cardinality columns (e.g. Age with 67 values)
                combined_labels = pd.concat([sel_labels, non_sel_labels])
                if combined_labels.nunique() > _MAX_ENRICH_ROWS:
                    numeric = pd.to_numeric(combined_labels, errors='coerce')
                    if numeric.notna().sum() > len(combined_labels) * 0.5:
                        try:
                            n_bins = min(12, max(5, combined_labels.nunique() // 5))
                            binned = pd.cut(numeric, bins=n_bins, duplicates='drop')
                            combined_str = binned.astype(str).fillna("N/A")
                        except Exception:
                            top = combined_labels.value_counts().head(_MAX_ENRICH_ROWS - 1).index
                            combined_str = combined_labels.where(
                                combined_labels.isin(top), "Other")
                    else:
                        top = combined_labels.value_counts().head(_MAX_ENRICH_ROWS - 1).index
                        combined_str = combined_labels.where(
                            combined_labels.isin(top), "Other")
                    sel_labels = combined_str[combined_str.index.isin(sel_gsms)]
                    non_sel_labels = combined_str[~combined_str.index.isin(sel_gsms)]

                group_key = (region['label'], lcol)
                group_rows = []

                for val in sorted(sel_labels.unique()):
                    if val.lower() in ('nan', 'none', 'n/a', 'not specified', ''):
                        continue
                    a = int((sel_labels == val).sum())
                    b = int(n_sel - a)
                    c = int((non_sel_labels == val).sum())
                    d = int(n_non - c)
                    if a == 0:
                        continue

                    try:
                        odds, p_val = fisher_exact(np.array([[a, b], [c, d]]),
                                                    alternative='greater')
                    except Exception:
                        p_val, odds = 1.0, 0.0

                    sel_frac = a / max(1, n_sel)
                    bg_frac = c / max(1, n_non)
                    enrich_ratio = sel_frac / bg_frac if bg_frac > 0 else float('inf')
                    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

                    row = {
                        'Region': region['label'],
                        'Label Column': lcol,
                        'Value': val,
                        'a': a, 'n_sel': n_sel, 'c': c, 'n_non': n_non,
                        'Sel%': sel_frac * 100, 'BG%': bg_frac * 100,
                        'Enrichment': enrich_ratio, 'p-value': p_val, 'Sig': sig
                    }
                    group_rows.append(row)
                    all_rows.append(row)

                if group_rows:
                    plot_groups[group_key] = sorted(group_rows, key=lambda x: x['p-value'])

        if not all_rows:
            ttk.Label(self.t_enrich,
                      text="No enrichment data could be computed.\n"
                           "Ensure labels have matching GSMs with the platform.",
                      font=("Segoe UI", 11), foreground="orange").pack(pady=30)
            return

        all_rows.sort(key=lambda x: x['p-value'])
        n_sig = sum(1 for r in all_rows if r['Sig'] != 'ns')
        ttk.Label(self.t_enrich,
                  text=f"OK {n_sig} significantly enriched (p<0.05) / {len(all_rows)} tested  |  "
                       f"{len(plot_groups)} group(s) across {len(self.regions)} region(s)",
                  font=("Segoe UI", 9, "bold"),
                  foreground="#C62828" if n_sig > 0 else "#666").pack(fill=tk.X, padx=10, pady=2)

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
            top_rows = sig_only[:_MAX_ENRICH_ROWS]
            n_omitted = len(sig_only) - len(top_rows)

            # ════════════════════════════════════════════════════════════
            #  PLOT 1: Paired bar chart - Selected % vs Rest % (top labels)
            # ════════════════════════════════════════════════════════════
            n_bars = len(top_rows)
            if n_bars == 0:
                continue
            fig_h = max(4, min(20, 0.45 * n_bars + 1.5))
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, fig_h), gridspec_kw={'width_ratios': [3, 2]})

            labels = [_tr(r['Value'], 28) for r in top_rows]
            sel_pcts = [r['Sel%'] for r in top_rows]
            bg_pcts = [r['BG%'] for r in top_rows]
            sigs = [r['Sig'] for r in top_rows]
            y_pos = np.arange(n_bars)
            bar_h = 0.35

            # Left panel: Paired horizontal bars
            bars_sel = ax1.barh(y_pos - bar_h / 2, sel_pcts, bar_h,
                                label=f'Selected Region (n={n_sel})',
                                color='#C62828', edgecolor='black', lw=0.4, alpha=0.85)
            bars_bg = ax1.barh(y_pos + bar_h / 2, bg_pcts, bar_h,
                               label=f'Rest of Platform (n={n_non:,})',
                               color='#78909C', edgecolor='black', lw=0.4, alpha=0.65)

            # Annotate with counts + significance
            for i, r in enumerate(top_rows):
                # Selected bar annotation
                ax1.text(max(sel_pcts[i] + 0.5, 1), y_pos[i] - bar_h / 2,
                         f" {r['a']}/{n_sel}  {r['Sig']}", va='center', fontsize=7,
                         fontweight='bold' if r['Sig'] != 'ns' else 'normal',
                         color='#C62828' if r['Sig'] != 'ns' else '#999')
                # BG bar annotation
                ax1.text(max(bg_pcts[i] + 0.5, 1), y_pos[i] + bar_h / 2,
                         f" {r['c']}/{n_non:,}", va='center', fontsize=6.5, color='#546E7A')

            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(labels, fontsize=7.5)
            ax1.set_xlabel("Frequency (%)", fontsize=9)
            ax1.set_title(f"Selected vs Rest - {nice_col}", fontsize=10, weight='bold')
            ax1.legend(fontsize=7, loc='lower right', framealpha=0.9)
            ax1.invert_yaxis()
            ax1.grid(axis='x', alpha=0.2)

            # ════════════════════════════════════════════════════════════
            #  PLOT 2: Enrichment ratio bars (colored by significance)
            # ════════════════════════════════════════════════════════════
            enrich_vals = [min(r['Enrichment'], 30) for r in top_rows]  # cap at 30 for display
            bar_colors = ['#C62828' if s == '***' else '#E65100' if s == '**'
                          else '#F9A825' if s == '*' else '#BDBDBD' for s in sigs]

            ax2.barh(y_pos, enrich_vals, 0.55, color=bar_colors,
                     edgecolor='black', lw=0.4, alpha=0.85)
            ax2.axvline(1.0, color='black', ls='--', lw=1, alpha=0.5)

            for i, r in enumerate(top_rows):
                ev = min(r['Enrichment'], 30)
                p_str = f"p={r['p-value']:.1e}" if r['p-value'] < 0.01 else f"p={r['p-value']:.3f}"
                ax2.text(ev + 0.15, y_pos[i],
                         f" {r['Enrichment']:.1f}x  {p_str}",
                         va='center', fontsize=6.5,
                         fontweight='bold' if r['Sig'] != 'ns' else 'normal',
                         color=bar_colors[i])

            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(['' for _ in top_rows])  # labels already on left plot
            ax2.set_xlabel("Enrichment Ratio (fold)", fontsize=9)
            ax2.set_title(f"Enrichment - {nice_col}", fontsize=10, weight='bold')
            ax2.invert_yaxis()
            ax2.grid(axis='x', alpha=0.2)

            # Significance legend
            from matplotlib.patches import Patch as _Patch
            ax2.legend(handles=[
                _Patch(facecolor='#C62828', label='p<0.001 ***'),
                _Patch(facecolor='#E65100', label='p<0.01 **'),
                _Patch(facecolor='#F9A825', label='p<0.05 *'),
                _Patch(facecolor='#BDBDBD', label='ns'),
            ], fontsize=6, loc='lower right', framealpha=0.9)

            fig.suptitle(f"{reg_label} - Fisher Enrichment: {nice_col}",
                         fontsize=11, weight='bold', y=1.01)
            try:
                plt.tight_layout()
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
        volcano_rows = [r for r in all_rows if r['p-value'] < 1.0
                        and 0 < r['Enrichment'] < 100]
        if len(volcano_rows) >= 3:
            ttk.Separator(sf, orient='horizontal').pack(fill=tk.X, pady=6)
            ttk.Label(sf, text="> Volcano Plot - All Regions x All Label Columns",
                      font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, padx=10, pady=(4, 2))

            fig, ax = plt.subplots(figsize=(14, 7))
            x_vals = [np.log2(max(r['Enrichment'], 0.01)) for r in volcano_rows]
            y_vals = [-np.log10(max(r['p-value'], 1e-50)) for r in volcano_rows]
            v_sigs = [r['Sig'] for r in volcano_rows]
            v_colors = ['#C62828' if s == '***' else '#E65100' if s == '**'
                        else '#F9A825' if s == '*' else '#BDBDBD' for s in v_sigs]

            ax.scatter(x_vals, y_vals, c=v_colors, s=45, alpha=0.75,
                       edgecolor='black', lw=0.3, zorder=3,
                       picker=True, pickradius=5)
            ax.axhline(-np.log10(0.05), color='gray', ls='--', lw=1, alpha=0.5)
            ax.axvline(0, color='gray', ls='--', lw=1, alpha=0.5)

            # NO auto-labels — click to inspect instead
            ax.text(0.5, -0.06,
                    'Click points to inspect  •  Shift+click to multi-select  •  Double-click to clear',
                    transform=ax.transAxes, fontsize=7.5, ha='center',
                    color='#777777', style='italic')

            ax.set_xlabel("log2(Enrichment Ratio)", fontsize=10)
            ax.set_ylabel("-log10(p-value)", fontsize=10)
            ax.set_title("Enrichment Volcano - Selected Region vs Rest of Platform",
                         fontsize=12, weight='bold')
            from matplotlib.patches import Patch as _Patch2
            ax.legend(handles=[
                _Patch2(facecolor='#C62828', label='p<0.001 ***'),
                _Patch2(facecolor='#E65100', label='p<0.01 **'),
                _Patch2(facecolor='#F9A825', label='p<0.05 *'),
                _Patch2(facecolor='#BDBDBD', label='ns'),
            ], fontsize=7, loc='upper left', framealpha=0.9)
            ax.grid(alpha=0.15)
            try:
                plt.tight_layout()
            except Exception:
                pass
            self._embed(fig, sf, f"enrich_volcano")

            # ── Click-to-inspect for enrichment volcano ──
            x_arr = np.array(x_vals)
            y_arr = np.array(y_vals)
            _v_sel_anns = {}  # idx -> annotation

            # Info table below volcano
            _v_info_frame = ttk.LabelFrame(sf,
                text="Selected Enrichments (click points above)")
            _v_info_frame.pack(fill=tk.X, padx=5, pady=(0, 4))

            _v_info_cols = ('Region', 'Label Column', 'Value', 'Sel', 'Sel%',
                            'BG%', 'Enrichment', 'p-value', 'Sig')
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
                    _v_tree.insert('', tk.END, values=(
                        r.get('Region', ''),
                        r.get('Label Column', ''),
                        _tr(r.get('Value', ''), 22),
                        r.get('Sel', ''),
                        f"{r.get('Sel%', 0):.1f}%",
                        f"{r.get('BG%', 0):.1f}%",
                        f"{r.get('Enrichment', 0):.2f}",
                        f"{r.get('p-value', 1):.2e}",
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
                 "Enrichment", "p-value", "Sig")
        tree = ttk.Treeview(tbl_frame, columns=tcols, show="headings", height=20)
        widths = {"Region": 140, "Column": 110, "Value": 170, "Sel": 80,
                  "Sel%": 60, "BG%": 60, "Enrichment": 85, "p-value": 95, "Sig": 45}
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

        for row in sig_rows:
            enr_str = f"{row['Enrichment']:.2f}" if row['Enrichment'] != float('inf') else "INF"
            p_str = f"{row['p-value']:.2e}" if row['p-value'] < 0.001 else f"{row['p-value']:.4f}"
            tag = "sig3" if row['Sig'] == "***" else "sig2" if row['Sig'] == "**" else "sig1"
            tree.insert("", tk.END, values=(
                row['Region'], row['Label Column'], _tr(row['Value'], 35),
                f"{row['a']}/{row['n_sel']}",
                f"{row['Sel%']:.1f}%", f"{row['BG%']:.1f}%",
                enr_str, p_str, row['Sig']
            ), tags=(tag,))

        if not sig_rows:
            ttk.Label(sf, text="No significantly enriched labels found (p<0.05).\n"
                               "Check the Frequency Analysis tab for full label breakdown.",
                      font=("Segoe UI", 10), foreground="gray").pack(pady=10)

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

        vals = {r['label']: pd.to_numeric(r['expression_values'], errors='coerce').dropna() for r in self.regions}
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
            m = self._mc.get(r['label'], pd.DataFrame()).copy()
            if not m.empty: m['Region'] = r['label']; frames.append(m)
        if not frames:
            ttk.Label(self.t_table, text="No metadata.", font=("Segoe UI", 11),
                      foreground="gray").pack(pady=30)
            return

        combined = pd.concat(frames, ignore_index=True)
        self._table_df = combined  # keep ref for compare

        # Priority columns
        cls_cols = sorted([c for c in combined.columns if c not in ('GSM', 'Region', 'Expression') and combined[c].dtype == 'object'])
        pri = ['Region', 'GSM', 'series_id', 'title', 'source_name_ch1'] + cls_cols
        drop = {'contact', 'supplementary_file', 'data_row_count', 'channel_count',
                'status', 'submission_date', 'last_update_date'}
        cols = [c for c in pri if c in combined.columns]
        cols += [c for c in combined.columns if c not in cols and c not in drop]
        cols = cols[:25]
        self._table_cols = cols

        # ── Top controls bar ──
        ctrl = ttk.Frame(self.t_table)
        ctrl.pack(fill=tk.X, padx=5, pady=(4, 2))

        ttk.Label(ctrl,
                  text="* Click column header -> Color By  |  Select rows -> highlight + compare",
                  font=("Segoe UI", 9, "italic"), foreground="#1976D2").pack(side=tk.LEFT)

        self._sel_count_lbl = ttk.Label(ctrl, text="0 selected", font=("Segoe UI", 9, "bold"),
                                         foreground="#666")
        self._sel_count_lbl.pack(side=tk.RIGHT, padx=8)

        btn_bar = ttk.Frame(self.t_table)
        btn_bar.pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(btn_bar, text="[v] Select All", width=12,
                   command=self._table_select_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_bar, text="[ ] Clear", width=8,
                   command=self._table_clear_sel).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_bar, text="* Compare Selected ->",
                   command=self._compare_selected_samples).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btn_bar, text="[Chart] Highlight Rows",
                   command=self._highlight_selection).pack(side=tk.RIGHT, padx=4)

        # ── Treeview ──
        tree_frame = ttk.Frame(self.t_table)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)

        self._sample_tree = ttk.Treeview(tree_frame, columns=cols, show="headings",
                                          selectmode="extended")
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self._sample_tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self._sample_tree.xview)
        self._sample_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        for c in cols:
            self._sample_tree.heading(c, text=c.replace('_', ' '))
            self._sample_tree.column(c, width=130, stretch=False)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self._sample_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

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
            foreground="#C62828" if n > 0 else "#666"
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
            foreground="#1B5E20"
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

        for grp_name, grp_df in sel_df.groupby('_group'):
            expr = pd.to_numeric(grp_df[expr_col], errors="coerce").dropna()
            if expr.empty: continue
            label = f"{_tr(grp_name, 35)} (n={len(expr)})"
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
        meta_df['Group'] = meta_df['_group']
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

    # ═══════════════════════════════════════════════════════════════════
    #  Embed / Close / Export helpers
    # ═══════════════════════════════════════════════════════════════════
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
        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 6))

        self.figs[key] = fig
        self.canvases[key] = canvas
        self.toolbars[key] = toolbar

    def _close_figs(self):
        for key in list(self.figs.keys()):
            try: plt.close(self.figs[key])
            except: pass
        self.figs.clear()
        for key in list(self.canvases.keys()):
            try: self.canvases[key].get_tk_widget().destroy()
            except: pass
        self.canvases.clear()
        for key in list(self.toolbars.keys()):
            try: self.toolbars[key].destroy()
            except: pass
        self.toolbars.clear()

    def _export(self):
        d = filedialog.askdirectory(title="Export Folder", parent=self)
        if not d: return
        out = Path(d); out.mkdir(parents=True, exist_ok=True)
        for key, fig in self.figs.items():
            fig.savefig(out / f"region_{key}.png", dpi=150, bbox_inches='tight')
        frames = [self._mc.get(r['label'], pd.DataFrame()).assign(Region=r['label'])
                  for r in self.regions]
        frames = [f for f in frames if not f.empty]
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(out / "region_samples.csv", index=False)
        messagebox.showinfo("Exported",
                            f"Saved to {out}\n- {len(self.figs)} figures\n- region_samples.csv",
                            parent=self)
        try: self.app.enqueue_log(f"[Export] Region analysis -> {out}")
        except: pass

    def _on_close(self):
        """Thorough cleanup to prevent 'main thread is not in main loop' crashes.
        All tkinter objects must be destroyed HERE (main thread) before GC runs."""
        import gc

        # 1. Close all matplotlib figures and destroy their tk widgets
        for key in list(self.figs.keys()):
            try: plt.close(self.figs[key])
            except: pass
        self.figs.clear()

        for key in list(self.canvases.keys()):
            try: self.canvases[key].get_tk_widget().destroy()
            except: pass
        self.canvases.clear()

        for key in list(self.toolbars.keys()):
            try: self.toolbars[key].destroy()
            except: pass
        self.toolbars.clear()

        # 2. Destroy all child widgets (this kills their Variables/Images)
        for w in self.winfo_children():
            try: w.destroy()
            except: pass

        # 3. Clear data references that might hold tkinter objects
        self._mc.clear()
        self._mc_total.clear()
        self.regions = []
        self.platform_labels_df = None
        self.filter_values = set()

        # 4. Nullify tkinter Variables so __del__ won't fire from GC
        for attr in ('plot_mode', 'gse_scope', 'color_column',
                     'ai_label_col', 'filter_search_var', 'merge_regions', 'overlay'):
            v = getattr(self, attr, None)
            if v is not None:
                try: v.set('')
                except: pass

        # 5. Force GC while still on main thread
        gc.collect()

        # 6. Destroy the toplevel itself
        self.destroy()
