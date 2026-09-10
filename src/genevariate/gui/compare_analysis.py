"""
GeneVariate - Compare Analysis Module v2
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, colorchooser
import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import seaborn as sns

from genevariate.core.analysis.bimodality import robust_kde
from genevariate.core.analysis import (
    distance_matrix as _distance_matrix,
    group_summary as _group_summary,
    pairwise_distances as _pairwise_distances,
)
from scipy.spatial.distance import pdist, squareform

from genevariate.gui.theme import (
    AERO, UI_FONT, MONO_FONT, labelframe, style_toolbar, ensure_theme, style_window,
)

# Unified GeneVariate plot stylesheet (graceful fallback if utils missing)
try:
    from genevariate.utils.viz_style import (
        apply_genevariate_style as _apply_gv_style,
        palette_for as _palette_for,
        cmap_for as _cmap_for,
        smart_figsize as _smart_figsize,
        cap_figsize as _cap_figsize,
        make_interactive as _make_interactive,
        attach_point_labels as _point_labels,
        EXPORT_DPI as _EXPORT_DPI,
    )
    _apply_gv_style()
except Exception:
    _EXPORT_DPI = 300
    def _palette_for(n, use_case="discrete"):
        if n <= 10: p = sns.color_palette("tab10", n)
        elif n <= 20: p = sns.color_palette("tab20", n)
        else: p = sns.color_palette("husl", n)
        return [mcolors.to_hex(c) for c in p]
    def _cmap_for(kind="sequential"):
        return "viridis" if kind != "diverging" else "RdBu_r"
    def _smart_figsize(kind="default"): return (10, 6)
    def _cap_figsize(w, h, max_w=16.0, max_h=10.0):
        return (min(w, max_w), min(h, max_h))
    def _make_interactive(fig, **kw): return None
    def _point_labels(artist, labels): return None

# ─── Lazy ML imports ───────────────────────────────────────────────
def _get_sklearn():
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics import silhouette_score
    return PCA, StandardScaler, KMeans, DBSCAN, silhouette_score

def _get_umap():
    try:
        from umap import UMAP; return UMAP
    except ImportError:
        return None

# ─── Constants ─────────────────────────────────────────────────────
AI_COLS = ['Classified_Condition', 'Classified_Tissue', 'Classified_Treatment',
           'Classified_Age']
_LW = 2.3; _LA = 0.88


def _lim(key):
    """How much this window shows, as the user set it."""
    from genevariate.gui import display_limits
    n = display_limits.get(key)
    return 10 ** 9 if n is None else n

# Separator between the sample id and its group in a point's hover label.
_LABEL_SEP = " · "

# ─── Shared helpers ────────────────────────────────────────────────
def _kde(vals, n=300, x_range=None):
    v = np.asarray(vals, dtype=float); v = v[np.isfinite(v)]
    if len(v) < 2 or np.ptp(v) == 0: return None
    try:
        # The estimator the Distribution Classifier counts modes on, so a curve
        # drawn anywhere in the program shows the shape the program names.
        k = robust_kde(v)
        if x_range:
            xs = np.linspace(x_range[0], x_range[1], n)
        else:
            bw = k.factor * v.std(ddof=1)
            pad = max(3.0 * bw, 0.05 * np.ptp(v), 0.01)
            xs = np.linspace(v.min() - pad, v.max() + pad, n)
        return xs, np.maximum(k(xs), 0)
    except: return None

def _clrs(n):
    # Delegate to unified stylesheet for a colorblind-safe, consistent palette.
    return _palette_for(n, use_case="discrete")

def _tr(s, m=28):
    s = str(s); return (s[:m-1] + '\u2026') if len(s) > m else s

def _interactive_legend(fig, legend, artist_map):
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
            for s in ('set_color', 'set_facecolor', 'set_edgecolor'):
                try: getattr(h, s)(r[1])
                except: pass
            for a in arts:
                for s in ('set_color', 'set_facecolor', 'set_edgecolor'):
                    try: getattr(a, s)(r[1])
                    except: pass
            fig.canvas.draw_idle()
    fig.canvas.mpl_connect('pick_event', _pick)

class ScrollFrame(ttk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        # A bare tk.Canvas defaults to platform grey and ttk cannot reach it.
        self.canvas = tk.Canvas(self, highlightthickness=0, bg=AERO["bg_top"])
        vs = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.sf = ttk.Frame(self.canvas)
        self.sf.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        # NOT self._w: tkinter stores this widget's own Tk path name there, and
        # overwriting it with a canvas item id made every pack() of this frame
        # fail with 'bad argument "1": must be name of window'. The failure
        # happened inside an after() callback, so Tk swallowed it and the tab
        # just came up empty.
        self._win_id = self.canvas.create_window((0, 0), window=self.sf, anchor="nw")
        self.canvas.configure(yscrollcommand=vs.set)
        vs.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(self._win_id, width=e.width))
    @property
    def scrollable_frame(self): return self.sf
    def clear(self):
        for w in self.sf.winfo_children(): w.destroy()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CompareDistributionsWindow - THE SINGLE COMPARISON ENGINE
#
#  Used for ALL comparisons: regions, platforms, species, genes.
#  CompareRegionsWindow inherits this and just pre-loads data.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CompareDistributionsWindow(tk.Toplevel):
    """
    Core data structures:
      data_map      : {group_label: pd.Series of expression values}
      bg_map        : {label: pd.Series of background values}
      metadata_df   : Combined DataFrame with GSM, Expression, Group, + AI labels
      group_gsm_map : {group_label: [gsm_ids]}
    """

    def __init__(self, parent, app_ref, title_text="Distribution Comparison"):
        super().__init__(parent)
        self.app = app_ref
        self.title(title_text)
        self.geometry("1750x1080")
        self.transient(parent)

        # Core data
        self.data_map = {}
        self.bg_map = {}
        self.metadata_df = pd.DataFrame()
        self.group_gsm_map = {}
        self.grouping_column = None

        # Plot state
        self.figs = {}; self.canvases = {}; self.toolbars = {}
        self.plot_mode = tk.StringVar(value="both")
        self.color_by = tk.StringVar(value="Group")

        self._install_styles()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _value_axis_label(self):
        """Axis label for the compared platform, generic when they disagree."""
        get = getattr(self.app, 'platform_measurement_label', None)
        md = self.metadata_df
        if get is None or md is None or md.empty or 'Platform' not in md.columns:
            return "expression"
        try:
            labels = {get(p) for p in md['Platform'].dropna().unique()}
        except Exception:
            return "expression"
        return labels.pop() if len(labels) == 1 else "expression"

    def _install_styles(self):
        """Install the shared app theme + this window's named styles so it
        matches the rest of GeneVariate instead of falling back to bare clam."""
        ensure_theme(self)
        style_window(self)
        s = ttk.Style(self)
        s.configure('Section.TLabel', foreground=AERO['accent_dark'],
                    font=(UI_FONT, 14, 'bold'))
        s.configure('Hint.TLabel', foreground=AERO['muted'],
                    font=(UI_FONT, 9, 'italic'))
        s.configure('Field.TLabel', foreground=AERO['text'],
                    font=(UI_FONT, 9, 'bold'))
        s.configure('Value.TLabel', foreground=AERO['accent_dark'],
                    font=(UI_FONT, 9, 'bold'))
        s.configure('Empty.TLabel', foreground=AERO['muted'],
                    font=(UI_FONT, 11))
        s.configure('Error.TLabel', foreground=AERO['danger'],
                    font=(UI_FONT, 11))

    # ══════════════════════════════════════════════════════════════════
    #  UI BUILD
    # ══════════════════════════════════════════════════════════════════
    def _build_ui(self):
        main = ttk.Frame(self, padding=5); main.pack(fill=tk.BOTH, expand=True)

        # Header
        hdr = ttk.Frame(main); hdr.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(hdr, text="Distribution Comparison Engine",
                  style='Section.TLabel').pack(side=tk.LEFT)
        self.status_label = ttk.Label(hdr, text="Ready", style='Hint.TLabel')
        self.status_label.pack(side=tk.RIGHT)

        paned = ttk.PanedWindow(main, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # ── LEFT CONTROLS ──
        ctrl = ttk.Frame(paned, width=300)
        paned.add(ctrl, weight=0)

        # Group selector
        lf_grp = labelframe(ctrl, text="Groups", padding=4)
        lf_grp.pack(fill=tk.BOTH, expand=True, pady=4)
        self.group_listbox = tk.Listbox(lf_grp, selectmode=tk.EXTENDED, height=12,
                                         font=(MONO_FONT, 9), bd=0, relief="flat",
                                         highlightthickness=1,
                                         highlightbackground=AERO['border'],
                                         highlightcolor=AERO['accent'],
                                         background=AERO['panel'], foreground=AERO['text'],
                                         selectbackground=AERO['accent_light'],
                                         selectforeground=AERO['accent_dark'])
        gsb = ttk.Scrollbar(lf_grp, orient="vertical", command=self.group_listbox.yview)
        self.group_listbox.configure(yscrollcommand=gsb.set)
        self.group_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        gsb.pack(side=tk.RIGHT, fill=tk.Y)

        gbtn = ttk.Frame(ctrl); gbtn.pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(gbtn, text="\u2611 All", width=6,
                   command=lambda: self.group_listbox.select_set(0, tk.END)).pack(side=tk.LEFT, padx=2)
        ttk.Button(gbtn, text="\u2610 None", width=6,
                   command=lambda: self.group_listbox.select_clear(0, tk.END)).pack(side=tk.LEFT, padx=2)
        ttk.Button(gbtn, text="\u21bb Refresh", style="Action.TButton",
                   command=self._refresh_all_plots).pack(side=tk.RIGHT, padx=2)

        gi = ttk.Frame(ctrl); gi.pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(gi, text="Grouping:", style='Field.TLabel').pack(side=tk.LEFT)
        self.lbl_grouping = ttk.Label(gi, text="\u2014", style='Value.TLabel')
        self.lbl_grouping.pack(side=tk.LEFT, padx=5)

        # Plot mode
        pm = labelframe(ctrl, text="Plot Mode", padding=2); pm.pack(fill=tk.X, padx=4, pady=4)
        for v, t in [("density", "Density"), ("rug", "Rug"), ("both", "Both")]:
            ttk.Radiobutton(pm, text=t, variable=self.plot_mode,
                            value=v, command=self._refresh_all_plots).pack(side=tk.LEFT, padx=3)

        # Color By
        cb = labelframe(ctrl, text="Color By (PCA/Clustering)", padding=2)
        cb.pack(fill=tk.X, padx=4, pady=2)
        self.color_combo = ttk.Combobox(cb, textvariable=self.color_by, width=22, state='readonly',
                                         values=["Group"])
        self.color_combo.pack(fill=tk.X)
        self.color_combo.bind('<<ComboboxSelected>>',
                              lambda e: self._refresh_all_plots())

        ttk.Button(ctrl, text="Export All", style="Primary.TButton",
                   command=self._export).pack(fill=tk.X, padx=4, pady=(8, 2))

        # ── RIGHT TABS ──
        self.nb = ttk.Notebook(paned, padding=3)
        paned.add(self.nb, weight=1)

        self.t_overlay = ttk.Frame(self.nb); self.nb.add(self.t_overlay, text=" Overlay ")
        self.t_pca     = ttk.Frame(self.nb); self.nb.add(self.t_pca, text=" PCA / UMAP ")
        self.t_cluster = ttk.Frame(self.nb); self.nb.add(self.t_cluster, text=" Clustering ")
        self.t_dist    = ttk.Frame(self.nb); self.nb.add(self.t_dist, text=" Distance ")
        self.t_ai      = ttk.Frame(self.nb); self.nb.add(self.t_ai, text=" Labels ")
        self.t_sep     = ttk.Frame(self.nb); self.nb.add(self.t_sep, text=" Separation ")
        self.t_dpc     = ttk.Frame(self.nb); self.nb.add(self.t_dpc, text=" DPC ")
        self.t_stats   = ttk.Frame(self.nb); self.nb.add(self.t_stats, text=" Statistics ")
        self.t_table   = ttk.Frame(self.nb); self.nb.add(self.t_table, text=" Data Table ")

    # ══════════════════════════════════════════════════════════════════
    #  DATA INJECTION
    # ══════════════════════════════════════════════════════════════════
    def inject_data(self, data_map, bg_map=None, metadata_df=None,
                    group_gsm_map=None, grouping_col=None):
        """
        Inject comparison data. Called by CompareRegionsWindow (subclass)
        or by any external workflow (e.g. open_compare_window).
        """
        self.data_map = data_map or {}
        self.bg_map = bg_map or {}
        self.metadata_df = metadata_df if metadata_df is not None else pd.DataFrame()
        self.group_gsm_map = group_gsm_map or {}
        self.grouping_column = grouping_col

        self.group_listbox.delete(0, tk.END)
        for label in self.data_map:
            n = len(self.data_map[label])
            self.group_listbox.insert(tk.END, f"{label} (n={n})")
        self.group_listbox.select_set(0, tk.END)

        if self.grouping_column:
            self.lbl_grouping.config(text=self.grouping_column)

        opts = ["Group"]
        if not self.metadata_df.empty:
            # Add ALL Classified_* columns (from AI or loaded labels file)
            for c in self.metadata_df.columns:
                if c.startswith('Classified_') and self.metadata_df[c].notna().sum() > 0:
                    opts.append(c)
            # Also add standard metadata columns
            for c in ['Platform', 'Gene', 'Species', 'series_id']:
                if c in self.metadata_df.columns and self.metadata_df[c].notna().sum() > 0:
                    if c not in opts:
                        opts.append(c)
        self.color_combo['values'] = opts
        self.color_by.set("Group")
        self.status_label.config(text=f"{len(data_map)} groups loaded")

    def _get_selected_groups(self):
        sel = self.group_listbox.curselection()
        keys = list(self.data_map.keys())
        if not sel: return keys
        return [keys[i] for i in sel if i < len(keys)]

    # ══════════════════════════════════════════════════════════════════
    #  EMBED / CLOSE
    # ══════════════════════════════════════════════════════════════════
    def _embed(self, fig, parent, key):
        for old_key, store in [(key, self.canvases), (key, self.toolbars), (key, self.figs)]:
            if old_key in store:
                try:
                    if store is self.canvases: store[old_key].get_tk_widget().destroy()
                    elif store is self.toolbars: store[old_key].destroy()
                    else: plt.close(store[old_key])
                except: pass
        c = FigureCanvasTkAgg(fig, master=parent); c.draw()
        _make_interactive(fig, on_pick=self._open_sample_card)
        tb = NavigationToolbar2Tk(c, parent); tb.update()
        style_toolbar(tb)
        tb.pack(side=tk.TOP, fill=tk.X)
        c.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 6))
        self.figs[key] = fig; self.canvases[key] = c; self.toolbars[key] = tb

    def _clear_tab(self, tab):
        for w in tab.winfo_children(): w.destroy()

    def _on_close(self):
        import gc
        for f in self.figs.values():
            try: plt.close(f)
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
        for w in self.winfo_children():
            try: w.destroy()
            except: pass
        gc.collect()
        self.destroy()

    # ══════════════════════════════════════════════════════════════════
    #  REFRESH ALL TABS
    # ══════════════════════════════════════════════════════════════════
    def _refresh_all_plots(self):
        for f in list(self.figs.values()):
            try: plt.close(f)
            except: pass
        self.figs.clear(); self.canvases.clear(); self.toolbars.clear()
        for tab in [self.t_overlay, self.t_pca, self.t_cluster, self.t_dist,
                    self.t_ai, self.t_sep, self.t_dpc, self.t_stats, self.t_table]:
            self._clear_tab(tab)

        self._render_overlay()
        self._render_pca_umap()
        self._render_clustering()
        self._render_dist_matrix()
        self._render_ai_labels()
        self._render_separation()
        self._render_dpc()
        self._render_stats()
        self._render_table()

    # ══════════════════════════════════════════════════════════════════
    #  TAB 1: OVERLAY DENSITIES
    # ══════════════════════════════════════════════════════════════════
    def _render_overlay(self):
        sel = self._get_selected_groups()
        if not sel: return
        mode = self.plot_mode.get()
        scroll = ScrollFrame(self.t_overlay); scroll.pack(fill=tk.BOTH, expand=True)
        fig = Figure(figsize=(16, 8))
        ax = fig.subplots()

        # Background
        for lbl, bg in self.bg_map.items():
            v = bg.dropna().astype(float)
            if len(v) > 10:
                ax.hist(v, bins=min(150, max(60, int(np.sqrt(len(v))))),
                        color='#AAAAAA', alpha=0.4, density=True,
                        edgecolor='#777777', linewidth=0.3, zorder=1)

        # x_range from all data
        all_v = []
        for k in sel:
            if k in self.data_map: all_v.append(self.data_map[k].dropna())
        for bg in self.bg_map.values(): all_v.append(bg.dropna())
        if all_v:
            cat = pd.concat(all_v).astype(float)
            xr = (cat.min() * 0.97, cat.max() * 1.03)
        else:
            xr = None

        colors = _clrs(len(sel))
        amap = {}; handles = []
        for i, label in enumerate(sel):
            if label not in self.data_map: continue
            expr = self.data_map[label].dropna().astype(float)
            if expr.empty: continue
            clr = colors[i]
            lb = f"{_tr(label, 35)} (n={len(expr)})"
            arts = []
            if mode in ("density", "both"):
                kd = _kde(expr, x_range=xr)
                if kd:
                    ln, = ax.plot(kd[0], kd[1], color=clr, lw=_LW, alpha=_LA, zorder=4)
                    arts.append(ln)
                elif len(expr) == 1:
                    vl = ax.axvline(expr.iloc[0], color=clr, ls=':', lw=_LW, alpha=0.7, zorder=4)
                    arts.append(vl)
            if mode in ("rug", "both"):
                sns.rugplot(x=expr, ax=ax, color=clr, height=0.04, alpha=0.4, zorder=5)
                if ax.collections: arts.append(ax.collections[-1])
            amap[lb] = arts
            handles.append(mlines.Line2D([], [], color=clr, lw=_LW, label=lb))

        ax.set_xlabel(self._value_axis_label()); ax.set_ylabel("Density"); ax.set_ylim(bottom=0)
        ax.set_title(f"Distribution Overlay - {len(sel)} groups ({mode})", fontsize=13, weight='bold')
        if handles:
            leg = ax.legend(handles=handles, fontsize=9, loc='upper left',
                            bbox_to_anchor=(1.01, 1.0), framealpha=0.92)
            _interactive_legend(fig, leg, amap)
        fig.subplots_adjust(right=0.72)
        self._embed(fig, scroll.scrollable_frame, "overlay")

    # ══════════════════════════════════════════════════════════════════
    #  TAB 2: PCA / UMAP
    # ══════════════════════════════════════════════════════════════════
    def _render_pca_umap(self):
        sel = self._get_selected_groups()
        df = self._build_ml_df(sel)
        if df.empty or len(sel) < 2:
            ttk.Label(self.t_pca, text="Need \u22652 groups with data.",
                      style='Empty.TLabel').pack(pady=40)
            return
        try:
            PCA, StandardScaler, _, _, _ = _get_sklearn()
        except ImportError:
            ttk.Label(self.t_pca, text="scikit-learn not installed.\npip install scikit-learn",
                      style='Error.TLabel').pack(pady=40)
            return

        X, _, _keep = self._extract_features(df)
        df = df[_keep]
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        pca = PCA(n_components=2)
        coords_pca = pca.fit_transform(X_s)

        UMAP = _get_umap()
        coords_umap = None
        if UMAP and X_s.shape[0] > 15:
            try:
                coords_umap = UMAP(n_components=2, n_neighbors=min(15, X_s.shape[0]-1),
                                    random_state=42).fit_transform(X_s)
            except: pass

        scroll = ScrollFrame(self.t_pca); scroll.pack(fill=tk.BOTH, expand=True)
        sf = scroll.scrollable_frame
        ev = pca.explained_variance_ratio_

        # Plot 1: PCA by Group
        fig1 = Figure(figsize=(14, 8))
        ax1 = fig1.subplots()
        self._scatter(ax1, coords_pca, df['Group'].values, "PCA - by Group",
                      f"PC1 ({ev[0]:.1%})", f"PC2 ({ev[1]:.1%})",
                      ids=self._gsms(df))
        fig1.suptitle("Dimensionality Reduction", fontsize=14, weight='bold')
        fig1.tight_layout(); self._embed(fig1, sf, "pca_group")

        # Plot 2: PCA by the column the "Color By" box names, else the best
        # label column. Honouring the box is the whole point of it being
        # there: it is populated with every usable label column, but until
        # now nothing read it, so choosing a column changed nothing.
        label_col = None
        chosen = self.color_by.get()
        if chosen and chosen != "Group":
            if chosen in df.columns and df[chosen].notna().sum() > 0:
                label_col = chosen
        if not label_col and self.grouping_column \
                and self.grouping_column.startswith('Classified_'):
            if self.grouping_column in df.columns and df[self.grouping_column].notna().sum() > 0:
                label_col = self.grouping_column
        if not label_col:
            label_col = self._best_ai_col(df)

        fig2 = Figure(figsize=(14, 8))
        ax2 = fig2.subplots()
        if label_col:
            nice = label_col.replace('Classified_', '')
            self._scatter(ax2, coords_pca, df[label_col].fillna('N/A').astype(str).values,
                          f"PCA - by {nice}", f"PC1 ({ev[0]:.1%})", f"PC2 ({ev[1]:.1%})",
                          ids=self._gsms(df))
        else:
            ax2.text(0.5, 0.5, "No labels available", ha='center', va='center',
                     transform=ax2.transAxes, color='gray')
            ax2.set_title("PCA - Labels N/A", fontsize=10)
        fig2.tight_layout(); self._embed(fig2, sf, "pca_label")

        # Plot 3: UMAP (if available)
        if coords_umap is not None:
            fig3 = Figure(figsize=(14, 8))
            ax3 = fig3.subplots()
            self._scatter(ax3, coords_umap, df['Group'].values,
                          "UMAP - by Group", "UMAP 1", "UMAP 2",
                          ids=self._gsms(df))
            fig3.tight_layout(); self._embed(fig3, sf, "pca_umap")

    # ══════════════════════════════════════════════════════════════════
    #  TAB 3: CLUSTERING
    # ══════════════════════════════════════════════════════════════════
    def _render_clustering(self):
        sel = self._get_selected_groups()
        df = self._build_ml_df(sel)
        if df.empty or len(sel) < 2: return
        try:
            PCA, StandardScaler, KMeans, DBSCAN, silhouette_score = _get_sklearn()
        except ImportError: return

        X, _, _keep = self._extract_features(df)
        df = df[_keep]
        scaler = StandardScaler(); X_s = scaler.fit_transform(X)
        pca = PCA(n_components=2); coords = pca.fit_transform(X_s)

        # Auto-K via silhouette
        max_k = min(8, len(sel) + 2, X_s.shape[0] - 1)
        best_k = max(2, len(sel)); best_sc = -1
        for k in range(2, max(3, max_k + 1)):
            try:
                km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_s)
                sc = silhouette_score(X_s, km.labels_)
                if sc > best_sc: best_sc = sc; best_k = k
            except: pass

        km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X_s)

        scroll = ScrollFrame(self.t_cluster); scroll.pack(fill=tk.BOTH, expand=True)
        sf = scroll.scrollable_frame

        # Plot 1: K-Means
        fig1 = Figure(figsize=(14, 8))
        ax1 = fig1.subplots()
        self._scatter(ax1, coords, [f"C{c}" for c in km.labels_],
                      f"K-Means (k={best_k}, sil={best_sc:.2f})", "PC1", "PC2",
                      ids=self._gsms(df))
        fig1.suptitle("Clustering Analysis", fontsize=14, weight='bold')
        fig1.tight_layout(); self._embed(fig1, sf, "cluster_km")

        # Plot 2: Ground Truth
        fig2 = Figure(figsize=(14, 8))
        ax2 = fig2.subplots()
        self._scatter(ax2, coords, df['Group'].values, "Groups (Ground Truth)",
                      "PC1", "PC2", ids=self._gsms(df))
        fig2.tight_layout(); self._embed(fig2, sf, "cluster_gt")

        # Plot 3: DBSCAN
        try:
            db = DBSCAN(eps=0.8, min_samples=max(3, X_s.shape[0] // 50)).fit(X_s)
            n_db = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
            lbl = [f"C{c}" if c >= 0 else "Noise" for c in db.labels_]
            fig3 = Figure(figsize=(14, 8))
            ax3 = fig3.subplots()
            self._scatter(ax3, coords, lbl, f"DBSCAN ({n_db} clusters)", "PC1", "PC2",
                          ids=self._gsms(df))
            fig3.tight_layout(); self._embed(fig3, sf, "cluster_db")
        except Exception as e:
            ttk.Label(sf, text=f"DBSCAN Error: {e}", style='Error.TLabel').pack(pady=10)

    # ══════════════════════════════════════════════════════════════════
    #  TAB 4: DISTANCE MATRIX
    # ══════════════════════════════════════════════════════════════════
    def _groups_for_stats(self, sel):
        """The selected groups as plain arrays, the shape core.analysis takes."""
        return {g: self.data_map[g].dropna().astype(float).to_numpy()
                for g in sel if g in self.data_map}

    def _render_dist_matrix(self):
        sel = self._get_selected_groups()
        if len(sel) < 2: return
        n = len(sel)

        scroll = ScrollFrame(self.t_dist); scroll.pack(fill=tk.BOTH, expand=True)
        sf = scroll.scrollable_frame

        # One pass over the pairs, three views of it. The distances come from
        # analysis.distribution_compare so the heatmap here, the Statistics tab
        # below and the assistant all read the same numbers.
        pairs = _pairwise_distances(self._groups_for_stats(sel))
        if pairs.empty:
            ttk.Label(sf, text="No pair of groups had enough values to compare.",
                      style='Empty.TLabel').pack(pady=40)
            return
        present = [g for g in sel
                   if g in set(pairs['group_a']) | set(pairs['group_b'])]
        for mi, (mname, col) in enumerate(
                (('Wasserstein', 'wasserstein'),
                 ('Delta-Mean', 'delta_mean'),
                 ('Jensen-Shannon', 'jensen_shannon'))):
            mat = _distance_matrix(pairs, col, groups=present)
            tl = [_tr(k, 22) for k in present]
            fig_h = max(6, 2.5 + n * 0.5)
            fig = Figure(figsize=(max(10, 2.5 + n * 0.5), fig_h))
            ax = fig.subplots()
            sns.heatmap(pd.DataFrame(mat.to_numpy(), index=tl, columns=tl),
                        annot=True, fmt=".3f", cmap="Blues", ax=ax,
                        linewidths=0.6, linecolor='black')
            ax.set_title(f"Pairwise Distance - {mname}", fontsize=13, weight='bold')
            ax.tick_params(labelsize=9)
            fig.tight_layout()
            self._embed(fig, sf, f"distmat_{mi}")

    # ══════════════════════════════════════════════════════════════════
    #  TAB 5: AI LABELS
    # ══════════════════════════════════════════════════════════════════
    def _render_ai_labels(self):
        if self.metadata_df.empty or 'Group' not in self.metadata_df.columns:
            ttk.Label(self.t_ai, text="No metadata.", style='Empty.TLabel').pack(pady=40)
            return
        active = [c for c in self.metadata_df.columns
                  if c.startswith('Classified_') and self.metadata_df[c].notna().sum() > 0]
        if not active:
            ttk.Label(self.t_ai,
                      text="No label columns available.\n\n"
                           "Load a labels file or run classification first.",
                      style='Empty.TLabel').pack(pady=40)
            return

        scroll = ScrollFrame(self.t_ai); scroll.pack(fill=tk.BOTH, expand=True)
        fig = Figure(figsize=_cap_figsize(18, len(active) * 5))
        axes = fig.subplots(len(active), 2, squeeze=False)

        for ri, lc in enumerate(active):
            nice = lc.replace('Classified_', '').replace('_', ' ')

            # Smart-bin high-cardinality columns before crosstab
            raw = self.metadata_df[lc].fillna('N/A').astype(str)
            n_uniq = raw.nunique()
            binned = False
            if n_uniq > _lim('groups'):
                numeric = pd.to_numeric(self.metadata_df[lc], errors='coerce')
                if numeric.notna().sum() > len(self.metadata_df) * 0.5:
                    try:
                        n_bins = min(12, max(5, n_uniq // 5))
                        raw = pd.cut(numeric, bins=n_bins, duplicates='drop').astype(str).fillna('N/A')
                        binned = True
                    except Exception:
                        top = raw.value_counts().head(_lim('groups') - 1).index
                        raw = raw.where(raw.isin(top), 'Other')
                else:
                    top = raw.value_counts().head(_lim('groups') - 1).index
                    raw = raw.where(raw.isin(top), 'Other')

            suffix = " (binned)" if binned else ""

            # Left: Heatmap %
            ct = pd.crosstab(self.metadata_df['Group'], raw)
            ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
            top_cols = ct.sum().nlargest(_lim('groups')).index.tolist()
            ct_pct = ct_pct[[c for c in top_cols if c in ct_pct.columns]]

            sns.heatmap(ct_pct, annot=True, fmt=".1f", cmap="YlGnBu",
                        ax=axes[ri, 0], linewidths=0.6, linecolor='black',
                        cbar_kws={'label': '%'})
            axes[ri, 0].set_title(f"{nice}{suffix} - % per Group", fontsize=9, weight='bold')
            axes[ri, 0].tick_params(labelsize=7)

            # Right: Grouped bars (top values)
            top_vals = raw.value_counts().head(10).index
            groups = self.metadata_df['Group'].unique()
            x = np.arange(len(top_vals))
            w = 0.8 / max(1, len(groups))
            clrs = _clrs(len(groups))
            for gi, g in enumerate(groups):
                mask = self.metadata_df['Group'] == g
                sub_raw = raw[mask]
                vc = sub_raw.value_counts()
                vals = [vc.get(v, 0) for v in top_vals]
                axes[ri, 1].bar(x + gi * w, vals, w, label=_tr(g, 18),
                                color=clrs[gi], edgecolor='black', lw=0.3)
            axes[ri, 1].set_xticks(x + w * len(groups) / 2)
            axes[ri, 1].set_xticklabels([_tr(v, 18) for v in top_vals],
                                         rotation=35, ha='right', fontsize=7)
            axes[ri, 1].set_title(f"{nice} - Counts", fontsize=9, weight='bold')
            axes[ri, 1].legend(fontsize=6, ncol=2)

        fig.suptitle("Label Cross-Analysis", fontsize=14, weight='bold')
        fig.tight_layout()
        self._embed(fig, scroll.scrollable_frame, "ai")

    # ══════════════════════════════════════════════════════════════════
    #  TAB 6: SEPARATION (Strip + Violin)
    # ══════════════════════════════════════════════════════════════════
    def _render_separation(self):
        sel = self._get_selected_groups()
        dfs = []
        for g in sel:
            if g not in self.data_map: continue
            v = self.data_map[g].dropna().astype(float)
            dfs.append(pd.DataFrame({'Expression': v, 'Group': g}))
        if not dfs: return
        full = pd.concat(dfs, ignore_index=True)
        colors = _clrs(len(sel))
        cmap = {g: c for g, c in zip(sel, colors)}

        scroll = ScrollFrame(self.t_sep); scroll.pack(fill=tk.BOTH, expand=True)
        sf = scroll.scrollable_frame

        # Plot 1: Strip + Mean
        fig1 = Figure(figsize=(16, 8))
        ax1 = fig1.subplots()
        sns.stripplot(data=full, x='Group', y='Expression', hue='Group',
                      palette=cmap, jitter=0.25, ax=ax1, legend=False, alpha=0.5, size=4)
        sns.pointplot(data=full, x='Group', y='Expression', estimator='mean',
                      color='black', linestyles='none', capsize=0.1, markers='D', ax=ax1)
        ax1.set_title("Strip + Mean", fontsize=13, weight='bold')
        plt.setp(ax1.get_xticklabels(), rotation=30, ha='right')
        fig1.tight_layout(); self._embed(fig1, sf, "sep_strip")

        # Plot 2: Violin
        fig2 = Figure(figsize=(16, 8))
        ax2 = fig2.subplots()
        sns.violinplot(data=full, x='Group', y='Expression', hue='Group',
                       palette=cmap, ax=ax2, inner='box', legend=False)
        ax2.set_title("Violin Plot", fontsize=13, weight='bold')
        plt.setp(ax2.get_xticklabels(), rotation=30, ha='right')
        fig2.tight_layout(); self._embed(fig2, sf, "sep_violin")

    # ══════════════════════════════════════════════════════════════════
    #  TAB 7: DPC DECISION GRAPH (Density Peak Clustering)
    # ══════════════════════════════════════════════════════════════════
    def _render_dpc(self):
        sel = self._get_selected_groups()
        all_v, lbls = [], []
        for g in sel:
            if g not in self.data_map: continue
            v = self.data_map[g].dropna().astype(float).tolist()
            all_v.extend(v); lbls.extend([g] * len(v))
        if len(all_v) < 10: return

        scroll = ScrollFrame(self.t_dpc); scroll.pack(fill=tk.BOTH, expand=True)
        sf = scroll.scrollable_frame

        X = np.array(all_v).reshape(-1, 1)
        L = np.array(lbls)

        try:
            dists = squareform(pdist(X))
            dc = np.percentile(dists, 2)
            if dc == 0: dc = 1e-5

            rho = np.sum(np.exp(-(dists / dc) ** 2), axis=1) - 1
            delta = np.zeros(len(X))
            ord_rho = np.argsort(-rho)
            for i, idx in enumerate(ord_rho):
                if i == 0:
                    delta[idx] = dists[idx, :].max()
                else:
                    higher = ord_rho[:i]
                    delta[idx] = dists[idx, higher].min()

            fig = Figure(figsize=(14, 8))
            ax = fig.subplots()
            uniq = list(dict.fromkeys(lbls))
            colors = _clrs(len(uniq))
            cmap_d = {g: c for g, c in zip(uniq, colors)}
            for g in uniq:
                mask = L == g
                ax.scatter(rho[mask], delta[mask], c=cmap_d[g],
                           label=f"{_tr(g)} ({mask.sum()})", alpha=0.7, s=30,
                           edgecolors='black', lw=0.3)

            ax.set_xlabel("Local Density (rho)", fontsize=11)
            ax.set_ylabel("Min Distance to Higher Density (delta)", fontsize=11)
            ax.set_title("Density Peak Clustering - Decision Graph", fontsize=13, weight='bold')
            ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
            fig.tight_layout()
            self._embed(fig, sf, "dpc")

        except Exception as e:
            ttk.Label(sf, text=f"DPC Error: {e}", style='Error.TLabel').pack(pady=30)

    # ══════════════════════════════════════════════════════════════════
    #  TAB 8: STATISTICS
    # ══════════════════════════════════════════════════════════════════
    def _render_stats(self):
        sel = self._get_selected_groups()
        cols = ("Group A", "Group B", "Metric", "Value", "Sig")
        tree = ttk.Treeview(self.t_stats, columns=cols, show="headings")
        for c, w in zip(cols, [200, 200, 160, 130, 90]):
            tree.heading(c, text=c)
            tree.column(c, width=w, minwidth=70, anchor=tk.CENTER)
        sb = ttk.Scrollbar(self.t_stats, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); sb.pack(side=tk.RIGHT, fill=tk.Y)

        groups = self._groups_for_stats(sel)
        summary = _group_summary(groups)
        pairs = _pairwise_distances(groups)
        # Both frames are attached for export, so a saved CSV is the whole
        # comparison rather than the rows that fit in the widget.
        tree._export_frame = pairs

        for _, r in summary.iterrows():
            for m, txt in (("N", f"{int(r['n'])}"), ("Mean", f"{r['mean']:.4f}"),
                           ("Median", f"{r['median']:.4f}"),
                           ("Std", f"{r['sd']:.4f}"), ("IQR", f"{r['iqr']:.4f}")):
                tree.insert("", tk.END, values=(_tr(r['group'], 25), "-", m, txt, ""))

        if not pairs.empty:
            tree.insert("", tk.END, values=("─" * 16, "─" * 16, "PAIRWISE", "─" * 10, ""))
            for _, r in pairs.iterrows():
                a, b = _tr(r['group_a'], 20), _tr(r['group_b'], 20)
                sig = r['significance']
                tree.insert("", tk.END, values=(a, b, "Wilcoxon Z",
                                                f"{r['rank_sum_z']:.4f}", sig))
                tree.insert("", tk.END, values=(a, b, "p-value",
                                                f"{r['p_value']:.2e}", sig))
                # The star now follows the BH q, not the raw p. With k groups
                # there are k(k-1)/2 comparisons on this one screen, and at ten
                # groups two of them clear p<0.05 on noise alone; marking those
                # significant was reporting the multiplicity as a result.
                tree.insert("", tk.END, values=(a, b, "q (BH, all pairs)",
                                                f"{r['q_value']:.2e}", sig))
                tree.insert("", tk.END, values=(a, b, "Wasserstein",
                                                f"{r['wasserstein']:.4f}",
                                                r['separation'].capitalize()))
                tree.insert("", tk.END, values=(a, b, "Jensen-Shannon",
                                                f"{r['jensen_shannon']:.4f}", ""))

    # ══════════════════════════════════════════════════════════════════
    #  TAB 9: DATA TABLE
    # ══════════════════════════════════════════════════════════════════
    def _render_table(self):
        if self.metadata_df.empty:
            ttk.Label(self.t_table, text="No metadata loaded.",
                      style='Empty.TLabel').pack(pady=30)
            return
        df = self.metadata_df
        # Build priority columns: standard ones + ALL Classified_* columns
        cls_cols = sorted([c for c in df.columns if c.startswith('Classified_')])
        pri = ['Group', 'GSM', 'Expression', 'series_id', 'title',
               'source_name_ch1'] + cls_cols + ['Platform', 'Gene', 'Species']
        cols = [c for c in pri if c in df.columns]
        cols += [c for c in df.columns if c not in cols]
        cols = cols[:_lim('table_columns')]

        tree = ttk.Treeview(self.t_table, columns=cols, show="headings", selectmode="extended")
        vsb = ttk.Scrollbar(self.t_table, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(self.t_table, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        for c in cols:
            tree.heading(c, text=c.replace('_', ' '))
            tree.column(c, width=120, minwidth=80, stretch=False, anchor=tk.CENTER)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        max_rows = 2000
        for _, row in df.head(max_rows).iterrows():
            tree.insert("", tk.END, values=[str(row.get(c, ''))[:80] for c in cols])
        if len(df) > max_rows:
            ttk.Label(self.t_table,
                      text=f"Showing {max_rows:,} / {len(df):,} rows.",
                      style='Hint.TLabel').pack(pady=2)

    # ══════════════════════════════════════════════════════════════════
    #  ML HELPERS (shared by PCA, Clustering, etc.)
    # ══════════════════════════════════════════════════════════════════
    def _build_ml_df(self, sel):
        frames = []
        # Detect ALL Classified_* columns in metadata dynamically
        all_cls_cols = [c for c in self.metadata_df.columns
                        if c.startswith('Classified_')] if not self.metadata_df.empty else []
        for g in sel:
            if g not in self.data_map: continue
            expr = self.data_map[g].dropna().astype(float)
            sub = pd.DataFrame({'Expression': expr, 'Group': g})
            if not self.metadata_df.empty and g in self.metadata_df['Group'].values:
                grp_meta = self.metadata_df[self.metadata_df['Group'] == g]
                # GSM has to travel with the row: _scatter turns it into the
                # point label that _open_sample_card reads back. Without it
                # _gsms() returned None and clicking a point did nothing.
                # The extra names are exactly the non-Classified entries
                # inject_data offers in the Color By box; without them here
                # picking one of those left the PCA uncoloured.
                carry = [c for c in (['GSM'] + all_cls_cols
                                     + ['Platform', 'Gene', 'Species', 'series_id'])
                         if c in grp_meta.columns]
                # Both frames descend from the same platform table, so the
                # index is the trustworthy join. Fall back to positional only
                # when a caller has already discarded it.
                by_index = (grp_meta.index.is_unique and sub.index.is_unique
                            and sub.index.isin(grp_meta.index).all())
                for ac in carry:
                    if by_index:
                        sub[ac] = grp_meta[ac].reindex(sub.index)
                    else:
                        vals = grp_meta[ac].values
                        sub[ac] = vals[:len(sub)] if len(vals) >= len(sub) else \
                                  list(vals) + ['N/A'] * (len(sub) - len(vals))
            frames.append(sub)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _extract_features(self, df):
        """Feature matrix, the rows that survived, and the feature names.

        A sample with no expression value is dropped, not entered at zero.
        Zero is an ordinary level on this scale, so filling with it gathers
        the unmeasured samples into one place in the projection and draws
        that as if it were structure in the data.
        """
        keep = df['Expression'].notna().to_numpy()
        df = df[keep]
        feat_cols = ['Expression']
        X_parts = [df[['Expression']].values]
        # Use ALL Classified_* columns present in the df
        cls_cols = [c for c in df.columns if c.startswith('Classified_')]
        for ac in cls_cols:
            if ac in df.columns:
                dum = pd.get_dummies(df[ac].fillna('N/A').astype(str), prefix=ac[:10])
                X_parts.append(dum.values); feat_cols.extend(dum.columns.tolist())
        X = np.column_stack(X_parts) if len(X_parts) > 1 else X_parts[0]
        if X.shape[1] < 2:
            # A single feature cannot be projected in two dimensions. The pad
            # column is drawn from a seeded generator so the same data gives
            # the same picture twice; an unseeded draw moved the axis
            # percentages, the auto-selected k and the silhouette on every
            # redraw of an unchanged comparison.
            rng = np.random.default_rng(0)
            X = np.column_stack([X, rng.standard_normal(X.shape[0]) * 0.001])
        return X, feat_cols, keep

    def _best_ai_col(self, df):
        # Standard AI_COLS, then any Classified_*, then a curator's own label
        # column: a hand-made sheet uses none of those names and would
        # otherwise leave the embedding uncoloured.
        for ac in AI_COLS:
            if ac in df.columns and df[ac].notna().sum() > 0: return ac
        for c in df.columns:
            if c.startswith('Classified_') and df[c].notna().sum() > 0: return c
        try:
            from genevariate.core.label_entities import label_value_columns
            for c in label_value_columns(df):
                if df[c].notna().sum() > 0: return c
        except Exception:
            pass
        return None

    @staticmethod
    def _gsms(df):
        """Accessions in row order, or ``None`` if this frame has none.

        The embeddings are computed from ``df`` without dropping rows, so row
        order is what ties a point back to its sample.
        """
        if 'GSM' not in df.columns:
            return None
        return df['GSM'].astype(str).values

    def _open_sample_card(self, pick):
        """Show the record for the embedding point the user clicked."""
        from genevariate.gui.windows.sample_card import show_sample_card

        gsm = str(pick.label).split(_LABEL_SEP)[0].strip()
        show_sample_card(self, gsm, self.metadata_df)

    def _scatter(self, ax, coords, labels, title, xlabel, ylabel, ids=None):
        """``ids`` names the sample behind each point so it can be clicked."""
        uniq = list(pd.Series(labels).value_counts().head(_lim('groups')).index)
        clrs = _clrs(len(uniq))
        ids = None if ids is None else np.asarray(ids, dtype=object)

        def _names(mask, group):
            # The legend entry carries a count, which is noise in a tooltip.
            if ids is None:
                return [group] * int(mask.sum())
            return [f"{i}{_LABEL_SEP}{group}" for i in ids[mask]]

        for i, u in enumerate(uniq):
            mask = np.array(labels) == u
            sc = ax.scatter(coords[mask, 0], coords[mask, 1], c=clrs[i],
                       label=f"{_tr(u)} ({mask.sum()})", alpha=0.7, s=25,
                       edgecolors='black', lw=0.3)
            _point_labels(sc, _names(mask, _tr(u)))
        other = ~np.isin(labels, uniq)
        if other.any():
            sc = ax.scatter(coords[other, 0], coords[other, 1], c='#CCCCCC',
                       label=f"Other ({other.sum()})", alpha=0.3, s=15)
            _point_labels(sc, _names(other, "Other"))
        ax.set_title(title, fontsize=10, weight='bold')
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, ncol=max(1, len(uniq) // 6), framealpha=0.9)

    def _export(self):
        d = filedialog.askdirectory(title="Export Folder", parent=self)
        if not d: return
        out = Path(d); out.mkdir(parents=True, exist_ok=True)
        # Every figure and every table on screen, not just the figures.
        from genevariate.gui.exporting import export_window
        shown = export_window(self, out, figures=self.figs, prefix="compare_")
        if not self.metadata_df.empty:
            self.metadata_df.to_csv(out / "compare_data.csv", index=False)
        messagebox.showinfo("Exported",
                            f"Saved to {out}\n- {len(shown)} figures and tables",
                            parent=self)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CompareRegionsWindow - THIN SUBCLASS
#
#  Inherits 100% of CompareDistributionsWindow.
#  Only difference: packages selected regions -> inject_data() -> auto-run.
#  Same tabs, same PCA, same clustering, same everything.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CompareRegionsWindow(CompareDistributionsWindow):
    """
    Called from _compare_regions_logic() with pre-selected histogram regions.

    regions_data: list[dict] with keys:
        label, gene, platform, column, range, color,
        expression_values, gsm_list, meta_df, ai_labels_df, platform_df
    """

    def __init__(self, parent, app_ref, regions_data):
        n = len(regions_data)
        super().__init__(parent, app_ref, title_text=f"Compare Regions ({n} regions)")

        # ── Package regions into the engine's data structures ──
        data_map = {}
        bg_map = {}
        group_gsm_map = {}
        frames = []

        for r in regions_data:
            label = r['label']
            col = r['column']
            expr = r['expression_values'].dropna().astype(float)
            data_map[label] = expr
            group_gsm_map[label] = r['gsm_list']

            # Background (once per platform)
            bg_key = f"Platform ({r.get('platform', 'BG')})"
            if bg_key not in bg_map:
                bg_df = r.get('platform_df', pd.DataFrame())
                if not bg_df.empty and col in bg_df.columns:
                    bg_map[bg_key] = bg_df[col].dropna().astype(float)

            # Build metadata per-GSM
            gsms = r['gsm_list']
            bg_df = r.get('platform_df', pd.DataFrame())
            meta = r.get('meta_df', pd.DataFrame())
            ai = r.get('ai_labels_df', pd.DataFrame())

            if not bg_df.empty and 'GSM' in bg_df.columns:
                sub = bg_df[bg_df['GSM'].isin(set(gsms))][['GSM', col]].copy()
                sub.rename(columns={col: 'Expression'}, inplace=True)
            else:
                sub = pd.DataFrame({'GSM': gsms, 'Expression': expr.values[:len(gsms)]})

            # A left merge on a de-duplicated key keeps row order but hands
            # back a fresh RangeIndex. The platform index is what ties these
            # rows to the expression Series in data_map, so restore it.
            src_index = sub.index

            if not meta.empty:
                mc = 'gsm' if ('gsm' in meta.columns and 'GSM' not in meta.columns) else 'GSM'
                ms = meta.rename(columns={mc: 'GSM'}) if mc != 'GSM' else meta
                kp = ['GSM'] + [c for c in ms.columns if c != 'GSM' and c not in sub.columns]
                sub = sub.merge(ms[kp].drop_duplicates('GSM'), on='GSM', how='left')

            if not ai.empty:
                ac = 'GSM' if 'GSM' in ai.columns else 'gsm'
                ais = ai.rename(columns={ac: 'GSM'}) if ac != 'GSM' else ai
                cls = ['GSM'] + [c for c in ais.columns if c.startswith('Classified_')]
                sub = sub.merge(ais[cls].drop_duplicates('GSM'), on='GSM', how='left')

            if len(sub) == len(src_index):
                sub.index = src_index

            sub['Group'] = label
            sub['Platform'] = r.get('platform', 'Unknown')
            sub['Gene'] = r.get('gene', col)
            frames.append(sub)

        # Keep the per-region index: _build_ml_df aligns metadata to the
        # expression Series through it.
        metadata_df = pd.concat(frames) if frames else pd.DataFrame()

        # ── Inject into the engine ──
        self.inject_data(
            data_map=data_map,
            bg_map=bg_map,
            metadata_df=metadata_df,
            group_gsm_map=group_gsm_map,
            grouping_col="Region"
        )

        # ── Auto-trigger all analysis ──
        self.after(200, self._refresh_all_plots)
