"""
GeneVariate - Compare Analysis Module v2
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, colorchooser
import pandas as pd
import numpy as np
import itertools
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.stats import gaussian_kde, ranksums, wasserstein_distance
from scipy.spatial.distance import pdist, squareform, jensenshannon

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
           'Classified_Age', 'Classified_Treatment_Time']
_MAX_GRP = 20; _LW = 2.3; _LA = 0.88

# ─── Shared helpers ────────────────────────────────────────────────
def _kde(vals, n=300, x_range=None):
    v = np.asarray(vals, dtype=float); v = v[np.isfinite(v)]
    if len(v) < 2 or np.ptp(v) == 0: return None
    try:
        k = gaussian_kde(v)
        if x_range:
            xs = np.linspace(x_range[0], x_range[1], n)
        else:
            bw = k.factor * v.std(ddof=1)
            pad = max(3.0 * bw, 0.05 * np.ptp(v), 0.01)
            xs = np.linspace(v.min() - pad, v.max() + pad, n)
        return xs, np.maximum(k(xs), 0)
    except: return None

def _clrs(n):
    if n <= 10: p = sns.color_palette("tab10", n)
    elif n <= 20: p = sns.color_palette("tab20", n)
    else: p = sns.color_palette("gist_ncar", n)
    return [mcolors.to_hex(c) for c in p]

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
        self.canvas = tk.Canvas(self, highlightthickness=0)
        vs = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.sf = ttk.Frame(self.canvas)
        self.sf.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self._w = self.canvas.create_window((0, 0), window=self.sf, anchor="nw")
        self.canvas.configure(yscrollcommand=vs.set)
        vs.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(self._w, width=e.width))
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

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ══════════════════════════════════════════════════════════════════
    #  UI BUILD
    # ══════════════════════════════════════════════════════════════════
    def _build_ui(self):
        main = ttk.Frame(self, padding=5); main.pack(fill=tk.BOTH, expand=True)

        # Header
        hdr = ttk.Frame(main); hdr.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(hdr, text="Distribution Comparison Engine",
                  font=("Segoe UI", 14, "bold"), foreground="#333").pack(side=tk.LEFT)
        self.status_label = ttk.Label(hdr, text="Ready", foreground="grey")
        self.status_label.pack(side=tk.RIGHT)

        paned = ttk.PanedWindow(main, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # ── LEFT CONTROLS ──
        ctrl = ttk.Frame(paned, width=300)
        paned.add(ctrl, weight=0)

        # Group selector
        lf_grp = ttk.LabelFrame(ctrl, text="Groups", padding=4)
        lf_grp.pack(fill=tk.BOTH, expand=True, pady=4)
        self.group_listbox = tk.Listbox(lf_grp, selectmode=tk.EXTENDED, height=12,
                                         font=("Consolas", 9))
        gsb = ttk.Scrollbar(lf_grp, orient="vertical", command=self.group_listbox.yview)
        self.group_listbox.configure(yscrollcommand=gsb.set)
        self.group_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        gsb.pack(side=tk.RIGHT, fill=tk.Y)

        gbtn = ttk.Frame(ctrl); gbtn.pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(gbtn, text="[v] All", width=6,
                   command=lambda: self.group_listbox.select_set(0, tk.END)).pack(side=tk.LEFT, padx=2)
        ttk.Button(gbtn, text="[ ] None", width=6,
                   command=lambda: self.group_listbox.select_clear(0, tk.END)).pack(side=tk.LEFT, padx=2)
        ttk.Button(gbtn, text="> Refresh", command=self._refresh_all_plots).pack(side=tk.RIGHT, padx=2)

        gi = ttk.Frame(ctrl); gi.pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(gi, text="Grouping:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        self.lbl_grouping = ttk.Label(gi, text="[None]", foreground="#1976D2")
        self.lbl_grouping.pack(side=tk.LEFT, padx=5)

        # Plot mode
        pm = ttk.LabelFrame(ctrl, text="Plot Mode", padding=2); pm.pack(fill=tk.X, padx=4, pady=4)
        for v, t in [("density", "Density"), ("rug", "Rug"), ("both", "Both")]:
            ttk.Radiobutton(pm, text=t, variable=self.plot_mode,
                            value=v, command=self._refresh_all_plots).pack(side=tk.LEFT, padx=3)

        # Color By
        cb = ttk.LabelFrame(ctrl, text="Color By (PCA/Clustering)", padding=2)
        cb.pack(fill=tk.X, padx=4, pady=2)
        self.color_combo = ttk.Combobox(cb, textvariable=self.color_by, width=22, state='readonly',
                                         values=["Group"])
        self.color_combo.pack(fill=tk.X)

        ttk.Button(ctrl, text="* Export All", command=self._export).pack(fill=tk.X, padx=4, pady=(8, 2))

        # ── RIGHT TABS ──
        self.nb = ttk.Notebook(paned, padding=3)
        paned.add(self.nb, weight=1)

        self.t_overlay = ttk.Frame(self.nb); self.nb.add(self.t_overlay, text="* Overlay")
        self.t_pca     = ttk.Frame(self.nb); self.nb.add(self.t_pca, text="[Chart] PCA / UMAP")
        self.t_cluster = ttk.Frame(self.nb); self.nb.add(self.t_cluster, text="* Clustering")
        self.t_dist    = ttk.Frame(self.nb); self.nb.add(self.t_dist, text="* Distance Matrix")
        self.t_ai      = ttk.Frame(self.nb); self.nb.add(self.t_ai, text="* Labels")
        self.t_sep     = ttk.Frame(self.nb); self.nb.add(self.t_sep, text="* Separation")
        self.t_dpc     = ttk.Frame(self.nb); self.nb.add(self.t_dpc, text="* DPC")
        self.t_stats   = ttk.Frame(self.nb); self.nb.add(self.t_stats, text="* Statistics")
        self.t_table   = ttk.Frame(self.nb); self.nb.add(self.t_table, text="[File] Data Table")

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
        tb = NavigationToolbar2Tk(c, parent); tb.update()
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
        fig, ax = plt.subplots(figsize=(16, 8))

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

        ax.set_xlabel("Expression"); ax.set_ylabel("Density"); ax.set_ylim(bottom=0)
        ax.set_title(f"Distribution Overlay - {len(sel)} groups ({mode})", fontsize=13, weight='bold')
        if handles:
            leg = ax.legend(handles=handles, fontsize=7, loc='upper left',
                            bbox_to_anchor=(1.01, 1.0), framealpha=0.92)
            _interactive_legend(fig, leg, amap)
        plt.subplots_adjust(right=0.72)
        self._embed(fig, scroll.scrollable_frame, "overlay")

    # ══════════════════════════════════════════════════════════════════
    #  TAB 2: PCA / UMAP
    # ══════════════════════════════════════════════════════════════════
    def _render_pca_umap(self):
        sel = self._get_selected_groups()
        df = self._build_ml_df(sel)
        if df.empty or len(sel) < 2:
            ttk.Label(self.t_pca, text="Need >=2 groups with data.",
                      foreground="gray", font=("Segoe UI", 11)).pack(pady=40)
            return
        try:
            PCA, StandardScaler, _, _, _ = _get_sklearn()
        except ImportError:
            ttk.Label(self.t_pca, text="scikit-learn not installed.\npip install scikit-learn",
                      foreground="red", font=("Segoe UI", 11)).pack(pady=40)
            return

        X, _ = self._extract_features(df)
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
        fig1, ax1 = plt.subplots(figsize=(14, 8))
        self._scatter(ax1, coords_pca, df['Group'].values, "PCA - by Group",
                      f"PC1 ({ev[0]:.1%})", f"PC2 ({ev[1]:.1%})")
        fig1.suptitle("Dimensionality Reduction", fontsize=14, weight='bold')
        plt.tight_layout(); self._embed(fig1, sf, "pca_group")

        # Plot 2: PCA by best label column
        label_col = None
        if self.grouping_column and self.grouping_column.startswith('Classified_'):
            if self.grouping_column in df.columns and df[self.grouping_column].notna().sum() > 0:
                label_col = self.grouping_column
        if not label_col:
            label_col = self._best_ai_col(df)

        fig2, ax2 = plt.subplots(figsize=(14, 8))
        if label_col:
            nice = label_col.replace('Classified_', '')
            self._scatter(ax2, coords_pca, df[label_col].fillna('N/A').astype(str).values,
                          f"PCA - by {nice}", f"PC1 ({ev[0]:.1%})", f"PC2 ({ev[1]:.1%})")
        else:
            ax2.text(0.5, 0.5, "No labels available", ha='center', va='center',
                     transform=ax2.transAxes, color='gray')
            ax2.set_title("PCA - Labels N/A", fontsize=10)
        plt.tight_layout(); self._embed(fig2, sf, "pca_label")

        # Plot 3: UMAP (if available)
        if coords_umap is not None:
            fig3, ax3 = plt.subplots(figsize=(14, 8))
            self._scatter(ax3, coords_umap, df['Group'].values,
                          "UMAP - by Group", "UMAP 1", "UMAP 2")
            plt.tight_layout(); self._embed(fig3, sf, "pca_umap")

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

        X, _ = self._extract_features(df)
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
        fig1, ax1 = plt.subplots(figsize=(14, 8))
        self._scatter(ax1, coords, [f"C{c}" for c in km.labels_],
                      f"K-Means (k={best_k}, sil={best_sc:.2f})", "PC1", "PC2")
        fig1.suptitle("Clustering Analysis", fontsize=14, weight='bold')
        plt.tight_layout(); self._embed(fig1, sf, "cluster_km")

        # Plot 2: Ground Truth
        fig2, ax2 = plt.subplots(figsize=(14, 8))
        self._scatter(ax2, coords, df['Group'].values, "Groups (Ground Truth)", "PC1", "PC2")
        plt.tight_layout(); self._embed(fig2, sf, "cluster_gt")

        # Plot 3: DBSCAN
        try:
            db = DBSCAN(eps=0.8, min_samples=max(3, X_s.shape[0] // 50)).fit(X_s)
            n_db = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
            lbl = [f"C{c}" if c >= 0 else "Noise" for c in db.labels_]
            fig3, ax3 = plt.subplots(figsize=(14, 8))
            self._scatter(ax3, coords, lbl, f"DBSCAN ({n_db} clusters)", "PC1", "PC2")
            plt.tight_layout(); self._embed(fig3, sf, "cluster_db")
        except Exception as e:
            ttk.Label(sf, text=f"DBSCAN Error: {e}", foreground="red").pack(pady=10)

    # ══════════════════════════════════════════════════════════════════
    #  TAB 4: DISTANCE MATRIX
    # ══════════════════════════════════════════════════════════════════
    def _render_dist_matrix(self):
        sel = self._get_selected_groups()
        if len(sel) < 2: return
        n = len(sel)

        scroll = ScrollFrame(self.t_dist); scroll.pack(fill=tk.BOTH, expand=True)
        sf = scroll.scrollable_frame

        metrics = {
            'Wasserstein': lambda a, b: wasserstein_distance(a, b),
            'Delta-Mean': lambda a, b: abs(a.mean() - b.mean()),
            'Jensen-Shannon': lambda a, b: self._js_dist(a, b),
        }
        for mi, (mname, mfn) in enumerate(metrics.items()):
            mat = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        d1 = self.data_map.get(sel[i], pd.Series(dtype=float)).dropna()
                        d2 = self.data_map.get(sel[j], pd.Series(dtype=float)).dropna()
                        try: mat[i, j] = mfn(d1, d2)
                        except: mat[i, j] = np.nan
            tl = [_tr(k, 22) for k in sel]
            fig_h = max(6, 2.5 + n * 0.5)
            fig, ax = plt.subplots(figsize=(max(10, 2.5 + n * 0.5), fig_h))
            sns.heatmap(pd.DataFrame(mat, index=tl, columns=tl),
                        annot=True, fmt=".3f", cmap="YlOrRd", ax=ax,
                        linewidths=0.5, linecolor='white')
            ax.set_title(f"Pairwise Distance - {mname}", fontsize=13, weight='bold')
            ax.tick_params(labelsize=9)
            plt.tight_layout()
            self._embed(fig, sf, f"distmat_{mi}")

    @staticmethod
    def _js_dist(a, b):
        lo = min(a.min(), b.min()); hi = max(a.max(), b.max())
        bins = np.linspace(lo, hi, 50)
        p, _ = np.histogram(a, bins=bins, density=True)
        q, _ = np.histogram(b, bins=bins, density=True)
        p = p / (p.sum() + 1e-10); q = q / (q.sum() + 1e-10)
        return jensenshannon(p, q)

    # ══════════════════════════════════════════════════════════════════
    #  TAB 5: AI LABELS
    # ══════════════════════════════════════════════════════════════════
    def _render_ai_labels(self):
        if self.metadata_df.empty or 'Group' not in self.metadata_df.columns:
            ttk.Label(self.t_ai, text="No metadata.", foreground="gray").pack(pady=40)
            return
        active = [c for c in self.metadata_df.columns
                  if c.startswith('Classified_') and self.metadata_df[c].notna().sum() > 0]
        if not active:
            ttk.Label(self.t_ai,
                      text="No label columns available.\n\n"
                           "Load a labels file or run classification first.",
                      font=("Segoe UI", 11), foreground="gray").pack(pady=40)
            return

        scroll = ScrollFrame(self.t_ai); scroll.pack(fill=tk.BOTH, expand=True)
        fig, axes = plt.subplots(len(active), 2, figsize=(18, len(active) * 5), squeeze=False)

        for ri, lc in enumerate(active):
            nice = lc.replace('Classified_', '').replace('_', ' ')

            # Smart-bin high-cardinality columns before crosstab
            raw = self.metadata_df[lc].fillna('N/A').astype(str)
            n_uniq = raw.nunique()
            binned = False
            if n_uniq > _MAX_GRP:
                numeric = pd.to_numeric(self.metadata_df[lc], errors='coerce')
                if numeric.notna().sum() > len(self.metadata_df) * 0.5:
                    try:
                        n_bins = min(12, max(5, n_uniq // 5))
                        raw = pd.cut(numeric, bins=n_bins, duplicates='drop').astype(str).fillna('N/A')
                        binned = True
                    except Exception:
                        top = raw.value_counts().head(_MAX_GRP - 1).index
                        raw = raw.where(raw.isin(top), 'Other')
                else:
                    top = raw.value_counts().head(_MAX_GRP - 1).index
                    raw = raw.where(raw.isin(top), 'Other')

            suffix = " (binned)" if binned else ""

            # Left: Heatmap %
            ct = pd.crosstab(self.metadata_df['Group'], raw)
            ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
            top_cols = ct.sum().nlargest(_MAX_GRP).index.tolist()
            ct_pct = ct_pct[[c for c in top_cols if c in ct_pct.columns]]

            sns.heatmap(ct_pct, annot=True, fmt=".1f", cmap="YlGnBu",
                        ax=axes[ri, 0], linewidths=0.5, cbar_kws={'label': '%'})
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
        plt.tight_layout()
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
        fig1, ax1 = plt.subplots(figsize=(16, 8))
        sns.stripplot(data=full, x='Group', y='Expression', hue='Group',
                      palette=cmap, jitter=0.25, ax=ax1, legend=False, alpha=0.5, size=4)
        sns.pointplot(data=full, x='Group', y='Expression', estimator='mean',
                      color='black', linestyles='none', capsize=0.1, markers='D', ax=ax1)
        ax1.set_title("Strip + Mean", fontsize=13, weight='bold')
        plt.setp(ax1.get_xticklabels(), rotation=30, ha='right')
        plt.tight_layout(); self._embed(fig1, sf, "sep_strip")

        # Plot 2: Violin
        fig2, ax2 = plt.subplots(figsize=(16, 8))
        sns.violinplot(data=full, x='Group', y='Expression', hue='Group',
                       palette=cmap, ax=ax2, inner='box', legend=False)
        ax2.set_title("Violin Plot", fontsize=13, weight='bold')
        plt.setp(ax2.get_xticklabels(), rotation=30, ha='right')
        plt.tight_layout(); self._embed(fig2, sf, "sep_violin")

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

            fig, ax = plt.subplots(figsize=(14, 8))
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
            plt.tight_layout()
            self._embed(fig, sf, "dpc")

        except Exception as e:
            ttk.Label(sf, text=f"DPC Error: {e}", foreground="red").pack(pady=30)

    # ══════════════════════════════════════════════════════════════════
    #  TAB 8: STATISTICS
    # ══════════════════════════════════════════════════════════════════
    def _render_stats(self):
        sel = self._get_selected_groups()
        cols = ("Group A", "Group B", "Metric", "Value", "Sig")
        tree = ttk.Treeview(self.t_stats, columns=cols, show="headings")
        for c, w in zip(cols, [200, 200, 160, 130, 90]):
            tree.heading(c, text=c); tree.column(c, width=w)
        sb = ttk.Scrollbar(self.t_stats, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); sb.pack(side=tk.RIGHT, fill=tk.Y)

        for g in sel:
            v = self.data_map.get(g, pd.Series(dtype=float)).dropna().astype(float)
            if v.empty: continue
            for m, fn in [("N", lambda x: str(len(x))), ("Mean", lambda x: f"{x.mean():.4f}"),
                          ("Median", lambda x: f"{x.median():.4f}"),
                          ("Std", lambda x: f"{x.std():.4f}"),
                          ("IQR", lambda x: f"{x.quantile(.75)-x.quantile(.25):.4f}")]:
                tree.insert("", tk.END, values=(_tr(g, 25), "-", m, fn(v), ""))

        if len(sel) >= 2:
            tree.insert("", tk.END, values=("─" * 16, "─" * 16, "PAIRWISE", "─" * 10, ""))
            for k1, k2 in itertools.combinations(sel, 2):
                d1 = self.data_map.get(k1, pd.Series(dtype=float)).dropna()
                d2 = self.data_map.get(k2, pd.Series(dtype=float)).dropna()
                if d1.empty or d2.empty: continue
                try:
                    s, p = ranksums(d1, d2)
                    sig = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"
                    tree.insert("", tk.END, values=(_tr(k1,20), _tr(k2,20), "Wilcoxon Z", f"{s:.4f}", sig))
                    tree.insert("", tk.END, values=(_tr(k1,20), _tr(k2,20), "p-value", f"{p:.2e}", sig))
                except: pass
                try:
                    wd = wasserstein_distance(d1, d2)
                    tree.insert("", tk.END, values=(_tr(k1,20), _tr(k2,20), "Wasserstein", f"{wd:.4f}",
                        "High" if wd > 1 else "Mod" if wd > .5 else "Low"))
                except: pass

    # ══════════════════════════════════════════════════════════════════
    #  TAB 9: DATA TABLE
    # ══════════════════════════════════════════════════════════════════
    def _render_table(self):
        if self.metadata_df.empty:
            ttk.Label(self.t_table, text="No metadata loaded.",
                      foreground="gray").pack(pady=30)
            return
        df = self.metadata_df
        # Build priority columns: standard ones + ALL Classified_* columns
        cls_cols = sorted([c for c in df.columns if c.startswith('Classified_')])
        pri = ['Group', 'GSM', 'Expression', 'series_id', 'title',
               'source_name_ch1'] + cls_cols + ['Platform', 'Gene', 'Species']
        cols = [c for c in pri if c in df.columns]
        cols += [c for c in df.columns if c not in cols]
        cols = cols[:25]

        tree = ttk.Treeview(self.t_table, columns=cols, show="headings", selectmode="extended")
        vsb = ttk.Scrollbar(self.t_table, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(self.t_table, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        for c in cols:
            tree.heading(c, text=c.replace('_', ' ')); tree.column(c, width=120, stretch=False)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        max_rows = 2000
        for _, row in df.head(max_rows).iterrows():
            tree.insert("", tk.END, values=[str(row.get(c, ''))[:80] for c in cols])
        if len(df) > max_rows:
            ttk.Label(self.t_table,
                      text=f"Showing {max_rows:,} / {len(df):,} rows.",
                      font=("Segoe UI", 8, "italic"), foreground="gray").pack(pady=2)

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
                for ac in all_cls_cols:
                    if ac in grp_meta.columns:
                        vals = grp_meta[ac].values
                        sub[ac] = vals[:len(sub)] if len(vals) >= len(sub) else \
                                  list(vals) + ['N/A'] * (len(sub) - len(vals))
            frames.append(sub)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _extract_features(self, df):
        feat_cols = ['Expression']
        X_parts = [df[['Expression']].fillna(0).values]
        # Use ALL Classified_* columns present in the df
        cls_cols = [c for c in df.columns if c.startswith('Classified_')]
        for ac in cls_cols:
            if ac in df.columns:
                dum = pd.get_dummies(df[ac].fillna('N/A').astype(str), prefix=ac[:10])
                X_parts.append(dum.values); feat_cols.extend(dum.columns.tolist())
        X = np.column_stack(X_parts) if len(X_parts) > 1 else X_parts[0]
        if X.shape[1] < 2:
            X = np.column_stack([X, np.random.randn(X.shape[0]) * 0.001])
        return X, feat_cols

    def _best_ai_col(self, df):
        # First try standard AI_COLS, then any Classified_* column
        for ac in AI_COLS:
            if ac in df.columns and df[ac].notna().sum() > 0: return ac
        for c in df.columns:
            if c.startswith('Classified_') and df[c].notna().sum() > 0: return c
        return None

    def _scatter(self, ax, coords, labels, title, xlabel, ylabel):
        uniq = list(pd.Series(labels).value_counts().head(_MAX_GRP).index)
        clrs = _clrs(len(uniq))
        for i, u in enumerate(uniq):
            mask = np.array(labels) == u
            ax.scatter(coords[mask, 0], coords[mask, 1], c=clrs[i],
                       label=f"{_tr(u)} ({mask.sum()})", alpha=0.7, s=25,
                       edgecolors='black', lw=0.3)
        other = ~np.isin(labels, uniq)
        if other.any():
            ax.scatter(coords[other, 0], coords[other, 1], c='#CCCCCC',
                       label=f"Other ({other.sum()})", alpha=0.3, s=15)
        ax.set_title(title, fontsize=10, weight='bold')
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.legend(fontsize=6, ncol=max(1, len(uniq) // 6), framealpha=0.9)

    def _export(self):
        d = filedialog.askdirectory(title="Export Folder", parent=self)
        if not d: return
        out = Path(d); out.mkdir(parents=True, exist_ok=True)
        for key, fig in self.figs.items():
            fig.savefig(out / f"compare_{key}.png", dpi=150, bbox_inches='tight')
        if not self.metadata_df.empty:
            self.metadata_df.to_csv(out / "compare_data.csv", index=False)
        messagebox.showinfo("Exported", f"Saved to {out}", parent=self)


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

            sub['Group'] = label
            sub['Platform'] = r.get('platform', 'Unknown')
            sub['Gene'] = r.get('gene', col)
            frames.append(sub)

        metadata_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

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
