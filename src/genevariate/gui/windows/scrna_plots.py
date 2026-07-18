"""
GeneVariate — Cell-level single-cell plots.

These are the plots that pseudo-bulking would destroy, so they are
computed directly from the cell-level AnnData returned by the CELLxGENE
Census (or any other scRNA source):

* Composition   — stacked bar of cell-type proportions per donor / tissue
* UMAP          — 2-D embedding coloured by cell-type / tissue / gene
* Dot plot      — gene-by-group mean-expression and fraction-expressing
* QC            — n_genes, total_counts, pct_mito per cell

Every value plotted is a real measurement from a public submission — no
simulated data. Pseudo-bulk is NOT used here; that is the aggregated
platform path in the main GeneVariate windows.
"""

from __future__ import annotations

import threading
import traceback
from typing import Any, List, Optional, Sequence

import tkinter as tk
from tkinter import ttk, messagebox


# ────────────────────────────────────────────────────────────────────────────
# Lazy imports (kept inside functions for fast app startup without scanpy)
# ────────────────────────────────────────────────────────────────────────────
def _require_mpl():
    import matplotlib
    matplotlib.use("TkAgg", force=False)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg, NavigationToolbar2Tk,
    )
    return Figure, FigureCanvasTkAgg, NavigationToolbar2Tk


def _require_numpy_pandas():
    import numpy as np
    import pandas as pd
    return np, pd


# ────────────────────────────────────────────────────────────────────────────
# Window
# ────────────────────────────────────────────────────────────────────────────
class ScrnaPlotsWindow(tk.Toplevel):
    """Four cell-level plot tabs over an in-memory AnnData."""

    def __init__(self, parent, adata):
        super().__init__(parent)
        self.title("Single-cell plots — cell-level AnnData")
        self.geometry("1180x760")
        try:
            self.transient(parent)
        except Exception:
            pass

        self.adata = adata
        self._umap_coords = None  # cached numpy array of shape (n_obs, 2)

        # Header banner
        banner = ttk.Frame(self, padding=(10, 6))
        banner.pack(fill=tk.X)
        nobs = getattr(adata, "n_obs", "?")
        nvars = getattr(adata, "n_vars", "?")
        ttk.Label(
            banner,
            text=(f"AnnData: {nobs:,} cells × {nvars:,} genes. "
                  "All values are real scRNA-seq measurements; no simulated data."),
            font=("Segoe UI", 9, "italic"),
            foreground="#0A5B9A",
            wraplength=1150, justify=tk.LEFT,
        ).pack(fill=tk.X)

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self._build_composition_tab()
        self._build_umap_tab()
        self._build_dotplot_tab()
        self._build_qc_tab()

        ttk.Button(self, text="Close", command=self.destroy
                   ).pack(side=tk.RIGHT, padx=8, pady=(0, 8))

    # ────────────────────────────────────────────────────────────────
    # Shared helpers
    # ────────────────────────────────────────────────────────────────
    def _obs_columns(self) -> List[str]:
        try:
            return list(self.adata.obs.columns.astype(str))
        except Exception:
            return []

    def _candidate_label_cols(self) -> List[str]:
        """obs columns that are plausibly categorical labels."""
        prefer = ["cell_type", "tissue", "tissue_general", "disease",
                  "assay", "donor_id", "sex", "development_stage",
                  "self_reported_ethnicity", "dataset_id",
                  "suspension_type", "is_primary_data"]
        cols = self._obs_columns()
        # keep preferred ordering first, then append whatever else is categorical
        out = [c for c in prefer if c in cols]
        for c in cols:
            if c in out:
                continue
            try:
                s = self.adata.obs[c]
                if s.dtype == "object" or str(s.dtype).startswith("category"):
                    out.append(c)
            except Exception:
                continue
        return out

    def _gene_index(self):
        from genevariate.utils.anndata_io import _coerce_gene_index
        return _coerce_gene_index(self.adata.var)

    def _new_figure_panel(self, parent):
        """Return (fig, ax, canvas_widget) embedded into parent."""
        Figure, FigureCanvasTkAgg, NavigationToolbar2Tk = _require_mpl()
        fig = Figure(figsize=(8, 5.2), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        w = canvas.get_tk_widget()
        toolbar = NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        w.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        return fig, canvas

    def _error(self, title: str, exc: Exception):
        tb = traceback.format_exc()
        messagebox.showerror(title, tb, parent=self)

    # ────────────────────────────────────────────────────────────────
    # 1. Composition — stacked bar of cell_type proportions per group
    # ────────────────────────────────────────────────────────────────
    def _build_composition_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Composition")

        top = ttk.Frame(tab, padding=6)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Group by (x-axis):").pack(side=tk.LEFT)
        labels = self._candidate_label_cols()
        default_group = "donor_id" if "donor_id" in labels else (labels[0] if labels else "")
        self.comp_group_var = tk.StringVar(value=default_group)
        ttk.Combobox(top, textvariable=self.comp_group_var, values=labels,
                     width=22, state="readonly"
                     ).pack(side=tk.LEFT, padx=4)

        ttk.Label(top, text="Stack by (colour):").pack(side=tk.LEFT, padx=(10, 2))
        default_stack = "cell_type" if "cell_type" in labels else (
            labels[1] if len(labels) > 1 else "")
        self.comp_stack_var = tk.StringVar(value=default_stack)
        ttk.Combobox(top, textvariable=self.comp_stack_var, values=labels,
                     width=22, state="readonly"
                     ).pack(side=tk.LEFT, padx=4)

        ttk.Label(top, text="Max stack cats:").pack(side=tk.LEFT, padx=(10, 2))
        self.comp_topn_var = tk.StringVar(value="12")
        ttk.Entry(top, textvariable=self.comp_topn_var, width=5
                  ).pack(side=tk.LEFT)

        ttk.Label(top, text="Max x groups:").pack(side=tk.LEFT, padx=(8, 2))
        self.comp_xgroups_var = tk.StringVar(value="40")
        ttk.Entry(top, textvariable=self.comp_xgroups_var, width=5
                  ).pack(side=tk.LEFT)

        self.comp_proportion_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Proportions (else counts)",
                        variable=self.comp_proportion_var
                        ).pack(side=tk.LEFT, padx=(12, 2))

        ttk.Button(top, text="Plot", command=self._draw_composition
                   ).pack(side=tk.RIGHT, padx=4)

        body = ttk.Frame(tab)
        body.pack(fill=tk.BOTH, expand=True)
        self.comp_fig, self.comp_canvas = self._new_figure_panel(body)

    def _draw_composition(self):
        try:
            np, pd = _require_numpy_pandas()
            group = self.comp_group_var.get().strip()
            stack = self.comp_stack_var.get().strip()
            try:
                topn = max(1, int(self.comp_topn_var.get()))
            except Exception:
                topn = 12
            if not group or not stack:
                messagebox.showinfo("Composition",
                                     "Pick a group-by and a stack-by column.",
                                     parent=self)
                return
            if group not in self.adata.obs.columns or stack not in self.adata.obs.columns:
                messagebox.showerror("Composition",
                                      f"{group!r} or {stack!r} not in obs.",
                                      parent=self)
                return

            obs = self.adata.obs[[group, stack]].astype(str)
            ct = (obs.groupby([group, stack]).size()
                     .unstack(fill_value=0))

            # Keep the top-N stack categories (sum across groups); rest → "other"
            totals = ct.sum(axis=0).sort_values(ascending=False)
            keep = totals.head(topn).index.tolist()
            if len(totals) > topn:
                other = ct.drop(columns=keep).sum(axis=1)
                ct = ct[keep]
                ct["other"] = other

            # Order x-axis by total cell count, then cap to max x groups
            try:
                x_cap = max(1, int(self.comp_xgroups_var.get()))
            except Exception:
                x_cap = 40
            ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]
            dropped_x = 0
            if len(ct) > x_cap:
                dropped_x = len(ct) - x_cap
                ct = ct.iloc[:x_cap]

            counts = ct.copy()  # keep raw counts for hover
            if self.comp_proportion_var.get():
                ct = ct.div(ct.sum(axis=1).replace(0, 1), axis=0)
                ylabel = "Proportion"
            else:
                ylabel = "Cell count"

            self.comp_fig.clear()
            ax = self.comp_fig.add_subplot(111)
            bottoms = np.zeros(len(ct), dtype=float)
            x = np.arange(len(ct))
            import matplotlib.cm as cm
            base_cmap = cm.get_cmap("tab20")
            bars_per_col = {}
            for i, col in enumerate(ct.columns):
                vals = ct[col].to_numpy()
                bars = ax.bar(x, vals, bottom=bottoms, label=str(col),
                               color=base_cmap(i % 20),
                               edgecolor="white", linewidth=0.3, picker=True)
                bars_per_col[str(col)] = (bars, counts[col].to_numpy(),
                                            vals, bottoms.copy())
                bottoms += vals
            ax.set_xticks(x)
            ax.set_xticklabels([str(s) for s in ct.index], rotation=45,
                               ha="right", fontsize=8)
            ax.set_ylabel(ylabel)
            extra = f" — top {x_cap} of {x_cap + dropped_x}" if dropped_x else ""
            ax.set_title(f"{stack} composition per {group}{extra}  "
                         f"(n_cells={int(self.adata.n_obs):,})")
            ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                      fontsize=8, frameon=False)

            # Hover annotation — show group / stack / count / fraction
            annot = ax.annotate("", xy=(0, 0), xytext=(12, 12),
                                 textcoords="offset points",
                                 bbox=dict(boxstyle="round,pad=0.3",
                                            fc="#FFFFCC", ec="#888", alpha=0.95),
                                 fontsize=8)
            annot.set_visible(False)
            x_labels = [str(s) for s in ct.index]
            row_totals = counts.sum(axis=1).to_numpy()

            def _on_move(event):
                if event.inaxes != ax:
                    if annot.get_visible():
                        annot.set_visible(False)
                        self.comp_canvas.draw_idle()
                    return
                for col, (bars, raw_cnt, disp_v, disp_bot) in bars_per_col.items():
                    cont, info = bars.contains(event)
                    if cont and info.get("ind", []).size > 0:
                        i = int(info["ind"][0])
                        frac = (raw_cnt[i] / row_totals[i]) if row_totals[i] else 0.0
                        annot.xy = (x[i], disp_bot[i] + disp_v[i] / 2)
                        annot.set_text(
                            f"{group} = {x_labels[i]}\n"
                            f"{stack} = {col}\n"
                            f"cells = {int(raw_cnt[i]):,}\n"
                            f"fraction = {frac:.2%}")
                        annot.set_visible(True)
                        self.comp_canvas.draw_idle()
                        return
                if annot.get_visible():
                    annot.set_visible(False)
                    self.comp_canvas.draw_idle()

            if getattr(self, "_comp_hover_cid", None) is not None:
                try:
                    self.comp_canvas.mpl_disconnect(self._comp_hover_cid)
                except Exception:
                    pass
            self._comp_hover_cid = self.comp_canvas.mpl_connect(
                "motion_notify_event", _on_move)

            self.comp_fig.tight_layout()
            self.comp_canvas.draw_idle()
        except Exception as exc:
            self._error("Composition plot failed", exc)

    # ────────────────────────────────────────────────────────────────
    # 2. UMAP — 2-D embedding (uses obsm['X_umap'] if present, else
    #    PCA + umap-learn fallback, else plain PCA as a stand-in.)
    # ────────────────────────────────────────────────────────────────
    def _build_umap_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="UMAP / PCA")

        top = ttk.Frame(tab, padding=6)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Colour by:").pack(side=tk.LEFT)
        # All obs columns are offered (not just the curated candidates) and the
        # combo stays writable so users can paste any column name.
        labels_all = self._obs_columns()
        default = "cell_type" if "cell_type" in labels_all else (
            labels_all[0] if labels_all else "")
        self.umap_color_var = tk.StringVar(value=default)
        self.umap_color_combo = ttk.Combobox(
            top, textvariable=self.umap_color_var,
            values=labels_all, width=22)  # not readonly — free-text allowed
        self.umap_color_combo.pack(side=tk.LEFT, padx=4)

        ttk.Label(top, text="or gene:").pack(side=tk.LEFT, padx=(10, 2))
        self.umap_gene_var = tk.StringVar(value="")
        ttk.Entry(top, textvariable=self.umap_gene_var, width=14
                  ).pack(side=tk.LEFT, padx=4)

        ttk.Label(top, text="Max cells:").pack(side=tk.LEFT, padx=(10, 2))
        self.umap_maxcells_var = tk.StringVar(value="20000")
        ttk.Entry(top, textvariable=self.umap_maxcells_var, width=8
                  ).pack(side=tk.LEFT)

        ttk.Button(top, text="Compute & plot",
                   command=self._draw_umap_async).pack(side=tk.RIGHT, padx=4)

        self.umap_status_var = tk.StringVar(
            value="Tip: if the AnnData has no obsm['X_umap'], a PCA-only "
                  "fallback is computed (umap-learn will be used if installed).")
        ttk.Label(tab, textvariable=self.umap_status_var,
                  font=("Segoe UI", 9, "italic"),
                  foreground="#0A5B9A").pack(fill=tk.X, padx=8)

        body = ttk.Frame(tab)
        body.pack(fill=tk.BOTH, expand=True)
        self.umap_fig, self.umap_canvas = self._new_figure_panel(body)

    def _draw_umap_async(self):
        self.umap_status_var.set("Computing embedding…")
        # Invalidate cache so max_cells changes take effect
        self._umap_coords = None

        def _worker():
            try:
                coords, method = self._get_umap_coords()
                self.after(0, lambda: self._draw_umap_plot(coords, method))
            except Exception as exc:
                tb = traceback.format_exc()
                self.after(0, lambda: (self.umap_status_var.set("Failed."),
                                         messagebox.showerror("UMAP failed", tb,
                                                               parent=self)))

        threading.Thread(target=_worker, daemon=True).start()

    def _get_umap_coords(self):
        """Return (coords[n×2], method_str). Cached across calls."""
        if self._umap_coords is not None:
            return self._umap_coords
        np, pd = _require_numpy_pandas()
        adata = self.adata

        try:
            maxc = max(500, int(self.umap_maxcells_var.get()))
        except Exception:
            maxc = 20_000

        # Subsample for speed
        if adata.n_obs > maxc:
            rs = np.random.default_rng(0)
            idx = np.sort(rs.choice(adata.n_obs, size=maxc, replace=False))
        else:
            idx = np.arange(adata.n_obs)
        self._umap_subset_idx = idx

        # 1. Use pre-computed UMAP if AnnData already has one
        if hasattr(adata, "obsm") and "X_umap" in adata.obsm:
            coords = np.asarray(adata.obsm["X_umap"])[idx]
            self._umap_coords = (coords, "X_umap (from AnnData)")
            return self._umap_coords

        # 2. Otherwise compute a PCA on the subset
        X = adata.X[idx]
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        # log1p if data looks like counts (max very large, all non-negative)
        if X.min() >= 0.0 and X.max() > 50:
            X = np.log1p(X)

        # Center + PCA via truncated SVD for memory
        from sklearn.decomposition import TruncatedSVD
        k = min(30, min(X.shape) - 1)
        if k <= 2:
            # Degenerate — just return raw first two columns
            coords = X[:, :2]
            method = "raw (too few features for PCA)"
        else:
            svd = TruncatedSVD(n_components=k, random_state=0)
            pcs = svd.fit_transform(X)
            # 3. UMAP if available, else plain PCA (first 2 PCs)
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=0,
                                     n_neighbors=min(15, pcs.shape[0] - 1))
                coords = reducer.fit_transform(pcs)
                method = f"UMAP on {k} PCs (subset={len(idx):,})"
            except Exception:
                coords = pcs[:, :2]
                method = f"PCA[1,2] (subset={len(idx):,} — install umap-learn for UMAP)"

        self._umap_coords = (coords, method)
        return self._umap_coords

    def _draw_umap_plot(self, coords, method):
        np, pd = _require_numpy_pandas()
        idx = self._umap_subset_idx
        gene_q = self.umap_gene_var.get().strip()

        self.umap_fig.clear()
        ax = self.umap_fig.add_subplot(111)

        hover_text: list = [""] * len(coords)

        if gene_q:
            # Colour by gene expression
            var_idx = self._gene_index()
            var_list = [str(v) for v in var_idx]
            if gene_q not in var_list:
                self.umap_status_var.set(f"Gene {gene_q!r} not in var.")
                return
            j = var_list.index(gene_q)
            col = self.adata.X[idx, j]
            if hasattr(col, "toarray"):
                col = col.toarray().ravel()
            col = np.asarray(col).ravel().astype(float)
            scatter = ax.scatter(coords[:, 0], coords[:, 1], c=col,
                                  cmap="viridis", s=8, alpha=0.75, linewidths=0,
                                  picker=True)
            self.umap_fig.colorbar(scatter, ax=ax, label=f"{gene_q} (raw X)")
            title = f"{method} — coloured by {gene_q}"
            # Hover includes gene value + any cell_type/tissue/donor we find
            extra_cols = [c for c in ("cell_type", "tissue", "donor_id")
                          if c in self.adata.obs.columns]
            extras = {c: self.adata.obs[c].astype(str).to_numpy()[idx]
                      for c in extra_cols}
            for i in range(len(coords)):
                parts = [f"{gene_q}={col[i]:.3g}"]
                for c in extra_cols:
                    parts.append(f"{c}={extras[c][i]}")
                hover_text[i] = "\n".join(parts)
        else:
            color = self.umap_color_var.get().strip()
            if color and color in self.adata.obs.columns:
                vals = self.adata.obs[color].astype(str).to_numpy()[idx]
                categories = list(pd.unique(vals))
                import matplotlib.cm as cm
                base_cmap = cm.get_cmap("tab20")
                lookup = {c: i for i, c in enumerate(categories)}
                colors = [base_cmap(lookup[v] % 20) for v in vals]
                scatter = ax.scatter(coords[:, 0], coords[:, 1], c=colors,
                                      s=8, alpha=0.75, linewidths=0,
                                      picker=True)
                # Compact legend — top 20 categories by count
                vc = pd.Series(vals).value_counts().head(20)
                from matplotlib.patches import Patch
                handles = [Patch(facecolor=base_cmap(lookup[name] % 20),
                                  label=f"{name} ({n})")
                           for name, n in vc.items()]
                ax.legend(handles=handles, loc="center left",
                           bbox_to_anchor=(1.01, 0.5), fontsize=8,
                           frameon=False)
                title = f"{method} — coloured by {color}"
                for i in range(len(coords)):
                    hover_text[i] = f"{color}={vals[i]}"
            else:
                scatter = ax.scatter(coords[:, 0], coords[:, 1],
                                      s=8, alpha=0.75, linewidths=0,
                                      picker=True)
                title = method

        ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
        ax.set_title(title, fontsize=10)

        # Interactive hover annotation
        annot = ax.annotate("", xy=(0, 0), xytext=(12, 12),
                             textcoords="offset points",
                             bbox=dict(boxstyle="round,pad=0.3",
                                        fc="#FFFFCC", ec="#888", alpha=0.95),
                             fontsize=8)
        annot.set_visible(False)
        self._umap_hover_text = hover_text
        self._umap_scatter = scatter
        self._umap_annot = annot

        def _on_move(event):
            if event.inaxes != ax or scatter is None:
                if annot.get_visible():
                    annot.set_visible(False)
                    self.umap_canvas.draw_idle()
                return
            cont, info = scatter.contains(event)
            if cont and info.get("ind", []).size > 0:
                i = int(info["ind"][0])
                xy = scatter.get_offsets()[i]
                annot.xy = (xy[0], xy[1])
                annot.set_text(hover_text[i] or f"cell {i}")
                annot.set_visible(True)
                self.umap_canvas.draw_idle()
            elif annot.get_visible():
                annot.set_visible(False)
                self.umap_canvas.draw_idle()

        # Disconnect any previous handler before connecting a new one
        if getattr(self, "_umap_hover_cid", None) is not None:
            try:
                self.umap_canvas.mpl_disconnect(self._umap_hover_cid)
            except Exception:
                pass
        self._umap_hover_cid = self.umap_canvas.mpl_connect(
            "motion_notify_event", _on_move)

        self.umap_fig.tight_layout()
        self.umap_canvas.draw_idle()
        self.umap_status_var.set(method + "  — hover a point for details.")

    # ────────────────────────────────────────────────────────────────
    # 3. Dot plot — gene × group (mean expression + fraction expressing)
    # ────────────────────────────────────────────────────────────────
    def _build_dotplot_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Dot plot")

        top = ttk.Frame(tab, padding=6)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Genes (comma-separated):").pack(side=tk.LEFT)
        self.dot_genes_var = tk.StringVar(value="")
        ttk.Entry(top, textvariable=self.dot_genes_var, width=40
                  ).pack(side=tk.LEFT, padx=4)

        ttk.Label(top, text="Group by:").pack(side=tk.LEFT, padx=(10, 2))
        labels = self._candidate_label_cols()
        default = "cell_type" if "cell_type" in labels else (labels[0] if labels else "")
        self.dot_group_var = tk.StringVar(value=default)
        ttk.Combobox(top, textvariable=self.dot_group_var, values=labels,
                     width=22, state="readonly"
                     ).pack(side=tk.LEFT, padx=4)

        ttk.Label(top, text="Max groups:").pack(side=tk.LEFT, padx=(10, 2))
        self.dot_maxgroups_var = tk.StringVar(value="40")
        ttk.Entry(top, textvariable=self.dot_maxgroups_var, width=5
                  ).pack(side=tk.LEFT)

        ttk.Button(top, text="Plot", command=self._draw_dotplot
                   ).pack(side=tk.RIGHT, padx=4)

        hint = ttk.Label(tab,
            text="Circle size = fraction of cells with X>0 for that gene in that group.  "
                 "Colour = mean expression across cells in the group.",
            font=("Segoe UI", 9, "italic"), foreground="#0A5B9A")
        hint.pack(fill=tk.X, padx=8)

        body = ttk.Frame(tab)
        body.pack(fill=tk.BOTH, expand=True)
        self.dot_fig, self.dot_canvas = self._new_figure_panel(body)

    def _draw_dotplot(self):
        try:
            np, pd = _require_numpy_pandas()
            raw = self.dot_genes_var.get().strip()
            if not raw:
                messagebox.showinfo("Dot plot",
                                     "Enter one or more gene symbols.",
                                     parent=self)
                return
            wanted = [g.strip() for g in raw.replace(",", " ").split() if g.strip()]
            group = self.dot_group_var.get().strip()
            if group not in self.adata.obs.columns:
                messagebox.showerror("Dot plot",
                                      f"{group!r} is not an obs column.",
                                      parent=self)
                return

            var_idx = self._gene_index()
            var_list = [str(v) for v in var_idx]
            pairs = [(g, var_list.index(g)) for g in wanted if g in var_list]
            missing = [g for g in wanted if g not in var_list]
            if not pairs:
                messagebox.showinfo("Dot plot",
                                     f"None of these genes are in the fetched "
                                     f"data: {wanted}",
                                     parent=self)
                return

            cols = [j for _, j in pairs]
            X_sub = self.adata.X[:, cols]
            if hasattr(X_sub, "toarray"):
                X_sub = X_sub.toarray()
            X_sub = np.asarray(X_sub, dtype=np.float32)

            groups = self.adata.obs[group].astype(str).to_numpy()
            # Order groups by size (largest first) and cap
            try:
                max_groups = max(1, int(self.dot_maxgroups_var.get()))
            except Exception:
                max_groups = 40
            vc_all = pd.Series(groups).value_counts()
            cats = list(vc_all.index)
            dropped_g = 0
            if len(cats) > max_groups:
                dropped_g = len(cats) - max_groups
                cats = cats[:max_groups]
            n_g = len(cats)
            n_k = len(pairs)
            mean_mat = np.zeros((n_g, n_k))
            frac_mat = np.zeros((n_g, n_k))
            for gi, c in enumerate(cats):
                m = groups == c
                if not m.any():
                    continue
                block = X_sub[m]
                mean_mat[gi] = block.mean(axis=0)
                frac_mat[gi] = (block > 0).mean(axis=0)

            self.dot_fig.clear()
            ax = self.dot_fig.add_subplot(111)
            xs, ys, sizes, colors, tips = [], [], [], [], []
            gene_names = [g for g, _ in pairs]
            for gi in range(n_g):
                for ki in range(n_k):
                    xs.append(ki)
                    ys.append(gi)
                    sizes.append(20 + 260 * float(frac_mat[gi, ki]))
                    colors.append(float(mean_mat[gi, ki]))
                    tips.append(
                        f"{gene_names[ki]} × {cats[gi]}\n"
                        f"mean = {mean_mat[gi, ki]:.3g}\n"
                        f"frac = {100 * frac_mat[gi, ki]:.1f}%")
            sc = ax.scatter(xs, ys, s=sizes, c=colors, cmap="viridis",
                             edgecolors="black", linewidths=0.3, picker=True)
            ax.set_xticks(range(n_k))
            ax.set_xticklabels(gene_names, rotation=40,
                                ha="right", fontsize=9)
            ax.set_yticks(range(n_g))
            ax.set_yticklabels(list(cats), fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("Gene")
            ax.set_ylabel(group)
            title = f"Dot plot — {n_k} gene(s) × {n_g} group(s)"
            if dropped_g:
                title += f" (top {n_g} of {n_g + dropped_g})"
            if missing:
                title += f"   [not found: {', '.join(missing)}]"
            ax.set_title(title, fontsize=10)
            self.dot_fig.colorbar(sc, ax=ax, label="Mean expression",
                                    fraction=0.03, pad=0.01)

            # Hover annotation (mean, frac, labels)
            annot = ax.annotate("", xy=(0, 0), xytext=(12, 12),
                                 textcoords="offset points",
                                 bbox=dict(boxstyle="round,pad=0.3",
                                            fc="#FFFFCC", ec="#888", alpha=0.95),
                                 fontsize=8)
            annot.set_visible(False)

            def _on_move(event):
                if event.inaxes != ax:
                    if annot.get_visible():
                        annot.set_visible(False)
                        self.dot_canvas.draw_idle()
                    return
                cont, info = sc.contains(event)
                if cont and info.get("ind", []).size > 0:
                    i = int(info["ind"][0])
                    annot.xy = (xs[i], ys[i])
                    annot.set_text(tips[i])
                    annot.set_visible(True)
                    self.dot_canvas.draw_idle()
                elif annot.get_visible():
                    annot.set_visible(False)
                    self.dot_canvas.draw_idle()

            if getattr(self, "_dot_hover_cid", None) is not None:
                try:
                    self.dot_canvas.mpl_disconnect(self._dot_hover_cid)
                except Exception:
                    pass
            self._dot_hover_cid = self.dot_canvas.mpl_connect(
                "motion_notify_event", _on_move)

            self.dot_fig.tight_layout()
            self.dot_canvas.draw_idle()
        except Exception as exc:
            self._error("Dot plot failed", exc)

    # ────────────────────────────────────────────────────────────────
    # 4. QC — n_genes, total_counts, pct_mito per cell
    # ────────────────────────────────────────────────────────────────
    def _build_qc_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="QC")

        top = ttk.Frame(tab, padding=6)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Group by (optional):").pack(side=tk.LEFT)
        labels = [""] + self._candidate_label_cols()
        self.qc_group_var = tk.StringVar(value="")
        ttk.Combobox(top, textvariable=self.qc_group_var, values=labels,
                     width=22, state="readonly"
                     ).pack(side=tk.LEFT, padx=4)

        ttk.Label(top, text="Mito prefix:").pack(side=tk.LEFT, padx=(10, 2))
        self.qc_mito_var = tk.StringVar(value="MT-")
        ttk.Entry(top, textvariable=self.qc_mito_var, width=8
                  ).pack(side=tk.LEFT)

        ttk.Button(top, text="Plot", command=self._draw_qc
                   ).pack(side=tk.RIGHT, padx=4)

        hint = ttk.Label(tab,
            text="n_genes = genes with X>0 per cell. "
                 "total_counts = sum of X per cell. "
                 "pct_mito = 100 × (sum over genes starting with prefix) / total_counts.",
            font=("Segoe UI", 9, "italic"), foreground="#0A5B9A")
        hint.pack(fill=tk.X, padx=8)

        body = ttk.Frame(tab)
        body.pack(fill=tk.BOTH, expand=True)
        self.qc_fig, self.qc_canvas = self._new_figure_panel(body)

    def _draw_qc(self):
        try:
            np, pd = _require_numpy_pandas()
            X = self.adata.X
            if hasattr(X, "toarray"):
                # sparse path — avoid densifying the whole matrix
                total = np.asarray(X.sum(axis=1)).ravel().astype(float)
                n_genes = np.asarray((X > 0).sum(axis=1)).ravel().astype(float)
            else:
                Xa = np.asarray(X, dtype=float)
                total = Xa.sum(axis=1)
                n_genes = (Xa > 0).sum(axis=1).astype(float)

            # Mito fraction
            prefix = self.qc_mito_var.get().strip()
            pct_mito = np.zeros_like(total)
            if prefix:
                var_idx = self._gene_index()
                var_list = [str(v) for v in var_idx]
                mito_cols = [i for i, g in enumerate(var_list)
                             if g.upper().startswith(prefix.upper())]
                if mito_cols:
                    if hasattr(self.adata.X, "toarray"):
                        sub = self.adata.X[:, mito_cols]
                        mito_total = np.asarray(sub.sum(axis=1)).ravel()
                    else:
                        mito_total = np.asarray(self.adata.X[:, mito_cols]).sum(axis=1)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        pct_mito = 100.0 * mito_total / np.where(total > 0, total, 1.0)

            group = self.qc_group_var.get().strip()

            self.qc_fig.clear()
            axes = self.qc_fig.subplots(1, 3)

            # Per-axis metadata for hover
            axis_info = {}  # id(ax) -> dict(mode, ...)

            def _violin_or_hist(ax, values, ylabel):
                if group and group in self.adata.obs.columns:
                    cats = pd.unique(self.adata.obs[group].astype(str))
                    data, labels_kept = [], []
                    for c in cats:
                        m = (self.adata.obs[group].astype(str).to_numpy() == c)
                        if m.any():
                            data.append(values[m])
                            labels_kept.append(str(c)[:20])
                    if data:
                        ax.violinplot(data, showmedians=True)
                        ax.set_xticks(range(1, len(labels_kept) + 1))
                        ax.set_xticklabels(labels_kept, rotation=40,
                                            ha="right", fontsize=7)
                        # Pre-compute summary stats per violin for hover
                        stats = []
                        for arr in data:
                            arr = np.asarray(arr, dtype=float)
                            stats.append({
                                "n": int(arr.size),
                                "median": float(np.median(arr)) if arr.size else 0.0,
                                "mean":   float(arr.mean()) if arr.size else 0.0,
                                "q25":    float(np.percentile(arr, 25)) if arr.size else 0.0,
                                "q75":    float(np.percentile(arr, 75)) if arr.size else 0.0,
                                "min":    float(arr.min()) if arr.size else 0.0,
                                "max":    float(arr.max()) if arr.size else 0.0,
                            })
                        axis_info[id(ax)] = dict(mode="violin", labels=labels_kept,
                                                   stats=stats, ylabel=ylabel)
                else:
                    counts, bin_edges, _ = ax.hist(values, bins=60,
                                                     color="#0A5B9A",
                                                     edgecolor="white",
                                                     linewidth=0.3)
                    axis_info[id(ax)] = dict(mode="hist",
                                               counts=np.asarray(counts),
                                               edges=np.asarray(bin_edges),
                                               ylabel=ylabel)
                ax.set_ylabel(ylabel)

            _violin_or_hist(axes[0], n_genes, "n_genes per cell")
            _violin_or_hist(axes[1], total, "total_counts per cell")
            _violin_or_hist(axes[2], pct_mito, f"% mito ({prefix or 'disabled'})")

            self.qc_fig.suptitle(
                f"QC — {int(self.adata.n_obs):,} cells"
                + (f" grouped by {group}" if group else ""),
                fontsize=10)

            # Hover: histogram → bin range + count; violin → per-group stats
            annot = self.qc_fig.text(
                0.01, 0.97, "", fontsize=8, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="#FFFFCC",
                           ec="#888", alpha=0.95))
            annot.set_visible(False)

            def _on_move(event):
                if event.inaxes is None or event.xdata is None:
                    if annot.get_visible():
                        annot.set_visible(False)
                        self.qc_canvas.draw_idle()
                    return
                info = axis_info.get(id(event.inaxes))
                if info is None:
                    return
                if info["mode"] == "hist":
                    edges = info["edges"]; counts = info["counts"]
                    x = float(event.xdata)
                    if x < edges[0] or x > edges[-1]:
                        if annot.get_visible():
                            annot.set_visible(False)
                            self.qc_canvas.draw_idle()
                        return
                    b = int(np.searchsorted(edges, x, side="right") - 1)
                    b = max(0, min(b, len(counts) - 1))
                    annot.set_text(
                        f"{info['ylabel']}\n"
                        f"bin: [{edges[b]:.3g}, {edges[b+1]:.3g})\n"
                        f"cells: {int(counts[b]):,}")
                else:
                    # violin — map xdata → 1-based violin index
                    labels_kept = info["labels"]; stats = info["stats"]
                    if not labels_kept:
                        return
                    i = int(round(event.xdata)) - 1
                    if i < 0 or i >= len(labels_kept):
                        if annot.get_visible():
                            annot.set_visible(False)
                            self.qc_canvas.draw_idle()
                        return
                    s = stats[i]
                    annot.set_text(
                        f"{info['ylabel']}\n"
                        f"{labels_kept[i]}  (n={s['n']:,})\n"
                        f"median={s['median']:.3g}  mean={s['mean']:.3g}\n"
                        f"IQR=[{s['q25']:.3g}, {s['q75']:.3g}]\n"
                        f"range=[{s['min']:.3g}, {s['max']:.3g}]")
                annot.set_visible(True)
                self.qc_canvas.draw_idle()

            if getattr(self, "_qc_hover_cid", None) is not None:
                try:
                    self.qc_canvas.mpl_disconnect(self._qc_hover_cid)
                except Exception:
                    pass
            self._qc_hover_cid = self.qc_canvas.mpl_connect(
                "motion_notify_event", _on_move)

            self.qc_fig.tight_layout()
            self.qc_canvas.draw_idle()
        except Exception as exc:
            self._error("QC plot failed", exc)
