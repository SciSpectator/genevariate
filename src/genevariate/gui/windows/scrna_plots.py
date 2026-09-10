"""
GeneVariate - Cell-level single-cell plots.

These are the plots that pseudo-bulking would destroy, so they are
computed directly from the cell-level AnnData returned by the CELLxGENE
Census (or any other scRNA source):

* Composition   - stacked bar of cell-type proportions per donor / tissue
* UMAP          - 2-D embedding coloured by cell-type / tissue / gene
* Dot plot      - gene-by-group mean-expression and fraction-expressing
* QC            - n_genes, total_counts, pct_mito per cell

Every value plotted is a real measurement from a public submission - no
simulated data. Pseudo-bulk is NOT used here; that is the aggregated
platform path in the main GeneVariate windows.
"""

from __future__ import annotations

import threading
import traceback
from typing import Any, List, Optional, Sequence

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from genevariate.gui.theme import (AERO, UI_FONT, labelframe, style_toolbar, ensure_theme,
                                   style_window, wrap_to_parent)

try:
    from genevariate.utils.viz_style import (
        make_interactive as _make_interactive,
        palette_for as _palette_for,
    )
except Exception:  # pragma: no cover
    def _make_interactive(fig, **kw): return None

    def _palette_for(n, use_case="discrete"):
        import matplotlib.colors as _mc
        import seaborn as _sns
        return [_mc.to_hex(c) for c in _sns.color_palette("husl", max(1, n))]


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
        ensure_theme(self)
        style_window(self)
        self.title("Single-cell plots - cell-level AnnData")
        self.geometry("1180x760")
        try:
            self.transient(parent)
        except Exception:
            pass

        # _adata_full never changes; self.adata is the currently kept subset and
        # is what every tab reads, so unticking a study moves all four plots.
        self._adata_full = adata
        self.adata = adata
        self._umap_coords = None  # cached numpy array of shape (n_obs, 2)
        # What each tab computed, kept so an export carries the numbers behind
        # the picture and not only the picture.
        self._tables = {}
        self._drawn = set()

        # Header banner
        banner = ttk.Frame(self, padding=(10, 6))
        banner.pack(fill=tk.X)
        nobs = getattr(adata, "n_obs", "?")
        nvars = getattr(adata, "n_vars", "?")
        wrap_to_parent(ttk.Label(
            banner,
            text=(f"AnnData: {nobs:,} cells × {nvars:,} genes. "
                  "All values are real scRNA-seq measurements; no simulated data."),
            font=(UI_FONT, 9, "italic"),
            foreground=AERO["accent_dark"],
            justify=tk.LEFT,
        )).pack(fill=tk.X)

        outer = ttk.Frame(self)
        outer.pack(fill=tk.BOTH, expand=True)

        self._build_study_filter(outer)

        self.nb = ttk.Notebook(outer)
        self.nb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        self._build_composition_tab()
        self._build_umap_tab()
        self._build_dotplot_tab()
        self._build_qc_tab()
        self._figs = {"composition": self.comp_fig, "umap": self.umap_fig,
                      "dotplot": self.dot_fig, "qc": self.qc_fig}

        ttk.Button(self, text="Close", command=self.destroy
                   ).pack(side=tk.RIGHT, padx=8, pady=(0, 8))
        ttk.Button(self, text="Export All", command=self._export_all
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
        # These panels replot into the same figure as the user changes the
        # gene or grouping, so interactivity is attached once to the figure
        # rather than after each redraw.
        _make_interactive(fig)
        w = canvas.get_tk_widget()
        toolbar = NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
        toolbar.update()
        style_toolbar(toolbar)
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        w.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        return fig, canvas

    def _error(self, title: str, exc: Exception):
        tb = traceback.format_exc()
        messagebox.showerror(title, tb, parent=self)

    # ────────────────────────────────────────────────────────────────
    # Study filter - the cell-level twin of the bulk Filter Values panel
    # ────────────────────────────────────────────────────────────────
    def _study_columns(self) -> List[str]:
        """obs columns that plausibly say which study a cell came from."""
        prefer = ["dataset_id", "series_id", "gse", "study", "study_id",
                  "batch", "donor_id"]
        cols = self._obs_columns()
        out = [c for c in prefer if c in cols]
        for c in cols:
            if c in out:
                continue
            try:
                s = self._adata_full.obs[c]
                if s.dtype == "object" or str(s.dtype).startswith("category"):
                    if 1 < s.nunique() <= 500:
                        out.append(c)
            except Exception:
                continue
        return out

    def _build_study_filter(self, parent):
        cols = self._study_columns()
        self._study_checks = {}
        self._study_counts = {}
        self.study_col_var = tk.StringVar(value=cols[0] if cols else "")
        self.study_count_var = tk.StringVar(value="")
        if not cols:
            return

        panel = labelframe(parent, text=" Studies ", padding=6)
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 0), pady=6)

        combo = ttk.Combobox(panel, textvariable=self.study_col_var,
                             values=cols, width=20, state="readonly")
        combo.pack(fill=tk.X)
        combo.bind("<<ComboboxSelected>>",
                   lambda _e: self._populate_study_filter())

        holder = ttk.Frame(panel)
        holder.pack(fill=tk.BOTH, expand=True, pady=(6, 4))
        tree = ttk.Treeview(holder, columns=("check", "value", "n"),
                            show="headings", height=18, selectmode="none")
        tree.heading("check", text="\u2713")
        tree.heading("value", text="Study")
        tree.heading("n", text="cells")
        tree.column("check", width=34, anchor="center", stretch=False)
        tree.column("value", width=140, minwidth=90, anchor="center")
        tree.column("n", width=64, anchor="center", stretch=False)
        vs = ttk.Scrollbar(holder, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=vs.set)
        vs.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree.bind("<Button-1>", self._on_study_click)
        self._study_tree = tree

        row = ttk.Frame(panel)
        row.pack(fill=tk.X)
        ttk.Button(row, text="\u2611 All", width=8,
                   command=lambda: self._set_all_studies(True)
                   ).pack(side=tk.LEFT)
        ttk.Button(row, text="\u2610 None", width=8,
                   command=lambda: self._set_all_studies(False)
                   ).pack(side=tk.LEFT, padx=4)

        ttk.Label(panel, textvariable=self.study_count_var,
                  font=(UI_FONT, 9), foreground=AERO["accent_dark"]
                  ).pack(fill=tk.X, pady=(6, 0))

        self._populate_study_filter()

    def _populate_study_filter(self):
        col = self.study_col_var.get().strip()
        self._study_checks = {}
        self._study_counts = {}
        if col and col in self._adata_full.obs.columns:
            vc = self._adata_full.obs[col].astype(str).value_counts()
            self._study_counts = {str(v): int(n) for v, n in vc.items()}
            self._study_checks = {v: True for v in self._study_counts}
        self._render_study_tree()
        self._apply_study_filter()

    def _render_study_tree(self):
        tree = getattr(self, "_study_tree", None)
        if tree is None:
            return
        tree.delete(*tree.get_children())
        tree.tag_configure("on", foreground=AERO["text"])
        tree.tag_configure("off", foreground=AERO["muted"])
        for v, on in self._study_checks.items():
            tree.insert("", "end", iid=v,
                        values=("\u2611" if on else "\u2610", v,
                                f"{self._study_counts.get(v, 0):,}"),
                        tags=("on" if on else "off",))

    def _on_study_click(self, event):
        row = self._study_tree.identify_row(event.y)
        if not row or row not in self._study_checks:
            return
        # Refuse to untick the last one: an empty AnnData has nothing to plot.
        if self._study_checks[row] and sum(self._study_checks.values()) <= 1:
            return
        self._study_checks[row] = not self._study_checks[row]
        self._render_study_tree()
        self._apply_study_filter()

    def _set_all_studies(self, state: bool):
        if not self._study_checks:
            return
        if state:
            self._study_checks = {v: True for v in self._study_checks}
        else:
            first = next(iter(self._study_checks))
            self._study_checks = {v: v == first for v in self._study_checks}
        self._render_study_tree()
        self._apply_study_filter()

    def _cell_mask(self):
        """Boolean mask over the full AnnData, or None when nothing is dropped."""
        col = self.study_col_var.get().strip()
        if not col or not self._study_checks or all(self._study_checks.values()):
            return None
        np, _pd = _require_numpy_pandas()
        keep = [v for v, on in self._study_checks.items() if on]
        vals = self._adata_full.obs[col].astype(str).to_numpy()
        return np.isin(vals, keep)

    def _apply_study_filter(self):
        mask = self._cell_mask()
        self.adata = self._adata_full if mask is None else self._adata_full[mask]
        self.study_count_var.set(
            f"{int(self.adata.n_obs):,} of {int(self._adata_full.n_obs):,} cells")
        self._umap_coords = None
        for name, draw in (("composition", self._draw_composition),
                           ("dotplot", self._draw_dotplot),
                           ("qc", self._draw_qc)):
            if name in self._drawn:
                draw()
        if "umap" in self._drawn:
            # Re-embedding is minutes of work, so it waits for the button.
            self.umap_status_var.set(
                "Study filter changed - press Compute & plot to re-embed.")

    def _export_all(self):
        """Draw every tab that can be drawn, then write the figures and tables.

        Each tab here plots only when its button is pressed, so an export that
        saved what happened to be on screen would ship whichever one or two
        plots the user had looked at. The dot plot is the exception: it needs
        gene symbols typed in, and inventing a gene list would be answering a
        question the user never asked.
        """
        from pathlib import Path
        from genevariate.gui.exporting import export_window
        d = filedialog.askdirectory(title="Export folder", parent=self)
        if not d:
            return
        out = Path(d)
        out.mkdir(parents=True, exist_ok=True)

        self.configure(cursor="watch")
        self.update_idletasks()
        try:
            for draw in (self._draw_composition, self._draw_qc):
                try:
                    draw()
                except Exception:
                    pass
            if self.dot_genes_var.get().strip():
                try:
                    self._draw_dotplot()
                except Exception:
                    pass
            try:
                maxc = max(500, int(self.umap_maxcells_var.get()))
            except Exception:
                maxc = 20_000
            try:
                coords, method = self._get_umap_coords(maxc)
                self._draw_umap_plot(coords, method)
            except Exception:
                pass
        finally:
            self.configure(cursor="")

        written = export_window(self, out, figures=self._figs, prefix="scrna_")
        for name, table in self._tables.items():
            path = out / f"scrna_{name}.csv"
            try:
                table.to_csv(path)
                written.append(path)
            except Exception:
                pass
        note = ("" if self.dot_genes_var.get().strip() else
                "\n\nThe dot plot was skipped: it needs gene symbols.")
        messagebox.showinfo("Exported",
                            f"Saved {len(written)} file(s) to:\n{out}{note}",
                            parent=self)

    # ────────────────────────────────────────────────────────────────
    # 1. Composition - stacked bar of cell_type proportions per group
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

            try:
                x_cap = max(1, int(self.comp_xgroups_var.get()))
            except Exception:
                x_cap = 40

            # The same crosstab the assistant answers a composition question
            # with, and the same top-N/"other" pooling and group cap. Drawing
            # it here a second time is how the window and the assistant came
            # to disagree about what fraction of a group a cell type is.
            from genevariate.core.analysis import cell_composition
            res = cell_composition(self.adata, group, stack,
                                   top_stack=topn, max_groups=x_cap)
            counts = res.counts
            dropped_x = res.dropped_groups
            self._tables["composition"] = counts
            if self.comp_proportion_var.get():
                ct, ylabel = res.fractions, "Proportion"
            else:
                ct, ylabel = counts, "Cell count"

            self.comp_fig.clear()
            ax = self.comp_fig.add_subplot(111)
            bottoms = np.zeros(len(ct), dtype=float)
            x = np.arange(len(ct))
            pal = _palette_for(len(ct.columns))
            bars_per_col = {}
            for i, col in enumerate(ct.columns):
                vals = ct[col].to_numpy()
                bars = ax.bar(x, vals, bottom=bottoms, label=str(col),
                               color=pal[i],
                               edgecolor="white", linewidth=0.3, picker=True)
                bars_per_col[str(col)] = (bars, counts[col].to_numpy(),
                                            vals, bottoms.copy())
                bottoms += vals
            ax.set_xticks(x)
            ax.set_xticklabels([str(s) for s in ct.index], rotation=45,
                               ha="right", fontsize=8)
            ax.set_ylabel(ylabel)
            extra = f" - top {x_cap} of {x_cap + dropped_x}" if dropped_x else ""
            ax.set_title(f"{stack} composition per {group}{extra}  "
                         f"(n_cells={int(self.adata.n_obs):,})")
            ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                      fontsize=8, frameon=False)

            # Hover annotation - show group / stack / count / fraction
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
                    # ax.bar returns a BarContainer, which is a tuple of
                    # Rectangles and has no .contains of its own - asking it
                    # for one raised AttributeError on the first mouse move,
                    # so this tooltip never appeared. Hit-test the patches.
                    hit = next((k for k, rect in enumerate(bars)
                                if rect.contains(event)[0]), None)
                    if hit is not None:
                        i = hit
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
            self._drawn.add("composition")
        except Exception as exc:
            self._error("Composition plot failed", exc)

    # ────────────────────────────────────────────────────────────────
    # 2. UMAP - 2-D embedding (uses obsm['X_umap'] if present, else
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
            values=labels_all, width=22)  # not readonly - free-text allowed
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
                  foreground=AERO["accent_dark"]).pack(fill=tk.X, padx=8)

        body = ttk.Frame(tab)
        body.pack(fill=tk.BOTH, expand=True)
        self.umap_fig, self.umap_canvas = self._new_figure_panel(body)

    def _draw_umap_async(self):
        self.umap_status_var.set("Computing embedding…")
        # Invalidate cache so max_cells changes take effect
        self._umap_coords = None

        # Read the cell cap here, on the Tk thread. Tk is not thread-safe, and
        # a worker calling .get() on a Tk variable reaches into the interpreter
        # from the wrong thread.
        try:
            maxc = max(500, int(self.umap_maxcells_var.get()))
        except Exception:
            maxc = 20_000

        def _worker():
            try:
                coords, method = self._get_umap_coords(maxc)
                self.after(0, lambda: self._draw_umap_plot(coords, method))
            except Exception as exc:
                tb = traceback.format_exc()
                self.after(0, lambda: (self.umap_status_var.set("Failed."),
                                         messagebox.showerror("UMAP failed", tb,
                                                               parent=self)))

        threading.Thread(target=_worker, daemon=True).start()

    def _get_umap_coords(self, maxc=20_000):
        """Return (coords[n×2], method_str). Cached across calls.

        *maxc* is passed in rather than read from the Tk variable here,
        because this runs on a worker thread.
        """
        if self._umap_coords is not None:
            return self._umap_coords
        # Same embedding, same seed, same subsample as the assistant's, so the
        # two never place the same cell in two different spots.
        from genevariate.core.analysis import cell_embedding
        coords, method, idx = cell_embedding(self.adata, max_cells=maxc, seed=0)
        self._umap_subset_idx = idx
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
            title = f"{method} - coloured by {gene_q}"
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
                pal = _palette_for(len(categories))
                cmap = {c: pal[i] for i, c in enumerate(categories)}
                colors = [cmap[v] for v in vals]
                scatter = ax.scatter(coords[:, 0], coords[:, 1], c=colors,
                                      s=8, alpha=0.75, linewidths=0,
                                      picker=True)
                # Compact legend - top 20 categories by count
                vc = pd.Series(vals).value_counts().head(20)
                from matplotlib.patches import Patch
                handles = [Patch(facecolor=cmap[name],
                                  label=f"{name} ({n})")
                           for name, n in vc.items()]
                ax.legend(handles=handles, loc="center left",
                           bbox_to_anchor=(1.01, 0.5), fontsize=8,
                           frameon=False)
                title = f"{method} - coloured by {color}"
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
        self.umap_status_var.set(method + "  - hover a point for details.")
        self._drawn.add("umap")

    # ────────────────────────────────────────────────────────────────
    # 3. Dot plot - gene × group (mean expression + fraction expressing)
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
            font=("Segoe UI", 9, "italic"), foreground=AERO["accent_dark"])
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

            try:
                max_groups = max(1, int(self.dot_maxgroups_var.get()))
            except Exception:
                max_groups = 40

            # Same means and expressing fractions the assistant reports.
            from genevariate.core.analysis import marker_dotplot
            try:
                dot = marker_dotplot(self.adata, wanted, group,
                                     max_groups=max_groups)
            except KeyError:
                messagebox.showinfo("Dot plot",
                                     f"None of these genes are in the fetched "
                                     f"data: {wanted}",
                                     parent=self)
                return

            gene_names = list(dot.genes)
            missing = list(dot.missing)
            cats = list(dot.mean.index)
            mean_mat = dot.mean.to_numpy()
            frac_mat = dot.fraction.to_numpy()
            dropped_g = dot.dropped_groups
            n_g, n_k = len(cats), len(gene_names)
            self._tables["dotplot"] = pd.concat(
                [dot.mean.add_prefix("mean_"),
                 dot.fraction.add_prefix("frac_")], axis=1)

            self.dot_fig.clear()
            ax = self.dot_fig.add_subplot(111)
            xs, ys, sizes, colors, tips = [], [], [], [], []
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
            title = f"Dot plot - {n_k} gene(s) × {n_g} group(s)"
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
            self._drawn.add("dotplot")
        except Exception as exc:
            self._error("Dot plot failed", exc)

    # ────────────────────────────────────────────────────────────────
    # 4. QC - n_genes, total_counts, pct_mito per cell
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
            font=("Segoe UI", 9, "italic"), foreground=AERO["accent_dark"])
        hint.pack(fill=tk.X, padx=8)

        body = ttk.Frame(tab)
        body.pack(fill=tk.BOTH, expand=True)
        self.qc_fig, self.qc_canvas = self._new_figure_panel(body)

    def _draw_qc(self):
        try:
            np, pd = _require_numpy_pandas()
            prefix = self.qc_mito_var.get().strip()
            group = self.qc_group_var.get().strip()
            # Same three per-cell metrics the assistant reports, computed off
            # the sparse matrix without densifying it.
            from genevariate.core.analysis import cell_qc
            qc_tbl = cell_qc(self.adata, mito_prefix=prefix,
                             group_by=group or None)
            self._tables["qc"] = qc_tbl
            n_genes = qc_tbl["n_genes"].to_numpy()
            total = qc_tbl["total_counts"].to_numpy()
            pct_mito = qc_tbl["pct_mito"].to_numpy()

            self.qc_fig.clear()
            axes = self.qc_fig.subplots(1, 3)

            # Per-axis metadata for hover
            axis_info = {}  # id(ax) -> dict(mode, ...)

            # Convert the grouping column once, and cap how many violins are
            # drawn. A census cell_type column carries hundreds of categories:
            # one violin each was an unreadable smear that also ran a KDE per
            # category, and the mask used to re-stringify the whole column on
            # every one of them.
            _QC_MAX_GROUPS = 20
            gvals, gcats, n_cats_all = None, [], 0
            if group and group in self.adata.obs.columns:
                gvals = self.adata.obs[group].astype(str).to_numpy()
                vc_g = pd.Series(gvals).value_counts()
                n_cats_all = len(vc_g)
                gcats = list(vc_g.head(_QC_MAX_GROUPS).index)

            def _violin_or_hist(ax, values, ylabel):
                if gvals is not None and gcats:
                    data, labels_kept = [], []
                    for c in gcats:
                        m = gvals == c
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

            grouped_note = ""
            if gcats:
                grouped_note = f" grouped by {group}"
                if n_cats_all > len(gcats):
                    grouped_note += (f" (top {len(gcats)} of {n_cats_all} "
                                     f"by cell count)")
            self.qc_fig.suptitle(
                f"QC - {int(self.adata.n_obs):,} cells" + grouped_note,
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
                    # violin - map xdata → 1-based violin index
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
            self._drawn.add("qc")
        except Exception as exc:
            self._error("QC plot failed", exc)
