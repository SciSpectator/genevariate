"""
CELLxGENE Discover Census browser window.

Flow
----
1. User picks organism → schema populates tissue / disease / cell-type / assay
   combo boxes from the live Census.
2. User clicks "Preview matching cells" → a count + top-label distributions
   render in the right-hand panel.
3. User sets a cell-count cap (default 50,000) and an optional gene list,
   then clicks "Fetch AnnData". A background thread runs the query.
4. Once fetched, three terminal actions are offered:
     * "Load as platform (pseudo-bulk)" — aggregates by donor × cell_type
       and registers the resulting DataFrame as a new entry in
       ``app.gpl_datasets``, so every existing analysis window sees it.
     * "Open cell-level plots" — launches the scRNA-specific plot
       windows (composition / UMAP / dot plot / QC).
     * "Save as .h5ad" — writes the AnnData to disk for scanpy users.

All values shown are **real measurements** from public CELLxGENE
submissions. The pseudo-bulk path is a transparent mean/sum aggregation
(see ``utils/pseudobulk.py``), not synthetic data generation.
"""

from __future__ import annotations

import threading
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import tkinter as tk
from tkinter import ttk, messagebox, filedialog


# ────────────────────────────────────────────────────────────────────────────
# Lazy module wiring (so the app still launches if scRNA deps missing)
# ────────────────────────────────────────────────────────────────────────────
def _try_import():
    try:
        from genevariate.sources.cellxgene import CensusClient, ORGANISMS
        from genevariate.utils.anndata_io import (
            summarize_adata, save_h5ad, anndata_to_platform_df,
        )
        from genevariate.utils.pseudobulk import (
            pseudobulk, pseudobulk_to_platform_df, describe_pseudobulk,
        )
        return dict(CensusClient=CensusClient, ORGANISMS=ORGANISMS,
                    summarize_adata=summarize_adata,
                    save_h5ad=save_h5ad,
                    anndata_to_platform_df=anndata_to_platform_df,
                    pseudobulk=pseudobulk,
                    pseudobulk_to_platform_df=pseudobulk_to_platform_df,
                    describe_pseudobulk=describe_pseudobulk)
    except Exception as exc:
        return {"_error": str(exc)}


_MODS = _try_import()


# ────────────────────────────────────────────────────────────────────────────
# Browser window
# ────────────────────────────────────────────────────────────────────────────
class CellxGeneBrowserWindow(tk.Toplevel):
    """Tk Toplevel for fetching CELLxGENE Census data into GeneVariate."""

    COMMON_TISSUES = (
        "", "lung", "liver", "kidney", "heart", "brain", "pancreas",
        "skin", "blood", "bone marrow", "thymus", "spleen", "colon",
        "small intestine", "stomach", "breast", "prostate", "ovary",
        "testis", "skeletal muscle", "adipose tissue",
    )
    COMMON_DISEASES = (
        "", "normal", "COVID-19", "lung adenocarcinoma", "Alzheimer disease",
        "type 2 diabetes mellitus", "breast cancer", "melanoma",
        "Parkinson disease", "Crohn disease",
    )

    def __init__(self, parent):
        super().__init__(parent)
        self.app = parent
        self.title("CELLxGENE Census — Single-cell data source")
        self.geometry("1200x780")
        try:
            self.transient(parent)
        except Exception:
            pass

        self._adata = None          # last fetched AnnData
        self._client = None          # CensusClient (opened lazily)
        self._fetch_thread = None

        if "_error" in _MODS:
            self._render_missing_deps(_MODS["_error"])
            return

        self._build_ui()
        self._populate_organisms()

    # ───── Missing-deps fallback ───────────────────────────────────────
    def _render_missing_deps(self, err: str):
        f = ttk.Frame(self, padding=20)
        f.pack(fill=tk.BOTH, expand=True)
        ttk.Label(f,
                  text="Single-cell support requires additional packages.",
                  font=("Segoe UI", 12, "bold")).pack(pady=6)
        ttk.Label(f,
                  text="Install them from a terminal:",
                  font=("Segoe UI", 10)).pack()
        cmd = "pip install --user cellxgene-census anndata scanpy"
        entry = ttk.Entry(f, width=70)
        entry.insert(0, cmd)
        entry.config(state="readonly")
        entry.pack(pady=6)
        ttk.Label(f, text=f"Import error:\n{err}",
                  foreground="#C62828",
                  font=("Consolas", 9)).pack(pady=10)
        ttk.Button(f, text="Close", command=self.destroy).pack()

    # ───── UI build ────────────────────────────────────────────────────
    def _build_ui(self):
        # Top banner — remind users that all data is real
        banner = ttk.Frame(self, padding=(10, 6))
        banner.pack(fill=tk.X)
        ttk.Label(
            banner,
            text=("CELLxGENE Discover Census — all values are real measurements "
                  "from public scRNA-seq submissions. "
                  "Pseudo-bulk aggregation collapses real cells into "
                  "donor × cell-type groups (mean / sum / median)."),
            wraplength=1150, justify=tk.LEFT,
            font=("Segoe UI", 9, "italic"),
            foreground="#0A5B9A",
        ).pack(fill=tk.X)

        # Main two-column layout
        main = ttk.Frame(self, padding=8)
        main.pack(fill=tk.BOTH, expand=True)
        main.columnconfigure(0, weight=2)
        main.columnconfigure(1, weight=3)
        main.rowconfigure(0, weight=1)

        # ── Left: query builder ─────────────────────────────────────
        left = ttk.LabelFrame(main, text="Census query", padding=10)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        r0 = ttk.Frame(left); r0.pack(fill=tk.X, pady=4)
        ttk.Label(r0, text="Organism:", width=13).pack(side=tk.LEFT)
        self.organism_var = tk.StringVar(value="homo_sapiens")
        self.organism_combo = ttk.Combobox(
            r0, textvariable=self.organism_var, width=24, state="readonly"
        )
        self.organism_combo.pack(side=tk.LEFT, padx=4)
        self.organism_combo.bind("<<ComboboxSelected>>",
                                  lambda e: self._refresh_schema())

        def _combo_row(parent, label, default_values=()):
            row = ttk.Frame(parent); row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=13).pack(side=tk.LEFT)
            var = tk.StringVar()
            combo = ttk.Combobox(row, textvariable=var,
                                  values=default_values, width=36)
            combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
            ttk.Button(row, text="…", width=3,
                       command=lambda: self._populate_combo_async(combo, label)
                       ).pack(side=tk.LEFT)
            return var, combo

        self.tissue_var, self.tissue_combo = _combo_row(left, "Tissue:",
                                                         self.COMMON_TISSUES)
        self.disease_var, self.disease_combo = _combo_row(left, "Disease:",
                                                            self.COMMON_DISEASES)
        self.celltype_var, self.celltype_combo = _combo_row(left, "Cell type:", ())
        self.assay_var, self.assay_combo = _combo_row(left, "Assay:", ())
        self.sex_var, self.sex_combo = _combo_row(left, "Sex:",
                                                    ("", "male", "female"))

        r1 = ttk.Frame(left); r1.pack(fill=tk.X, pady=(8, 4))
        self.primary_only_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(r1, text="Primary data only (excludes duplicate reanalyses)",
                        variable=self.primary_only_var
                        ).pack(side=tk.LEFT)

        r2 = ttk.Frame(left); r2.pack(fill=tk.X, pady=4)
        ttk.Label(r2, text="Max cells:", width=13).pack(side=tk.LEFT)
        self.max_cells_var = tk.StringVar(value="50000")
        ttk.Entry(r2, textvariable=self.max_cells_var, width=12).pack(side=tk.LEFT, padx=4)
        ttk.Label(r2, text="(hard cap — larger fetches are subsampled)",
                  font=("Segoe UI", 8, "italic"), foreground="gray"
                  ).pack(side=tk.LEFT, padx=4)

        r3 = ttk.Frame(left); r3.pack(fill=tk.X, pady=4)
        ttk.Label(r3, text="Genes (optional):", width=13).pack(side=tk.LEFT, anchor="n")
        self.genes_text = tk.Text(r3, height=4, width=36, wrap=tk.WORD,
                                   font=("Consolas", 9))
        self.genes_text.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Label(r3, text="comma/space/newline\nseparated",
                  font=("Segoe UI", 8, "italic"), foreground="gray"
                  ).pack(side=tk.LEFT, anchor="n")

        r4 = ttk.Frame(left); r4.pack(fill=tk.X, pady=(10, 4))
        try:
            self.preview_btn = ttk.Button(
                r4, text="Preview matching cells", command=self._do_preview,
                style="Secondary.TButton")
        except tk.TclError:
            self.preview_btn = ttk.Button(r4, text="Preview matching cells",
                                           command=self._do_preview)
        self.preview_btn.pack(side=tk.LEFT, padx=2)
        try:
            self.fetch_btn = ttk.Button(r4, text="Fetch AnnData",
                                         command=self._do_fetch,
                                         style="Primary.TButton")
        except tk.TclError:
            self.fetch_btn = ttk.Button(r4, text="Fetch AnnData",
                                         command=self._do_fetch)
        self.fetch_btn.pack(side=tk.LEFT, padx=2)

        r5 = ttk.Frame(left); r5.pack(fill=tk.X, pady=(12, 2))
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(r5, textvariable=self.status_var,
                  font=("Segoe UI", 9, "italic"),
                  foreground="#0A5B9A").pack(side=tk.LEFT)

        # ── Right: preview / results panel ─────────────────────────
        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.nb = ttk.Notebook(right)
        self.nb.grid(row=0, column=0, sticky="nsew")

        # Preview tab
        tab_p = ttk.Frame(self.nb); self.nb.add(tab_p, text="Preview")
        self.preview_tree = ttk.Treeview(
            tab_p, columns=("key", "value"), show="headings", height=18
        )
        self.preview_tree.heading("key", text="Field")
        self.preview_tree.heading("value", text="Value")
        self.preview_tree.column("key", width=180, anchor="w")
        self.preview_tree.column("value", width=500, anchor="w")
        sb_p = ttk.Scrollbar(tab_p, command=self.preview_tree.yview)
        self.preview_tree.configure(yscrollcommand=sb_p.set)
        self.preview_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)
        sb_p.pack(side=tk.RIGHT, fill=tk.Y)

        # Fetched-AnnData tab (populated after fetch)
        tab_a = ttk.Frame(self.nb); self.nb.add(tab_a, text="Fetched AnnData")
        self.fetched_tree = ttk.Treeview(
            tab_a, columns=("key", "value"), show="headings", height=18
        )
        self.fetched_tree.heading("key", text="Field")
        self.fetched_tree.heading("value", text="Value")
        self.fetched_tree.column("key", width=220, anchor="w")
        self.fetched_tree.column("value", width=500, anchor="w")
        sb_a = ttk.Scrollbar(tab_a, command=self.fetched_tree.yview)
        self.fetched_tree.configure(yscrollcommand=sb_a.set)
        self.fetched_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)
        sb_a.pack(side=tk.RIGHT, fill=tk.Y)

        # ── Bottom: terminal actions ────────────────────────────────
        bottom = ttk.LabelFrame(self, text=" What to do with the fetched data ",
                                  padding=10)
        bottom.pack(fill=tk.X, padx=8, pady=(4, 8))

        # Pseudo-bulk config
        pb = ttk.Frame(bottom); pb.pack(fill=tk.X, pady=4)
        ttk.Label(pb, text="Pseudo-bulk groupby:",
                  font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        self.pb_group_var = tk.StringVar(value="donor_id,cell_type")
        ttk.Entry(pb, textvariable=self.pb_group_var, width=28
                  ).pack(side=tk.LEFT, padx=4)
        ttk.Label(pb, text="Aggregation:").pack(side=tk.LEFT, padx=(10, 2))
        self.pb_agg_var = tk.StringVar(value="mean")
        ttk.Combobox(pb, textvariable=self.pb_agg_var, width=8,
                      state="readonly",
                      values=("mean", "sum", "median")
                      ).pack(side=tk.LEFT, padx=2)
        ttk.Label(pb, text="Min cells/group:").pack(side=tk.LEFT, padx=(10, 2))
        self.pb_min_var = tk.StringVar(value="10")
        ttk.Entry(pb, textvariable=self.pb_min_var, width=6
                  ).pack(side=tk.LEFT)

        # Action buttons
        br = ttk.Frame(bottom); br.pack(fill=tk.X, pady=(6, 2))
        self.load_platform_btn = ttk.Button(
            br, text="Load as platform (pseudo-bulk)",
            command=self._load_as_platform, state=tk.DISABLED)
        self.load_platform_btn.pack(side=tk.LEFT, padx=2)

        self.cell_plots_btn = ttk.Button(
            br, text="Open cell-level plots",
            command=self._open_cell_plots, state=tk.DISABLED)
        self.cell_plots_btn.pack(side=tk.LEFT, padx=2)

        self.save_h5ad_btn = ttk.Button(
            br, text="Save as .h5ad",
            command=self._save_h5ad, state=tk.DISABLED)
        self.save_h5ad_btn.pack(side=tk.LEFT, padx=2)

        ttk.Button(br, text="Close", command=self.destroy
                   ).pack(side=tk.RIGHT, padx=2)

    # ───── Census & schema helpers ─────────────────────────────────
    def _get_client(self):
        if self._client is None:
            self._client = _MODS["CensusClient"]()
        return self._client

    def _populate_organisms(self):
        self.organism_combo["values"] = list(_MODS["ORGANISMS"])

    def _refresh_schema(self):
        """Called when organism changes — clears tissue/disease/etc. combos."""
        for combo in (self.tissue_combo, self.disease_combo,
                       self.celltype_combo, self.assay_combo):
            combo["values"] = []
        self._set_status(f"Organism set to {self.organism_var.get()}. "
                          "Click '…' next to a field to load its values.")

    def _populate_combo_async(self, combo: ttk.Combobox, label: str):
        """Load unique values of a Census obs column into a combo box."""
        colmap = {
            "Tissue:": "tissue",
            "Disease:": "disease",
            "Cell type:": "cell_type",
            "Assay:": "assay",
            "Sex:": "sex",
        }
        col = colmap.get(label)
        if not col:
            return
        self._set_status(f"Loading unique {col} values from Census…")

        def _worker():
            try:
                client = self._get_client()
                vals = client.unique_values(col, organism=self.organism_var.get(),
                                              limit=1000)
                self.after(0, lambda: self._apply_combo(combo, vals, col))
            except Exception as exc:
                err = "".join(traceback.format_exception_only(type(exc), exc))
                self.after(0, lambda: self._set_status(f"Error: {err.strip()}"))

        threading.Thread(target=_worker, daemon=True).start()

    def _apply_combo(self, combo, vals, col):
        combo["values"] = [""] + vals
        self._set_status(f"Loaded {len(vals)} {col} values.")

    # ───── Preview ─────────────────────────────────────────────────
    def _collect_filters(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        for name, var in (("tissue", self.tissue_var),
                           ("disease", self.disease_var),
                           ("cell_type", self.celltype_var),
                           ("assay", self.assay_var),
                           ("sex", self.sex_var)):
            v = var.get().strip()
            if v:
                d[name] = v
        d["is_primary_data"] = bool(self.primary_only_var.get())
        return d

    def _parse_gene_list(self) -> Optional[List[str]]:
        raw = self.genes_text.get("1.0", tk.END).strip()
        if not raw:
            return None
        parts = [p.strip() for p in raw.replace(",", " ").replace("\n", " ").split()]
        genes = [p for p in parts if p]
        return genes or None

    def _do_preview(self):
        filters = self._collect_filters()
        self._set_status("Previewing from Census…")
        self.preview_btn.config(state=tk.DISABLED)

        def _worker():
            try:
                client = self._get_client()
                summary = client.preview(
                    organism=self.organism_var.get(), **filters)
                self.after(0, lambda: self._show_preview(summary))
            except Exception as exc:
                tb = traceback.format_exc()
                self.after(0, lambda: self._on_error("Preview failed", tb))
            finally:
                self.after(0, lambda: self.preview_btn.config(state=tk.NORMAL))

        threading.Thread(target=_worker, daemon=True).start()

    def _show_preview(self, summary: Dict[str, Any]):
        self.preview_tree.delete(*self.preview_tree.get_children())
        def _add(k, v):
            self.preview_tree.insert("", "end", values=(k, v))
        _add("Filter", summary["filter"])
        _add("Cells matched", f"{summary['n_cells_matched']:,}")
        _add("Cells previewed (sample)", f"{summary['n_cells_previewed']:,}")
        _add("Distinct datasets", f"{summary['n_datasets']:,}")
        _add("Distinct donors", f"{summary['n_donors']:,}")
        for col in ("cell_type", "tissue", "disease", "assay", "sex"):
            if col in summary:
                _add(f"Top {col}", f"{summary.get(f'{col}_n_unique', '?')} unique")
                for name, n in summary[col]:
                    _add(f"  · {name}", f"{n:,}")
        self.nb.select(0)
        self._set_status(f"Preview complete: "
                          f"{summary['n_cells_matched']:,} cells match.")

    # ───── Fetch ────────────────────────────────────────────────────
    def _do_fetch(self):
        filters = self._collect_filters()
        genes = self._parse_gene_list()
        try:
            max_cells = int(self.max_cells_var.get())
            if max_cells <= 0:
                max_cells = None
        except Exception:
            max_cells = 50_000

        self._set_status("Fetching from CELLxGENE Census…")
        self.fetch_btn.config(state=tk.DISABLED)
        self.preview_btn.config(state=tk.DISABLED)

        def _progress(msg: str):
            self.after(0, lambda: self._set_status(msg))

        def _worker():
            try:
                client = self._get_client()
                adata = client.fetch(
                    organism=self.organism_var.get(),
                    genes=genes,
                    max_cells=max_cells,
                    progress_callback=_progress,
                    **filters,
                )
                self.after(0, lambda: self._on_fetched(adata))
            except Exception as exc:
                tb = traceback.format_exc()
                self.after(0, lambda: self._on_error("Fetch failed", tb))
            finally:
                self.after(0, lambda: (
                    self.fetch_btn.config(state=tk.NORMAL),
                    self.preview_btn.config(state=tk.NORMAL),
                ))

        self._fetch_thread = threading.Thread(target=_worker, daemon=True)
        self._fetch_thread.start()

    def _on_fetched(self, adata):
        self._adata = adata
        summary = _MODS["summarize_adata"](adata)
        self.fetched_tree.delete(*self.fetched_tree.get_children())
        def _add(k, v):
            self.fetched_tree.insert("", "end", values=(k, v))
        _add("Cells fetched (n_obs)", f"{summary['n_cells']:,}")
        _add("Genes (n_vars)", f"{summary['n_genes']:,}")
        for col in ("cell_type", "tissue", "disease", "assay",
                     "development_stage", "sex"):
            if col in summary:
                _add(f"Unique {col}", summary[f"{col}_n_unique"])
                for name, n in list(summary[col].items())[:8]:
                    _add(f"  · {name}", n)
        src = adata.uns.get("source", {})
        _add("Source",            src.get("origin", "CELLxGENE Census"))
        _add("Census version",    src.get("census_version", "stable"))
        _add("Filter",            src.get("obs_filter", ""))
        self.nb.select(1)
        self._set_status(f"Fetched {adata.n_obs:,} cells × "
                          f"{adata.n_vars:,} genes. "
                          "Pick an action below.")
        for b in (self.load_platform_btn, self.cell_plots_btn,
                   self.save_h5ad_btn):
            b.config(state=tk.NORMAL)

    # ───── Terminal actions ────────────────────────────────────────
    def _load_as_platform(self):
        if self._adata is None:
            return
        groupby = [p.strip() for p in self.pb_group_var.get().split(",") if p.strip()]
        if not groupby:
            messagebox.showerror("Pseudo-bulk", "Groupby cannot be empty.",
                                  parent=self)
            return
        try:
            agg = self.pb_agg_var.get()
            min_cells = max(1, int(self.pb_min_var.get()))
        except Exception:
            agg, min_cells = "mean", 10

        # Validate groupby columns exist before doing expensive work
        missing = [g for g in groupby if g not in self._adata.obs.columns]
        if missing:
            avail = ", ".join(sorted(self._adata.obs.columns.astype(str)))
            messagebox.showerror(
                "Groupby column missing",
                f"These columns are not in the fetched data: {missing}\n\n"
                f"Available obs columns:\n{avail}",
                parent=self)
            return

        self._set_status(f"Pseudo-bulking ({agg}, groupby={groupby}, "
                          f"min_cells={min_cells})…")

        def _worker():
            try:
                pb = _MODS["pseudobulk"](
                    self._adata, groupby=groupby, agg=agg,
                    min_cells=min_cells)
                df = _MODS["pseudobulk_to_platform_df"](pb)
                self.after(0, lambda: self._register_platform(df, pb))
            except Exception as exc:
                tb = traceback.format_exc()
                self.after(0, lambda: self._on_error(
                    "Pseudo-bulk failed", tb))

        threading.Thread(target=_worker, daemon=True).start()

    def _register_platform(self, df, pb_adata):
        # Build a platform name like "CellxGene_lung_normal_mean"
        src = self._adata.uns.get("source", {}) if self._adata is not None else {}
        filt = src.get("obs_filter", "") or ""
        tag_parts = []
        for key, val in self._collect_filters().items():
            if key in ("is_primary_data",):
                continue
            if isinstance(val, (list, tuple)):
                tag_parts.append("_".join(str(v) for v in val)[:20])
            elif val:
                tag_parts.append(str(val).replace(" ", "_")[:20])
        tag = "_".join(tag_parts) or "all"
        name = f"CellxGene_{tag}_{pb_adata.uns['pseudobulk']['agg']}"

        # Register on the main app so every downstream window sees it
        if not hasattr(self.app, "gpl_datasets") or self.app.gpl_datasets is None:
            self.app.gpl_datasets = {}
        self.app.gpl_datasets[name] = df
        # Also stash the AnnData for later cell-level analyses
        if not hasattr(self.app, "scrna_datasets"):
            self.app.scrna_datasets = {}
        self.app.scrna_datasets[name] = {
            "cells": self._adata,
            "pseudobulk": pb_adata,
        }
        try:
            self.app._update_platform_status()
        except Exception:
            pass
        msg = (f"Registered platform '{name}'\n"
               f"  Pseudo-bulk groups (one row per group): {pb_adata.n_obs:,}\n"
               f"  Genes: {pb_adata.n_vars:,}\n"
               f"  Aggregation: {pb_adata.uns['pseudobulk']['agg']} of "
               f"{pb_adata.uns['pseudobulk']['n_cells_kept']:,} real cells\n"
               f"  (every value = {pb_adata.uns['pseudobulk']['agg']} of real measurements, "
               f"no simulation)\n\n"
               f"You can now open Gene Explorer, Label Enrichment, "
               f"or Compare Distributions to analyze it.")
        messagebox.showinfo("Loaded as platform", msg, parent=self)
        self._set_status(f"Registered platform '{name}'.")

    def _open_cell_plots(self):
        if self._adata is None:
            return
        try:
            from genevariate.gui.windows.scrna_plots import ScrnaPlotsWindow
        except Exception as exc:
            messagebox.showerror("Cell-level plots",
                                  f"Could not load plot window:\n{exc}",
                                  parent=self)
            return
        ScrnaPlotsWindow(self, self._adata)

    def _save_h5ad(self):
        if self._adata is None:
            return
        p = filedialog.asksaveasfilename(
            defaultextension=".h5ad",
            filetypes=[("AnnData h5ad", "*.h5ad")],
            initialfile="cellxgene_fetch.h5ad",
            parent=self)
        if not p:
            return
        try:
            _MODS["save_h5ad"](self._adata, p)
            self._set_status(f"Saved to {p}")
            messagebox.showinfo("Saved", f"AnnData saved to:\n{p}", parent=self)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc), parent=self)

    # ───── Misc helpers ─────────────────────────────────────────────
    def _set_status(self, msg: str):
        self.status_var.set(msg)

    def _on_error(self, title: str, tb: str):
        self._set_status(f"{title}")
        messagebox.showerror(title, tb, parent=self)

    def destroy(self):
        try:
            if self._client is not None:
                self._client.close()
        except Exception:
            pass
        super().destroy()
