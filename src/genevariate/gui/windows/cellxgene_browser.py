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
     * "Load as platform (pseudo-bulk)" - aggregates by donor × cell_type,
       normalizes the aggregate the way bulk RNA-seq counts are normalized
       (see ``utils/pseudobulk.normalize_pseudobulk``), and registers the
       resulting DataFrame as a new entry in ``app.gpl_datasets``, so every
       existing analysis window sees it on the same scale as the rest.
     * "Open cell-level plots" - launches the scRNA-specific plot
       windows (composition / UMAP / dot plot / QC).
     * "Save as .h5ad" - writes the AnnData to disk for scanpy users.
   A saved .h5ad (or any scanpy file) can be read back in with
   "Load .h5ad from disk", which takes the same three actions - so the
   single-cell path does not require a live Census query.

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

from genevariate.gui.theme import (AERO, UI_FONT, labelframe, ensure_theme, style_window,
                                   wrap_to_parent)


# ────────────────────────────────────────────────────────────────────────────
# Lazy module wiring (so the app still launches if scRNA deps missing)
# ────────────────────────────────────────────────────────────────────────────
def _try_import():
    try:
        from genevariate.sources.cellxgene import CensusClient, ORGANISMS
        from genevariate.utils.anndata_io import (
            summarize_adata, save_h5ad, load_h5ad, anndata_to_platform_df,
        )
        from genevariate.utils.pseudobulk import (
            pseudobulk, pseudobulk_to_platform_df, describe_pseudobulk,
            is_raw_counts, normalize_pseudobulk, aggregate_to_platform,
        )
        return dict(CensusClient=CensusClient, ORGANISMS=ORGANISMS,
                    summarize_adata=summarize_adata,
                    save_h5ad=save_h5ad, load_h5ad=load_h5ad,
                    anndata_to_platform_df=anndata_to_platform_df,
                    pseudobulk=pseudobulk,
                    pseudobulk_to_platform_df=pseudobulk_to_platform_df,
                    describe_pseudobulk=describe_pseudobulk,
                    is_raw_counts=is_raw_counts,
                    normalize_pseudobulk=normalize_pseudobulk,
                    aggregate_to_platform=aggregate_to_platform)
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
        ensure_theme(self)
        style_window(self)
        self.title("CELLxGENE Census - Single-cell data source")
        self.geometry("1200x780")
        try:
            self.transient(parent)
        except Exception:
            pass

        self._adata = None          # last fetched AnnData
        self._loaded_name = None    # file stem when _adata came from disk
        self._raw_counts = False    # whether _adata.X holds read counts
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
                  foreground=AERO["danger"],
                  font=("Consolas", 9)).pack(pady=10)
        ttk.Button(f, text="Close", command=self.destroy).pack()

    # ───── UI build ────────────────────────────────────────────────────
    def _build_ui(self):
        # Top banner - remind users that all data is real
        banner = ttk.Frame(self, padding=(10, 6))
        banner.pack(fill=tk.X)
        wrap_to_parent(ttk.Label(
            banner,
            text=("CELLxGENE Discover Census - all values are real measurements "
                  "from public scRNA-seq submissions. "
                  "Pseudo-bulk aggregation collapses real cells into "
                  "donor × cell-type groups (mean / sum / median)."),
            justify=tk.LEFT,
            font=(UI_FONT, 9, "italic"),
            foreground=AERO["accent_dark"],
        )).pack(fill=tk.X)

        # Main two-column layout
        main = ttk.Frame(self, padding=8)
        main.pack(fill=tk.BOTH, expand=True)
        main.columnconfigure(0, weight=2)
        main.columnconfigure(1, weight=3)
        main.rowconfigure(0, weight=1)

        # ── Left: query builder ─────────────────────────────────────
        left = labelframe(main, text="Census query", padding=10)
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
            ttk.Label(row, text=label, width=17).pack(side=tk.LEFT)
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
        # tissue_general is a coarser column than tissue, not a synonym for it:
        # "lung" against "upper lobe of left lung". A survey that means to cover
        # an organ has to filter on this one, and until it was exposed here that
        # query could only be written in a script.
        self.tissue_general_var, self.tissue_general_combo = _combo_row(
            left, "Tissue (general):", ())
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
        ttk.Label(r2, text="Max cells:", width=17).pack(side=tk.LEFT)
        self.max_cells_var = tk.StringVar(value="0")
        ttk.Entry(r2, textvariable=self.max_cells_var, width=12).pack(side=tk.LEFT, padx=4)
        ttk.Label(r2, text="Seed:").pack(side=tk.LEFT, padx=(8, 0))
        # A capped fetch draws a random subsample, so the seed is part of what
        # produced the matrix. It is shown and recorded rather than left at a
        # hidden default, which is what makes the fetch repeatable.
        self.seed_var = tk.StringVar(value="0")
        ttk.Entry(r2, textvariable=self.seed_var, width=8).pack(side=tk.LEFT, padx=4)
        ttk.Label(r2, text="(0 cells = all matching; a cap draws a seeded random subsample)",
                  font=("Segoe UI", 8, "italic"), foreground=AERO["muted"]
                  ).pack(side=tk.LEFT, padx=4)

        r3 = ttk.Frame(left); r3.pack(fill=tk.X, pady=4)
        ttk.Label(r3, text="Genes (optional):", width=17).pack(side=tk.LEFT, anchor="n")
        self.genes_text = tk.Text(r3, height=4, width=36, wrap=tk.WORD,
                                   font=("Consolas", 9))
        self.genes_text.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Label(r3, text="comma/space/newline\nseparated",
                  font=("Segoe UI", 8, "italic"), foreground=AERO["muted"]
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
                  foreground=AERO["accent_dark"]).pack(side=tk.LEFT)

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
        bottom = labelframe(self, text=" What to do with the fetched data ",
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

        # Always enabled: a saved .h5ad is the only way back into the
        # single-cell path without querying the Census again.
        ttk.Button(br, text="Load .h5ad from disk",
                   command=self._load_h5ad_file).pack(side=tk.LEFT, padx=2)

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
        """Called when organism changes - clears tissue/disease/etc. combos."""
        for combo in (self.tissue_combo, self.tissue_general_combo,
                       self.disease_combo,
                       self.celltype_combo, self.assay_combo):
            combo["values"] = []
        self._set_status(f"Organism set to {self.organism_var.get()}. "
                          "Click '…' next to a field to load its values.")

    def _populate_combo_async(self, combo: ttk.Combobox, label: str):
        """Load unique values of a Census obs column into a combo box."""
        colmap = {
            "Tissue:": "tissue",
            "Tissue (general):": "tissue_general",
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
    @staticmethod
    def _filter_value(raw: str):
        """One value, or several separated by ';'.

        Semicolon rather than comma: Census ontology labels contain commas
        ("CD4-positive, alpha-beta T cell"), so splitting on those would quietly
        turn one real cell type into two that match nothing.
        """
        parts = [p.strip() for p in str(raw).split(";") if p.strip()]
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else parts

    def _collect_filters(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        for name, var in (("tissue", self.tissue_var),
                           ("disease", self.disease_var),
                           ("cell_type", self.celltype_var),
                           ("assay", self.assay_var),
                           ("sex", self.sex_var)):
            v = self._filter_value(var.get())
            if v is not None:
                d[name] = v
        d["is_primary_data"] = bool(self.primary_only_var.get())
        # tissue_general is not one of preview()/fetch()'s named parameters, so
        # it travels in extra=, which builds the same obs filter clause.
        tg = self._filter_value(self.tissue_general_var.get())
        if tg is not None:
            d["extra"] = {"tissue_general": tg}
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
            max_cells = None
        try:
            seed = int(self.seed_var.get())
        except Exception:
            seed = 0
        self._last_seed = seed

        self._set_status("Fetching from CELLxGENE Census…")
        self._loaded_name = None
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
                    random_seed=seed,
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

    def _load_h5ad_file(self):
        p = filedialog.askopenfilename(
            title="Open .h5ad",
            filetypes=[("AnnData h5ad", "*.h5ad"), ("All files", "*")],
            parent=self)
        if not p:
            return
        self._set_status(f"Reading {Path(p).name}…")

        def _worker():
            try:
                adata = _MODS["load_h5ad"](p)
                self.after(0, lambda: self._on_loaded_from_disk(adata, p))
            except Exception:
                tb = traceback.format_exc()
                self.after(0, lambda: self._on_error("Could not read .h5ad", tb))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_loaded_from_disk(self, adata, path):
        self._loaded_name = Path(path).stem
        self._on_fetched(adata)
        # A file off disk need not follow the Census schema, so the default
        # donor_id,cell_type groupby would fail before it started.
        obs = list(adata.obs.columns.astype(str))
        donor = next((c for c in ("donor_id", "sample_id", "patient_id",
                                  "patient", "batch") if c in obs), None)
        ctype = next((c for c in ("cell_type", "celltype", "CellType",
                                  "cell_ontology_class", "cluster",
                                  "leiden", "louvain") if c in obs), None)
        pref = [c for c in (donor, ctype) if c]
        if pref:
            self.pb_group_var.set(",".join(pref))
        self._set_status(f"Loaded {Path(path).name}: {adata.n_obs:,} cells × "
                          f"{adata.n_vars:,} genes. Pick an action below.")

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
        _add("Random seed",       getattr(self, "_last_seed", 0))

        # The Census raw layer is read counts, and summing them is what makes a
        # group a library the bulk RNA-seq normalization can act on. A mean of
        # raw counts is neither counts nor normalized, so it is not the default
        # for this input.
        self._raw_counts = bool(_MODS["is_raw_counts"](adata.X))
        _add("Value scale", "raw counts" if self._raw_counts else "normalized")
        if self._raw_counts:
            self.pb_agg_var.set("sum")

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
                # The same pipeline the assistant's tools run. The three
                # settings stay editable here - a user may want a mean, or a
                # different grouping - but the steps after them are the
                # program's, so a platform loaded by clicking and one loaded
                # by asking are built the same way.
                df, pb = _MODS["aggregate_to_platform"](
                    self._adata, groupby=groupby, agg=agg,
                    min_cells=min_cells)
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
        agg = pb_adata.uns["pseudobulk"]["agg"]
        if self._loaded_name:
            # The Census filter combos say nothing about a file off disk, so
            # the file is what names the platform.
            safe = "".join(c if c.isalnum() else "_"
                           for c in self._loaded_name)[:40]
            name = f"scRNA_{safe}_{agg}"
        else:
            name = f"CellxGene_{tag}_{agg}"

        # Stash the AnnData first: the app resolves a single-cell platform's
        # organism through scrna_datasets, and registration reads it back.
        if not hasattr(self.app, "scrna_datasets"):
            self.app.scrna_datasets = {}
        self.app.scrna_datasets[name] = {
            "cells": self._adata,
            "pseudobulk": pb_adata,
        }
        # Register on the main app so every downstream window sees it. This
        # indexes the gene columns as well as storing the frame; a frame
        # stored without its index is a platform with zero genes everywhere
        # a gene symbol has to be resolved.
        self.app.register_platform_frame(name, df)
        try:
            self.app._update_platform_status()
        except Exception:
            pass
        info = pb_adata.uns["pseudobulk"]
        norm = info.get("normalization")
        norm_line = (
            f"  Normalization: {norm} ({info['normalization_scope']}), "
            f"{info['genes_dropped_low_count']:,} low-count genes dropped\n"
            if norm else
            "  Normalization: none - values were already normalized\n")
        msg = (f"Registered platform '{name}'\n"
               f"  Pseudo-bulk groups (one row per group): {pb_adata.n_obs:,}\n"
               f"  Genes: {pb_adata.n_vars:,}\n"
               f"  Aggregation: {info['agg']} of "
               f"{info['n_cells_kept']:,} real cells\n"
               + norm_line +
               f"  (every value derives from real measurements, "
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
