"""
RNA-seq differential-expression window (raw counts → DESeq2 → GSEA).

Completes the NGS path: load a raw count matrix (CSV/TSV, 10x MTX dir, or
h5ad), assign samples to case / control, then run QC → DESeq2 median-of-ratios
normalisation → DESeq2 negative-binomial DE (pydeseq2) → GSEA prerank on the
Wald statistic. Results render in QC / DE-table / GSEA tabs, and the normalised
matrix can be registered as a GeneVariate platform so every other analysis
window can use it.

Structure mirrors ``gui/windows/cellxgene_browser.py``: lazy ``_try_import``,
``_render_missing_deps`` fallback, worker threads that marshal UI updates via
``self.after(0, ...)`` and drive the shared progress bar through the app's
``_acquire_progress`` / ``update_progress`` / ``_release_progress`` API.
"""
from __future__ import annotations

import threading
import traceback
from pathlib import Path
from typing import Dict, Optional

import tkinter as tk
from tkinter import ttk, messagebox, filedialog


# ────────────────────────────────────────────────────────────────────────────
# Lazy wiring (app still launches if pydeseq2 / gseapy absent)
# ────────────────────────────────────────────────────────────────────────────
def _try_import():
    try:
        from genevariate.core.count_io import load_counts
        from genevariate.core.analysis.rnaseq import (
            compute_qc_metrics, deseq2_size_factors, cpm_normalize,
            run_deseq2, deseq_results_to_ranked, counts_to_platform_df,
            _HAS_PYDESEQ2,
        )
        from genevariate.core.analysis import run_prerank_gsea, DEFAULT_LIBRARIES
        return dict(
            load_counts=load_counts,
            compute_qc_metrics=compute_qc_metrics,
            deseq2_size_factors=deseq2_size_factors,
            cpm_normalize=cpm_normalize,
            run_deseq2=run_deseq2,
            deseq_results_to_ranked=deseq_results_to_ranked,
            counts_to_platform_df=counts_to_platform_df,
            run_prerank_gsea=run_prerank_gsea,
            DEFAULT_LIBRARIES=DEFAULT_LIBRARIES,
            _HAS_PYDESEQ2=_HAS_PYDESEQ2,
        )
    except Exception as exc:  # pragma: no cover - only when deps missing
        return {"_error": str(exc)}


_MODS = _try_import()


def _require_mpl():
    import matplotlib
    matplotlib.use("Agg", force=False)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    try:
        from genevariate.utils.viz_style import apply_genevariate_style
        apply_genevariate_style()
    except Exception:
        pass
    return Figure, FigureCanvasTkAgg


class RnaSeqDEWindow(tk.Toplevel):
    """Raw-count RNA-seq differential expression + GSEA."""

    def __init__(self, parent):
        super().__init__(parent)
        self.app = parent
        self.title("RNA-seq Differential Expression (counts → DESeq2 → GSEA)")
        self.geometry("1180x760")
        try:
            self.transient(parent)
        except Exception:
            pass

        self._counts = None          # genes x samples raw counts
        self._sample_meta = None     # optional per-sample metadata
        self._norm = None            # log-CPM normalised (for registration)
        self._ranked = None          # DESeq2 ranked frame
        self._busy = False

        if "_error" in _MODS:
            self._render_missing_deps(_MODS["_error"])
            return
        self._build_ui()

    # ───── Missing-deps fallback ───────────────────────────────────────
    def _render_missing_deps(self, err: str):
        f = ttk.Frame(self, padding=20)
        f.pack(fill=tk.BOTH, expand=True)
        ttk.Label(f, text="RNA-seq DE requires additional packages.",
                  font=("Segoe UI", 12, "bold")).pack(pady=6)
        ttk.Label(f, text="Install them from a terminal:",
                  font=("Segoe UI", 10)).pack()
        entry = ttk.Entry(f, width=70)
        entry.insert(0, "pip install --user genevariate[rnaseq] gseapy")
        entry.config(state="readonly")
        entry.pack(pady=6)
        ttk.Label(f, text=f"Import error:\n{err}", foreground="#C62828",
                  font=("Consolas", 9)).pack(pady=10)
        ttk.Button(f, text="Close", command=self.destroy).pack()

    # ───── UI build ────────────────────────────────────────────────────
    def _build_ui(self):
        banner = ttk.Frame(self, padding=(10, 6))
        banner.pack(fill=tk.X)
        ttk.Label(
            banner,
            text=("Load raw RNA-seq counts (CSV/TSV, 10x MTX folder, or .h5ad), "
                  "assign case vs control, then run DESeq2 + GSEA."),
            font=("Segoe UI", 10)).pack(anchor="w")

        body = ttk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True)

        # Left control panel
        left = ttk.Frame(body, padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left, text="1. Count matrix", font=("Segoe UI", 10, "bold")
                  ).pack(anchor="w")
        self._path_var = tk.StringVar()
        pe = ttk.Entry(left, textvariable=self._path_var, width=34)
        pe.pack(anchor="w", pady=2)
        row = ttk.Frame(left); row.pack(anchor="w")
        ttk.Button(row, text="File…", command=self._pick_file).pack(side=tk.LEFT)
        ttk.Button(row, text="10x folder…", command=self._pick_dir
                   ).pack(side=tk.LEFT, padx=4)
        ttk.Button(left, text="Load", command=self._load).pack(anchor="w", pady=4)
        self._info_lbl = ttk.Label(left, text="No matrix loaded.",
                                   foreground="#555")
        self._info_lbl.pack(anchor="w", pady=(0, 8))

        ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=6)
        ttk.Label(left, text="2. Groups", font=("Segoe UI", 10, "bold")
                  ).pack(anchor="w")
        ttk.Label(left, text="Case samples (comma/space sep, or a "
                             "Classified_* value):").pack(anchor="w")
        self._case_txt = tk.Text(left, width=34, height=3)
        self._case_txt.pack(anchor="w", pady=2)
        ttk.Label(left, text="Control samples:").pack(anchor="w")
        self._ctrl_txt = tk.Text(left, width=34, height=3)
        self._ctrl_txt.pack(anchor="w", pady=2)

        ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=6)
        ttk.Label(left, text="3. GSEA libraries (comma sep)").pack(anchor="w")
        self._libs_var = tk.StringVar(
            value=",".join(_MODS["DEFAULT_LIBRARIES"]))
        ttk.Entry(left, textvariable=self._libs_var, width=34).pack(anchor="w")
        self._gsea_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="Run GSEA after DE",
                        variable=self._gsea_var).pack(anchor="w", pady=2)

        self._run_btn = ttk.Button(left, text="Run DESeq2",
                                   command=self._run, style="ToolGreen.TButton")
        self._run_btn.pack(anchor="w", pady=8)
        self._reg_btn = ttk.Button(left, text="Register normalized as platform",
                                   command=self._register, state="disabled")
        self._reg_btn.pack(anchor="w")
        self._status = ttk.Label(left, text="", foreground="#00695C")
        self._status.pack(anchor="w", pady=6)

        if not _MODS.get("_HAS_PYDESEQ2", False):
            ttk.Label(left,
                      text="⚠ pydeseq2 not installed — QC/normalization work,\n"
                           "but DE needs `pip install genevariate[rnaseq]`.",
                      foreground="#B26A00", font=("Segoe UI", 8)
                      ).pack(anchor="w", pady=4)

        # Right results notebook
        self._nb = ttk.Notebook(body)
        self._nb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)
        self._qc_tab = ttk.Frame(self._nb)
        self._de_tab = ttk.Frame(self._nb)
        self._gsea_tab = ttk.Frame(self._nb)
        self._nb.add(self._qc_tab, text="QC")
        self._nb.add(self._de_tab, text="DE table")
        self._nb.add(self._gsea_tab, text="GSEA")
        self._de_tree = self._make_tree(
            self._de_tab, ("gene", "log2FC", "pvalue", "padj", "baseMean"))
        self._gsea_tree = self._make_tree(
            self._gsea_tab, ("Term", "NES", "pval", "fdr", "library"))

    def _make_tree(self, parent, cols):
        tree = ttk.Treeview(parent, columns=cols, show="headings")
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=140, anchor="w")
        vsb = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(fill=tk.BOTH, expand=True)
        return tree

    # ───── File selection / load ───────────────────────────────────────
    def _pick_file(self):
        p = filedialog.askopenfilename(
            parent=self, title="Select count matrix",
            filetypes=[("Counts", "*.csv *.tsv *.txt *.h5ad"), ("All", "*.*")])
        if p:
            self._path_var.set(p)

    def _pick_dir(self):
        p = filedialog.askdirectory(parent=self, title="Select 10x MTX folder")
        if p:
            self._path_var.set(p)

    def _load(self):
        path = self._path_var.get().strip()
        if not path:
            messagebox.showwarning("RNA-seq DE", "Pick a count matrix first.",
                                   parent=self)
            return
        self._set_status("Loading counts…")

        def _worker():
            try:
                counts, meta = _MODS["load_counts"](path)
                self.after(0, lambda: self._on_loaded(counts, meta))
            except Exception:
                tb = traceback.format_exc()
                self.after(0, lambda: self._on_error("Load failed", tb))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_loaded(self, counts, meta):
        self._counts = counts
        self._sample_meta = meta
        n_g, n_s = counts.shape
        cls_cols = []
        if meta is not None:
            cls_cols = [c for c in meta.columns
                        if str(c).startswith("Classified_")]
        self._info_lbl.config(
            text=f"{n_g:,} genes × {n_s:,} samples\n"
                 f"samples: {', '.join(map(str, counts.columns[:6]))}"
                 f"{'…' if n_s > 6 else ''}")
        if cls_cols:
            self._info_lbl.config(
                text=self._info_lbl.cget("text")
                + f"\nmetadata: {', '.join(cls_cols[:4])}")
        self._render_qc()
        self._set_status("Counts loaded. Assign groups and Run DESeq2.")

    # ───── QC plot ──────────────────────────────────────────────────────
    def _render_qc(self):
        for w in self._qc_tab.winfo_children():
            w.destroy()
        try:
            Figure, FigureCanvasTkAgg = _require_mpl()
            qc = _MODS["compute_qc_metrics"](self._counts)
            fig = Figure(figsize=(8, 5), dpi=100)
            for i, col in enumerate(
                    ("library_size", "n_genes_detected", "pct_mito"), start=1):
                ax = fig.add_subplot(1, 3, i)
                ax.bar(range(len(qc)), qc[col].values)
                ax.set_title(col, fontsize=9)
                ax.set_xticks(range(len(qc)))
                ax.set_xticklabels(qc.index, rotation=90, fontsize=6)
            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=self._qc_tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except Exception as exc:
            ttk.Label(self._qc_tab, text=f"QC plot failed: {exc}",
                      foreground="#C62828").pack(pady=10)

    # ───── Group resolution ─────────────────────────────────────────────
    def _resolve_group(self, text: str):
        """A group spec is either an explicit sample list or a single
        Classified_* value that selects samples from the metadata."""
        toks = [t.strip() for t in text.replace(",", " ").split() if t.strip()]
        cols = set(map(str, self._counts.columns))
        explicit = [t for t in toks if t in cols]
        if explicit:
            return explicit
        # Treat the whole spec as a metadata value.
        if self._sample_meta is not None and toks:
            value = text.strip()
            for c in self._sample_meta.columns:
                if str(c).startswith("Classified_"):
                    hit = self._sample_meta.index[
                        self._sample_meta[c].astype(str) == value].tolist()
                    if hit:
                        return [s for s in hit if s in cols]
        return explicit

    # ───── Run DESeq2 ───────────────────────────────────────────────────
    def _run(self):
        if self._counts is None:
            messagebox.showwarning("RNA-seq DE", "Load a count matrix first.",
                                   parent=self)
            return
        if self._busy:
            return
        case = self._resolve_group(self._case_txt.get("1.0", tk.END))
        ctrl = self._resolve_group(self._ctrl_txt.get("1.0", tk.END))
        if len(case) < 2 or len(ctrl) < 2:
            messagebox.showwarning(
                "RNA-seq DE",
                f"Need ≥2 samples per group (case={len(case)}, "
                f"control={len(ctrl)}).", parent=self)
            return
        libs = [x.strip() for x in self._libs_var.get().split(",") if x.strip()]
        run_gsea = self._gsea_var.get()

        import pandas as pd
        design = pd.DataFrame(
            {"condition": (["case"] * len(case)) + (["control"] * len(ctrl))},
            index=case + ctrl)

        self._busy = True
        self._run_btn.config(state="disabled")
        try:
            self.app._acquire_progress()
        except Exception:
            pass

        def _prog(frac, msg):
            try:
                self.app.update_progress(value=int(frac * 100), text=msg)
            except Exception:
                pass

        def _worker():
            try:
                _prog(0.1, "Computing size factors…")
                self._norm = _MODS["cpm_normalize"](self._counts, log=True)
                _prog(0.35, "Running DESeq2…")
                res = _MODS["run_deseq2"](
                    self._counts, design, ("condition", "case", "control"))
                ranked = _MODS["deseq_results_to_ranked"](res)
                self._ranked = ranked
                gsea = None
                if run_gsea:
                    _prog(0.7, "Running GSEA prerank…")
                    gsea = _MODS["run_prerank_gsea"](ranked, gene_sets=libs)
                self.after(0, lambda: self._on_de_done(res, ranked, gsea))
            except Exception:
                tb = traceback.format_exc()
                self.after(0, lambda: self._on_error("DESeq2 failed", tb))
            finally:
                try:
                    self.app._release_progress()
                except Exception:
                    pass

        threading.Thread(target=_worker, daemon=True).start()

    def _on_de_done(self, res, ranked, gsea):
        self._busy = False
        self._run_btn.config(state="normal")
        self._reg_btn.config(state="normal")

        # DE table (top 200 by padj)
        for i in self._de_tree.get_children():
            self._de_tree.delete(i)
        show = res.copy()
        if "padj" in show.columns:
            show = show.sort_values("padj", na_position="last")
        for gene, r in show.head(200).iterrows():
            self._de_tree.insert("", tk.END, values=(
                gene,
                f"{r.get('log2FoldChange', float('nan')):.3f}",
                f"{r.get('pvalue', float('nan')):.2e}",
                f"{r.get('padj', float('nan')):.2e}",
                f"{r.get('baseMean', float('nan')):.1f}"))
        self._nb.select(self._de_tab)

        # GSEA table
        for i in self._gsea_tree.get_children():
            self._gsea_tree.delete(i)
        if gsea is not None and not gsea.empty:
            def _col(df, *names):
                for n in names:
                    if n in df.columns:
                        return n
                return None
            term = _col(gsea, "Term", "Name")
            nes = _col(gsea, "NES", "nes")
            pv = _col(gsea, "NOM p-val", "pval", "p_value")
            fdr = _col(gsea, "FDR q-val", "fdr", "fdr_q_val")
            for _, r in gsea.head(200).iterrows():
                self._gsea_tree.insert("", tk.END, values=(
                    str(r.get(term, ""))[:60] if term else "",
                    r.get(nes, "") if nes else "",
                    r.get(pv, "") if pv else "",
                    r.get(fdr, "") if fdr else "",
                    r.get("library", "")))
        self._set_status(f"DE complete — {len(res):,} genes tested. "
                         f"You can register the normalized matrix as a platform.")

    def _register(self):
        if self._norm is None:
            return
        try:
            df = _MODS["counts_to_platform_df"](self._norm, self._sample_meta)
        except Exception as exc:
            messagebox.showerror("Register", f"Failed: {exc}", parent=self)
            return
        stem = Path(self._path_var.get()).stem or "rnaseq"
        name = f"RNAseq_{stem}_log2cpm"
        if not getattr(self.app, "gpl_datasets", None):
            self.app.gpl_datasets = {}
        self.app.gpl_datasets[name] = df
        try:
            self.app._update_platform_status()
        except Exception:
            pass
        messagebox.showinfo(
            "Registered platform",
            f"Registered '{name}' ({df.shape[0]} samples × "
            f"{df.shape[1] - 1} gene columns).\n\n"
            f"Open Gene Explorer, Compare Distributions, or Label Enrichment "
            f"to analyze it.", parent=self)
        self._set_status(f"Registered platform '{name}'.")

    # ───── helpers ──────────────────────────────────────────────────────
    def _set_status(self, msg: str):
        try:
            self._status.config(text=msg)
        except Exception:
            pass

    def _on_error(self, title: str, tb: str):
        self._busy = False
        try:
            self._run_btn.config(state="normal")
        except Exception:
            pass
        messagebox.showerror(title, tb[-1500:], parent=self)
        self._set_status(f"{title}.")
