"""
Activity-inference window — TF (CollecTRI) & pathway (PROGENy) activities via
decoupleR.

Placed under *Tools → Activity Inference…* (an advanced/optional analysis, not a
main toolbar button). Pick a loaded platform, choose TF or pathway activity, and
score each sample with decoupleR's ULM. Results render as a ranked table + a
markdown report.

Structure mirrors ``gui/windows/rnaseq_de.py``: lazy ``_try_import`` with a
``_render_missing_deps`` fallback (decoupler is optional), and a worker thread
that marshals UI updates via ``self.after(0, ...)`` and drives the shared
progress bar through the app's ``_acquire_progress`` / ``update_progress`` /
``_release_progress`` API.
"""
from __future__ import annotations

import threading
import traceback
from typing import Optional

import tkinter as tk
from tkinter import ttk, messagebox


def _try_import():
    try:
        from genevariate.core.analysis.activity import run_activity, _HAS_DECOUPLER
        return dict(run_activity=run_activity, _HAS_DECOUPLER=_HAS_DECOUPLER)
    except Exception as exc:  # pragma: no cover - only when deps missing
        return {"_error": str(exc)}


_MODS = _try_import()


class ActivityInferenceWindow(tk.Toplevel):
    """TF / pathway activity inference over a loaded platform (decoupleR)."""

    def __init__(self, parent):
        super().__init__(parent)
        self.app = parent
        self.title("Activity Inference — TF (CollecTRI) / pathway (PROGENy)")
        self.geometry("820x620")
        try:
            self.transient(parent)
        except Exception:
            pass
        self._busy = False

        if "_error" in _MODS:
            self._render_missing_deps(_MODS["_error"])
            return
        self._build_ui()

    # ───── Missing-deps fallback ───────────────────────────────────────
    def _render_missing_deps(self, err: str):
        f = ttk.Frame(self, padding=20)
        f.pack(fill=tk.BOTH, expand=True)
        ttk.Label(f, text="Activity inference requires the decoupler package.",
                  font=("Segoe UI", 12, "bold")).pack(pady=6)
        ttk.Label(f, text="Install it from a terminal:",
                  font=("Segoe UI", 10)).pack()
        entry = ttk.Entry(f, width=60)
        entry.insert(0, "pip install --user decoupler")
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
            text=("Score which regulators are active per sample: transcription "
                  "factors (CollecTRI) or pathways (PROGENy), via decoupleR ULM."),
            font=("Segoe UI", 10)).pack(anchor="w")

        ctl = ttk.Frame(self, padding=(10, 6))
        ctl.pack(fill=tk.X)

        ttk.Label(ctl, text="Platform:").grid(row=0, column=0, sticky="w",
                                              padx=(0, 6), pady=4)
        self._plat_var = tk.StringVar()
        plats = list(getattr(self.app, "gpl_datasets", {}) or {})
        self._plat_combo = ttk.Combobox(ctl, textvariable=self._plat_var,
                                        values=plats, width=28, state="readonly")
        if plats:
            self._plat_combo.current(0)
        self._plat_combo.grid(row=0, column=1, sticky="w", pady=4)

        ttk.Label(ctl, text="Kind:").grid(row=0, column=2, sticky="w",
                                          padx=(16, 6), pady=4)
        self._kind_var = tk.StringVar(value="tf")
        ttk.Radiobutton(ctl, text="TF (CollecTRI)", variable=self._kind_var,
                        value="tf").grid(row=0, column=3, sticky="w", pady=4)
        ttk.Radiobutton(ctl, text="Pathway (PROGENy)", variable=self._kind_var,
                        value="pathway").grid(row=0, column=4, sticky="w", pady=4)

        ttk.Label(ctl, text="Organism:").grid(row=1, column=0, sticky="w",
                                              padx=(0, 6), pady=4)
        self._org_var = tk.StringVar(value="human")
        ttk.Entry(ctl, textvariable=self._org_var, width=14).grid(
            row=1, column=1, sticky="w", pady=4)

        self._run_btn = ttk.Button(ctl, text="Run", command=self._run)
        self._run_btn.grid(row=1, column=3, columnspan=2, sticky="w", pady=4)

        # Results: ranked table + report
        body = ttk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=(4, 10))

        cols = ("source", "mean_activity", "abs_activity")
        self._tree = ttk.Treeview(body, columns=cols, show="headings", height=12)
        for c, w in zip(cols, (240, 140, 140)):
            self._tree.heading(c, text=c)
            self._tree.column(c, width=w, anchor="w")
        tvsb = ttk.Scrollbar(body, orient=tk.VERTICAL,
                             command=self._tree.yview)
        self._tree.configure(yscrollcommand=tvsb.set)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tvsb.pack(side=tk.LEFT, fill=tk.Y)

        self._report = tk.Text(body, width=42, wrap=tk.WORD,
                               font=("Consolas", 9), state=tk.DISABLED)
        self._report.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))

    # ───── Run ──────────────────────────────────────────────────────────
    def _run(self):
        if self._busy:
            return
        key = self._plat_var.get().strip()
        plats = getattr(self.app, "gpl_datasets", {}) or {}
        if key not in plats:
            messagebox.showwarning("Activity Inference",
                                   "Load and select a platform first.",
                                   parent=self)
            return
        df = plats[key]
        kind = self._kind_var.get()
        organism = self._org_var.get().strip() or "human"

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
                _prog(0.2, f"Fetching {kind} network + scoring…")
                res = _MODS["run_activity"](df, kind=kind, organism=organism)
                self.after(0, lambda: self._on_done(res))
            except Exception:
                tb = traceback.format_exc()
                self.after(0, lambda: self._on_error(tb))
            finally:
                try:
                    self.app._release_progress()
                except Exception:
                    pass

        threading.Thread(target=_worker, daemon=True).start()

    def _on_done(self, res):
        self._busy = False
        self._run_btn.config(state="normal")
        for i in self._tree.get_children():
            self._tree.delete(i)
        ranked = res.get("ranked")
        if ranked is not None:
            for name, row in ranked.iterrows():
                self._tree.insert(
                    "", tk.END,
                    values=(name, f"{row['mean_activity']:+.4f}",
                            f"{row['abs_activity']:.4f}"))
        self._report.configure(state=tk.NORMAL)
        self._report.delete("1.0", tk.END)
        self._report.insert(tk.END, res.get("report", "") or "")
        self._report.configure(state=tk.DISABLED)

    def _on_error(self, tb: str):
        self._busy = False
        self._run_btn.config(state="normal")
        messagebox.showerror("Activity Inference", tb, parent=self)
