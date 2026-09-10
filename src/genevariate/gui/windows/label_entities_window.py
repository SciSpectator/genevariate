"""
GeneVariate - what each extracted label was resolved to.

A normalized label file does not only say "Liver". It says which entry of
which vocabulary the extractor matched, and the three vocabularies it draws on
mean different things: a MeSH heading is a concept that already existed, a
Cellosaurus registration means the sample is a catalogued cell line rather
than a piece of tissue, and a locally minted identifier means nothing
recognised the value and it was grouped on spelling alone.

Without this window those accessions sit unread in the file and every value
looks equally well founded. With it a user can see, before running anything,
that a third of the samples labelled with a tissue are actually cell lines, or
that a whole column resolved to nothing.

Read-only, and built from frames already in memory, so opening it costs
nothing.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import pandas as pd

from genevariate.core import label_entities as le
from genevariate.gui.theme import AERO, UI_FONT, ensure_theme, style_window

_COLS = ("Platform", "Field", "Value", "Accession", "Source", "Stage", "n")
_WIDTHS = {"Platform": 110, "Field": 90, "Value": 260, "Accession": 120,
           "Source": 110, "Stage": 110, "n": 70}

_SOURCE_ORDER = (le.CELLOSAURUS, le.MESH, le.LOCAL, le.UNLINKED)


class LabelEntitiesWindow(tk.Toplevel):
    """The vocabulary entry behind every distinct label value."""

    def __init__(self, parent, platform_labels):
        super().__init__(parent)
        ensure_theme(self)
        style_window(self)
        self.title("Label Entities")
        self.geometry("1040x640")
        self.transient(parent)

        self._table = self._collect(platform_labels)
        self._source_var = tk.StringVar(value="All")
        self._search_var = tk.StringVar()

        self._build_header()
        self._build_controls()
        self._build_table()
        self._render()

    # ------------------------------------------------------------ data ----
    @staticmethod
    def _collect(platform_labels):
        """One entity table across every loaded platform."""
        frames = []
        for plat, df in sorted((platform_labels or {}).items()):
            tbl = le.entity_table(df)
            if tbl.empty:
                continue
            tbl.insert(0, "Platform", plat)
            frames.append(tbl)
        if not frames:
            return pd.DataFrame(columns=_COLS)
        return pd.concat(frames, ignore_index=True)

    def _filtered(self):
        df = self._table
        if df.empty:
            return df
        src = self._source_var.get()
        if src != "All":
            df = df[df["Source"] == src]
        needle = self._search_var.get().strip().lower()
        if needle:
            hit = (df["Value"].astype(str).str.lower().str.contains(needle, regex=False)
                   | df["Accession"].astype(str).str.lower().str.contains(needle, regex=False))
            df = df[hit]
        return df

    # ------------------------------------------------------------- ui ----
    def _build_header(self):
        head = ttk.Frame(self)
        head.pack(fill=tk.X, padx=14, pady=(12, 2))
        ttk.Label(head, text="Label Entities",
                  font=(UI_FONT, 15, "bold")).pack(anchor=tk.W)

        if self._table.empty:
            note = ("No entity links in the loaded labels. Phase 1 and phase 1b "
                    "record the value verbatim; the accessions are written by "
                    "the phase 2 normalization pass.")
        else:
            counts = self._table.groupby("Source")["n"].sum()
            parts = [f"{s} {int(counts[s]):,}" for s in _SOURCE_ORDER
                     if s in counts.index]
            note = "  |  ".join(parts) + "   (samples per vocabulary)"
        ttk.Label(head, text=note, font=(UI_FONT, 9),
                  foreground=AERO["muted"], wraplength=980,
                  justify=tk.LEFT).pack(anchor=tk.W, pady=(2, 0))

    def _build_controls(self):
        bar = ttk.Frame(self)
        bar.pack(fill=tk.X, padx=14, pady=(8, 4))

        ttk.Label(bar, text="Vocabulary:", font=(UI_FONT, 9, "bold")).pack(side=tk.LEFT)
        present = [s for s in _SOURCE_ORDER
                   if not self._table.empty and s in set(self._table["Source"])]
        box = ttk.Combobox(bar, textvariable=self._source_var, state="readonly",
                           width=16, values=["All", *present])
        box.pack(side=tk.LEFT, padx=(6, 16))
        box.bind("<<ComboboxSelected>>", lambda _e: self._render())

        ttk.Label(bar, text="Find:", font=(UI_FONT, 9, "bold")).pack(side=tk.LEFT)
        entry = ttk.Entry(bar, textvariable=self._search_var, width=28)
        entry.pack(side=tk.LEFT, padx=6)
        entry.bind("<KeyRelease>", lambda _e: self._render())

        self._count_lbl = ttk.Label(bar, text="", font=(UI_FONT, 9),
                                    foreground=AERO["muted"])
        self._count_lbl.pack(side=tk.RIGHT)

    def _build_table(self):
        wrap = ttk.Frame(self)
        wrap.pack(fill=tk.BOTH, expand=True, padx=14, pady=(4, 12))

        self.tree = ttk.Treeview(wrap, columns=_COLS, show="headings",
                                 selectmode="browse")
        for c in _COLS:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=_WIDTHS[c], minwidth=60,
                             anchor="center", stretch=True)
        vsb = ttk.Scrollbar(wrap, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # A catalogued cell line is the one row a reader must not mistake for
        # a tissue, so it is the only one that carries its own colour.
        self.tree.tag_configure("cellline", foreground=AERO["green_dark"])
        self.tree.tag_configure("unresolved", foreground=AERO["muted"])
        self.tree.tag_configure("plain", foreground=AERO["text"])

    def _render(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        df = self._filtered()
        for row in df.itertuples(index=False):
            source = getattr(row, "Source", "")
            tag = ("cellline" if source == le.CELLOSAURUS
                   else "unresolved" if source in (le.LOCAL, le.UNLINKED)
                   else "plain")
            self.tree.insert("", tk.END, tags=(tag,), values=(
                row.Platform, row.Field, row.Value, row.Accession,
                source, row.Stage, f"{int(row.n):,}"))
        total = 0 if self._table.empty else len(self._table)
        self._count_lbl.config(text=f"{len(df):,} of {total:,} values")
