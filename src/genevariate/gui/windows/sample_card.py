"""
GeneVariate - the record behind a single plotted point.

Clicking a sample in any plot opens this: everything known about that one
GSM, in one place. It exists because a scatter can only ever show two numbers
about a sample, while the question a user actually has when a point looks odd
is "what *is* this one?" - which study, which tissue, which treatment, and how
its expression compares with the region it was drawn in.

The window is deliberately read-only and cheap to open. It takes an already
loaded metadata frame rather than fetching anything, so clicking a point never
blocks on the network.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Optional, Sequence

from genevariate.gui.theme import AERO, MONO_FONT, UI_FONT, ensure_theme, style_window

# Fields that describe the sample itself, shown first and in this order.
# Everything else is still shown, just below these.
_PRIORITY = ("GSM", "series_id", "title", "source_name_ch1", "organism_ch1",
             "Tissue", "Condition", "Treatment", "Age", "Sex",
             "characteristics_ch1")

# Bulk text fields that would otherwise push the useful rows off the screen.
_LONG = ("characteristics_ch1", "title", "description", "treatment_protocol_ch1")

_SKIP = {"_platform"}


class SampleCardWindow(tk.Toplevel):
    """Read-only detail view for one sample."""

    def __init__(self, parent, gsm: str, metadata_row=None,
                 expression: Optional[Sequence] = None,
                 region_label: Optional[str] = None):
        super().__init__(parent)
        ensure_theme(self)
        style_window(self)
        self.title(f"Sample {gsm}")
        self.geometry("620x560")
        self.transient(parent)

        head = ttk.Frame(self)
        head.pack(fill=tk.X, padx=14, pady=(12, 4))
        ttk.Label(head, text=str(gsm),
                  font=(UI_FONT, 15, "bold")).pack(anchor=tk.W)
        if region_label:
            ttk.Label(head, text=f"in region {region_label}",
                      font=(UI_FONT, 10), foreground=AERO["muted"]).pack(anchor=tk.W)

        if expression is not None:
            self._expression_line(expression)

        body = ttk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True, padx=14, pady=(8, 4))
        tree = ttk.Treeview(body, columns=("field", "value"), show="headings",
                            selectmode="browse")
        tree.heading("field", text="Field")
        tree.heading("value", text="Value")
        tree.column("field", width=190, stretch=False, anchor=tk.W)
        tree.column("value", width=380, stretch=True, anchor=tk.W)
        vsb = ttk.Scrollbar(body, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.tree = tree

        for field, value in self._rows(gsm, metadata_row):
            tree.insert("", tk.END, values=(field, value))
        # A cell can hold a paragraph of free text that the column cannot show;
        # clicking the row spells it out in full underneath.
        tree.bind("<<TreeviewSelect>>", self._show_full_value)

        self.detail = tk.Text(self, height=4, wrap=tk.WORD, relief=tk.FLAT,
                              bg=AERO["panel"], fg=AERO["text"],
                              font=(MONO_FONT, 9), padx=8, pady=6)
        self.detail.pack(fill=tk.X, padx=14, pady=(0, 6))
        self.detail.configure(state=tk.DISABLED)

        ttk.Button(self, text="Close", style="Secondary.TButton",
                   command=self.destroy).pack(side=tk.RIGHT, padx=14, pady=(0, 12))

    # ── content ──────────────────────────────────────────────────────
    def _expression_line(self, expression) -> None:
        try:
            values = [float(v) for v in expression]
        except (TypeError, ValueError):
            return
        if not values:
            return
        text = "  ".join(f"{v:.3f}" for v in values[:8])
        if len(values) > 8:
            text += f"  (+{len(values) - 8} more)"
        row = ttk.Frame(self)
        row.pack(fill=tk.X, padx=14, pady=(6, 0))
        ttk.Label(row, text="Expression:", font=(UI_FONT, 9, "bold"),
                  foreground=AERO["muted"]).pack(side=tk.LEFT)
        ttk.Label(row, text=text, font=(MONO_FONT, 10),
                  foreground=AERO["accent_dark"]).pack(side=tk.LEFT, padx=(6, 0))

    def _rows(self, gsm, metadata_row):
        """Priority fields first, then whatever else the record carries."""
        if metadata_row is None:
            return [("GSM", str(gsm)),
                    ("", "No metadata loaded for this sample.")]
        try:
            record = dict(metadata_row)
        except (TypeError, ValueError):
            return [("GSM", str(gsm))]

        rows, seen = [], set()
        for field in _PRIORITY:
            if field in record:
                seen.add(field)
                rows.append((field, self._clean(record[field], field)))
        for field in sorted(record):
            if field in seen or field in _SKIP:
                continue
            rows.append((field, self._clean(record[field], field)))
        return [(f, v) for f, v in rows if v not in ("", "nan", "None")]

    @staticmethod
    def _clean(value, field) -> str:
        text = str(value).replace("\n", " ").replace("\r", " ").strip()
        if field in _LONG and len(text) > 300:
            text = text[:300] + " ..."
        return text

    def _show_full_value(self, _event=None) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        field, value = self.tree.item(selection[0], "values")
        self.detail.configure(state=tk.NORMAL)
        self.detail.delete("1.0", tk.END)
        self.detail.insert("1.0", f"{field}\n{value}")
        self.detail.configure(state=tk.DISABLED)


def show_sample_card(parent, gsm, metadata_df=None, expression=None,
                     region_label=None):
    """Open the card for ``gsm``, looking its row up in ``metadata_df``.

    Returns the window, or ``None`` if it could not be opened - a click on a
    plot must never be able to take the application down.
    """
    row = None
    try:
        if metadata_df is not None and not metadata_df.empty:
            key = "GSM" if "GSM" in metadata_df.columns else metadata_df.columns[0]
            match = metadata_df[metadata_df[key].astype(str) == str(gsm)]
            if not match.empty:
                row = match.iloc[0]
    except Exception:
        row = None
    try:
        return SampleCardWindow(parent, gsm, row, expression, region_label)
    except Exception:
        return None
