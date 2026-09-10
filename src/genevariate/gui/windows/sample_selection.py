"""
GeneVariate - the records behind a selection of plotted points.

A single point opens a sample card; a lassoed cluster asks a different
question. "These twenty samples sit apart from the rest - what do they have
in common?" cannot be answered by the two coordinates that were plotted, so
this window shows the whole row for every selected sample and lets the user
colour those rows by any field: the extracted labels, the study (GSE /
series_id), or any expression column.

Opened from :class:`genevariate.utils.viz_style.PlotInteractor`, so it is
reachable from every plot in the application rather than from the handful of
scatter panels that used to carry their own selector.

Latency is a design constraint, not an afterthought. A selection can land on
a frame with hundreds of thousands of rows and thousands of columns, and the
window has to appear immediately:

* rows are looked up with one vectorised ``isin`` against a cached position
  index, never row by row;
* the tree shows a bounded slice of columns and rows - Export always writes
  the full selection, so nothing is lost by not drawing it;
* recolouring re-tags existing rows instead of rebuilding the tree.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd

from genevariate.gui.theme import AERO, MONO_FONT, UI_FONT, ensure_theme, style_window
from genevariate.utils.viz_style import palette_for, sample_table_of

#: Identity and label fields, shown first because they are what a user reads
#: when asking what a cluster has in common. Everything else follows.
_PRIORITY = ("GSM", "series_id", "GSE", "gpl", "platform", "title",
             "Tissue", "Condition", "Treatment", "Age", "Sex",
             "source_name_ch1", "characteristics_ch1")

#: A platform frame can carry thousands of gene columns. Drawing them all
#: would stall the window for seconds to produce a table nobody can scroll
#: sideways through; Export writes every column regardless.
#: Likewise for rows - a lasso over a dense embedding can take thousands.
#: Past this many distinct values a categorical legend stops being a legend.


def _tint(hex_color: str, amount: float = 0.72) -> str:
    """Mix ``hex_color`` toward white, for a row background that stays legible.

    A saturated fill behind 9pt text is a table nobody can read, so the row
    carries a wash of the colour and the legend swatch carries the colour.
    """
    h = hex_color.lstrip("#")
    rgb = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    mixed = [int(c + (255 - c) * amount) for c in rgb]
    return "#%02X%02X%02X" % tuple(mixed)



def _lim(key):
    """How much this window lists, as the user set it."""
    from genevariate.gui import display_limits
    n = display_limits.get(key)
    return 10 ** 9 if n is None else n


class SampleSelectionWindow(tk.Toplevel):
    """All data for the samples the user selected, coloured by a chosen field."""

    def __init__(self, parent, names, table=None, key_col="GSM",
                 source=""):
        super().__init__(parent)
        ensure_theme(self)
        style_window(self)
        self.title(f"Selected samples ({len(names)})")
        self.geometry("1080x620")
        self.transient(parent)

        self.names = [str(n) for n in names]
        self.frame = self._rows_for(table, key_col)
        self._items = []          # (item_id, colour key) in tree order
        self._colour_of = {}      # colour key -> hex

        self._build_head(source)
        self._build_tree()
        self._build_foot()
        self._fill_tree()
        self._recolour()

    # ── data ─────────────────────────────────────────────────────────
    def _rows_for(self, table, key_col) -> pd.DataFrame:
        """The selected rows, or a bare accession list if no table is known."""
        bare = pd.DataFrame({key_col: self.names})
        if table is None or getattr(table, "empty", True):
            return bare
        if key_col not in table.columns:
            return bare
        try:
            keys = table[key_col].astype(str)
            sub = table[keys.isin(set(self.names))]
            if sub.empty:
                return bare
            # Duplicate accessions across merged sources would multiply rows
            # and misstate n; the first record wins.
            return sub.drop_duplicates(subset=[key_col]).reset_index(drop=True)
        except Exception:
            return bare

    def _colour_fields(self):
        """Columns worth colouring by, cheapest-to-read first.

        A column with one value cannot separate anything and a column with a
        distinct value per sample is a list of identifiers, not a grouping;
        neither is offered.
        """
        out = []
        n = len(self.frame)
        for col in self.frame.columns:
            try:
                k = self.frame[col].nunique(dropna=True)
            except Exception:
                continue
            if k <= 1:
                continue
            if k >= n and not pd.api.types.is_numeric_dtype(self.frame[col]):
                continue
            out.append(col)
        # Labels and study membership are what a user reaches for first.
        head = [c for c in _PRIORITY if c in out]
        return head + [c for c in out if c not in head]

    def _ordered_columns(self):
        cols = list(self.frame.columns)
        head = [c for c in _PRIORITY if c in cols]
        return head + [c for c in cols if c not in head]

    # ── layout ───────────────────────────────────────────────────────
    def _build_head(self, source) -> None:
        head = ttk.Frame(self)
        head.pack(fill=tk.X, padx=14, pady=(12, 4))

        title = f"{len(self.frame)} samples selected"
        if source:
            title += f"  \u2022  {source}"
        ttk.Label(head, text=title,
                  font=(UI_FONT, 13, "bold")).pack(side=tk.LEFT)

        fields = self._colour_fields()
        ttk.Label(head, text="Colour by").pack(side=tk.LEFT, padx=(18, 6))
        self.colour_var = tk.StringVar(value=fields[0] if fields else "(none)")
        box = ttk.Combobox(head, textvariable=self.colour_var, width=24,
                           state="readonly",
                           values=["(none)"] + fields)
        box.pack(side=tk.LEFT)
        box.bind("<<ComboboxSelected>>", lambda _e: self._recolour())

        self.legend = ttk.Frame(self)
        self.legend.pack(fill=tk.X, padx=14, pady=(6, 2))

    def _build_tree(self) -> None:
        body = ttk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True, padx=14, pady=(4, 4))

        self.cols = self._ordered_columns()[:_lim('table_columns')]
        tree = ttk.Treeview(body, columns=self.cols, show="headings",
                            selectmode="browse")
        for c in self.cols:
            tree.heading(c, text=c)
            tree.column(c, width=140, minwidth=80, stretch=True,
                        anchor=tk.CENTER)
        vsb = ttk.Scrollbar(body, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(body, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # Double-click a row for the full single-sample record.
        tree.bind("<Double-1>", self._open_card)
        self.tree = tree

    def _build_foot(self) -> None:
        self.note = ttk.Label(self, text="", foreground=AERO["muted"],
                              font=(UI_FONT, 9, "italic"))
        self.note.pack(anchor=tk.W, padx=14)

        foot = ttk.Frame(self)
        foot.pack(fill=tk.X, padx=14, pady=(4, 12))
        ttk.Button(foot, text="Export selection (CSV)",
                   command=self._export).pack(side=tk.LEFT)
        ttk.Button(foot, text="Close", style="Secondary.TButton",
                   command=self.destroy).pack(side=tk.RIGHT)

    # ── content ──────────────────────────────────────────────────────
    def _fill_tree(self) -> None:
        shown = self.frame.head(_lim('table_rows'))
        # One conversion of the whole block, not one per cell: str() per cell
        # on a wide selection is what used to make a table like this crawl.
        block = shown[self.cols].astype(str).values
        for row in block:
            self._items.append(self.tree.insert("", tk.END, values=tuple(row)))
        notes = []
        if len(self.frame) > _lim('table_rows'):
            notes.append(f"showing first {_lim('table_rows')} of {len(self.frame)} rows")
        n_all = len(self.frame.columns)
        if n_all > len(self.cols):
            notes.append(f"showing {len(self.cols)} of {n_all} columns")
        if notes:
            notes.append("Export writes the full selection")
        self.note.configure(text="  \u2022  ".join(notes))

    def _recolour(self) -> None:
        field = self.colour_var.get()
        for child in self.legend.winfo_children():
            child.destroy()
        if field in ("(none)", "") or field not in self.frame.columns:
            for item in self._items:
                self.tree.item(item, tags=())
            return

        values = self.frame[field].head(len(self._items))
        keys, colours, legend = self._palette(values, field)
        for key, hexc in colours.items():
            tag = f"c{abs(hash(key)) % 10 ** 8}"
            self.tree.tag_configure(tag, background=_tint(hexc),
                                    foreground=AERO["text"])
            colours[key] = (hexc, tag)
        for item, key in zip(self._items, keys):
            self.tree.item(item, tags=(colours[key][1],))

        for label, hexc, count in legend:
            cell = ttk.Frame(self.legend)
            cell.pack(side=tk.LEFT, padx=(0, 12))
            swatch = tk.Canvas(cell, width=12, height=12, highlightthickness=1,
                               highlightbackground=AERO["border"], bg=hexc)
            swatch.pack(side=tk.LEFT, pady=1)
            ttk.Label(cell, text=f" {label}  ({count})",
                      font=(UI_FONT, 9)).pack(side=tk.LEFT)

    def _palette(self, values, field):
        """Map each row to a colour, and summarise the mapping for the legend.

        A field with a handful of values gets one hue each. A numeric field
        with many gets a viridis ramp, because forty arbitrary hues over a
        continuum reads as noise where a ramp reads as an ordering.
        """
        text = values.astype(str)
        counts = text.value_counts()
        numeric = pd.to_numeric(values, errors="coerce")
        continuous = (pd.api.types.is_numeric_dtype(values)
                      or numeric.notna().mean() > 0.9)

        if len(counts) > _lim('legend_groups') and continuous:
            import matplotlib as mpl
            import matplotlib.colors as mcolors
            finite = numeric[np.isfinite(numeric)]
            lo = float(finite.min()) if len(finite) else 0.0
            hi = float(finite.max()) if len(finite) else 1.0
            span = (hi - lo) or 1.0
            cmap = mpl.colormaps["viridis"]
            keys, colours = [], {}
            for v in numeric:
                key = "n/a" if not np.isfinite(v) else f"{float(v):.6g}"
                keys.append(key)
                if key not in colours:
                    frac = 0.0 if key == "n/a" else (float(v) - lo) / span
                    colours[key] = (AERO["muted"] if key == "n/a"
                                    else mcolors.to_hex(cmap(frac)))
            legend = [(f"{field} {lo:.4g}", mcolors.to_hex(cmap(0.0)), len(finite)),
                      (f"{field} {hi:.4g}", mcolors.to_hex(cmap(1.0)), len(finite))]
            return keys, colours, legend

        top = list(counts.index[:_lim('legend_groups')])
        pal = palette_for(len(top), "discrete")
        colours = dict(zip(top, pal))
        other = AERO["muted"]
        keys = [v if v in colours else "\u2026 other" for v in text]
        if "\u2026 other" in keys:
            colours["\u2026 other"] = other
        legend = [(v, colours[v], int(counts[v])) for v in top]
        n_other = len(counts) - len(top)
        if n_other > 0:
            legend.append((f"\u2026 {n_other} more", other,
                           int(counts.iloc[len(top):].sum())))
        return keys, colours, legend

    # ── actions ──────────────────────────────────────────────────────
    def _open_card(self, _event=None) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        from genevariate.gui.windows.sample_card import show_sample_card
        idx = self._items.index(sel[0])
        key = self.cols[0]
        show_sample_card(self, str(self.frame.iloc[idx][key]), self.frame)

    def _export(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")], parent=self,
            initialfile=f"selection_{len(self.frame)}_samples.csv")
        if not path:
            return
        try:
            self.frame.to_csv(path, index=False)
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc), parent=self)
            return
        messagebox.showinfo(
            "Exported",
            f"{len(self.frame)} samples \u00d7 {len(self.frame.columns)} columns",
            parent=self)


def show_selection(fig, names, ax=None, table=None, key_col="GSM",
                   parent=None, source=""):
    """Open the selection window for ``names``, over the table ``fig`` carries.

    Returns the window, or ``None`` - a drag on a plot must never be able to
    take the application down.
    """
    if not names:
        return None
    if table is None:
        table, registered_key = sample_table_of(fig)
        if registered_key:
            key_col = registered_key
    if callable(table):
        try:
            table = table()
        except Exception:
            table = None
    if parent is None:
        try:
            parent = fig.canvas.get_tk_widget().winfo_toplevel()
        except Exception:
            parent = None
    if parent is None:
        return None
    if not source and ax is not None:
        source = (ax.get_title() or "").strip()
    try:
        return SampleSelectionWindow(parent, names, table, key_col, source)
    except Exception:
        return None
