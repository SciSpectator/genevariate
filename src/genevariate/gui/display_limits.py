"""The Display limits dialog. The settings themselves are program-wide
and live in :mod:`genevariate.utils.display_limits`; this adds the form
the user fills in and re-exports the rest so existing callers still work.
"""
from __future__ import annotations

from typing import Any, Dict, Optional  # noqa: F401

from genevariate.utils.display_limits import (  # noqa: F401
    ANALYSIS, DISPLAY, cap_note, get, head, head_series, reset, set_all,
    _all,
)


def open_dialog(parent, on_change=None) -> None:
    """Let the user raise or remove every limit, then redraw.

    Blank means no limit. The two groups are kept apart on the form because
    changing one of them changes the picture and changing the other changes
    the numbers.
    """
    import tkinter as tk
    from tkinter import ttk, messagebox

    from genevariate.gui.theme import ensure_theme, labelframe, style_window

    win = tk.Toplevel(parent)
    ensure_theme(win)
    style_window(win)
    win.title("GeneVariate - Display limits")
    win.transient(parent)
    win.resizable(False, False)

    intro = ttk.Label(
        win, justify=tk.LEFT, style='Hint.TLabel',
        text=("How much of each result is drawn or listed. Leave a box empty "
              "to remove the limit and show everything.\nNo limit here ever "
              "changes a statistic except the two marked below, which decide "
              "what is tested."))
    intro.pack(anchor=tk.W, padx=12, pady=(10, 6))

    entries: Dict[str, Any] = {}

    def _section(title, spec, caution=None):
        box = labelframe(win, title)
        box.pack(fill=tk.X, padx=12, pady=(0, 8))
        if caution:
            c = ttk.Label(box, text=caution, justify=tk.LEFT,
                          style='Caution.TLabel', wraplength=520)
            c.grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=8, pady=(4, 6))
        base = 1 if caution else 0
        for i, (key, (default, what, where)) in enumerate(spec.items()):
            ttk.Label(box, text=what, wraplength=300, justify=tk.LEFT).grid(
                row=base + i, column=0, sticky=tk.W, padx=(8, 6), pady=2)
            var = tk.StringVar(value="" if get(key) is None else str(get(key)))
            ttk.Entry(box, textvariable=var, width=8).grid(
                row=base + i, column=1, sticky=tk.W, pady=2)
            # Wrapped, or a long sentence here pushes the dialog past the
            # screen and takes the buttons with it.
            ttk.Label(box, text=where, style='Footnote.TLabel',
                      wraplength=330, justify=tk.LEFT).grid(
                row=base + i, column=2, sticky=tk.W, padx=(8, 8), pady=2)
            entries[key] = var

    _section("How much is shown", DISPLAY)
    _section("What is tested", ANALYSIS,
             caution=("These two decide which label values enter the test, so "
                      "raising one enlarges the multiple-testing family and "
                      "every q value moves. Change them deliberately."))

    bar = ttk.Frame(win)
    bar.pack(fill=tk.X, padx=12, pady=(2, 12))

    def _apply():
        values: Dict[str, Optional[int]] = {}
        for key, var in entries.items():
            raw = var.get().strip()
            if not raw:
                values[key] = None
                continue
            try:
                n = int(raw)
            except ValueError:
                messagebox.showwarning(
                    "Display limits",
                    f"{_all()[key][1]}: '{raw}' is not a whole number. "
                    "Leave it empty to show everything.", parent=win)
                return
            if n < 1:
                messagebox.showwarning(
                    "Display limits",
                    f"{_all()[key][1]}: a limit has to be at least 1. "
                    "Leave it empty to show everything.", parent=win)
                return
            values[key] = n
        set_all(values)
        win.destroy()
        if on_change:
            on_change()

    def _show_all():
        for var in entries.values():
            var.set("")

    def _defaults():
        for key, var in entries.items():
            d = _all()[key][0]
            var.set("" if d is None else str(d))

    ttk.Button(bar, text="Apply", style="Action.TButton",
               command=_apply).pack(side=tk.RIGHT, padx=4)
    ttk.Button(bar, text="Cancel", command=win.destroy).pack(side=tk.RIGHT, padx=4)
    ttk.Button(bar, text="Show everything", command=_show_all).pack(side=tk.LEFT)
    ttk.Button(bar, text="Defaults", command=_defaults).pack(side=tk.LEFT, padx=6)
    win.update_idletasks()
    # Centred on the window that opened it, and never wider than the screen.
    w, h = win.winfo_reqwidth(), win.winfo_reqheight()
    sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
    w, h = min(w, sw - 40), min(h, sh - 60)
    win.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 3)}")
