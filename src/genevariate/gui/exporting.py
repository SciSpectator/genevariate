"""Saving whatever is on screen, from anywhere in the program.

Every table and every plot GeneVariate draws is a result, and a result the user
cannot take away is of no use to them. Rather than hand-wiring an export button
onto each of the several dozen tables scattered across the windows - and
forgetting the next one somebody adds - this module gives every
``ttk.Treeview`` a Save button, a right-click menu and Ctrl+S, from one place:

    install_table_export()      once, before any window is built

Figures are handled by :func:`save_figure`, which the plot toolbars and the
"Save plot" buttons call.

Nothing here computes or reformats anything: a table is written exactly as it is
displayed, so an exported file and the screen always agree.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pandas as pd

__all__ = [
    "dataframe_from_tree", "save_dataframe", "save_tree", "save_figure",
    "attach_table_export", "attach_figure_export", "install_table_export",
]

_TABLE_TYPES = [("CSV", "*.csv"), ("Tab-separated", "*.tsv"),
                ("Excel", "*.xlsx")]
_FIGURE_TYPES = [("PNG image", "*.png"), ("PDF", "*.pdf"),
                 ("SVG", "*.svg"), ("TIFF", "*.tif")]


def _safe(text, default="table"):
    keep = [c if (c.isalnum() or c in "-_") else "_" for c in str(text).strip()]
    out = "".join(keep).strip("_")
    return out[:60] or default


# ── reading a table back out of the widget ─────────────────────────────
def dataframe_from_tree(tree) -> pd.DataFrame:
    """Everything the Treeview is showing, as a DataFrame.

    Column headings become column names, and if the widget shows the tree
    column (a grouped table) its label is kept as the first column so the
    grouping is not lost. Child rows are flattened in display order.
    """
    cols = list(tree["columns"] or ())
    names = []
    for c in cols:
        try:
            names.append(str(tree.heading(c).get("text") or c))
        except Exception:
            names.append(str(c))

    show = str(tree.cget("show"))
    with_tree_col = "tree" in show
    if with_tree_col:
        names.insert(0, str(tree.heading("#0").get("text") or "Item"))

    rows = []

    def walk(node):
        for iid in tree.get_children(node):
            vals = list(tree.item(iid, "values"))
            vals += [""] * (len(cols) - len(vals))
            if with_tree_col:
                vals.insert(0, tree.item(iid, "text"))
            rows.append(vals[:len(names)])
            walk(iid)

    walk("")
    return pd.DataFrame(rows, columns=names)


# ── writing ────────────────────────────────────────────────────────────
def save_dataframe(df: pd.DataFrame, parent=None, name="table") -> str | None:
    """Ask for a path and write *df* there. Returns the path, or None."""
    if df is None or df.empty:
        messagebox.showinfo("Nothing to save",
                            "This table is empty.", parent=parent)
        return None
    path = filedialog.asksaveasfilename(
        parent=parent, title="Save table",
        defaultextension=".csv", filetypes=_TABLE_TYPES,
        initialfile=f"{_safe(name)}.csv")
    if not path:
        return None
    try:
        low = path.lower()
        if low.endswith(".xlsx"):
            df.to_excel(path, index=False)
        elif low.endswith((".tsv", ".txt")):
            df.to_csv(path, sep="\t", index=False)
        else:
            df.to_csv(path, index=False)
    except Exception as exc:
        messagebox.showerror("Save failed", str(exc), parent=parent)
        return None
    messagebox.showinfo("Saved", f"{len(df):,} row(s) written to:\n{path}",
                        parent=parent)
    return path


def save_tree(tree, name=None) -> str | None:
    """Save the contents of a Treeview.

    A tab that caps what it displays attaches the frame it is a view of, and
    this route has to prefer it for the same reason ``export_window`` does:
    scraping the widget returns the display strings, so a table shortened to
    fit a column or rounded to fit a cell would be written out shortened and
    rounded. The two buttons wrote different files until this looked here too.
    """
    parent = tree.winfo_toplevel()
    if name is None:
        name = tree.winfo_toplevel().title()
    df = getattr(tree, "_export_frame", None)
    if df is None:
        df = dataframe_from_tree(tree)
    return save_dataframe(df, parent=parent, name=name)


def save_figure(fig, parent=None, name="plot") -> str | None:
    """Ask for a path and write *fig* there, vector formats included."""
    if fig is None:
        return None
    path = filedialog.asksaveasfilename(
        parent=parent, title="Save plot",
        defaultextension=".png", filetypes=_FIGURE_TYPES,
        initialfile=f"{_safe(name, 'plot')}.png")
    if not path:
        return None
    try:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    except Exception as exc:
        messagebox.showerror("Save failed", str(exc), parent=parent)
        return None
    messagebox.showinfo("Saved", f"Plot written to:\n{path}", parent=parent)
    return path


# ── wiring it onto the widgets ─────────────────────────────────────────
def _menu_for(tree):
    menu = tk.Menu(tree, tearoff=0)
    menu.add_command(label="Save table as\u2026",
                     command=lambda: save_tree(tree))
    menu.add_command(label="Copy selected rows",
                     command=lambda: _copy_selection(tree))
    return menu


def _copy_selection(tree):
    sel = tree.selection() or tree.get_children("")
    lines = ["\t".join(str(v) for v in tree.item(i, "values")) for i in sel]
    if not lines:
        return
    tree.clipboard_clear()
    tree.clipboard_append("\n".join(lines))


def attach_table_export(tree, name=None, label="\u2913 Save table\u2026"):
    """Add a Save button, Ctrl+S and a right-click menu to one Treeview.

    The button goes into the frame that holds the table, below it. Both
    geometry managers are handled: with ``pack`` the bar is inserted *before*
    the table in the packing order, otherwise an already-expanded table would
    leave it no room; with ``grid`` it takes a new full-width row underneath.
    A table laid out with ``place`` is left alone - the menu and Ctrl+S still
    work, so nothing is unreachable.
    """
    if getattr(tree, "_export_attached", False):
        return None
    tree._export_attached = True

    menu = _menu_for(tree)

    def popup(event):
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
    tree.bind("<Button-3>", popup, add="+")
    tree.bind("<Control-s>", lambda e: (save_tree(tree, name), "break")[1],
              add="+")

    holder = tree.master
    manager = tree.winfo_manager()
    try:
        bar = ttk.Frame(holder)
        if manager == "grid":
            ncols, nrows = holder.grid_size()
            bar.grid(row=nrows, column=0, columnspan=max(1, ncols),
                     sticky="w", pady=(4, 0))
        elif manager == "pack":
            bar.pack(before=tree, side=tk.BOTTOM, fill=tk.X, pady=(4, 0))
        else:
            bar.destroy()
            return None
        btn = ttk.Button(bar, text=label, style="Secondary.TButton",
                         command=lambda: save_tree(tree, name))
        btn.pack(side=tk.LEFT)
        tree._export_button = btn
        return btn
    except Exception:
        return None


def attach_figure_export(container, fig, name=None,
                         label="\u2913 Save plot\u2026"):
    """Add a Save button for *fig* into *container* (packed frames only)."""
    try:
        bar = ttk.Frame(container)
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Button(bar, text=label, style="Secondary.TButton",
                   command=lambda: save_figure(
                       fig, parent=container.winfo_toplevel(),
                       name=name or container.winfo_toplevel().title())
                   ).pack(side=tk.LEFT, padx=4, pady=2)
        return bar
    except Exception:
        return None


def _table_name(tree, fallback="table"):
    """A filename for a table, taken from the tab or frame it lives in."""
    w = tree
    for _ in range(8):
        parent = getattr(w, "master", None)
        if parent is None:
            break
        if isinstance(parent, ttk.Notebook):
            try:
                return _safe(parent.tab(w, "text"), fallback)
            except Exception:
                pass
        if isinstance(parent, (tk.LabelFrame, ttk.LabelFrame)):
            try:
                text = parent.cget("text")
                if text:
                    return _safe(text, fallback)
            except Exception:
                pass
        w = parent
    return fallback


def iter_tables(root):
    """Every Treeview under *root*, as ``(name, tree)`` in creation order."""
    found = []

    def walk(w):
        for child in w.winfo_children():
            if isinstance(child, ttk.Treeview):
                found.append((_table_name(child), child))
            walk(child)

    walk(root)
    seen = {}
    out = []
    for name, tree in found:
        seen[name] = seen.get(name, 0) + 1
        out.append((name if seen[name] == 1 else f"{name}_{seen[name]}", tree))
    return out


def iter_texts(root):
    """Every non-empty ``tk.Text`` under *root*, as ``(name, text)``.

    A window's written interpretation is a result too, and it lives in a Text
    rather than a Treeview, so an export built only on tables would drop it.
    """
    found = []

    def walk(w):
        for child in w.winfo_children():
            if isinstance(child, tk.Text):
                found.append((_table_name(child, "notes"), child))
            walk(child)

    walk(root)
    seen = {}
    out = []
    for name, widget in found:
        seen[name] = seen.get(name, 0) + 1
        out.append((name if seen[name] == 1 else f"{name}_{seen[name]}", widget))
    return out


def export_window(win, out_dir, figures=None, prefix=""):
    """Write every table and every figure of a window into *out_dir*.

    Used by the "Export All" buttons so that saving a window means saving all
    of it - the tables shown in each tab as well as the plots - instead of the
    handful of frames whoever wrote the button happened to think of.
    Returns the list of paths written.
    """
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for key, fig in (figures or {}).items():
        if fig is None:
            continue
        # PNG at 300 dpi for reading, PDF for submission: journals want line
        # art as vector, and a raster figure cannot be rescaled afterwards.
        for ext in ("png", "pdf"):
            path = out / f"{prefix}{_safe(key, 'plot')}.{ext}"
            try:
                fig.savefig(path, dpi=300, bbox_inches="tight")
                written.append(path)
            except Exception:
                pass
    for name, tree in iter_tables(win):
        # A tab that caps what it displays attaches the frame it is a view of,
        # so the export is the whole result and not the visible corner of it.
        # Scraping the widget is the fallback for tables built row by row.
        df = getattr(tree, "_export_frame", None)
        if df is None:
            df = dataframe_from_tree(tree)
        if df.empty:
            continue
        path = out / f"{prefix}{name}.csv"
        try:
            df.to_csv(path, index=False)
            written.append(path)
        except Exception:
            pass
    for name, widget in iter_texts(win):
        try:
            body = widget.get("1.0", tk.END).strip()
        except Exception:
            continue
        if not body:
            continue
        path = out / f"{prefix}{name}.txt"
        try:
            path.write_text(body, encoding="utf-8")
            written.append(path)
        except Exception:
            pass
    return written


_PATCHED = False


def install_table_export():
    """Make every Treeview built from now on exportable.

    Wrapping the constructor is what keeps the promise "every table in the
    program can be saved" true for tables that do not exist yet, instead of
    depending on whoever adds the next one remembering to wire a button.
    The button is attached on idle, once the widget has actually been laid
    out - before that its geometry manager is not yet known.
    """
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True
    original = ttk.Treeview.__init__

    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        try:
            self.after_idle(lambda: _attach_if_alive(self))
        except Exception:
            pass

    ttk.Treeview.__init__ = __init__


def _attach_if_alive(tree):
    try:
        if tree.winfo_exists() and not getattr(tree, "_no_export", False):
            attach_table_export(tree)
    except Exception:
        pass
