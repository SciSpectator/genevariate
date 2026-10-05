"""The Frutiger Aero palette every GeneVariate window draws from."""

import tkinter as tk
from tkinter import ttk

AERO = {
    "bg":            "#FFFFFF",
    "bg_top":        "#EAF6FE",   # sky gradient top
    "bg_mid":        "#FFFFFF",
    "bg_bot":        "#F4FBF2",   # nature gradient bottom
    "panel":         "#FFFFFF",
    "panel_top":     "#FCFEFF",   # glossy highlight
    "panel_bot":     "#EDF7FF",   # reflected-sky base
    "border":        "#C5DAEA",
    "border_soft":   "#E0EEF7",
    "text":          "#0E2A45",
    "muted":         "#5F7D95",
    # sky (primary)
    "accent":        "#1E90E0",
    "accent_dark":   "#0A5B9A",
    "accent_light":  "#B9E3FA",
    "sky_top":       "#6DC8F3",
    "sky_bot":       "#2B8BD6",
    # nature (secondary)
    "green":         "#4CAF50",
    "green_dark":    "#2E7D32",
    "green_light":   "#C9EFC7",
    "leaf_top":      "#8FD98F",
    "leaf_bot":      "#3FAA45",
    # states
    "success":       "#2E7D32",
    "danger":        "#C0392B",
    "danger_hover":  "#9E2B1F",
    "warn":          "#E67E22",
    "hover_sky":     "#E8F5FD",
    "pressed_sky":   "#BFE1F6",
    "glass_hilite":  "#F4FAFE",
    "table_rule":    "#000000",
}

UI_FONT = "Segoe UI"
MONO_FONT = "Consolas"


def _to_rgb(hex_color):
    """'#RRGGBB' -> (r, g, b)."""
    h = hex_color.lstrip('#')
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def style_window(win, bg=None):
    """Give a ``tk.Toplevel`` the application background."""
    try:
        win.configure(bg=bg or AERO["bg_top"])
    except Exception:
        pass          # some Toplevels are override-redirect and refuse bg
    return win


def choice_dialog(parent, title, options, intro=""):
    """Ask the user to pick one of several tools, in a themed modal window.

    Returns the key of the card the user picked, or ``None``.
    """
    ensure_theme(parent)
    win = tk.Toplevel(parent)
    win.title(title)
    style_window(win)
    win.transient(parent)
    win.resizable(False, False)
    picked = {"key": None}

    outer = ttk.Frame(win, padding=(18, 16, 18, 14))
    outer.pack(fill=tk.BOTH, expand=True)
    ttk.Label(outer, text=title, font=(UI_FONT, 13, "bold"),
              foreground=AERO["accent_dark"]).pack(anchor="w")
    if intro:
        ttk.Label(outer, text=intro, font=(UI_FONT, 10),
                  foreground=AERO["muted"], wraplength=430,
                  justify=tk.LEFT).pack(anchor="w", pady=(4, 0))

    def choose(key):
        picked["key"] = key
        win.destroy()

    for key, head, desc in options:
        card = tk.Frame(outer, bg=AERO["panel"], bd=0,
                        highlightthickness=1,
                        highlightbackground=AERO["border"],
                        highlightcolor=AERO["border"], cursor="hand2")
        card.pack(fill=tk.X, pady=(12, 0))
        inner = tk.Frame(card, bg=AERO["panel"])
        inner.pack(fill=tk.X, padx=14, pady=11)
        t = tk.Label(inner, text=head, bg=AERO["panel"],
                     fg=AERO["accent_dark"], font=(UI_FONT, 11, "bold"),
                     anchor="w")
        t.pack(fill=tk.X)
        d = tk.Label(inner, text=desc, bg=AERO["panel"], fg=AERO["muted"],
                     font=(UI_FONT, 9), anchor="w", justify=tk.LEFT,
                     wraplength=400)
        d.pack(fill=tk.X, pady=(3, 0))

        def paint(colour, frames=(card, inner), labels=(t, d)):
            for f in frames:
                f.configure(bg=colour)
            for lb in labels:
                lb.configure(bg=colour)

        for w in (card, inner, t, d):
            w.bind("<Button-1>", lambda _e, k=key: choose(k))
            w.bind("<Enter>", lambda _e: paint(AERO["hover_sky"]))
            w.bind("<Leave>", lambda _e: paint(AERO["panel"]))

    ttk.Button(outer, text="Cancel", style="Ghost.TButton",
               command=win.destroy).pack(anchor="e", pady=(14, 0))

    win.update_idletasks()
    try:
        px, py = parent.winfo_rootx(), parent.winfo_rooty()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        x = px + max(0, (pw - win.winfo_width()) // 2)
        y = py + max(0, (ph - win.winfo_height()) // 3)
        win.geometry(f"+{x}+{y}")
    except Exception:
        pass
    win.grab_set()
    win.focus_set()
    win.bind("<Escape>", lambda _e: win.destroy())
    parent.wait_window(win)
    return picked["key"]


def style_toolbar(toolbar, bg=None):
    """Repaint a matplotlib ``NavigationToolbar2Tk`` in the app's colours."""
    bg = bg or AERO["bg_top"]
    try:
        toolbar.configure(bg=bg, relief="flat", borderwidth=0)
    except Exception:
        return toolbar
    for child in toolbar.winfo_children():
        for opts in ({"bg": bg, "relief": "flat", "borderwidth": 0,
                      "highlightthickness": 0},
                     # only the buttons accept these; the readout Label does not
                     {"activebackground": AERO["hover_sky"]},
                     # only the readout Label carries text
                     {"fg": AERO["muted"]}):
            try:
                child.configure(**opts)
            except Exception:
                pass
    return toolbar


# ── The ttk theme itself ──────────────────────────────────────────────────
_PILL_IMGS = []
_ROUND_IMGS = []
_INSTALLED = False


def wrap_to_parent(label, pad=32, minimum=220):
    """Keep a label's wraplength tied to its container's width."""
    def _resize(event):
        if not label.winfo_exists():
            return
        w = max(minimum, event.width - pad)
        if label.cget('wraplength') != w:
            label.configure(wraplength=w)
    label.master.bind('<Configure>', _resize, add='+')
    return label


def labelframe(parent, text="", padding=0, bg=None, fg=None, font=None, **kw):
    """A titled container - the app's LabelFrame, without the stall."""
    if isinstance(padding, (list, tuple)):
        pads = [int(p) for p in padding]
        padx = max(pads[0::2]) if len(pads) > 2 else pads[0]
        pady = max(pads[1::2]) if len(pads) > 1 else pads[0]
    else:
        padx = pady = int(padding)

    import tkinter as tk
    frame = tk.LabelFrame(
        parent,
        text=text,
        bg=bg or AERO["bg_top"],
        fg=fg or AERO["accent_dark"],
        font=font or (UI_FONT, 10, "bold"),
        padx=padx, pady=pady,
        bd=0, relief="flat",
        highlightthickness=1,
        highlightbackground=AERO["border"],
        highlightcolor=AERO["border"],
        **kw
    )
    return frame


def ensure_theme(widget):
    """Install the theme once per interpreter. Safe to call from any window."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    install_theme(widget)


def install_classic_defaults(widget):
    """Give the classic Tk widgets the palette as their default."""
    o = widget.option_add
    ui = (UI_FONT, 10)
    for cls in ("Toplevel", "Frame", "Labelframe", "Label", "Message",
                "Canvas", "Panedwindow"):
        o(f"*{cls}.background", AERO["bg_top"])
    for cls in ("Label", "Message", "Checkbutton", "Radiobutton"):
        o(f"*{cls}.foreground", AERO["text"])
        o(f"*{cls}.font", ui)
    for cls in ("Labelframe", "Checkbutton", "Radiobutton", "Scale",
                "Spinbox", "Button", "Menubutton"):
        o(f"*{cls}.background", AERO["bg_top"])
    o("*Labelframe.foreground", AERO["accent_dark"])
    o("*Labelframe.font", (UI_FONT, 10, "bold"))
    o("*Labelframe.borderWidth", 1)
    o("*Labelframe.relief", "solid")

    for cls in ("Button", "Menubutton"):
        o(f"*{cls}.foreground", AERO["text"])
        o(f"*{cls}.font", ui)
        o(f"*{cls}.relief", "flat")
        o(f"*{cls}.borderWidth", 0)
        o(f"*{cls}.highlightThickness", 1)
        o(f"*{cls}.highlightBackground", AERO["border"])
        o(f"*{cls}.highlightColor", AERO["accent"])
        o(f"*{cls}.activeBackground", AERO["hover_sky"])
        o(f"*{cls}.activeForeground", AERO["accent_dark"])
        o(f"*{cls}.padX", 10)
        o(f"*{cls}.padY", 4)

    for cls in ("Entry", "Spinbox", "Listbox", "Text"):
        o(f"*{cls}.background", AERO["bg"])
        o(f"*{cls}.foreground", AERO["text"])
        o(f"*{cls}.relief", "flat")
        o(f"*{cls}.borderWidth", 1)
        o(f"*{cls}.highlightThickness", 1)
        o(f"*{cls}.highlightBackground", AERO["border"])
        o(f"*{cls}.highlightColor", AERO["accent"])
        o(f"*{cls}.selectBackground", AERO["accent_light"])
        o(f"*{cls}.selectForeground", AERO["text"])
        o(f"*{cls}.insertBackground", AERO["text"])
    for cls in ("Entry", "Spinbox", "Listbox"):
        o(f"*{cls}.font", ui)
    o("*Text.font", (MONO_FONT, 10))
    o("*Listbox.activeStyle", "none")

    for cls in ("Checkbutton", "Radiobutton"):
        o(f"*{cls}.activeBackground", AERO["hover_sky"])
        o(f"*{cls}.activeForeground", AERO["accent_dark"])
        o(f"*{cls}.selectColor", AERO["bg"])
        o(f"*{cls}.highlightThickness", 0)
        o(f"*{cls}.borderWidth", 0)

    o("*Menu.background", AERO["bg"])
    o("*Menu.foreground", AERO["text"])
    o("*Menu.font", ui)
    o("*Menu.activeBackground", AERO["accent_light"])
    o("*Menu.activeForeground", AERO["text"])
    o("*Menu.selectColor", AERO["accent_dark"])
    o("*Menu.relief", "flat")
    o("*Menu.borderWidth", 1)
    o("*Menu.activeBorderWidth", 0)

    o("*Scale.troughColor", AERO["border_soft"])
    o("*Scale.highlightThickness", 0)
    o("*Scale.borderWidth", 0)
    o("*Scale.font", ui)


def install_theme(widget):
    """Configure TTK styles - Frutiger Aero theme."""
    install_classic_defaults(widget)
    style = ttk.Style(widget)

    available_themes = style.theme_names()
    if 'clam' in available_themes:
        style.theme_use('clam')
    elif 'alt' in available_themes:
        style.theme_use('alt')

    # Try to pick up the window bg from ctk; fallback to Aero sky-white
    try:
        widget.configure(fg_color=AERO["bg_top"])
    except Exception:
        try:
            widget.configure(bg=AERO["bg_top"])
        except Exception:
            pass

    # ── Base frames / labels ─────────────────────────────────
    style.configure("TFrame", background=AERO["bg_top"])
    style.configure("TLabel",
                    background=AERO["bg_top"],
                    foreground=AERO["text"],
                    font=('Segoe UI', 10))
    style.configure("TLabelframe",
                    background=AERO["bg_top"],
                    foreground=AERO["accent_dark"],
                    bordercolor=AERO["border"],
                    lightcolor=AERO["border_soft"],
                    darkcolor=AERO["border"],
                    borderwidth=1,
                    relief="solid")
    style.configure("TLabelframe.Label",
                    background=AERO["bg_top"],
                    foreground=AERO["accent_dark"],
                    font=('Segoe UI', 10, 'bold'))
    style.configure("TSeparator", background=AERO["border"])

    # ── Buttons ──────────────────────────────────────────────
    style.configure("TButton",
                    font=('Segoe UI', 10),
                    background=AERO["glass_hilite"],
                    foreground=AERO["text"],
                    bordercolor=AERO["border"],
                    lightcolor="#FFFFFF",
                    darkcolor=AERO["border"],
                    focuscolor=AERO["accent_light"],
                    relief="flat",
                    padding=(10, 6))
    style.map("TButton",
              background=[('pressed', AERO["pressed_sky"]),
                          ('active', AERO["hover_sky"])],
              bordercolor=[('active', AERO["accent"]),
                           ('pressed', AERO["accent_dark"])],
              foreground=[('disabled', AERO["muted"])])

    # Green "Add" style - nature accent
    style.configure("Add.TButton",
                    font=('Segoe UI', 11, 'bold'),
                    background=AERO["green_light"],
                    foreground=AERO["green_dark"],
                    bordercolor=AERO["green"],
                    padding=(12, 8),
                    relief="flat")
    style.map("Add.TButton",
              background=[('active', "#A5D6A7"), ('pressed', "#81C784")],
              bordercolor=[('active', AERO["green_dark"])])

    # Blue "Action" style - primary sky accent
    style.configure("Action.TButton",
                    font=('Segoe UI', 10, 'bold'),
                    background=AERO["accent_light"],
                    foreground=AERO["accent_dark"],
                    bordercolor=AERO["accent"],
                    padding=(12, 8),
                    relief="flat")
    style.map("Action.TButton",
              background=[('active', AERO["sky_top"]),
                          ('pressed', AERO["accent"])],
              foreground=[('active', AERO["accent_dark"]),
                          ('pressed', "#FFFFFF")],
              bordercolor=[('active', AERO["accent_dark"])])

    # ── Unified named button styles ──────────
    # Primary: the main call-to-action of a panel (sky)
    style.configure("Primary.TButton",
                    font=('Segoe UI', 10, 'bold'),
                    background=AERO["accent"],
                    foreground="#FFFFFF",
                    bordercolor=AERO["accent_dark"],
                    padding=(14, 8),
                    relief="flat")
    style.map("Primary.TButton",
              background=[('disabled', AERO["border_soft"]),
                          ('pressed', AERO["accent_dark"]),
                          ('active', AERO["sky_top"])],
              foreground=[('disabled', AERO["muted"])],
              bordercolor=[('active', AERO["accent_dark"])])

    # Secondary: supporting actions (neutral sky)
    style.configure("Secondary.TButton",
                    font=('Segoe UI', 10),
                    background=AERO["panel_bot"],
                    foreground=AERO["accent_dark"],
                    bordercolor=AERO["border"],
                    padding=(12, 6),
                    relief="flat")
    style.map("Secondary.TButton",
              background=[('disabled', AERO["border_soft"]),
                          ('pressed', AERO["pressed_sky"]),
                          ('active', AERO["hover_sky"])],
              foreground=[('disabled', AERO["muted"])])

    # Destructive: clear / delete / deselect (danger)
    style.configure("Destructive.TButton",
                    font=('Segoe UI', 10, 'bold'),
                    background="#F7DCDA",
                    foreground=AERO["danger"],
                    bordercolor=AERO["danger"],
                    padding=(12, 6),
                    relief="flat")
    style.map("Destructive.TButton",
              background=[('disabled', AERO["border_soft"]),
                          ('pressed', AERO["danger_hover"]),
                          ('active', "#F1B5AF")],
              foreground=[('disabled', AERO["muted"]),
                          ('pressed', "#FFFFFF")])

    # Warn / Download: orange emphasis
    style.configure("Warn.TButton",
                    font=('Segoe UI', 10, 'bold'),
                    background="#FFE7D1",
                    foreground="#9A4A06",
                    bordercolor=AERO["warn"],
                    padding=(12, 6),
                    relief="flat")
    style.map("Warn.TButton",
              background=[('disabled', AERO["border_soft"]),
                          ('pressed', "#D35400"),
                          ('active', "#F7C08A")],
              foreground=[('disabled', AERO["muted"]),
                          ('pressed', "#FFFFFF")])

    # Tool: large flagship buttons in the Analysis Tools panel
    style.configure("Tool.TButton",
                    font=('Segoe UI', 10, 'bold'),
                    background=AERO["accent_light"],
                    foreground=AERO["accent_dark"],
                    bordercolor=AERO["accent"],
                    padding=(18, 14),
                    relief="flat")
    style.map("Tool.TButton",
              background=[('disabled', AERO["border_soft"]),
                          ('pressed', AERO["accent"]),
                          ('active', AERO["sky_top"])],
              foreground=[('disabled', AERO["muted"]),
                          ('pressed', "#FFFFFF")])

    # ToolGreen / ToolWarn: matching flavors for the tool row
    style.configure("ToolGreen.TButton",
                    font=('Segoe UI', 10, 'bold'),
                    background=AERO["green_light"],
                    foreground=AERO["green_dark"],
                    bordercolor=AERO["green"],
                    padding=(18, 14),
                    relief="flat")
    style.map("ToolGreen.TButton",
              background=[('disabled', AERO["border_soft"]),
                          ('pressed', AERO["green_dark"]),
                          ('active', "#A5D6A7")],
              foreground=[('disabled', AERO["muted"]),
                          ('pressed', "#FFFFFF")])

    style.configure("ToolWarn.TButton",
                    font=('Segoe UI', 10, 'bold'),
                    background="#FFE7D1",
                    foreground="#9A4A06",
                    bordercolor=AERO["warn"],
                    padding=(18, 14),
                    relief="flat")
    style.map("ToolWarn.TButton",
              background=[('disabled', AERO["border_soft"]),
                          ('pressed', "#D35400"),
                          ('active', "#F7C08A")],
              foreground=[('disabled', AERO["muted"]),
                          ('pressed', "#FFFFFF")])

    # Toggle (chevron): compact flat pill used by Step 1 collapse
    style.configure("Toggle.TButton",
                    font=('Segoe UI', 9, 'bold'),
                    background=AERO["panel_bot"],
                    foreground=AERO["accent_dark"],
                    bordercolor=AERO["border"],
                    padding=(10, 3),
                    relief="flat")
    style.map("Toggle.TButton",
              background=[('active', AERO["hover_sky"]),
                          ('pressed', AERO["pressed_sky"])])

    # Ghost: minimal, for inline refresh-style actions
    style.configure("Ghost.TButton",
                    font=('Segoe UI', 9),
                    background=AERO["bg_top"],
                    foreground=AERO["accent_dark"],
                    bordercolor=AERO["border_soft"],
                    padding=(8, 3),
                    relief="flat")
    style.map("Ghost.TButton",
              background=[('active', AERO["hover_sky"]),
                          ('pressed', AERO["pressed_sky"])])

    # ── Rounded "pill" restyle (capsule buttons) ──────────────────────
    try:
        _install_pill_button_styles(style)
    except Exception as _pill_err:  # pragma: no cover - visual nicety only
        try:
            print(f"[UI] pill-button restyle skipped: {_pill_err}")
        except Exception:
            pass

    try:
        _install_rounded_container_styles(style)
    except Exception as _round_err:  # pragma: no cover - visual nicety only
        try:
            print(f"[UI] rounded-container restyle skipped: {_round_err}")
        except Exception:
            pass

    try:
        _install_treeview_gridlines(style)
    except Exception as _grid_err:  # pragma: no cover - visual nicety only
        try:
            print(f"[UI] table gridlines skipped: {_grid_err}")
        except Exception:
            pass

    # ── Entries / comboboxes ─────────────────────────────────
    style.configure("TEntry",
                    fieldbackground="#FFFFFF",
                    background="#FFFFFF",
                    foreground=AERO["text"],
                    bordercolor=AERO["border"],
                    lightcolor=AERO["border_soft"],
                    darkcolor=AERO["border"],
                    insertcolor=AERO["accent_dark"],
                    padding=4)
    style.map("TEntry",
              bordercolor=[('focus', AERO["accent"])],
              lightcolor=[('focus', AERO["accent_light"])])

    style.configure("TCombobox",
                    fieldbackground="#FFFFFF",
                    background=AERO["panel_bot"],
                    foreground=AERO["text"],
                    bordercolor=AERO["border"],
                    arrowcolor=AERO["accent_dark"],
                    padding=4)
    style.map("TCombobox",
              bordercolor=[('focus', AERO["accent"])],
              fieldbackground=[('readonly', "#FFFFFF")])

    # ── Radio / check ────────────────────────────────────────
    style.configure("TRadiobutton",
                    background=AERO["bg_top"],
                    foreground=AERO["text"],
                    font=('Segoe UI', 10))
    style.map("TRadiobutton",
              background=[('active', AERO["hover_sky"])],
              indicatorcolor=[('selected', AERO["accent"]),
                              ('!selected', "#FFFFFF")])

    style.configure("TCheckbutton",
                    background=AERO["bg_top"],
                    foreground=AERO["text"],
                    font=('Segoe UI', 10))
    style.map("TCheckbutton",
              background=[('active', AERO["hover_sky"])],
              indicatorcolor=[('selected', AERO["accent"]),
                              ('!selected', "#FFFFFF")])

    # ── Scrollbars ───────────────────────────────────────────
    style.configure("Vertical.TScrollbar",
                    background=AERO["panel_bot"],
                    troughcolor=AERO["bg_top"],
                    bordercolor=AERO["border_soft"],
                    arrowcolor=AERO["accent_dark"],
                    lightcolor=AERO["border_soft"],
                    darkcolor=AERO["border"])
    style.map("Vertical.TScrollbar",
              background=[('active', AERO["accent_light"]),
                          ('pressed', AERO["accent"])])
    style.configure("Horizontal.TScrollbar",
                    background=AERO["panel_bot"],
                    troughcolor=AERO["bg_top"],
                    bordercolor=AERO["border_soft"],
                    arrowcolor=AERO["accent_dark"])
    style.map("Horizontal.TScrollbar",
              background=[('active', AERO["accent_light"]),
                          ('pressed', AERO["accent"])])

    # ── Progress bar ────────
    style.configure("Horizontal.TProgressbar",
                    background=AERO["green"],
                    troughcolor=AERO["border_soft"],
                    bordercolor=AERO["border_soft"],
                    lightcolor=AERO["leaf_top"],
                    darkcolor=AERO["green_dark"],
                    thickness=18)
    # the blue variant the secondary windows and dialogs use
    style.configure("Accent.Horizontal.TProgressbar",
                    background=AERO["accent"],
                    troughcolor=AERO["border_soft"],
                    bordercolor=AERO["border_soft"],
                    lightcolor=AERO["sky_top"],
                    darkcolor=AERO["accent_dark"],
                    thickness=14)
    style.configure("TProgressbar",
                    background=AERO["green"],
                    troughcolor=AERO["border_soft"],
                    bordercolor=AERO["border_soft"],
                    lightcolor=AERO["leaf_top"],
                    darkcolor=AERO["green_dark"],
                    thickness=18)

    # ── Notebook (if any) ────────────────────────────────────
    style.configure("TNotebook",
                    background=AERO["bg_top"],
                    bordercolor=AERO["bg_top"],
                    lightcolor=AERO["bg_top"],
                    darkcolor=AERO["bg_top"],
                    borderwidth=0,
                    tabmargins=(2, 4, 2, 0))
    style.configure("TNotebook.Tab",
                    background=AERO["panel_bot"],
                    foreground=AERO["text"],
                    padding=(8, 6),
                    font=('Segoe UI', 10))
    style.map("TNotebook.Tab",
              background=[('selected', "#FFFFFF"),
                          ('active', AERO["hover_sky"])],
              foreground=[('selected', AERO["accent_dark"])])

    # ── Treeview ─────────────────────────────────────────────
    style.configure("Treeview",
                    background="#FFFFFF",
                    fieldbackground="#FFFFFF",
                    foreground=AERO["text"],
                    bordercolor=AERO["border"],
                    font=(UI_FONT, 10),
                    rowheight=26)
    style.configure("Treeview.Heading",
                    background=AERO["panel_bot"],
                    foreground=AERO["accent_dark"],
                    borderwidth=1,
                    relief="flat",
                    font=(UI_FONT, 10, 'bold'))
    style.map("Treeview",
              background=[('selected', AERO["accent_light"])],
              foreground=[('selected', AERO["accent_dark"])])

def _install_treeview_gridlines(style):
    """Rule every ttk.Treeview like a spreadsheet: black lines between cells."""
    from PIL import Image, ImageTk

    line = _to_rgb(AERO.get("table_rule", "#000000"))

    def _photo(pixels, size):
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        img.putdata(pixels)
        photo = ImageTk.PhotoImage(img)
        _PILL_IMGS.append(photo)   # Tk keeps only a weak hold on images
        return photo

    def _row(fill_hex):
        """4x4: three rows of fill over a 1px rule along the bottom."""
        fill = (*_to_rgb(fill_hex), 255)
        rule = (*line, 255)
        return _photo([fill] * 12 + [rule] * 4, (4, 4))

    _clear, _rule = (0, 0, 0, 0), (*line, 255)
    cell = _photo([_clear, _clear, _clear, _rule] * 4, (4, 4))

    style.element_create(
        "GV.Treeitem.row", "image", _row("#FFFFFF"),
        ("selected", _row(AERO["accent_light"])),
        border=(0, 0, 0, 1), sticky="nswe")
    style.element_create(
        "GV.Treedata.rule", "image", cell,
        border=(0, 0, 1, 0), sticky="nswe")

    style.layout("Treeview.Row", [("GV.Treeitem.row", {"sticky": "nswe"})])
    style.layout("Treeview.Cell", [
        ("GV.Treedata.rule", {"sticky": "nswe", "children": [
            ("Treedata.padding", {"sticky": "nswe", "children": [
                ("Treeitem.text", {"sticky": "nswe"})]})]})])


def _install_pill_button_styles(style):
    """Re-skin the named ttk button styles as rounded capsule 'pills'."""
    from PIL import Image, ImageDraw, ImageTk

    # module-level so the images outlive any single window

    def _clamp(v):
        return max(0, min(255, int(v)))

    def _shade(hex_color, factor):
        """factor < 1 darkens, > 1 lightens; returns an (r, g, b) tuple."""
        h = hex_color.lstrip('#')
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return (_clamp(r * factor), _clamp(g * factor), _clamp(b * factor))

    def _rgba(hex_color, a=255):
        h = hex_color.lstrip('#')
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), a)

    def _pill_image(h, fill_hex, radius, border_rgb=None, border_px=0):
        """Horizontal 9-slice rounded rect. Returns (photo, cap)."""
        ss = 4  # supersample for smooth anti-aliased edges
        cap = max(2, min(int(radius), h // 2))   # fixed left/right slice
        w = 2 * cap + 8                          # caps + ~8px stretch center
        W, H = w * ss, h * ss
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        outline = (_rgba('%02x%02x%02x' % border_rgb)
                   if border_rgb else None)
        d.rounded_rectangle([0, 0, W - 1, H - 1], radius=cap * ss,
                            fill=_rgba(fill_hex), outline=outline,
                            width=(border_px * ss if border_rgb else 0))
        img = img.resize((w, h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        _PILL_IMGS.append(photo)
        return photo, cap

    def _hex_rgb(hex_color):
        h = hex_color.lstrip('#')
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    def _skin(name, fill, fg, edge, hover, pressed, font, pad_x, h):
        """Outline-pill skin: light fill + thin coloured border + coloured text."""
        rad = h // 2                       # full capsule
        edge_rgb = _hex_rgb(edge)
        normal_img, cap = _pill_image(h, fill, rad, edge_rgb, 1)
        hover_img, _ = _pill_image(h, hover, rad, edge_rgb, 1)
        press_img, _ = _pill_image(h, pressed, rad, edge_rgb, 1)
        dis_fill = AERO["border_soft"]
        dis_img, _ = _pill_image(h, dis_fill, rad, _shade(dis_fill, 0.9), 1)
        elem = "pill_" + name.replace('.', '_')
        try:
            style.element_create(
                elem, "image", normal_img,
                ("pressed", press_img),
                ("active", hover_img),
                ("disabled", dis_img),
                border=(cap, 0, cap, 0), sticky="nsew", height=h)
        except Exception:
            return  # element exists already / imaging unsupported -> keep square
        style.layout(name, [
            (elem, {"sticky": "nsew", "children": [
                ("Button.padding", {"sticky": "nsew", "children": [
                    ("Button.label", {"sticky": "nsew"})]})]})])
        bg = AERO["bg_top"]
        style.configure(name, font=font, foreground=fg, background=bg,
                        bordercolor=bg, lightcolor=bg, darkcolor=bg,
                        focuscolor=bg, padding=(max(2, pad_x), 2),
                        anchor="center", relief="flat", borderwidth=0)
        style.map(name,
                  foreground=[('disabled', AERO["muted"]),
                              ('pressed', fg), ('active', fg)],
                  background=[('disabled', bg), ('pressed', bg),
                              ('active', bg), ('!active', bg)],
                  bordercolor=[('focus', bg), ('active', bg)],
                  lightcolor=[('pressed', bg), ('active', bg)],
                  darkcolor=[('pressed', bg), ('active', bg)])

    A = AERO
    WHITE = "#FFFFFF"
    # (style, fill, fg/text, border, hover, pressed, font, pad_x, height)
    specs = [
        ("TButton",             WHITE, A["accent_dark"], A["border"],  A["hover_sky"], A["pressed_sky"], ('Segoe UI', 10),         12, 32),
        ("Add.TButton",         WHITE, A["green_dark"],  A["green"],    A["green_light"], "#B8E6B6",      ('Segoe UI', 11, 'bold'), 16, 38),
        ("Action.TButton",      WHITE, A["accent_dark"], A["accent"],   A["hover_sky"], A["pressed_sky"], ('Segoe UI', 10, 'bold'), 16, 38),
        ("Primary.TButton",     WHITE, A["accent"],      A["accent"],   A["hover_sky"], A["pressed_sky"], ('Segoe UI', 10, 'bold'), 18, 36),
        ("Secondary.TButton",   WHITE, A["accent_dark"], A["border"],   A["hover_sky"], A["pressed_sky"], ('Segoe UI', 10),         14, 32),
        ("Destructive.TButton", WHITE, A["danger"],      A["danger"],   "#FBE9E7",      "#F5CDC8",        ('Segoe UI', 10, 'bold'), 14, 32),
        ("Warn.TButton",        WHITE, "#9A4A06",        A["warn"],     "#FDF0E4",      "#F7DFC6",        ('Segoe UI', 10, 'bold'), 14, 32),
        ("Tool.TButton",        WHITE, A["accent_dark"], A["accent"],   A["hover_sky"], A["pressed_sky"], ('Segoe UI', 9, 'bold'),   2, 34),
        ("ToolGreen.TButton",   WHITE, A["green_dark"],  A["green"],    A["green_light"], "#B8E6B6",      ('Segoe UI', 9, 'bold'),   2, 34),
        ("ToolWarn.TButton",    WHITE, "#9A4A06",        A["warn"],     "#FDF0E4",      "#F7DFC6",        ('Segoe UI', 9, 'bold'),   2, 34),
        ("Toggle.TButton",      WHITE, A["accent_dark"], A["border"],   A["hover_sky"], A["pressed_sky"], ('Segoe UI', 9, 'bold'),   8, 26),
        ("Ghost.TButton",       WHITE, A["accent_dark"], A["border_soft"], A["hover_sky"], A["pressed_sky"], ('Segoe UI', 9),       8, 26),
    ]
    for spec in specs:
        _skin(*spec)

def _install_rounded_container_styles(style):
    """Round the LabelFrame cards, Entry/Combobox fields and Notebook tabs."""
    from PIL import Image, ImageDraw, ImageTk

    # module-level so the images outlive any single window
    A = AERO

    def _rgba(hex_color, a=255):
        h = hex_color.lstrip('#')
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), a)

    def _card_image(fill_hex, radius, border_hex=None, border_px=1,
                    round_bottom=True):
        """Four-corner (or top-only) rounded-rect 9-slice. Returns (photo, r)."""
        ss = 4
        r = max(3, int(radius))
        pad = border_px + 1
        w = h = 2 * r + 8                       # fixed corners + small centre
        W, H = w * ss, h * ss
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        outline = _rgba(border_hex) if border_hex else None
        box = [pad * ss, pad * ss, W - 1 - pad * ss, H - 1 - pad * ss]
        d.rounded_rectangle(box, radius=r * ss, fill=_rgba(fill_hex),
                            outline=outline,
                            width=(border_px * ss if outline else 0))
        if not round_bottom:
            # Square off the lower half so tabs sit flush on the notebook body.
            d.rectangle([box[0], (H // 2), box[2], box[3] + border_px * ss],
                        fill=_rgba(fill_hex))
            if outline:
                for x in (box[0], box[2]):
                    d.line([(x, H // 2), (x, box[3] + border_px * ss)],
                           fill=outline, width=border_px * ss)
        img = img.resize((w, h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        _ROUND_IMGS.append(photo)
        return photo, r

    # ── LabelFrame → rounded card (border blends with the page fill) ──────
    try:
        card_img, r = _card_image(A["bg_top"], 14, A["border"], 1)
        style.element_create("Rounded.Labelframe.border", "image", card_img,
                             border=(r, r, r, r), sticky="nsew")
        style.layout("TLabelframe",
                     [("Rounded.Labelframe.border", {"sticky": "nsew"})])
        style.configure("TLabelframe", background=A["bg_top"])
    except Exception:
        pass

    # ── Entry → rounded white field ──────────────────────────────────────
    try:
        e_norm, er = _card_image("#FFFFFF", 10, A["border"], 1)
        e_foc, _ = _card_image("#FFFFFF", 10, A["accent"], 1)
        style.element_create("Rounded.Entry.field", "image", e_norm,
                             ("focus", e_foc),
                             border=(er, er, er, er), sticky="nsew")
        style.layout("TEntry", [
            ("Rounded.Entry.field", {"sticky": "nsew", "children": [
                ("Entry.padding", {"sticky": "nsew", "children": [
                    ("Entry.textarea", {"sticky": "nsew"})]})]})])
        style.configure("TEntry", padding=(8, 5))
    except Exception:
        pass

    # ── Combobox → rounded white field (keep the dropdown arrow) ─────────
    try:
        c_norm, cr = _card_image("#FFFFFF", 10, A["border"], 1)
        c_foc, _ = _card_image("#FFFFFF", 10, A["accent"], 1)
        style.element_create("Rounded.Combobox.field", "image", c_norm,
                             ("focus", c_foc),
                             border=(cr, cr, cr, cr), sticky="nsew")
        style.layout("TCombobox", [
            ("Rounded.Combobox.field", {"sticky": "nsew", "children": [
                ("Combobox.downarrow", {"side": "right", "sticky": "ns"}),
                ("Combobox.padding", {"sticky": "nsew", "children": [
                    ("Combobox.textarea", {"sticky": "nsew"})]})]})])
        style.configure("TCombobox", padding=(8, 5))
    except Exception:
        pass

    # ── Notebook tabs → rounded tops ─────────────────────────────────────
    try:
        t_norm, tr = _card_image(A["panel_bot"], 10, A["border"], 1,
                                 round_bottom=False)
        t_sel, _ = _card_image("#FFFFFF", 10, A["border"], 1,
                               round_bottom=False)
        t_act, _ = _card_image(A["hover_sky"], 10, A["border"], 1,
                               round_bottom=False)
        style.element_create("Rounded.Notebook.tab", "image", t_norm,
                             ("selected", t_sel), ("active", t_act),
                             border=(tr, tr, tr, 2), sticky="nsew")
        style.layout("TNotebook.Tab", [
            ("Rounded.Notebook.tab", {"sticky": "nsew", "children": [
                ("Notebook.padding", {"side": "top", "sticky": "nsew",
                                      "children": [
                    ("Notebook.label", {"side": "top", "sticky": ""})]})]})])
        style.configure("TNotebook.Tab", padding=(8, 6))
    except Exception:
        pass


__all__ = ["AERO", "UI_FONT", "MONO_FONT", "style_window",
           "install_theme", "ensure_theme", "wrap_to_parent"]
