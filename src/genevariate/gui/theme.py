"""
The Frutiger Aero palette every GeneVariate window draws from.

The main window builds its ttk theme from these tokens; secondary windows
(region analysis, comparison, evaluation) import the same dict so a heading in
one window is the same blue as a heading in another. Colours belong here rather
than inline at the call site - a hex literal in a widget is how two windows
drift apart.
"""

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
    # rules between table rows/columns - deliberately black, not the soft
    # border tint, because their whole job is to be unmissable
    "table_rule":    "#000000",
}

UI_FONT = "Segoe UI"
MONO_FONT = "Consolas"


def _to_rgb(hex_color):
    """'#RRGGBB' -> (r, g, b)."""
    h = hex_color.lstrip('#')
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def style_window(win, bg=None):
    """Give a ``tk.Toplevel`` the application background.

    The ttk theme paints ``TFrame`` and ``TLabel`` with an AERO background, but
    a bare ``tk.Toplevel`` keeps Tk's default grey. The result is pale-blue
    widgets floating on a grey slab - which is what every dialog in the app
    looked like. Setting the container to match is the whole fix, and it has to
    happen at the window rather than at each widget, or the next dialog someone
    adds inherits the same grey.
    """
    try:
        win.configure(bg=bg or AERO["bg_top"])
    except Exception:
        pass          # some Toplevels are override-redirect and refuse bg
    return win


def style_toolbar(toolbar, bg=None):
    """Repaint a matplotlib ``NavigationToolbar2Tk`` in the app's colours.

    The toolbar is plain Tk, not ttk, so the theme never reaches it and every
    embedded figure in the app sat on a strip of Windows-grey. Its children are
    ordinary ``tk.Button``/``tk.Label`` widgets, so recolouring the frame alone
    would leave a grey square behind each icon - both levels have to be set.
    """
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
#
# This used to live on the main window, which meant a secondary window only
# looked like the app because the main window happened to have run first: ttk
# styles are per-interpreter, so the region window was inheriting a theme it
# never asked for. Opened standalone it fell back to bare clam - square grey
# 3D buttons on a grey slab. Installing the theme from here, and calling it
# from every window that owns a Style, is what makes that impossible.
#
# The generated PhotoImages must outlive the window that triggered their
# creation, so they are held at module level rather than on a widget.
_PILL_IMGS = []
_ROUND_IMGS = []
_INSTALLED = False


def wrap_to_parent(label, pad=32, minimum=220):
    """Keep a label's wraplength tied to its container's width.

    A label with no wraplength is clipped at the window edge, and a fixed one
    is wrong the moment the window is resized - which is how the explanatory
    banners ended up running off the right edge instead of wrapping.
    """
    def _resize(event):
        if not label.winfo_exists():
            return
        w = max(minimum, event.width - pad)
        if label.cget('wraplength') != w:
            label.configure(wraplength=w)
    label.master.bind('<Configure>', _resize, add='+')
    return label


def labelframe(parent, text="", padding=0, bg=None, fg=None, font=None, **kw):
    """A titled container - the app's LabelFrame, without the stall.

    Creating a ``ttk.LabelFrame`` makes a synchronous request to the X server
    and blocks until the window manager answers it. Against a bare server that
    costs about 1 ms, but under a compositing desktop it measured 180 ms per
    widget, and the cost is paid once per card: a dialog with twenty of them
    took four seconds to build and the main window took nine. All of it is
    spent inside widget construction at 0% CPU with no event loop running, so
    it does not reach the user as a slow window, it reaches them as an
    application that has died. Nothing in the program can predict how slow
    somebody else's window manager will be, which is why this is not a
    tuning problem.

    ``tk.LabelFrame`` draws the same titled box, makes no such request, and
    costs 0.04 ms under both servers. Building it here rather than at each
    call site is what keeps every card in the application the same fill,
    border and title font.
    """
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
        # A 1px highlight border takes the theme's colour, where `relief`
        # would force Tk's black hairline or a 3D bevel.
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


def install_theme(widget):
    """Configure TTK styles - Frutiger Aero theme (glossy white + sky-blue +
    nature-green). Matches CellTracker's desktop_gui aesthetic."""
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

    # ── Unified named button styles (Phase 1 redesign) ──────────
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

    # ── Rounded "pill" restyle (CopilotKit-inspired capsules) ──────────
    # ttk's clam theme paints square-cornered buttons. We overlay each
    # named style with a 9-slice, PIL-rendered rounded-capsule image so
    # every existing ttk.Button becomes a smooth pill - no call-site
    # changes. Semantic colors + hover/pressed states are preserved.
    # Wrapped in try/except: if imaging is unavailable the app simply
    # keeps the original (square) styles.
    try:
        _install_pill_button_styles(style)
    except Exception as _pill_err:  # pragma: no cover - visual nicety only
        try:
            print(f"[UI] pill-button restyle skipped: {_pill_err}")
        except Exception:
            pass

    # Round the *containers* too (cards / fields / tabs) so the whole GUI
    # reads soft instead of angular - same PIL 9-slice trick as the pills.
    try:
        _install_rounded_container_styles(style)
    except Exception as _round_err:  # pragma: no cover - visual nicety only
        try:
            print(f"[UI] rounded-container restyle skipped: {_round_err}")
        except Exception:
            pass

    # Rule the tables like a spreadsheet. Borderless rows of numbers are hard
    # to read across, especially in the wide result tables.
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

    # ── Progress bar - green glossy like CellTracker ────────
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
    # No border on the notebook itself: the tabs already carry one, and the
    # two together drew a second rule across the window under the tab strip.
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
    # Flat 1px heading border so the header separators sit exactly on the body
    # gridlines instead of a raised 3-D bevel a couple of pixels off them.
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
    """Rule every ttk.Treeview like a spreadsheet: black lines between cells.

    ttk has no gridline option, so the row and cell backgrounds are replaced
    with 9-slice image elements. The border spec pins one edge of the image at
    its natural size while the rest stretches, which is what keeps the rule
    exactly 1px tall however far the row is stretched.

    Painting the row background as an image means this code now owns the
    selection highlight too - hence the ``selected`` state images, without
    which a selected row would simply stop changing colour.
    """
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

    # Transparent but for a 1px rule down the right edge, so it can be laid
    # over the row background to separate the columns.
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
    """Re-skin the named ttk button styles as rounded capsule 'pills'.

    ttk's clam theme paints square corners. We render anti-aliased
    rounded-rectangle images with PIL and register them as horizontal
    9-slice ttk image elements (rounded left/right caps stay fixed, the
    middle stretches), so every existing ``ttk.Button`` keeps its semantic
    color + hover/pressed states but gains fully rounded corners in the
    style of the CopilotKit capsule buttons. Generated PhotoImages are
    retained on ``self`` so Tk cannot garbage-collect them.
    """
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
        """Horizontal 9-slice rounded rect. Returns (photo, cap).

        ``radius`` is the corner radius in px (capped at h/2). The left/right
        fixed 9-slice caps equal that radius, so the child label is inset by
        only ~radius on each side - keeping room for text on narrow buttons.
        """
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
        """Outline-pill skin: light fill + thin coloured border + coloured
        text (matches the reference 'Refresh' capsule). ``edge``/``fg`` carry
        the semantic colour; the fill stays light so the text/border read."""
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
        # Reset the leftover solid ``background`` from the base style so no
        # square colour band shows through the pill's transparent corners -
        # the capsule floats directly on the container colour.
        bg = AERO["bg_top"]
        style.configure(name, font=font, foreground=fg, background=bg,
                        bordercolor=bg, lightcolor=bg, darkcolor=bg,
                        focuscolor=bg, padding=(max(2, pad_x), 2),
                        anchor="center", relief="flat", borderwidth=0)
        # Outline style keeps coloured text in every state (never white);
        # background stays the container colour so the corners never fill.
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
    # Outline-pill palette (blue/green as before): white fill, coloured
    # border + coloured text; hover/pressed are faint tints of that colour.
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
    """Round the *container* chrome - LabelFrame cards, Entry/Combobox
    fields and Notebook tabs - with the same anti-aliased PIL 9-slice
    images used for the pill buttons, so the whole GUI reads soft instead
    of angular. Each family is wrapped in its own try/except: any failure
    keeps that widget's original square style, and generated PhotoImages
    are retained on ``self`` so Tk cannot garbage-collect them.
    """
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
            # The bottom edge is deliberately left unstroked: closing the
            # outline all the way round drew a line under every tab, which is
            # what made the tab strip read as a row of detached boxes with
            # doubled rules between them.
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
