"""
The Frutiger Aero palette every GeneVariate window draws from.

The main window builds its ttk theme from these tokens; secondary windows
(region analysis, comparison, evaluation) import the same dict so a heading in
one window is the same blue as a heading in another. Colours belong here rather
than inline at the call site — a hex literal in a widget is how two windows
drift apart.
"""

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
}

UI_FONT = "Segoe UI"
MONO_FONT = "Consolas"

__all__ = ["AERO", "UI_FONT", "MONO_FONT"]
