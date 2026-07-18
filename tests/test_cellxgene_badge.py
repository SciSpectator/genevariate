"""Unit tests for the 'Pre-classified' CELLxGENE badge in the platform list.

The badge is drawn inside ``GeneVariateApp._refresh_platform_buttons`` for any
platform whose name starts with ``CellxGene_`` (the prefix set in
``cellxgene_browser.py`` when a pseudo-bulk slice is registered).

We don't construct the full app — we bind the unbound method to a minimal
stub that exposes only the four attributes the method touches, then walk
the resulting Tk widget tree.

Skips automatically when no display is available (headless CI).
"""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace

import pandas as pd
import pytest


# ── Skip the whole module if Tk has no display ──────────────────────────
def _tk_available() -> bool:
    try:
        r = tk.Tk()
    except tk.TclError:
        return False
    r.withdraw()
    r.destroy()
    return True


pytestmark = pytest.mark.skipif(
    not _tk_available(),
    reason="No DISPLAY available — skipping Tk widget tests",
)


# ── Helpers ─────────────────────────────────────────────────────────────
def _mk_df(n_samples: int = 4) -> pd.DataFrame:
    return pd.DataFrame({
        "GSM": [f"PB_{i}" for i in range(n_samples)],
        "geneA": list(range(n_samples)),
        "geneB": list(range(n_samples)),
    })


def _render(gpl_datasets: dict) -> tuple[tk.Tk, ttk.Frame]:
    """Build a stub app, call _refresh_platform_buttons, return (root, frame)."""
    from genevariate.gui.app import GeoWorkflowGUI

    root = tk.Tk()
    root.withdraw()
    frame = ttk.Frame(root)

    stub = SimpleNamespace(
        _plat_btn_frame=frame,
        gpl_datasets=gpl_datasets,
        gpl_available_files={},
        data_dir=None,
        _user_data_dirs=[],
        # _refresh_platform_buttons calls _discover_available_platforms;
        # for these tests we stub it to return empty so only `loaded` matters.
        _discover_available_platforms=lambda: {},
        # _smart_load_gpl is never invoked in tests (no clicks).
        _smart_load_gpl=lambda p: None,
    )
    GeoWorkflowGUI._refresh_platform_buttons(stub)
    return root, frame


def _find_labels(widget) -> list[tk.Label]:
    """Recursively collect every tk.Label descendant."""
    out: list[tk.Label] = []
    for child in widget.winfo_children():
        if isinstance(child, tk.Label):
            out.append(child)
        out.extend(_find_labels(child))
    return out


def _badge_labels(widget) -> list[tk.Label]:
    return [lbl for lbl in _find_labels(widget)
            if "Pre-classified" in (lbl.cget("text") or "")]


# ── Tests ───────────────────────────────────────────────────────────────
class TestPreClassifiedBadge:
    def test_badge_present_for_cellxgene_platform(self):
        """A CellxGene_* platform must render a 'Pre-classified' badge."""
        root, frame = _render({
            "CellxGene_lung_normal_mean": _mk_df(10),
        })
        try:
            badges = _badge_labels(frame)
            assert len(badges) == 1, \
                f"Expected exactly 1 badge, got {len(badges)}"
            text = badges[0].cget("text")
            assert "CELLxGENE Cell Ontology" in text
            assert "Pre-classified" in text
        finally:
            root.destroy()

    def test_no_badge_for_regular_gpl_platform(self):
        """A regular GPL platform must NOT render the badge."""
        root, frame = _render({
            "GPL570": _mk_df(20),
            "GPL96":  _mk_df(15),
        })
        try:
            badges = _badge_labels(frame)
            assert badges == [], \
                f"Did not expect badges on GPL platforms, found: " \
                f"{[b.cget('text') for b in badges]}"
        finally:
            root.destroy()

    def test_badge_only_attached_to_cellxgene_in_mixed_list(self):
        """When both kinds are present, only CellxGene_* gets badged."""
        root, frame = _render({
            "GPL570":                    _mk_df(20),
            "CellxGene_kidney_normal_mean": _mk_df(8),
            "GPL96":                     _mk_df(15),
            "CellxGene_lung_disease_sum":  _mk_df(12),
        })
        try:
            badges = _badge_labels(frame)
            assert len(badges) == 2, \
                f"Expected 2 badges (one per CELLxGENE platform), got {len(badges)}"
            for b in badges:
                assert "CELLxGENE Cell Ontology" in b.cget("text")
        finally:
            root.destroy()

    def test_badge_is_visually_distinct(self):
        """Badge must have the light-blue palette so users notice it."""
        root, frame = _render({
            "CellxGene_lung_normal_mean": _mk_df(10),
        })
        try:
            badge = _badge_labels(frame)[0]
            # Light-blue background + navy foreground (Material Blue-50 / 900)
            assert badge.cget("bg") == "#E3F2FD"
            assert badge.cget("fg") == "#0D47A1"
            # Bordered so it reads as a badge, not body text
            assert int(badge.cget("borderwidth")) >= 1
        finally:
            root.destroy()

    def test_badge_packed_below_button_in_same_frame(self):
        """Badge and button live in the same per-platform sub-frame so they
        move together when the layout reflows."""
        root, frame = _render({
            "CellxGene_lung_normal_mean": _mk_df(10),
        })
        try:
            badge = _badge_labels(frame)[0]
            parent = badge.master
            # Same parent should also contain exactly one tk.Button.
            buttons = [c for c in parent.winfo_children()
                       if isinstance(c, tk.Button)]
            assert len(buttons) == 1, \
                f"Expected 1 button next to the badge, got {len(buttons)}"
            # Button text should include the platform name.
            assert "CellxGene_lung_normal_mean" in buttons[0].cget("text")
        finally:
            root.destroy()

    def test_no_false_positive_on_substring_match(self):
        """A platform that merely *contains* 'CellxGene' but doesn't start
        with the prefix must NOT be badged."""
        root, frame = _render({
            "MyCellxGene_custom": _mk_df(5),   # doesn't start with prefix
            "GPL_CellxGene_x":    _mk_df(5),   # doesn't start with prefix
        })
        try:
            badges = _badge_labels(frame)
            assert badges == [], \
                "Badge should only attach when name STARTS WITH 'CellxGene_'"
        finally:
            root.destroy()


class TestEmptyState:
    def test_no_badge_when_no_platforms_loaded(self):
        root, frame = _render({})
        try:
            assert _badge_labels(frame) == []
        finally:
            root.destroy()
