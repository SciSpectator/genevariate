"""Headless chart rendering + data-grounded chart interpretation."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # matplotlib is a core dep, but stay import-guarded like the rest of the app
    from matplotlib.figure import Figure
    _HAS_MPL = True
except Exception:  # pragma: no cover - only when matplotlib is absent
    Figure = None  # type: ignore
    _HAS_MPL = False

try:
    from genevariate.utils.viz_style import AERO, apply_genevariate_style
    apply_genevariate_style()
    _C_A = AERO.get("accent", "#1E90E0")       # blue
    _C_B = AERO.get("green_dark", "#2E7D32")   # green (Frutiger-Aero, never orange)
    _C_LINE = AERO.get("text", "#0E2A45")
    _C_MUTED = AERO.get("muted", "#5F7D95")
    _C_A_DARK = AERO.get("accent_dark", "#0A5B9A")
except Exception:  # pragma: no cover
    _C_A, _C_B, _C_LINE = "#1E90E0", "#2E7D32", "#0E2A45"
    _C_MUTED, _C_A_DARK = "#5F7D95", "#0A5B9A"


# ---------------------------------------------------------------- descriptors
from genevariate.core.analysis.figures import (  # noqa: E402
    describe_values, draw_distribution_overlay, draw_gene_histogram,
    draw_distance_heatmap, draw_group_scatter, draw_group_violin,
    draw_gsea_terms,
    draw_top_differences,
    draw_unique_gene_counts,
    gsea_term_rows,
)


def _shape_phrase(d: Mapping[str, Any]) -> str:
    if not d.get("n"):
        return "no data"
    sk = float(d.get("skew", 0.0) or 0.0)
    sym = ("roughly symmetric" if abs(sk) < 0.5 else
           ("right-skewed" if sk > 0 else "left-skewed"))
    modes = int(d.get("n_modes", 1) or 1)
    mode_txt = ("single peak" if modes <= 1 else
                ("two peaks" if modes == 2 else "several peaks"))
    cls = d.get("class") or "?"
    return f"{cls.lower()}, {mode_txt}, {sym} (skew={sk:+.2f})"


# ------------------------------------------------------------------- builders
def fig_histogram(values: Sequence[float], gene: str, label: str = "",
                  dist_class: str = "", x_label: str = "Expression",
                  bounds: Optional[Tuple[float, float]] = None,
                  ) -> Tuple[Any, Dict[str, Any]]:
    """The Explorer's histogram on a bare ``Figure``, for an executor thread."""
    desc = describe_values(values, dist_class)
    if not _HAS_MPL:
        return None, desc
    fig = Figure(figsize=(6.2, 3.6), dpi=100)
    ax = fig.add_subplot(111)
    desc = draw_gene_histogram(ax, values, gene=gene, label=label,
                               dist_class=dist_class, x_label=x_label,
                               bounds=bounds)
    desc.pop("bins", None); desc.pop("patches", None); desc.pop("modes", None)
    fig.tight_layout()
    return fig, desc


def fig_overlay(vectors: Mapping[str, Sequence[float]], gene: str
                ) -> Tuple[Any, Dict[str, Any]]:
    """The Compare window's overlay on a bare ``Figure``."""
    per: Dict[str, Dict[str, Any]] = {
        k: describe_values(v) for k, v in vectors.items()}
    means = {k: d.get("mean") for k, d in per.items() if d.get("n")}
    desc: Dict[str, Any] = {"gene": gene, "per_source": per}
    if len(means) >= 2:
        hi = max(means, key=means.get); lo = min(means, key=means.get)
        desc["higher"] = hi; desc["lower"] = lo
        desc["mean_gap"] = float(means[hi] - means[lo])
    if not _HAS_MPL:
        return None, desc
    fig = Figure(figsize=(6.2, 3.6), dpi=100)
    ax = fig.add_subplot(111)
    shared = draw_distribution_overlay(ax, vectors, x_label="expression",
                                       title=f"{gene} across sources")
    if shared.get("handles"):
        ax.legend(handles=shared["handles"], fontsize=8, frameon=False)
    desc.update({k: v for k, v in shared.items()
                 if k not in ("handles", "artists")})
    desc["gene"] = gene
    fig.tight_layout()
    return fig, desc


def fig_from_enrichment(ranked: Any, gsea: Any, title: str
                        ) -> Tuple[Any, Dict[str, Any]]:
    """The enrichment window's term chart on a bare ``Figure``."""
    _, desc = gsea_term_rows(gsea, ranked)
    desc["title"] = title
    if not _HAS_MPL or not desc["top"]:
        return None, desc
    fig = Figure(figsize=(6.4, max(3.0, 0.32 * len(desc["top"]) + 1.0)), dpi=100)
    ax = fig.add_subplot(111)
    desc = draw_gsea_terms(ax, gsea, ranked, title=title)
    fig.tight_layout()
    return fig, desc


# ------------------------------------------------------- markdown descriptors
def describe_distribution_block(desc: Mapping[str, Any]) -> str:
    if not desc.get("n"):
        return ""
    return (
        "\n## What the chart shows\n"
        f"- **Shape**: {_shape_phrase(desc)}\n"
        f"- **Center**: mean {desc['mean']:.3g}, median {desc['median']:.3g}\n"
        f"- **Spread**: std {desc['std']:.3g} "
        f"(CV {desc.get('cv', float('nan')):.2f}), IQR {desc.get('iqr', 0):.3g}\n"
        f"- **Outliers**: {desc.get('n_outliers', 0)} beyond 1.5×IQR\n"
        f"- **Range**: {desc['min']:.3g} - {desc['max']:.3g}\n"
    )


def describe_overlay_block(desc: Mapping[str, Any]) -> str:
    per = desc.get("per_source") or {}
    if not per:
        return ""
    lines = ["\n## What the chart shows"]
    for k, d in per.items():
        if d.get("n"):
            lines.append(f"- **{k}**: mean {d['mean']:.3g}, median "
                         f"{d['median']:.3g}, spread (std) {d['std']:.3g}")
        else:
            lines.append(f"- **{k}**: no data")
    if "higher" in desc:
        lines.append(f"- **Difference**: {desc['higher']} sits higher than "
                     f"{desc['lower']} by {desc.get('mean_gap', 0):.3g} on average.")
    return "\n".join(lines) + "\n"


def fig_group_violin(groups: Mapping[str, Sequence[float]], title: str,
                     value_label: str = "expression", top: int = 12,
                     ) -> Tuple[Any, Dict[str, Any]]:
    """The Separation tab's violins on a bare ``Figure``."""
    if not _HAS_MPL:
        return None, {"title": title, "n_groups": 0, "hidden": 0, "groups": []}
    n = min(len([g for g in groups.values()]), top)
    fig = Figure(figsize=(max(6.4, 0.9 * max(n, 1) + 2.0), 4.2), dpi=100)
    desc = draw_group_violin(fig.add_subplot(111), groups, title=title,
                             value_label=value_label, top=top)
    fig.tight_layout()
    return fig, desc


def fig_distance_heatmap(matrix: Any, title: str,
                         ) -> Tuple[Any, Dict[str, Any]]:
    """The Distance tab's heatmap on a bare ``Figure``."""
    names = [str(x) for x in matrix.index]
    if not _HAS_MPL or not names:
        return None, {"title": title, "n_groups": len(names)}
    side = max(4.0, 0.55 * len(names) + 2.2)
    fig = Figure(figsize=(side, side * 0.85), dpi=100)
    desc = draw_distance_heatmap(fig.add_subplot(111), matrix, title=title)
    fig.tight_layout()
    return fig, desc


def describe_group_violin_block(desc: Mapping[str, Any]) -> str:
    rows = desc.get("groups") or []
    if not rows:
        return ""
    lines = ["\n## What the chart shows"]
    top = max(rows, key=lambda r: r["median"])
    bot = min(rows, key=lambda r: r["median"])
    lines.append(f"- **Highest**: {top['group']} (median {top['median']:.3g}); "
                 f"**lowest**: {bot['group']} (median {bot['median']:.3g}).")
    widest = max(rows, key=lambda r: r["iqr"])
    lines.append(f"- **Widest**: {widest['group']} spans an IQR of "
                 f"{widest['iqr']:.3g} - a wide violin means the group is not "
                 f"one population, and its mean is not a summary of it.")
    if desc.get("hidden"):
        lines.append(f"- {desc['hidden']} smaller group(s) are not drawn.")
    return "\n".join(lines) + "\n"


def describe_bar_block(desc: Mapping[str, Any]) -> str:
    top = desc.get("top") or []
    if not top:
        return ""
    n_total = desc.get("n_total")
    listed = top[:8]
    n_ref = n_total if n_total and n_total > len(listed) else len(top)
    of = (f" of {n_ref}" if n_ref > len(listed) else "")
    lines = ["\n## What the chart shows",
             f"- **Top {len(listed)}{of}** by |{'score'}|:"]
    for t in listed:
        arrow = "▲" if t["value"] > 0 else ("▼" if t["value"] < 0 else "•")
        lines.append(f"  - {arrow} {t['label']}: {t['value']:.3g}")
    if desc.get("n_up") or desc.get("n_down"):
        lines.append(f"- **Direction**: {desc.get('n_up', 0)} up, "
                     f"{desc.get('n_down', 0)} down among the top bars.")
    return "\n".join(lines) + "\n"


def fig_scatter(coords, groups: Sequence[str], title: str,
                xlabel: str = "1", ylabel: str = "2", max_groups: int = 12,
                ) -> Tuple[Any, Dict[str, Any]]:
    """The projection scatter on a bare ``Figure``."""
    if not _HAS_MPL:
        return None, {"title": title, "n": 0, "groups": [], "hidden": 0}
    fig = Figure(figsize=(6.6, 5.0), dpi=100)
    desc = draw_group_scatter(fig.add_subplot(111), coords, groups,
                              title=title, xlabel=xlabel, ylabel=ylabel,
                              max_groups=max_groups)
    desc.pop("artists", None)
    fig.tight_layout()
    return fig, desc


def _bare_bars(drawer, *args, **kw) -> Tuple[Any, Dict[str, Any]]:
    """Own a Figure for one of the shared bar routines and hand it back."""
    if not _HAS_MPL:
        return None, {}
    fig = Figure(figsize=(6.4, 4.2), dpi=100)
    desc = drawer(fig.add_subplot(111), *args, **kw)
    fig.tight_layout()
    return fig, desc


def fig_unique_gene_counts(unique_genes) -> Tuple[Any, Dict[str, Any]]:
    """The Gene Overlap bar, drawn by the routine the export also calls."""
    return _bare_bars(draw_unique_gene_counts, unique_genes)


def fig_top_differences(de_genes, top: int = 12) -> Tuple[Any, Dict[str, Any]]:
    """The cross-platform difference bar, drawn by the shared routine."""
    return _bare_bars(draw_top_differences, de_genes, top=top)
