"""Headless chart rendering + data-grounded chart interpretation.

This module is deliberately Tk-free: figures are built with the Matplotlib
``Figure`` object (Agg), never ``pyplot`` global state, so executors can create
them on a worker thread and the sidebar embeds them on the main thread.

Every builder returns ``(figure, descriptor)`` where ``descriptor`` is a plain
dict of the numbers a human would read off the chart (center, spread, shape,
outliers, top terms, direction). ``descriptor_block`` turns that dict into a
short markdown "What the chart shows" section that is appended to the tool
report — which is what gives the local text model *chart understanding*: it
narrates the graph from the extracted numbers rather than from pixels.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
except Exception:  # pragma: no cover
    _C_A, _C_B, _C_LINE = "#1E90E0", "#2E7D32", "#0E2A45"


# ---------------------------------------------------------------- descriptors
def describe_values(values: Sequence[float], dist_class: str = "") -> Dict[str, Any]:
    """Extract the numbers a reader would take off a distribution chart."""
    v = np.asarray(list(values), dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    q1, med, q3 = (float(x) for x in np.percentile(v, [25, 50, 75]))
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(np.count_nonzero((v < lo) | (v > hi)))
    mean, std = float(np.mean(v)), float(np.std(v, ddof=1)) if v.size > 1 else 0.0
    try:
        from scipy.stats import skew as _skew
        skew = float(_skew(v)) if v.size > 2 else 0.0
    except Exception:
        skew = 0.0
    cls = (dist_class or "").strip()
    n_modes = 2 if cls == "Bimodal" else (3 if cls == "Multimodal" else 1)
    return {
        "n": int(v.size), "mean": mean, "median": med, "std": std,
        "cv": (std / mean) if mean else float("nan"),
        "iqr": iqr, "skew": skew, "n_modes": n_modes, "n_outliers": n_out,
        "min": float(v.min()), "max": float(v.max()), "class": cls,
    }


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
                  dist_class: str = "") -> Tuple[Any, Dict[str, Any]]:
    """Single-gene distribution histogram + its descriptor."""
    desc = describe_values(values, dist_class)
    if not _HAS_MPL:
        return None, desc
    v = np.asarray(list(values), dtype=float)
    v = v[np.isfinite(v)]
    fig = Figure(figsize=(6.2, 3.6), dpi=100)
    ax = fig.add_subplot(111)
    if v.size:
        bins = int(np.clip(np.sqrt(v.size), 8, 40))
        ax.hist(v, bins=bins, color=_C_A, alpha=0.85, edgecolor="white",
                linewidth=0.5)
        ax.axvline(desc["mean"], color=_C_B, ls="--", lw=1.6, label="mean")
        ax.axvline(desc["median"], color=_C_LINE, ls=":", lw=1.6, label="median")
        ax.legend(fontsize=8, frameon=False)
    title = f"{gene} distribution" + (f" — {label}" if label else "")
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xlabel("expression"); ax.set_ylabel("samples")
    fig.tight_layout()
    return fig, desc


def fig_overlay(vectors: Mapping[str, Sequence[float]], gene: str
                ) -> Tuple[Any, Dict[str, Any]]:
    """Overlaid per-source histograms for one gene + a comparison descriptor."""
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
    palette = [_C_A, _C_B, "#0A5B9A", "#4CAF50", "#3AA6B9"]
    for i, (k, vec) in enumerate(vectors.items()):
        v = np.asarray(list(vec), dtype=float); v = v[np.isfinite(v)]
        if not v.size:
            continue
        bins = int(np.clip(np.sqrt(v.size), 8, 30))
        ax.hist(v, bins=bins, alpha=0.5, label=f"{k} (n={v.size})",
                color=palette[i % len(palette)], edgecolor="white", linewidth=0.4)
    ax.set_title(f"{gene} across sources", fontsize=11, weight="bold")
    ax.set_xlabel("expression"); ax.set_ylabel("samples")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig, desc


def fig_bar(labels: Sequence[str], values: Sequence[float], title: str,
            xlabel: str = "score", top: int = 12,
            ) -> Tuple[Any, Dict[str, Any]]:
    """Horizontal bar of the top |value| entries (enrichment terms / ranked genes)."""
    pairs = [(str(l), float(x)) for l, x in zip(labels, values)
             if np.isfinite(x)]
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    pairs = pairs[:top]
    desc = {"title": title,
            "top": [{"label": l, "value": round(x, 4)} for l, x in pairs],
            "n_up": sum(1 for _, x in pairs if x > 0),
            "n_down": sum(1 for _, x in pairs if x < 0)}
    if not _HAS_MPL or not pairs:
        return None, desc
    pairs = pairs[::-1]  # largest at top
    labs = [l if len(l) <= 42 else l[:39] + "…" for l, _ in pairs]
    vals = [x for _, x in pairs]
    colors = [_C_A if x >= 0 else _C_B for x in vals]
    fig = Figure(figsize=(6.4, max(3.0, 0.32 * len(pairs) + 1.0)), dpi=100)
    ax = fig.add_subplot(111)
    ax.barh(range(len(vals)), vals, color=colors, edgecolor="white", linewidth=0.4)
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(labs, fontsize=8)
    ax.axvline(0, color=_C_LINE, lw=0.8)
    ax.set_title(title, fontsize=11, weight="bold"); ax.set_xlabel(xlabel)
    fig.tight_layout()
    return fig, desc


def fig_from_enrichment(ranked: Any, gsea: Any, title: str
                        ) -> Tuple[Any, Dict[str, Any]]:
    """Bar chart for an enrichment result: GSEA NES per term when terms passed
    the filter, otherwise the top differential genes from the ranked table."""
    # 1) prefer real enriched GSEA terms (NES bars)
    try:
        g = gsea
        if g is not None and hasattr(g, "columns") and len(g):
            cols = {c.lower(): c for c in g.columns}
            if "nes" in cols and "error" not in cols:
                term = cols.get("term") or cols.get("name") or g.columns[0]
                sub = g[[term, cols["nes"]]].dropna()
                if len(sub):
                    fig, desc = fig_bar(sub[term].astype(str).tolist(),
                                        sub[cols["nes"]].tolist(),
                                        title, xlabel="NES")
                    desc["kind"] = "gsea"
                    return fig, desc
    except Exception:
        pass
    # 2) fall back to the ranked gene table
    try:
        r = ranked
        if r is not None and hasattr(r, "columns") and len(r):
            cols = {c.lower(): c for c in r.columns}
            val = (cols.get("rank") or cols.get("t_stat") or cols.get("stat")
                   or cols.get("logfc") or cols.get("score"))
            labels = (r.index.astype(str).tolist()
                      if r.index.name or not cols.get("gene")
                      else r[cols["gene"]].astype(str).tolist())
            if val:
                fig, desc = fig_bar(labels, r[val].tolist(),
                                    title, xlabel=val)
                desc["kind"] = "ranked"
                return fig, desc
    except Exception:
        pass
    return None, {"title": title, "top": []}


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
        f"- **Range**: {desc['min']:.3g} – {desc['max']:.3g}\n"
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


def describe_bar_block(desc: Mapping[str, Any]) -> str:
    top = desc.get("top") or []
    if not top:
        return ""
    lines = ["\n## What the chart shows",
             f"- **Top {len(top)}** by |{'score'}|:"]
    for t in top[:8]:
        arrow = "▲" if t["value"] > 0 else ("▼" if t["value"] < 0 else "•")
        lines.append(f"  - {arrow} {t['label']}: {t['value']:.3g}")
    if desc.get("n_up") or desc.get("n_down"):
        lines.append(f"- **Direction**: {desc.get('n_up', 0)} up, "
                     f"{desc.get('n_down', 0)} down among the top bars.")
    return "\n".join(lines) + "\n"
