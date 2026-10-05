"""The program's drawing routines, in one place and free of Tkinter."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = ["describe_values", "draw_gene_histogram", "histogram_title"]

_REGION_FILL = "#1E90E0"
_REGION_EDGE = "#0A5B9A"
_MODE_LINE = "#7D3C98"


def describe_values(values: Sequence[float],
                    dist_class: str = "") -> Dict[str, Any]:
    """The numbers a reader would take off a distribution chart."""
    v = np.asarray(list(values), dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    q1, med, q3 = (float(x) for x in np.percentile(v, [25, 50, 75]))
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(np.count_nonzero((v < lo) | (v > hi)))
    mean = float(np.mean(v))
    std = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
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


def histogram_title(head: str, dist_class: str, desc: Mapping[str, Any],
                    modes: Sequence[Tuple[float, float]]) -> str:
    """The Explorer's title line, so both routes name a chart the same way."""
    n = int(desc.get("n", 0) or 0)
    if len(modes) > 1:
        shape = "  ".join(f"{x:.1f}" for x, _ in modes)
        return f"{head} | {dist_class} | n={n:,} | modes at {shape}"
    return (f"{head} | {dist_class or '?'} | n={n:,} | "
            f"u={desc.get('mean', float('nan')):.2f} "
            f"SD={desc.get('std', float('nan')):.2f}")


def draw_gene_histogram(ax, values: Sequence[float], *,
                        gene: str = "",
                        label: str = "",
                        dist_class: str = "",
                        x_label: str = "Expression",
                        bounds: Optional[Tuple[float, float]] = None,
                        title: bool = True) -> Dict[str, Any]:
    """Draw one gene's distribution on *ax*, the way the Explorer draws it."""
    from genevariate.config import CONFIG
    from genevariate.core.analysis.bimodality import (
        BIMODAL_TAGS, density_modes, optimal_bins, robust_kde,
    )

    cfg = CONFIG['plotting']['histogram']
    v = np.asarray(list(values), dtype=float)
    v = v[np.isfinite(v)]
    desc = describe_values(v, dist_class)
    desc["bins"], desc["patches"], desc["modes"] = None, None, []
    if not v.size:
        return desc

    modes: list = []
    if dist_class in BIMODAL_TAGS:
        try:
            modes = density_modes(v)
        except Exception:
            modes = []

    counts, bins, patches = ax.hist(
        v, bins=optimal_bins(v, method='auto'), density=True,
        edgecolor=cfg['edge_color'], alpha=cfg['alpha'],
        color=cfg['default_color'], linewidth=0.5)
    y_max = max(counts) if len(counts) else 0

    if (v.size >= cfg['min_samples_for_kde']
            and np.var(v) > cfg['min_variance_for_kde']):
        try:
            x_range = np.linspace(v.min(), v.max(), 200)
            kde_vals = robust_kde(v)(x_range)
            ax.plot(x_range, kde_vals, 'r-', alpha=0.6, linewidth=2,
                    label='KDE')
            ax.legend(loc='upper right', fontsize=7, framealpha=0.7)
            y_max = max(y_max, kde_vals.max())
        except Exception:
            pass
    if y_max > 0:
        ax.set_ylim(0, y_max * 1.15)

    for x_mode, _ in (modes if len(modes) > 1 else []):
        ax.axvline(x_mode, color=_MODE_LINE, linestyle='--',
                   linewidth=1.2, alpha=0.85, zorder=3)

    if bounds is not None:
        lo, hi = float(bounds[0]), float(bounds[1])
        n_in = int(((v >= lo) & (v <= hi)).sum())
        ax.axvspan(lo, hi, color=_REGION_FILL, alpha=0.18, zorder=0,
                   label=f"region {lo:.3g}-{hi:.3g} (n={n_in:,})")
        ax.axvline(lo, color=_REGION_EDGE, lw=1.4, zorder=2)
        ax.legend(loc='upper right', fontsize=7, framealpha=0.7)
        desc["region"] = {"low": lo, "high": hi, "n_in": n_in,
                          "pct_in": round(100.0 * n_in / max(1, v.size), 2)}

    if title:
        head = f"{label} - {gene}" if label else gene
        ax.set_title(histogram_title(head, dist_class, desc, modes),
                     fontsize=10, weight='bold', pad=15)
    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2, linestyle='--')

    desc["bins"], desc["patches"], desc["modes"] = bins, patches, modes
    return desc


# ── Region enrichment ──────────────────────────────────────────────────────
_SIG_COLOR = {"***": "#C62828", "**": "#E53935", "*": "#EF9A9A"}
_SIG_NS = "#BDBDBD"
_SEL_COLOR = "#C62828"
_BG_COLOR = "#78909C"


def _trunc(s, m=28):
    s = "".join(ch if ord(ch) >= 32 else " " for ch in str(s)).strip()
    return (s[:m - 1] + "..") if len(s) > m else s


def enrichment_rows(table) -> list:
    """``region_label_enrichment`` output as the row dicts the plot reads."""
    out = []
    for rec in table.to_dict("records"):
        n_gse = rec.get("n_gse")
        n_gse = (None if n_gse is None or not np.isfinite(n_gse)
                 else int(n_gse))
        out.append({
            "Region": rec["region"], "Label Column": rec["label_column"],
            "Value": rec["value"],
            "a": rec["a"], "n_sel": rec["n_region"],
            "c": rec["c"], "n_non": rec["n_background"],
            "Sel%": rec["region_pct"], "BG%": rec["background_pct"],
            "Enrichment": rec["enrichment"], "p-value": rec["p_value"],
            "padj": rec["q_value"], "Sig": rec["significance"],
            "n_gse": n_gse, "rho": rec.get("rho"), "n_eff": rec.get("n_eff"),
            "ci_low": rec.get("ci_low"), "ci_high": rec.get("ci_high"),
        })
    return out


def draw_enrichment(ax_freq, ax_fold, rows: Sequence[Mapping[str, Any]], *,
                    label_column: str) -> Dict[str, Any]:
    """Draw one region x label-column enrichment onto two axes."""
    rows = list(rows)
    if not rows:
        return {"label_column": label_column, "n": 0}
    n_sel = rows[0]["n_sel"]
    n_non = rows[0]["n_non"]

    labels = [_trunc(r["Value"]) for r in rows]
    sel_pcts = [r["Sel%"] for r in rows]
    bg_pcts = [r["BG%"] for r in rows]
    sigs = [r["Sig"] for r in rows]
    y_pos = np.arange(len(rows))
    bar_h = 0.35

    ax_freq.barh(y_pos - bar_h / 2, sel_pcts, bar_h,
                 label=f"Selected Region (n={n_sel})",
                 color=_SEL_COLOR, edgecolor="black", lw=0.4, alpha=0.85)
    ax_freq.barh(y_pos + bar_h / 2, bg_pcts, bar_h,
                 label=f"Rest of Loaded Samples (n={n_non:,})",
                 color=_BG_COLOR, edgecolor="black", lw=0.4, alpha=0.65)
    for i, r in enumerate(rows):
        ax_freq.text(max(sel_pcts[i] + 0.5, 1), y_pos[i] - bar_h / 2,
                     f" {r['a']}/{n_sel}  {r['Sig']}", va="center", fontsize=7,
                     fontweight="bold" if r["Sig"] != "ns" else "normal",
                     color=_SEL_COLOR if r["Sig"] != "ns" else "#999")
        ax_freq.text(max(bg_pcts[i] + 0.5, 1), y_pos[i] + bar_h / 2,
                     f" {r['c']}/{n_non:,}", va="center", fontsize=6.5,
                     color="#546E7A")
    ax_freq.set_xlim(0, max(sel_pcts + bg_pcts + [1.0]) * 1.38)
    ax_freq.set_yticks(y_pos)
    ax_freq.set_yticklabels(labels, fontsize=7.5)
    ax_freq.set_xlabel("Frequency (%)", fontsize=9)
    ax_freq.set_title(f"Selected vs rest - {label_column}", fontsize=10,
                      weight="bold")
    ax_freq.legend(fontsize=9, loc="lower right", framealpha=0.9)
    ax_freq.invert_yaxis()
    ax_freq.grid(axis="x", alpha=0.2)

    finite = [r["Enrichment"] for r in rows
              if np.isfinite(r["Enrichment"]) and r["Enrichment"] > 0]
    top = max(finite) if finite else 1.0
    inf_at = top * 2.2
    fold = [r["Enrichment"] if np.isfinite(r["Enrichment"]) and r["Enrichment"] > 0
            else inf_at for r in rows]
    colors = [_SIG_COLOR.get(s, _SIG_NS) for s in sigs]
    base = min([f for f in fold if f > 0] + [1.0]) * 0.5
    base = min(base, 0.5)
    ax_fold.barh(y_pos, [f - base for f in fold], 0.55, left=base,
                 color=colors, edgecolor="black", lw=0.4, alpha=0.85)
    ax_fold.set_xscale("log")
    ax_fold.axvline(1.0, color="black", ls="--", lw=1, alpha=0.5)
    for i, r in enumerate(rows):
        q = r.get("padj", float("nan"))
        q_str = (f"q={q:.1e}" if np.isfinite(q) and q < 0.01
                 else f"q={q:.3f}" if np.isfinite(q) else "q=n/a")
        gse = r.get("n_gse")
        gse_str = "" if gse is None else f"  [{gse} GSE]"
        shown = (f"{r['Enrichment']:.1f}x" if np.isfinite(r["Enrichment"])
                 else "infinite")
        ax_fold.text(fold[i] * 1.06, y_pos[i],
                     f" {shown}  {q_str}{gse_str}",
                     va="center", fontsize=6.5,
                     fontweight="bold" if r["Sig"] != "ns" else "normal",
                     color=colors[i])
        lo, hi = r.get("ci_low", float("nan")), r.get("ci_high", float("nan"))
        if np.isfinite(lo) and np.isfinite(hi) and lo > 0:
            ax_fold.plot([lo, hi], [y_pos[i], y_pos[i]], color="#37474F",
                         lw=1.1, alpha=0.9, zorder=4)
            for _end in (lo, hi):
                ax_fold.plot([_end, _end], [y_pos[i] - 0.14, y_pos[i] + 0.14],
                             color="#37474F", lw=1.1, alpha=0.9, zorder=4)
    hi_ci = [r.get("ci_high", 0.0) for r in rows
             if np.isfinite(r.get("ci_high", float("nan")))]
    ax_fold.set_xlim(base, max(fold + hi_ci + [1.0]) * 6.0)
    ax_fold.set_yticks(y_pos)
    ax_fold.set_yticklabels(["" for _ in rows])
    ax_fold.set_xlabel(
        "Enrichment ratio (fold, log axis; whiskers = 95% CI by study)",
        fontsize=9)
    ax_fold.set_title(f"Enrichment - {label_column}", fontsize=10, weight="bold")
    ax_fold.invert_yaxis()
    ax_fold.grid(axis="x", alpha=0.2)
    try:
        from matplotlib.patches import Patch
        ax_fold.legend(handles=[
            Patch(facecolor=_SIG_COLOR["***"], label="q<0.001 ***"),
            Patch(facecolor=_SIG_COLOR["**"], label="q<0.01 **"),
            Patch(facecolor=_SIG_COLOR["*"], label="q<0.05 *"),
            Patch(facecolor=_SIG_NS, label="ns"),
        ], fontsize=6, loc="lower right", framealpha=0.9)
    except Exception:
        pass

    single_study = sum(1 for r in rows
                       if r.get("n_gse") is not None and r["n_gse"] < 3)
    return {
        "label_column": label_column, "n": len(rows),
        "n_selected": n_sel, "n_background": n_non,
        "single_study": single_study,
    }


# ── Pooled effect forest ───────────────────────────────────────────
def draw_pooled_forest(ax, rows, *, gene: str, label_column: str,
                       palette: Optional[Mapping[str, str]] = None,
                       ) -> Dict[str, Any]:
    """Pooled log odds ratio per label value, with each platform marked."""
    p = dict(palette or {})
    c_conc = p.get("green_dark", "#2E7D32")
    c_sig = p.get("accent", "#1E90E0")
    c_muted = p.get("muted", "#5F7D95")

    from genevariate.utils import display_limits as _dl
    n_given = len(rows)
    rows = [r for r in rows if r.get("k", 0) >= 2]
    n_poolable = len(rows)
    n_single = n_given - n_poolable
    rows = _dl.head(rows, "enrichment_rows")
    if not rows:
        return {"gene": gene, "label_column": label_column, "n": 0,
                "n_total": n_given, "n_poolable": 0,
                "n_single_platform": n_single}
    ys = np.arange(len(rows))[::-1]
    for y, r in zip(ys, rows):
        lo = np.log(max(r["ci_low"], 1e-6))
        hi = np.log(max(r["ci_high"], 1e-6))
        sig = np.isfinite(r["q"]) and r["q"] < 0.05
        col = (c_conc if sig and r.get("concordant") else
               c_sig if sig else c_muted)
        ax.plot([lo, hi], [y, y], color=col, lw=1.6, zorder=2)
        ax.plot([r["pooled_log_or"]], [y], "o", color=col, ms=5, zorder=3)
        for pv in (r.get("per_platform") or {}).values():
            ax.plot([pv["log_or"]], [y], "|", color="#5A6B7A", ms=7, mew=1.0,
                    zorder=4)
    ax.axvline(0.0, color="#888", lw=1.0, ls="--", zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([_trunc(r["value"], 26) for r in rows], fontsize=8)
    ax.set_xlabel("Pooled log odds ratio for the label, inside versus outside the region",
                  fontsize=9)
    _parts = [x for x in (
        _dl.cap_note(len(rows), n_poolable, noun="values"),
        (f"{n_single:,} of {n_given:,} values occur on one platform and are not pooled"
         if n_single else "")) if x]
    ax.set_title(f"{gene} - {label_column}  (bars 95% CI; "
                 f"ticks = single platforms)"
                 + (f"  ({'; '.join(_parts)})" if _parts else ""), fontsize=9)

    n_sig = sum(1 for r in rows if np.isfinite(r["q"]) and r["q"] < 0.05)
    return {
        "gene": gene, "label_column": label_column, "n": len(rows),
        "n_total": n_given,
        "n_poolable": n_poolable,
        "n_single_platform": n_single,
        "n_sig": n_sig,
        "n_concordant": sum(1 for r in rows if r.get("concordant")),
    }


_TERM_UP = "#2E7D63"
_TERM_DOWN = "#B03A2E"
_TERM_AXIS = "#7F8C8D"


def gsea_term_rows(gsea, ranked, *, top: int = 12) -> Tuple[list, Dict[str, Any]]:
    """The bars a gene-set result stands for, and what was left out."""
    pairs: list = []
    kind = ""
    for frame, keys, k in ((gsea, ("nes",), "gsea"),
                           (ranked, ("rank", "t_stat", "stat", "logfc",
                                     "score"), "ranked")):
        if pairs or frame is None or not hasattr(frame, "columns") or not len(frame):
            continue
        cols = {str(c).lower(): c for c in frame.columns}
        if k == "gsea" and "error" in cols:
            continue
        val = next((cols[x] for x in keys if x in cols), None)
        if val is None:
            continue
        if k == "gsea":
            term = cols.get("term") or cols.get("name") or frame.columns[0]
            names = frame[term].astype(str).tolist()
        elif cols.get("gene"):
            names = frame[cols["gene"]].astype(str).tolist()
        else:
            names = frame.index.astype(str).tolist()
        pairs = [(str(n), float(v)) for n, v in zip(names, frame[val].tolist())
                 if np.isfinite(v)]
        kind = k
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    n_given = len(pairs)
    pairs = pairs[:top]
    desc = {"top": [{"label": n, "value": round(v, 4)} for n, v in pairs],
            "n_total": n_given,
            "hidden": n_given - len(pairs),
            "n_up": sum(1 for _, v in pairs if v > 0),
            "n_down": sum(1 for _, v in pairs if v < 0),
            "kind": kind}
    return pairs, desc


def draw_gsea_terms(ax, gsea, ranked, *, title: str = "",
                    top: int = 12) -> Dict[str, Any]:
    """Horizontal bars for a gene-set enrichment result, on a bare axis."""
    pairs, desc = gsea_term_rows(gsea, ranked, top=top)
    desc["title"] = title
    if not pairs:
        ax.text(0.5, 0.5, "No enriched terms and no ranked genes to draw",
                ha="center", va="center", transform=ax.transAxes,
                color=_TERM_AXIS)
        ax.set_axis_off()
        return desc
    pairs = pairs[::-1]                      # largest at the top
    labs = [n if len(n) <= 42 else n[:39] + "…" for n, _ in pairs]
    vals = [v for _, v in pairs]
    ax.barh(range(len(vals)), vals,
            color=[_TERM_UP if v >= 0 else _TERM_DOWN for v in vals],
            edgecolor="white", linewidth=0.4)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labs, fontsize=8)
    ax.axvline(0, color=_TERM_AXIS, lw=0.8)
    ax.set_xlabel("NES" if desc["kind"] == "gsea" else "rank statistic")
    head = title or ("Gene-set enrichment" if desc["kind"] == "gsea"
                     else "Top ranked genes")
    if desc["hidden"]:
        # The cut is named inside the title so it survives export.
        head += f" (top {len(vals)} of {desc['n_total']})"
    ax.set_title(head, fontsize=11, weight="bold")
    return desc


def kde_curve(values: Sequence[float], n: int = 300,
              x_range: Optional[Tuple[float, float]] = None):
    """The density curve the Distribution Classifier counts modes on."""
    from .bimodality import robust_kde
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2 or np.ptp(v) == 0:
        return None
    try:
        k = robust_kde(v)
        if x_range:
            xs = np.linspace(x_range[0], x_range[1], n)
        else:
            bw = k.factor * v.std(ddof=1)
            pad = max(3.0 * bw, 0.05 * np.ptp(v), 0.01)
            xs = np.linspace(v.min() - pad, v.max() + pad, n)
        return xs, np.maximum(k(xs), 0)
    except Exception:
        return None


def draw_distribution_overlay(ax, groups: Mapping[str, Sequence[float]], *,
                              background: Optional[Mapping[str, Sequence[float]]] = None,
                              mode: str = "both", x_label: str = "Expression",
                              title: Optional[str] = None,
                              max_label: int = 35) -> Dict[str, Any]:
    """Overlaid per-group distributions on a bare axis."""
    from genevariate.utils.viz_style import palette_for, rug_alpha
    import matplotlib.lines as mlines

    series = {}
    for k, v in groups.items():
        a = np.asarray(list(v), dtype=float)
        a = a[np.isfinite(a)]
        if a.size:
            series[str(k)] = a

    for bg in (background or {}).values():
        b = np.asarray(list(bg), dtype=float)
        b = b[np.isfinite(b)]
        if b.size > 10:
            ax.hist(b, bins=int(min(150, max(60, np.sqrt(b.size)))),
                    color="#AAAAAA", alpha=0.4, density=True,
                    edgecolor="#777777", linewidth=0.3, zorder=1)

    pool = [a for a in series.values()]
    pool += [np.asarray(list(b), dtype=float) for b in (background or {}).values()]
    pool = [p[np.isfinite(p)] for p in pool if len(p)]
    if pool:
        cat = np.concatenate(pool)
        x_range = (float(cat.min()) * 0.97, float(cat.max()) * 1.03)
    else:
        x_range = None

    colors = palette_for(max(len(series), 1), use_case="discrete")
    per: Dict[str, Dict[str, Any]] = {}
    artists: Dict[str, list] = {}
    handles = []
    for i, (name, a) in enumerate(series.items()):
        per[name] = describe_values(a)
        clr = colors[i % len(colors)]
        short = name if len(name) <= max_label else name[:max_label - 1] + "…"
        lb = f"{short} (n={a.size})"
        arts = []
        if mode in ("density", "both"):
            kd = kde_curve(a, x_range=x_range)
            if kd is not None:
                line, = ax.plot(kd[0], kd[1], color=clr, lw=2.3, alpha=0.88,
                                zorder=4)
                arts.append(line)
            elif a.size == 1:
                arts.append(ax.axvline(a[0], color=clr, ls=":", lw=2.3,
                                       alpha=0.7, zorder=4))
        if mode in ("rug", "both"):
            ax.vlines(a, 0, 0.04, transform=ax.get_xaxis_transform(),
                      color=clr, alpha=rug_alpha(a.size), linewidth=0.8,
                      zorder=5)
            arts.append(ax.collections[-1])
        artists[lb] = arts
        handles.append(mlines.Line2D([], [], color=clr, lw=2.3, label=lb))

    ax.set_xlabel(x_label)
    ax.set_ylabel("Density")
    ax.set_ylim(bottom=0)
    ax.set_title(title or f"Distribution overlay, {len(series)} groups ({mode})",
                 fontsize=13, weight="bold")

    means = {k: d.get("mean") for k, d in per.items() if d.get("n")}
    desc: Dict[str, Any] = {"per_source": per, "n_groups": len(series)}
    if len(means) >= 2:
        hi = max(means, key=means.get)
        lo = min(means, key=means.get)
        desc["higher"], desc["lower"] = hi, lo
        desc["mean_gap"] = float(means[hi] - means[lo])
    desc["handles"] = handles
    desc["artists"] = artists
    return desc


def _bar_desc(pairs, n_given: int, title: str) -> Dict[str, Any]:
    return {"title": title,
            "top": [{"label": n, "value": round(float(v), 4)} for n, v in pairs],
            "n_total": n_given,
            "hidden": n_given - len(pairs)}


def _draw_labelled_bars(ax, pairs, *, n_given: int, title: str, x_label: str,
                        colour_by_sign: bool = True) -> Dict[str, Any]:
    desc = _bar_desc(pairs, n_given, title)
    if not pairs:
        ax.text(0.5, 0.5, "Nothing to draw", ha="center", va="center",
                transform=ax.transAxes, color=_TERM_AXIS)
        ax.set_axis_off()
        return desc
    rows = pairs[::-1]
    labs = [n if len(n) <= 42 else n[:39] + "…" for n, _ in rows]
    vals = [float(v) for _, v in rows]
    colours = ([_TERM_UP if v >= 0 else _TERM_DOWN for v in vals]
               if colour_by_sign else [_TERM_UP] * len(vals))
    ax.barh(range(len(vals)), vals, color=colours, edgecolor="white",
            linewidth=0.4)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labs, fontsize=8)
    ax.axvline(0, color=_TERM_AXIS, lw=0.8)
    ax.set_xlabel(x_label)
    head = title
    if desc["hidden"]:
        head += f" (top {len(vals)} of {n_given})"
    ax.set_title(head, fontsize=11, weight="bold")
    return desc


def draw_unique_gene_counts(ax, unique_genes: Mapping[str, Any], *,
                            title: str = "Genes carried by one platform alone"
                            ) -> Dict[str, Any]:
    """How many genes each platform measures that no other platform does."""
    pairs = [(f"{k} only", len(v)) for k, v in unique_genes.items()]
    pairs.sort(key=lambda p: p[1], reverse=True)
    return _draw_labelled_bars(ax, pairs, n_given=len(pairs), title=title,
                               x_label="genes", colour_by_sign=False)


def draw_top_differences(ax, de_genes: Sequence[Mapping[str, Any]], *,
                         top: int = 12,
                         title: str = "Largest cross-platform differences"
                         ) -> Dict[str, Any]:
    """The genes whose means disagree most between platforms."""
    rows = [(str(g.get("gene", "")), float(g.get("max_abs_delta", 0.0)))
            for g in de_genes if g.get("gene")]
    n_given = len(rows)
    rows.sort(key=lambda p: abs(p[1]), reverse=True)
    return _draw_labelled_bars(ax, rows[:top], n_given=n_given, title=title,
                               x_label="max |delta mean|",
                               colour_by_sign=False)


def draw_distance_heatmap(ax, matrix, *, title: str = "",
                          max_label: int = 20) -> Dict[str, Any]:
    """Annotated square heatmap of a pairwise-distance matrix."""
    names = [str(x) for x in matrix.index]
    m = np.asarray(matrix.to_numpy(), dtype=float)
    off = m[~np.eye(len(names), dtype=bool)] if len(names) > 1 else np.array([])
    desc: Dict[str, Any] = {"title": title, "n_groups": len(names),
                            "max": float(np.nanmax(off)) if off.size else None,
                            "min": float(np.nanmin(off)) if off.size else None}
    if off.size:
        i, j = np.unravel_index(np.nanargmax(np.where(
            np.eye(len(names), dtype=bool), -np.inf, m)), m.shape)
        desc["furthest"] = [names[i], names[j]]
    if not names:
        return desc

    im = ax.imshow(m, cmap="Blues", aspect="auto")
    ax.figure.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    short = [n if len(n) <= max_label else n[:max_label - 3] + "…" for n in names]
    ax.set_xticklabels(short, rotation=40, ha="right", fontsize=7)
    ax.set_yticklabels(short, fontsize=7)
    hi = np.nanmax(m) if np.isfinite(m).any() else 1.0
    for i in range(len(names)):
        for j in range(len(names)):
            if np.isfinite(m[i, j]):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center",
                        fontsize=6,
                        color="white" if m[i, j] > hi * 0.6 else "#0E2A45")
    ax.set_title(title, fontsize=11, weight="bold")
    return desc


def draw_group_violin(ax, groups: Mapping[str, Sequence[float]], *,
                      title: str = "", value_label: str = "expression",
                      top: int = 12) -> Dict[str, Any]:
    """One violin per group, largest first, with the box kept inside."""
    usable = []
    for name, values in groups.items():
        v = np.asarray(list(values), dtype=float)
        v = v[np.isfinite(v)]
        if v.size >= 2:
            usable.append((str(name), v))
    usable.sort(key=lambda p: p[1].size, reverse=True)
    hidden = max(0, len(usable) - top)
    usable = usable[:top]
    desc: Dict[str, Any] = {
        "title": title, "n_groups": len(usable), "hidden": hidden,
        "groups": [{"group": n, "n": int(v.size), "median": float(np.median(v)),
                    "iqr": float(np.subtract(*np.percentile(v, [75, 25])))}
                   for n, v in usable]}
    if not usable:
        ax.text(0.5, 0.5, "No group has two values to draw",
                ha="center", va="center", transform=ax.transAxes,
                color=_TERM_AXIS)
        ax.set_axis_off()
        return desc

    data = [v for _, v in usable]
    pos = np.arange(1, len(data) + 1)
    parts = ax.violinplot(data, positions=pos, showextrema=False, widths=0.85)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor("#1E90E0" if i % 2 == 0 else "#2E7D32")
        body.set_edgecolor("#0E2A45")
        body.set_alpha(0.45)
        body.set_linewidth(0.6)
    ax.boxplot(data, positions=pos, widths=0.10, showfliers=False,
               medianprops=dict(color="#0E2A45", lw=1.4),
               boxprops=dict(color="#0E2A45", lw=0.9),
               whiskerprops=dict(color="#0E2A45", lw=0.9),
               capprops=dict(color="#0E2A45", lw=0.9))
    ax.set_xticks(pos)
    ax.set_xticklabels(
        [(n if len(n) <= 18 else n[:15] + "…") + f"\nn={v.size:,}"
         for n, v in usable], fontsize=7, rotation=30, ha="right")
    ax.set_ylabel(value_label)
    head = title
    if hidden:
        head += f" (largest {len(usable)} of {len(usable) + hidden})"
    ax.set_title(head, fontsize=11, weight="bold")
    return desc


try:
    from genevariate.utils.viz_style import POINT_LABEL_SEP as _POINT_SEP
except Exception:  # pragma: no cover - stylesheet is optional
    _POINT_SEP = " · "


def draw_group_scatter(ax, coords, groups: Sequence[str], *, title: str = "",
                       xlabel: str = "1", ylabel: str = "2",
                       max_groups: int = 12,
                       ids: Optional[Sequence[Any]] = None) -> Dict[str, Any]:
    """Two coordinates per sample, coloured by whatever the caller grouped on."""
    xy = np.asarray(coords, dtype=float)
    if xy.ndim == 1:
        xy = xy.reshape(-1, 1)
    if xy.size and xy.shape[1] < 2:
        xy = np.column_stack([xy[:, 0], np.zeros(len(xy))])
    names = [str(g) for g in groups]
    counts: Dict[str, int] = {}
    for g in names:
        counts[g] = counts.get(g, 0) + 1
    ordered = sorted(counts, key=lambda g: -counts[g])
    shown, hidden = ordered[:max_groups], ordered[max_groups:]
    desc: Dict[str, Any] = {
        "title": title, "n": int(xy.shape[0]),
        "groups": [{"name": g, "n": counts[g]} for g in shown],
        "hidden": len(hidden), "artists": []}
    if xy.size == 0:
        return desc

    ids_arr = None if ids is None else np.asarray(ids, dtype=object)

    def _names(mask, group):
        if ids_arr is None:
            return [group] * int(mask.sum())
        return [f"{i}{_POINT_SEP}{group}" for i in ids_arr[mask]]

    try:                       # matplotlib >= 3.5
        from matplotlib import colormaps
        cmap = colormaps["tab20"]
    except Exception:          # pragma: no cover - older matplotlib
        from matplotlib import cm
        cmap = cm.get_cmap("tab20")
    for i, g in enumerate(shown):
        m = np.array([n == g for n in names])
        sc = ax.scatter(xy[m, 0], xy[m, 1], s=14, alpha=0.8, linewidths=0,
                        color=cmap(i % 20), label=f"{g} ({counts[g]})")
        desc["artists"].append((sc, _names(m, g)))
    if hidden:
        m = np.array([n in set(hidden) for n in names])
        sc = ax.scatter(xy[m, 0], xy[m, 1], s=10, alpha=0.4, linewidths=0,
                        color="#9aa0a6", label=f"{len(hidden)} smaller groups")
        desc["artists"].append((sc, _names(m, "smaller groups")))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, weight="bold")
    ax.legend(fontsize=7, markerscale=1.4, frameon=False,
              loc="best", ncol=1 if len(shown) <= 8 else 2)
    return desc
