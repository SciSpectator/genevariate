"""Label enrichment pooled across platforms, one stratum per platform."""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from genevariate.core.analysis.enrichment import benjamini_hochberg

try:
    from scipy.stats import chi2, norm

    _HAS_SCIPY = True
except Exception:  # pragma: no cover - scipy is a hard dep in practice
    _HAS_SCIPY = False


__all__ = [
    "pooled_label_enrichment",
    "summarize_pooled_enrichment",
]

_HALDANE = 0.5

DROP_ABSENT = "not measured here"
DROP_CONSTANT = "constant on this platform"
DROP_RARE = "too rare here"


def _logor(a: float, b: float, c: float, d: float):
    """Woolf log odds ratio and variance, Haldane-corrected."""
    a, b, c, d = (x + _HALDANE for x in (a, b, c, d))
    return float(np.log((a * d) / (b * c))), float(1 / a + 1 / b + 1 / c + 1 / d)


def _deff(hit: np.ndarray, grp: Optional[np.ndarray]) -> float:
    """Design effect of *hit* under study clumping, 1.0 when unknowable."""
    if grp is None:
        return 1.0
    from genevariate.core.analysis.overdispersion import (
        design_effect, estimate_rho, group_counts,
    )
    try:
        succ, sizes = group_counts(hit, grp)
        live = sizes > 0
        if live.sum() < 2:
            return 1.0
        rho = estimate_rho(succ[live], sizes[live])
        return max(1.0, design_effect(float(sizes[live].mean()), rho))
    except Exception:
        return 1.0


def _prepare(stratum: Mapping) -> Optional[dict]:
    """One platform's arrays, aligned and stripped of unlabelled samples."""
    lab = np.asarray(stratum.get("labels"), dtype=object)
    inr = np.asarray(stratum.get("in_region"), dtype=bool)
    if lab.shape[0] != inr.shape[0]:
        raise ValueError("in_region and labels must have the same length")
    grp = stratum.get("groups")
    grp = None if grp is None else np.asarray(grp, dtype=object).astype(str)
    if grp is not None and grp.shape[0] != lab.shape[0]:
        raise ValueError("groups must have the same length as labels")

    keep = np.array([x is not None and x == x and str(x).strip() != ""
                     for x in lab], dtype=bool)
    if not keep.any():
        return None
    lab = lab.astype(str)[keep]
    inr = inr[keep]
    grp = None if grp is None else grp[keep]
    if not inr.any() or inr.all():
        # No contrast: the region is empty or it is the whole platform.
        return None
    return {"labels": lab, "in_region": inr, "groups": grp,
            "n": int(lab.shape[0]), "n_region": int(inr.sum()),
            "technology": str(stratum.get("technology") or "unknown")}


def random_effects(y, var):
    """Returns ``{pooled, se, z, p, tau2, i2, q_stat, df, p_het}``."""
    y = np.asarray(y, dtype=float)
    var = np.asarray(var, dtype=float)
    k = y.size
    w = 1.0 / var
    fe = float((w * y).sum() / w.sum())
    Q = float((w * (y - fe) ** 2).sum())
    df = k - 1
    if df > 0:
        C = float(w.sum() - (w ** 2).sum() / w.sum())
        tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0
        i2 = float(max(0.0, (Q - df) / Q) * 100.0) if Q > 0 else 0.0
        p_het = float(chi2.sf(Q, df)) if _HAS_SCIPY else float("nan")
    else:
        tau2, i2, p_het = 0.0, float("nan"), float("nan")
    ws = 1.0 / (var + tau2)
    pooled = float((ws * y).sum() / ws.sum())
    se = float(np.sqrt(1.0 / ws.sum()))
    z = pooled / se if se > 0 else float("nan")
    p = (float(2.0 * norm.sf(abs(z))) if _HAS_SCIPY and np.isfinite(z)
         else float("nan"))
    return {"pooled": pooled, "se": se, "z": float(z), "p": p,
            "tau2": float(tau2), "i2": i2, "q_stat": Q, "df": df,
            "p_het": p_het}


def pooled_label_enrichment(
    strata: Mapping[str, Mapping],
    values: Optional[Sequence[str]] = None,
    min_count: int = 3,
    max_values: int = 40,
) -> dict:
    """Returns ``{platforms, technologies, cross_technology, values, rows, n_dropped}``."""
    prepared: Dict[str, dict] = {}
    for name, st in strata.items():
        p = _prepare(st)
        if p is not None:
            prepared[str(name)] = p
    if not prepared:
        return {"platforms": [], "technologies": {}, "cross_technology": False,
                "values": [], "rows": [], "n_dropped": 0}

    names = list(prepared)
    techs = {n: prepared[n]["technology"] for n in names}
    known = {t for t in techs.values() if t not in ("unknown", "custom")}
    cross_tech = len(known) > 1

    if values is None:
        seen: Dict[str, List[int]] = {}
        for n in names:
            v, c = np.unique(prepared[n]["labels"], return_counts=True)
            for vi, ci in zip(v, c):
                e = seen.setdefault(str(vi), [0, 0])
                e[0] += 1
                e[1] += int(ci)
        values = [v for v, _ in sorted(seen.items(),
                                       key=lambda kv: (-kv[1][0], -kv[1][1]))
                  ][:max_values]
    values = [str(v) for v in values]

    rows: List[dict] = []
    n_dropped = 0
    for v in values:
        ys, vs, per, dropped = [], [], {}, {}
        for n in names:
            P = prepared[n]
            hit = P["labels"] == v
            total = int(hit.sum())
            if total == 0:
                dropped[n] = DROP_ABSENT
                continue
            if total == P["n"]:
                dropped[n] = DROP_CONSTANT
                continue
            if total < min_count:
                dropped[n] = DROP_RARE
                continue
            inr = P["in_region"]
            a = int((hit & inr).sum())
            b = P["n_region"] - a
            c = total - a
            d = (P["n"] - P["n_region"]) - c
            lor, var = _logor(a, b, c, d)
            deff = _deff(hit, P["groups"])
            var *= deff
            ys.append(lor)
            vs.append(var)
            per[n] = {"log_or": lor, "se": float(np.sqrt(var)), "a": a,
                      "n_region": P["n_region"], "n": P["n"],
                      "rate": a / max(1, P["n_region"]),
                      "bg_rate": c / max(1, P["n"] - P["n_region"]),
                      "deff": float(deff)}
        n_dropped += len(dropped)
        k = len(ys)
        if k < 1:
            continue

        y = np.asarray(ys, dtype=float)
        re = random_effects(y, vs)
        pooled, se, tau2 = re["pooled"], re["se"], re["tau2"]
        i2, Q, df, p_het = re["i2"], re["q_stat"], re["df"], re["p_het"]
        z, p = re["z"], re["p"]
        same = int(np.sum(np.sign(y) == np.sign(pooled))) if pooled != 0 else 0

        rows.append({
            "value": v, "k": k,
            "pooled_log_or": pooled, "pooled_or": float(np.exp(pooled)),
            "se": se, "z": float(z), "p": p, "q": float("nan"),
            "ci_low": float(np.exp(pooled - 1.959964 * se)),
            "ci_high": float(np.exp(pooled + 1.959964 * se)),
            "tau2": float(tau2), "i2": i2,
            "q_stat": Q, "df": df, "p_het": p_het, "q_het": float("nan"),
            "n_same_sign": same,
            "concordant": bool(k >= 2 and same == k
                               and np.isfinite(i2) and i2 < 25.0),
            "per_platform": per, "dropped": dropped,
        })

    for key, qkey in (("p", "q"), ("p_het", "q_het")):
        ps = np.array([r[key] for r in rows], dtype=float)
        ok = np.isfinite(ps)
        if ok.any():
            qs = np.full(ps.shape, np.nan)
            qs[ok] = benjamini_hochberg(ps[ok])
            for r, qv in zip(rows, qs):
                r[qkey] = float(qv)

    rows.sort(key=lambda r: (r["q"] if np.isfinite(r["q"]) else 1.0))
    return {"platforms": names, "technologies": techs,
            "cross_technology": cross_tech, "values": values,
            "rows": rows, "n_dropped": n_dropped}


def _fmt_or(x: float) -> str:
    if not np.isfinite(x):
        return "n/a"
    return f"{x:.2f}" if x < 100 else f"{x:.3g}"


def summarize_pooled_enrichment(result: dict, lcol: str = "label",
                                alpha: float = 0.05) -> str:
    """Markdown summary, with the cross-technology reading rule attached."""
    rows = result.get("rows") or []
    plats = result.get("platforms") or []
    techs = result.get("technologies") or {}
    if not rows:
        return (f"No label value of '{lcol}' could be pooled across "
                f"{len(plats)} platform(s).")

    lines = [f"### Pooled enrichment of '{lcol}' across {len(plats)} platforms",
             ""]
    lines.append("Platforms: "
                 + ", ".join(f"{p} ({techs.get(p, 'unknown')})" for p in plats))
    lines.append("")

    hits = [r for r in rows if np.isfinite(r["q"]) and r["q"] < alpha]
    if not hits:
        lines.append(f"No value reaches q < {alpha} after pooling.")
    else:
        lines.append(f"**{len(hits)} value(s) at q < {alpha}:**")
        for r in hits[:15]:
            tag = "concordant" if r["concordant"] else f"I2={r['i2']:.0f}%"
            lines.append(
                f"  - {r['value']}: OR {_fmt_or(r['pooled_or'])} "
                f"[{_fmt_or(r['ci_low'])}, {_fmt_or(r['ci_high'])}], "
                f"q={r['q']:.2g}, {r['k']}/{len(plats)} platforms, {tag}")

    split = [r for r in rows if r["k"] >= 2 and np.isfinite(r["q_het"])
             and r["q_het"] < alpha and r["i2"] >= 50]
    if split:
        lines.append("")
        lines.append("**Platforms disagree (q_het < %g, I2 >= 50%%):** %s"
                     % (alpha, ", ".join(r["value"] for r in split[:10])))
        if result.get("cross_technology"):
            lines.append(
                "  These platforms use different technologies, so the "
                "disagreement cannot be attributed to biology. A monotone "
                "rescaling preserves a quantile but not which samples sit in "
                "it, so the same brush rule selects different samples on "
                "different technologies. Agreement across technologies is "
                "evidence; disagreement is not interpretable.")

    single = [r for r in rows if r["k"] < 2]
    if single:
        lines.append("")
        lines.append(f"{len(single)} value(s) were testable on one platform "
                     "only and are not pooled.")
    return "\n".join(lines)
