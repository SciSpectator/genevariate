"""Comparing many regions against each other, not each against the background."""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from genevariate.core.analysis.enrichment import benjamini_hochberg
from genevariate.core.analysis.overdispersion import enrichment_diagnostics

try:
    from scipy.stats import chi2, fisher_exact

    _HAS_SCIPY = True
except Exception:  # pragma: no cover - scipy is a hard dep in practice
    _HAS_SCIPY = False


__all__ = [
    "overlap_matrix",
    "enrichment_matrix",
    "pairwise_differential",
    "heterogeneity",
    "cluster_regions",
    "region_separability",
    "summarize_comparison",
]

_HALDANE = 0.5


def _as_masks(region_masks: Mapping[str, Sequence[bool]], n: int
              ) -> Tuple[List[str], np.ndarray]:
    names = list(region_masks)
    if not names:
        raise ValueError("no regions given")
    M = np.zeros((len(names), n), dtype=bool)
    for i, k in enumerate(names):
        m = np.asarray(region_masks[k], dtype=bool)
        if m.shape[0] != n:
            raise ValueError(f"region {k!r} mask has length {m.shape[0]}, "
                             f"expected {n}")
        M[i] = m
    return names, M


def _clean(labels: Sequence, groups: Optional[Sequence]):
    lab = np.asarray(labels, dtype=object)
    keep = np.array([x is not None and x == x and str(x).strip() != ""
                     for x in lab], dtype=bool)
    grp = None if groups is None else np.asarray(groups, dtype=object).astype(str)
    return lab.astype(str), grp, keep


def overlap_matrix(region_masks: Mapping[str, Sequence[bool]]) -> dict:
    """Jaccard overlap of every region pair."""
    n = len(next(iter(region_masks.values())))
    names, M = _as_masks(region_masks, n)
    k = len(names)
    J = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            inter = float((M[i] & M[j]).sum())
            union = float((M[i] | M[j]).sum())
            J[i, j] = inter / union if union else 0.0
    return {"regions": names, "jaccard": J}


def enrichment_matrix(
    region_masks: Mapping[str, Sequence[bool]],
    labels: Sequence,
    groups: Optional[Sequence] = None,
    min_count: int = 3,
    max_values: int = 40,
    n_boot: int = 300,
    seed: int = 0,
) -> dict:
    """Regions x label values, as one grid with one multiple-testing correction."""
    lab, grp, keep = _clean(labels, groups)
    n = lab.shape[0]
    names, M = _as_masks(region_masks, n)
    M = M & keep                      # unlabelled samples cannot enrich anything
    lab_k = lab[keep]

    vals, counts = np.unique(lab_k, return_counts=True)
    order = np.argsort(-counts)
    vals = [str(v) for v in vals[order][:max_values]]
    N = int(keep.sum())

    R, V = len(names), len(vals)
    a = np.zeros((R, V), dtype=int)
    log2_lift = np.full((R, V), np.nan)
    p = np.full((R, V), np.nan)
    n_eff = np.full((R, V), np.nan)
    rho = np.full((R, V), np.nan)
    n_gse = np.zeros((R, V), dtype=int)
    n_region = np.array([int(m.sum()) for m in M])
    background = np.array([(((lab == v) & keep).sum() / N) if N else np.nan
                           for v in vals], dtype=float)

    for i, m in enumerate(M):
        n_r = int(m.sum())
        # one study-aware pass per region covers every value at once
        diag = {}
        if n_r and grp is not None:
            try:
                diag = enrichment_diagnostics(m[keep], lab_k, grp[keep],
                                              values=vals, n_boot=n_boot,
                                              seed=seed)
            except Exception:
                diag = {}
        for j, v in enumerate(vals):
            hit = (lab == v) & keep
            a_ij = int((hit & m).sum())
            a[i, j] = a_ij
            if not n_r:
                continue
            rate = (a_ij + _HALDANE) / (n_r + 1.0)
            base = background[j]
            if base and base > 0:
                log2_lift[i, j] = float(np.log2(rate / base))
            if _HAS_SCIPY and 0 < n_r < N:
                c = int(hit.sum()) - a_ij
                try:
                    p[i, j] = float(fisher_exact(
                        [[a_ij, n_r - a_ij], [c, (N - n_r) - c]],
                        alternative="greater")[1])
                except Exception:
                    pass
            d = diag.get(v)
            if d:
                n_eff[i, j] = d.get("n_eff_sel", np.nan)
                rho[i, j] = d.get("rho", np.nan)
                n_gse[i, j] = d.get("n_gse") or 0

    flat = p.ravel()
    ok = np.isfinite(flat)
    q = np.full(flat.shape, np.nan)
    if ok.any():
        q[ok] = benjamini_hochberg(flat[ok])
    q = q.reshape(p.shape)

    thin = a < min_count
    return {
        "regions": names, "values": vals,
        "log2_lift": log2_lift, "a": a, "n_region": n_region,
        "p": p, "q": q, "n_eff": n_eff, "rho": rho, "n_gse": n_gse,
        "background": background, "n_labelled": N, "thin": thin,
        "min_count": int(min_count),
    }


def _logor(a, b, c, d):
    """Woolf log odds ratio and variance, Haldane-corrected."""
    a, b, c, d = (x + _HALDANE for x in (a, b, c, d))
    return float(np.log((a * d) / (b * c))), float(1 / a + 1 / b + 1 / c + 1 / d)


def pairwise_differential(
    region_masks: Mapping[str, Sequence[bool]],
    labels: Sequence,
    groups: Optional[Sequence] = None,
    values: Optional[Sequence[str]] = None,
    min_count: int = 3,
    n_boot: int = 400,
    alpha: float = 0.05,
    seed: int = 0,
) -> List[dict]:
    """Every region pair, every label value: is it commoner in A than in B?"""
    lab, grp, keep = _clean(labels, groups)
    n = lab.shape[0]
    names, M = _as_masks(region_masks, n)
    M = M & keep
    jac = overlap_matrix(region_masks)["jaccard"]

    if values is None:
        v_all, cnt = np.unique(lab[keep], return_counts=True)
        values = [str(v) for v in v_all[np.argsort(-cnt)][:40]]

    inv = n_groups = None
    if grp is not None:
        _, inv = np.unique(grp, return_inverse=True)
        n_groups = int(inv.max()) + 1 if inv.size else 0

    rng = np.random.default_rng(seed)
    boot_w = None
    if inv is not None and n_boot and n_groups > 1:
        boot_w = rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups),
                                 size=int(n_boot)).astype(np.float32)

    rows: List[dict] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            mi, mj = M[i], M[j]
            ni, nj = int(mi.sum()), int(mj.sum())
            if not ni or not nj:
                continue
            for v in values:
                hit = (lab == v) & keep
                ai, aj = int((hit & mi).sum()), int((hit & mj).sum())
                if max(ai, aj) < min_count:
                    continue
                lor, var = _logor(ai, ni - ai, aj, nj - aj)
                pv = np.nan
                if _HAS_SCIPY:
                    try:
                        pv = float(fisher_exact([[ai, ni - ai], [aj, nj - aj]])[1])
                    except Exception:
                        pass

                lo = hi = float("nan")
                if boot_w is not None:
                    def _per_study(mask):
                        return np.bincount(inv, weights=mask.astype(float),
                                           minlength=n_groups).astype(np.float32)
                    ai_g, ni_g = _per_study(hit & mi), _per_study(mi)
                    aj_g, nj_g = _per_study(hit & mj), _per_study(mj)
                    A = ai_g @ boot_w.T
                    Ni = ni_g @ boot_w.T
                    B = aj_g @ boot_w.T
                    Nj = nj_g @ boot_w.T
                    with np.errstate(divide="ignore", invalid="ignore"):
                        b_lor = np.log(((A + _HALDANE) * (Nj - B + _HALDANE)) /
                                       ((Ni - A + _HALDANE) * (B + _HALDANE)))
                    b_lor = b_lor[np.isfinite(b_lor)]
                    if b_lor.size:
                        lo = float(np.percentile(b_lor, 100 * alpha / 2))
                        hi = float(np.percentile(b_lor, 100 * (1 - alpha / 2)))

                rows.append({
                    "region_a": names[i], "region_b": names[j], "value": v,
                    "a_a": ai, "n_a": ni, "a_b": aj, "n_b": nj,
                    "rate_a": ai / ni, "rate_b": aj / nj,
                    "log_or": lor, "se": float(np.sqrt(var)),
                    "ci_low": lo, "ci_high": hi,
                    "p": pv, "q": float("nan"),
                    "jaccard": float(jac[i, j]),
                })

    ps = np.array([r["p"] for r in rows], dtype=float)
    ok = np.isfinite(ps)
    if ok.any():
        qq = np.full(ps.shape, np.nan)
        qq[ok] = benjamini_hochberg(ps[ok])
        for r, q in zip(rows, qq):
            r["q"] = float(q)
    rows.sort(key=lambda r: (np.inf if not np.isfinite(r["q"]) else r["q"]))
    return rows


def heterogeneity(
    region_masks: Mapping[str, Sequence[bool]],
    labels: Sequence,
    groups: Optional[Sequence] = None,
    values: Optional[Sequence[str]] = None,
    min_count: int = 3,
) -> List[dict]:
    """Per label: is the enrichment the same in every region, or region-specific?"""
    from genevariate.core.analysis.overdispersion import (
        design_effect, estimate_rho, group_counts,
    )

    lab, grp, keep = _clean(labels, groups)
    n = lab.shape[0]
    names, M = _as_masks(region_masks, n)
    M = M & keep
    N = int(keep.sum())

    if values is None:
        v_all, cnt = np.unique(lab[keep], return_counts=True)
        values = [str(v) for v in v_all[np.argsort(-cnt)][:40]]

    out: List[dict] = []
    for v in values:
        hit = (lab == v) & keep
        ys, ws, per = [], [], {}
        for i, m in enumerate(M):
            n_r = int(m.sum())
            a = int((hit & m).sum())
            if not n_r or a < min_count:
                continue
            c = int(hit.sum()) - a
            lor, var = _logor(a, n_r - a, c, (N - n_r) - c)
            deff = 1.0
            if grp is not None:
                try:
                    succ, sizes = group_counts(hit[m], grp[m])
                    live = sizes > 0
                    if live.sum() >= 2:
                        r = estimate_rho(succ[live], sizes[live])
                        deff = max(1.0, design_effect(
                            float(sizes[live].mean()), r))
                except Exception:
                    deff = 1.0
            var *= deff
            ys.append(lor)
            ws.append(1.0 / var)
            per[names[i]] = {"log_or": lor, "se": float(np.sqrt(var)), "a": a,
                             "n": n_r, "rate": a / n_r, "deff": float(deff)}
        k = len(ys)
        if k < 2:
            continue
        ys, ws = np.array(ys), np.array(ws)
        pooled = float((ws * ys).sum() / ws.sum())
        Q = float((ws * (ys - pooled) ** 2).sum())
        df = k - 1
        pv = float(chi2.sf(Q, df)) if _HAS_SCIPY else float("nan")
        i2 = float(max(0.0, (Q - df) / Q) * 100.0) if Q > 0 else 0.0
        out.append({"value": v, "k": k, "q_stat": Q, "df": df, "p": pv,
                    "q": float("nan"), "i2": i2, "pooled_log_or": pooled,
                    "pooled_or": float(np.exp(pooled)), "per_region": per})

    ps = np.array([r["p"] for r in out], dtype=float)
    ok = np.isfinite(ps)
    if ok.any():
        qs = np.full(ps.shape, np.nan)
        qs[ok] = benjamini_hochberg(ps[ok])
        for r, q in zip(out, qs):
            r["q"] = float(q)

    out.sort(key=lambda r: -r["i2"])
    return out


def cluster_regions(matrix: dict) -> dict:
    """Order regions by how similar their enrichment profiles are."""
    L = np.asarray(matrix["log2_lift"], dtype=float)
    names = list(matrix["regions"])
    L = np.where(np.isfinite(L), L, 0.0)
    k = L.shape[0]
    if k < 2:
        return {"regions": names, "order": list(range(k)),
                "correlation": np.ones((k, k)), "linkage": None}

    with np.errstate(invalid="ignore"):
        C = np.corrcoef(L)
    C = np.where(np.isfinite(C), C, 0.0)

    order, Z = list(range(k)), None
    try:
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform
        D = np.clip(1.0 - C, 0.0, 2.0)
        np.fill_diagonal(D, 0.0)
        D = (D + D.T) / 2.0
        Z = linkage(squareform(D, checks=False), method="average")
        order = list(dendrogram(Z, no_plot=True)["leaves"])
    except Exception:
        pass
    return {"regions": names, "order": order, "correlation": C, "linkage": Z}


def region_separability(
    region_masks: Mapping[str, Sequence[bool]],
    label_frame,
    groups: Optional[Sequence] = None,
    max_levels: int = 60,
    seed: int = 0,
) -> dict:
    """How distinct is each region once the folds respect study structure?"""
    import pandas as pd

    from genevariate.core.analysis.box_model import fit_label_model

    lf = pd.DataFrame(label_frame).reset_index(drop=True)
    n = len(lf)
    names, M = _as_masks(region_masks, n)

    blocks, feats = {}, []
    for col in lf.columns:
        s = lf[col].astype(str).fillna("")
        top = [v for v in s.value_counts().index[:max_levels] if v.strip()]
        if len(top) < 2:
            continue
        idx = []
        for v in top:
            feats.append(f"{col}={v}")
            idx.append(len(feats) - 1)
        blocks[col] = idx
    if not feats:
        return {"features": [], "regions": {},
                "error": "no label column had two or more values"}

    X = np.zeros((n, len(feats)), dtype=float)
    for col, idx in blocks.items():
        s = lf[col].astype(str).fillna("")
        for k, f in zip(idx, [feats[i].split("=", 1)[1] for i in idx]):
            X[:, k] = (s == f).to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    out: Dict[str, dict] = {}
    for i, name in enumerate(names):
        y = M[i]
        if int(y.sum()) < 10 or int((~y).sum()) < 10:
            out[name] = {"skipped": "fewer than 10 samples on one side"}
            continue
        try:
            m = fit_label_model(X, y, groups, feature_names=feats, seed=seed)
        except Exception as exc:
            out[name] = {"skipped": str(exc)}
            continue

        imp = {}
        if np.isfinite(m.auc):
            from sklearn.metrics import roc_auc_score
            try:
                intact = float(roc_auc_score(y, m.predict(X)))
            except Exception:
                intact = float("nan")
            for col, idx in blocks.items():
                Xp = X.copy()
                perm = rng.permutation(n)
                Xp[:, idx] = X[np.ix_(perm, idx)]
                try:
                    drop = intact - float(roc_auc_score(y, m.predict(Xp)))
                except Exception:
                    drop = float("nan")
                imp[col] = float(drop)
        out[name] = {"auc": float(m.auc), "grouped": bool(m.grouped),
                     "n_pos": int(y.sum()), "n_splits": int(m.n_splits),
                     "importance": imp}
    return {"features": feats, "regions": out}


def _fmt_or(x):
    return "inf" if not np.isfinite(x) else f"{np.exp(x):.2f}x"


def summarize_comparison(matrix: dict,
                         hetero: Optional[List[dict]] = None,
                         pairs: Optional[List[dict]] = None,
                         separability: Optional[dict] = None,
                         q_cut: float = 0.05,
                         top: int = 5) -> str:
    """A written read of the comparison, computed rather than generated."""
    regions = matrix["regions"]
    values = matrix["values"]
    q, lift, a = matrix["q"], matrix["log2_lift"], matrix["a"]
    n_eff, n_gse = matrix["n_eff"], matrix["n_gse"]
    n_region = matrix["n_region"]
    lines: List[str] = []

    sig = np.isfinite(q) & (q < q_cut) & (a >= matrix["min_count"])
    lines.append(
        f"{len(regions)} regions x {len(values)} label values "
        f"({matrix['n_labelled']:,} labelled samples). "
        f"{int(sig.sum())} of {int(np.isfinite(q).sum())} cells pass BH-FDR "
        f"at q<{q_cut:g}, corrected across the whole matrix.")

    per_region = sig.sum(axis=1)
    if per_region.max() > 0 and (per_region == per_region.max()).sum() == 1:
        best = int(np.argmax(per_region))
        worst = int(np.argmin(per_region))
        lines.append(
            f"Most distinctive region: {regions[best]} "
            f"({per_region[best]} enriched labels). "
            f"Least: {regions[worst]} ({per_region[worst]}).")

    ranked = sorted(
        ((i, j) for i in range(len(regions)) for j in range(len(values))
         if sig[i, j]),
        key=lambda ij: -lift[ij])
    if ranked:
        lines.append("")
        lines.append(f"Strongest enrichments (top {min(top, len(ranked))}):")
        for i, j in ranked[:top]:
            ne = n_eff[i, j]
            ne_txt = (f", {a[i, j]} hits; region worth {ne:.0f} of "
                      f"{n_region[i]} samples across {n_gse[i, j]} studies"
                      if np.isfinite(ne) else f", {a[i, j]} hits")
            lines.append(f"  - {values[j]} in {regions[i]}: "
                         f"{2 ** lift[i, j]:.1f}x, q={q[i, j]:.2g}{ne_txt}")
        thin = [(i, j) for i, j in ranked[:top]
                if np.isfinite(n_eff[i, j]) and n_eff[i, j] < 0.25 * n_region[i]]
        if thin:
            lines.append(
                f"  Caution: {len(thin)} of these lose over three quarters of "
                f"their apparent evidence to study clumping.")

    if hetero:
        shared = [h for h in hetero if h["i2"] < 25 and h["k"] >= 2]
        specific = [h for h in hetero if h["i2"] >= 50 and np.isfinite(h["q"])
                    and h["q"] < q_cut]
        lines.append("")
        if specific:
            lines.append("Region-specific labels (enrichment differs by region):")
            for h in specific[:top]:
                lines.append(f"  - {h['value']}: I2={h['i2']:.0f}%, "
                             f"Q={h['q_stat']:.1f} on {h['df']} df, "
                             f"q={h['q']:.2g}")
        else:
            lines.append("No label shows significant between-region "
                         "heterogeneity - the regions are not disagreeing.")
        if shared:
            lines.append(
                "Consistent across regions: "
                + ", ".join(h["value"] for h in shared[:top])
                + " (I2 under 25%, so these are platform-wide, not "
                  "region-specific).")

    if pairs:
        strong = [r for r in pairs
                  if np.isfinite(r["q"]) and r["q"] < q_cut
                  and np.isfinite(r["ci_low"]) and r["ci_low"] * r["ci_high"] > 0]
        lines.append("")
        if strong:
            lines.append(f"Region-vs-region differences ({len(strong)} "
                         f"significant with a CI excluding 1):")
            for r in strong[:top]:
                warn = ("  [regions share "
                        f"{r['jaccard']:.0%} of their samples - not independent]"
                        if r["jaccard"] > 0.2 else "")
                lines.append(
                    f"  - {r['value']}: {r['region_a']} {r['rate_a']:.1%} vs "
                    f"{r['region_b']} {r['rate_b']:.1%}, "
                    f"OR {_fmt_or(r['log_or'])}, q={r['q']:.2g}{warn}")
        else:
            lines.append("No label separates any pair of regions after FDR - "
                         "they may be enriched against the platform without "
                         "being distinguishable from each other.")

    if separability and separability.get("regions"):
        lines.append("")
        lines.append("Cross-fitted separability (can a model trained on other "
                     "studies recognise this region from its labels alone?):")
        for name, rec in separability["regions"].items():
            if "skipped" in rec:
                lines.append(f"  - {name}: not fitted ({rec['skipped']})")
                continue
            imp = rec.get("importance") or {}
            top_lab = max(imp, key=imp.get) if imp else None
            drv = (f", driven by {top_lab} (AUC -{imp[top_lab]:.3f} when "
                   f"permuted)" if top_lab and np.isfinite(imp[top_lab]) else "")
            verdict = ("indistinguishable" if rec["auc"] < 0.6
                       else "clearly distinct" if rec["auc"] > 0.8
                       else "partly distinct")
            caveat = ("" if rec["grouped"]
                      else " [folds NOT grouped by study - treat as optimistic]")
            lines.append(f"  - {name}: AUC {rec['auc']:.3f} "
                         f"({verdict}){caveat}{drv}")

    return "\n".join(lines)
