"""
Study-clumping (overdispersion) corrections for counts and for measurements.

GEO samples do not arrive independently: they come in study-sized clumps, so a
region holding 7,754 GSMs drawn from a few hundred GSEs carries far less
information than 7,754 independent draws. Treating the counts as binomial
therefore produces confidence intervals that are fictitiously tight and
p-values that are confidently wrong.

The same holds for a measurement: two expression values from one study are not
two independent readings of the platform, so a test that compares platforms
sample by sample is testing more samples than it has evidence for.

This module supplies the corrections that make either kind of number honest:

* ``estimate_rho``            beta-binomial intra-cluster correlation of a label
                              across studies (0 = independent, 1 = fully clumped)
* ``icc_oneway``              the same correlation for a continuous measurement
* ``effective_sample_size``   n / (1 + (m_bar - 1) * rho)
* ``enrichment_diagnostics``  per-label-value rho, n_eff, contributing-study
                              count and a bootstrap-by-study CI on the
                              enrichment ratio

Everything is pure numpy/scipy and Tk-free so the GUI, the chatbot tools and the
tests can all share it.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.optimize import minimize
    from scipy.special import betaln

    _HAS_SCIPY = True
except Exception:  # pragma: no cover - scipy is a hard dep in practice
    _HAS_SCIPY = False


__all__ = [
    "group_counts",
    "estimate_rho",
    "icc_oneway",
    "effective_sample_size",
    "design_effect",
    "enrichment_diagnostics",
]


def icc_oneway(values: Sequence[float], groups: Sequence) -> Tuple[float, float]:
    """One-way random-effects ICC of a continuous measurement across studies.

    ``estimate_rho`` answers the same question for a yes/no label; this answers
    it for a number, so an expression value measured on study-sized clumps of
    samples can be given the same design effect as a count. Returns
    ``(rho, mean_cluster_size)``; rho is 0 when there is one study or no
    between-study variance to speak of.
    """
    y = np.asarray(values, dtype=float)
    g = np.asarray(groups, dtype=object).astype(str)
    if y.shape[0] != g.shape[0]:
        raise ValueError("values and groups must have the same length")
    keep = np.isfinite(y)
    y, g = y[keep], g[keep]
    if y.size < 3:
        return 0.0, float(y.size)
    _, inv = np.unique(g, return_inverse=True)
    k = int(inv.max()) + 1
    n_i = np.bincount(inv, minlength=k).astype(float)
    N = float(y.size)
    m_bar = float(N / k)
    if k < 2 or N <= k:
        return 0.0, m_bar
    sums = np.bincount(inv, weights=y, minlength=k)
    mean_i = sums / n_i
    grand = float(y.sum() / N)
    msb = float((n_i * (mean_i - grand) ** 2).sum() / (k - 1))
    msw = float((((y - mean_i[inv]) ** 2).sum()) / (N - k))
    n0 = (N - float((n_i ** 2).sum()) / N) / (k - 1)
    denom = msb + (n0 - 1.0) * msw
    if denom <= 0:
        return 0.0, m_bar
    return float(np.clip((msb - msw) / denom, 0.0, 1.0)), m_bar


def group_counts(hits: Sequence[bool], groups: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    """Collapse per-sample hits into per-group (successes, size) arrays."""
    h = np.asarray(hits, dtype=bool)
    g = np.asarray(groups)
    if h.shape[0] != g.shape[0]:
        raise ValueError("hits and groups must have the same length")
    _, inv = np.unique(g, return_inverse=True)
    n_groups = int(inv.max()) + 1 if inv.size else 0
    sizes = np.bincount(inv, minlength=n_groups).astype(float)
    succ = np.bincount(inv, weights=h.astype(float), minlength=n_groups)
    return succ, sizes


def _moment_rho(k: np.ndarray, m: np.ndarray) -> float:
    """ANOVA-type moment estimator of the intra-cluster correlation."""
    n_total = float(m.sum())
    n_groups = int(m.size)
    if n_groups < 2 or n_total <= n_groups:
        return 0.0
    p_bar = float(k.sum() / n_total)
    if p_bar <= 0.0 or p_bar >= 1.0:
        return 0.0
    p_i = k / m
    msb = float((m * (p_i - p_bar) ** 2).sum() / (n_groups - 1))
    msw = float((m * p_i * (1.0 - p_i)).sum() / (n_total - n_groups))
    m0 = (n_total - float((m ** 2).sum()) / n_total) / (n_groups - 1)
    denom = msb + (m0 - 1.0) * msw
    if denom <= 0:
        return 0.0
    return float(np.clip((msb - msw) / denom, 0.0, 1.0))


def estimate_rho(successes: Sequence[float], sizes: Sequence[float]) -> float:
    """
    Beta-binomial intra-cluster correlation rho for one label across studies.

    ``successes[i]`` samples out of ``sizes[i]`` in study *i* carry the label.
    rho = 1 / (1 + a + b) of the fitted Beta(a, b) mixing distribution, so
    rho -> 0 when the label is scattered independently across studies and
    rho -> 1 when whole studies are all-or-nothing for it.
    """
    k = np.asarray(successes, dtype=float)
    m = np.asarray(sizes, dtype=float)
    if k.shape != m.shape:
        raise ValueError("successes and sizes must have the same shape")
    keep = m > 0
    k, m = k[keep], m[keep]
    if k.size < 2:
        return 0.0
    total = float(k.sum())
    if total <= 0.0 or total >= float(m.sum()):
        # label absent everywhere or present everywhere - no dispersion to fit
        return 0.0
    if not _HAS_SCIPY:
        return _moment_rho(k, m)

    mu0 = float(np.clip(total / m.sum(), 1e-6, 1 - 1e-6))

    def nll(theta: np.ndarray) -> float:
        logit_mu, log_s = theta
        if not np.isfinite(logit_mu) or not np.isfinite(log_s):
            return 1e12
        mu = 1.0 / (1.0 + np.exp(-np.clip(logit_mu, -30, 30)))
        s = float(np.exp(np.clip(log_s, -20, 20)))
        a, b = mu * s, (1.0 - mu) * s
        if a <= 0 or b <= 0 or not np.isfinite(a) or not np.isfinite(b):
            return 1e12
        val = -float((betaln(k + a, m - k + b) - betaln(a, b)).sum())
        return val if np.isfinite(val) else 1e12

    x0 = np.array([np.log(mu0 / (1.0 - mu0)), np.log(10.0)])
    try:
        res = minimize(nll, x0, method="Nelder-Mead",
                       options={"maxiter": 600, "xatol": 1e-4, "fatol": 1e-4})
        if not np.isfinite(res.fun) or res.fun >= 1e11:
            return _moment_rho(k, m)
        s = float(np.exp(np.clip(res.x[1], -20, 20)))
    except Exception:
        return _moment_rho(k, m)
    rho = 1.0 / (1.0 + s)
    if not np.isfinite(rho):
        return _moment_rho(k, m)
    return float(np.clip(rho, 0.0, 1.0))


def design_effect(mean_cluster_size: float, rho: float) -> float:
    """Kish design effect 1 + (m_bar - 1) * rho."""
    if not np.isfinite(rho) or rho <= 0 or not np.isfinite(mean_cluster_size):
        return 1.0
    return float(1.0 + (max(float(mean_cluster_size), 1.0) - 1.0) * float(rho))


def effective_sample_size(n: float, mean_cluster_size: float, rho: float) -> float:
    """n / (1 + (m_bar - 1) * rho) - how many independent samples n is worth."""
    n = float(n)
    if n <= 0:
        return 0.0
    deff = design_effect(mean_cluster_size, rho)
    return n / deff if deff > 0 else n


def enrichment_diagnostics(
    in_region: Sequence[bool],
    labels: Sequence,
    groups: Optional[Sequence] = None,
    values: Optional[Iterable] = None,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, dict]:
    """
    Study-aware diagnostics for every label value of one region x label column.

    All three sequences are aligned over the platform samples that carry a
    label: ``in_region`` marks region membership, ``labels`` is the label value
    and ``groups`` the study id (GSE / series_id). ``groups=None`` means no
    study information is available, in which case the clumping fields come back
    as NaN/None rather than being silently faked.

    Returns ``{value: {n_gse, rho, mean_cluster, n_eff_sel, ci_low, ci_high,
    a, c}}``. The CI is a percentile bootstrap that resamples **studies**, not
    samples, so it reflects the real amount of independent evidence.
    """
    reg = np.asarray(in_region, dtype=bool)
    lab = np.asarray(labels, dtype=object).astype(str)
    if reg.shape[0] != lab.shape[0]:
        raise ValueError("in_region and labels must have the same length")
    vals = list(values) if values is not None else list(np.unique(lab))
    n_sel = int(reg.sum())

    out: Dict[str, dict] = {}

    if groups is None:
        for v in vals:
            hit = lab == str(v)
            out[str(v)] = {
                "n_gse": None, "rho": float("nan"), "mean_cluster": float("nan"),
                "n_eff_sel": float(n_sel), "ci_low": float("nan"),
                "ci_high": float("nan"),
                "a": int((hit & reg).sum()), "c": int((hit & ~reg).sum()),
            }
        return out

    grp = np.asarray(groups, dtype=object).astype(str)
    if grp.shape[0] != lab.shape[0]:
        raise ValueError("groups must have the same length as labels")
    _, inv = np.unique(grp, return_inverse=True)
    n_groups = int(inv.max()) + 1 if inv.size else 0

    sel_g = np.bincount(inv, weights=reg.astype(float), minlength=n_groups)
    non_g = np.bincount(inv, weights=(~reg).astype(float), minlength=n_groups)
    sizes = sel_g + non_g
    mean_cluster = float(sizes[sizes > 0].mean()) if n_groups else float("nan")

    weights = None
    if n_boot and n_boot > 0 and n_groups > 1:
        rng = np.random.default_rng(seed)
        weights = rng.multinomial(
            n_groups, np.full(n_groups, 1.0 / n_groups), size=int(n_boot)
        ).astype(np.float32)
        sel_b = weights @ sel_g.astype(np.float32)
        non_b = weights @ non_g.astype(np.float32)
        sel_b = np.where(sel_b > 0, sel_b, np.nan)
        non_b = np.where(non_b > 0, non_b, np.nan)

    for v in vals:
        key = str(v)
        hit = lab == key
        a_g = np.bincount(inv, weights=(hit & reg).astype(float), minlength=n_groups)
        c_g = np.bincount(inv, weights=(hit & ~reg).astype(float), minlength=n_groups)

        rho = estimate_rho(a_g + c_g, sizes)
        lo = hi = float("nan")
        if weights is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = ((weights @ a_g.astype(np.float32)) / sel_b) / \
                        ((weights @ c_g.astype(np.float32)) / non_b)
            ratio = ratio[np.isfinite(ratio)]
            if ratio.size >= 20:
                lo = float(np.quantile(ratio, alpha / 2.0))
                hi = float(np.quantile(ratio, 1.0 - alpha / 2.0))

        out[key] = {
            "n_gse": int((a_g > 0).sum()),
            "rho": float(rho),
            "mean_cluster": mean_cluster,
            "n_eff_sel": effective_sample_size(n_sel, mean_cluster, rho),
            "ci_low": lo,
            "ci_high": hi,
            "a": int(a_g.sum()),
            "c": int(c_g.sum()),
        }
    return out
