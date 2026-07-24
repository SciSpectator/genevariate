"""
Multi-gene conjunction boxes and the multiplicative-null synergy score.

Brushing a range on one gene gives a slab; brushing ranges on *k* genes gives an
axis-aligned box (the intersection of the slabs). The question that box answers
is not "is this label enriched here" - each single gene already answers that -
but "is the combination doing something the genes do not do on their own?".

Two nulls are reported, because they answer different questions.

*How many samples did we expect?* - the multiplicative-lift null, which is what
a reader intuitively pictures:

    lift(S)  = P(label | S) / P(label)
    exp_a    = P(label) * n_box * prod_g lift(slab_g)

*Is the combination doing anything?* - the k-way interaction of the log-linear
model on the 2^k x label table, i.e. the ratio of odds ratios:

    synergy  = prod over the 2^k cells of odds(cell) ^ (-1)^(zeros in cell)

For two genes that is the familiar ``(o11 * o00) / (o10 * o01)``. Unlike a lift
ratio it does not saturate against the 1/P(label) ceiling, so it can register
positive synergy for common labels as well as redundancy. synergy > 1 means the
genes reinforce each other, synergy < 1 means they are redundant - they mark the
same samples twice - and synergy ~ 1 means the box is exactly what the single
genes already told you.

A confidence interval is bootstrapped over **studies**, because a box that
shrinks onto a handful of GSEs is exactly where a naive count lies.

Pure numpy/scipy and Tk-free, shared by the GUI, the chatbot tools and tests.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from .overdispersion import effective_sample_size, estimate_rho

try:
    from scipy.stats import fisher_exact

    _HAS_SCIPY = True
except Exception:  # pragma: no cover - scipy is a hard dep in practice
    _HAS_SCIPY = False


__all__ = [
    "conjunction_mask",
    "multiplicative_null",
    "synergy_diagnostics",
]


def conjunction_mask(masks: Mapping[str, Sequence[bool]]) -> np.ndarray:
    """AND every gene slab together into the box membership mask."""
    if not masks:
        raise ValueError("at least one gene mask is required")
    arrays = [np.asarray(m, dtype=bool) for m in masks.values()]
    n = arrays[0].shape[0]
    if any(a.shape[0] != n for a in arrays):
        raise ValueError("all gene masks must have the same length")
    box = arrays[0].copy()
    for a in arrays[1:]:
        box &= a
    return box


def multiplicative_null(marginal_lifts: Sequence[float]) -> float:
    """Expected box lift if the genes carried independent information."""
    exp = 1.0
    for l in marginal_lifts:
        if not np.isfinite(l):
            return float("nan")
        exp *= float(l)
    return float(exp)


def _cell_masks(gene_masks):
    """The 2^k cells of the gene x gene table, with their log-linear signs.

    Yields ``(mask, sign)`` where sign is +1 for cells with an even number of
    genes outside their slab and -1 for odd - the contrast that isolates the
    k-way interaction from all lower-order terms.
    """
    from itertools import product

    names = list(gene_masks)
    n = len(next(iter(gene_masks.values())))
    for combo in product((True, False), repeat=len(names)):
        mask = np.ones(n, dtype=bool)
        for name, inside in zip(names, combo):
            mask &= gene_masks[name] if inside else ~gene_masks[name]
        sign = -1.0 if (len(combo) - sum(combo)) % 2 else 1.0
        yield mask, sign


def _interaction_or(counts, sizes, signs):
    """prod odds(cell)^sign with a Haldane-Anscombe 0.5 correction."""
    a = np.asarray(counts, dtype=float)
    n = np.asarray(sizes, dtype=float)
    odds = (a + 0.5) / (n - a + 0.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.exp((np.asarray(signs, dtype=float) * np.log(odds)).sum()))


def synergy_diagnostics(
    masks: Mapping[str, Sequence[bool]],
    labels: Sequence,
    groups: Optional[Sequence] = None,
    values: Optional[Sequence] = None,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, dict]:
    """
    Per-label-value synergy of a multi-gene conjunction box.

    ``masks`` maps gene name -> boolean membership in that gene's brushed slab;
    all masks, ``labels`` and ``groups`` are aligned over the platform samples
    that carry a label. ``groups`` are study ids (GSE); ``None`` means no study
    information, in which case the clumping fields come back as NaN/None rather
    than pretending the samples are independent draws.

    Returns ``{value: {...}}`` with, per value:

    ``n_box``          samples in the box
    ``a``              of those, how many carry the label
    ``lift_box``       P(label | box) / P(label)
    ``marginal_lifts`` {gene: P(label | slab) / P(label)}
    ``expected_lift``  product of the marginal lifts
    ``exp_a``          label count that multiplicative-lift null predicts in the box
    ``synergy``        k-way log-linear interaction odds ratio (see module docs)
    ``ci_low/ci_high`` percentile CI on ``synergy``, bootstrapped over studies
    ``n_gse``          studies contributing at least one labelled box sample
    ``rho``            beta-binomial clumping of the label across studies
    ``n_eff_box``      box size discounted for that clumping
    ``p``              one-sided Fisher p for box vs rest (plain enrichment)
    ``empty_cells``    gene-combination cells with no samples; ``synergy`` is NaN
                       whenever this is non-zero, because the interaction is then
                       not identified rather than merely uncertain
    """
    box = conjunction_mask(masks)
    lab = np.asarray(labels, dtype=object).astype(str)
    if lab.shape[0] != box.shape[0]:
        raise ValueError("labels must have the same length as the gene masks")

    gene_masks = {g: np.asarray(m, dtype=bool) for g, m in masks.items()}
    vals = list(values) if values is not None else list(np.unique(lab))

    n_all = int(lab.shape[0])
    n_box = int(box.sum())
    gene_n = {g: int(m.sum()) for g, m in gene_masks.items()}

    inv = None
    n_groups = 0
    weights = None
    if groups is not None:
        grp = np.asarray(groups, dtype=object).astype(str)
        if grp.shape[0] != lab.shape[0]:
            raise ValueError("groups must have the same length as labels")
        _, inv = np.unique(grp, return_inverse=True)
        n_groups = int(inv.max()) + 1 if inv.size else 0
        if n_boot and n_boot > 0 and n_groups > 1:
            rng = np.random.default_rng(seed)
            weights = rng.multinomial(
                n_groups, np.full(n_groups, 1.0 / n_groups), size=int(n_boot)
            ).astype(np.float32)

    def _per_study(mask):
        return np.bincount(inv, weights=mask.astype(float),
                           minlength=n_groups).astype(np.float32)

    # the 2^k gene-combination cells the log-linear interaction contrasts
    cells = list(_cell_masks(gene_masks))
    cell_masks = [m for m, _ in cells]
    cell_signs = np.array([s for _, s in cells], dtype=float)
    cell_n = np.array([int(m.sum()) for m in cell_masks], dtype=float)
    n_empty = int((cell_n == 0).sum())

    if inv is not None:
        sizes_f = np.bincount(inv, minlength=n_groups).astype(float)
        if weights is not None:
            # (n_cells, n_groups) -> (n_cells, n_boot) once, reused per value
            cell_n_g = np.stack([_per_study(m) for m in cell_masks])
            cell_n_b = cell_n_g @ weights.T

    out: Dict[str, dict] = {}
    k = len(gene_masks)

    for v in vals:
        key = str(v)
        hit = lab == key
        a = int((hit & box).sum())
        n_hit = int(hit.sum())
        p_all = n_hit / n_all if n_all else float("nan")

        lift_box = (a / n_box) / p_all if (n_box and p_all > 0) else float("nan")
        marg = {}
        for g, m in gene_masks.items():
            ng = gene_n[g]
            ag = int((hit & m).sum())
            marg[g] = (ag / ng) / p_all if (ng and p_all > 0) else float("nan")
        exp_lift = multiplicative_null(list(marg.values()))

        cell_a = np.array([float((hit & m).sum()) for m in cell_masks])
        synergy = (float("nan") if n_empty
                   else _interaction_or(cell_a, cell_n, cell_signs))

        p_val = float("nan")
        if _HAS_SCIPY and n_box and n_box < n_all:
            c = n_hit - a
            try:
                p_val = float(fisher_exact([[a, n_box - a],
                                            [c, (n_all - n_box) - c]],
                                           alternative="greater")[1])
            except Exception:
                p_val = float("nan")

        rec = {
            "n_box": n_box,
            "a": a,
            "n_hit": n_hit,
            "lift_box": float(lift_box),
            "marginal_lifts": marg,
            "expected_lift": float(exp_lift),
            "synergy": float(synergy),
            "exp_a": float(exp_lift * p_all * n_box) if np.isfinite(exp_lift) else float("nan"),
            "empty_cells": n_empty,
            "p": p_val,
            "n_gse": None,
            "rho": float("nan"),
            "n_eff_box": float(n_box),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }

        if inv is not None:
            hit_box_g = _per_study(hit & box)
            rec["n_gse"] = int((hit_box_g > 0).sum())
            rho = estimate_rho(_per_study(hit).astype(float), sizes_f)
            mean_cluster = float(sizes_f[sizes_f > 0].mean()) if n_groups else float("nan")
            rec["rho"] = float(rho)
            rec["n_eff_box"] = effective_sample_size(n_box, mean_cluster, rho)

            if weights is not None and not n_empty:
                cell_a_b = np.stack([_per_study(hit & m) for m in cell_masks]) @ weights.T
                odds = (cell_a_b + 0.5) / (cell_n_b - cell_a_b + 0.5)
                with np.errstate(divide="ignore", invalid="ignore"):
                    syn_b = np.exp(cell_signs @ np.log(np.where(odds > 0, odds, np.nan)))
                syn_b = syn_b[np.isfinite(syn_b)]
                if syn_b.size >= 20:
                    rec["ci_low"] = float(np.quantile(syn_b, alpha / 2.0))
                    rec["ci_high"] = float(np.quantile(syn_b, 1.0 - alpha / 2.0))

        rec["n_genes"] = k
        out[key] = rec

    return out
