"""Calibrated model of P(label | genes), and how to read a box off it."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    _HAS_SKLEARN = True
except Exception:  # pragma: no cover - sklearn is an optional heavy dep
    _HAS_SKLEARN = False


__all__ = [
    "BoxLabelModel",
    "fit_label_model",
    "reliability_curve",
    "integrate_box",
    "relaxation_attribution",
]


@dataclass
class BoxLabelModel:
    """A cross-fitted, isotonic-calibrated P(label | genes)."""

    features: List[str]
    fold_models: List[object]
    calibrator: object
    p_oof: np.ndarray          # calibrated, cross-fitted P(label) per sample
    p_raw: np.ndarray          # before the isotonic layer
    y: np.ndarray
    n_splits: int
    grouped: bool              # were the folds split by study?
    auc: float
    brier_raw: float
    brier_cal: float
    ece_raw: float
    ece_cal: float
    prevalence: float
    data_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def predict(self, X) -> np.ndarray:
        """Calibrated P(label) averaged over the fold ensemble."""
        X = np.asarray(X, dtype=float)
        raw = np.mean([m.predict_proba(X)[:, 1] for m in self.fold_models], axis=0)
        return np.clip(self.calibrator.predict(raw), 0.0, 1.0)


def _ece(y, p, bins=10):
    """Expected calibration error: mean |predicted - observed| over p-bins."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    if y.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    err = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        err += m.sum() / y.size * abs(p[m].mean() - y[m].mean())
    return float(err)


def fit_label_model(
    X,
    y,
    groups: Optional[Sequence] = None,
    feature_names: Optional[Sequence[str]] = None,
    n_splits: int = 5,
    seed: int = 0,
    max_iter: int = 200,
) -> BoxLabelModel:
    """Cross-fit a gradient-boosted P(label | genes) and calibrate it."""
    if not _HAS_SKLEARN:
        raise RuntimeError("scikit-learn is required for the box model "
                           "(pip install scikit-learn)")
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional (samples x genes)")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    n_pos = int(y.sum())
    if n_pos < 10 or n_pos == y.size:
        raise ValueError(f"need at least 10 positive and 10 negative samples "
                         f"(got {n_pos} positive of {y.size})")

    names = list(feature_names) if feature_names is not None else \
        [f"f{i}" for i in range(X.shape[1])]

    grouped = groups is not None
    if grouped:
        g = np.asarray(groups, dtype=object).astype(str)
        if g.shape[0] != y.shape[0]:
            raise ValueError("groups must have the same length as y")
        n_splits = int(min(n_splits, len(np.unique(g))))
        if n_splits < 2:
            grouped = False
    if grouped:
        splits = list(GroupKFold(n_splits=n_splits).split(X, y, g))
    else:
        n_splits = int(min(n_splits, n_pos, y.size - n_pos))
        splits = list(StratifiedKFold(n_splits=max(2, n_splits), shuffle=True,
                                      random_state=seed).split(X, y))

    p_raw = np.full(y.size, np.nan)
    fold_models = []
    for tr, te in splits:
        if y[tr].sum() == 0 or y[tr].sum() == tr.size:
            continue        # this fold's studies carry no signal to learn from
        m = HistGradientBoostingClassifier(max_iter=max_iter, random_state=seed)
        m.fit(X[tr], y[tr])
        p_raw[te] = m.predict_proba(X[te])[:, 1]
        fold_models.append(m)
    if not fold_models:
        raise ValueError("no fold contained both classes - the label is confined "
                         "to too few studies to model")

    ok = np.isfinite(p_raw)

    p_cal = np.full(y.size, np.nan)
    for tr, te in splits:
        tr = tr[np.isfinite(p_raw[tr])]
        te = te[np.isfinite(p_raw[te])]
        if tr.size == 0 or te.size == 0 or y[tr].sum() in (0, tr.size):
            continue
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p_raw[tr], y[tr])
        p_cal[te] = np.clip(iso.predict(p_raw[te]), 0.0, 1.0)
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(p_raw[ok], y[ok])
    honest = np.isfinite(p_cal)
    missing = ok & ~honest
    if missing.any():
        p_cal[missing] = np.clip(calibrator.predict(p_raw[missing]), 0.0, 1.0)

    try:
        auc = float(roc_auc_score(y[ok], p_raw[ok]))
    except Exception:
        auc = float("nan")

    scored = np.unique(y[ok]).size >= 2 if ok.any() else False
    honest_scored = np.unique(y[honest]).size >= 2 if honest.any() else False
    nan = float("nan")

    return BoxLabelModel(
        features=names,
        fold_models=fold_models,
        calibrator=calibrator,
        p_oof=p_cal,
        p_raw=p_raw,
        y=y,
        n_splits=len(fold_models),
        grouped=grouped,
        auc=auc,
        brier_raw=(float(np.mean((p_raw[ok] - y[ok]) ** 2)) if scored else nan),
        # calibrated metrics only over points the calibrator never saw
        brier_cal=(float(np.mean((p_cal[honest] - y[honest]) ** 2))
                   if honest_scored else nan),
        ece_raw=(_ece(y[ok], p_raw[ok]) if scored else nan),
        ece_cal=(_ece(y[honest], p_cal[honest]) if honest_scored else nan),
        prevalence=float(y.mean()),
        data_bounds={n: (float(X[:, i].min()), float(X[:, i].max()))
                     for i, n in enumerate(names)},
    )


def reliability_curve(model: BoxLabelModel, bins: int = 10):
    """(predicted, observed, count) per probability bin, for a calibration plot."""
    p = model.p_oof
    y = model.y
    ok = np.isfinite(p)
    p, y = p[ok], y[ok].astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    pred, obs, cnt = [], [], []
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        pred.append(float(p[m].mean()))
        obs.append(float(y[m].mean()))
        cnt.append(int(m.sum()))
    return np.array(pred), np.array(obs), np.array(cnt)


def _mc_points(bounds: Sequence[Tuple[float, float]], n_mc: int, rng) -> np.ndarray:
    lows = np.array([b[0] for b in bounds], dtype=float)
    highs = np.array([b[1] for b in bounds], dtype=float)
    return lows + rng.random((int(n_mc), len(bounds))) * (highs - lows)


def integrate_box(
    model: BoxLabelModel,
    bounds: Sequence[Tuple[float, float]],
    n_mc: int = 4000,
    seed: int = 0,
) -> dict:
    """Integrate the calibrated surface over a box by Monte Carlo. Returns ``{p_uniform, fold_low, fold_high, n_mc}``."""
    rng = np.random.default_rng(seed)
    pts = _mc_points(bounds, n_mc, rng)
    per_fold = []
    for m in model.fold_models:
        raw = m.predict_proba(pts)[:, 1]
        per_fold.append(float(np.mean(np.clip(model.calibrator.predict(raw), 0, 1))))
    per_fold = np.array(per_fold, dtype=float)
    return {
        "p_uniform": float(per_fold.mean()),
        "fold_low": float(per_fold.min()),
        "fold_high": float(per_fold.max()),
        "n_mc": int(n_mc),
    }


def relaxation_attribution(
    model: BoxLabelModel,
    bounds: Sequence[Tuple[float, float]],
    n_mc: int = 4000,
    seed: int = 0,
) -> Dict[str, dict]:
    """How much each gene's bound is holding the box up, as ``{p_relaxed, drop}`` per gene."""
    base = integrate_box(model, bounds, n_mc=n_mc, seed=seed)["p_uniform"]
    out: Dict[str, dict] = {}
    for i, name in enumerate(model.features):
        relaxed = list(bounds)
        relaxed[i] = model.data_bounds.get(name, bounds[i])
        p = integrate_box(model, relaxed, n_mc=n_mc, seed=seed)["p_uniform"]
        out[name] = {"p_relaxed": p, "drop": float(base - p)}
    return out
