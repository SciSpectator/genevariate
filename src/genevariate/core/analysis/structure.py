"""Unsupervised structure over samples: projection, clustering, density peaks."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "scale_features",
    "project",
    "cluster",
    "density_peaks",
    "cluster_composition",
]


def _sklearn():
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    return PCA, StandardScaler, KMeans, DBSCAN, silhouette_score


def scale_features(X) -> np.ndarray:
    """Standardised copy of ``X``."""
    _, StandardScaler, _, _, _ = _sklearn()
    return StandardScaler().fit_transform(np.asarray(X, dtype=float))


def project(X, *, method: str = "pca", seed: int = 42,
            n_neighbors: int = 15) -> Dict[str, object]:
    """Returns ``coords``, ``method`` and ``explained``, for ``pca``, ``umap`` or ``tsne``."""
    X_s = np.asarray(X, dtype=float)
    if X_s.ndim == 1:
        X_s = X_s.reshape(-1, 1)
    if X_s.shape[0] < 3:
        raise ValueError("at least three samples are needed to project them")
    if X_s.shape[1] < 2:
        col = X_s[:, 0]
        return {"coords": np.column_stack([col, np.zeros_like(col)]),
                "method": "single feature", "explained": [1.0, 0.0]}
    want = str(method or "pca").lower()
    if want in ("tsne", "t-sne"):
        from sklearn.manifold import TSNE
        perp = max(2.0, min(float(n_neighbors), (X_s.shape[0] - 1) / 4.0, 50.0))
        coords = TSNE(n_components=2, perplexity=perp, random_state=seed,
                      init="pca", learning_rate="auto").fit_transform(X_s)
        return {"coords": np.asarray(coords), "method": "tsne",
                "explained": None, "perplexity": float(perp)}
    if want == "umap":
        try:
            from umap import UMAP
        except ImportError:
            UMAP = None
        if UMAP is not None and X_s.shape[0] > 15:
            coords = UMAP(n_components=2,
                          n_neighbors=min(n_neighbors, X_s.shape[0] - 1),
                          random_state=seed).fit_transform(X_s)
            return {"coords": np.asarray(coords), "method": "umap",
                    "explained": None}
    PCA, _, _, _, _ = _sklearn()
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_s)
    return {"coords": np.asarray(coords), "method": "pca",
            "explained": [float(v) for v in pca.explained_variance_ratio_]}


def cluster(X, *, method: str = "kmeans", k: Optional[int] = None,
            max_k: int = 8, seed: int = 42, eps: float = 0.8,
            min_samples: Optional[int] = None,
            against: Optional[Sequence] = None) -> Dict[str, object]:
    """Returns ``method``, ``labels``, ``names``, ``k``, ``n_clusters``, ``n_noise``, ``silhouette``, ``ari``."""
    X_s = np.asarray(X, dtype=float)
    if X_s.ndim == 1:
        X_s = X_s.reshape(-1, 1)
    n = X_s.shape[0]
    if n < 3:
        raise ValueError("at least three samples are needed to cluster them")
    _, _, KMeans, DBSCAN, silhouette_score = _sklearn()
    want = str(method or "kmeans").lower()

    def _ari(labels):
        if against is None:
            return None
        try:
            from sklearn.metrics import adjusted_rand_score
            from sklearn.preprocessing import LabelEncoder
            truth = LabelEncoder().fit_transform(
                [str(v) for v in list(against)[:len(labels)]])
            return float(adjusted_rand_score(truth, labels))
        except Exception:
            return None

    if want == "dbscan":
        mins = min_samples if min_samples else max(3, n // 50)
        labels = DBSCAN(eps=eps, min_samples=mins).fit_predict(X_s)
        n_clusters = len(set(labels) - {-1})
        keep = labels != -1
        sil = None
        if n_clusters >= 2 and keep.sum() > n_clusters:
            try:
                sil = float(silhouette_score(X_s[keep], labels[keep]))
            except Exception:
                sil = None
        return {"method": "dbscan", "labels": labels,
                "names": [f"C{c}" if c >= 0 else "Noise" for c in labels],
                "n_clusters": int(n_clusters),
                "n_noise": int((labels == -1).sum()),
                "eps": float(eps), "min_samples": int(mins),
                "silhouette": sil, "k": None, "ari": _ari(labels)}

    if k:
        best_k, best_sc = max(2, min(int(k), n)), None
    else:
        top = max(3, min(int(max_k), n - 1) + 1)
        best_k, best_sc = 2, -1.0
        for cand in range(2, top):
            try:
                lab = KMeans(n_clusters=cand, n_init=10,
                             random_state=seed).fit_predict(X_s)
                sc = float(silhouette_score(X_s, lab))
            except Exception:
                continue
            if sc > best_sc:
                best_sc, best_k = sc, cand
        if best_sc is not None and best_sc < -0.5:
            best_sc = None
    km = KMeans(n_clusters=best_k, n_init=10, random_state=seed).fit(X_s)
    labels = km.labels_
    if best_sc is None and best_k < n:
        try:
            best_sc = float(silhouette_score(X_s, labels))
        except Exception:
            best_sc = None
    return {"method": "kmeans", "labels": labels,
            "names": [f"C{c}" for c in labels],
            "k": int(best_k), "silhouette": best_sc,
            "n_clusters": int(best_k), "n_noise": 0, "ari": _ari(labels),
            "centers": np.asarray(km.cluster_centers_),
            "inertia": float(km.inertia_)}


def density_peaks(X, *, percent: float = 2.0, max_n: Optional[int] = None,
                  seed: int = 0) -> Dict[str, object]:
    """Rodriguez and Laio decision graph, as ``rho``, ``delta``, ``dc``, ``order``, ``index``, ``distances``."""
    from scipy.spatial.distance import pdist, squareform
    X_s = np.asarray(X, dtype=float)
    if X_s.ndim == 1:
        X_s = X_s.reshape(-1, 1)
    if X_s.shape[0] < 10:
        raise ValueError("at least ten samples are needed for a decision graph")
    index = np.arange(X_s.shape[0])
    if max_n and X_s.shape[0] > max_n:
        index = np.sort(np.random.default_rng(seed).choice(
            X_s.shape[0], size=int(max_n), replace=False))
        X_s = X_s[index]
    dists = squareform(pdist(X_s))
    dc = float(np.percentile(dists, percent)) or 1e-5
    rho = np.sum(np.exp(-(dists / dc) ** 2), axis=1) - 1
    delta = np.zeros(len(X_s))
    order = np.argsort(-rho)
    for i, idx in enumerate(order):
        if i == 0:
            delta[idx] = dists[idx, :].max()
        else:
            delta[idx] = dists[idx, order[:i]].min()
    return {"rho": rho, "delta": delta, "dc": dc, "order": order,
            "index": index,
            "distances": dists}


def cluster_composition(names: Sequence[str],
                        groups: Sequence[str]) -> pd.DataFrame:
    """What each cluster is made of, as ``cluster``, ``group``, ``samples``, ``share``."""
    tab = pd.crosstab(pd.Series(list(names), name="cluster"),
                      pd.Series(list(groups), name="group"))
    out = tab.reset_index().melt(id_vars="cluster", var_name="group",
                                 value_name="samples")
    out = out[out["samples"] > 0].copy()
    totals = out.groupby("cluster")["samples"].transform("sum")
    out["share"] = (out["samples"] / totals).round(4)
    return out.sort_values(["cluster", "samples"],
                           ascending=[True, False]).reset_index(drop=True)
