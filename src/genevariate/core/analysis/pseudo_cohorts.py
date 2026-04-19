"""
Embedding-clustered pseudo-cohorts.

Uses an embedding model (nomic-embed-text via the local Ollama server, same
backbone MemoryAgent uses) to vectorise the LLM-curated condition labels
and cluster samples *without* a priori case/control assignment. Returns
auto-discovered cohort IDs that can then be fed into the mean or ΔVariance
enrichment pipeline.

Why this matters:
  * GEO series often have unstructured text labels ("MCF7 treated with 1uM
    tamoxifen 24h" vs "MCF7 DMSO 24h") that the user doesn't want to hand-
    label into case/control.
  * Curating those labels into embeddings + KMeans gives you natural
    cohorts derived from semantic similarity, not string matching.
  * Each discovered cluster pair can be run through enrichment → a grid of
    pathway findings across all auto-discovered comparisons.

Pure-Python fallback when Ollama is unavailable: TF-IDF char n-gram vectors
from sklearn. That's a weaker signal but the feature still functions on
machines without LLMs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import requests
    _HAS_REQ = True
except Exception:
    requests = None
    _HAS_REQ = False

try:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import silhouette_score
    _HAS_SK = True
except Exception:
    _HAS_SK = False


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


# -----------------------------------------------------------------
# Embedding backends
# -----------------------------------------------------------------
def _ollama_embed(texts: Sequence[str], model: str = "nomic-embed-text",
                  url: str = OLLAMA_URL) -> Optional[np.ndarray]:
    if not _HAS_REQ:
        return None
    try:
        resp = requests.post(f"{url.rstrip('/')}/api/embed",
                             json={"model": model, "input": list(texts)},
                             timeout=60)
        if resp.status_code == 404:
            # Fall back to singular /api/embeddings endpoint
            out = []
            for t in texts:
                r = requests.post(f"{url.rstrip('/')}/api/embeddings",
                                  json={"model": model, "prompt": t},
                                  timeout=30)
                r.raise_for_status()
                out.append(r.json().get("embedding"))
            vecs = np.array(out, dtype=np.float32)
        else:
            resp.raise_for_status()
            vecs = np.array(resp.json().get("embeddings", []), dtype=np.float32)
        if vecs.size == 0:
            return None
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms
    except Exception:
        return None


def _tfidf_embed(texts: Sequence[str]) -> np.ndarray:
    """Pure-Python fallback: character n-gram TF-IDF + L2 norm."""
    if not _HAS_SK:
        raise RuntimeError("scikit-learn is required for the TF-IDF fallback.")
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    mat = vec.fit_transform([str(t) for t in texts]).toarray().astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def embed_labels(labels: Sequence[str],
                 prefer: str = "ollama") -> Tuple[np.ndarray, str]:
    """
    Returns (embeddings, backend_used). Tries Ollama first if prefer='ollama',
    otherwise TF-IDF. Always falls back to TF-IDF if the primary fails.
    """
    if prefer == "ollama":
        vecs = _ollama_embed(list(labels))
        if vecs is not None and len(vecs) == len(labels):
            return vecs, "ollama:nomic-embed-text"
    return _tfidf_embed(list(labels)), "tfidf_char_ngram"


# -----------------------------------------------------------------
# Cohort discovery
# -----------------------------------------------------------------
@dataclass
class PseudoCohortResult:
    cluster_of: Dict[str, int]          # GSM -> cluster id
    label_used: Dict[str, str]          # GSM -> raw label text
    k_selected: int
    silhouette: float
    backend: str
    cluster_stability: Optional[Dict[int, float]] = None  # cluster_id -> mean bootstrap Jaccard
    stability_floor: float = 0.6        # below this → refuse cohort for enrichment
    n_bootstrap: int = 0


def _bootstrap_jaccard_stability(X: np.ndarray,
                                  reference_labels: np.ndarray,
                                  k: int,
                                  n_bootstrap: int = 50,
                                  subsample_frac: float = 0.8,
                                  random_state: int = 0) -> Dict[int, float]:
    """
    Hennig-style (2007) bootstrap cluster stability.

    For each bootstrap resample (subsample of rows), re-run KMeans at the same
    `k` and compute the best-match Jaccard of each reference cluster against
    any bootstrap cluster. Returns {cluster_id -> mean max-Jaccard across
    bootstraps}. Values ≥0.75 indicate a stable cluster; 0.6–0.75 is patterny-
    but-doubtful; <0.6 is noise-level and should be refused for enrichment.

    Reference: Hennig, "Cluster-wise assessment of cluster stability",
    Computational Statistics & Data Analysis 52 (2008) 258–271.
    """
    n = X.shape[0]
    m = max(2, int(round(subsample_frac * n)))
    rng = np.random.default_rng(random_state)
    ref_clusters = {c: set(np.where(reference_labels == c)[0])
                    for c in sorted(set(reference_labels))}
    scores: Dict[int, List[float]] = {c: [] for c in ref_clusters}

    for b in range(n_bootstrap):
        idx = rng.choice(n, size=m, replace=False)
        Xb = X[idx]
        try:
            km = KMeans(n_clusters=k, n_init=5,
                        random_state=int(rng.integers(0, 2**31 - 1))).fit(Xb)
        except Exception:
            continue
        for c, ref_set in ref_clusters.items():
            ref_in_boot = ref_set.intersection(idx.tolist())
            if not ref_in_boot:
                continue
            # indices in the bootstrap array that belong to reference cluster c
            ref_boot_pos = {i for i, orig in enumerate(idx) if orig in ref_in_boot}
            best_j = 0.0
            for bc in set(km.labels_):
                boot_set = set(np.where(km.labels_ == bc)[0].tolist())
                inter = len(ref_boot_pos & boot_set)
                union = len(ref_boot_pos | boot_set)
                if union == 0:
                    continue
                best_j = max(best_j, inter / union)
            scores[c].append(best_j)

    return {c: float(np.mean(vals)) if vals else 0.0 for c, vals in scores.items()}


def discover_pseudo_cohorts(sample_labels: Dict[str, str],
                             k_range: Tuple[int, int] = (2, 8),
                             random_state: int = 0,
                             prefer_backend: str = "ollama",
                             n_bootstrap: int = 50,
                             stability_floor: float = 0.6) -> PseudoCohortResult:
    """
    Cluster samples by embedding-similarity of their LLM-curated labels.

    sample_labels: {GSM -> curated free-text label (e.g. 'MCF7 tamoxifen 24h')}
    k_range:      (k_min, k_max) inclusive range to sweep.
    n_bootstrap:  number of resamples for Hennig cluster-stability (0 disables).
    stability_floor: Jaccard below this disqualifies a cluster for enrichment.
    Returns PseudoCohortResult with the k chosen by silhouette score and
    per-cluster bootstrap Jaccard stability.
    """
    if not _HAS_SK:
        raise RuntimeError("scikit-learn is required for cohort discovery.")

    gsms = [g for g, v in sample_labels.items() if isinstance(v, str) and v.strip()]
    texts = [sample_labels[g].strip() for g in gsms]
    if len(gsms) < max(k_range[0], 2):
        raise ValueError(f"Need at least {k_range[0]} labelled samples, got {len(gsms)}")

    X, backend = embed_labels(texts, prefer=prefer_backend)

    k_min, k_max = k_range
    k_max = min(k_max, len(gsms) - 1)
    k_min = max(2, k_min)

    best = {"k": k_min, "sil": -1.0, "labels": None}
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X)
        if len(set(km.labels_)) < 2:
            continue
        try:
            sil = silhouette_score(X, km.labels_)
        except Exception:
            sil = -1.0
        if sil > best["sil"]:
            best = {"k": k, "sil": sil, "labels": km.labels_}

    if best["labels"] is None:
        raise RuntimeError("Failed to find a valid clustering.")

    cluster_of = {g: int(c) for g, c in zip(gsms, best["labels"])}

    stability: Optional[Dict[int, float]] = None
    if n_bootstrap and n_bootstrap > 0 and len(gsms) >= 6:
        try:
            stability = _bootstrap_jaccard_stability(
                X, np.asarray(best["labels"]), k=best["k"],
                n_bootstrap=n_bootstrap, random_state=random_state,
            )
            stability = {int(c): float(v) for c, v in stability.items()}
        except Exception:
            stability = None

    return PseudoCohortResult(
        cluster_of=cluster_of,
        label_used={g: sample_labels[g] for g in gsms},
        k_selected=best["k"],
        silhouette=float(best["sil"]),
        backend=backend,
        cluster_stability=stability,
        stability_floor=float(stability_floor),
        n_bootstrap=int(n_bootstrap or 0),
    )


def cohort_summary(result: PseudoCohortResult) -> pd.DataFrame:
    """Summarise discovered cohorts: n_samples + representative labels + stability."""
    rows = []
    stab = result.cluster_stability or {}
    for cid in sorted(set(result.cluster_of.values())):
        members = [g for g, c in result.cluster_of.items() if c == cid]
        labels = [result.label_used[g] for g in members]
        rep = pd.Series(labels).mode()
        rep_txt = str(rep.iloc[0]) if len(rep) else ""
        j = stab.get(int(cid))
        rows.append({
            "cluster":       cid,
            "n_samples":     len(members),
            "representative_label": rep_txt[:100],
            "sample_labels": "; ".join(sorted(set(labels))[:3])[:200],
            "bootstrap_jaccard": float(j) if j is not None else np.nan,
            "stable": (j is not None and j >= result.stability_floor),
        })
    return pd.DataFrame(rows)


def cohort_pairs(result: PseudoCohortResult,
                 min_size: int = 3,
                 enforce_stability: bool = True) -> List[Tuple[int, int]]:
    """
    Enumerate all (i, j) cluster pairs with each cluster ≥ `min_size` samples.

    If `enforce_stability` is True and bootstrap Jaccard scores are available,
    pairs with either cluster below `result.stability_floor` are refused
    (returned list excludes them). This prevents wasting enrichment compute on
    noise-level clusters.
    """
    sizes = {c: sum(1 for v in result.cluster_of.values() if v == c)
             for c in set(result.cluster_of.values())}
    kept = sorted([c for c, n in sizes.items() if n >= min_size])

    if enforce_stability and result.cluster_stability:
        floor = result.stability_floor
        kept = [c for c in kept
                if result.cluster_stability.get(int(c), 0.0) >= floor]

    return [(i, j) for idx, i in enumerate(kept) for j in kept[idx + 1:]]
