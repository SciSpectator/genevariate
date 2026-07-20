"""
Activity inference — transcription-factor and pathway activities via decoupleR.

Standard over-representation / GSEA answers "which gene sets are over-represented".
Activity inference answers the mechanistic follow-up: *which regulators are active*.
:func:`tf_activity` scores transcription-factor activity from the CollecTRI
regulon, and :func:`pathway_activity` scores pathway activity from PROGENy, both
via `decoupler` (Saez-Rodriguez lab).

decoupler is an optional dependency (``pip install decoupler``) and its public
API has changed across major versions, so every call is import-guarded and the
1.x and 2.x entry points are both attempted. When decoupler is absent every
function raises a clear ``RuntimeError`` — nothing else in GeneVariate imports it.

Input is GeneVariate's canonical platform DataFrame (rows = samples, a ``GSM``
column, optional ``Classified_*`` metadata, remaining columns numeric genes).
"""
from __future__ import annotations

from importlib.util import find_spec
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# decoupler is optional and heavy; probe for it here and import it lazily in
# _require_decoupler so importing this module stays cheap.
_dc = None
_HAS_DECOUPLER = find_spec("decoupler") is not None


_META_PREFIXES = ("GSM", "series_id")


def _require_decoupler() -> None:
    global _dc
    if not _HAS_DECOUPLER:
        raise RuntimeError(
            "decoupler is not installed. `pip install decoupler` to run "
            "activity inference (TF activities via CollecTRI, pathway "
            "activities via PROGENy).")
    if _dc is None:
        import decoupler as _dc_mod
        _dc = _dc_mod


def _expr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical platform DataFrame -> samples x genes numeric matrix."""
    meta = [c for c in df.columns
            if c in _META_PREFIXES or str(c).startswith("Classified_")]
    genes = [c for c in df.columns if c not in meta]
    idx = (df["GSM"].astype(str).values if "GSM" in df.columns
           else df.index.astype(str))
    mat = df[genes].apply(pd.to_numeric, errors="coerce")
    mat.index = idx
    mat.columns = [str(c).upper() for c in mat.columns]
    return mat.fillna(0.0)


def _get_net(kind: str, organism: str) -> pd.DataFrame:
    """Fetch a CollecTRI (``tf``) or PROGENy (``pathway``) network, version-safe."""
    if kind == "tf":
        for getter in ("get_collectri",):
            fn = getattr(_dc, getter, None)
            if fn is not None:
                return fn(organism=organism)
        op = getattr(_dc, "op", None)  # decoupler 2.x
        if op is not None and hasattr(op, "collectri"):
            return op.collectri(organism=organism)
    else:
        fn = getattr(_dc, "get_progeny", None)
        if fn is not None:
            return fn(organism=organism, top=500)
        op = getattr(_dc, "op", None)
        if op is not None and hasattr(op, "progeny"):
            return op.progeny(organism=organism, top=500)
    raise RuntimeError("Could not locate a network getter in this decoupler "
                       "version.")


def _run_ulm(mat: pd.DataFrame, net: pd.DataFrame
             ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Univariate linear model activity, returning (activities, p-values).

    Handles both decoupler 1.x (``dc.run_ulm`` -> tuple of DataFrames) and 2.x
    (``dc.mt.ulm`` writing into an AnnData-like object).
    """
    run_ulm = getattr(_dc, "run_ulm", None)
    if run_ulm is not None:
        acts, pvals = run_ulm(mat=mat, net=net, verbose=False)
        return acts, pvals
    mt = getattr(_dc, "mt", None)
    if mt is not None and hasattr(mt, "ulm"):
        try:
            import anndata as ad
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("decoupler 2.x needs anndata installed") from exc
        adata = ad.AnnData(mat)
        mt.ulm(data=adata, net=net, verbose=False)
        acts = adata.obsm["score_ulm"]
        pvals = adata.obsm.get("padj_ulm", adata.obsm.get("pvals_ulm"))
        return pd.DataFrame(acts), pd.DataFrame(pvals)
    raise RuntimeError("This decoupler version exposes neither run_ulm nor mt.ulm.")


def _summarise(acts: pd.DataFrame, kind: str, organism: str) -> Dict[str, object]:
    """Rank sources by mean activity across samples and build a report."""
    mean_act = acts.mean(axis=0)
    order = mean_act.abs().sort_values(ascending=False)
    ranked = pd.DataFrame({
        "mean_activity": mean_act.reindex(order.index),
        "abs_activity": order,
    })
    label = "transcription factors (CollecTRI)" if kind == "tf" \
        else "pathways (PROGENy)"
    top = ranked.head(15)
    lines = [f"# Activity inference — {label}\n",
             f"Scored **{acts.shape[1]}** sources across "
             f"**{acts.shape[0]}** samples (ULM, {organism}).\n",
             "**Most active (|mean score|):**"]
    for name, row in top.iterrows():
        lines.append(f"- **{name}**: {row['mean_activity']:+.3f}")
    return {
        "activities": acts,
        "ranked": ranked,
        "summary": (f"Inferred activity for {acts.shape[1]} {label} across "
                    f"{acts.shape[0]} samples. Top: "
                    + ", ".join(list(top.index[:8])) + "."),
        "report": "\n".join(lines),
    }


def tf_activity(df: pd.DataFrame, organism: str = "human") -> Dict[str, object]:
    """Transcription-factor activity per sample from the CollecTRI regulon.

    Returns a dict with ``activities`` (samples x TFs), ``ranked`` (TFs by mean
    |activity|), ``summary`` and markdown ``report``.
    """
    _require_decoupler()
    mat = _expr_matrix(df)
    net = _get_net("tf", organism)
    acts, _ = _run_ulm(mat, net)
    return _summarise(acts, "tf", organism)


def pathway_activity(df: pd.DataFrame, organism: str = "human"
                     ) -> Dict[str, object]:
    """Pathway activity per sample from PROGENy.

    Returns a dict with ``activities`` (samples x pathways), ``ranked``,
    ``summary`` and markdown ``report``.
    """
    _require_decoupler()
    mat = _expr_matrix(df)
    net = _get_net("pathway", organism)
    acts, _ = _run_ulm(mat, net)
    return _summarise(acts, "pathway", organism)


def run_activity(df: pd.DataFrame, kind: str = "tf",
                 organism: str = "human") -> Dict[str, object]:
    """Dispatch to :func:`tf_activity` (``kind='tf'``) or
    :func:`pathway_activity` (``kind='pathway'``)."""
    if kind == "pathway":
        return pathway_activity(df, organism=organism)
    return tf_activity(df, organism=organism)
