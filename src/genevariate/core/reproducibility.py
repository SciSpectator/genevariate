"""
Per-run reproducibility manifest.

An agent-driven science tool must be able to say exactly *how* a result was
produced. :func:`build_manifest` captures the four things needed to reproduce an
analysis — parameters, software versions, random seeds, and a content hash of
the input data — into a plain dict that can be serialised to JSON or rendered as
markdown and attached to any result.

Tk-free, headless, and dependency-light: version probing degrades gracefully
when a package is absent, and data hashing works on DataFrames, arrays, dicts of
those, or file paths.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata as _im
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

# Packages whose versions materially affect analysis output.
_TRACKED_PACKAGES: tuple[str, ...] = (
    "genevariate", "numpy", "pandas", "scipy", "scikit-learn",
    "gseapy", "anndata", "diptest", "decoupler",
    "harmonypy", "combat", "statsmodels",
)


def _package_versions(packages: Iterable[str] = _TRACKED_PACKAGES
                      ) -> Dict[str, str]:
    """Installed version of each tracked package (``"not installed"`` if absent)."""
    out: Dict[str, str] = {}
    for name in packages:
        try:
            out[name] = _im.version(name)
        except Exception:
            out[name] = "not installed"
    return out


def _hash_frame(df: pd.DataFrame) -> str:
    """Stable SHA-256 of a DataFrame's shape, columns and values."""
    h = hashlib.sha256()
    h.update(f"{df.shape}".encode())
    h.update(",".join(str(c) for c in df.columns).encode())
    try:
        h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    except Exception:
        # Fall back to a bytes view of the numeric block.
        h.update(np.ascontiguousarray(df.select_dtypes("number").to_numpy(
            dtype=float, na_value=np.nan)).tobytes())
    return h.hexdigest()


def hash_data(obj: Any) -> str:
    """Content hash of an analysis input.

    Handles a DataFrame, a numpy array, a dict/list of those, or a filesystem
    path (hashed by contents). Returns a short SHA-256 hex digest (16 chars).
    Never raises — an unhashable input yields ``"unhashable"``.
    """
    try:
        h = hashlib.sha256()
        if isinstance(obj, pd.DataFrame):
            h.update(_hash_frame(obj).encode())
        elif isinstance(obj, pd.Series):
            h.update(_hash_frame(obj.to_frame()).encode())
        elif isinstance(obj, np.ndarray):
            h.update(np.ascontiguousarray(obj).tobytes())
        elif isinstance(obj, dict):
            for k in sorted(obj, key=str):
                h.update(str(k).encode())
                h.update(hash_data(obj[k]).encode())
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                h.update(hash_data(item).encode())
        elif isinstance(obj, str):
            import os
            if os.path.exists(obj):
                with open(obj, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
            else:
                h.update(obj.encode())
        else:
            h.update(repr(obj).encode())
        return h.hexdigest()[:16]
    except Exception:
        return "unhashable"


def build_manifest(tool: str,
                   params: Optional[Dict[str, Any]] = None,
                   inputs: Optional[Dict[str, Any]] = None,
                   seed: Optional[int] = None,
                   packages: Iterable[str] = _TRACKED_PACKAGES
                   ) -> Dict[str, Any]:
    """Assemble a reproducibility manifest for a single analysis run.

    Parameters
    ----------
    tool : name of the analysis/tool that produced the result.
    params : the resolved parameters the analysis was run with.
    inputs : named analysis inputs (DataFrames/arrays/paths) to content-hash.
    seed : the random seed used, if any.

    Returns a JSON-serialisable dict with ``tool, timestamp, params, seed,
    data_hashes, versions, environment``.
    """
    data_hashes = {name: hash_data(obj)
                   for name, obj in (inputs or {}).items()}
    return {
        "tool": tool,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "params": _jsonable(params or {}),
        "seed": seed,
        "data_hashes": data_hashes,
        "versions": _package_versions(packages),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion of params to JSON-friendly values."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, pd.DataFrame):
        return f"<DataFrame {obj.shape[0]}x{obj.shape[1]}>"
    return str(obj)


def manifest_to_markdown(manifest: Dict[str, Any]) -> str:
    """Render a manifest as a compact markdown block for a result report."""
    lines = ["### Reproducibility",
             f"- **tool**: `{manifest.get('tool')}`",
             f"- **timestamp**: {manifest.get('timestamp')}",
             f"- **seed**: {manifest.get('seed')}"]
    params = manifest.get("params") or {}
    if params:
        lines.append(f"- **params**: `{json.dumps(params, sort_keys=True)}`")
    dh = manifest.get("data_hashes") or {}
    if dh:
        lines.append("- **data hashes**: "
                     + ", ".join(f"{k}=`{v}`" for k, v in dh.items()))
    versions = manifest.get("versions") or {}
    shown = {k: v for k, v in versions.items() if v != "not installed"}
    if shown:
        lines.append("- **versions**: "
                     + ", ".join(f"{k} {v}" for k, v in shown.items()))
    env = manifest.get("environment") or {}
    if env:
        lines.append(f"- **python**: {env.get('python')} "
                     f"({env.get('platform')})")
    return "\n".join(lines)
