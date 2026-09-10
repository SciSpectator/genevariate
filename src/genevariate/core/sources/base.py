"""
Base class for all GeneVariate data sources.

Contract: every source's .fetch(...) returns a pandas.DataFrame with
columns (GSM, series_id, <gene columns...>) and one row per sample.
Gene columns are HGNC symbols where possible; NaN values are preserved.
"""

from __future__ import annotations

import abc
import os
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd


CANONICAL_META_COLS = ("GSM", "series_id")


@dataclass
class SourceInfo:
    name: str
    technology: str
    species: str = "human"
    description: str = ""


class BaseSource(abc.ABC):
    """
    Abstract base class for GeneVariate data sources.

    Subclasses implement `fetch(query, **kwargs)` which must return a tidy
    DataFrame in GeneVariate's canonical wide format.
    """

    info: SourceInfo = SourceInfo(name="base", technology="unknown")

    @abc.abstractmethod
    def fetch(self, query: str, **kwargs) -> pd.DataFrame:
        ...

    @staticmethod
    def to_canonical(expr: pd.DataFrame,
                     gsm_to_series: Optional[dict] = None) -> pd.DataFrame:
        """
        Transpose a genes x samples expression matrix into GeneVariate's
        canonical samples x genes layout with GSM + series_id metadata columns.
        """
        if expr.empty:
            return pd.DataFrame(columns=list(CANONICAL_META_COLS))

        t = expr.T.copy()
        gsms = [str(i).strip().upper() for i in t.index]
        series = [(gsm_to_series or {}).get(g, np.nan) for g in gsms]
        t.insert(0, "series_id", series)
        t.insert(0, "GSM", gsms)
        t.reset_index(drop=True, inplace=True)
        return t

    @staticmethod
    def save_csv(df: pd.DataFrame, out_dir: str, filename: str,
                 provenance: Optional[dict] = None) -> str:
        """
        Save a canonical DataFrame to .csv.gz. If `provenance` is given,
        write a sidecar .meta.json file that pins the source, version,
        and any extra fields needed to reproduce the analysis later.
        """
        import json
        from datetime import datetime, timezone

        os.makedirs(out_dir, exist_ok=True)
        if not filename.endswith(".csv.gz"):
            filename = filename + ".csv.gz"
        path = os.path.join(out_dir, filename)
        df.to_csv(path, index=False, compression="gzip")
        if provenance is not None:
            meta = {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "genevariate_version": _genevariate_version(),
                "shape": list(df.shape),
                "columns": list(df.columns)[:12] + (["..."] if df.shape[1] > 12 else []),
            }
            meta.update(provenance)
            with open(path.replace(".csv.gz", ".meta.json"), "w") as fh:
                json.dump(meta, fh, indent=2, default=str)
        return path


def _genevariate_version() -> str:
    try:
        from importlib.metadata import version
        return version("genevariate")
    except Exception:
        return "unknown"


ProgressCallback = Callable[[str, float], None]
