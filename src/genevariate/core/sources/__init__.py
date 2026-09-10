"""
GeneVariate data sources (multi-technology ingestion).

Every source returns a DataFrame in the GeneVariate canonical format:
    GSM | series_id | GENE1 | GENE2 | ...
(one row per sample, NaN values preserved)

so that downstream tools (distribution explorer, cross-platform analysis,
classification, enrichment) can consume microarray, bulk RNA-seq and scRNA-seq
pseudobulk matrices identically. Those are expression measurements; methylation
betas and peak coverage are not, and are refused at ingestion rather than
loaded into statistics written for expression.
"""

from .base import BaseSource, CANONICAL_META_COLS

try:
    from .archs4 import Archs4Source
except Exception:
    Archs4Source = None

__all__ = ["BaseSource", "CANONICAL_META_COLS", "Archs4Source"]
