"""GeneVariate data sources (multi-technology ingestion)."""

from .base import BaseSource, CANONICAL_META_COLS

try:
    from .archs4 import Archs4Source
except Exception:
    Archs4Source = None

__all__ = ["BaseSource", "CANONICAL_META_COLS", "Archs4Source"]
