"""Vendored LLM-GEO-Label-Extractor pipeline (SciSpectator/LLM-GEO-Label-Extractor).

Four-phase GEO sample metadata extraction:
    Phase 1   verbatim per-label LLM extraction (Tissue, Condition, Treatment)
    Phase 1b  GSE-context inference for "Not Specified" outputs
    Phase 1c  deterministic consensus + optional BioLORD semantic curator
    Phase 2   multi-agent MeSH-cascade canonicalisation

The upstream source is checked in under this package. Files use bare imports
(e.g. ``from phase1 import ...``), so this ``__init__`` inserts the package
directory into ``sys.path`` to keep those imports working when invoked as a
genevariate sub-module.

Entry point: :func:`genevariate.core.llm_extractor.cli.main`.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

__all__ = ["cli"]
