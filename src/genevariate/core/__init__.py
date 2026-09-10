"""
Core analysis and extraction modules for GeneVariate.

Nothing heavy is imported here. The package used to re-export a local-ollama
sample classifier (``nlp.classify_sample``), a MemGPT-style ``GSEContext`` and
two scipy helper classes, none of which any caller reached - but importing the
package pulled all of them in, which made the ``ollama`` client a hard import
dependency of every ``genevariate.core.*`` module. Extraction goes through
:mod:`genevariate.core.geo_extract_driver`; import submodules directly.
"""

from .gpl_downloader import GPLDownloader

__all__ = [
    'GPLDownloader',
]
