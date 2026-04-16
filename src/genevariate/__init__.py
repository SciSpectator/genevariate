"""
GeneVariate - Gene Expression Variability Analysis Platform

A comprehensive Python-based platform for gene expression variability analysis
and biological metadata extraction using AI-powered label extraction with Ollama.

The GUI (tkinter/matplotlib/seaborn) is loaded lazily on first access so that
headless extraction workloads never pay the GUI import cost.
"""

__version__ = '2.1.0'
__author__ = 'GeneVariate Development Team'

from .config import CONFIG

__all__ = ['GeoWorkflowGUI', 'CONFIG']


def __getattr__(name):
    """Lazy attribute access — only imports the GUI when explicitly requested."""
    if name == 'GeoWorkflowGUI':
        from .gui.app import GeoWorkflowGUI
        return GeoWorkflowGUI
    raise AttributeError(f"module 'genevariate' has no attribute {name!r}")
