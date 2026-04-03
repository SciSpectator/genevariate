"""
GeneVariate - Gene Expression Variability Analysis Platform

A comprehensive Python-based platform for gene expression variability analysis
and biological metadata extraction using AI-powered label extraction with Ollama.
"""

__version__ = '2.0.0'
__author__ = 'GeneVariate Development Team'

from .gui.app import GeoWorkflowGUI
from .config import CONFIG

__all__ = ['GeoWorkflowGUI', 'CONFIG']
