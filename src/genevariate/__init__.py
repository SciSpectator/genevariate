"""GeneVariate - Gene Expression Variability Analysis Platform"""

__version__ = '1.0.0'
__author__ = 'GeneVariate Development Team'

from .config import CONFIG

__all__ = ['GeoWorkflowGUI', 'CONFIG']


def __getattr__(name):
    """Lazy attribute access - only imports the GUI when explicitly requested."""
    if name == 'GeoWorkflowGUI':
        from .gui.app import GeoWorkflowGUI
        return GeoWorkflowGUI
    raise AttributeError(f"module 'genevariate' has no attribute {name!r}")
