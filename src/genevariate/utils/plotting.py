"""
Plotting utilities for GeneVariate.
Provides helper functions for creating consistent, publication-quality plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis


class Plotter:
    """Utility class for plotting operations."""
    
    @staticmethod
    def get_optimal_bins(data, method='auto'):
        """
        Calculate optimal number of bins for histogram.
        
        Args:
            data: Array-like data
            method: 'auto', 'sturges', 'scott', or 'fd'
            
        Returns:
            Integer number of bins
        """
        data = np.array(data)
        n = len(data)
        
        if method == 'sturges':
            bins = int(np.ceil(np.log2(n) + 1))
        elif method == 'scott':
            h = 3.5 * np.std(data) / (n ** (1/3))
            bins = int(np.ceil((data.max() - data.min()) / h)) if h > 0 else 30
        elif method == 'fd':
            q75, q25 = np.percentile(data, [75, 25])
            iqr = q75 - q25
            h = 2 * iqr / (n ** (1/3))
            bins = int(np.ceil((data.max() - data.min()) / h)) if h > 0 else 30
        else:  # auto
            bins = min(int(np.sqrt(n)), 100)
        
        # Ensure reasonable bounds
        bins = max(10, min(bins, 100))
        
        return bins
    
    @staticmethod
    def get_distinct_colors(n):
        """
        Generate n visually distinct colors using golden ratio.
        
        Args:
            n: Number of colors to generate
            
        Returns:
            List of hex color strings
        """
        import colorsys
        
        golden_ratio = 1.618033988749895
        colors = []
        
        for i in range(n):
            hue = (i * golden_ratio) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(rgb[0] * 255),
                int(rgb[1] * 255),
                int(rgb[2] * 255)
            )
            colors.append(hex_color)
        
        return colors
    
    @staticmethod
    def add_significance_stars(ax, x1, x2, y, p_value, height=0.05):
        """
        Add significance stars to plot.
        
        Args:
            ax: Matplotlib axes
            x1, x2: X positions for comparison
            y: Y position for stars
            p_value: P-value
            height: Height of connector line
        """
        if p_value < 0.001:
            stars = '***'
        elif p_value < 0.01:
            stars = '**'
        elif p_value < 0.05:
            stars = '*'
        else:
            stars = 'ns'
        
        y_max = ax.get_ylim()[1]
        bar_height = y_max * height
        
        ax.plot([x1, x1, x2, x2], [y, y + bar_height, y + bar_height, y], 
                lw=1.5, c='black')
        ax.text((x1 + x2) / 2, y + bar_height, stars, 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    @staticmethod
    def style_axis(ax, xlabel=None, ylabel=None, title=None, grid=True):
        """
        Apply consistent styling to axis.
        
        Args:
            ax: Matplotlib axes
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Plot title
            grid: Whether to show grid
        """
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=10, fontweight='bold')
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
        if title:
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        
        if grid:
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        ax.tick_params(labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
