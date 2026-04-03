"""
Statistical analysis utilities for gene expression data.
"""

import numpy as np
from scipy.stats import ranksums, mannwhitneyu, ttest_ind, wasserstein_distance


class BioStats:
    """Statistical analysis utilities."""
    
    @staticmethod
    def compare_distributions(data1, data2, method='wilcoxon'):
        """
        Compare two distributions statistically.
        
        Args:
            data1, data2: Array-like data
            method: 'wilcoxon', 'ttest', or 'wasserstein'
            
        Returns:
            Tuple of (statistic, p_value)
        """
        data1 = np.array(data1).astype(float)
        data2 = np.array(data2).astype(float)
        
        data1 = data1[~np.isnan(data1)]
        data2 = data2[~np.isnan(data2)]
        
        if len(data1) < 3 or len(data2) < 3:
            return (np.nan, np.nan)
        
        try:
            if method == 'wilcoxon':
                stat, pval = ranksums(data1, data2)
            elif method == 'ttest':
                stat, pval = ttest_ind(data1, data2)
            elif method == 'wasserstein':
                stat = wasserstein_distance(data1, data2)
                pval = np.nan  # Wasserstein doesn't provide p-value
            else:
                stat, pval = ranksums(data1, data2)
            
            return (stat, pval)
            
        except Exception as e:
            return (np.nan, np.nan)
    
    @staticmethod
    def calculate_effect_size(data1, data2, method='cohens_d'):
        """
        Calculate effect size between two groups.
        
        Args:
            data1, data2: Array-like data
            method: 'cohens_d' or 'cliff_delta'
            
        Returns:
            Effect size value
        """
        data1 = np.array(data1).astype(float)
        data2 = np.array(data2).astype(float)
        
        data1 = data1[~np.isnan(data1)]
        data2 = data2[~np.isnan(data2)]
        
        if len(data1) < 2 or len(data2) < 2:
            return np.nan
        
        if method == 'cohens_d':
            mean1, mean2 = np.mean(data1), np.mean(data2)
            std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
            n1, n2 = len(data1), len(data2)
            
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
            
            if pooled_std == 0:
                return np.nan
            
            return (mean1 - mean2) / pooled_std
        
        elif method == 'cliff_delta':
            matrix = np.sign(data1[:, None] - data2[None, :])
            return np.mean(matrix)
        
        return np.nan
    
    @staticmethod
    def bootstrap_confidence_interval(data, statistic=np.mean, n_bootstrap=1000, confidence=0.95):
        """
        Calculate bootstrap confidence interval.
        
        Args:
            data: Array-like data
            statistic: Function to calculate (default: mean)
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level (0-1)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        data = np.array(data).astype(float)
        data = data[~np.isnan(data)]
        
        if len(data) < 2:
            return (np.nan, np.nan)
        
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stats.append(statistic(sample))
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
        upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
        
        return (lower, upper)
