"""
AI Engine for gene expression analysis.
Provides distribution analysis and pattern recognition.

Uses scipy excess kurtosis convention:
    Normal distribution: excess kurtosis ~ 0
    Uniform distribution: excess kurtosis ~ -1.2
    Heavy-tailed (Cauchy): excess kurtosis >> 3
"""

import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks


class BioAI_Engine:
    """AI engine for biological data analysis."""

    @staticmethod
    def analyze_gene_distribution(expression_values):
        """
        Analyze gene expression distribution and classify its type.

        Classification cascade (ordered by specificity):
            1. Bimodal    - two or more peaks detected via KDE
            2. Heavy-tailed - excess kurtosis > 3 (long tails, extreme outliers)
            3. Uniform    - excess kurtosis < -1 and low skewness
            4. Normal     - low skewness and kurtosis near 0
            5. Lognormal  - right-skewed and all values >= 0
            6. Right-skewed - positive skewness > 1
            7. Left-skewed  - negative skewness < -1
            8. Mixed      - doesn't fit any clean category

        Args:
            expression_values: Array-like of expression values

        Returns:
            String describing distribution type
        """
        try:
            expr = np.array(expression_values).astype(float)
            expr = expr[~np.isnan(expr)]

            if len(expr) == 0:
                return "Insufficient Data"
            if len(expr) < 10:
                return "Insufficient Data"

            # Calculate statistics (scipy uses excess kurtosis: normal=0)
            skewness = skew(expr)
            kurt = kurtosis(expr)  # excess kurtosis

            # ── 1. Bimodal detection (KDE-based for gene expression) ──
            # Use adaptive bins based on sample size for better sensitivity
            n_bins = min(max(int(np.sqrt(len(expr))), 30), 100)
            hist, bin_edges = np.histogram(expr, bins=n_bins)

            # Smooth histogram to reduce noise before peak detection
            if len(hist) >= 5:
                kernel = np.array([1, 2, 3, 2, 1]) / 9.0
                hist_smooth = np.convolve(hist, kernel, mode='same')
            else:
                hist_smooth = hist.astype(float)

            # Require peaks to be at least 5% of max to be significant
            min_prominence = max(hist_smooth) * 0.05
            peaks, props = find_peaks(hist_smooth, prominence=min_prominence)

            if len(peaks) >= 2:
                # Verify the valley between peaks is deep enough
                # (at least 30% drop from the lower peak)
                peak_heights = hist_smooth[peaks]
                sorted_heights = sorted(peak_heights, reverse=True)
                if len(sorted_heights) >= 2:
                    lower_peak = sorted_heights[1]
                    # Find valley between the two highest peaks
                    p1, p2 = sorted(peaks[np.argsort(peak_heights)[-2:]])
                    valley_min = hist_smooth[p1:p2 + 1].min()
                    if valley_min < lower_peak * 0.7:
                        return "Bimodal"

            # ── 2. Lognormal (right-skewed, all positive) ──
            # Check before heavy-tailed since lognormal has high kurtosis
            if skewness > 0.5 and np.all(expr >= 0):
                return "Lognormal"

            # ── 3. Right-skewed (not all positive) ──
            if skewness > 1.0:
                return "Right-skewed"

            # ── 4. Left-skewed ──
            if skewness < -1.0:
                return "Left-skewed"

            # ── 5. Heavy-tailed (extreme kurtosis with moderate skewness) ──
            # Very high kurtosis (>6) overrides skewness since the heavy tails
            # dominate the distribution shape regardless of asymmetry
            if kurt > 6.0 or (kurt > 3.0 and abs(skewness) < 1.0):
                return "Heavy-tailed"

            # ── 6. Uniform (flat, negative excess kurtosis) ──
            if kurt < -1.0 and abs(skewness) < 0.5:
                return "Uniform"

            # ── 7. Normal (symmetric, kurtosis near 0) ──
            if abs(skewness) < 0.5 and -1.0 <= kurt <= 1.5:
                return "Normal"

            # ── 8. Fallback ──
            return "Mixed"

        except Exception:
            return "Unknown"

    @staticmethod
    def detect_outliers(data, method='zscore', threshold=3.0):
        """
        Detect outliers in data.

        Args:
            data: Array-like data
            method: 'zscore' or 'iqr'
            threshold: Z-score threshold or IQR multiplier

        Returns:
            Boolean array indicating outliers
        """
        data = np.array(data).astype(float)
        data = data[~np.isnan(data)]

        if len(data) < 3:
            return np.zeros(len(data), dtype=bool)

        if method == 'zscore':
            std = np.std(data)
            if std < 1e-10:
                return np.zeros(len(data), dtype=bool)
            z_scores = np.abs((data - np.mean(data)) / std)
            return z_scores > threshold

        elif method == 'iqr':
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            if iqr < 1e-10:
                return np.zeros(len(data), dtype=bool)
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            return (data < lower) | (data > upper)

        return np.zeros(len(data), dtype=bool)

    @staticmethod
    def suggest_transformation(expression_values):
        """
        Suggest appropriate transformation for expression data.

        Validates data domain before suggesting transformations that
        require positive values (log, sqrt, Box-Cox).

        Args:
            expression_values: Array-like of expression values

        Returns:
            String with transformation suggestion
        """
        expr = np.array(expression_values).astype(float)
        expr = expr[~np.isnan(expr)]

        if len(expr) < 10:
            return "None"

        skewness = skew(expr)
        has_negative = np.any(expr < 0)
        has_zero = np.any(expr == 0)
        all_positive = np.all(expr > 0)

        if abs(skewness) < 0.5:
            return "None (already approximately normal)"
        elif skewness > 1.0 and all_positive:
            return "Log transformation"
        elif skewness > 1.0 and not has_negative:
            return "Log(x+1) transformation"
        elif skewness > 0.5 and not has_negative:
            return "Square root transformation"
        elif skewness > 0.5 and has_negative:
            return "Yeo-Johnson transformation (handles negatives)"
        elif skewness < -1.0:
            return "Reflect and log transformation"
        elif skewness < -0.5:
            return "Square transformation or Yeo-Johnson"
        elif all_positive:
            return "Box-Cox transformation"
        else:
            return "Yeo-Johnson transformation"
