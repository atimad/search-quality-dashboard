"""
Statistical Utilities for Search Quality Metrics Dashboard.

This module provides statistical analysis functions for search metrics:
- Confidence interval calculation
- A/B test significance testing
- Trend detection
- Anomaly detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

# Set up logging
logger = logging.getLogger(__name__)

class StatisticalUtils:
    """Statistical utilities for search metrics analysis."""
    
    @staticmethod
    def calculate_binomial_ci(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for a binomial proportion (e.g., CTR, conversion rate).
        
        Args:
            successes: Number of successes (e.g., clicks)
            trials: Number of trials (e.g., impressions)
            confidence: Confidence level (default: 0.95 for 95% CI)
            
        Returns:
            Tuple of (lower bound, upper bound)
        """
        if trials == 0:
            return (0.0, 0.0)
        
        proportion = successes / trials
        
        # Calculate CI using the Wilson score interval (better than normal approximation for small samples)
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        denominator = 1 + z**2 / trials
        
        center = (proportion + z**2 / (2 * trials)) / denominator
        halfwidth = z * np.sqrt(proportion * (1 - proportion) / trials + z**2 / (4 * trials**2)) / denominator
        
        lower = max(0.0, center - halfwidth)
        upper = min(1.0, center + halfwidth)
        
        return (lower, upper)
    
    @staticmethod
    def calculate_continuous_ci(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for a continuous metric (e.g., time to click).
        
        Args:
            values: List of metric values
            confidence: Confidence level (default: 0.95 for 95% CI)
            
        Returns:
            Tuple of (lower bound, upper bound)
        """
        if not values or len(values) < 2:
            return (None, None)
        
        values = np.array(values)
        values = values[~np.isnan(values)]  # Remove NaN values
        
        if len(values) < 2:
            return (None, None)
        
        mean = np.mean(values)
        sem = stats.sem(values)
        ci = stats.t.interval(confidence, len(values) - 1, loc=mean, scale=sem)
        
        return ci
    
    @staticmethod
    def binomial_significance_test(
        control_successes: int, 
        control_trials: int,
        treatment_successes: int,
        treatment_trials: int,
        confidence: float = 0.95
    ) -> Dict[str, Any]:
        """
        Perform significance test for binomial metrics (e.g., CTR, conversion rate).
        
        Args:
            control_successes: Number of successes in control group
            control_trials: Number of trials in control group
            treatment_successes: Number of successes in treatment group
            treatment_trials: Number of trials in treatment group
            confidence: Confidence level (default: 0.95 for 95% CI)
            
        Returns:
            Dictionary with test results
        """
        if control_trials == 0 or treatment_trials == 0:
            return {
                'p_value': None,
                'significant': False,
                'relative_difference': None,
                'control_rate': 0 if control_trials == 0 else control_successes / control_trials,
                'treatment_rate': 0 if treatment_trials == 0 else treatment_successes / treatment_trials
            }
        
        # Calculate proportions
        control_prop = control_successes / control_trials
        treatment_prop = treatment_successes / treatment_trials
        
        # Calculate relative difference
        if control_prop == 0:
            relative_diff = np.inf if treatment_prop > 0 else 0
        else:
            relative_diff = (treatment_prop - control_prop) / control_prop
        
        # Perform Fisher's exact test
        table = np.array([
            [control_successes, control_trials - control_successes],
            [treatment_successes, treatment_trials - treatment_successes]
        ])
        
        _, p_value = stats.fisher_exact(table)
        
        # Determine significance
        significant = p_value < (1 - confidence)
        
        return {
            'p_value': p_value,
            'significant': significant,
            'relative_difference': relative_diff,
            'control_rate': control_prop,
            'treatment_rate': treatment_prop,
            'absolute_difference': treatment_prop - control_prop
        }
    
    @staticmethod
    def continuous_significance_test(
        control_values: List[float],
        treatment_values: List[float],
        confidence: float = 0.95
    ) -> Dict[str, Any]:
        """
        Perform significance test for continuous metrics (e.g., time to click).
        
        Args:
            control_values: List of metric values for control group
            treatment_values: List of metric values for treatment group
            confidence: Confidence level (default: 0.95 for 95% CI)
            
        Returns:
            Dictionary with test results
        """
        # Remove NaN values
        control_values = np.array(control_values)
        treatment_values = np.array(treatment_values)
        
        control_values = control_values[~np.isnan(control_values)]
        treatment_values = treatment_values[~np.isnan(treatment_values)]
        
        if len(control_values) < 2 or len(treatment_values) < 2:
            return {
                'p_value': None,
                'significant': False,
                'relative_difference': None,
                'control_mean': np.mean(control_values) if len(control_values) > 0 else None,
                'treatment_mean': np.mean(treatment_values) if len(treatment_values) > 0 else None
            }
        
        # Calculate means
        control_mean = np.mean(control_values)
        treatment_mean = np.mean(treatment_values)
        
        # Calculate relative difference
        if control_mean == 0:
            relative_diff = np.inf if treatment_mean > 0 else 0
        else:
            relative_diff = (treatment_mean - control_mean) / control_mean
        
        # Perform Mann-Whitney U test (non-parametric, doesn't assume normal distribution)
        try:
            _, p_value = stats.mannwhitneyu(control_values, treatment_values, alternative='two-sided')
        except ValueError:
            # Fall back to t-test if Mann-Whitney fails
            _, p_value = stats.ttest_ind(control_values, treatment_values, equal_var=False)
        
        # Determine significance
        significant = p_value < (1 - confidence)
        
        return {
            'p_value': p_value,
            'significant': significant,
            'relative_difference': relative_diff,
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'absolute_difference': treatment_mean - control_mean
        }
    
    @staticmethod
    def detect_trend(
        time_series: pd.Series,
        window: int = 7
    ) -> Dict[str, Any]:
        """
        Detect trend in a time series.
        
        Args:
            time_series: Time series data (pandas Series with datetime index)
            window: Rolling window size for trend smoothing
            
        Returns:
            Dictionary with trend analysis results
        """
        if len(time_series) < 2 * window:
            return {
                'has_trend': False,
                'trend_direction': None,
                'trend_magnitude': None,
                'p_value': None,
                'decomposition': None
            }
        
        # Create a copy of the time series with a complete datetime index
        ts = time_series.copy()
        
        # Handle missing values
        ts = ts.fillna(method='ffill').fillna(method='bfill')
        
        # Perform time series decomposition
        try:
            decomposition = seasonal_decompose(ts, model='additive', period=min(window, len(ts) // 2))
            trend = decomposition.trend
            trend = trend.dropna()
        except:
            # If decomposition fails, use simple moving average
            trend = ts.rolling(window=window, center=True).mean()
            trend = trend.dropna()
            decomposition = None
        
        if len(trend) < 2:
            return {
                'has_trend': False,
                'trend_direction': None,
                'trend_magnitude': None,
                'p_value': None,
                'decomposition': None
            }
        
        # Perform linear regression on the trend
        X = np.arange(len(trend)).reshape(-1, 1)
        y = trend.values
        
        model = sm.OLS(y, sm.add_constant(X)).fit()
        
        # Extract slope and p-value
        slope = model.params[1]
        p_value = model.pvalues[1]
        
        # Determine trend direction and significance
        has_trend = p_value < 0.05
        trend_direction = 'up' if slope > 0 else 'down' if slope < 0 else 'flat'
        
        # Normalize trend magnitude as percentage change over the period
        if np.mean(y) == 0:
            trend_magnitude = 0
        else:
            trend_magnitude = (slope * len(trend)) / np.mean(y)
        
        return {
            'has_trend': has_trend,
            'trend_direction': trend_direction,
            'trend_magnitude': trend_magnitude,
            'p_value': p_value,
            'slope': slope,
            'decomposition': decomposition
        }
    
    @staticmethod
    def detect_anomalies(
        time_series: pd.Series,
        window: int = 14,
        sigma: float = 3.0
    ) -> Dict[str, Any]:
        """
        Detect anomalies in a time series using a moving window approach.
        
        Args:
            time_series: Time series data (pandas Series with datetime index)
            window: Window size for moving statistics
            sigma: Number of standard deviations to consider as anomaly threshold
            
        Returns:
            Dictionary with anomaly detection results
        """
        if len(time_series) < window + 1:
            return {
                'has_anomalies': False,
                'anomalies': pd.Series(dtype=bool),
                'anomaly_indices': []
            }
        
        # Create a copy and handle missing values
        ts = time_series.copy()
        ts = ts.fillna(method='ffill').fillna(method='bfill')
        
        # Calculate rolling mean and standard deviation
        rolling_mean = ts.rolling(window=window, min_periods=window//2).mean()
        rolling_std = ts.rolling(window=window, min_periods=window//2).std()
        
        # Define thresholds
        upper_threshold = rolling_mean + sigma * rolling_std
        lower_threshold = rolling_mean - sigma * rolling_std
        
        # Detect anomalies
        anomalies = (ts > upper_threshold) | (ts < lower_threshold)
        
        # Get anomaly indices and values
        anomaly_indices = anomalies[anomalies].index.tolist()
        anomaly_values = ts[anomalies].tolist()
        
        # Calculate deviation from expected value
        if anomaly_indices:
            deviations = [(ts[idx] - rolling_mean[idx]) / rolling_std[idx] if rolling_std[idx] > 0 else np.inf 
                          for idx in anomaly_indices]
        else:
            deviations = []
        
        return {
            'has_anomalies': len(anomaly_indices) > 0,
            'anomalies': anomalies,
            'anomaly_indices': anomaly_indices,
            'anomaly_values': anomaly_values,
            'deviations': deviations,
            'upper_threshold': upper_threshold,
            'lower_threshold': lower_threshold,
            'rolling_mean': rolling_mean
        }
    
    @staticmethod
    def compare_distributions(
        dist1: List[float],
        dist2: List[float],
        bins: int = 10
    ) -> Dict[str, Any]:
        """
        Compare two distributions using statistical tests.
        
        Args:
            dist1: First distribution
            dist2: Second distribution
            bins: Number of bins for histogram
            
        Returns:
            Dictionary with distribution comparison results
        """
        # Remove NaN values
        dist1 = np.array(dist1)
        dist2 = np.array(dist2)
        
        dist1 = dist1[~np.isnan(dist1)]
        dist2 = dist2[~np.isnan(dist2)]
        
        if len(dist1) < 2 or len(dist2) < 2:
            return {
                'statistically_different': None,
                'ks_statistic': None,
                'p_value': None,
                'effect_size': None
            }
        
        # Perform Kolmogorov-Smirnov test
        ks_statistic, p_value = stats.ks_2samp(dist1, dist2)
        
        # Calculate effect size (Cohen's d)
        mean1, mean2 = np.mean(dist1), np.mean(dist2)
        std1, std2 = np.std(dist1), np.std(dist2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((len(dist1) - 1) * std1**2 + (len(dist2) - 1) * std2**2) / 
                             (len(dist1) + len(dist2) - 2))
        
        if pooled_std == 0:
            effect_size = 0
        else:
            effect_size = abs(mean1 - mean2) / pooled_std
        
        # Create histograms
        hist1, edges1 = np.histogram(dist1, bins=bins, density=True)
        hist2, edges2 = np.histogram(dist2, bins=bins, density=True)
        
        return {
            'statistically_different': p_value < 0.05,
            'ks_statistic': ks_statistic,
            'p_value': p_value,
            'effect_size': effect_size,
            'effect_size_interpretation': interpret_effect_size(effect_size),
            'hist1': hist1,
            'edges1': edges1,
            'hist2': hist2,
            'edges2': edges2,
            'mean1': mean1,
            'mean2': mean2,
            'std1': std1,
            'std2': std2
        }
    
    @staticmethod
    def power_analysis(
        baseline_rate: float,
        mde: float,
        confidence: float = 0.95,
        power: float = 0.8
    ) -> Dict[str, Any]:
        """
        Calculate sample size needed for an A/B test.
        
        Args:
            baseline_rate: Baseline conversion rate (e.g., CTR)
            mde: Minimum Detectable Effect (relative change, e.g., 0.1 for 10% lift)
            confidence: Confidence level (default: 0.95 for 95% confidence)
            power: Statistical power (default: 0.8 for 80% power)
            
        Returns:
            Dictionary with power analysis results
        """
        if baseline_rate <= 0 or baseline_rate >= 1:
            return {
                'sample_size_per_variant': None,
                'total_sample_size': None
            }
        
        # Calculate alpha and beta
        alpha = 1 - confidence
        beta = 1 - power
        
        # Calculate critical values
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(1 - beta)
        
        # Calculate expected rate in treatment
        treatment_rate = baseline_rate * (1 + mde)
        
        # Calculate pooled standard error
        pooled_var = baseline_rate * (1 - baseline_rate) + treatment_rate * (1 - treatment_rate)
        
        # Calculate sample size per variant
        sample_size = (z_alpha + z_beta)**2 * pooled_var / (baseline_rate - treatment_rate)**2
        
        # Round up to nearest integer
        sample_size = int(np.ceil(sample_size))
        
        return {
            'sample_size_per_variant': sample_size,
            'total_sample_size': 2 * sample_size,
            'baseline_rate': baseline_rate,
            'treatment_rate': treatment_rate,
            'mde': mde,
            'mde_absolute': baseline_rate * mde,
            'confidence': confidence,
            'power': power
        }

def interpret_effect_size(effect_size: float) -> str:
    """
    Interpret Cohen's d effect size.
    
    Args:
        effect_size: Effect size value
        
    Returns:
        String interpretation of effect size
    """
    if effect_size < 0.2:
        return 'negligible'
    elif effect_size < 0.5:
        return 'small'
    elif effect_size < 0.8:
        return 'medium'
    else:
        return 'large'
