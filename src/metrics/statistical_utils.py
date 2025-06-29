"""
Statistical utilities for search quality analysis.

This module provides functions for statistical analysis of search quality metrics,
including hypothesis testing, confidence intervals, and power analysis for A/B testing.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Dict, List, Tuple, Union, Optional

def calculate_confidence_interval(data: np.ndarray, 
                                  confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate the confidence interval for a sample of data.
    
    Args:
        data: Sample data
        confidence: Confidence level (default: 0.95 for 95% confidence)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(data) < 2:
        return (np.nan, np.nan)
    
    mean = np.mean(data)
    std_error = stats.sem(data)
    
    # Calculate the margin of error
    margin = std_error * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    
    return (mean - margin, mean + margin)

def calculate_sample_size(baseline_rate: float,
                          minimum_detectable_effect: float,
                          power: float = 0.8,
                          significance: float = 0.05,
                          two_sided: bool = True) -> int:
    """
    Calculate the required sample size for an A/B test.
    
    Args:
        baseline_rate: The baseline conversion rate (between 0 and 1)
        minimum_detectable_effect: The smallest effect size you want to be able to detect
        power: Statistical power (default: 0.8)
        significance: Significance level (default: 0.05)
        two_sided: Whether to use a two-sided test (default: True)
        
    Returns:
        Required sample size per variant
    """
    # Calculate the effect size
    effect_size = minimum_detectable_effect / np.sqrt(baseline_rate * (1 - baseline_rate))
    
    # Calculate the z-scores
    z_alpha = stats.norm.ppf(1 - significance/2) if two_sided else stats.norm.ppf(1 - significance)
    z_beta = stats.norm.ppf(power)
    
    # Calculate the sample size
    n = ((z_alpha + z_beta) / effect_size) ** 2
    
    # Return the ceiling of the result
    return int(np.ceil(n))

def estimate_test_duration(sample_size: int, 
                           daily_traffic: int,
                           traffic_allocation: float = 0.5) -> float:
    """
    Estimate the duration of an A/B test in days.
    
    Args:
        sample_size: Required sample size per variant
        daily_traffic: Daily number of users/sessions
        traffic_allocation: Proportion of traffic allocated to the test (default: 0.5)
        
    Returns:
        Estimated test duration in days
    """
    # Total sample size needed across both variants
    total_sample_size = sample_size * 2
    
    # Daily sample collected
    daily_sample = daily_traffic * traffic_allocation
    
    # Calculate duration
    duration = total_sample_size / daily_sample
    
    return duration

def run_ab_test_analysis(control_data: np.ndarray,
                        experiment_data: np.ndarray,
                        metric_name: str,
                        alpha: float = 0.05) -> Dict[str, Union[float, str, bool]]:
    """
    Perform statistical analysis for an A/B test.
    
    Args:
        control_data: Data from the control group
        experiment_data: Data from the experiment group
        metric_name: Name of the metric being analyzed
        alpha: Significance level (default: 0.05)
        
    Returns:
        Dictionary with test results
    """
    results = {
        'metric': metric_name,
        'control_mean': np.mean(control_data),
        'experiment_mean': np.mean(experiment_data),
        'absolute_difference': np.mean(experiment_data) - np.mean(control_data),
        'relative_difference_percent': (np.mean(experiment_data) - np.mean(control_data)) / np.mean(control_data) * 100 if np.mean(control_data) != 0 else np.nan,
        'control_sample_size': len(control_data),
        'experiment_sample_size': len(experiment_data),
        'p_value': None,
        'significant': False,
        'confidence_interval': None,
        'test_type': None
    }
    
    # Check if we have enough data
    if len(control_data) < 2 or len(experiment_data) < 2:
        results['error'] = 'Insufficient data for statistical analysis'
        return results
    
    # Determine appropriate test based on data characteristics
    # Check for normality using Shapiro-Wilk test
    try:
        control_normal = stats.shapiro(control_data)[1] >= 0.05
        experiment_normal = stats.shapiro(experiment_data)[1] >= 0.05
    except:
        # If Shapiro-Wilk fails (e.g., due to sample size), assume non-normal
        control_normal = False
        experiment_normal = False
    
    # Check for equal variances
    try:
        equal_var = stats.levene(control_data, experiment_data)[1] >= 0.05
    except:
        equal_var = False
    
    # Run appropriate test
    if control_normal and experiment_normal:
        # If both groups are approximately normal
        if equal_var:
            # Equal variances: Student's t-test
            t_stat, p_value = stats.ttest_ind(control_data, experiment_data, equal_var=True)
            results['test_type'] = "Student's t-test"
        else:
            # Unequal variances: Welch's t-test
            t_stat, p_value = stats.ttest_ind(control_data, experiment_data, equal_var=False)
            results['test_type'] = "Welch's t-test"
    else:
        # Non-normal data: Mann-Whitney U test
        u_stat, p_value = stats.mannwhitneyu(control_data, experiment_data, alternative='two-sided')
        results['test_type'] = "Mann-Whitney U test"
    
    results['p_value'] = p_value
    results['significant'] = p_value < alpha
    
    # Calculate confidence interval for the difference in means
    # Use bootstrap method for robustness
    n_bootstrap = 10000
    bootstrap_diffs = []
    
    for _ in range(n_bootstrap):
        control_sample = np.random.choice(control_data, size=len(control_data), replace=True)
        experiment_sample = np.random.choice(experiment_data, size=len(experiment_data), replace=True)
        bootstrap_diffs.append(np.mean(experiment_sample) - np.mean(control_sample))
    
    # Calculate percentile confidence interval
    lower_bound = np.percentile(bootstrap_diffs, (alpha/2) * 100)
    upper_bound = np.percentile(bootstrap_diffs, (1 - alpha/2) * 100)
    results['confidence_interval'] = (lower_bound, upper_bound)
    
    return results

def analyze_experiment_results(baseline_df: pd.DataFrame, 
                              experiment_df: pd.DataFrame,
                              metrics: List[str]) -> pd.DataFrame:
    """
    Analyze A/B test results for multiple metrics.
    
    Args:
        baseline_df: DataFrame containing baseline (control) data
        experiment_df: DataFrame containing experiment data
        metrics: List of metric columns to analyze
        
    Returns:
        DataFrame with test results for each metric
    """
    results = []
    
    for metric in metrics:
        # Extract data for this metric
        control_data = baseline_df[metric].dropna().values
        experiment_data = experiment_df[metric].dropna().values
        
        # Run statistical analysis
        test_results = run_ab_test_analysis(
            control_data=control_data,
            experiment_data=experiment_data,
            metric_name=metric
        )
        
        results.append(test_results)
    
    return pd.DataFrame(results)

def calculate_confidence_intervals_for_timeseries(df: pd.DataFrame, 
                                                 metric_col: str,
                                                 date_col: str = 'date',
                                                 confidence: float = 0.95) -> pd.DataFrame:
    """
    Calculate confidence intervals for a metric over time.
    
    Args:
        df: DataFrame containing the data
        metric_col: Column containing the metric values
        date_col: Column containing the dates
        confidence: Confidence level (default: 0.95)
        
    Returns:
        DataFrame with confidence intervals for each date
    """
    # Group by date
    grouped = df.groupby(date_col)
    
    results = []
    for date, group in grouped:
        # Extract metric values
        values = group[metric_col].dropna().values
        
        # Calculate statistics
        mean_value = np.mean(values) if len(values) > 0 else np.nan
        
        # Calculate confidence interval
        if len(values) >= 2:
            lower, upper = calculate_confidence_interval(values, confidence)
        else:
            lower, upper = np.nan, np.nan
        
        results.append({
            date_col: date,
            f'{metric_col}_mean': mean_value,
            f'{metric_col}_lower': lower,
            f'{metric_col}_upper': upper,
            f'{metric_col}_sample_size': len(values)
        })
    
    return pd.DataFrame(results)

def detect_anomalies(df: pd.DataFrame, 
                    metric_col: str,
                    date_col: str = 'date',
                    window: int = 7,
                    threshold: float = 3.0) -> pd.DataFrame:
    """
    Detect anomalies in time series data using a rolling Z-score method.
    
    Args:
        df: DataFrame containing the data
        metric_col: Column containing the metric values
        date_col: Column containing the dates
        window: Rolling window size for baseline calculation
        threshold: Z-score threshold for anomaly detection
        
    Returns:
        DataFrame with anomaly flags
    """
    # Ensure the dataframe is sorted by date
    df = df.sort_values(date_col).copy()
    
    # Calculate rolling mean and standard deviation
    df[f'{metric_col}_rolling_mean'] = df[metric_col].rolling(window=window, min_periods=3).mean()
    df[f'{metric_col}_rolling_std'] = df[metric_col].rolling(window=window, min_periods=3).std()
    
    # Calculate Z-scores
    df[f'{metric_col}_zscore'] = (df[metric_col] - df[f'{metric_col}_rolling_mean']) / df[f'{metric_col}_rolling_std']
    
    # Flag anomalies
    df[f'{metric_col}_is_anomaly'] = abs(df[f'{metric_col}_zscore']) > threshold
    
    return df

def calculate_mde(baseline_rate: float,
                 sample_size: int,
                 power: float = 0.8,
                 significance: float = 0.05,
                 two_sided: bool = True) -> float:
    """
    Calculate the Minimum Detectable Effect (MDE) for a given sample size.
    
    Args:
        baseline_rate: The baseline conversion rate (between 0 and 1)
        sample_size: Sample size per variant
        power: Statistical power (default: 0.8)
        significance: Significance level (default: 0.05)
        two_sided: Whether to use a two-sided test (default: True)
        
    Returns:
        Minimum detectable effect size as an absolute change
    """
    # Calculate the z-scores
    z_alpha = stats.norm.ppf(1 - significance/2) if two_sided else stats.norm.ppf(1 - significance)
    z_beta = stats.norm.ppf(power)
    
    # Calculate the standard error
    se = np.sqrt(baseline_rate * (1 - baseline_rate) * (2 / sample_size))
    
    # Calculate the MDE
    mde = (z_alpha + z_beta) * se
    
    return mde

def segment_impact_analysis(df: pd.DataFrame,
                           segment_col: str,
                           metric_col: str,
                           experiment_col: str = 'variant',
                           control_value: str = 'control',
                           experiment_value: str = 'experiment') -> pd.DataFrame:
    """
    Analyze the impact of an experiment across different segments.
    
    Args:
        df: DataFrame containing the data
        segment_col: Column defining the segments to analyze
        metric_col: Column containing the metric values
        experiment_col: Column indicating the experiment variant
        control_value: Value in experiment_col that indicates the control group
        experiment_value: Value in experiment_col that indicates the experiment group
        
    Returns:
        DataFrame with segment-level analysis
    """
    segments = df[segment_col].unique()
    results = []
    
    for segment in segments:
        # Filter data for this segment
        segment_data = df[df[segment_col] == segment]
        
        # Split into control and experiment
        control_data = segment_data[segment_data[experiment_col] == control_value][metric_col].dropna().values
        experiment_data = segment_data[segment_data[experiment_col] == experiment_value][metric_col].dropna().values
        
        # Skip segments with insufficient data
        if len(control_data) < 10 or len(experiment_data) < 10:
            continue
        
        # Run analysis for this segment
        segment_results = run_ab_test_analysis(
            control_data=control_data,
            experiment_data=experiment_data,
            metric_name=f"{metric_col} for {segment_col}={segment}"
        )
        
        # Add segment information
        segment_results['segment'] = segment
        segment_results['segment_size'] = len(segment_data)
        segment_results['segment_size_percent'] = len(segment_data) / len(df) * 100
        
        results.append(segment_results)
    
    return pd.DataFrame(results) if results else pd.DataFrame()
