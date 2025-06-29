"""
Relevance metrics module for search quality analysis.

This module provides functions to calculate various relevance metrics for search queries,
including user satisfaction, ranking quality, and reformulation patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional

def calculate_ndcg(clicked_positions: List[int], 
                  click_satisfaction_scores: Optional[List[float]] = None,
                  k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG) for a single query.
    
    Args:
        clicked_positions: List of positions that were clicked (1-indexed)
        click_satisfaction_scores: Optional list of relevance scores for each click
        k: Number of positions to consider
        
    Returns:
        NDCG score from 0 to 1
    """
    if not clicked_positions:
        return 0.0
    
    # If no satisfaction scores provided, assume all clicks have relevance 1.0
    if click_satisfaction_scores is None:
        click_satisfaction_scores = [1.0] * len(clicked_positions)
    
    # Calculate DCG
    dcg = 0.0
    for pos, score in zip(clicked_positions, click_satisfaction_scores):
        if pos <= k:  # Only consider positions up to k
            # DCG formula: relevance / log2(position + 1)
            dcg += score / np.log2(pos + 1)
    
    # Calculate ideal DCG (IDCG)
    # Sort relevance scores in descending order for ideal ranking
    ideal_scores = sorted(click_satisfaction_scores, reverse=True)
    idcg = 0.0
    for i, score in enumerate(ideal_scores):
        if i < k:  # Only consider up to k positions
            idcg += score / np.log2(i + 2)  # i+2 because i is 0-indexed
    
    # Avoid division by zero
    if idcg == 0:
        return 0.0
        
    return dcg / idcg

def calculate_mrr(first_click_positions: List[int]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) for a set of queries.
    
    Args:
        first_click_positions: List of positions of the first click for each query (1-indexed)
        
    Returns:
        MRR score from 0 to 1
    """
    if not first_click_positions:
        return 0.0
    
    # Calculate reciprocal rank for each query
    reciprocal_ranks = [1.0/pos if pos > 0 else 0.0 for pos in first_click_positions]
    
    # Calculate mean
    return np.mean(reciprocal_ranks)

def calculate_query_success_rate(df: pd.DataFrame) -> float:
    """
    Calculate the query success rate - proportion of queries that result in at least one click.
    
    Args:
        df: DataFrame containing search session data with 'query_id' and 'clicked' columns
        
    Returns:
        Success rate from 0 to 1
    """
    # Group by query_id and check if there was at least one click
    query_success = df.groupby('query_id')['clicked'].any()
    
    # Calculate the proportion of successful queries
    return query_success.mean()

def calculate_reformulation_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate metrics related to query reformulations.
    
    Args:
        df: DataFrame containing search session data with 'session_id', 'query_id', 
            'query_text', 'clicked', and 'timestamp' columns
        
    Returns:
        Dictionary of reformulation metrics
    """
    # Group data by session
    sessions = df.sort_values(['session_id', 'timestamp']).groupby('session_id')
    
    # Calculate metrics
    results = {}
    
    # 1. Average number of queries per session
    results['avg_queries_per_session'] = sessions['query_id'].nunique().mean()
    
    # 2. Reformulation rate - proportion of sessions with more than one query
    results['reformulation_rate'] = (sessions['query_id'].nunique() > 1).mean()
    
    # 3. Abandonment rate - proportion of sessions where no query resulted in a click
    results['abandonment_rate'] = (~sessions['clicked'].any()).mean()
    
    # 4. Zero-click reformulation rate - proportion of queries followed by another query without a click
    # Create a helper function to analyze each session
    def calculate_zero_click_reformulations(group):
        # Check if query had clicks
        query_had_clicks = group.groupby('query_id')['clicked'].any()
        
        # Count queries without clicks followed by another query
        zero_click_count = 0
        total_queries = len(query_had_clicks)
        
        if total_queries <= 1:
            return 0
        
        # Iterate through all except the last query
        for i in range(total_queries-1):
            if not query_had_clicks.iloc[i]:
                zero_click_count += 1
                
        return zero_click_count / (total_queries - 1)  # Denominator excludes last query
    
    # Apply helper function to each session
    zero_click_rates = sessions.apply(calculate_zero_click_reformulations)
    results['zero_click_reformulation_rate'] = zero_click_rates.mean()
    
    return results

def calculate_satisfaction_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate a search satisfaction score for each query based on click behavior.
    
    Args:
        df: DataFrame containing search session data
        
    Returns:
        DataFrame with satisfaction scores for each query
    """
    # Group by query_id
    query_groups = df.groupby('query_id')
    
    # Collect metrics for satisfaction calculation
    satisfaction_data = []
    
    for query_id, group in query_groups:
        # Basic click metrics
        click_count = group['clicked'].sum()
        has_click = click_count > 0
        clicks_in_top_3 = group[group['position'] <= 3]['clicked'].sum()
        
        # Time metrics - dwell time is a good indicator of satisfaction
        avg_dwell_time = group[group['clicked']]['dwell_time'].mean() if has_click else 0
        
        # Advanced engagement metrics
        click_depth = group[group['clicked']]['position'].max() if has_click else 0
        last_result_clicked = group[group['clicked']]['position'].max() if has_click else 0
        time_to_first_click = group[group['clicked']]['time_to_click'].min() if has_click else float('inf')
        
        # Calculate a composite satisfaction score (0-100)
        # This is a simplified example - in production, you would want to validate and tune this
        base_score = 50 if has_click else 20  # Base score
        
        # Add points for clicks in top positions
        position_score = min(25, clicks_in_top_3 * 10)
        
        # Add points for longer dwell time (capped at 20 points)
        dwell_score = min(20, avg_dwell_time / 2) if not np.isnan(avg_dwell_time) else 0
        
        # Subtract points for high click depth or slow time to first click
        depth_penalty = min(10, max(0, (click_depth - 3) * 2)) if click_depth > 0 else 0
        time_penalty = min(10, max(0, (time_to_first_click - 2) * 2)) if time_to_first_click < float('inf') else 5
        
        # Final satisfaction score
        satisfaction_score = min(100, max(0, base_score + position_score + dwell_score - depth_penalty - time_penalty))
        
        # Store the results
        satisfaction_data.append({
            'query_id': query_id,
            'click_count': click_count,
            'clicks_in_top_3': clicks_in_top_3,
            'avg_dwell_time': avg_dwell_time,
            'click_depth': click_depth,
            'time_to_first_click': time_to_first_click if time_to_first_click < float('inf') else None,
            'satisfaction_score': satisfaction_score
        })
    
    return pd.DataFrame(satisfaction_data)

def analyze_ranking_quality(df: pd.DataFrame) -> Dict[str, float]:
    """
    Analyze the quality of search result rankings.
    
    Args:
        df: DataFrame containing search session data
        
    Returns:
        Dictionary of ranking quality metrics
    """
    # Prepare data
    # We need to know the click positions and possibly satisfaction for each query
    query_groups = df.groupby('query_id')
    
    # Collect data for NDCG and MRR calculation
    first_click_positions = []
    all_clicked_positions = []
    satisfaction_scores = []
    
    for query_id, group in query_groups:
        # Filter for clicked results
        clicked_results = group[group['clicked']].sort_values('timestamp')
        
        if not clicked_results.empty:
            # Record position of first click (for MRR)
            first_click_positions.append(clicked_results.iloc[0]['position'])
            
            # Record all clicked positions (for NDCG)
            positions = clicked_results['position'].tolist()
            all_clicked_positions.append(positions)
            
            # Use dwell_time as a proxy for satisfaction/relevance
            # Normalize to [0, 1] range with a cap at 60 seconds
            dwell_times = clicked_results['dwell_time'].fillna(0).tolist()
            normalized_satisfaction = [min(time / 60.0, 1.0) for time in dwell_times]
            satisfaction_scores.append(normalized_satisfaction)
        else:
            # No clicks for this query
            first_click_positions.append(0)  # 0 indicates no click
            all_clicked_positions.append([])
            satisfaction_scores.append([])
    
    # Calculate metrics
    results = {}
    
    # MRR calculation
    results['mrr'] = calculate_mrr(first_click_positions)
    
    # NDCG calculation - need to calculate for each query then average
    ndcg_scores = []
    for positions, scores in zip(all_clicked_positions, satisfaction_scores):
        if positions:  # Only calculate for queries with clicks
            ndcg_scores.append(calculate_ndcg(positions, scores))
    
    results['ndcg'] = np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    # Additional metrics
    results['successful_query_rate'] = calculate_query_success_rate(df)
    
    # Calculate average click position (lower is better)
    clicked_df = df[df['clicked']]
    results['avg_click_position'] = clicked_df['position'].mean() if not clicked_df.empty else 0
    
    # Calculate rate of clicks on first result
    results['first_position_ctr'] = df[df['position'] == 1]['clicked'].mean()
    
    return results

def compare_relevance_metrics(baseline_df: pd.DataFrame, 
                             experiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare relevance metrics between baseline and experiment data.
    Useful for A/B test analysis.
    
    Args:
        baseline_df: DataFrame containing baseline search data
        experiment_df: DataFrame containing experiment search data
        
    Returns:
        DataFrame with metric comparisons
    """
    # Calculate metrics for both datasets
    baseline_metrics = analyze_ranking_quality(baseline_df)
    experiment_metrics = analyze_ranking_quality(experiment_df)
    
    # Calculate reformulation metrics
    baseline_reformulation = calculate_reformulation_metrics(baseline_df)
    experiment_reformulation = calculate_reformulation_metrics(experiment_df)
    
    # Combine all metrics
    baseline_metrics.update(baseline_reformulation)
    experiment_metrics.update(experiment_reformulation)
    
    # Create comparison dataframe
    comparison = []
    for metric, baseline_value in baseline_metrics.items():
        experiment_value = experiment_metrics.get(metric, 0)
        abs_diff = experiment_value - baseline_value
        rel_diff = abs_diff / baseline_value * 100 if baseline_value != 0 else float('inf')
        
        comparison.append({
            'metric': metric,
            'baseline': baseline_value,
            'experiment': experiment_value,
            'absolute_diff': abs_diff,
            'relative_diff_percent': rel_diff
        })
    
    return pd.DataFrame(comparison)
