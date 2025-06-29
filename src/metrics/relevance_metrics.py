"""
Relevance Metrics for Search Quality Dashboard.

This module calculates relevance metrics from search log data, including:
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)
- Precision@k
- Click Position Distribution
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
import json
from scipy import stats

# Set up logging
logger = logging.getLogger(__name__)

class RelevanceMetrics:
    """Calculate relevance metrics from search log data."""
    
    def __init__(self, config: Dict = None):
        """Initialize with configuration."""
        self.config = config or {}
        self.metrics_config = self.config.get('metrics', {})
        
        # Configure NDCG settings
        ndcg_config = self.metrics_config.get('ndcg', {})
        self.k_values = ndcg_config.get('k_values', [3, 5, 10])
        
        # Configure MRR settings
        mrr_config = self.metrics_config.get('mrr', {})
        self.consider_no_clicks = mrr_config.get('consider_no_clicks', True)
    
    def calculate_mrr(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Mean Reciprocal Rank (MRR) metrics.
        
        Args:
            df: DataFrame containing search log data
            
        Returns:
            Dictionary of MRR metrics
        """
        logger.info("Calculating MRR metrics")
        
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Calculate the reciprocal rank for each query
        df_copy['reciprocal_rank'] = df_copy.apply(self._calculate_reciprocal_rank, axis=1)
        
        # Filter based on configuration
        if not self.consider_no_clicks:
            df_copy = df_copy[df_copy['reciprocal_rank'] > 0]
        
        # Calculate overall MRR
        overall_mrr = df_copy['reciprocal_rank'].mean()
        
        # Calculate MRR by day
        df_copy['date'] = pd.to_datetime(df_copy['timestamp']).dt.date
        daily_mrr = df_copy.groupby('date')['reciprocal_rank'].agg(['mean', 'std', 'count']).reset_index()
        
        # Add confidence intervals
        daily_mrr['ci_lower'] = daily_mrr.apply(
            lambda row: row['mean'] - stats.t.ppf(0.975, row['count'] - 1) * row['std'] / np.sqrt(row['count']) 
            if row['count'] > 1 else row['mean'], 
            axis=1
        )
        daily_mrr['ci_upper'] = daily_mrr.apply(
            lambda row: row['mean'] + stats.t.ppf(0.975, row['count'] - 1) * row['std'] / np.sqrt(row['count']) 
            if row['count'] > 1 else row['mean'], 
            axis=1
        )
        
        # Calculate MRR by device type
        device_mrr = df_copy.groupby('device_type')['reciprocal_rank'].agg(['mean', 'std', 'count']).reset_index()
        
        # Add confidence intervals
        device_mrr['ci_lower'] = device_mrr.apply(
            lambda row: row['mean'] - stats.t.ppf(0.975, row['count'] - 1) * row['std'] / np.sqrt(row['count']) 
            if row['count'] > 1 else row['mean'], 
            axis=1
        )
        device_mrr['ci_upper'] = device_mrr.apply(
            lambda row: row['mean'] + stats.t.ppf(0.975, row['count'] - 1) * row['std'] / np.sqrt(row['count']) 
            if row['count'] > 1 else row['mean'], 
            axis=1
        )
        
        # Calculate MRR by query type
        query_type_mrr = df_copy.groupby('query_type')['reciprocal_rank'].agg(['mean', 'std', 'count']).reset_index()
        
        # Add confidence intervals
        query_type_mrr['ci_lower'] = query_type_mrr.apply(
            lambda row: row['mean'] - stats.t.ppf(0.975, row['count'] - 1) * row['std'] / np.sqrt(row['count']) 
            if row['count'] > 1 else row['mean'], 
            axis=1
        )
        query_type_mrr['ci_upper'] = query_type_mrr.apply(
            lambda row: row['mean'] + stats.t.ppf(0.975, row['count'] - 1) * row['std'] / np.sqrt(row['count']) 
            if row['count'] > 1 else row['mean'], 
            axis=1
        )
        
        # Return all metrics
        return {
            'overall_mrr': overall_mrr,
            'daily_mrr': daily_mrr,
            'device_mrr': device_mrr,
            'query_type_mrr': query_type_mrr
        }
    
    def calculate_ndcg(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate NDCG metrics.
        
        Args:
            df: DataFrame containing search log data
            
        Returns:
            Dictionary of NDCG metrics
        """
        logger.info("Calculating NDCG metrics")
        
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Calculate NDCG for each k value
        metrics = {}
        
        for k in self.k_values:
            # Calculate NDCG@k for each query
            df_copy[f'ndcg_{k}'] = df_copy.apply(lambda row: self._calculate_ndcg_k(row, k), axis=1)
            
            # Calculate overall NDCG@k
            overall_ndcg = df_copy[f'ndcg_{k}'].mean()
            metrics[f'overall_ndcg_{k}'] = overall_ndcg
            
            # Calculate NDCG@k by day
            df_copy['date'] = pd.to_datetime(df_copy['timestamp']).dt.date
            daily_ndcg = df_copy.groupby('date')[f'ndcg_{k}'].agg(['mean', 'std', 'count']).reset_index()
            
            # Add confidence intervals
            daily_ndcg['ci_lower'] = daily_ndcg.apply(
                lambda row: row['mean'] - stats.t.ppf(0.975, row['count'] - 1) * row['std'] / np.sqrt(row['count']) 
                if row['count'] > 1 else row['mean'], 
                axis=1
            )
            daily_ndcg['ci_upper'] = daily_ndcg.apply(
                lambda row: row['mean'] + stats.t.ppf(0.975, row['count'] - 1) * row['std'] / np.sqrt(row['count']) 
                if row['count'] > 1 else row['mean'], 
                axis=1
            )
            
            metrics[f'daily_ndcg_{k}'] = daily_ndcg
            
            # Calculate NDCG@k by device type
            device_ndcg = df_copy.groupby('device_type')[f'ndcg_{k}'].agg(['mean', 'std', 'count']).reset_index()
            
            # Add confidence intervals
            device_ndcg['ci_lower'] = device_ndcg.apply(
                lambda row: row['mean'] - stats.t.ppf(0.975, row['count'] - 1) * row['std'] / np.sqrt(row['count']) 
                if row['count'] > 1 else row['mean'], 
                axis=1
            )
            device_ndcg['ci_upper'] = device_ndcg.apply(
                lambda row: row['mean'] + stats.t.ppf(0.975, row['count'] - 1) * row['std'] / np.sqrt(row['count']) 
                if row['count'] > 1 else row['mean'], 
                axis=1
            )
            
            metrics[f'device_ndcg_{k}'] = device_ndcg
            
            # Calculate NDCG@k by query type
            query_type_ndcg = df_copy.groupby('query_type')[f'ndcg_{k}'].agg(['mean', 'std', 'count']).reset_index()
            
            # Add confidence intervals
            query_type_ndcg['ci_lower'] = query_type_ndcg.apply(
                lambda row: row['mean'] - stats.t.ppf(0.975, row['count'] - 1) * row['std'] / np.sqrt(row['count']) 
                if row['count'] > 1 else row['mean'], 
                axis=1
            )
            query_type_ndcg['ci_upper'] = query_type_ndcg.apply(
                lambda row: row['mean'] + stats.t.ppf(0.975, row['count'] - 1) * row['std'] / np.sqrt(row['count']) 
                if row['count'] > 1 else row['mean'], 
                axis=1
            )
            
            metrics[f'query_type_ndcg_{k}'] = query_type_ndcg
        
        return metrics
    
    def calculate_precision_at_k(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Precision@k metrics.
        
        Args:
            df: DataFrame containing search log data
            
        Returns:
            Dictionary of Precision@k metrics
        """
        logger.info("Calculating Precision@k metrics")
        
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Calculate Precision@k for each k value
        metrics = {}
        
        for k in self.k_values:
            # Calculate Precision@k for each query
            df_copy[f'precision_{k}'] = df_copy.apply(lambda row: self._calculate_precision_k(row, k), axis=1)
            
            # Calculate overall Precision@k
            overall_precision = df_copy[f'precision_{k}'].mean()
            metrics[f'overall_precision_{k}'] = overall_precision
            
            # Calculate Precision@k by day
            df_copy['date'] = pd.to_datetime(df_copy['timestamp']).dt.date
            daily_precision = df_copy.groupby('date')[f'precision_{k}'].agg(['mean', 'std', 'count']).reset_index()
            
            # Add confidence intervals
            daily_precision['ci_lower'] = daily_precision.apply(
                lambda row: row['mean'] - stats.t.ppf(0.975, row['count'] - 1) * row['std'] / np.sqrt(row['count']) 
                if row['count'] > 1 else row['mean'], 
                axis=1
            )
            daily_precision['ci_upper'] = daily_precision.apply(
                lambda row: row['mean'] + stats.t.ppf(0.975, row['count'] - 1) * row['std'] / np.sqrt(row['count']) 
                if row['count'] > 1 else row['mean'], 
                axis=1
            )
            
            metrics[f'daily_precision_{k}'] = daily_precision
            
            # Calculate Precision@k by device type
            device_precision = df_copy.groupby('device_type')[f'precision_{k}'].agg(['mean', 'std', 'count']).reset_index()
            
            # Add confidence intervals
            device_precision['ci_lower'] = device_precision.apply(
                lambda row: row['mean'] - stats.t.ppf(0.975, row['count'] - 1) * row['std'] / np.sqrt(row['count']) 
                if row['count'] > 1 else row['mean'], 
                axis=1
            )
            device_precision['ci_upper'] = device_precision.apply(
                lambda row: row['mean'] + stats.t.ppf(0.975, row['count'] - 1) * row['std'] / np.sqrt(row['count']) 
                if row['count'] > 1 else row['mean'], 
                axis=1
            )
            
            metrics[f'device_precision_{k}'] = device_precision
            
            # Calculate Precision@k by query type
            query_type_precision = df_copy.groupby('query_type')[f'precision_{k}'].agg(['mean', 'std', 'count']).reset_index()
            
            # Add confidence intervals
            query_type_precision['ci_lower'] = query_type_precision.apply(
                lambda row: row['mean'] - stats.t.ppf(0.975, row['count'] - 1) * row['std'] / np.sqrt(row['count']) 
                if row['count'] > 1 else row['mean'], 
                axis=1
            )
            query_type_precision['ci_upper'] = query_type_precision.apply(
                lambda row: row['mean'] + stats.t.ppf(0.975, row['count'] - 1) * row['std'] / np.sqrt(row['count']) 
                if row['count'] > 1 else row['mean'], 
                axis=1
            )
            
            metrics[f'query_type_precision_{k}'] = query_type_precision
        
        return metrics
    
    def calculate_click_position_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate click position distribution metrics.
        
        Args:
            df: DataFrame containing search log data
            
        Returns:
            Dictionary of click position distribution metrics
        """
        logger.info("Calculating click position distribution metrics")
        
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Ensure clicked_results is properly parsed
        if 'clicked_results' in df_copy.columns and isinstance(df_copy['clicked_results'].iloc[0], str):
            df_copy['clicked_results'] = df_copy['clicked_results'].apply(json.loads)
        
        # Ensure results is properly parsed
        if 'results' in df_copy.columns and isinstance(df_copy['results'].iloc[0], str):
            df_copy['results'] = df_copy['results'].apply(json.loads)
        
        # Get all clicked positions
        all_click_positions = []
        
        for _, row in df_copy.iterrows():
            clicked_results = row.get('clicked_results', [])
            results = row.get('results', [])
            
            if not isinstance(clicked_results, list) or not isinstance(results, list):
                continue
            
            # Get positions of clicked results (1-indexed)
            positions = []
            for clicked in clicked_results:
                try:
                    pos = results.index(clicked) + 1
                    positions.append(pos)
                except (ValueError, TypeError):
                    continue
            
            all_click_positions.extend(positions)
        
        # Calculate distribution
        if all_click_positions:
            position_counts = pd.Series(all_click_positions).value_counts().sort_index()
            position_distribution = position_counts / position_counts.sum()
            
            # Convert to DataFrame
            position_df = pd.DataFrame({
                'position': position_distribution.index,
                'frequency': position_distribution.values
            })
            
            # Calculate cumulative distribution
            position_df['cumulative_frequency'] = position_df['frequency'].cumsum()
            
            # Get various percentiles
            p50 = position_df[position_df['cumulative_frequency'] >= 0.5].iloc[0]['position'] if len(position_df[position_df['cumulative_frequency'] >= 0.5]) > 0 else None
            p75 = position_df[position_df['cumulative_frequency'] >= 0.75].iloc[0]['position'] if len(position_df[position_df['cumulative_frequency'] >= 0.75]) > 0 else None
            p90 = position_df[position_df['cumulative_frequency'] >= 0.9].iloc[0]['position'] if len(position_df[position_df['cumulative_frequency'] >= 0.9]) > 0 else None
            
            return {
                'position_distribution': position_df,
                'median_click_position': p50,
                'p75_click_position': p75,
                'p90_click_position': p90,
                'total_clicks': len(all_click_positions)
            }
        else:
            return {
                'position_distribution': pd.DataFrame(columns=['position', 'frequency', 'cumulative_frequency']),
                'median_click_position': None,
                'p75_click_position': None,
                'p90_click_position': None,
                'total_clicks': 0
            }
    
    def _calculate_reciprocal_rank(self, row):
        """Calculate reciprocal rank for a single query."""
        # Parse clicked_results if it's a string
        clicked_results = row.get('clicked_results', [])
        if isinstance(clicked_results, str):
            clicked_results = json.loads(clicked_results)
        
        # Parse results if it's a string
        results = row.get('results', [])
        if isinstance(results, str):
            results = json.loads(results)
        
        # If no clicks or no results, return 0
        if not clicked_results or not results:
            return 0
        
        # Get the first clicked result
        first_clicked = clicked_results[0]
        
        # Find its position (1-indexed)
        try:
            position = results.index(first_clicked) + 1
            return 1.0 / position
        except (ValueError, TypeError):
            return 0
    
    def _calculate_ndcg_k(self, row, k):
        """Calculate NDCG@k for a single query."""
        # Parse clicked_results if it's a string
        clicked_results = row.get('clicked_results', [])
        if isinstance(clicked_results, str):
            clicked_results = json.loads(clicked_results)
        
        # Parse results if it's a string
        results = row.get('results', [])
        if isinstance(results, str):
            results = json.loads(results)
        
        # Parse result_relevance if it's a string
        result_relevance = row.get('result_relevance', {})
        if isinstance(result_relevance, str):
            result_relevance = json.loads(result_relevance)
        
        # If no results or no relevance scores, return 0
        if not results or not result_relevance:
            return 0
        
        # Limit to top k results
        results_k = results[:k]
        
        # Get relevance scores for top k results
        rel_scores = [float(result_relevance.get(r, 0)) for r in results_k]
        
        # Get ideal ordering of relevance scores
        ideal_rel = sorted(rel_scores, reverse=True)
        
        # Calculate DCG and IDCG
        dcg = self._calculate_dcg(rel_scores)
        idcg = self._calculate_dcg(ideal_rel)
        
        # Calculate NDCG
        if idcg == 0:
            return 0
        else:
            return dcg / idcg
    
    def _calculate_dcg(self, relevance_scores):
        """Calculate Discounted Cumulative Gain."""
        dcg = 0
        for i, rel in enumerate(relevance_scores):
            # Use log base 2 for the discount
            dcg += (2 ** rel - 1) / np.log2(i + 2)  # i+2 because i is 0-indexed and log base 2 of 1 is 0
        return dcg
    
    def _calculate_precision_k(self, row, k):
        """Calculate Precision@k for a single query."""
        # Parse clicked_results if it's a string
        clicked_results = row.get('clicked_results', [])
        if isinstance(clicked_results, str):
            clicked_results = json.loads(clicked_results)
        
        # Parse results if it's a string
        results = row.get('results', [])
        if isinstance(results, str):
            results = json.loads(results)
        
        # If no results or no clicks, return 0
        if not results or not clicked_results:
            return 0
        
        # Limit to top k results
        results_k = results[:k]
        
        # Count clicked results in top k
        clicked_in_top_k = sum(1 for r in clicked_results if r in results_k)
        
        # Calculate precision
        return clicked_in_top_k / k

    def calculate_all_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate all relevance metrics.
        
        Args:
            df: DataFrame containing search log data
            
        Returns:
            Dictionary of all relevance metrics
        """
        logger.info("Calculating all relevance metrics")
        
        # Calculate individual metric sets
        mrr_metrics = self.calculate_mrr(df)
        ndcg_metrics = self.calculate_ndcg(df)
        precision_metrics = self.calculate_precision_at_k(df)
        click_position_metrics = self.calculate_click_position_distribution(df)
        
        # Combine all metrics
        all_metrics = {
            'mrr': mrr_metrics,
            'ndcg': ndcg_metrics,
            'precision': precision_metrics,
            'click_position': click_position_metrics
        }
        
        return all_metrics
