"""
Engagement Metrics for Search Quality Dashboard.

This module calculates engagement metrics from search log data, including:
- Click-Through Rate (CTR)
- Zero-Click Rate
- Time to First Click
- Query Reformulation Rate
- Session Abandonment Rate
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime, timedelta
import json
from scipy import stats

# Set up logging
logger = logging.getLogger(__name__)

class EngagementMetrics:
    """Calculate engagement metrics from search log data."""
    
    def __init__(self, config: Dict = None):
        """Initialize with configuration."""
        self.config = config or {}
        self.metrics_config = self.config.get('metrics', {})
        
        # Configure time-to-click settings
        ttc_config = self.metrics_config.get('time_to_click', {})
        self.max_time_to_click = ttc_config.get('max_time', 600)  # 10 minutes by default
        self.ttc_outlier_threshold = ttc_config.get('outlier_threshold', 0.95)
        
        # Configure session timeout
        self.session_timeout = self.config.get('session_timeout', 1800)  # 30 minutes by default
        
    def calculate_ctr(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Click-Through Rate (CTR) metrics.
        
        Args:
            df: DataFrame containing search log data with at least 'query_id' and 'clicked_results' columns
            
        Returns:
            Dictionary of CTR metrics
        """
        logger.info("Calculating CTR metrics")
        
        # Ensure 'clicked_results' is parsed from JSON if necessary
        if df['clicked_results'].dtype == 'object' and isinstance(df['clicked_results'].iloc[0], str):
            df['clicked_results'] = df['clicked_results'].apply(json.loads)
        
        # Add flag for queries with clicks
        df['has_clicks'] = df['clicked_results'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)
        
        # Overall CTR
        total_queries = len(df)
        queries_with_clicks = df['has_clicks'].sum()
        overall_ctr = queries_with_clicks / total_queries if total_queries > 0 else 0
        
        # Calculate 95% confidence interval for CTR
        if total_queries > 0:
            ctr_ci = stats.binom.interval(0.95, total_queries, overall_ctr)
            ctr_ci = (ctr_ci[0] / total_queries, ctr_ci[1] / total_queries)
        else:
            ctr_ci = (0, 0)
        
        # CTR by day
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        daily_ctr = df.groupby('date').apply(
            lambda x: (
                x['has_clicks'].sum() / len(x) if len(x) > 0 else 0,
                len(x)  # Include count for confidence interval calculation
            )
        ).reset_index()
        daily_ctr.columns = ['date', 'ctr_and_count']
        daily_ctr['ctr'] = daily_ctr['ctr_and_count'].apply(lambda x: x[0])
        daily_ctr['count'] = daily_ctr['ctr_and_count'].apply(lambda x: x[1])
        
        # Add confidence intervals to daily CTR
        daily_ctr['ctr_ci_lower'] = daily_ctr.apply(
            lambda row: stats.binom.interval(0.95, row['count'], row['ctr'])[0] / row['count'] 
            if row['count'] > 0 else 0, 
            axis=1
        )
        daily_ctr['ctr_ci_upper'] = daily_ctr.apply(
            lambda row: stats.binom.interval(0.95, row['count'], row['ctr'])[1] / row['count'] 
            if row['count'] > 0 else 0, 
            axis=1
        )
        
        daily_ctr = daily_ctr.drop('ctr_and_count', axis=1)
        
        # CTR by device type
        device_ctr = df.groupby('device_type').apply(
            lambda x: (
                x['has_clicks'].sum() / len(x) if len(x) > 0 else 0,
                len(x)
            )
        ).reset_index()
        device_ctr.columns = ['device_type', 'ctr_and_count']
        device_ctr['ctr'] = device_ctr['ctr_and_count'].apply(lambda x: x[0])
        device_ctr['count'] = device_ctr['ctr_and_count'].apply(lambda x: x[1])
        
        # Add confidence intervals to device CTR
        device_ctr['ctr_ci_lower'] = device_ctr.apply(
            lambda row: stats.binom.interval(0.95, row['count'], row['ctr'])[0] / row['count'] 
            if row['count'] > 0 else 0, 
            axis=1
        )
        device_ctr['ctr_ci_upper'] = device_ctr.apply(
            lambda row: stats.binom.interval(0.95, row['count'], row['ctr'])[1] / row['count'] 
            if row['count'] > 0 else 0, 
            axis=1
        )
        
        device_ctr = device_ctr.drop('ctr_and_count', axis=1)
        
        # CTR by query type
        query_type_ctr = df.groupby('query_type').apply(
            lambda x: (
                x['has_clicks'].sum() / len(x) if len(x) > 0 else 0,
                len(x)
            )
        ).reset_index()
        query_type_ctr.columns = ['query_type', 'ctr_and_count']
        query_type_ctr['ctr'] = query_type_ctr['ctr_and_count'].apply(lambda x: x[0])
        query_type_ctr['count'] = query_type_ctr['ctr_and_count'].apply(lambda x: x[1])
        
        # Add confidence intervals to query type CTR
        query_type_ctr['ctr_ci_lower'] = query_type_ctr.apply(
            lambda row: stats.binom.interval(0.95, row['count'], row['ctr'])[0] / row['count'] 
            if row['count'] > 0 else 0, 
            axis=1
        )
        query_type_ctr['ctr_ci_upper'] = query_type_ctr.apply(
            lambda row: stats.binom.interval(0.95, row['count'], row['ctr'])[1] / row['count'] 
            if row['count'] > 0 else 0, 
            axis=1
        )
        
        query_type_ctr = query_type_ctr.drop('ctr_and_count', axis=1)
        
        # Zero-Click Rate (ZCR)
        zcr = 1 - overall_ctr
        zcr_ci = (1 - ctr_ci[1], 1 - ctr_ci[0])  # Invert the CTR CI
        
        # Results per click
        df['results_per_click'] = df.apply(
            lambda row: len(row['results']) / len(row['clicked_results']) 
            if isinstance(row['clicked_results'], list) and len(row['clicked_results']) > 0 
            else np.nan, 
            axis=1
        )
        avg_results_per_click = df['results_per_click'].mean()
        
        # Return all metrics
        return {
            'overall_ctr': overall_ctr,
            'ctr_ci': ctr_ci,
            'daily_ctr': daily_ctr,
            'device_ctr': device_ctr,
            'query_type_ctr': query_type_ctr,
            'zero_click_rate': zcr,
            'zcr_ci': zcr_ci,
            'avg_results_per_click': avg_results_per_click
        }
    
    def calculate_time_to_click(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Time to Click metrics.
        
        Args:
            df: DataFrame containing search log data
            
        Returns:
            Dictionary of Time to Click metrics
        """
        logger.info("Calculating Time to Click metrics")
        
        # Parse click timestamps if needed
        if df['click_timestamps'].dtype == 'object' and isinstance(df['click_timestamps'].iloc[0], str):
            df['click_timestamps'] = df['click_timestamps'].apply(json.loads)
        
        # Filter to queries with at least one click
        click_df = df[df['clicked_results'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)].copy()
        
        if len(click_df) == 0:
            logger.warning("No clicks found in data, cannot calculate time to click metrics")
            return {
                'mean_time_to_first_click': None,
                'median_time_to_first_click': None,
                'daily_ttc': pd.DataFrame(),
                'device_ttc': pd.DataFrame(),
                'query_type_ttc': pd.DataFrame()
            }
        
        # Calculate time to first click for each query
        click_df['time_to_first_click'] = click_df.apply(self._calculate_time_to_first_click, axis=1)
        
        # Remove outliers
        if len(click_df) > 0:
            ttc_threshold = click_df['time_to_first_click'].quantile(self.ttc_outlier_threshold)
            click_df = click_df[click_df['time_to_first_click'] <= ttc_threshold]
        
        # Calculate overall metrics
        mean_ttc = click_df['time_to_first_click'].mean()
        median_ttc = click_df['time_to_first_click'].median()
        
        # Calculate time to first click by day
        click_df['date'] = pd.to_datetime(click_df['timestamp']).dt.date
        daily_ttc = click_df.groupby('date')['time_to_first_click'].agg(['mean', 'median', 'std', 'count']).reset_index()
        
        # Add confidence intervals (using t-distribution)
        daily_ttc['ci_lower'] = daily_ttc.apply(
            lambda row: row['mean'] - stats.t.ppf(0.975, row['count'] - 1) * row['std'] / np.sqrt(row['count']) 
            if row['count'] > 1 else row['mean'], 
            axis=1
        )
        daily_ttc['ci_upper'] = daily_ttc.apply(
            lambda row: row['mean'] + stats.t.ppf(0.975, row['count'] - 1) * row['std'] / np.sqrt(row['count']) 
            if row['count'] > 1 else row['mean'], 
            axis=1
        )
        
        # Calculate time to first click by device type
        device_ttc = click_df.groupby('device_type')['time_to_first_click'].agg(['mean', 'median', 'std', 'count']).reset_index()
        
        # Add confidence intervals
        device_ttc['ci_lower'] = device_ttc.apply(
            lambda row: row['mean'] - stats.t.ppf(0.975, row['count'] - 1) * row['std'] / np.sqrt(row['count']) 
            if row['count'] > 1 else row['mean'], 
            axis=1
        )
        device_ttc['ci_upper'] = device_ttc.apply(
            lambda row: row['mean'] + stats.t.ppf(0.975, row['count'] - 1) * row['std'] / np.sqrt(row['count']) 
            if row['count'] > 1 else row['mean'], 
            axis=1
        )
        
        # Calculate time to first click by query type
        query_type_ttc = click_df.groupby('query_type')['time_to_first_click'].agg(['mean', 'median', 'std', 'count']).reset_index()
        
        # Add confidence intervals
        query_type_ttc['ci_lower'] = query_type_ttc.apply(
            lambda row: row['mean'] - stats.t.ppf(0.975, row['count'] - 1) * row['std'] / np.sqrt(row['count']) 
            if row['count'] > 1 else row['mean'], 
            axis=1
        )
        query_type_ttc['ci_upper'] = query_type_ttc.apply(
            lambda row: row['mean'] + stats.t.ppf(0.975, row['count'] - 1) * row['std'] / np.sqrt(row['count']) 
            if row['count'] > 1 else row['mean'], 
            axis=1
        )
        
        # Return all metrics
        return {
            'mean_time_to_first_click': mean_ttc,
            'median_time_to_first_click': median_ttc,
            'daily_ttc': daily_ttc,
            'device_ttc': device_ttc,
            'query_type_ttc': query_type_ttc
        }
    
    def calculate_reformulation_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Query Reformulation metrics.
        
        Args:
            df: DataFrame containing search log data
            
        Returns:
            Dictionary of Query Reformulation metrics
        """
        logger.info("Calculating Query Reformulation metrics")
        
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Sort by session_id and timestamp
        df_copy = df_copy.sort_values(['session_id', 'timestamp'])
        
        # Calculate time difference between consecutive queries in the same session
        df_copy['next_timestamp'] = df_copy.groupby('session_id')['timestamp'].shift(-1)
        df_copy['time_to_next_query'] = (df_copy['next_timestamp'] - df_copy['timestamp']).dt.total_seconds()
        
        # Filter out queries that are the last in their session or where time to next query is > session timeout
        df_copy = df_copy[
            (df_copy['time_to_next_query'].notna()) & 
            (df_copy['time_to_next_query'] <= self.session_timeout)
        ]
        
        # Calculate reformulation rate
        total_non_last_queries = len(df_copy)
        reformulation_count = df_copy['is_reformulation'].shift(-1).sum()
        
        if total_non_last_queries > 0:
            reformulation_rate = reformulation_count / total_non_last_queries
        else:
            reformulation_rate = 0
        
        # Calculate reformulation rate by day
        df_copy['date'] = pd.to_datetime(df_copy['timestamp']).dt.date
        daily_reformulation = df_copy.groupby('date').apply(
            lambda x: (
                x['is_reformulation'].shift(-1).sum() / len(x) if len(x) > 0 else 0,
                len(x)
            )
        ).reset_index()
        daily_reformulation.columns = ['date', 'reform_and_count']
        daily_reformulation['reformulation_rate'] = daily_reformulation['reform_and_count'].apply(lambda x: x[0])
        daily_reformulation['count'] = daily_reformulation['reform_and_count'].apply(lambda x: x[1])
        daily_reformulation = daily_reformulation.drop('reform_and_count', axis=1)
        
        # Calculate reformulation rate by device type
        device_reformulation = df_copy.groupby('device_type').apply(
            lambda x: (
                x['is_reformulation'].shift(-1).sum() / len(x) if len(x) > 0 else 0,
                len(x)
            )
        ).reset_index()
        device_reformulation.columns = ['device_type', 'reform_and_count']
        device_reformulation['reformulation_rate'] = device_reformulation['reform_and_count'].apply(lambda x: x[0])
        device_reformulation['count'] = device_reformulation['reform_and_count'].apply(lambda x: x[1])
        device_reformulation = device_reformulation.drop('reform_and_count', axis=1)
        
        # Calculate reformulation rate by query type
        query_type_reformulation = df_copy.groupby('query_type').apply(
            lambda x: (
                x['is_reformulation'].shift(-1).sum() / len(x) if len(x) > 0 else 0,
                len(x)
            )
        ).reset_index()
        query_type_reformulation.columns = ['query_type', 'reform_and_count']
        query_type_reformulation['reformulation_rate'] = query_type_reformulation['reform_and_count'].apply(lambda x: x[0])
        query_type_reformulation['count'] = query_type_reformulation['reform_and_count'].apply(lambda x: x[1])
        query_type_reformulation = query_type_reformulation.drop('reform_and_count', axis=1)
        
        # Return all metrics
        return {
            'overall_reformulation_rate': reformulation_rate,
            'daily_reformulation': daily_reformulation,
            'device_reformulation': device_reformulation,
            'query_type_reformulation': query_type_reformulation
        }
    
    def calculate_session_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Session-level metrics.
        
        Args:
            df: DataFrame containing search log data
            
        Returns:
            Dictionary of Session metrics
        """
        logger.info("Calculating Session metrics")
        
        # Group by session_id
        session_groups = df.groupby('session_id')
        
        session_data = []
        
        for session_id, session_df in session_groups:
            # Sort by timestamp within session
            session_df = session_df.sort_values('timestamp')
            
            # Basic session metrics
            start_time = session_df['timestamp'].min()
            end_time = session_df['timestamp'].max()
            duration = (end_time - start_time).total_seconds()
            
            # User and device info
            user_id = session_df['user_id'].iloc[0]
            device_type = session_df['device_type'].iloc[0]
            location = session_df['location'].iloc[0]
            
            # Query and click counts
            num_queries = len(session_df)
            
            # Calculate total clicks in session
            if 'clicked_results' in session_df.columns:
                if isinstance(session_df['clicked_results'].iloc[0], str):
                    total_clicks = session_df['clicked_results'].apply(
                        lambda x: len(json.loads(x)) if x else 0
                    ).sum()
                else:
                    total_clicks = session_df['clicked_results'].apply(
                        lambda x: len(x) if isinstance(x, list) else 0
                    ).sum()
            else:
                total_clicks = session_df['num_clicks'].sum()
            
            # Check if session had any clicks
            has_clicks = total_clicks > 0
            
            # Calculate reformulations
            num_reformulations = session_df['is_reformulation'].sum()
            
            # Session success metrics
            # 1. Has at least one click
            success_has_click = has_clicks
            
            # 2. Last query has a click (didn't abandon)
            if 'clicked_results' in session_df.columns:
                if isinstance(session_df['clicked_results'].iloc[-1], str):
                    last_query_has_click = len(json.loads(session_df['clicked_results'].iloc[-1])) > 0
                else:
                    last_query_has_click = (isinstance(session_df['clicked_results'].iloc[-1], list) and 
                                         len(session_df['clicked_results'].iloc[-1]) > 0)
            else:
                last_query_has_click = session_df['num_clicks'].iloc[-1] > 0
            
            # 3. Low reformulation ratio
            reformulation_ratio = num_reformulations / (num_queries - 1) if num_queries > 1 else 0
            low_reformulation = reformulation_ratio < 0.5  # Threshold
            
            # Overall success score (0-3 scale)
            success_score = sum([
                success_has_click,
                last_query_has_click,
                low_reformulation
            ])
            
            # Add to session data
            session_record = {
                'session_id': session_id,
                'user_id': user_id,
                'device_type': device_type,
                'location': location,
                'start_time': start_time,
                'end_time': end_time,
                'duration_seconds': duration,
                'num_queries': num_queries,
                'total_clicks': total_clicks,
                'clicks_per_query': total_clicks / num_queries if num_queries > 0 else 0,
                'num_reformulations': num_reformulations,
                'reformulation_ratio': reformulation_ratio,
                'has_clicks': has_clicks,
                'last_query_has_click': last_query_has_click,
                'success_score': success_score,
                'date': pd.to_datetime(start_time).date()
            }
            
            session_data.append(session_record)
        
        # Create session DataFrame
        sessions_df = pd.DataFrame(session_data)
        
        # Calculate overall session metrics
        total_sessions = len(sessions_df)
        
        # Abandonment rate (sessions with no clicks)
        abandonment_count = (~sessions_df['has_clicks']).sum()
        abandonment_rate = abandonment_count / total_sessions if total_sessions > 0 else 0
        
        # Success rate (sessions with success score >= 2)
        success_count = (sessions_df['success_score'] >= 2).sum()
        success_rate = success_count / total_sessions if total_sessions > 0 else 0
        
        # Average queries per session
        avg_queries_per_session = sessions_df['num_queries'].mean()
        
        # Average session duration
        avg_session_duration = sessions_df['duration_seconds'].mean()
        
        # Single-query session rate
        single_query_count = (sessions_df['num_queries'] == 1).sum()
        single_query_rate = single_query_count / total_sessions if total_sessions > 0 else 0
        
        # Daily metrics
        daily_session_metrics = sessions_df.groupby('date').agg({
            'session_id': 'count',
            'has_clicks': lambda x: (~x).mean(),  # Abandonment rate
            'success_score': lambda x: (x >= 2).mean(),  # Success rate
            'num_queries': 'mean',  # Avg queries per session
            'duration_seconds': 'mean',  # Avg session duration
        }).reset_index()
        
        daily_session_metrics.columns = [
            'date', 'num_sessions', 'abandonment_rate', 'success_rate', 
            'avg_queries_per_session', 'avg_duration'
        ]
        
        # Device metrics
        device_session_metrics = sessions_df.groupby('device_type').agg({
            'session_id': 'count',
            'has_clicks': lambda x: (~x).mean(),  # Abandonment rate
            'success_score': lambda x: (x >= 2).mean(),  # Success rate
            'num_queries': 'mean',  # Avg queries per session
            'duration_seconds': 'mean',  # Avg session duration
        }).reset_index()
        
        device_session_metrics.columns = [
            'device_type', 'num_sessions', 'abandonment_rate', 'success_rate', 
            'avg_queries_per_session', 'avg_duration'
        ]
        
        # Return all metrics
        return {
            'sessions_df': sessions_df,
            'total_sessions': total_sessions,
            'abandonment_rate': abandonment_rate,
            'success_rate': success_rate,
            'avg_queries_per_session': avg_queries_per_session,
            'avg_session_duration': avg_session_duration,
            'single_query_rate': single_query_rate,
            'daily_session_metrics': daily_session_metrics,
            'device_session_metrics': device_session_metrics
        }
    
    def _calculate_time_to_first_click(self, row):
        """Helper method to calculate time to first click in seconds."""
        # Parse timestamps if they're in string format
        click_timestamps = row['click_timestamps']
        if isinstance(click_timestamps, str):
            click_timestamps = json.loads(click_timestamps)
        
        # If no clicks, return None
        if not click_timestamps or len(click_timestamps) == 0:
            return None
        
        # Get first click timestamp
        first_click_time = pd.to_datetime(click_timestamps[0])
        query_time = row['timestamp']
        
        # Calculate time difference in seconds
        time_diff = (first_click_time - query_time).total_seconds()
        
        # Apply maximum time constraint
        if time_diff > self.max_time_to_click or time_diff < 0:
            return None
        
        return time_diff

    def calculate_all_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate all engagement metrics.
        
        Args:
            df: DataFrame containing search log data
            
        Returns:
            Dictionary of all engagement metrics
        """
        logger.info("Calculating all engagement metrics")
        
        # Calculate individual metric sets
        ctr_metrics = self.calculate_ctr(df)
        time_to_click_metrics = self.calculate_time_to_click(df)
        reformulation_metrics = self.calculate_reformulation_metrics(df)
        session_metrics = self.calculate_session_metrics(df)
        
        # Combine all metrics
        all_metrics = {
            'ctr': ctr_metrics,
            'time_to_click': time_to_click_metrics,
            'reformulation': reformulation_metrics,
            'session': session_metrics
        }
        
        return all_metrics
