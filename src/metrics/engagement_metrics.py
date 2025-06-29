import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

class EngagementMetrics:
    """
    Calculate user engagement metrics for search quality analysis.
    
    This class provides methods to compute various engagement metrics
    that indicate how users interact with search results, including
    click-through rates, time spent, and behavioral patterns.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with search event data.
        
        Args:
            data: DataFrame containing search events with required columns:
                  - session_id, user_id, timestamp, event_type, query, etc.
        """
        self.data = data.copy()
        self.search_events = data[data['event_type'] == 'search'].copy()
        self.click_events = data[data['event_type'] == 'click'].copy()
    
    def calculate_click_through_rate(self, 
                                   groupby_cols: Optional[List[str]] = None,
                                   time_period: str = 'daily') -> pd.DataFrame:
        """
        Calculate click-through rate (CTR) - percentage of searches that result in clicks.
        
        Args:
            groupby_cols: Additional columns to group by (e.g., ['user_segment', 'device_type'])
            time_period: Time aggregation period ('daily', 'weekly', 'monthly')
            
        Returns:
            DataFrame with CTR metrics
        """
        # Add time period column
        self.search_events['period'] = self._get_time_period(
            self.search_events['timestamp'], time_period
        )
        self.click_events['period'] = self._get_time_period(
            self.click_events['timestamp'], time_period
        )
        
        base_groups = ['period']
        if groupby_cols:
            base_groups.extend(groupby_cols)
        
        # Count total searches
        search_counts = self.search_events.groupby(base_groups).size().reset_index(name='total_searches')
        
        # Count searches with clicks
        searches_with_clicks = self.click_events.groupby(
            base_groups + ['session_id', 'query_index_in_session']
        ).size().reset_index(name='click_count')
        
        clicks_per_period = searches_with_clicks.groupby(base_groups).size().reset_index(name='searches_with_clicks')
        
        # Merge and calculate CTR
        ctr_data = search_counts.merge(clicks_per_period, on=base_groups, how='left').fillna(0)
        ctr_data['click_through_rate'] = ctr_data['searches_with_clicks'] / ctr_data['total_searches']
        ctr_data['ctr_percentage'] = ctr_data['click_through_rate'] * 100
        
        return ctr_data
    
    def calculate_time_to_click(self, 
                              groupby_cols: Optional[List[str]] = None,
                              time_period: str = 'daily') -> pd.DataFrame:
        """
        Calculate average time from search to first click.
        
        Args:
            groupby_cols: Additional columns to group by
            time_period: Time aggregation period
            
        Returns:
            DataFrame with time to click metrics
        """
        # Merge search and click events
        merged_data = self.search_events.merge(
            self.click_events,
            on=['session_id', 'query_index_in_session'],
            suffixes=('_search', '_click')
        )
        
        # Calculate time to click
        merged_data['time_to_click'] = (
            merged_data['timestamp_click'] - merged_data['timestamp_search']
        ).dt.total_seconds()
        
        # Add time period
        merged_data['period'] = self._get_time_period(
            merged_data['timestamp_search'], time_period
        )
        
        # Group and aggregate
        base_groups = ['period']
        if groupby_cols:
            base_groups.extend([f"{col}_search" for col in groupby_cols])
        
        ttc_metrics = merged_data.groupby(base_groups).agg({
            'time_to_click': ['mean', 'median', 'std', 'count'],
            'session_id': 'nunique'
        }).round(2)
        
        # Flatten column names
        ttc_metrics.columns = [
            'avg_time_to_click_seconds', 'median_time_to_click_seconds',
            'std_time_to_click_seconds', 'total_clicks', 'unique_sessions'
        ]
        
        return ttc_metrics.reset_index()
    
    def calculate_session_engagement(self, 
                                   time_period: str = 'daily') -> pd.DataFrame:
        """
        Calculate session-level engagement metrics.
        
        Args:
            time_period: Time aggregation period
            
        Returns:
            DataFrame with session engagement metrics
        """
        # Calculate session metrics
        session_metrics = []
        
        for session_id, session_data in self.search_events.groupby('session_id'):
            session_clicks = self.click_events[
                self.click_events['session_id'] == session_id
            ]
            
            # Basic session info
            session_start = session_data['timestamp'].min()
            session_end = session_data['timestamp'].max()
            session_duration = (session_end - session_start).total_seconds()
            
            # Engagement metrics
            num_queries = len(session_data)
            num_clicks = len(session_clicks)
            unique_queries = session_data['query'].nunique()
            
            # Calculate reformulation rate
            reformulation_rate = (num_queries - unique_queries) / num_queries if num_queries > 0 else 0
            
            # Session engagement score (composite metric)
            engagement_score = self._calculate_session_engagement_score(
                num_queries, num_clicks, session_duration, reformulation_rate
            )
            
            session_metrics.append({
                'session_id': session_id,
                'user_id': session_data['user_id'].iloc[0],
                'user_segment': session_data['user_segment'].iloc[0],
                'device_type': session_data['device_type'].iloc[0],
                'period': self._get_time_period(pd.Series([session_start]), time_period)[0],
                'session_duration_seconds': session_duration,
                'num_queries': num_queries,
                'num_clicks': num_clicks,
                'unique_queries': unique_queries,
                'reformulation_rate': reformulation_rate,
                'engagement_score': engagement_score
            })
        
        return pd.DataFrame(session_metrics)
    
    def calculate_click_position_metrics(self, 
                                       groupby_cols: Optional[List[str]] = None,
                                       time_period: str = 'daily') -> pd.DataFrame:
        """
        Calculate metrics related to click positions in search results.
        
        Args:
            groupby_cols: Additional columns to group by
            time_period: Time aggregation period
            
        Returns:
            DataFrame with click position metrics
        """
        if self.click_events.empty:
            return pd.DataFrame()
        
        # Add time period
        self.click_events['period'] = self._get_time_period(
            self.click_events['timestamp'], time_period
        )
        
        base_groups = ['period']
        if groupby_cols:
            base_groups.extend(groupby_cols)
        
        # Calculate position metrics
        position_metrics = self.click_events.groupby(base_groups).agg({
            'click_position': ['mean', 'median', 'std'],
            'session_id': 'count'
        }).round(2)
        
        # Flatten column names
        position_metrics.columns = [
            'avg_click_position', 'median_click_position', 
            'std_click_position', 'total_clicks'
        ]
        
        # Calculate percentage of clicks in top positions
        for top_n in [1, 3, 5]:
            top_clicks = self.click_events[
                self.click_events['click_position'] <= top_n
            ].groupby(base_groups).size()
            
            total_clicks = self.click_events.groupby(base_groups).size()
            top_percentage = (top_clicks / total_clicks * 100).fillna(0)
            position_metrics[f'top_{top_n}_click_percentage'] = top_percentage
        
        return position_metrics.reset_index()
    
    def calculate_result_engagement(self, 
                                  time_period: str = 'daily') -> pd.DataFrame:
        """
        Calculate engagement metrics for search results.
        
        Args:
            time_period: Time aggregation period
            
        Returns:
            DataFrame with result engagement metrics
        """
        if self.click_events.empty:
            return pd.DataFrame()
        
        # Add time period
        self.click_events['period'] = self._get_time_period(
            self.click_events['timestamp'], time_period
        )
        
        # Calculate result engagement metrics
        result_metrics = self.click_events.groupby(['period', 'result_type']).agg({
            'time_on_result_seconds': ['mean', 'median', 'std'],
            'is_satisfied': 'mean',
            'session_id': 'count'
        }).round(2)
        
        # Flatten column names
        result_metrics.columns = [
            'avg_time_on_result', 'median_time_on_result', 
            'std_time_on_result', 'satisfaction_rate', 'total_clicks'
        ]
        
        return result_metrics.reset_index()
    
    def calculate_user_engagement_segments(self) -> pd.DataFrame:
        """
        Segment users based on their engagement patterns.
        
        Returns:
            DataFrame with user engagement segments
        """
        # Calculate user-level metrics
        user_metrics = []
        
        for user_id, user_data in self.search_events.groupby('user_id'):
            user_clicks = self.click_events[
                self.click_events['user_id'] == user_id
            ]
            
            # Calculate engagement metrics
            total_searches = len(user_data)
            total_clicks = len(user_clicks)
            unique_sessions = user_data['session_id'].nunique()
            avg_session_duration = user_data.groupby('session_id')['timestamp'].apply(
                lambda x: (x.max() - x.min()).total_seconds()
            ).mean()
            
            # Calculate satisfaction rate
            satisfaction_rate = user_clicks['is_satisfied'].mean() if not user_clicks.empty else 0
            
            user_metrics.append({
                'user_id': user_id,
                'user_segment': user_data['user_segment'].iloc[0],
                'total_searches': total_searches,
                'total_clicks': total_clicks,
                'unique_sessions': unique_sessions,
                'click_through_rate': total_clicks / total_searches if total_searches > 0 else 0,
                'avg_session_duration': avg_session_duration,
                'satisfaction_rate': satisfaction_rate
            })
        
        user_df = pd.DataFrame(user_metrics)
        
        # Define engagement segments based on behavior
        def classify_engagement(row):
            if row['click_through_rate'] >= 0.7 and row['satisfaction_rate'] >= 0.6:
                return 'highly_engaged'
            elif row['click_through_rate'] >= 0.4 and row['satisfaction_rate'] >= 0.3:
                return 'moderately_engaged'
            elif row['click_through_rate'] >= 0.1:
                return 'low_engaged'
            else:
                return 'disengaged'
        
        user_df['engagement_segment'] = user_df.apply(classify_engagement, axis=1)
        
        return user_df
    
    def _calculate_session_engagement_score(self, 
                                          num_queries: int, 
                                          num_clicks: int,
                                          session_duration: float, 
                                          reformulation_rate: float) -> float:
        """
        Calculate a composite engagement score for a session.
        
        Args:
            num_queries: Number of queries in the session
            num_clicks: Number of clicks in the session
            session_duration: Duration of the session in seconds
            reformulation_rate: Rate of query reformulation
            
        Returns:
            Engagement score between 0 and 1
        """
        # Normalize factors
        query_score = min(num_queries / 5, 1.0)  # Normalize to max 5 queries
        click_score = min(num_clicks / 10, 1.0)  # Normalize to max 10 clicks
        duration_score = min(session_duration / 300, 1.0)  # Normalize to max 5 minutes
        reformulation_penalty = max(0, 1 - reformulation_rate)  # Penalty for too much reformulation
        
        # Weighted combination
        engagement_score = (
            query_score * 0.2 +
            click_score * 0.4 +
            duration_score * 0.3 +
            reformulation_penalty * 0.1
        )
        
        return round(engagement_score, 3)
    
    def _get_time_period(self, timestamps: pd.Series, period: str) -> pd.Series:
        """
        Convert timestamps to time period labels.
        
        Args:
            timestamps: Series of timestamps
            period: Time period ('daily', 'weekly', 'monthly')
            
        Returns:
            Series with period labels
        """
        if period == 'daily':
            return timestamps.dt.date
        elif period == 'weekly':
            return timestamps.dt.to_period('W').astype(str)
        elif period == 'monthly':
            return timestamps.dt.to_period('M').astype(str)
        else:
            raise ValueError(f"Unsupported time period: {period}")
    
    def get_engagement_summary(self) -> Dict:
        """
        Get a summary of key engagement metrics.
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {}
        
        # Overall CTR
        total_searches = len(self.search_events)
        searches_with_clicks = self.click_events.groupby(
            ['session_id', 'query_index_in_session']
        ).size().shape[0]
        summary['overall_ctr'] = searches_with_clicks / total_searches if total_searches > 0 else 0
        
        # Average time to click
        if not self.click_events.empty:
            merged_data = self.search_events.merge(
                self.click_events,
                on=['session_id', 'query_index_in_session'],
                suffixes=('_search', '_click')
            )
            time_to_click = (
                merged_data['timestamp_click'] - merged_data['timestamp_search']
            ).dt.total_seconds()
            summary['avg_time_to_click'] = time_to_click.mean()
        else:
            summary['avg_time_to_click'] = None
        
        # Average click position
        if not self.click_events.empty:
            summary['avg_click_position'] = self.click_events['click_position'].mean()
            summary['top_3_click_rate'] = (
                self.click_events['click_position'] <= 3
            ).mean()
        else:
            summary['avg_click_position'] = None
            summary['top_3_click_rate'] = None
        
        # Session metrics
        session_metrics = self.calculate_session_engagement()
        if not session_metrics.empty:
            summary['avg_session_duration'] = session_metrics['session_duration_seconds'].mean()
            summary['avg_queries_per_session'] = session_metrics['num_queries'].mean()
            summary['avg_engagement_score'] = session_metrics['engagement_score'].mean()
        
        return summary

if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('..')
    from data.data_generator import SearchDataGenerator
    
    # Generate sample data
    generator = SearchDataGenerator(seed=42)
    data = generator.generate_search_logs(num_sessions=100)
    
    # Calculate engagement metrics
    engagement = EngagementMetrics(data)
    
    # Get various metrics
    ctr_metrics = engagement.calculate_click_through_rate(
        groupby_cols=['user_segment'], 
        time_period='daily'
    )
    print("CTR Metrics:")
    print(ctr_metrics.head())
    
    # Get summary
    summary = engagement.get_engagement_summary()
    print("\nEngagement Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
