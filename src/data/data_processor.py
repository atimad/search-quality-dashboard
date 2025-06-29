import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchDataProcessor:
    """
    Processes search log data and calculates metrics for the dashboard.
    
    This class takes raw search event data and processes it to calculate
    various search quality metrics including engagement, relevance, and
    user satisfaction indicators.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the data processor.
        
        Args:
            data_path: Path to the raw search data CSV file
        """
        self.data_path = data_path
        self.raw_data = None
        self.processed_metrics = {}
        
    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load search data from CSV file.
        
        Args:
            file_path: Path to the data file (optional if set in constructor)
            
        Returns:
            DataFrame with loaded search data
        """
        path = file_path or self.data_path
        if not path:
            raise ValueError("No data path provided")
            
        try:
            self.raw_data = pd.read_csv(path)
            self.raw_data['timestamp'] = pd.to_datetime(self.raw_data['timestamp'])
            logger.info(f"Loaded {len(self.raw_data)} records from {path}")
            return self.raw_data
        except Exception as e:
            logger.error(f"Error loading data from {path}: {e}")
            raise
    
    def process_session_metrics(self) -> pd.DataFrame:
        """
        Calculate session-level metrics from search events.
        
        Returns:
            DataFrame with session-level aggregated metrics
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info("Processing session-level metrics...")
        
        # Separate search and click events
        search_events = self.raw_data[self.raw_data['event_type'] == 'search'].copy()
        click_events = self.raw_data[self.raw_data['event_type'] == 'click'].copy()
        
        # Calculate session metrics
        session_metrics = []
        
        for session_id, session_data in search_events.groupby('session_id'):
            # Basic session info
            session_start = session_data['timestamp'].min()
            session_end = session_data['timestamp'].max()
            session_duration = (session_end - session_start).total_seconds()
            
            # Query metrics
            num_queries = len(session_data)
            avg_query_length = session_data['query_length'].mean()
            unique_queries = session_data['query'].nunique()
            
            # Click metrics for this session
            session_clicks = click_events[click_events['session_id'] == session_id]
            num_clicks = len(session_clicks)
            
            # Calculate click-through rate
            ctr = num_clicks / num_queries if num_queries > 0 else 0
            
            # Calculate satisfaction metrics
            satisfied_clicks = session_clicks[session_clicks['is_satisfied'] == True] if not session_clicks.empty else pd.DataFrame()
            satisfaction_rate = len(satisfied_clicks) / num_clicks if num_clicks > 0 else 0
            
            # Time metrics
            avg_time_to_click = self._calculate_avg_time_to_click(session_data, session_clicks)
            
            # Reformulation metrics
            reformulation_rate = (num_queries - unique_queries) / num_queries if num_queries > 0 else 0
            
            session_metric = {
                'session_id': session_id,
                'user_id': session_data['user_id'].iloc[0],
                'user_segment': session_data['user_segment'].iloc[0],
                'device_type': session_data['device_type'].iloc[0],
                'session_start': session_start,
                'session_duration_seconds': session_duration,
                'num_queries': num_queries,
                'num_clicks': num_clicks,
                'click_through_rate': ctr,
                'satisfaction_rate': satisfaction_rate,
                'avg_query_length': avg_query_length,
                'unique_queries': unique_queries,
                'reformulation_rate': reformulation_rate,
                'avg_time_to_click': avg_time_to_click,
                'date': session_start.date()
            }
            session_metrics.append(session_metric)
        
        session_df = pd.DataFrame(session_metrics)
        self.processed_metrics['sessions'] = session_df
        logger.info(f"Processed {len(session_df)} sessions")
        
        return session_df
    
    def process_query_metrics(self) -> pd.DataFrame:
        """
        Calculate query-level metrics from search events.
        
        Returns:
            DataFrame with query-level metrics
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info("Processing query-level metrics...")
        
        search_events = self.raw_data[self.raw_data['event_type'] == 'search'].copy()
        click_events = self.raw_data[self.raw_data['event_type'] == 'click'].copy()
        
        query_metrics = []
        
        for _, query_event in search_events.iterrows():
            session_id = query_event['session_id']
            query_index = query_event['query_index_in_session']
            
            # Find clicks for this specific query
            query_clicks = click_events[
                (click_events['session_id'] == session_id) & 
                (click_events['query_index_in_session'] == query_index)
            ]
            
            # Calculate metrics
            num_clicks = len(query_clicks)
            has_click = num_clicks > 0
            
            # Position metrics
            if not query_clicks.empty:
                avg_click_position = query_clicks['click_position'].mean()
                first_click_position = query_clicks['click_position'].min()
                avg_time_on_result = query_clicks['time_on_result_seconds'].mean()
            else:
                avg_click_position = None
                first_click_position = None
                avg_time_on_result = None
            
            # Satisfaction metrics
            satisfied_clicks = query_clicks[query_clicks['is_satisfied'] == True] if not query_clicks.empty else pd.DataFrame()
            has_satisfied_click = len(satisfied_clicks) > 0
            
            query_metric = {
                'session_id': session_id,
                'user_id': query_event['user_id'],
                'query': query_event['query'],
                'query_index_in_session': query_index,
                'user_segment': query_event['user_segment'],
                'device_type': query_event['device_type'],
                'timestamp': query_event['timestamp'],
                'query_length': query_event['query_length'],
                'query_words': query_event['query_words'],
                'results_returned': query_event['results_returned'],
                'search_latency_ms': query_event['search_latency_ms'],
                'num_clicks': num_clicks,
                'has_click': has_click,
                'avg_click_position': avg_click_position,
                'first_click_position': first_click_position,
                'avg_time_on_result': avg_time_on_result,
                'has_satisfied_click': has_satisfied_click,
                'date': query_event['timestamp'].date()
            }
            query_metrics.append(query_metric)
        
        query_df = pd.DataFrame(query_metrics)
        self.processed_metrics['queries'] = query_df
        logger.info(f"Processed {len(query_df)} queries")
        
        return query_df
    
    def process_daily_aggregates(self) -> pd.DataFrame:
        """
        Calculate daily aggregated metrics.
        
        Returns:
            DataFrame with daily metrics
        """
        if 'queries' not in self.processed_metrics:
            self.process_query_metrics()
        
        logger.info("Processing daily aggregates...")
        
        query_df = self.processed_metrics['queries']
        
        daily_metrics = query_df.groupby('date').agg({
            'session_id': 'nunique',  # unique sessions
            'user_id': 'nunique',     # unique users
            'query': 'count',         # total queries
            'has_click': 'mean',      # click-through rate
            'has_satisfied_click': 'mean',  # satisfaction rate
            'search_latency_ms': 'mean',    # avg search latency
            'query_length': 'mean',         # avg query length
            'first_click_position': 'mean', # avg first click position
            'avg_time_on_result': 'mean'    # avg time on result
        }).round(4)
        
        daily_metrics.columns = [
            'unique_sessions', 'unique_users', 'total_queries',
            'click_through_rate', 'satisfaction_rate', 'avg_search_latency_ms',
            'avg_query_length', 'avg_first_click_position', 'avg_time_on_result'
        ]
        
        daily_metrics = daily_metrics.reset_index()
        self.processed_metrics['daily'] = daily_metrics
        logger.info(f"Processed {len(daily_metrics)} daily aggregates")
        
        return daily_metrics
    
    def process_user_segment_metrics(self) -> pd.DataFrame:
        """
        Calculate metrics broken down by user segment.
        
        Returns:
            DataFrame with user segment metrics
        """
        if 'queries' not in self.processed_metrics:
            self.process_query_metrics()
        
        logger.info("Processing user segment metrics...")
        
        query_df = self.processed_metrics['queries']
        
        segment_metrics = query_df.groupby(['user_segment', 'date']).agg({
            'session_id': 'nunique',
            'user_id': 'nunique',
            'query': 'count',
            'has_click': 'mean',
            'has_satisfied_click': 'mean',
            'search_latency_ms': 'mean',
            'query_length': 'mean',
            'first_click_position': 'mean'
        }).round(4)
        
        segment_metrics.columns = [
            'unique_sessions', 'unique_users', 'total_queries',
            'click_through_rate', 'satisfaction_rate', 'avg_search_latency_ms',
            'avg_query_length', 'avg_first_click_position'
        ]
        
        segment_metrics = segment_metrics.reset_index()
        self.processed_metrics['user_segments'] = segment_metrics
        logger.info(f"Processed {len(segment_metrics)} user segment metrics")
        
        return segment_metrics
    
    def calculate_search_quality_score(self) -> pd.DataFrame:
        """
        Calculate a composite search quality score based on multiple metrics.
        
        Returns:
            DataFrame with search quality scores by date
        """
        if 'daily' not in self.processed_metrics:
            self.process_daily_aggregates()
        
        daily_df = self.processed_metrics['daily'].copy()
        
        # Normalize metrics to 0-1 scale for scoring
        # Higher values are better for these metrics
        positive_metrics = ['click_through_rate', 'satisfaction_rate', 'avg_time_on_result']
        # Lower values are better for these metrics
        negative_metrics = ['avg_search_latency_ms', 'avg_first_click_position']
        
        # Calculate normalized scores
        for metric in positive_metrics:
            if metric in daily_df.columns:
                daily_df[f'{metric}_score'] = daily_df[metric] / daily_df[metric].max()
        
        for metric in negative_metrics:
            if metric in daily_df.columns:
                daily_df[f'{metric}_score'] = 1 - (daily_df[metric] / daily_df[metric].max())
        
        # Calculate composite score (weighted average)
        weights = {
            'click_through_rate_score': 0.3,
            'satisfaction_rate_score': 0.4,
            'avg_search_latency_ms_score': 0.1,
            'avg_first_click_position_score': 0.1,
            'avg_time_on_result_score': 0.1
        }
        
        daily_df['search_quality_score'] = 0
        for metric, weight in weights.items():
            if metric in daily_df.columns:
                daily_df['search_quality_score'] += daily_df[metric] * weight
        
        # Scale to 0-100
        daily_df['search_quality_score'] *= 100
        
        quality_df = daily_df[['date', 'search_quality_score']].copy()
        self.processed_metrics['quality_scores'] = quality_df
        
        return quality_df
    
    def _calculate_avg_time_to_click(self, search_events: pd.DataFrame, 
                                   click_events: pd.DataFrame) -> float:
        """
        Calculate average time from search to first click in a session.
        
        Args:
            search_events: Search events for the session
            click_events: Click events for the session
            
        Returns:
            Average time to click in seconds
        """
        if click_events.empty:
            return None
        
        times_to_click = []
        
        for _, search in search_events.iterrows():
            search_time = search['timestamp']
            query_index = search['query_index_in_session']
            
            # Find first click for this query
            query_clicks = click_events[
                click_events['query_index_in_session'] == query_index
            ].sort_values('timestamp')
            
            if not query_clicks.empty:
                first_click_time = query_clicks.iloc[0]['timestamp']
                time_to_click = (first_click_time - search_time).total_seconds()
                times_to_click.append(time_to_click)
        
        return np.mean(times_to_click) if times_to_click else None
    
    def save_processed_data(self, output_dir: str = "data/processed/"):
        """
        Save all processed metrics to CSV files.
        
        Args:
            output_dir: Directory to save processed data files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for metric_name, metric_df in self.processed_metrics.items():
            file_path = output_path / f"{metric_name}_metrics.csv"
            metric_df.to_csv(file_path, index=False)
            logger.info(f"Saved {metric_name} metrics to {file_path}")
    
    def get_summary_statistics(self) -> Dict:
        """
        Get summary statistics for the processed data.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.processed_metrics:
            return {"error": "No processed metrics available"}
        
        summary = {}
        
        if 'queries' in self.processed_metrics:
            query_df = self.processed_metrics['queries']
            summary['total_queries'] = len(query_df)
            summary['unique_sessions'] = query_df['session_id'].nunique()
            summary['unique_users'] = query_df['user_id'].nunique()
            summary['overall_ctr'] = query_df['has_click'].mean()
            summary['overall_satisfaction'] = query_df['has_satisfied_click'].mean()
            summary['avg_search_latency'] = query_df['search_latency_ms'].mean()
        
        if 'daily' in self.processed_metrics:
            daily_df = self.processed_metrics['daily']
            summary['date_range'] = {
                'start': daily_df['date'].min().strftime('%Y-%m-%d'),
                'end': daily_df['date'].max().strftime('%Y-%m-%d')
            }
        
        return summary

if __name__ == "__main__":
    # Example usage
    processor = SearchDataProcessor("data/raw/synthetic_search_logs.csv")
    
    # Load and process data
    processor.load_data()
    processor.process_session_metrics()
    processor.process_query_metrics()
    processor.process_daily_aggregates()
    processor.process_user_segment_metrics()
    processor.calculate_search_quality_score()
    
    # Save processed data
    processor.save_processed_data()
    
    # Print summary
    summary = processor.get_summary_statistics()
    print("Data Processing Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
