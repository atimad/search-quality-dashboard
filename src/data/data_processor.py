"""
Data Processor for Search Quality Metrics Dashboard.

This module processes raw search logs into structured metrics datasets
that can be used for dashboard visualization and analysis.
"""

import os
import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from datetime import datetime, timedelta
import sqlite3
from sqlalchemy import create_engine
import logging
from tqdm import tqdm
import difflib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SearchDataProcessor:
    """Process raw search logs into metrics for analysis and visualization."""
    
    def __init__(self, config_path=None):
        """Initialize the data processor with configuration."""
        self.config = self._load_config(config_path)
        self.raw_data_path = Path(self.config.get('raw_data_path', 'data/raw/synthetic_search_logs.csv'))
        self.processed_data_dir = Path(self.config.get('processed_data_dir', 'data/processed/'))
        self.db_path = self.config.get('db_path', 'data/search_metrics.db')
        self.session_timeout = self.config.get('session_timeout', 1800)  # 30 minutes in seconds
        
        # Create processed data directory if it doesn't exist
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
    def _load_config(self, config_path):
        """Load configuration from YAML file or use defaults."""
        default_config = {
            'raw_data_path': 'data/raw/synthetic_search_logs.csv',
            'processed_data_dir': 'data/processed/',
            'db_path': 'data/search_metrics.db',
            'session_timeout': 1800,  # 30 minutes in seconds
            'min_session_length': 2,
            'max_session_length': 50,
            'metrics': {
                'ctr': {'enabled': True},
                'time_to_click': {'enabled': True, 'max_time': 600, 'outlier_threshold': 0.95},
                'ndcg': {'enabled': True, 'k_values': [3, 5, 10]},
                'reformulation': {'enabled': True, 'max_time_between_queries': 600}
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                # Merge with defaults for each section
                if 'metrics' in config and 'metrics' in default_config:
                    for metric, settings in default_config['metrics'].items():
                        if metric in config['metrics']:
                            settings.update(config['metrics'][metric])
                        config['metrics'][metric] = settings
                
                # Merge top-level configs
                default_config.update(config)
        
        return default_config
    
    def load_data(self):
        """Load raw search log data."""
        logger.info(f"Loading raw data from {self.raw_data_path}")
        
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Raw data file not found: {self.raw_data_path}")
        
        df = pd.read_csv(self.raw_data_path)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Parse JSON columns
        json_columns = ['results', 'result_relevance', 'clicked_results', 'click_timestamps']
        
        for col in json_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        logger.info(f"Loaded {len(df)} search log entries")
        return df
    
    def process_data(self):
        """Process raw data into structured metrics datasets."""
        df = self.load_data()
        
        # Process sessions and queries
        sessions_df = self._process_sessions(df)
        queries_df = self._process_queries(df)
        
        # Calculate metrics
        metrics = {}
        
        # Click-through rate metrics
        if self.config['metrics']['ctr']['enabled']:
            metrics['ctr'] = self._calculate_ctr_metrics(df)
        
        # Time to click metrics
        if self.config['metrics']['time_to_click']['enabled']:
            metrics['time_to_click'] = self._calculate_time_to_click_metrics(df)
        
        # NDCG metrics
        if self.config['metrics']['ndcg']['enabled']:
            metrics['ndcg'] = self._calculate_ndcg_metrics(df)
        
        # Query reformulation metrics
        if self.config['metrics']['reformulation']['enabled']:
            metrics['reformulation'] = self._calculate_reformulation_metrics(df)
        
        # Save to SQLite database
        self._save_to_database(df, sessions_df, queries_df, metrics)
        
        # Save to CSV files
        self._save_to_csv(df, sessions_df, queries_df, metrics)
        
        return {
            'raw_data': df,
            'sessions': sessions_df,
            'queries': queries_df,
            'metrics': metrics
        }
    
    def _process_sessions(self, df):
        """Process raw data into session-level metrics."""
        logger.info("Processing session-level metrics")
        
        # Group by session_id
        session_groups = df.groupby('session_id')
        
        sessions_data = []
        
        for session_id, session_df in tqdm(session_groups):
            # Sort by timestamp within session
            session_df = session_df.sort_values('timestamp')
            
            # Basic session metrics
            start_time = session_df['timestamp'].min()
            end_time = session_df['timestamp'].max()
            duration = (end_time - start_time).total_seconds()
            
            # Get the user_id (should be the same for all queries in the session)
            user_id = session_df['user_id'].iloc[0]
            
            # Get device type and location (should be the same for a session)
            device_type = session_df['device_type'].iloc[0]
            location = session_df['location'].iloc[0]
            
            # Count queries and clicks
            num_queries = len(session_df)
            total_clicks = session_df['num_clicks'].sum()
            
            # Check if any query had clicks
            has_clicks = total_clicks > 0
            
            # Count reformulations
            num_reformulations = session_df['is_reformulation'].sum()
            
            # Query diversity: count unique queries
            unique_queries = len(session_df['query_text'].unique())
            query_diversity = unique_queries / num_queries if num_queries > 0 else 0
            
            # Session satisfaction proxies
            # 1. Click-based: did they click on results?
            # 2. Reformulation-based: did they need to reformulate a lot?
            # 3. Abandonment: did they leave without clicking?
            
            # Click satisfaction (simple proxy: clicks per query)
            clicks_per_query = total_clicks / num_queries if num_queries > 0 else 0
            
            # Reformulation ratio (lower is better)
            reformulation_ratio = num_reformulations / (num_queries - 1) if num_queries > 1 else 0
            
            # Abandonment (no clicks in the session)
            abandoned = not has_clicks
            
            # Success metrics
            # - Success 1: Clicked on a result
            # - Success 2: Didn't need many reformulations
            # - Success 3: Session ended with a click (didn't abandon after last query)
            
            success_clicked = has_clicks
            success_low_reformulation = reformulation_ratio < 0.5  # Threshold: less than 50% of queries were reformulations
            success_ended_with_click = session_df.iloc[-1]['num_clicks'] > 0  # Last query had clicks
            
            # Overall success score (1-3 points)
            success_score = sum([
                success_clicked,
                success_low_reformulation,
                success_ended_with_click
            ])
            
            # Query patterns
            query_types = session_df['query_type'].value_counts().to_dict()
            
            # Create session record
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
                'clicks_per_query': clicks_per_query,
                'num_reformulations': num_reformulations,
                'reformulation_ratio': reformulation_ratio,
                'query_diversity': query_diversity,
                'abandoned': abandoned,
                'success_clicked': success_clicked,
                'success_low_reformulation': success_low_reformulation,
                'success_ended_with_click': success_ended_with_click,
                'success_score': success_score,
                'query_type_navigational': query_types.get('navigational', 0),
                'query_type_informational': query_types.get('informational', 0),
                'query_type_transactional': query_types.get('transactional', 0),
                'date': start_time.date()
            }
            
            sessions_data.append(session_record)
        
        # Create DataFrame
        sessions_df = pd.DataFrame(sessions_data)
        
        logger.info(f"Processed {len(sessions_df)} sessions")
        return sessions_df
    
    def _process_queries(self, df):
        """Process raw data into query-level metrics."""
        logger.info("Processing query-level metrics")
        
        # Clone the dataframe to avoid modifying the original
        queries_df = df.copy()
        
        # Parse JSON columns and extract metrics
        queries_df['num_results'] = queries_df['results'].apply(len)
        queries_df['first_click_position'] = queries_df.apply(
            lambda row: self._get_first_click_position(row), axis=1
        )
        queries_df['time_to_first_click'] = queries_df.apply(
            lambda row: self._get_time_to_first_click(row), axis=1
        )
        
        # Calculate MRR (Mean Reciprocal Rank)
        queries_df['reciprocal_rank'] = queries_df['first_click_position'].apply(
            lambda pos: 1.0 / pos if pos > 0 else 0
        )
        
        # Calculate NDCG for different k values
        k_values = self.config['metrics']['ndcg']['k_values']
        
        for k in k_values:
            queries_df[f'ndcg_{k}'] = queries_df.apply(
                lambda row: self._calculate_ndcg(row, k), axis=1
            )
        
        # Add date column for easier time-based analysis
        queries_df['date'] = queries_df['timestamp'].dt.date
        
        # Clean up redundant columns for storage efficiency
        columns_to_drop = ['results', 'result_relevance', 'clicked_results', 'click_timestamps']
        queries_df = queries_df.drop(columns=columns_to_drop)
        
        logger.info(f"Processed {len(queries_df)} queries")
        return queries_df
    
    def _get_first_click_position(self, row):
        """Get the position of the first clicked result (1-indexed)."""
        clicked_results = row['clicked_results']
        
        if not clicked_results:
            return 0  # No clicks
        
        results = row['results']
        
        # Find position of first clicked result (1-indexed)
        first_clicked = clicked_results[0]
        
        try:
            position = results.index(first_clicked) + 1
            return position
        except ValueError:
            return 0  # Clicked result not found in results (shouldn't happen)
    
    def _get_time_to_first_click(self, row):
        """Get the time to first click in seconds."""
        clicked_results = row['clicked_results']
        click_timestamps = row['click_timestamps']
        
        if not clicked_results or not click_timestamps:
            return None  # No clicks
        
        # Convert timestamps to datetime
        query_time = row['timestamp']
        first_click_time = pd.to_datetime(click_timestamps[0])
        
        # Calculate time difference in seconds
        time_to_click = (first_click_time - query_time).total_seconds()
        
        # Apply maximum time constraint
        max_time = self.config['metrics']['time_to_click']['max_time']
        
        if time_to_click > max_time:
            return None  # Outlier, discard
        
        return time_to_click
    
    def _calculate_ndcg(self, row, k):
        """Calculate NDCG@k for a query."""
        results = row['results']
        relevance_scores = row['result_relevance']
        clicked_results = row['clicked_results']
        
        if not results or len(results) == 0:
            return 0.0
        
        # Use only the top k results
        results = results[:k]
        
        # Get relevance for each result
        rel = [relevance_scores.get(r, 0.0) for r in results]
        
        # Create ideal ordering (sort by relevance)
        ideal_rel = sorted(rel, reverse=True)
        
        # Calculate DCG and IDCG
        dcg = self._calculate_dcg(rel)
        idcg = self._calculate_dcg(ideal_rel)
        
        # Calculate NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        return ndcg
    
    def _calculate_dcg(self, relevance_scores):
        """Calculate DCG (Discounted Cumulative Gain)."""
        dcg = 0.0
        
        for i, rel in enumerate(relevance_scores):
            # i is 0-indexed, but we need 1-indexed positions for the formula
            pos = i + 1
            dcg += (2 ** rel - 1) / np.log2(pos + 1)
        
        return dcg
    
    def _calculate_ctr_metrics(self, df):
        """Calculate click-through rate metrics."""
        logger.info("Calculating CTR metrics")
        
        # Overall CTR
        total_queries = len(df)
        queries_with_clicks = df[df['num_clicks'] > 0].shape[0]
        overall_ctr = queries_with_clicks / total_queries if total_queries > 0 else 0
        
        # CTR by day
        df['date'] = df['timestamp'].dt.date
        daily_ctr = df.groupby('date').apply(
            lambda x: x[x['num_clicks'] > 0].shape[0] / x.shape[0] if x.shape[0] > 0 else 0
        ).reset_index()
        daily_ctr.columns = ['date', 'ctr']
        
        # CTR by device type
        device_ctr = df.groupby('device_type').apply(
            lambda x: x[x['num_clicks'] > 0].shape[0] / x.shape[0] if x.shape[0] > 0 else 0
        ).reset_index()
        device_ctr.columns = ['device_type', 'ctr']
        
        # CTR by query type
        query_type_ctr = df.groupby('query_type').apply(
            lambda x: x[x['num_clicks'] > 0].shape[0] / x.shape[0] if x.shape[0] > 0 else 0
        ).reset_index()
        query_type_ctr.columns = ['query_type', 'ctr']
        
        # CTR by location
        location_ctr = df.groupby('location').apply(
            lambda x: x[x['num_clicks'] > 0].shape[0] / x.shape[0] if x.shape[0] > 0 else 0
        ).reset_index()
        location_ctr.columns = ['location', 'ctr']
        
        # Zero-click rate (ZCR)
        zcr = 1 - overall_ctr
        
        # Return metrics
        return {
            'overall_ctr': overall_ctr,
            'zero_click_rate': zcr,
            'daily_ctr': daily_ctr,
            'device_ctr': device_ctr,
            'query_type_ctr': query_type_ctr,
            'location_ctr': location_ctr
        }
    
    def _calculate_time_to_click_metrics(self, df):
        """Calculate time to click metrics."""
        logger.info("Calculating time to click metrics")
        
        # Filter out queries with no clicks or invalid time_to_first_click
        click_df = df[df['time_to_first_click'].notna()].copy()
        
        if len(click_df) == 0:
            logger.warning("No valid click data found for time to click metrics")
            return {
                'mean_time_to_click': None,
                'median_time_to_click': None,
                'daily_time_to_click': pd.DataFrame(),
                'device_time_to_click': pd.DataFrame(),
                'query_type_time_to_click': pd.DataFrame()
            }
        
        # Overall time to click
        mean_ttc = click_df['time_to_first_click'].mean()
        median_ttc = click_df['time_to_first_click'].median()
        
        # Time to click by day
        click_df['date'] = click_df['timestamp'].dt.date
        daily_ttc = click_df.groupby('date')['time_to_first_click'].agg(['mean', 'median', 'count']).reset_index()
        
        # Time to click by device type
        device_ttc = click_df.groupby('device_type')['time_to_first_click'].agg(['mean', 'median', 'count']).reset_index()
        
        # Time to click by query type
        query_type_ttc = click_df.groupby('query_type')['time_to_first_click'].agg(['mean', 'median', 'count']).reset_index()
        
        # Return metrics
        return {
            'mean_time_to_click': mean_ttc,
            'median_time_to_click': median_ttc,
            'daily_time_to_click': daily_ttc,
            'device_time_to_click': device_ttc,
            'query_type_time_to_click': query_type_ttc
        }
    
    def _calculate_ndcg_metrics(self, df):
        """Calculate NDCG metrics."""
        logger.info("Calculating NDCG metrics")
        
        k_values = self.config['metrics']['ndcg']['k_values']
        metrics = {}
        
        for k in k_values:
            col_name = f'ndcg_{k}'
            
            # Overall NDCG@k
            overall_ndcg = df[col_name].mean()
            metrics[f'overall_ndcg_{k}'] = overall_ndcg
            
            # NDCG@k by day
            df['date'] = df['timestamp'].dt.date
            daily_ndcg = df.groupby('date')[col_name].mean().reset_index()
            daily_ndcg.columns = ['date', f'ndcg_{k}']
            metrics[f'daily_ndcg_{k}'] = daily_ndcg
            
            # NDCG@k by device type
            device_ndcg = df.groupby('device_type')[col_name].mean().reset_index()
            device_ndcg.columns = ['device_type', f'ndcg_{k}']
            metrics[f'device_ndcg_{k}'] = device_ndcg
            
            # NDCG@k by query type
            query_type_ndcg = df.groupby('query_type')[col_name].mean().reset_index()
            query_type_ndcg.columns = ['query_type', f'ndcg_{k}']
            metrics[f'query_type_ndcg_{k}'] = query_type_ndcg
        
        # Mean Reciprocal Rank (MRR)
        overall_mrr = df['reciprocal_rank'].mean()
        metrics['overall_mrr'] = overall_mrr
        
        # MRR by day
        daily_mrr = df.groupby('date')['reciprocal_rank'].mean().reset_index()
        daily_mrr.columns = ['date', 'mrr']
        metrics['daily_mrr'] = daily_mrr
        
        # MRR by device type
        device_mrr = df.groupby('device_type')['reciprocal_rank'].mean().reset_index()
        device_mrr.columns = ['device_type', 'mrr']
        metrics['device_mrr'] = device_mrr
        
        # MRR by query type
        query_type_mrr = df.groupby('query_type')['reciprocal_rank'].mean().reset_index()
        query_type_mrr.columns = ['query_type', 'mrr']
        metrics['query_type_mrr'] = query_type_mrr
        
        return metrics
    
    def _calculate_reformulation_metrics(self, df):
        """Calculate query reformulation metrics."""
        logger.info("Calculating query reformulation metrics")
        
        # Overall reformulation rate
        total_queries = len(df)
        reformulations = df['is_reformulation'].sum()
        overall_reformulation_rate = reformulations / total_queries if total_queries > 0 else 0
        
        # Reformulation rate by day
        df['date'] = df['timestamp'].dt.date
        daily_reformulation = df.groupby('date').apply(
            lambda x: x['is_reformulation'].sum() / x.shape[0] if x.shape[0] > 0 else 0
        ).reset_index()
        daily_reformulation.columns = ['date', 'reformulation_rate']
        
        # Reformulation rate by device type
        device_reformulation = df.groupby('device_type').apply(
            lambda x: x['is_reformulation'].sum() / x.shape[0] if x.shape[0] > 0 else 0
        ).reset_index()
        device_reformulation.columns = ['device_type', 'reformulation_rate']
        
        # Reformulation rate by query type
        query_type_reformulation = df.groupby('query_type').apply(
            lambda x: x['is_reformulation'].sum() / x.shape[0] if x.shape[0] > 0 else 0
        ).reset_index()
        query_type_reformulation.columns = ['query_type', 'reformulation_rate']
        
        # Count queries that led to reformulation
        # A query leads to reformulation if the next query in the same session is a reformulation
        # To do this, we need to sort by session_id and timestamp
        sorted_df = df.sort_values(['session_id', 'timestamp'])
        
        # Create a column indicating if the next query is a reformulation
        sorted_df['next_is_reformulation'] = sorted_df.groupby('session_id')['is_reformulation'].shift(-1).fillna(0)
        
        # Calculate the percentage of queries that led to reformulation
        led_to_reformulation = sorted_df['next_is_reformulation'].sum()
        led_to_reformulation_rate = led_to_reformulation / total_queries if total_queries > 0 else 0
        
        # Return metrics
        return {
            'overall_reformulation_rate': overall_reformulation_rate,
            'led_to_reformulation_rate': led_to_reformulation_rate,
            'daily_reformulation': daily_reformulation,
            'device_reformulation': device_reformulation,
            'query_type_reformulation': query_type_reformulation
        }
    
    def _save_to_database(self, raw_df, sessions_df, queries_df, metrics):
        """Save processed data to SQLite database."""
        logger.info(f"Saving data to SQLite database: {self.db_path}")
        
        # Create database connection
        engine = create_engine(f'sqlite:///{self.db_path}')
        
        # Save raw data (sample to save space)
        if len(raw_df) > 10000:
            sample_size = min(10000, int(len(raw_df) * 0.1))
            raw_sample = raw_df.sample(sample_size)
            
            # Convert JSON columns back to strings for storage
            for col in ['results', 'result_relevance', 'clicked_results', 'click_timestamps']:
                if col in raw_sample.columns:
                    raw_sample[col] = raw_sample[col].apply(lambda x: json.dumps(x) if x is not None else None)
            
            raw_sample.to_sql('raw_search_logs_sample', engine, index=False, if_exists='replace')
        else:
            # Convert JSON columns back to strings for storage
            raw_df_copy = raw_df.copy()
            for col in ['results', 'result_relevance', 'clicked_results', 'click_timestamps']:
                if col in raw_df_copy.columns:
                    raw_df_copy[col] = raw_df_copy[col].apply(lambda x: json.dumps(x) if x is not None else None)
            
            raw_df_copy.to_sql('raw_search_logs', engine, index=False, if_exists='replace')
        
        # Save sessions data
        sessions_df.to_sql('sessions', engine, index=False, if_exists='replace')
        
        # Save queries data
        queries_df.to_sql('queries', engine, index=False, if_exists='replace')
        
        # Save metrics
        for metric_type, metric_data in metrics.items():
            if isinstance(metric_data, dict):
                for name, data in metric_data.items():
                    if isinstance(data, pd.DataFrame):
                        data.to_sql(f'metric_{metric_type}_{name}', engine, index=False, if_exists='replace')
            elif isinstance(metric_data, pd.DataFrame):
                metric_data.to_sql(f'metric_{metric_type}', engine, index=False, if_exists='replace')
        
        logger.info("Database save complete")
    
    def _save_to_csv(self, raw_df, sessions_df, queries_df, metrics):
        """Save processed data to CSV files."""
        logger.info(f"Saving data to CSV files in {self.processed_data_dir}")
        
        # Save sessions data
        sessions_path = self.processed_data_dir / 'sessions.csv'
        sessions_df.to_csv(sessions_path, index=False)
        
        # Save queries data
        queries_path = self.processed_data_dir / 'queries.csv'
        queries_df.to_csv(queries_path, index=False)
        
        # Save metrics
        metrics_dir = self.processed_data_dir / 'metrics'
        os.makedirs(metrics_dir, exist_ok=True)
        
        for metric_type, metric_data in metrics.items():
            if isinstance(metric_data, dict):
                for name, data in metric_data.items():
                    if isinstance(data, pd.DataFrame):
                        path = metrics_dir / f'{metric_type}_{name}.csv'
                        data.to_csv(path, index=False)
            elif isinstance(metric_data, pd.DataFrame):
                path = metrics_dir / f'{metric_type}.csv'
                metric_data.to_csv(path, index=False)
        
        logger.info("CSV save complete")

def main():
    """Main function to process data when script is run directly."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process search log data into metrics')
    parser.add_argument('--config', type=str, help='Path to configuration YAML file')
    args = parser.parse_args()
    
    processor = SearchDataProcessor(config_path=args.config)
    processor.process_data()

if __name__ == "__main__":
    main()
