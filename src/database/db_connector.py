"""
Database connector for Search Quality Metrics Dashboard.

This module provides database connection and query functionality for the dashboard.
"""

import os
import pandas as pd
import sqlite3
import logging
from typing import Dict, List, Any, Optional, Union
from sqlalchemy import create_engine, text
from pathlib import Path
import yaml

# Set up logging
logger = logging.getLogger(__name__)

class DBConnector:
    """Database connector for search metrics data."""
    
    def __init__(self, config: Dict = None):
        """Initialize with configuration."""
        self.config = config or {}
        
        # Get database settings
        db_config = self.config.get('database', {})
        self.db_type = db_config.get('type', 'sqlite')
        
        if self.db_type == 'sqlite':
            self.db_path = db_config.get('sqlite_path', 'data/search_metrics.db')
            self.connection_string = f'sqlite:///{self.db_path}'
        elif self.db_type == 'postgresql':
            pg_config = db_config.get('postgresql', {})
            host = pg_config.get('host', 'localhost')
            port = pg_config.get('port', 5432)
            dbname = pg_config.get('dbname', 'search_metrics')
            user = pg_config.get('user', 'postgres')
            password = pg_config.get('password', '')
            
            self.connection_string = f'postgresql://{user}:{password}@{host}:{port}/{dbname}'
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
        
        # Create engine
        self.engine = None
    
    def connect(self):
        """Establish database connection."""
        logger.info(f"Connecting to {self.db_type} database")
        
        try:
            self.engine = create_engine(self.connection_string)
            logger.info("Database connection established")
            return True
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            return False
    
    def execute_query(self, query: str, params: Dict = None) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            DataFrame with query results
        """
        if self.engine is None:
            self.connect()
        
        try:
            if params:
                result = pd.read_sql_query(text(query), self.engine, params=params)
            else:
                result = pd.read_sql_query(query, self.engine)
            
            return result
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            logger.error(f"Query: {query}")
            if params:
                logger.error(f"Parameters: {params}")
            
            return pd.DataFrame()
    
    def get_table_names(self) -> List[str]:
        """
        Get list of tables in the database.
        
        Returns:
            List of table names
        """
        if self.engine is None:
            self.connect()
        
        try:
            if self.db_type == 'sqlite':
                query = "SELECT name FROM sqlite_master WHERE type='table'"
            elif self.db_type == 'postgresql':
                query = "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
            else:
                return []
            
            result = pd.read_sql_query(query, self.engine)
            
            if 'name' in result.columns:
                return result['name'].tolist()
            elif 'table_name' in result.columns:
                return result['table_name'].tolist()
            else:
                return []
        except Exception as e:
            logger.error(f"Error getting table names: {str(e)}")
            return []
    
    def get_table_schema(self, table_name: str) -> List[Dict[str, str]]:
        """
        Get schema for a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            List of column definitions
        """
        if self.engine is None:
            self.connect()
        
        try:
            if self.db_type == 'sqlite':
                query = f"PRAGMA table_info({table_name})"
                result = pd.read_sql_query(query, self.engine)
                
                columns = []
                for _, row in result.iterrows():
                    columns.append({
                        'name': row['name'],
                        'type': row['type'],
                        'nullable': not bool(row['notnull']),
                        'primary_key': bool(row['pk'])
                    })
                
                return columns
            elif self.db_type == 'postgresql':
                query = """
                SELECT column_name, data_type, is_nullable, 
                       (SELECT count(*) FROM information_schema.key_column_usage 
                        WHERE table_name = c.table_name AND column_name = c.column_name) > 0 as is_primary
                FROM information_schema.columns c
                WHERE table_name = :table_name
                """
                result = pd.read_sql_query(text(query), self.engine, params={'table_name': table_name})
                
                columns = []
                for _, row in result.iterrows():
                    columns.append({
                        'name': row['column_name'],
                        'type': row['data_type'],
                        'nullable': row['is_nullable'] == 'YES',
                        'primary_key': bool(row['is_primary'])
                    })
                
                return columns
            else:
                return []
        except Exception as e:
            logger.error(f"Error getting schema for table {table_name}: {str(e)}")
            return []
    
    def get_recent_queries(self, limit: int = 100) -> pd.DataFrame:
        """
        Get most recent search queries.
        
        Args:
            limit: Maximum number of queries to return
            
        Returns:
            DataFrame with recent queries
        """
        query = f"""
        SELECT query_id, query_text, timestamp, num_clicks, device_type, query_type
        FROM queries
        ORDER BY timestamp DESC
        LIMIT {limit}
        """
        
        return self.execute_query(query)
    
    def get_daily_metrics(self, metric_type: str, days: int = 30) -> pd.DataFrame:
        """
        Get daily metrics of a specific type.
        
        Args:
            metric_type: Type of metric (ctr, time_to_click, etc.)
            days: Number of days to include
            
        Returns:
            DataFrame with daily metrics
        """
        query = f"""
        SELECT date, ctr, count
        FROM metric_ctr_daily_ctr
        ORDER BY date DESC
        LIMIT {days}
        """
        
        if metric_type == 'ctr':
            query = f"""
            SELECT date, ctr, count
            FROM metric_ctr_daily_ctr
            ORDER BY date DESC
            LIMIT {days}
            """
        elif metric_type == 'time_to_click':
            query = f"""
            SELECT date, mean, median, count
            FROM metric_time_to_click_daily_time_to_click
            ORDER BY date DESC
            LIMIT {days}
            """
        elif metric_type == 'mrr':
            query = f"""
            SELECT date, mean, count
            FROM metric_mrr_daily_mrr
            ORDER BY date DESC
            LIMIT {days}
            """
        elif metric_type == 'ndcg':
            query = f"""
            SELECT date, ndcg_5
            FROM metric_ndcg_daily_ndcg_5
            ORDER BY date DESC
            LIMIT {days}
            """
        elif metric_type == 'reformulation':
            query = f"""
            SELECT date, reformulation_rate, count
            FROM metric_reformulation_daily_reformulation
            ORDER BY date DESC
            LIMIT {days}
            """
        
        return self.execute_query(query)
    
    def get_metric_by_device(self, metric_type: str) -> pd.DataFrame:
        """
        Get metrics broken down by device type.
        
        Args:
            metric_type: Type of metric (ctr, time_to_click, etc.)
            
        Returns:
            DataFrame with metrics by device
        """
        if metric_type == 'ctr':
            query = """
            SELECT device_type, ctr, count
            FROM metric_ctr_device_ctr
            """
        elif metric_type == 'time_to_click':
            query = """
            SELECT device_type, mean, median, count
            FROM metric_time_to_click_device_time_to_click
            """
        elif metric_type == 'mrr':
            query = """
            SELECT device_type, mean, count
            FROM metric_mrr_device_mrr
            """
        elif metric_type == 'ndcg':
            query = """
            SELECT device_type, ndcg_5
            FROM metric_ndcg_device_ndcg_5
            """
        elif metric_type == 'reformulation':
            query = """
            SELECT device_type, reformulation_rate, count
            FROM metric_reformulation_device_reformulation
            """
        else:
            return pd.DataFrame()
        
        return self.execute_query(query)
    
    def get_metric_by_query_type(self, metric_type: str) -> pd.DataFrame:
        """
        Get metrics broken down by query type.
        
        Args:
            metric_type: Type of metric (ctr, time_to_click, etc.)
            
        Returns:
            DataFrame with metrics by query type
        """
        if metric_type == 'ctr':
            query = """
            SELECT query_type, ctr, count
            FROM metric_ctr_query_type_ctr
            """
        elif metric_type == 'time_to_click':
            query = """
            SELECT query_type, mean, median, count
            FROM metric_time_to_click_query_type_time_to_click
            """
        elif metric_type == 'mrr':
            query = """
            SELECT query_type, mean, count
            FROM metric_mrr_query_type_mrr
            """
        elif metric_type == 'ndcg':
            query = """
            SELECT query_type, ndcg_5
            FROM metric_ndcg_query_type_ndcg_5
            """
        elif metric_type == 'reformulation':
            query = """
            SELECT query_type, reformulation_rate, count
            FROM metric_reformulation_query_type_reformulation
            """
        else:
            return pd.DataFrame()
        
        return self.execute_query(query)
    
    def get_sessions(self, limit: int = 100) -> pd.DataFrame:
        """
        Get recent search sessions.
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            DataFrame with session data
        """
        query = f"""
        SELECT session_id, user_id, device_type, start_time, end_time, 
               duration_seconds, num_queries, total_clicks, clicks_per_query,
               num_reformulations, success_score
        FROM sessions
        ORDER BY start_time DESC
        LIMIT {limit}
        """
        
        return self.execute_query(query)
    
    def get_session_details(self, session_id: str) -> Dict[str, Any]:
        """
        Get detailed information for a specific session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Dictionary with session details
        """
        # Get session metadata
        session_query = f"""
        SELECT *
        FROM sessions
        WHERE session_id = '{session_id}'
        """
        
        session_df = self.execute_query(session_query)
        
        if len(session_df) == 0:
            return {}
        
        # Get queries in this session
        queries_query = f"""
        SELECT *
        FROM queries
        WHERE session_id = '{session_id}'
        ORDER BY timestamp
        """
        
        queries_df = self.execute_query(queries_query)
        
        # Create result dictionary
        result = {
            'session': session_df.iloc[0].to_dict(),
            'queries': queries_df.to_dict('records')
        }
        
        return result
    
    def search_queries(self, search_term: str, limit: int = 100) -> pd.DataFrame:
        """
        Search for queries containing a specific term.
        
        Args:
            search_term: Search term
            limit: Maximum number of results to return
            
        Returns:
            DataFrame with matching queries
        """
        query = f"""
        SELECT query_id, query_text, timestamp, num_clicks, device_type, query_type
        FROM queries
        WHERE query_text LIKE '%{search_term}%'
        ORDER BY timestamp DESC
        LIMIT {limit}
        """
        
        return self.execute_query(query)
    
    def get_query_volume_by_hour(self, days: int = 7) -> pd.DataFrame:
        """
        Get query volume by hour of day.
        
        Args:
            days: Number of days to include in analysis
            
        Returns:
            DataFrame with query volume by hour
        """
        if self.db_type == 'sqlite':
            query = f"""
            SELECT strftime('%H', timestamp) as hour, COUNT(*) as volume
            FROM queries
            WHERE timestamp >= datetime('now', '-{days} days')
            GROUP BY hour
            ORDER BY hour
            """
        elif self.db_type == 'postgresql':
            query = f"""
            SELECT EXTRACT(HOUR FROM timestamp) as hour, COUNT(*) as volume
            FROM queries
            WHERE timestamp >= NOW() - INTERVAL '{days} days'
            GROUP BY hour
            ORDER BY hour
            """
        else:
            return pd.DataFrame()
        
        return self.execute_query(query)
    
    def get_query_volume_by_day(self, days: int = 30) -> pd.DataFrame:
        """
        Get query volume by day of week.
        
        Args:
            days: Number of days to include in analysis
            
        Returns:
            DataFrame with query volume by day
        """
        if self.db_type == 'sqlite':
            query = f"""
            SELECT strftime('%w', timestamp) as day, COUNT(*) as volume
            FROM queries
            WHERE timestamp >= datetime('now', '-{days} days')
            GROUP BY day
            ORDER BY day
            """
        elif self.db_type == 'postgresql':
            query = f"""
            SELECT EXTRACT(DOW FROM timestamp) as day, COUNT(*) as volume
            FROM queries
            WHERE timestamp >= NOW() - INTERVAL '{days} days'
            GROUP BY day
            ORDER BY day
            """
        else:
            return pd.DataFrame()
        
        result = self.execute_query(query)
        
        # Map day numbers to names
        day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        result['day_name'] = result['day'].astype(int).apply(lambda x: day_names[x])
        
        return result
    
    def get_top_queries(self, days: int = 30, limit: int = 20) -> pd.DataFrame:
        """
        Get most frequent search queries.
        
        Args:
            days: Number of days to include in analysis
            limit: Maximum number of results to return
            
        Returns:
            DataFrame with top queries
        """
        if self.db_type == 'sqlite':
            query = f"""
            SELECT query_text, COUNT(*) as frequency
            FROM queries
            WHERE timestamp >= datetime('now', '-{days} days')
            GROUP BY query_text
            ORDER BY frequency DESC
            LIMIT {limit}
            """
        elif self.db_type == 'postgresql':
            query = f"""
            SELECT query_text, COUNT(*) as frequency
            FROM queries
            WHERE timestamp >= NOW() - INTERVAL '{days} days'
            GROUP BY query_text
            ORDER BY frequency DESC
            LIMIT {limit}
            """
        else:
            return pd.DataFrame()
        
        return self.execute_query(query)
    
    def get_zero_click_queries(self, days: int = 30, limit: int = 20) -> pd.DataFrame:
        """
        Get most frequent queries with zero clicks.
        
        Args:
            days: Number of days to include in analysis
            limit: Maximum number of results to return
            
        Returns:
            DataFrame with zero-click queries
        """
        if self.db_type == 'sqlite':
            query = f"""
            SELECT query_text, COUNT(*) as frequency
            FROM queries
            WHERE timestamp >= datetime('now', '-{days} days')
              AND num_clicks = 0
            GROUP BY query_text
            ORDER BY frequency DESC
            LIMIT {limit}
            """
        elif self.db_type == 'postgresql':
            query = f"""
            SELECT query_text, COUNT(*) as frequency
            FROM queries
            WHERE timestamp >= NOW() - INTERVAL '{days} days'
              AND num_clicks = 0
            GROUP BY query_text
            ORDER BY frequency DESC
            LIMIT {limit}
            """
        else:
            return pd.DataFrame()
        
        return self.execute_query(query)
    
    def get_high_reformulation_queries(self, days: int = 30, limit: int = 20) -> pd.DataFrame:
        """
        Get queries that frequently lead to reformulations.
        
        Args:
            days: Number of days to include in analysis
            limit: Maximum number of results to return
            
        Returns:
            DataFrame with high-reformulation queries
        """
        if self.db_type == 'sqlite':
            query = f"""
            WITH query_pairs AS (
                SELECT q1.query_text as original_query,
                       q2.query_text as next_query,
                       q2.is_reformulation
                FROM queries q1
                JOIN queries q2 ON q1.session_id = q2.session_id
                                AND q2.timestamp > q1.timestamp
                                AND (julianday(q2.timestamp) - julianday(q1.timestamp)) * 86400 <= 600
                WHERE q1.timestamp >= datetime('now', '-{days} days')
            )
            SELECT original_query,
                   COUNT(*) as total_occurrences,
                   SUM(is_reformulation) as reformulation_count,
                   CAST(SUM(is_reformulation) AS FLOAT) / COUNT(*) as reformulation_rate
            FROM query_pairs
            GROUP BY original_query
            HAVING total_occurrences >= 5
            ORDER BY reformulation_rate DESC, total_occurrences DESC
            LIMIT {limit}
            """
        elif self.db_type == 'postgresql':
            query = f"""
            WITH query_pairs AS (
                SELECT q1.query_text as original_query,
                       q2.query_text as next_query,
                       q2.is_reformulation
                FROM queries q1
                JOIN queries q2 ON q1.session_id = q2.session_id
                                AND q2.timestamp > q1.timestamp
                                AND EXTRACT(EPOCH FROM (q2.timestamp - q1.timestamp)) <= 600
                WHERE q1.timestamp >= NOW() - INTERVAL '{days} days'
            )
            SELECT original_query,
                   COUNT(*) as total_occurrences,
                   SUM(is_reformulation) as reformulation_count,
                   CAST(SUM(is_reformulation) AS FLOAT) / COUNT(*) as reformulation_rate
            FROM query_pairs
            GROUP BY original_query
            HAVING COUNT(*) >= 5
            ORDER BY reformulation_rate DESC, total_occurrences DESC
            LIMIT {limit}
            """
        else:
            return pd.DataFrame()
        
        return self.execute_query(query)
    
    def get_query_performance_summary(self) -> pd.DataFrame:
        """
        Get overall performance summary by query type.
        
        Returns:
            DataFrame with query performance summary
        """
        query = """
        SELECT query_type,
               COUNT(*) as query_count,
               AVG(num_clicks > 0) as ctr,
               AVG(num_clicks) as avg_clicks,
               AVG(CASE WHEN reciprocal_rank > 0 THEN reciprocal_rank ELSE NULL END) as mrr,
               AVG(ndcg_5) as ndcg
        FROM queries
        GROUP BY query_type
        """
        
        return self.execute_query(query)
    
    def get_session_metrics_summary(self) -> pd.DataFrame:
        """
        Get summary of session metrics.
        
        Returns:
            DataFrame with session metrics summary
        """
        query = """
        SELECT device_type,
               COUNT(*) as session_count,
               AVG(duration_seconds) as avg_duration,
               AVG(num_queries) as avg_queries,
               AVG(total_clicks) as avg_clicks,
               AVG(num_reformulations) as avg_reformulations,
               AVG(abandoned) as abandonment_rate,
               AVG(success_score) / 3.0 as success_rate
        FROM sessions
        GROUP BY device_type
        """
        
        return self.execute_query(query)
    
    def close(self):
        """Close database connection."""
        if self.engine is not None:
            self.engine.dispose()
            self.engine = None
            logger.info("Database connection closed")
