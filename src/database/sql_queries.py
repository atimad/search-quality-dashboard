"""
SQL Queries for Search Quality Metrics Dashboard.

This module provides SQL query templates for common metrics calculations.
These queries can be used across different database systems with minimal modification.
"""

from typing import Dict, List, Any

# Main table schema definitions - used for database setup
SCHEMA_DEFINITIONS = {
    'queries': """
        CREATE TABLE IF NOT EXISTS queries (
            query_id TEXT PRIMARY KEY,
            session_id TEXT,
            user_id TEXT,
            query_text TEXT,
            query_type TEXT,
            timestamp TIMESTAMP,
            num_results INTEGER,
            num_clicks INTEGER,
            first_click_position INTEGER,
            time_to_first_click REAL,
            reciprocal_rank REAL,
            ndcg_3 REAL,
            ndcg_5 REAL,
            ndcg_10 REAL,
            device_type TEXT,
            location TEXT,
            is_reformulation INTEGER,
            date DATE
        )
    """,
    
    'sessions': """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT,
            device_type TEXT,
            location TEXT,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            duration_seconds REAL,
            num_queries INTEGER,
            total_clicks INTEGER,
            clicks_per_query REAL,
            num_reformulations INTEGER,
            reformulation_ratio REAL,
            query_diversity REAL,
            abandoned INTEGER,
            success_clicked INTEGER,
            success_low_reformulation INTEGER,
            success_ended_with_click INTEGER,
            success_score INTEGER,
            query_type_navigational INTEGER,
            query_type_informational INTEGER,
            query_type_transactional INTEGER,
            date DATE
        )
    """
}

# Query templates by category
QUERY_TEMPLATES = {
    # CTR-related queries
    'ctr': {
        'overall': """
            SELECT COUNT(*) as total_queries,
                   SUM(CASE WHEN num_clicks > 0 THEN 1 ELSE 0 END) as queries_with_clicks,
                   CAST(SUM(CASE WHEN num_clicks > 0 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) as ctr
            FROM queries
            WHERE timestamp BETWEEN :start_date AND :end_date
        """,
        
        'daily': """
            SELECT date,
                   COUNT(*) as total_queries,
                   SUM(CASE WHEN num_clicks > 0 THEN 1 ELSE 0 END) as queries_with_clicks,
                   CAST(SUM(CASE WHEN num_clicks > 0 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) as ctr
            FROM queries
            WHERE timestamp BETWEEN :start_date AND :end_date
            GROUP BY date
            ORDER BY date
        """,
        
        'by_device': """
            SELECT device_type,
                   COUNT(*) as total_queries,
                   SUM(CASE WHEN num_clicks > 0 THEN 1 ELSE 0 END) as queries_with_clicks,
                   CAST(SUM(CASE WHEN num_clicks > 0 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) as ctr
            FROM queries
            WHERE timestamp BETWEEN :start_date AND :end_date
            GROUP BY device_type
        """,
        
        'by_query_type': """
            SELECT query_type,
                   COUNT(*) as total_queries,
                   SUM(CASE WHEN num_clicks > 0 THEN 1 ELSE 0 END) as queries_with_clicks,
                   CAST(SUM(CASE WHEN num_clicks > 0 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) as ctr
            FROM queries
            WHERE timestamp BETWEEN :start_date AND :end_date
            GROUP BY query_type
        """,
        
        'zero_click_rate': """
            SELECT COUNT(*) as total_queries,
                   SUM(CASE WHEN num_clicks = 0 THEN 1 ELSE 0 END) as zero_click_queries,
                   CAST(SUM(CASE WHEN num_clicks = 0 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) as zero_click_rate
            FROM queries
            WHERE timestamp BETWEEN :start_date AND :end_date
        """
    },
    
    # Time to click queries
    'time_to_click': {
        'overall': """
            SELECT AVG(time_to_first_click) as mean_ttc,
                   MIN(time_to_first_click) as min_ttc,
                   MAX(time_to_first_click) as max_ttc,
                   COUNT(*) as total_queries_with_clicks
            FROM queries
            WHERE timestamp BETWEEN :start_date AND :end_date
              AND time_to_first_click IS NOT NULL
        """,
        
        'daily': """
            SELECT date,
                   AVG(time_to_first_click) as mean_ttc,
                   COUNT(*) as total_queries_with_clicks
            FROM queries
            WHERE timestamp BETWEEN :start_date AND :end_date
              AND time_to_first_click IS NOT NULL
            GROUP BY date
            ORDER BY date
        """,
        
        'by_device': """
            SELECT device_type,
                   AVG(time_to_first_click) as mean_ttc,
                   COUNT(*) as total_queries_with_clicks
            FROM queries
            WHERE timestamp BETWEEN :start_date AND :end_date
              AND time_to_first_click IS NOT NULL
            GROUP BY device_type
        """,
        
        'by_query_type': """
            SELECT query_type,
                   AVG(time_to_first_click) as mean_ttc,
                   COUNT(*) as total_queries_with_clicks
            FROM queries
            WHERE timestamp BETWEEN :start_date AND :end_date
              AND time_to_first_click IS NOT NULL
            GROUP BY query_type
        """
    },
    
    # MRR and NDCG queries
    'relevance': {
        'mrr_overall': """
            SELECT AVG(reciprocal_rank) as mean_reciprocal_rank,
                   COUNT(*) as total_queries
            FROM queries
            WHERE timestamp BETWEEN :start_date AND :end_date
        """,
        
        'mrr_daily': """
            SELECT date,
                   AVG(reciprocal_rank) as mean_reciprocal_rank,
                   COUNT(*) as total_queries
            FROM queries
            WHERE timestamp BETWEEN :start_date AND :end_date
            GROUP BY date
            ORDER BY date
        """,
        
        'ndcg_overall': """
            SELECT AVG(ndcg_3) as ndcg_3,
                   AVG(ndcg_5) as ndcg_5,
                   AVG(ndcg_10) as ndcg_10,
                   COUNT(*) as total_queries
            FROM queries
            WHERE timestamp BETWEEN :start_date AND :end_date
        """,
        
        'ndcg_by_query_type': """
            SELECT query_type,
                   AVG(ndcg_3) as ndcg_3,
                   AVG(ndcg_5) as ndcg_5,
                   AVG(ndcg_10) as ndcg_10,
                   COUNT(*) as total_queries
            FROM queries
            WHERE timestamp BETWEEN :start_date AND :end_date
            GROUP BY query_type
        """
    },
    
    # Query reformulation queries
    'reformulation': {
        'overall_rate': """
            SELECT COUNT(*) as total_queries,
                   SUM(is_reformulation) as reformulation_count,
                   CAST(SUM(is_reformulation) AS FLOAT) / COUNT(*) as reformulation_rate
            FROM queries
            WHERE timestamp BETWEEN :start_date AND :end_date
        """,
        
        'daily': """
            SELECT date,
                   COUNT(*) as total_queries,
                   SUM(is_reformulation) as reformulation_count,
                   CAST(SUM(is_reformulation) AS FLOAT) / COUNT(*) as reformulation_rate
            FROM queries
            WHERE timestamp BETWEEN :start_date AND :end_date
            GROUP BY date
            ORDER BY date
        """,
        
        'by_device': """
            SELECT device_type,
                   COUNT(*) as total_queries,
                   SUM(is_reformulation) as reformulation_count,
                   CAST(SUM(is_reformulation) AS FLOAT) / COUNT(*) as reformulation_rate
            FROM queries
            WHERE timestamp BETWEEN :start_date AND :end_date
            GROUP BY device_type
        """,
        
        'by_query_type': """
            SELECT query_type,
                   COUNT(*) as total_queries,
                   SUM(is_reformulation) as reformulation_count,
                   CAST(SUM(is_reformulation) AS FLOAT) / COUNT(*) as reformulation_rate
            FROM queries
            WHERE timestamp BETWEEN :start_date AND :end_date
            GROUP BY query_type
        """,
        
        'top_reformulated_queries': """
            WITH query_pairs AS (
                SELECT q1.query_text as original_query,
                       q2.query_text as reformulated_query,
                       q2.is_reformulation
                FROM queries q1
                JOIN queries q2 ON q1.session_id = q2.session_id
                                AND q2.timestamp > q1.timestamp
                                AND CAST(JULIANDAY(q2.timestamp) - JULIANDAY(q1.timestamp) AS FLOAT) * 86400 <= 600
                WHERE q1.timestamp BETWEEN :start_date AND :end_date
                  AND q2.is_reformulation = 1
            )
            SELECT original_query,
                   COUNT(*) as reformulation_count
            FROM query_pairs
            GROUP BY original_query
            ORDER BY reformulation_count DESC
            LIMIT :limit
        """
    },
    
    # Session-related queries
    'session': {
        'overall_metrics': """
            SELECT COUNT(*) as total_sessions,
                   AVG(duration_seconds) as avg_duration,
                   AVG(num_queries) as avg_queries_per_session,
                   AVG(total_clicks) as avg_clicks_per_session,
                   AVG(clicks_per_query) as avg_clicks_per_query,
                   AVG(num_reformulations) as avg_reformulations,
                   SUM(CASE WHEN abandoned = 1 THEN 1 ELSE 0 END) as abandoned_sessions,
                   CAST(SUM(CASE WHEN abandoned = 1 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) as abandonment_rate,
                   AVG(success_score) / 3.0 as avg_success_rate
            FROM sessions
            WHERE start_time BETWEEN :start_date AND :end_date
        """,
        
        'daily': """
            SELECT date,
                   COUNT(*) as total_sessions,
                   AVG(duration_seconds) as avg_duration,
                   AVG(num_queries) as avg_queries_per_session,
                   CAST(SUM(CASE WHEN abandoned = 1 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) as abandonment_rate,
                   AVG(success_score) / 3.0 as avg_success_rate
            FROM sessions
            WHERE start_time BETWEEN :start_date AND :end_date
            GROUP BY date
            ORDER BY date
        """,
        
        'by_device': """
            SELECT device_type,
                   COUNT(*) as total_sessions,
                   AVG(duration_seconds) as avg_duration,
                   AVG(num_queries) as avg_queries_per_session,
                   AVG(total_clicks) as avg_clicks_per_session,
                   CAST(SUM(CASE WHEN abandoned = 1 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) as abandonment_rate,
                   AVG(success_score) / 3.0 as avg_success_rate
            FROM sessions
            WHERE start_time BETWEEN :start_date AND :end_date
            GROUP BY device_type
        """,
        
        'query_pattern': """
            SELECT 
                CASE 
                    WHEN num_queries = 1 THEN '1 query'
                    WHEN num_queries = 2 THEN '2 queries'
                    WHEN num_queries = 3 THEN '3 queries'
                    WHEN num_queries = 4 THEN '4 queries'
                    WHEN num_queries = 5 THEN '5 queries'
                    WHEN num_queries BETWEEN 6 AND 10 THEN '6-10 queries'
                    ELSE '10+ queries'
                END as session_length,
                COUNT(*) as session_count,
                CAST(COUNT(*) AS FLOAT) / (SELECT COUNT(*) FROM sessions WHERE start_time BETWEEN :start_date AND :end_date) as percentage
            FROM sessions
            WHERE start_time BETWEEN :start_date AND :end_date
            GROUP BY session_length
            ORDER BY MIN(num_queries)
        """,
        
        'success_distribution': """
            SELECT success_score,
                   COUNT(*) as session_count,
                   CAST(COUNT(*) AS FLOAT) / (SELECT COUNT(*) FROM sessions WHERE start_time BETWEEN :start_date AND :end_date) as percentage
            FROM sessions
            WHERE start_time BETWEEN :start_date AND :end_date
            GROUP BY success_score
            ORDER BY success_score
        """
    },
    
    # User behavior queries
    'user_behavior': {
        'session_frequency': """
            WITH user_sessions AS (
                SELECT user_id, COUNT(*) as session_count
                FROM sessions
                WHERE start_time BETWEEN :start_date AND :end_date
                GROUP BY user_id
            )
            SELECT 
                CASE 
                    WHEN session_count = 1 THEN '1 session'
                    WHEN session_count = 2 THEN '2 sessions'
                    WHEN session_count = 3 THEN '3 sessions'
                    WHEN session_count = 4 THEN '4 sessions'
                    WHEN session_count = 5 THEN '5 sessions'
                    WHEN session_count BETWEEN 6 AND 10 THEN '6-10 sessions'
                    WHEN session_count BETWEEN 11 AND 20 THEN '11-20 sessions'
                    ELSE '20+ sessions'
                END as frequency_bucket,
                COUNT(*) as user_count,
                CAST(COUNT(*) AS FLOAT) / (SELECT COUNT(DISTINCT user_id) FROM sessions WHERE start_time BETWEEN :start_date AND :end_date) as percentage
            FROM user_sessions
            GROUP BY frequency_bucket
            ORDER BY MIN(session_count)
        """,
        
        'query_volume_by_hour': """
            SELECT strftime('%H', timestamp) as hour,
                   COUNT(*) as query_count
            FROM queries
            WHERE timestamp BETWEEN :start_date AND :end_date
            GROUP BY hour
            ORDER BY hour
        """,
        
        'query_volume_by_day': """
            SELECT strftime('%w', timestamp) as day_of_week,
                   COUNT(*) as query_count
            FROM queries
            WHERE timestamp BETWEEN :start_date AND :end_date
            GROUP BY day_of_week
            ORDER BY day_of_week
        """
    },
    
    # Query content analysis
    'query_content': {
        'top_queries': """
            SELECT query_text,
                   COUNT(*) as frequency,
                   AVG(CASE WHEN num_clicks > 0 THEN 1 ELSE 0 END) as ctr,
                   AVG(reciprocal_rank) as mrr
            FROM queries
            WHERE timestamp BETWEEN :start_date AND :end_date
            GROUP BY query_text
            HAVING COUNT(*) >= :min_frequency
            ORDER BY frequency DESC
            LIMIT :limit
        """,
        
        'zero_click_queries': """
            SELECT query_text,
                   COUNT(*) as frequency
            FROM queries
            WHERE timestamp BETWEEN :start_date AND :end_date
              AND num_clicks = 0
            GROUP BY query_text
            HAVING COUNT(*) >= :min_frequency
            ORDER BY frequency DESC
            LIMIT :limit
        """,
        
        'query_type_distribution': """
            SELECT query_type,
                   COUNT(*) as query_count,
                   CAST(COUNT(*) AS FLOAT) / (SELECT COUNT(*) FROM queries WHERE timestamp BETWEEN :start_date AND :end_date) as percentage
            FROM queries
            WHERE timestamp BETWEEN :start_date AND :end_date
            GROUP BY query_type
            ORDER BY query_count DESC
        """
    },
    
    # A/B testing queries
    'ab_testing': {
        'ctr_comparison': """
            SELECT 
                variant,
                COUNT(*) as total_queries,
                SUM(CASE WHEN num_clicks > 0 THEN 1 ELSE 0 END) as queries_with_clicks,
                CAST(SUM(CASE WHEN num_clicks > 0 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) as ctr
            FROM queries
            WHERE timestamp BETWEEN :start_date AND :end_date
              AND experiment_id = :experiment_id
            GROUP BY variant
        """,
        
        'time_to_click_comparison': """
            SELECT 
                variant,
                COUNT(*) as total_queries_with_clicks,
                AVG(time_to_first_click) as mean_ttc,
                MIN(time_to_first_click) as min_ttc,
                MAX(time_to_first_click) as max_ttc
            FROM queries
            WHERE timestamp BETWEEN :start_date AND :end_date
              AND experiment_id = :experiment_id
              AND time_to_first_click IS NOT NULL
            GROUP BY variant
        """,
        
        'relevance_comparison': """
            SELECT 
                variant,
                COUNT(*) as total_queries,
                AVG(reciprocal_rank) as mrr,
                AVG(ndcg_5) as ndcg_5
            FROM queries
            WHERE timestamp BETWEEN :start_date AND :end_date
              AND experiment_id = :experiment_id
            GROUP BY variant
        """,
        
        'session_comparison': """
            SELECT 
                variant,
                COUNT(*) as total_sessions,
                AVG(duration_seconds) as avg_duration,
                AVG(num_queries) as avg_queries_per_session,
                CAST(SUM(CASE WHEN abandoned = 1 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) as abandonment_rate,
                AVG(success_score) / 3.0 as avg_success_rate
            FROM sessions
            WHERE start_time BETWEEN :start_date AND :end_date
              AND experiment_id = :experiment_id
            GROUP BY variant
        """
    }
}

# Helper functions to generate SQL queries with parameters
def get_query(category: str, query_name: str) -> str:
    """
    Get a SQL query template by category and name.
    
    Args:
        category: Query category (e.g., 'ctr', 'time_to_click')
        query_name: Specific query name within the category
        
    Returns:
        SQL query template string
    """
    return QUERY_TEMPLATES.get(category, {}).get(query_name, "")

def get_schema_definition(table_name: str) -> str:
    """
    Get schema definition for a table.
    
    Args:
        table_name: Name of the table
        
    Returns:
        CREATE TABLE statement for the table
    """
    return SCHEMA_DEFINITIONS.get(table_name, "")

def adapt_query_for_postgresql(query: str) -> str:
    """
    Adapt a SQLite query for PostgreSQL.
    
    Args:
        query: SQLite query string
        
    Returns:
        PostgreSQL-compatible query string
    """
    # Replace SQLite-specific functions with PostgreSQL equivalents
    adaptations = {
        'strftime': 'EXTRACT',
        "strftime('%H', timestamp)": "EXTRACT(HOUR FROM timestamp)",
        "strftime('%w', timestamp)": "EXTRACT(DOW FROM timestamp)",
        "strftime('%Y-%m-%d', timestamp)": "DATE(timestamp)",
        "JULIANDAY": "EXTRACT(EPOCH FROM",
        "datetime('now'": "NOW(",
        "CAST(JULIANDAY(q2.timestamp) - JULIANDAY(q1.timestamp) AS FLOAT) * 86400": "EXTRACT(EPOCH FROM (q2.timestamp - q1.timestamp))"
    }
    
    adapted_query = query
    
    for sqlite_func, pg_func in adaptations.items():
        adapted_query = adapted_query.replace(sqlite_func, pg_func)
    
    return adapted_query
