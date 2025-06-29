"""
SQL queries module for search quality dashboard.

This module provides a collection of SQL query templates for retrieving
search quality metrics and data for the dashboard.
"""

from typing import Dict, List, Union, Optional, Any
import datetime

class SQLQueries:
    """Class containing SQL query templates for search quality analysis."""
    
    @staticmethod
    def get_daily_metrics(start_date: str, end_date: str, segment: Optional[str] = None) -> str:
        """
        Get daily metrics query.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            segment: User segment filter (optional)
            
        Returns:
            SQL query string
        """
        segment_filter = f"AND user_segment = '{segment}'" if segment else ""
        
        query = f"""
        SELECT 
            date,
            user_segment,
            device_type,
            total_sessions,
            total_queries,
            total_clicks,
            avg_ctr,
            avg_time_to_first_click,
            avg_queries_per_session,
            zero_click_rate,
            click_depth,
            avg_ndcg
        FROM 
            daily_metrics
        WHERE 
            date BETWEEN '{start_date}' AND '{end_date}'
            {segment_filter}
        ORDER BY 
            date ASC
        """
        return query
    
    @staticmethod
    def get_session_details(session_id: str) -> str:
        """
        Get details for a specific search session.
        
        Args:
            session_id: ID of the search session
            
        Returns:
            SQL query string
        """
        query = f"""
        SELECT 
            s.session_id,
            s.user_id,
            s.start_time,
            s.end_time,
            s.session_duration,
            s.device_type,
            s.browser,
            s.user_segment,
            s.total_queries,
            s.total_clicks,
            q.query_id,
            q.query_text,
            q.timestamp,
            q.result_count,
            q.has_click,
            q.time_to_first_click
        FROM 
            search_sessions s
        LEFT JOIN 
            search_queries q ON s.session_id = q.session_id
        WHERE 
            s.session_id = '{session_id}'
        ORDER BY 
            q.timestamp ASC
        """
        return query
    
    @staticmethod
    def get_query_results(query_id: str) -> str:
        """
        Get search results for a specific query.
        
        Args:
            query_id: ID of the search query
            
        Returns:
            SQL query string
        """
        query = f"""
        SELECT 
            r.result_id,
            r.position,
            r.document_id,
            r.clicked,
            r.click_time,
            r.dwell_time
        FROM 
            search_results r
        WHERE 
            r.query_id = '{query_id}'
        ORDER BY 
            r.position ASC
        """
        return query
    
    @staticmethod
    def get_user_sessions(user_id: str, limit: int = 10) -> str:
        """
        Get search sessions for a specific user.
        
        Args:
            user_id: ID of the user
            limit: Maximum number of sessions to return
            
        Returns:
            SQL query string
        """
        query = f"""
        SELECT 
            session_id,
            start_time,
            end_time,
            session_duration,
            device_type,
            browser,
            total_queries,
            total_clicks
        FROM 
            search_sessions
        WHERE 
            user_id = '{user_id}'
        ORDER BY 
            start_time DESC
        LIMIT {limit}
        """
        return query
    
    @staticmethod
    def get_top_queries(start_date: str, end_date: str, limit: int = 10) -> str:
        """
        Get top search queries by frequency.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            limit: Maximum number of queries to return
            
        Returns:
            SQL query string
        """
        query = f"""
        SELECT 
            query_text,
            COUNT(*) as query_count,
            SUM(has_click) as click_count,
            SUM(has_click) / COUNT(*) as click_rate,
            AVG(time_to_first_click) as avg_time_to_click
        FROM 
            search_queries
        WHERE 
            DATE(timestamp) BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY 
            query_text
        ORDER BY 
            query_count DESC
        LIMIT {limit}
        """
        return query
    
    @staticmethod
    def get_reformulation_patterns(start_date: str, end_date: str, limit: int = 10) -> str:
        """
        Get common query reformulation patterns.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            limit: Maximum number of patterns to return
            
        Returns:
            SQL query string
        """
        query = f"""
        WITH query_pairs AS (
            SELECT 
                q1.query_text as original_query,
                q2.query_text as reformulated_query,
                COUNT(*) as frequency
            FROM 
                search_queries q1
            JOIN 
                search_queries q2 ON q1.session_id = q2.session_id
                AND q2.reformulation_of = q1.query_id
            WHERE 
                DATE(q1.timestamp) BETWEEN '{start_date}' AND '{end_date}'
            GROUP BY 
                q1.query_text, q2.query_text
        )
        SELECT 
            original_query,
            reformulated_query,
            frequency,
            CASE 
                WHEN LENGTH(reformulated_query) > LENGTH(original_query) THEN 'expanded'
                WHEN LENGTH(reformulated_query) < LENGTH(original_query) THEN 'narrowed'
                ELSE 'rephrased'
            END as reformulation_type
        FROM 
            query_pairs
        ORDER BY 
            frequency DESC
        LIMIT {limit}
        """
        return query
    
    @staticmethod
    def get_engagement_by_position(start_date: str, end_date: str, segment: Optional[str] = None) -> str:
        """
        Get click engagement metrics by result position.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            segment: User segment filter (optional)
            
        Returns:
            SQL query string
        """
        segment_join = ""
        segment_filter = ""
        
        if segment:
            segment_join = "JOIN search_sessions s ON r.session_id = s.session_id"
            segment_filter = f"AND s.user_segment = '{segment}'"
        
        query = f"""
        SELECT 
            r.position,
            COUNT(*) as impression_count,
            SUM(r.clicked) as click_count,
            SUM(r.clicked) / COUNT(*) as ctr,
            AVG(r.dwell_time) as avg_dwell_time
        FROM 
            search_results r
        {segment_join}
        WHERE 
            DATE(r.click_time) BETWEEN '{start_date}' AND '{end_date}'
            {segment_filter}
        GROUP BY 
            r.position
        ORDER BY 
            r.position ASC
        LIMIT 20
        """
        return query
    
    @staticmethod
    def get_ab_test_results(test_id: str) -> str:
        """
        Get results for a specific A/B test.
        
        Args:
            test_id: ID of the A/B test
            
        Returns:
            SQL query string
        """
        query = f"""
        SELECT 
            test_id,
            test_name,
            start_date,
            end_date,
            metric,
            variant,
            value,
            sample_size,
            p_value,
            significant
        FROM 
            ab_test_results
        WHERE 
            test_id = '{test_id}'
        ORDER BY 
            metric, variant
        """
        return query
    
    @staticmethod
    def get_metric_timeseries(metric: str, start_date: str, end_date: str, 
                             group_by: str = 'user_segment', device_type: Optional[str] = None) -> str:
        """
        Get time series data for a specific metric.
        
        Args:
            metric: Name of the metric column
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            group_by: Field to group results by ('user_segment' or 'device_type')
            device_type: Device type filter (optional)
            
        Returns:
            SQL query string
        """
        device_filter = f"AND device_type = '{device_type}'" if device_type else ""
        
        query = f"""
        SELECT 
            date,
            {group_by},
            {metric}
        FROM 
            daily_metrics
        WHERE 
            date BETWEEN '{start_date}' AND '{end_date}'
            {device_filter}
        ORDER BY 
            date ASC, {group_by}
        """
        return query
    
    @staticmethod
    def get_user_segment_distribution(start_date: str, end_date: str) -> str:
        """
        Get distribution of sessions across user segments.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            SQL query string
        """
        query = f"""
        SELECT 
            user_segment,
            COUNT(*) as session_count,
            COUNT(*) * 100.0 / (SELECT COUNT(*) FROM search_sessions 
                              WHERE DATE(start_time) BETWEEN '{start_date}' AND '{end_date}') as percentage,
            AVG(total_queries) as avg_queries,
            AVG(total_clicks) as avg_clicks,
            AVG(session_duration) as avg_duration
        FROM 
            search_sessions
        WHERE 
            DATE(start_time) BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY 
            user_segment
        ORDER BY 
            session_count DESC
        """
        return query
    
    @staticmethod
    def get_device_type_distribution(start_date: str, end_date: str) -> str:
        """
        Get distribution of sessions across device types.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            SQL query string
        """
        query = f"""
        SELECT 
            device_type,
            COUNT(*) as session_count,
            COUNT(*) * 100.0 / (SELECT COUNT(*) FROM search_sessions 
                              WHERE DATE(start_time) BETWEEN '{start_date}' AND '{end_date}') as percentage,
            AVG(total_queries) as avg_queries,
            AVG(total_clicks) as avg_clicks,
            AVG(session_duration) as avg_duration
        FROM 
            search_sessions
        WHERE 
            DATE(start_time) BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY 
            device_type
        ORDER BY 
            session_count DESC
        """
        return query
    
    @staticmethod
    def get_low_satisfaction_queries(start_date: str, end_date: str, limit: int = 20) -> str:
        """
        Get queries with low user satisfaction.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            limit: Maximum number of queries to return
            
        Returns:
            SQL query string
        """
        query = f"""
        SELECT 
            q.query_text,
            COUNT(*) as query_count,
            SUM(q.has_click) / COUNT(*) as click_rate,
            AVG(CASE WHEN q.has_click THEN q.time_to_first_click ELSE NULL END) as avg_time_to_click,
            MAX(r.position) as deepest_click_position
        FROM 
            search_queries q
        LEFT JOIN 
            search_results r ON q.query_id = r.query_id AND r.clicked = 1
        WHERE 
            DATE(q.timestamp) BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY 
            q.query_text
        HAVING 
            query_count >= 5
            AND (click_rate < 0.3 OR avg_time_to_click > 10 OR deepest_click_position > 5)
        ORDER BY 
            click_rate ASC, query_count DESC
        LIMIT {limit}
        """
        return query
    
    @staticmethod
    def get_zero_click_stats(start_date: str, end_date: str) -> str:
        """
        Get statistics about zero-click queries.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            SQL query string
        """
        query = f"""
        SELECT 
            DATE(timestamp) as date,
            user_segment,
            COUNT(*) as query_count,
            SUM(CASE WHEN has_click = 0 THEN 1 ELSE 0 END) as zero_click_count,
            SUM(CASE WHEN has_click = 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as zero_click_percentage
        FROM 
            search_queries q
        JOIN 
            search_sessions s ON q.session_id = s.session_id
        WHERE 
            DATE(q.timestamp) BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY 
            date, user_segment
        ORDER BY 
            date ASC, user_segment
        """
        return query
    
    @staticmethod
    def get_query_complexity_stats(start_date: str, end_date: str) -> str:
        """
        Get statistics about query complexity.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            SQL query string
        """
        query = f"""
        SELECT 
            CASE 
                WHEN LENGTH(query_text) - LENGTH(REPLACE(query_text, ' ', '')) + 1 BETWEEN 1 AND 2 THEN '1-2 words'
                WHEN LENGTH(query_text) - LENGTH(REPLACE(query_text, ' ', '')) + 1 BETWEEN 3 AND 4 THEN '3-4 words'
                WHEN LENGTH(query_text) - LENGTH(REPLACE(query_text, ' ', '')) + 1 BETWEEN 5 AND 7 THEN '5-7 words'
                ELSE '8+ words'
            END as query_length,
            COUNT(*) as query_count,
            SUM(has_click) / COUNT(*) as click_rate,
            AVG(CASE WHEN has_click = 1 THEN time_to_first_click ELSE NULL END) as avg_time_to_click
        FROM 
            search_queries
        WHERE 
            DATE(timestamp) BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY 
            query_length
        ORDER BY 
            CASE query_length
                WHEN '1-2 words' THEN 1
                WHEN '3-4 words' THEN 2
                WHEN '5-7 words' THEN 3
                WHEN '8+ words' THEN 4
            END
        """
        return query
    
    @staticmethod
    def get_recent_activity(limit: int = 100) -> str:
        """
        Get recent search activity.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            SQL query string
        """
        query = f"""
        SELECT 
            q.query_id,
            q.session_id,
            q.query_text,
            q.timestamp,
            q.has_click,
            q.time_to_first_click,
            s.user_segment,
            s.device_type
        FROM 
            search_queries q
        JOIN 
            search_sessions s ON q.session_id = s.session_id
        ORDER BY 
            q.timestamp DESC
        LIMIT {limit}
        """
        return query
