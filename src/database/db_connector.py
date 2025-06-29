"""
Database connector module for search quality dashboard.

This module provides a class for interacting with the database, executing queries,
and retrieving data for the dashboard.
"""

import sqlite3
import pandas as pd
import logging
from typing import Dict, List, Union, Optional, Any
from pathlib import Path
import os

class DatabaseConnector:
    """Class for handling database connections and operations."""
    
    def __init__(self, db_path: str = 'search_metrics.db'):
        """
        Initialize the database connector.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.logger = logging.getLogger(__name__)
    
    def connect(self) -> None:
        """
        Establish a connection to the database.
        
        Raises:
            Exception: If connection fails
        """
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            self.logger.info(f"Connected to database at {self.db_path}")
        except Exception as e:
            self.logger.error(f"Error connecting to database: {str(e)}")
            raise
    
    def disconnect(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
            self.logger.info("Disconnected from database")
    
    def execute_query(self, query: str, params: Optional[Union[tuple, Dict[str, Any]]] = None) -> List[tuple]:
        """
        Execute a SQL query and return the results.
        
        Args:
            query: SQL query string
            params: Parameters to bind to the query
            
        Returns:
            List of tuples containing query results
            
        Raises:
            Exception: If query execution fails
        """
        if not self.conn:
            self.connect()
        
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            
            results = self.cursor.fetchall()
            return results
        except Exception as e:
            self.logger.error(f"Error executing query: {str(e)}\nQuery: {query}\nParams: {params}")
            raise
    
    def execute_query_to_df(self, query: str, params: Optional[Union[tuple, Dict[str, Any]]] = None) -> pd.DataFrame:
        """
        Execute a SQL query and return the results as a pandas DataFrame.
        
        Args:
            query: SQL query string
            params: Parameters to bind to the query
            
        Returns:
            DataFrame containing query results
            
        Raises:
            Exception: If query execution fails
        """
        if not self.conn:
            self.connect()
        
        try:
            if params:
                return pd.read_sql_query(query, self.conn, params=params)
            else:
                return pd.read_sql_query(query, self.conn)
        except Exception as e:
            self.logger.error(f"Error executing query to DataFrame: {str(e)}\nQuery: {query}\nParams: {params}")
            raise
    
    def execute_write_query(self, query: str, params: Optional[Union[tuple, Dict[str, Any]]] = None) -> int:
        """
        Execute a SQL write query (INSERT, UPDATE, DELETE) and return the number of affected rows.
        
        Args:
            query: SQL query string
            params: Parameters to bind to the query
            
        Returns:
            Number of affected rows
            
        Raises:
            Exception: If query execution fails
        """
        if not self.conn:
            self.connect()
        
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            
            self.conn.commit()
            return self.cursor.rowcount
        except Exception as e:
            self.conn.rollback()
            self.logger.error(f"Error executing write query: {str(e)}\nQuery: {query}\nParams: {params}")
            raise
    
    def execute_many(self, query: str, params_list: List[Union[tuple, Dict[str, Any]]]) -> int:
        """
        Execute a SQL query multiple times with different parameter sets.
        
        Args:
            query: SQL query string
            params_list: List of parameter sets to bind to the query
            
        Returns:
            Number of affected rows
            
        Raises:
            Exception: If query execution fails
        """
        if not self.conn:
            self.connect()
        
        try:
            self.cursor.executemany(query, params_list)
            self.conn.commit()
            return self.cursor.rowcount
        except Exception as e:
            self.conn.rollback()
            self.logger.error(f"Error executing many: {str(e)}\nQuery: {query}")
            raise
    
    def insert_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = 'append') -> int:
        """
        Insert a pandas DataFrame into a database table.
        
        Args:
            df: DataFrame to insert
            table_name: Name of the target table
            if_exists: How to behave if the table already exists
                       ('fail', 'replace', or 'append')
            
        Returns:
            Number of rows inserted
            
        Raises:
            Exception: If insertion fails
        """
        if not self.conn:
            self.connect()
        
        try:
            df.to_sql(table_name, self.conn, if_exists=if_exists, index=False)
            return len(df)
        except Exception as e:
            self.conn.rollback()
            self.logger.error(f"Error inserting DataFrame: {str(e)}\nTable: {table_name}")
            raise
    
    def create_table(self, table_name: str, columns: Dict[str, str], primary_key: Optional[str] = None) -> None:
        """
        Create a new database table.
        
        Args:
            table_name: Name of the table to create
            columns: Dictionary mapping column names to their SQL types
            primary_key: Name of the column to use as primary key (optional)
            
        Raises:
            Exception: If table creation fails
        """
        if not self.conn:
            self.connect()
        
        # Build column definitions
        column_defs = []
        for col_name, col_type in columns.items():
            if primary_key and col_name == primary_key:
                column_defs.append(f"{col_name} {col_type} PRIMARY KEY")
            else:
                column_defs.append(f"{col_name} {col_type}")
        
        # Build CREATE TABLE query
        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(column_defs)})"
        
        try:
            self.cursor.execute(query)
            self.conn.commit()
            self.logger.info(f"Created table: {table_name}")
        except Exception as e:
            self.conn.rollback()
            self.logger.error(f"Error creating table: {str(e)}\nQuery: {query}")
            raise
    
    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.
        
        Args:
            table_name: Name of the table to check
            
        Returns:
            True if the table exists, False otherwise
        """
        if not self.conn:
            self.connect()
        
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
        result = self.execute_query(query, (table_name,))
        return len(result) > 0
    
    def get_table_schema(self, table_name: str) -> List[tuple]:
        """
        Get the schema of a database table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            List of tuples containing column information
            
        Raises:
            Exception: If the table does not exist
        """
        if not self.conn:
            self.connect()
        
        if not self.table_exists(table_name):
            raise ValueError(f"Table '{table_name}' does not exist")
        
        query = f"PRAGMA table_info({table_name})"
        return self.execute_query(query)
    
    def initialize_database(self, schema_file: Optional[str] = None) -> None:
        """
        Initialize the database with tables from a schema file.
        
        Args:
            schema_file: Path to SQL schema file (optional)
            
        Raises:
            Exception: If initialization fails
        """
        if not self.conn:
            self.connect()
        
        try:
            if schema_file and os.path.exists(schema_file):
                with open(schema_file, 'r') as f:
                    schema_sql = f.read()
                self.conn.executescript(schema_sql)
                self.logger.info(f"Initialized database from schema file: {schema_file}")
            else:
                # Create default tables if no schema file is provided
                self.create_default_tables()
                self.logger.info("Initialized database with default tables")
            
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            self.logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def create_default_tables(self) -> None:
        """
        Create default tables for the search quality dashboard.
        
        Raises:
            Exception: If table creation fails
        """
        # Create table for search sessions
        self.create_table(
            table_name="search_sessions",
            columns={
                "session_id": "TEXT",
                "user_id": "TEXT",
                "start_time": "TIMESTAMP",
                "end_time": "TIMESTAMP",
                "session_duration": "REAL",
                "device_type": "TEXT",
                "browser": "TEXT",
                "user_segment": "TEXT",
                "total_queries": "INTEGER",
                "total_clicks": "INTEGER"
            },
            primary_key="session_id"
        )
        
        # Create table for search queries
        self.create_table(
            table_name="search_queries",
            columns={
                "query_id": "TEXT",
                "session_id": "TEXT",
                "query_text": "TEXT",
                "timestamp": "TIMESTAMP",
                "result_count": "INTEGER",
                "reformulation_of": "TEXT",
                "has_click": "BOOLEAN",
                "time_to_first_click": "REAL",
                "max_click_position": "INTEGER"
            },
            primary_key="query_id"
        )
        
        # Create table for search results
        self.create_table(
            table_name="search_results",
            columns={
                "result_id": "TEXT",
                "query_id": "TEXT",
                "session_id": "TEXT",
                "position": "INTEGER",
                "document_id": "TEXT",
                "clicked": "BOOLEAN",
                "click_time": "TIMESTAMP",
                "dwell_time": "REAL"
            },
            primary_key="result_id"
        )
        
        # Create table for daily aggregated metrics
        self.create_table(
            table_name="daily_metrics",
            columns={
                "date": "DATE",
                "user_segment": "TEXT",
                "device_type": "TEXT",
                "total_sessions": "INTEGER",
                "total_queries": "INTEGER",
                "total_clicks": "INTEGER",
                "avg_ctr": "REAL",
                "avg_time_to_first_click": "REAL",
                "avg_queries_per_session": "REAL",
                "zero_click_rate": "REAL",
                "click_depth": "REAL",
                "avg_ndcg": "REAL"
            }
        )
        
        # Create table for A/B test results
        self.create_table(
            table_name="ab_test_results",
            columns={
                "test_id": "TEXT",
                "test_name": "TEXT",
                "start_date": "DATE",
                "end_date": "DATE",
                "metric": "TEXT",
                "variant": "TEXT",
                "value": "REAL",
                "sample_size": "INTEGER",
                "p_value": "REAL",
                "significant": "BOOLEAN"
            }
        )
