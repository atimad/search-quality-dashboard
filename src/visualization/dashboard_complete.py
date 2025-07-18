"""
Dashboard visualization module for the Search Quality Metrics Dashboard.

This module implements the Streamlit-based dashboard for visualizing search quality metrics.
"""

import os
import sys
import logging
import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Import project modules
from database.db_connector import DatabaseConnector as SearchDatabaseConnector
from database.sql_queries import SQLQueries as SearchQueryBuilder
from metrics.engagement_metrics import EngagementMetrics
from metrics.relevance_metrics import RelevanceMetrics
from metrics import statistical_utils as stats_utils

# Dashboard colors and styling
COLORS = {
    'primary': '#1E88E5',   # Blue
    'secondary': '#FFC107', # Amber
    'success': '#4CAF50',   # Green
    'danger': '#E53935',    # Red
    'warning': '#FF9800',   # Orange
    'info': '#00ACC1',      # Cyan
    'light': '#F5F5F5',     # Light Grey
    'dark': '#212121',      # Dark Grey
    'background': '#FFFFFF',# White
    'text': '#212121',      # Dark Grey
}

# Streamlit page config
st.set_page_config(
    page_title="Search Quality Metrics Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    """Main function to run the dashboard."""
    # Initialize database connection
    db = SearchDatabaseConnector()
    query_builder = SearchQueryBuilder()
    
    # Set up sidebar
    setup_sidebar()
    
    # Get selected view from sidebar
    view = st.session_state.get('view', 'Overview')
    
    # Render the selected view
    if view == 'Overview':
        render_overview_page(db, query_builder)
    elif view == 'Query Analysis':
        render_query_analysis_page(db, query_builder)
    elif view == 'User Engagement':
        render_user_engagement_page(db, query_builder)
    elif view == 'Result Quality':
        render_result_quality_page(db, query_builder)
    elif view == 'Session Analysis':
        render_session_analysis_page(db, query_builder)
    elif view == 'Segmentation':
        render_segmentation_page(db, query_builder)
    elif view == 'Trends & Anomalies':
        render_trends_anomalies_page(db, query_builder)
    elif view == 'A/B Testing':
        render_ab_testing_page(db, query_builder)
    elif view == 'Data Explorer':
        render_data_explorer_page(db, query_builder)

def setup_sidebar():
    """Set up the sidebar with navigation and filters."""
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/atimad/search-quality-dashboard/main/assets/logo.png", width=200)
        st.title("Search Quality Dashboard")
        
        # Navigation
        st.header("Navigation")
        views = [
            'Overview', 
            'Query Analysis', 
            'User Engagement', 
            'Result Quality', 
            'Session Analysis', 
            'Segmentation', 
            'Trends & Anomalies',
            'A/B Testing',
            'Data Explorer'
        ]
        
        selected_view = st.radio("Select View", views, index=views.index(st.session_state.get('view', 'Overview')))
        st.session_state['view'] = selected_view
        
        # Date Range Filter
        st.header("Date Range")
        col1, col2 = st.columns(2)
        
        # Get min and max dates from data
        min_date = datetime.date(2023, 1, 1)  # Replace with actual min date from data
        max_date = datetime.date.today()
        
        with col1:
            start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        
        with col2:
            end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
        
        st.session_state['date_range'] = (start_date, end_date)
        
        # Common Filters
        st.header("Filters")
        
        # Device Type Filter
        device_types = ['All', 'Desktop', 'Mobile', 'Tablet']
        selected_device = st.selectbox("Device Type", device_types)
        st.session_state['device_filter'] = None if selected_device == 'All' else selected_device
        
        # Location Filter
        locations = ['All', 'North America', 'Europe', 'Asia', 'South America', 'Africa', 'Oceania']
        selected_location = st.selectbox("Location", locations)
        st.session_state['location_filter'] = None if selected_location == 'All' else selected_location
        
        # Query Type Filter
        query_types = ['All', 'Navigational', 'Informational', 'Transactional']
        selected_query_type = st.selectbox("Query Type", query_types)
        st.session_state['query_type_filter'] = None if selected_query_type == 'All' else selected_query_type
        
        # Advanced Filters (expandable)
        with st.expander("Advanced Filters"):
            # User Type
            user_types = ['All', 'New', 'Returning']
            selected_user_type = st.selectbox("User Type", user_types)
            st.session_state['user_type_filter'] = None if selected_user_type == 'All' else selected_user_type
            
            # Time of Day
            time_periods = ['All', 'Morning', 'Afternoon', 'Evening', 'Night']
            selected_time = st.selectbox("Time of Day", time_periods)
            st.session_state['time_filter'] = None if selected_time == 'All' else selected_time
            
            # User Segment
            segments = ['All', 'High Engagement', 'Medium Engagement', 'Low Engagement']
            selected_segment = st.selectbox("User Segment", segments)
            st.session_state['segment_filter'] = None if selected_segment == 'All' else selected_segment
        
        # Apply Filters Button
        if st.button("Apply Filters"):
            st.success("Filters applied!")

def render_overview_page(db, query_builder):
    """Render the overview dashboard page."""
    st.title("Search Quality Overview")
    st.subheader("Key metrics at a glance")
    
    # Get date range from session state
    start_date, end_date = st.session_state.get('date_range', (datetime.date(2023, 1, 1), datetime.date.today()))
    
    # KPI Metrics Row
    col1, col2, col3, col4 = st.columns(4)

    metrics_query = query_builder.get_daily_metrics(str(start_date), str(end_date))
    metrics_df = db.execute_query_to_df(metrics_query)

    if not metrics_df.empty:
        ctr = metrics_df['avg_ctr'].mean()
        zero_click = metrics_df['zero_click_rate'].mean()
        time_to_click = metrics_df['avg_time_to_first_click'].mean()
        reformulation = metrics_df['avg_queries_per_session'].mean() - 1
    else:
        ctr = zero_click = time_to_click = reformulation = np.nan

    with col1:
        st.metric(label="CTR", value=f"{ctr:.2%}" if ctr==ctr else 'N/A')

    with col2:
        st.metric(label="Zero-Click Rate", value=f"{zero_click:.2%}" if zero_click==zero_click else 'N/A')

    with col3:
        st.metric(label="Avg. Time to Click", value=f"{time_to_click:.1f}s" if time_to_click==time_to_click else 'N/A')

    with col4:
        st.metric(label="Reformulation Rate", value=f"{reformulation:.2%}" if reformulation==reformulation else 'N/A')
    
    # Time Series Charts
    st.subheader("Trends Over Time")
    
    if not metrics_df.empty:
        dates = pd.to_datetime(metrics_df['date'])
        ctr_values = metrics_df['avg_ctr']
        ttc_values = metrics_df['avg_time_to_first_click']
    else:
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        ctr_values = np.zeros(len(dates))
        ttc_values = np.zeros(len(dates))
    
    # Create a two-panel chart
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                         subplot_titles=("Click-Through Rate Over Time", "Time to First Click Over Time"),
                         vertical_spacing=0.1,
                         row_heights=[0.5, 0.5])
    
    # Add CTR line
    fig.add_trace(
        go.Scatter(x=dates, y=ctr_values, mode='lines', name='CTR', line=dict(color=COLORS['primary'], width=2)),
        row=1, col=1
    )
    
    # Add Time to First Click line
    fig.add_trace(
        go.Scatter(x=dates, y=ttc_values, mode='lines', name='Time to First Click', line=dict(color=COLORS['secondary'], width=2)),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40),
    )
    
    fig.update_yaxes(title_text="CTR", tickformat=".1%", row=1, col=1)
    fig.update_yaxes(title_text="Seconds", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Query Performance Summary
    st.subheader("Query Performance Summary")
    
    query_sql = query_builder.get_top_queries(str(start_date), str(end_date), limit=10)
    query_data = db.execute_query_to_df(query_sql)
    
    # Display query performance table
    st.dataframe(query_data)
    
    # Display additional insights or charts as needed
    st.subheader("Geographical Distribution")
    
    device_sql = query_builder.get_device_type_distribution(str(start_date), str(end_date))
    device_df = db.execute_query_to_df(device_sql)

    if not device_df.empty:
        fig = px.pie(device_df, names='device_type', values='session_count',
                     title='Sessions by Device Type')
        st.plotly_chart(fig, use_container_width=True)

def render_query_analysis_page(db, query_builder):
    """Render the query analysis page."""
    st.title("Query Analysis")
    st.subheader("Deep dive into search query patterns and performance")
    
    start_date, end_date = st.session_state.get('date_range', (datetime.date(2023, 1, 1), datetime.date.today()))

    st.subheader("Top Queries")
    top_df = db.execute_query_to_df(
        query_builder.get_top_queries(str(start_date), str(end_date), limit=20)
    )
    if not top_df.empty:
        st.dataframe(top_df)

    st.subheader("Reformulation Patterns")
    reform_df = db.execute_query_to_df(
        query_builder.get_reformulation_patterns(str(start_date), str(end_date), limit=10)
    )
    if not reform_df.empty:
        st.dataframe(reform_df)

    st.subheader("Zero-Click Rate")
    zero_df = db.execute_query_to_df(
        query_builder.get_zero_click_stats(str(start_date), str(end_date))
    )
    if not zero_df.empty:
        fig = px.line(zero_df, x='date', y='zero_click_percentage', color='user_segment', markers=True)
        fig.update_yaxes(tickformat='.1%')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Query Complexity")
    complexity_df = db.execute_query_to_df(
        query_builder.get_query_complexity_stats(str(start_date), str(end_date))
    )
    if not complexity_df.empty:
        fig = px.bar(complexity_df, x='query_length', y='query_count', title='Query Complexity Distribution')
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(complexity_df)

def render_user_engagement_page(db, query_builder):
    """Render the user engagement analysis page."""
    st.title("User Engagement Analysis")
    st.subheader("Understand how users interact with search results")
    
    start_date, end_date = st.session_state.get('date_range', (datetime.date(2023, 1, 1), datetime.date.today()))

    metrics_df = db.execute_query_to_df(
        query_builder.get_daily_metrics(str(start_date), str(end_date))
    )
    if not metrics_df.empty:
        st.subheader("Engagement Metrics Over Time")
        fig = px.line(metrics_df, x='date', y=['avg_ctr', 'zero_click_rate'], markers=True)
        fig.update_layout(yaxis_tickformat='.1%', legend_title="Metric")
        st.plotly_chart(fig, use_container_width=True)

    position_df = db.execute_query_to_df(
        query_builder.get_engagement_by_position(str(start_date), str(end_date))
    )
    if not position_df.empty:
        st.subheader("Engagement by Result Position")
        fig = px.bar(position_df, x='position', y='ctr', labels={'ctr': 'CTR'}, title='CTR by Position')
        fig.update_layout(yaxis_tickformat='.1%')
        st.plotly_chart(fig, use_container_width=True)

def render_result_quality_page(db, query_builder):
    """Render the result quality page."""
    st.title("Result Quality Analysis")
    st.subheader("Evaluate the relevance and ranking of search results")
    
    # Get date range from session state
    start_date, end_date = st.session_state.get('date_range', (datetime.date(2023, 1, 1), datetime.date.today()))
    
    # KPI Metrics Row
    st.subheader("Key Relevance Metrics")
    col1, col2, col3, col4 = st.columns(4)

    metrics_df = db.execute_query_to_df(
        query_builder.get_daily_metrics(str(start_date), str(end_date))
    )

    ndcg = metrics_df['avg_ndcg'].mean() if not metrics_df.empty else np.nan

    with col1:
        st.metric(label="NDCG@10", value=f"{ndcg:.2f}" if ndcg==ndcg else 'N/A')
    
    with col2:
        mrr = 0.72  # Mean Reciprocal Rank
        st.metric(label="MRR", value=f"{mrr:.2f}", delta="+0.02")
    
    with col3:
        map_score = 0.68  # Mean Average Precision
        st.metric(label="MAP@10", value=f"{map_score:.2f}", delta="+0.01")
    
    with col4:
        user_satisfaction = 0.79  # User satisfaction score
        st.metric(label="User Satisfaction", value=f"{user_satisfaction:.2f}", delta="+0.04")
    
    # NDCG Analysis
    st.subheader("NDCG Analysis")
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["NDCG Trend", "NDCG by Query Type", "NDCG Distribution"])
    
    with tab1:
        # NDCG Trend Over Time
        if not metrics_df.empty:
            dates = pd.to_datetime(metrics_df['date'])
            ndcg_values = metrics_df['avg_ndcg']
        else:
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            ndcg_values = np.zeros(len(dates))
        
        # Create time series chart
        fig = px.line(
            x=dates, 
            y=ndcg_values,
            labels={'x': 'Date', 'y': 'NDCG@10'},
            title='NDCG@10 Trend Over Time'
        )
        
        # Add trendline
        x = np.arange(len(dates))
        z = np.polyfit(x, ndcg_values, 1)
        y_hat = np.poly1d(z)(x)
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=y_hat,
                mode='lines',
                name='Trend',
                line=dict(color=COLORS['danger'], width=2, dash='dash')
            )
        )
        
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='NDCG@10',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # NDCG by Query Type
        # Sample data for NDCG by query type
        query_types_data = pd.DataFrame({
            'query_type': ['Informational', 'Navigational', 'Transactional'],
            'ndcg': [0.82, 0.91, 0.78],
            'mrr': [0.68, 0.85, 0.63],
            'map': [0.65, 0.81, 0.59]
        })
        
        # Create bar chart for NDCG by query type
        fig = px.bar(
            query_types_data,
            x='query_type',
            y=['ndcg', 'mrr', 'map'],
            title='Relevance Metrics by Query Type',
            labels={
                'query_type': 'Query Type',
                'value': 'Score',
                'variable': 'Metric'
            },
            barmode='group',
            color_discrete_sequence=[COLORS['primary'], COLORS['secondary'], COLORS['success']]
        )
        
        # Update trace names for better readability
        fig.data[0].name = 'NDCG@10'
        fig.data[1].name = 'MRR'
        fig.data[2].name = 'MAP@10'
        
        fig.update_layout(
            xaxis_title='Query Type',
            yaxis_title='Score',
            legend_title='Metric',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # NDCG Distribution
        # Sample data for NDCG distribution
        ndcg_bins = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-0.9', '0.9-1.0']
        ndcg_freq = [0.02, 0.05, 0.15, 0.30, 0.28, 0.20]
        
        # Create bar chart for NDCG distribution
        fig = px.bar(
            x=ndcg_bins,
            y=ndcg_freq,
            title='NDCG@10 Distribution',
            labels={'x': 'NDCG Range', 'y': 'Frequency'},
            color=ndcg_freq,
            color_continuous_scale=px.colors.sequential.Blues
        )
        
        fig.update_layout(
            xaxis_title='NDCG Range',
            yaxis_title='Frequency',
            yaxis_tickformat='.0%',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Result Position Analysis
    st.subheader("Result Position Analysis")
    
    position_sql = query_builder.get_engagement_by_position(str(start_date), str(end_date))
    position_data = db.execute_query_to_df(position_sql)
    
    if not position_data.empty:
        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                position_data,
                x='position',
                y='ctr',
                title='CTR by Result Position',
                labels={'position': 'Position', 'ctr': 'CTR'},
            )
            fig.update_layout(yaxis_tickformat='.1%')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.line(
                position_data,
                x='position',
                y='avg_dwell_time',
                markers=True,
                title='Avg. Dwell Time by Position',
                labels={'position': 'Position', 'avg_dwell_time': 'Seconds'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Content Freshness Analysis
    st.subheader("Content Freshness Analysis")
    
    # Sample data for content freshness
    freshness_data = pd.DataFrame({
        'age_bucket': ['0-1 day', '1-7 days', '7-30 days', '1-3 months', '3-6 months', '6-12 months', '1+ year'],
        'percentage': [0.12, 0.25, 0.30, 0.18, 0.08, 0.05, 0.02],
        'avg_relevance': [0.92, 0.90, 0.87, 0.82, 0.75, 0.70, 0.62],
        'avg_clicks': [0.38, 0.35, 0.32, 0.28, 0.22, 0.18, 0.15]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Content distribution by age
        fig = px.pie(
            freshness_data,
            values='percentage',
            names='age_bucket',
            title='Search Results by Content Age',
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=450)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Relevance by content age
        fig = px.bar(
            freshness_data,
            x='age_bucket',
            y=['avg_relevance', 'avg_clicks'],
            title='Relevance and Clicks by Content Age',
            labels={
                'age_bucket': 'Content Age', 
                'value': 'Score', 
                'variable': 'Metric'
            },
            barmode='group',
            color_discrete_sequence=[COLORS['primary'], COLORS['secondary']]
        )
        
        # Update trace names for better readability
        fig.data[0].name = 'Avg. Relevance'
        fig.data[1].name = 'Avg. Click Rate'
        
        fig.update_layout(
            xaxis_title='Content Age',
            yaxis_title='Score',
            legend_title='Metric',
            height=450
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # User Satisfaction Analysis
    st.subheader("User Satisfaction Analysis")
    
    # Sample data for user satisfaction by query category
    satisfaction_data = pd.DataFrame({
        'query_category': ['AI Research', 'Product Reviews', 'How-to Guides', 'News', 'Academic', 'Technical', 'Entertainment'],
        'satisfaction': [0.85, 0.78, 0.82, 0.75, 0.88, 0.81, 0.72],
        'coverage': [0.92, 0.85, 0.90, 0.82, 0.78, 0.88, 0.95]
    })
    
    # Create horizontal bar chart for satisfaction by query category
    fig = px.bar(
        satisfaction_data.sort_values('satisfaction'),
        y='query_category',
        x='satisfaction',
        orientation='h',
        title='User Satisfaction by Query Category',
        labels={'query_category': 'Query Category', 'satisfaction': 'Satisfaction Score'},
        color='satisfaction',
        color_continuous_scale=px.colors.sequential.Blues,
        height=500
    )
    
    fig.update_layout(
        xaxis_title='Satisfaction Score',
        yaxis_title='Query Category',
        yaxis={'categoryorder': 'total ascending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation between satisfaction and content quality metrics
    # Sample data for correlation matrix
    correlation_data = pd.DataFrame({
        'Metric': ['Relevance', 'Freshness', 'Authority', 'Comprehensiveness', 'Loading Speed', 'Formatting'],
        'Correlation': [0.85, 0.72, 0.68, 0.78, 0.45, 0.62]
    })
    
    # Create horizontal bar chart for correlation
    fig = px.bar(
        correlation_data.sort_values('Correlation'),
        y='Metric',
        x='Correlation',
        orientation='h',
        title='Correlation between Content Metrics and User Satisfaction',
        labels={'Metric': 'Content Metric', 'Correlation': 'Correlation with Satisfaction'},
        color='Correlation',
        color_continuous_scale=px.colors.sequential.Blues,
        height=400
    )
    
    fig.update_layout(
        xaxis_title='Correlation Coefficient',
        yaxis_title='Content Metric',
        yaxis={'categoryorder': 'total ascending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_session_analysis_page(db, query_builder):
    """Render the session analysis page."""
    st.title("Session Analysis")
    st.subheader("Analyze user session patterns and success metrics")
    
    start_date, end_date = st.session_state.get('date_range', (datetime.date(2023, 1, 1), datetime.date.today()))

    metrics_df = db.execute_query_to_df(
        query_builder.get_daily_metrics(str(start_date), str(end_date))
    )
    if not metrics_df.empty:
        st.subheader("Sessions Over Time")
        fig = px.line(metrics_df, x='date', y='total_sessions', markers=True)
        st.plotly_chart(fig, use_container_width=True)

    segment_df = db.execute_query_to_df(
        query_builder.get_user_segment_distribution(str(start_date), str(end_date))
    )
    if not segment_df.empty:
        st.subheader("Sessions by User Segment")
        fig = px.bar(segment_df, x='user_segment', y='session_count', text='percentage', title='User Segment Distribution')
        fig.update_traces(texttemplate='%{text:.1f}%')
        st.plotly_chart(fig, use_container_width=True)

def render_segmentation_page(db, query_builder):
    """Render the segmentation analysis page."""
    st.title("Segmentation Analysis")
    st.subheader("Metrics broken down by different user segments")
    
    start_date, end_date = st.session_state.get('date_range', (datetime.date(2023, 1, 1), datetime.date.today()))

    segment_df = db.execute_query_to_df(
        query_builder.get_user_segment_distribution(str(start_date), str(end_date))
    )
    device_df = db.execute_query_to_df(
        query_builder.get_device_type_distribution(str(start_date), str(end_date))
    )

    if not segment_df.empty:
        st.subheader("User Segments")
        fig = px.bar(segment_df, x='user_segment', y='session_count', text='percentage')
        fig.update_traces(texttemplate='%{text:.1f}%')
        st.plotly_chart(fig, use_container_width=True)

    if not device_df.empty:
        st.subheader("Device Types")
        fig = px.pie(device_df, names='device_type', values='session_count', title='Sessions by Device')
        st.plotly_chart(fig, use_container_width=True)

def render_trends_anomalies_page(db, query_builder):
    """Render the trends and anomalies analysis page."""
    st.title("Trends & Anomalies")
    st.subheader("Trend detection and anomaly identification in search metrics")
    
    start_date, end_date = st.session_state.get('date_range', (datetime.date(2023, 1, 1), datetime.date.today()))

    metrics_df = db.execute_query_to_df(
        query_builder.get_daily_metrics(str(start_date), str(end_date))
    )

    if metrics_df.empty:
        st.info("No data available for selected period")
        return

    st.subheader("CTR Trend with Anomaly Detection")
    ctr_ts = metrics_df[['date', 'avg_ctr']].rename(columns={'avg_ctr': 'ctr'})
    anomaly_df = stats_utils.detect_anomalies(ctr_ts, metric_col='ctr', date_col='date')
    fig = px.line(anomaly_df, x='date', y='ctr', markers=True)
    anomalies = anomaly_df[anomaly_df['ctr_is_anomaly']]
    if not anomalies.empty:
        fig.add_scatter(x=anomalies['date'], y=anomalies['ctr'], mode='markers', marker=dict(color='red', size=10), name='Anomaly')
    fig.update_layout(yaxis_tickformat='.1%')
    st.plotly_chart(fig, use_container_width=True)

def render_ab_testing_page(db, query_builder):
    """Render the A/B testing analysis page."""
    st.title("A/B Testing Analysis")
    st.subheader("Compare metrics between different search experiments")
    
    test_id = st.text_input("A/B Test ID")
    if not test_id:
        st.info("Enter a test ID to view results")
        return

    results_df = db.execute_query_to_df(query_builder.get_ab_test_results(test_id))
    if results_df.empty:
        st.warning("No results found for the specified test")
        return
    st.dataframe(results_df)

def render_data_explorer_page(db, query_builder):
    """Render the data explorer page."""
    st.title("Data Explorer")
    st.subheader("Custom exploration of underlying search data")
    
    user_query = st.text_area("SQL Query")
    if st.button("Run Query") and user_query.strip():
        try:
            df = db.execute_query_to_df(user_query)
            st.dataframe(df)
        except Exception as e:
            st.error(f"Query failed: {e}")

# Main entry point for the dashboard
if __name__ == "__main__":
    main()
