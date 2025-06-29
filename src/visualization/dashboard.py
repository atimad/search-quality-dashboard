.subheader("Session Explorer")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        device_filter = st.selectbox(
            "Device Type",
            ["All Devices", "Desktop", "Mobile", "Tablet"],
            index=0
        )
    
    with col2:
        query_count_filter = st.selectbox(
            "Number of Queries",
            ["All", "Single Query", "Multiple Queries"],
            index=0
        )
    
    with col3:
        success_filter = st.selectbox(
            "Session Success",
            ["All", "Successful", "Abandoned"],
            index=0
        )
    
    # Build query
    query = """
        SELECT 
            session_id,
            user_id,
            device_type,
            start_time,
            end_time,
            duration_seconds,
            num_queries,
            total_clicks,
            num_reformulations,
            abandoned,
            success_score
        FROM sessions
        WHERE date BETWEEN :start_date AND :end_date
    """
    
    params = {
        'start_date': st.session_state['start_date'].strftime('%Y-%m-%d'),
        'end_date': st.session_state['end_date'].strftime('%Y-%m-%d')
    }
    
    # Add device filter
    if device_filter != "All Devices":
        query += " AND device_type = :device_type"
        params['device_type'] = device_filter.lower()
    
    # Add query count filter
    if query_count_filter == "Single Query":
        query += " AND num_queries = 1"
    elif query_count_filter == "Multiple Queries":
        query += " AND num_queries > 1"
    
    # Add success filter
    if success_filter == "Successful":
        query += " AND abandoned = 0"
    elif success_filter == "Abandoned":
        query += " AND abandoned = 1"
    
    # Add ordering
    query += " ORDER BY start_time DESC LIMIT 1000"
    
    # Execute query
    sessions_data = self.db.execute_query(query, params)
    
    # Display results
    if len(sessions_data) > 0:
        # Format data for display
        display_data = sessions_data.copy()
        
        # Convert timestamps to datetime
        display_data['start_time'] = pd.to_datetime(display_data['start_time'])
        display_data['end_time'] = pd.to_datetime(display_data['end_time'])
        
        # Format duration
        display_data['duration'] = display_data['duration_seconds'].apply(
            lambda x: f"{int(x // 60)}m {int(x % 60)}s" if not pd.isna(x) else "N/A"
        )
        
        # Format success score
        display_data['success'] = display_data['success_score'].apply(
            lambda x: "High" if x == 3 else "Medium" if x == 2 else "Low" if x == 1 else "None"
        )
        
        # Format abandoned flag
        display_data['status'] = display_data['abandoned'].apply(
            lambda x: "Abandoned" if x == 1 else "Successful"
        )
        
        # Select columns for display
        columns_to_display = [
            'session_id', 'user_id', 'device_type', 'start_time', 'duration', 
            'num_queries', 'total_clicks', 'num_reformulations', 'status', 'success'
        ]
        
        display_data = display_data[columns_to_display]
        
        # Rename columns for display
        column_mapping = {
            'session_id': 'Session ID',
            'user_id': 'User ID',
            'device_type': 'Device',
            'start_time': 'Start Time',
            'duration': 'Duration',
            'num_queries': 'Queries',
            'total_clicks': 'Clicks',
            'num_reformulations': 'Reformulations',
            'status': 'Status',
            'success': 'Success'
        }
        
        display_data = display_data.rename(columns=column_mapping)
        
        # Show results count
        st.write(f"Found {len(display_data):,} sessions")
        
        # Show data table
        st.dataframe(display_data, hide_index=True)
        
        # Session details
        st.subheader("Session Details")
        
        # Select a session to view details
        selected_session = st.selectbox(
            "Select Session to View Details",
            sessions_data['session_id'].tolist(),
            format_func=lambda x: f"{x} - {sessions_data[sessions_data['session_id'] == x]['start_time'].iloc[0]}"
        )
        
        if selected_session:
            # Get session details
            session_details = self.db.get_session_details(selected_session)
            
            if session_details and 'session' in session_details and 'queries' in session_details:
                session = session_details['session']
                queries = session_details['queries']
                
                # Display session information
                st.write("**Session Information**")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        label="User ID",
                        value=session['user_id']
                    )
                
                with col2:
                    st.metric(
                        label="Device",
                        value=session['device_type']
                    )
                
                with col3:
                    st.metric(
                        label="Duration",
                        value=f"{int(session['duration_seconds'] // 60)}m {int(session['duration_seconds'] % 60)}s"
                    )
                
                with col4:
                    st.metric(
                        label="Success Score",
                        value=f"{session['success_score']} / 3"
                    )
                
                # Display queries in session
                st.write("**Queries in Session**")
                
                if len(queries) > 0:
                    queries_df = pd.DataFrame(queries)
                    
                    # Format for display
                    display_queries = queries_df.copy()
                    
                    # Convert timestamp to datetime
                    display_queries['timestamp'] = pd.to_datetime(display_queries['timestamp'])
                    
                    # Format metrics
                    if 'reciprocal_rank' in display_queries.columns:
                        display_queries['reciprocal_rank'] = display_queries['reciprocal_rank'].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A")
                    
                    if 'ndcg_5' in display_queries.columns:
                        display_queries['ndcg_5'] = display_queries['ndcg_5'].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A")
                    
                    if 'time_to_first_click' in display_queries.columns:
                        display_queries['time_to_first_click'] = display_queries['time_to_first_click'].apply(lambda x: f"{x:.1f}s" if not pd.isna(x) else "N/A")
                    
                    if 'is_reformulation' in display_queries.columns:
                        display_queries['is_reformulation'] = display_queries['is_reformulation'].apply(lambda x: "Yes" if x == 1 else "No")
                    
                    # Select columns for display
                    columns_to_show = [
                        'query_text', 'query_type', 'timestamp', 'num_results', 
                        'num_clicks', 'time_to_first_click', 'is_reformulation'
                    ]
                    
                    columns_to_show = [col for col in columns_to_show if col in display_queries.columns]
                    
                    display_queries = display_queries[columns_to_show]
                    
                    # Rename columns for display
                    column_mapping = {
                        'query_text': 'Query',
                        'query_type': 'Type',
                        'timestamp': 'Timestamp',
                        'num_results': 'Results',
                        'num_clicks': 'Clicks',
                        'time_to_first_click': 'Time to Click',
                        'is_reformulation': 'Reformulation'
                    }
                    
                    display_queries = display_queries.rename(columns=column_mapping)
                    
                    # Show data table
                    st.dataframe(display_queries, hide_index=True)
                else:
                    st.info("No queries found for this session.")
            else:
                st.warning("Session details not found.")
        
        # Add export option
        if st.button("Export to CSV"):
            csv = display_data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="session_data.csv",
                mime="text/csv"
            )
    else:
        st.info("No sessions found matching the criteria.")

def _show_metric_explorer(self):
    """Show metric explorer with custom calculations."""
    st.subheader("Metric Explorer")
    
    # Metric selection
    metric_options = [
        "Click-Through Rate (CTR)",
        "Mean Reciprocal Rank (MRR)",
        "NDCG@k",
        "Time to First Click",
        "Query Reformulation Rate",
        "Session Success Rate",
        "Session Abandonment Rate"
    ]
    
    selected_metric = st.selectbox("Select Metric", metric_options)
    
    # Segmentation selection
    segment_options = [
        "Overall",
        "By Date",
        "By Device Type",
        "By Query Type",
        "By Location",
        "By Time of Day"
    ]
    
    selected_segment = st.selectbox("Segment By", segment_options)
    
    # Build and execute query based on selections
    if selected_metric == "Click-Through Rate (CTR)":
        if selected_segment == "Overall":
            # Overall CTR
            query = """
                SELECT 
                    COUNT(*) as total_queries,
                    SUM(CASE WHEN num_clicks > 0 THEN 1 ELSE 0 END) as queries_with_clicks,
                    CAST(SUM(CASE WHEN num_clicks > 0 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) as ctr
                FROM queries
                WHERE date BETWEEN :start_date AND :end_date
            """
            
            metric_data = self.db.execute_query(query, {
                'start_date': st.session_state['start_date'].strftime('%Y-%m-%d'),
                'end_date': st.session_state['end_date'].strftime('%Y-%m-%d')
            })
            
            if len(metric_data) > 0:
                st.metric(
                    label="Overall Click-Through Rate",
                    value=f"{metric_data['ctr'].iloc[0]:.2%}"
                )
                
                st.metric(
                    label="Total Queries",
                    value=f"{metric_data['total_queries'].iloc[0]:,}"
                )
                
                st.metric(
                    label="Queries with Clicks",
                    value=f"{metric_data['queries_with_clicks'].iloc[0]:,}"
                )
            else:
                st.info("No data available for the selected date range.")
        
        elif selected_segment == "By Date":
            # CTR by date
            query = """
                SELECT 
                    date,
                    COUNT(*) as total_queries,
                    SUM(CASE WHEN num_clicks > 0 THEN 1 ELSE 0 END) as queries_with_clicks,
                    CAST(SUM(CASE WHEN num_clicks > 0 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) as ctr
                FROM queries
                WHERE date BETWEEN :start_date AND :end_date
                GROUP BY date
                ORDER BY date
            """
            
            metric_data = self.db.execute_query(query, {
                'start_date': st.session_state['start_date'].strftime('%Y-%m-%d'),
                'end_date': st.session_state['end_date'].strftime('%Y-%m-%d')
            })
            
            if len(metric_data) > 0:
                # Convert date to datetime if it's not
                if not pd.api.types.is_datetime64_any_dtype(metric_data['date']):
                    metric_data['date'] = pd.to_datetime(metric_data['date'])
                
                # Create chart
                fig = px.line(
                    metric_data,
                    x='date',
                    y='ctr',
                    title='Click-Through Rate by Date',
                    labels={'ctr': 'CTR', 'date': 'Date'},
                    height=400
                )
                
                fig.update_layout(
                    xaxis_title='Date',
                    yaxis_title='Click-Through Rate',
                    yaxis_tickformat='.1%',
                    margin=dict(l=40, r=40, t=40, b=40),
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show data table
                display_data = metric_data.copy()
                display_data['ctr'] = display_data['ctr'].apply(lambda x: f"{x:.2%}")
                
                column_mapping = {
                    'date': 'Date',
                    'total_queries': 'Total Queries',
                    'queries_with_clicks': 'Queries with Clicks',
                    'ctr': 'CTR'
                }
                
                display_data = display_data.rename(columns=column_mapping)
                
                st.dataframe(display_data, hide_index=True)
            else:
                st.info("No data available for the selected date range.")
        
        elif selected_segment in ["By Device Type", "By Query Type", "By Location"]:
            # CTR by segment
            segment_column = {
                "By Device Type": "device_type",
                "By Query Type": "query_type",
                "By Location": "location"
            }[selected_segment]
            
            segment_label = {
                "By Device Type": "Device Type",
                "By Query Type": "Query Type",
                "By Location": "Location"
            }[selected_segment]
            
            query = f"""
                SELECT 
                    {segment_column},
                    COUNT(*) as total_queries,
                    SUM(CASE WHEN num_clicks > 0 THEN 1 ELSE 0 END) as queries_with_clicks,
                    CAST(SUM(CASE WHEN num_clicks > 0 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) as ctr
                FROM queries
                WHERE date BETWEEN :start_date AND :end_date
                GROUP BY {segment_column}
                ORDER BY ctr DESC
            """
            
            metric_data = self.db.execute_query(query, {
                'start_date': st.session_state['start_date'].strftime('%Y-%m-%d'),
                'end_date': st.session_state['end_date'].strftime('%Y-%m-%d')
            })
            
            if len(metric_data) > 0:
                # Create chart
                fig = px.bar(
                    metric_data,
                    x=segment_column,
                    y='ctr',
                    title=f'Click-Through Rate by {segment_label}',
                    labels={'ctr': 'CTR', segment_column: segment_label},
                    height=400,
                    text_auto='.1%'
                )
                
                fig.update_layout(
                    xaxis_title=segment_label,
                    yaxis_title='Click-Through Rate',
                    yaxis_tickformat='.1%',
                    margin=dict(l=40, r=40, t=40, b=40),
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show data table
                display_data = metric_data.copy()
                display_data['ctr'] = display_data['ctr'].apply(lambda x: f"{x:.2%}")
                
                column_mapping = {
                    segment_column: segment_label,
                    'total_queries': 'Total Queries',
                    'queries_with_clicks': 'Queries with Clicks',
                    'ctr': 'CTR'
                }
                
                display_data = display_data.rename(columns=column_mapping)
                
                st.dataframe(display_data, hide_index=True)
            else:
                st.info("No data available for the selected date range.")
        
        elif selected_segment == "By Time of Day":
            # CTR by time of day
            query = """
                SELECT 
                    CASE
                        WHEN strftime('%H', timestamp) BETWEEN '00' AND '05' THEN 'Late Night (12AM-6AM)'
                        WHEN strftime('%H', timestamp) BETWEEN '06' AND '11' THEN 'Morning (6AM-12PM)'
                        WHEN strftime('%H', timestamp) BETWEEN '12' AND '17' THEN 'Afternoon (12PM-6PM)'
                        ELSE 'Evening (6PM-12AM)'
                    END as time_of_day,
                    COUNT(*) as total_queries,
                    SUM(CASE WHEN num_clicks > 0 THEN 1 ELSE 0 END) as queries_with_clicks,
                    CAST(SUM(CASE WHEN num_clicks > 0 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) as ctr
                FROM queries
                WHERE date BETWEEN :start_date AND :end_date
                GROUP BY time_of_day
                ORDER BY MIN(strftime('%H', timestamp))
            """
            
            metric_data = self.db.execute_query(query, {
                'start_date': st.session_state['start_date'].strftime('%Y-%m-%d'),
                'end_date': st.session_state['end_date'].strftime('%Y-%m-%d')
            })
            
            if len(metric_data) > 0:
                # Ensure correct time of day order
                time_order = ['Morning (6AM-12PM)', 'Afternoon (12PM-6PM)', 'Evening (6PM-12AM)', 'Late Night (12AM-6AM)']
                metric_data['time_of_day'] = pd.Categorical(metric_data['time_of_day'], categories=time_order, ordered=True)
                metric_data = metric_data.sort_values('time_of_day')
                
                # Create chart
                fig = px.bar(
                    metric_data,
                    x='time_of_day',
                    y='ctr',
                    title='Click-Through Rate by Time of Day',
                    labels={'ctr': 'CTR', 'time_of_day': 'Time of Day'},
                    height=400,
                    text_auto='.1%',
                    category_orders={'time_of_day': time_order}
                )
                
                fig.update_layout(
                    xaxis_title='Time of Day',
                    yaxis_title='Click-Through Rate',
                    yaxis_tickformat='.1%',
                    margin=dict(l=40, r=40, t=40, b=40),
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show data table
                display_data = metric_data.copy()
                display_data['ctr'] = display_data['ctr'].apply(lambda x: f"{x:.2%}")
                
                column_mapping = {
                    'time_of_day': 'Time of Day',
                    'total_queries': 'Total Queries',
                    'queries_with_clicks': 'Queries with Clicks',
                    'ctr': 'CTR'
                }
                
                display_data = display_data.rename(columns=column_mapping)
                
                st.dataframe(display_data, hide_index=True)
            else:
                st.info("No data available for the selected date range.")
    
    # Similar patterns would be implemented for other metrics and segmentations
    # Here I'm showing just the CTR examples for brevity, but the same approach
    # would be used for MRR, NDCG, Time to Click, etc.
    
    else:
        st.info("This metric and segmentation combination is not yet implemented in the explorer.")
        
        # Placeholder for future implementations
        st.write(f"Selected Metric: {selected_metric}")
        st.write(f"Selected Segmentation: {selected_segment}")
        
        # Custom SQL query option
        st.subheader("Custom SQL Query")
        
        custom_sql = st.text_area(
            "Enter Custom SQL Query",
            """-- Example: Daily CTR
SELECT 
    date,
    COUNT(*) as total_queries,
    SUM(CASE WHEN num_clicks > 0 THEN 1 ELSE 0 END) as queries_with_clicks,
    CAST(SUM(CASE WHEN num_clicks > 0 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) as ctr
FROM queries
WHERE date BETWEEN '2023-01-01' AND '2023-12-31'
GROUP BY date
ORDER BY date
"""
        )
        
        if st.button("Run Custom Query"):
            if custom_sql:
                try:
                    custom_data = self.db.execute_query(custom_sql)
                    
                    if len(custom_data) > 0:
                        st.write(f"Found {len(custom_data)} rows")
                        st.dataframe(custom_data)
                        
                        # Add export option
                        csv = custom_data.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name="custom_query_data.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("Query returned no results.")
                except Exception as e:
                    st.error(f"Error executing query: {str(e)}")
            else:
                st.warning("Please enter a SQL query.")

def main():
    """Main function to run the dashboard."""
    dashboard = SearchDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
