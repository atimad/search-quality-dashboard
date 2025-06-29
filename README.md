# Search Quality Metrics Dashboard

A comprehensive dashboard for analyzing and visualizing search quality metrics. This project is designed to help search product teams understand user behavior, measure search quality, and make data-driven decisions to improve search experiences.

## Features

- **Data Processing Pipeline**: Processes raw search logs into structured metrics
- **Comprehensive Metrics Suite**: 
  - Engagement metrics (CTR, time-to-click, abandonment rate)
  - Relevance metrics (NDCG, MRR, precision@k)
  - User satisfaction indicators
- **Statistical Analysis**: Confidence intervals, trend detection, and anomaly detection
- **Interactive Dashboard**: Built with Streamlit for real-time exploration of metrics
- **SQL Integration**: Production-ready SQL queries for metrics calculation

## Key Metrics Implemented

1. **Click-Through Rate (CTR)**: Percentage of searches that result in a click
2. **Zero-Click Rate**: Percentage of searches with no clicks
3. **Time to First Click**: Average time between search and first result click
4. **Query Reformulation Rate**: Percentage of searches followed by a refined query
5. **Session Success Rate**: Percentage of sessions ending with a successful interaction
6. **Mean Reciprocal Rank (MRR)**: Average position of the first clicked result
7. **Normalized Discounted Cumulative Gain (NDCG)**: Measures ranking quality
8. **Query Diversity**: Distribution of unique queries vs. repeated queries

## Dashboard Pages

1. **Overview**: Key metrics at a glance with time series trends
2. **Query Analysis**: Deep dive into query patterns and performance
3. **User Engagement**: User behavior and interaction patterns
4. **Result Quality**: Evaluation of ranking and relevance
5. **Session Analysis**: Session depth, duration, and success metrics
6. **Segmentation**: Metrics broken down by device, query type, location, and time of day
7. **Trends & Anomalies**: Trend detection and anomaly identification
8. **Data Explorer**: Custom exploration of underlying data

## Getting Started

### Prerequisites

- Python 3.9+
- pip
- virtualenv (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/atimad/search-quality-dashboard.git
cd search-quality-dashboard

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Generating Synthetic Data

The dashboard comes with a data generator that creates realistic synthetic search log data:

```bash
# Generate a full dataset
python init.py --num-users 1000 --num-days 30

# Generate a small sample dataset for testing
python init.py --sample --sample-size 1000
```

### Running the Dashboard

```bash
# Start the Streamlit dashboard
streamlit run main.py
```

The dashboard will be available at `http://localhost:8501`.

## Project Structure

```
search-quality-dashboard/
├── README.md                 # This file
├── requirements.txt          # Dependencies
├── main.py                   # Entry point for the dashboard
├── init.py                   # Data initialization script
├── config/                   # Configuration files
├── data/                     # Data directory
│   ├── raw/                  # Raw search logs
│   └── processed/            # Processed metrics
├── src/                      # Source code
│   ├── data/                 # Data processing modules
│   ├── metrics/              # Metrics calculation
│   ├── database/             # Database connectors and queries
│   └── visualization/        # Dashboard and visualization
├── notebooks/                # Jupyter notebooks for exploration
└── tests/                    # Unit and integration tests
```

## Data Schema

The synthetic search logs include:
- `session_id`: Unique identifier for user session
- `user_id`: Anonymized user identifier
- `query_id`: Unique identifier for each query
- `query_text`: The search query text
- `timestamp`: When the query was executed
- `results`: List of result IDs returned
- `clicked_results`: List of result IDs that were clicked
- `click_timestamps`: Timestamps for each click
- `device_type`: User device (mobile, desktop, tablet)
- `location`: Geographic region

## Usage Tips

1. **Date Range Selection**: Use the date picker in the sidebar to analyze data from specific time periods
2. **Segmentation**: Filter by device type or query type to understand performance for different segments
3. **Trend Detection**: The Trends & Anomalies page automatically identifies significant trends and anomalies
4. **Custom Exploration**: Use the Data Explorer page to run custom queries and analyses

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.