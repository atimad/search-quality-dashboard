"""
Sample notebook demonstrating search quality analysis using the Search Quality Metrics Dashboard.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add parent directory to path to import project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), "..")))

# Import project modules
from src.data.data_generator import SearchDataGenerator
from src.data.data_processor import SearchDataProcessor
from src.metrics.engagement_metrics import EngagementMetrics
from src.metrics.relevance_metrics import RelevanceMetrics
from src.metrics.statistical_utils import ABTestAnalyzer, TrendDetector, AnomalyDetector
from src.database.db_connector import SearchDatabaseConnector
from src.database.sql_queries import SearchQueryBuilder

# Set up plotting
plt.style.use('seaborn-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# %% [markdown]
# # Search Quality Analysis Notebook
# 
# This notebook demonstrates how to use the Search Quality Metrics Dashboard components to analyze search quality data. It shows examples of calculating metrics, visualizing results, and identifying trends and anomalies.

# %% [markdown]
# ## 1. Generate Sample Data
# 
# First, let's generate some sample search data using the data generator module.

# %%
# Initialize data generator
generator_config = {
    'num_users': 1000,
    'num_days': 30
}

generator = SearchDataGenerator(generator_config)

# Generate sample data
sample_data = generator.create_sample_data(sample_size=5000, return_df=True)

# Display sample data
sample_data.head()

# %% [markdown]
# ## 2. Calculate Engagement Metrics
# 
# Let's calculate some key engagement metrics from the sample data.

# %%
# Initialize engagement metrics calculator
engagement_metrics = EngagementMetrics()

# Calculate CTR
ctr = engagement_metrics.calculate_ctr(sample_data)
print(f"Overall CTR: {ctr:.2%}")

# Calculate CTR by device type
ctr_by_device = engagement_metrics.calculate_ctr_by_segment(sample_data, 'device_type')
print("\nCTR by Device Type:")
for device, value in ctr_by_device.items():
    print(f"{device}: {value:.2%}")

# Calculate time to first click
ttfc = engagement_metrics.calculate_time_to_first_click(sample_data)
print(f"\nAverage Time to First Click: {ttfc:.2f} seconds")

# Calculate zero-click rate
zcr = engagement_metrics.calculate_zero_click_rate(sample_data)
print(f"\nZero-Click Rate: {zcr:.2%}")

# %% [markdown]
# Let's visualize these engagement metrics:

# %%
# Visualize CTR by device type
plt.figure(figsize=(10, 6))
devices = list(ctr_by_device.keys())
ctrs = list(ctr_by_device.values())
bars = plt.bar(devices, ctrs, color=['#1E88E5', '#FFC107', '#4CAF50'])
plt.title('Click-Through Rate by Device Type', fontsize=16)
plt.ylabel('CTR')
plt.ylim(0, max(ctrs) * 1.2)

# Add percentage labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.2%}', ha='center', va='bottom', fontsize=12)

plt.show()

# %% [markdown]
# ## 3. Calculate Relevance Metrics
# 
# Now, let's calculate some relevance metrics to evaluate search quality.

# %%
# Initialize relevance metrics calculator
relevance_metrics = RelevanceMetrics()

# Calculate NDCG
ndcg = relevance_metrics.calculate_ndcg(sample_data)
print(f"Overall NDCG: {ndcg:.4f}")

# Calculate MRR (Mean Reciprocal Rank)
mrr = relevance_metrics.calculate_mrr(sample_data)
print(f"Mean Reciprocal Rank: {mrr:.4f}")

# Calculate query reformulation rate
qrr = relevance_metrics.calculate_reformulation_rate(sample_data)
print(f"Query Reformulation Rate: {qrr:.2%}")

# %% [markdown]
# ## 4. Perform A/B Test Analysis
# 
# Let's simulate an A/B test to compare two search algorithms.

# %%
# Initialize A/B test analyzer
ab_analyzer = ABTestAnalyzer()

# Simulate two groups with different CTRs
group_a_clicks = np.random.binomial(1, 0.32, 10000)
group_b_clicks = np.random.binomial(1, 0.35, 10000)

# Perform statistical test
result = ab_analyzer.compare_metric(group_a_clicks, group_b_clicks, 'CTR')

print(f"Group A CTR: {np.mean(group_a_clicks):.2%}")
print(f"Group B CTR: {np.mean(group_b_clicks):.2%}")
print(f"Absolute Improvement: {np.mean(group_b_clicks) - np.mean(group_a_clicks):.2%}")
print(f"Relative Improvement: {(np.mean(group_b_clicks) / np.mean(group_a_clicks) - 1):.2%}")
print(f"p-value: {result['p_value']:.6f}")
print(f"Statistically Significant: {result['significant']}")

# Visualize A/B test results
plt.figure(figsize=(10, 6))
means = [np.mean(group_a_clicks), np.mean(group_b_clicks)]
errors = [
    ab_analyzer.calculate_confidence_interval(group_a_clicks)[1] - np.mean(group_a_clicks),
    ab_analyzer.calculate_confidence_interval(group_b_clicks)[1] - np.mean(group_b_clicks)
]

bars = plt.bar(['Control (A)', 'Variant (B)'], means, yerr=errors, capsize=10, color=['#1E88E5', '#4CAF50'])
plt.title('A/B Test Results: Click-Through Rate', fontsize=16)
plt.ylabel('CTR')
plt.ylim(0, max(means) * 1.2)

# Add percentage labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
            f'{height:.2%}', ha='center', va='bottom', fontsize=12)

plt.show()

# %% [markdown]
# ## 5. Detect Trends and Anomalies
# 
# Let's generate some time series data and detect trends and anomalies.

# %%
# Initialize trend and anomaly detectors
trend_detector = TrendDetector()
anomaly_detector = AnomalyDetector()

# Generate sample time series data with trend and seasonality
dates = pd.date_range(start='2023-01-01', periods=90, freq='D')
base = 0.32  # Base CTR
trend = np.linspace(0, 0.05, 90)  # Upward trend
day_of_week = np.array([0.01 if d.dayofweek < 5 else -0.02 for d in dates])  # Weekday/weekend effect
noise = np.random.normal(0, 0.01, 90)  # Random noise

# Combine components
ctr_values = base + trend + day_of_week + noise

# Add a few anomalies
anomaly_indices = [15, 32, 67]
for idx in anomaly_indices:
    ctr_values[idx] = ctr_values[idx] + (0.1 if np.random.random() > 0.5 else -0.1)

# Create DataFrame
ts_data = pd.DataFrame({
    'date': dates,
    'ctr': ctr_values
})

# Detect trend
trend_result = trend_detector.detect_trend(ts_data['ctr'].values)
print(f"Trend Direction: {'Upward' if trend_result['slope'] > 0 else 'Downward'}")
print(f"Trend Strength: {trend_result['strength']:.4f}")
print(f"Statistical Significance: {trend_result['p_value']:.6f}")

# Detect anomalies
anomalies = anomaly_detector.detect_anomalies(ts_data['ctr'].values)
print(f"\nDetected {len(anomalies['indices'])} anomalies at indices: {anomalies['indices']}")

# Visualize time series with trend and anomalies
plt.figure(figsize=(14, 8))
plt.plot(dates, ctr_values, 'b-', label='CTR')

# Add trend line
x = np.arange(len(ctr_values))
trend_line = trend_result['intercept'] + trend_result['slope'] * x
plt.plot(dates, trend_line, 'r--', label='Trend Line')

# Mark anomalies
if anomalies['indices']:
    anomaly_dates = [dates[i] for i in anomalies['indices']]
    anomaly_values = [ctr_values[i] for i in anomalies['indices']]
    plt.scatter(anomaly_dates, anomaly_values, color='red', s=100, label='Anomalies')

plt.title('CTR Trend Analysis with Anomaly Detection', fontsize=16)
plt.ylabel('CTR')
plt.xlabel('Date')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Session Analysis
# 
# Let's analyze user session data to understand user behavior.

# %%
# Group sample data by session
session_data = sample_data.groupby('session_id').agg({
    'query_id': 'count',
    'clicked_results': lambda x: sum(len(clicks) if isinstance(clicks, list) else 0 for clicks in x),
    'user_id': 'first',
    'device_type': 'first',
    'timestamp': ['min', 'max']
}).reset_index()

# Rename columns
session_data.columns = ['session_id', 'num_queries', 'num_clicks', 'user_id', 'device_type', 'start_time', 'end_time']

# Calculate session duration in seconds
session_data['duration'] = (session_data['end_time'] - session_data['start_time']).dt.total_seconds()

# Calculate clicks per query
session_data['clicks_per_query'] = session_data['num_clicks'] / session_data['num_queries']

# Define session success (example definition: at least one click and session duration > 10 seconds)
session_data['success'] = (session_data['num_clicks'] > 0) & (session_data['duration'] > 10)

# Calculate overall session success rate
success_rate = session_data['success'].mean()
print(f"Overall Session Success Rate: {success_rate:.2%}")

# Visualize session metrics
fig, axs = plt.subplots(2, 2, figsize=(16, 12))

# Queries per session
axs[0, 0].hist(session_data['num_queries'], bins=10, color='#1E88E5', alpha=0.7)
axs[0, 0].set_title('Distribution of Queries per Session', fontsize=14)
axs[0, 0].set_xlabel('Number of Queries')
axs[0, 0].set_ylabel('Frequency')

# Clicks per session
axs[0, 1].hist(session_data['num_clicks'], bins=10, color='#FFC107', alpha=0.7)
axs[0, 1].set_title('Distribution of Clicks per Session', fontsize=14)
axs[0, 1].set_xlabel('Number of Clicks')
axs[0, 1].set_ylabel('Frequency')

# Session duration
axs[1, 0].hist(session_data['duration'], bins=20, color='#4CAF50', alpha=0.7)
axs[1, 0].set_title('Distribution of Session Duration', fontsize=14)
axs[1, 0].set_xlabel('Duration (seconds)')
axs[1, 0].set_ylabel('Frequency')

# Success rate by device type
success_by_device = session_data.groupby('device_type')['success'].mean()
axs[1, 1].bar(success_by_device.index, success_by_device.values, color=['#1E88E5', '#FFC107', '#4CAF50'])
axs[1, 1].set_title('Success Rate by Device Type', fontsize=14)
axs[1, 1].set_ylabel('Success Rate')
axs[1, 1].set_ylim(0, 1)

# Add percentage labels
for i, v in enumerate(success_by_device):
    axs[1, 1].text(i, v + 0.02, f'{v:.2%}', ha='center')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Query Analysis
# 
# Let's analyze query patterns to understand what users are searching for.

# %%
# Extract query text and calculate frequency
query_counts = sample_data['query_text'].value_counts().reset_index()
query_counts.columns = ['query_text', 'count']
query_counts = query_counts.sort_values('count', ascending=False).head(15)

# Visualize top queries
plt.figure(figsize=(12, 8))
bars = plt.barh(query_counts['query_text'][::-1], query_counts['count'][::-1], color='#1E88E5')
plt.title('Top 15 Search Queries', fontsize=16)
plt.xlabel('Frequency')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Conclusion
# 
# This notebook has demonstrated various analyses that can be performed using the Search Quality Metrics Dashboard components. The metrics calculated and visualizations created help in understanding user behavior, evaluating search quality, and identifying areas for improvement.
# 
# Key insights from this analysis:
# 
# 1. Engagement metrics show how users interact with search results
# 2. Relevance metrics quantify the quality of search results
# 3. A/B testing helps compare different search algorithms
# 4. Trend and anomaly detection identifies patterns and outliers in metrics
# 5. Session analysis reveals user behavior patterns
# 6. Query analysis shows what users are searching for
# 
# These insights can drive improvements in search algorithms, user interface design, and overall user experience.
