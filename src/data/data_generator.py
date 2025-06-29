"""
Data Generator for Search Quality Metrics Dashboard.

This module generates synthetic search log data that mimics real-world search behavior.
The generated data includes user sessions, queries, search results, and user interactions.
"""

import os
import pandas as pd
import numpy as np
import random
import datetime
import uuid
import json
import yaml
from pathlib import Path
from tqdm import tqdm

# Constants for data generation
DEVICE_TYPES = ["desktop", "mobile", "tablet"]
DEVICE_DISTRIBUTION = [0.55, 0.35, 0.1]  # Probability distribution for devices

LOCATION_TYPES = ["US", "Europe", "Asia", "Other"]
LOCATION_DISTRIBUTION = [0.45, 0.25, 0.2, 0.1]  # Probability distribution for locations

QUERY_TYPES = ["navigational", "informational", "transactional"]
QUERY_TYPE_DISTRIBUTION = [0.3, 0.5, 0.2]  # Probability distribution for query types

# User satisfaction levels - impacts click behavior
SATISFACTION_LEVELS = ["high", "medium", "low"]
SATISFACTION_DISTRIBUTION = [0.6, 0.3, 0.1]  # Probability distribution for satisfaction

# Common search queries by category
QUERIES = {
    "navigational": [
        "amazon login", "facebook", "youtube", "gmail", "twitter", "instagram",
        "netflix", "linkedin", "google maps", "yahoo mail", "reddit", "wikipedia",
        "amazon prime", "bank of america login", "ebay", "paypal login", "pinterest",
        "walmart", "target", "chase bank", "wells fargo", "spotify", "hulu",
        "github", "outlook", "office 365", "dropbox", "google drive", "apple store"
    ],
    "informational": [
        "how to make pasta", "best movies 2023", "covid symptoms", 
        "what is blockchain", "weather today", "news", "stock market today",
        "how to lose weight", "best laptop 2023", "vaccine side effects",
        "how to tie a tie", "best restaurants near me", "what is bitcoin",
        "how to change password", "history of internet", "who won the super bowl",
        "calories in banana", "how to make bread", "recipe for chicken soup",
        "who is the president", "how tall is mount everest", "distance to moon",
        "how to create website", "best books 2023", "how to fix wifi problems",
        "what is quantum computing", "history of artificial intelligence",
        "best exercise for back pain", "how to improve memory", "best diet for health"
    ],
    "transactional": [
        "buy iphone 14", "order pizza online", "amazon deals", "book flight tickets",
        "hotel booking", "car rental", "movie tickets", "concert tickets",
        "buy laptop", "online shopping", "food delivery", "order groceries online",
        "cheap flights", "discount codes", "buy ps5", "best price tv", "buy furniture",
        "streaming subscription", "buy shoes online", "car insurance quote",
        "home insurance", "vacation packages", "gym membership", "buy headphones",
        "online courses", "software download", "domain name registration",
        "web hosting services", "online banking", "crypto exchange"
    ]
}

# Predefined results for each query type
RESULTS_TEMPLATES = {
    "navigational": {
        "count": lambda: random.randint(3, 10),  # Fewer results for navigational queries
        "relevance": lambda: sorted([0.9] + [random.uniform(0.1, 0.5) for _ in range(9)], reverse=True)
    },
    "informational": {
        "count": lambda: random.randint(10, 20),
        "relevance": lambda: sorted([random.uniform(0.7, 0.9) for _ in range(3)] + 
                                    [random.uniform(0.3, 0.7) for _ in range(7)] +
                                    [random.uniform(0.1, 0.3) for _ in range(10)], reverse=True)
    },
    "transactional": {
        "count": lambda: random.randint(5, 15),
        "relevance": lambda: sorted([random.uniform(0.6, 0.8) for _ in range(5)] + 
                                    [random.uniform(0.2, 0.6) for _ in range(10)], reverse=True)
    }
}

class SearchDataGenerator:
    """Generate synthetic search log data for analysis."""
    
    def __init__(self, config_path=None):
        """Initialize the data generator with configuration."""
        self.config = self._load_config(config_path)
        self.output_path = Path(self.config.get('output_path', 'data/raw/synthetic_search_logs.csv'))
        self.num_users = self.config.get('num_users', 1000)
        self.num_days = self.config.get('num_days', 30)
        self.sessions_per_user_per_day = self.config.get('sessions_per_user_per_day', [0, 1, 2, 3])
        self.sessions_per_user_weights = self.config.get('sessions_per_user_weights', [0.3, 0.4, 0.2, 0.1])
        self.queries_per_session = self.config.get('queries_per_session', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.queries_per_session_weights = self.config.get('queries_per_session_weights', 
                                                           [0.2, 0.3, 0.2, 0.1, 0.07, 0.05, 0.03, 0.02, 0.02, 0.01])
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # Random seed for reproducibility
        random.seed(self.config.get('random_seed', 42))
        np.random.seed(self.config.get('random_seed', 42))
        
        # Generate user profiles
        self.user_profiles = self._generate_user_profiles()
        
    def _load_config(self, config_path):
        """Load configuration from YAML file or use defaults."""
        default_config = {
            'output_path': 'data/raw/synthetic_search_logs.csv',
            'num_users': 1000,
            'num_days': 30,
            'sessions_per_user_per_day': [0, 1, 2, 3],
            'sessions_per_user_weights': [0.3, 0.4, 0.2, 0.1],
            'queries_per_session': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'queries_per_session_weights': [0.2, 0.3, 0.2, 0.1, 0.07, 0.05, 0.03, 0.02, 0.02, 0.01],
            'random_seed': 42,
            'click_probability_high': 0.9,
            'click_probability_medium': 0.6,
            'click_probability_low': 0.3,
            'max_clicks_per_query': 3,
            'session_abandonment_rate': 0.15,
            'query_reformulation_rate': 0.25,
            'similar_query_rate': 0.7,
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                # Merge with defaults
                default_config.update(config)
        
        return default_config
    
    def _generate_user_profiles(self):
        """Generate profiles for users with consistent behavior patterns."""
        user_profiles = {}
        
        for user_id in range(1, self.num_users + 1):
            user_id = f"user_{user_id}"
            user_profiles[user_id] = {
                'device_preference': np.random.choice(DEVICE_TYPES, p=DEVICE_DISTRIBUTION),
                'location': np.random.choice(LOCATION_TYPES, p=LOCATION_DISTRIBUTION),
                'query_type_preference': np.random.choice(QUERY_TYPES, p=QUERY_TYPE_DISTRIBUTION),
                'satisfaction_level': np.random.choice(SATISFACTION_LEVELS, p=SATISFACTION_DISTRIBUTION),
                'avg_queries_per_session': random.uniform(1, 4),
                'avg_session_duration': random.uniform(2, 15),  # minutes
                'click_probability_modifier': random.uniform(0.8, 1.2),  # Modifier for click probabilities
                'time_to_click_avg': random.uniform(3, 20),  # Average seconds to click
            }
        
        return user_profiles
    
    def _generate_session(self, user_id, session_id, start_time):
        """Generate a search session for a user."""
        user_profile = self.user_profiles[user_id]
        
        # Determine number of queries in this session
        num_queries = np.random.choice(
            self.queries_per_session, 
            p=self.queries_per_session_weights
        )
        
        # Check if session is abandoned (no clicks)
        is_abandoned = random.random() < self.config['session_abandonment_rate']
        
        # Initialize session data
        session_data = []
        current_time = start_time
        
        # Previous query for potential reformulation
        previous_query = None
        previous_query_type = None
        
        for q_idx in range(num_queries):
            # Determine if this is a reformulation of previous query
            is_reformulation = (previous_query is not None and 
                                random.random() < self.config['query_reformulation_rate'])
            
            if is_reformulation and random.random() < self.config['similar_query_rate']:
                # Similar query (e.g. adding or changing a word)
                query_type = previous_query_type
                query_words = previous_query.split()
                
                if random.random() < 0.5 and len(query_words) > 1:
                    # Remove a word
                    word_to_remove = random.randint(0, len(query_words) - 1)
                    query_words.pop(word_to_remove)
                else:
                    # Add or change a word
                    if random.random() < 0.7 or len(query_words) > 5:
                        # Change a word
                        word_to_change = random.randint(0, len(query_words) - 1)
                        new_words = QUERIES[query_type][random.randint(0, len(QUERIES[query_type]) - 1)].split()
                        if new_words:
                            query_words[word_to_change] = new_words[random.randint(0, len(new_words) - 1)]
                    else:
                        # Add a word
                        new_words = QUERIES[query_type][random.randint(0, len(QUERIES[query_type]) - 1)].split()
                        if new_words:
                            query_words.append(new_words[random.randint(0, len(new_words) - 1)])
                
                query_text = " ".join(query_words)
            else:
                # New query
                # User preference influences query type, but with some randomness
                if random.random() < 0.7:  # 70% chance to use preferred query type
                    query_type = user_profile['query_type_preference']
                else:
                    query_type = np.random.choice(QUERY_TYPES, p=QUERY_TYPE_DISTRIBUTION)
                
                query_text = random.choice(QUERIES[query_type])
            
            # Update for potential next reformulation
            previous_query = query_text
            previous_query_type = query_type
            
            # Generate search results
            result_template = RESULTS_TEMPLATES[query_type]
            num_results = result_template["count"]()
            relevance_scores = result_template["relevance"]()[:num_results]
            
            # Create result IDs and their relevance
            results = [f"result_{uuid.uuid4().hex[:8]}" for _ in range(num_results)]
            result_relevance = dict(zip(results, relevance_scores))
            
            # Determine click behavior based on user satisfaction and result relevance
            clicked_results = []
            click_timestamps = []
            
            # Skip clicks if this is an abandoned session
            if not is_abandoned:
                # Base click probability determined by satisfaction level
                if user_profile['satisfaction_level'] == 'high':
                    base_click_prob = self.config['click_probability_high']
                elif user_profile['satisfaction_level'] == 'medium':
                    base_click_prob = self.config['click_probability_medium']
                else:
                    base_click_prob = self.config['click_probability_low']
                
                # Apply user's click modifier
                base_click_prob *= user_profile['click_probability_modifier']
                
                # Determine clicked results
                max_clicks = min(self.config['max_clicks_per_query'], num_results)
                potential_clicks = []
                
                for i, result in enumerate(results):
                    # Higher relevance results more likely to be clicked
                    click_prob = base_click_prob * result_relevance[result]
                    if random.random() < click_prob:
                        potential_clicks.append((result, result_relevance[result], i + 1))
                
                # Sort by relevance and position for realistic click order
                potential_clicks.sort(key=lambda x: (-x[1], x[2]))
                
                # Take up to max_clicks
                selected_clicks = potential_clicks[:random.randint(0, max_clicks)]
                
                # Sort by position to simulate chronological clicks
                selected_clicks.sort(key=lambda x: x[2])
                
                # Record clicks and timestamps
                current_click_time = current_time + datetime.timedelta(seconds=random.uniform(2, 10))
                
                for result, _, _ in selected_clicks:
                    clicked_results.append(result)
                    
                    # Time to click increases with each click
                    time_to_click = random.expovariate(1.0 / user_profile['time_to_click_avg'])
                    current_click_time += datetime.timedelta(seconds=time_to_click)
                    click_timestamps.append(current_click_time.strftime("%Y-%m-%d %H:%M:%S"))
            
            # Create query entry
            query_data = {
                'session_id': session_id,
                'user_id': user_id,
                'query_id': f"query_{uuid.uuid4().hex[:10]}",
                'query_text': query_text,
                'query_type': query_type,
                'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                'results': json.dumps(results),
                'result_relevance': json.dumps({r: float(s) for r, s in result_relevance.items()}),
                'clicked_results': json.dumps(clicked_results),
                'click_timestamps': json.dumps(click_timestamps),
                'num_results': len(results),
                'num_clicks': len(clicked_results),
                'device_type': user_profile['device_preference'],
                'location': user_profile['location'],
                'is_reformulation': 1 if is_reformulation else 0
            }
            
            session_data.append(query_data)
            
            # Update time for next query
            # Longer delay if there were clicks
            if clicked_results:
                # Use the last click time as a base and add some time for reading
                last_click_time = datetime.datetime.strptime(click_timestamps[-1], "%Y-%m-%d %H:%M:%S") if click_timestamps else current_time
                reading_time = datetime.timedelta(seconds=random.uniform(5, 60))
                current_time = last_click_time + reading_time
            else:
                # No clicks, shorter delay before next query
                current_time += datetime.timedelta(seconds=random.uniform(5, 20))
        
        return session_data
    
    def generate_data(self):
        """Generate the full search log dataset."""
        all_data = []
        
        # Start date is num_days before today
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=self.num_days)
        
        print(f"Generating search data for {self.num_users} users over {self.num_days} days...")
        
        # Use tqdm to show progress
        with tqdm(total=self.num_users) as pbar:
            for user_id in self.user_profiles:
                # Generate sessions for each day
                for day in range(self.num_days):
                    current_date = start_date + datetime.timedelta(days=day)
                    
                    # Number of sessions this user has today
                    num_sessions = np.random.choice(
                        self.sessions_per_user_per_day,
                        p=self.sessions_per_user_weights
                    )
                    
                    # Generate each session
                    for s_idx in range(num_sessions):
                        # Random time during the day
                        session_hour = random.randint(7, 23)  # Between 7am and 11pm
                        session_minute = random.randint(0, 59)
                        session_second = random.randint(0, 59)
                        
                        session_start = current_date.replace(
                            hour=session_hour,
                            minute=session_minute,
                            second=session_second
                        )
                        
                        session_id = f"session_{uuid.uuid4().hex[:8]}"
                        session_data = self._generate_session(user_id, session_id, session_start)
                        all_data.extend(session_data)
                
                pbar.update(1)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(all_data)
        
        # Sort by timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Reset datetime to string for CSV storage
        df['timestamp'] = df['timestamp'].dt.strftime("%Y-%m-%d %H:%M:%S")
        
        # Save to CSV
        df.to_csv(self.output_path, index=False)
        print(f"Generated {len(df)} search log entries from {self.num_users} users")
        print(f"Data saved to {self.output_path}")
        
        return df

    def create_sample_data(self, sample_size=1000):
        """Create a smaller sample dataset for testing."""
        if not os.path.exists(self.output_path):
            print("Full dataset not found. Generating full dataset first...")
            self.generate_data()
        
        # Load full dataset
        df = pd.read_csv(self.output_path)
        
        # Take a stratified sample by device type and query type
        sample_indices = []
        
        for device in DEVICE_TYPES:
            for query_type in QUERY_TYPES:
                mask = (df['device_type'] == device) & (df['query_type'] == query_type)
                subset = df[mask]
                
                if len(subset) > 0:
                    # Calculate proportion of this segment in the full dataset
                    prop = len(subset) / len(df)
                    
                    # Calculate how many samples to take from this segment
                    n_samples = max(1, int(prop * sample_size))
                    
                    # Sample from this segment
                    if len(subset) <= n_samples:
                        segment_indices = subset.index.tolist()
                    else:
                        segment_indices = subset.sample(n_samples).index.tolist()
                    
                    sample_indices.extend(segment_indices)
        
        # Get unique indices (in case of overlap)
        sample_indices = list(set(sample_indices))
        
        # If we don't have enough samples, add more randomly
        if len(sample_indices) < sample_size:
            remaining = sample_size - len(sample_indices)
            # Get indices not already in sample_indices
            remaining_indices = df.index.difference(sample_indices).tolist()
            
            if remaining_indices:
                # Sample from remaining indices
                additional_indices = random.sample(
                    remaining_indices, 
                    min(remaining, len(remaining_indices))
                )
                sample_indices.extend(additional_indices)
        
        # If we have too many samples, subsample
        if len(sample_indices) > sample_size:
            sample_indices = random.sample(sample_indices, sample_size)
        
        # Create sample dataframe
        sample_df = df.loc[sample_indices].copy()
        
        # Save sample to file
        sample_path = str(self.output_path).replace('.csv', '_sample.csv')
        sample_df.to_csv(sample_path, index=False)
        
        print(f"Created sample dataset with {len(sample_df)} entries")
        print(f"Sample data saved to {sample_path}")
        
        return sample_df

def main():
    """Main function to generate data when script is run directly."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic search log data')
    parser.add_argument('--config', type=str, help='Path to configuration YAML file')
    parser.add_argument('--sample', action='store_true', help='Create a sample dataset')
    parser.add_argument('--sample-size', type=int, default=1000, help='Size of sample dataset')
    args = parser.parse_args()
    
    generator = SearchDataGenerator(config_path=args.config)
    
    if args.sample:
        generator.create_sample_data(sample_size=args.sample_size)
    else:
        generator.generate_data()

if __name__ == "__main__":
    main()
