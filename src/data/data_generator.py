import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import uuid
from typing import Dict, List, Tuple

class SearchDataGenerator:
    """
    Generates synthetic search log data for testing and development.
    
    This class creates realistic search session data including:
    - User queries and query reformulations
    - Click behavior and engagement metrics
    - Temporal patterns and seasonality
    - Different user segments and behavior patterns
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the data generator with a random seed for reproducibility.
        
        Args:
            seed: Random seed for reproducible data generation
        """
        np.random.seed(seed)
        random.seed(seed)
        
        # Common search terms and topics
        self.search_topics = [
            "python programming", "machine learning", "data science", "web development",
            "artificial intelligence", "statistics", "database design", "cloud computing",
            "mobile development", "cybersecurity", "blockchain", "devops",
            "react framework", "node.js", "docker containers", "kubernetes",
            "tensorflow", "pytorch", "sql queries", "data visualization"
        ]
        
        # Query variations and reformulations
        self.query_modifiers = [
            "tutorial", "guide", "examples", "best practices", "tips",
            "2024", "beginner", "advanced", "free", "online",
            "how to", "what is", "why", "when", "where"
        ]
        
        # Result types
        self.result_types = ["web", "video", "image", "pdf", "stackoverflow", "github"]
        
        # Device types
        self.device_types = ["desktop", "mobile", "tablet"]
        
        # User segments
        self.user_segments = ["new_user", "regular_user", "power_user", "enterprise_user"]
    
    def generate_search_logs(self, 
                           num_sessions: int = 10000,
                           start_date: str = "2024-01-01",
                           end_date: str = "2024-12-31") -> pd.DataFrame:
        """
        Generate synthetic search session data.
        
        Args:
            num_sessions: Number of search sessions to generate
            start_date: Start date for the data range
            end_date: End date for the data range
            
        Returns:
            DataFrame with search session data
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        sessions = []
        
        for i in range(num_sessions):
            session_data = self._generate_session(start_dt, end_dt)
            sessions.extend(session_data)
        
        df = pd.DataFrame(sessions)
        return df.sort_values('timestamp').reset_index(drop=True)
    
    def _generate_session(self, start_dt: datetime, end_dt: datetime) -> List[Dict]:
        """
        Generate a single search session with multiple queries and interactions.
        
        Args:
            start_dt: Start datetime for the session
            end_dt: End datetime for the session
            
        Returns:
            List of dictionaries representing search events in the session
        """
        session_id = str(uuid.uuid4())
        user_id = str(uuid.uuid4())
        
        # Random timestamp within the date range
        session_start = start_dt + timedelta(
            seconds=random.randint(0, int((end_dt - start_dt).total_seconds()))
        )
        
        # User characteristics
        user_segment = random.choice(self.user_segments)
        device_type = random.choice(self.device_types)
        
        # Determine session characteristics based on user segment
        if user_segment == "power_user":
            num_queries = np.random.poisson(5) + 1
            avg_time_between_queries = 30  # seconds
        elif user_segment == "regular_user":
            num_queries = np.random.poisson(2) + 1
            avg_time_between_queries = 60
        else:  # new_user or enterprise_user
            num_queries = np.random.poisson(1) + 1
            avg_time_between_queries = 90
        
        session_events = []
        current_time = session_start
        initial_query = self._generate_query()
        
        for query_idx in range(num_queries):
            if query_idx == 0:
                query = initial_query
            else:
                # Generate reformulated query
                query = self._generate_reformulated_query(initial_query, query_idx)
            
            # Generate search results and interactions
            query_events = self._generate_query_events(
                session_id, user_id, query, current_time, 
                user_segment, device_type, query_idx
            )
            session_events.extend(query_events)
            
            # Update time for next query
            time_gap = np.random.exponential(avg_time_between_queries)
            current_time += timedelta(seconds=time_gap)
        
        return session_events
    
    def _generate_query(self) -> str:
        """Generate a realistic search query."""
        base_topic = random.choice(self.search_topics)
        
        if random.random() < 0.3:  # 30% chance of adding modifier
            modifier = random.choice(self.query_modifiers)
            if random.random() < 0.5:
                query = f"{modifier} {base_topic}"
            else:
                query = f"{base_topic} {modifier}"
        else:
            query = base_topic
        
        return query
    
    def _generate_reformulated_query(self, original_query: str, attempt: int) -> str:
        """Generate a reformulated version of the original query."""
        if attempt == 1:
            # First reformulation - usually add specificity
            modifiers = ["how to", "tutorial", "example", "guide"]
            modifier = random.choice(modifiers)
            return f"{modifier} {original_query}"
        elif attempt == 2:
            # Second reformulation - might remove words or change approach
            if len(original_query.split()) > 2:
                words = original_query.split()
                return " ".join(words[:-1])  # Remove last word
            else:
                return f"{original_query} examples"
        else:
            # Further reformulations - more specific or different angle
            specific_modifiers = ["advanced", "beginner", "free", "2024", "best"]
            modifier = random.choice(specific_modifiers)
            return f"{modifier} {original_query}"
    
    def _generate_query_events(self, session_id: str, user_id: str, 
                             query: str, timestamp: datetime,
                             user_segment: str, device_type: str, 
                             query_index: int) -> List[Dict]:
        """Generate events for a single query within a session."""
        events = []
        
        # Search event
        search_event = {
            'session_id': session_id,
            'user_id': user_id,
            'timestamp': timestamp,
            'event_type': 'search',
            'query': query,
            'query_length': len(query),
            'query_words': len(query.split()),
            'query_index_in_session': query_index,
            'user_segment': user_segment,
            'device_type': device_type,
            'results_returned': random.randint(5, 50),
            'search_latency_ms': np.random.gamma(2, 50)  # Realistic latency distribution
        }
        events.append(search_event)
        
        # Generate clicks based on user behavior patterns
        num_clicks = self._determine_num_clicks(user_segment, query_index)
        
        for click_idx in range(num_clicks):
            click_time = timestamp + timedelta(seconds=random.uniform(1, 30))
            
            # Click position bias - higher positions more likely to be clicked
            position = np.random.choice(range(1, 11), p=[0.3, 0.2, 0.15, 0.1, 0.08, 0.06, 0.04, 0.03, 0.02, 0.02])
            
            # Time spent on result
            time_on_result = self._generate_time_on_result(position, user_segment)
            
            click_event = {
                'session_id': session_id,
                'user_id': user_id,
                'timestamp': click_time,
                'event_type': 'click',
                'query': query,
                'query_index_in_session': query_index,
                'user_segment': user_segment,
                'device_type': device_type,
                'click_position': position,
                'result_type': random.choice(self.result_types),
                'time_on_result_seconds': time_on_result,
                'is_satisfied': time_on_result > 30,  # Simple satisfaction heuristic
                'click_index_in_query': click_idx
            }
            events.append(click_event)
        
        return events
    
    def _determine_num_clicks(self, user_segment: str, query_index: int) -> int:
        """Determine number of clicks based on user behavior patterns."""
        base_click_prob = {
            'power_user': 0.8,
            'regular_user': 0.6,
            'new_user': 0.4,
            'enterprise_user': 0.7
        }
        
        # Decrease click probability for later queries in session
        prob_modifier = 0.9 ** query_index
        click_prob = base_click_prob[user_segment] * prob_modifier
        
        if random.random() < click_prob:
            # Generate number of clicks (usually 1-3)
            return min(np.random.poisson(1.5) + 1, 5)
        else:
            return 0
    
    def _generate_time_on_result(self, position: int, user_segment: str) -> float:
        """Generate realistic time spent on clicked result."""
        # Base time influenced by position (higher positions get more time)
        base_time = max(10, 60 - (position * 5))
        
        # User segment influence
        segment_multiplier = {
            'power_user': 1.2,
            'regular_user': 1.0,
            'new_user': 0.8,
            'enterprise_user': 1.5
        }
        
        multiplier = segment_multiplier[user_segment]
        
        # Generate time with some randomness
        time_on_result = np.random.exponential(base_time * multiplier)
        
        return max(1.0, time_on_result)  # Minimum 1 second
    
    def save_data(self, df: pd.DataFrame, filepath: str):
        """Save generated data to CSV file."""
        df.to_csv(filepath, index=False)
        print(f"Generated {len(df)} search events saved to {filepath}")

if __name__ == "__main__":
    # Example usage
    generator = SearchDataGenerator(seed=42)
    
    # Generate data
    search_data = generator.generate_search_logs(
        num_sessions=1000,
        start_date="2024-01-01",
        end_date="2024-12-31"
    )
    
    # Save to file
    generator.save_data(search_data, "data/raw/synthetic_search_logs.csv")
    
    # Display basic statistics
    print(f"Total events: {len(search_data)}")
    print(f"Unique sessions: {search_data['session_id'].nunique()}")
    print(f"Unique users: {search_data['user_id'].nunique()}")
    print(f"Event types: {search_data['event_type'].value_counts()}")
