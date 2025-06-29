"""
Initialization script for Search Quality Metrics Dashboard.

This script generates synthetic data and initializes the database.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add src directory to path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
sys.path.append(src_dir)

from data.data_generator import SearchDataGenerator
from data.data_processor import SearchDataProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Initialize the dashboard with synthetic data."""
    parser = argparse.ArgumentParser(description="Initialize the Search Quality Metrics Dashboard")
    parser.add_argument("--num-users", type=int, default=1000, help="Number of users to generate")
    parser.add_argument("--num-days", type=int, default=30, help="Number of days of data to generate")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--sample", action="store_true", help="Generate a sample dataset for testing")
    parser.add_argument("--sample-size", type=int, default=1000, help="Size of sample dataset")
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # Generate synthetic data
    logger.info("Generating synthetic search data...")
    
    generator_config = {
        'num_users': args.num_users,
        'num_days': args.num_days
    }
    
    generator = SearchDataGenerator(generator_config)
    
    if args.sample:
        generator.create_sample_data(sample_size=args.sample_size)
    else:
        generator.generate_data()
    
    # Process data and calculate metrics
    logger.info("Processing data and calculating metrics...")
    
    processor = SearchDataProcessor(args.config)
    processor.process_data()
    
    logger.info("Initialization complete! Run 'streamlit run main.py' to start the dashboard.")

if __name__ == "__main__":
    main()
