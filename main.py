"""
Main entry point for the Search Quality Metrics Dashboard.

This script sets up the application and runs the dashboard.
"""

import os
import sys
import logging
from pathlib import Path

# Add src directory to path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
sys.path.append(src_dir)

from visualization.dashboard import main

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/dashboard.log"),
        logging.StreamHandler()
    ]
)

# Ensure log directory exists
os.makedirs("logs", exist_ok=True)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting Search Quality Metrics Dashboard")
    main()
