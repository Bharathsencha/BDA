"""
Configuration settings for Ride-Hailing Surge Price Prediction System.
"""
import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "ml" / "models"

# Ensure directories exist
for p in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC_RIDE_REQUESTS = "ride-requests"
KAFKA_TOPIC_SURGE_ALERTS = "surge-alerts"
KAFKA_CONSUMER_GROUP = "surge-prediction-group"

# HDFS Configuration
HDFS_BASE_URL = os.getenv("HDFS_BASE_URL", "hdfs://localhost:9000")
HDFS_RAW_PATH = "/data/raw/taxi_trips"
HDFS_PROCESSED_PATH = "/data/processed/features"

# Pricing & Surge Parameters (Standard City of Chicago Taxi Rates Baseline)
BASE_FLAG_DROP = 3.25       # Base start fare ($)
PER_MILE_RATE = 2.25        # Rate per mile ($)
PER_MINUTE_RATE = 0.40      # Rate per minute (trip_seconds / 60)
SURGE_MULTIPLIER_THRESHOLD = 1.5  # Surge multiplier threshold for classification

# Spatio-Temporal Aggregation
TIME_WINDOW_MINUTES = 15    # Aggregation window (15 mins)
PREDICTION_LEAD_MINUTES = 20 # Predict surge 20 minutes in advance
TOTAL_CHICAGO_ZONES = 77    # Chicago Community Areas (1 - 77)

# City of Chicago Socrata Open Data Portal Endpoint
CHICAGO_TAXI_API_ENDPOINT = "https://data.cityofchicago.org/resource/wrvz-psew.json"
