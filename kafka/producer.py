"""
Kafka Producer for Ride-Request Event Streaming.
Streams simulated or historical ride requests to Kafka topic 'ride-requests'.
Supports standalone dry-run simulation mode when a live Kafka cluster is not running.
"""
import sys
import os
import json
import time
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC_RIDE_REQUESTS,
    RAW_DATA_DIR,
    BASE_FLAG_DROP,
    PER_MILE_RATE,
    PER_MINUTE_RATE
)

try:
    from kafka import KafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

def create_kafka_producer():
    """Initializes KafkaProducer with JSON serialization."""
    if not KAFKA_AVAILABLE:
        print("[!] kafka-python library not available.")
        return None
    try:
        producer = KafkaProducer(
            bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            request_timeout_ms=5000,
            max_block_ms=3000
        )
        print(f"[] Connected to Kafka Broker at {KAFKA_BOOTSTRAP_SERVERS}")
        return producer
    except Exception as e:
        print(f"[!] Warning: Could not connect to Kafka Broker ({e}).")
        return None

def compute_surge_multiplier(fare, miles, seconds):
    """Calculates surge multiplier relative to Chicago baseline fare rate."""
    base_cost = BASE_FLAG_DROP + (miles * PER_MILE_RATE) + ((seconds / 60.0) * PER_MINUTE_RATE)
    if base_cost <= 0:
        return 1.0
    return round(fare / base_cost, 2)

def stream_ride_events(csv_path: Path, max_events: int = 100, delay_sec: float = 0.05, dry_run: bool = False):
    """Reads trips CSV and produces ride-request events to Kafka."""
    if not csv_path.exists():
        print(f"[!] Error: Raw data file not found at {csv_path}. Run scripts/download_data.py first.")
        return

    df = pd.read_csv(csv_path)
    print(f"[*] Loaded {len(df)} historical trips from {csv_path.name}")
    print(f"[*] Starting stream to topic '{KAFKA_TOPIC_RIDE_REQUESTS}' (max_events={max_events}, delay={delay_sec}s)...")

    producer = None if dry_run else create_kafka_producer()
    is_live_kafka = producer is not None

    count = 0
    for idx, row in df.iterrows():
        if count >= max_events:
            break

        pickup_val = row.get("pickup_community_area")
        pickup_zone = int(pickup_val) if pd.notna(pickup_val) else 32

        dropoff_val = row.get("dropoff_community_area")
        dropoff_zone = int(dropoff_val) if pd.notna(dropoff_val) else 8

        fare_val = row.get("fare")
        fare = float(fare_val) if pd.notna(fare_val) else 10.0

        miles_val = row.get("trip_miles")
        miles = float(miles_val) if pd.notna(miles_val) else 2.0

        seconds_val = row.get("trip_seconds")
        seconds = float(seconds_val) if pd.notna(seconds_val) else 600.0

        surge_mult = compute_surge_multiplier(fare, miles, seconds)

        event = {
            "event_type": "RIDE_REQUEST",
            "trip_id": str(row.get("trip_id", f"TRIP_{idx:07d}")),
            "timestamp": str(row.get("trip_start_timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))),
            "pickup_community_area": pickup_zone,
            "dropoff_community_area": dropoff_zone,
            "trip_miles": miles,
            "trip_seconds": seconds,
            "fare": fare,
            "surge_multiplier": surge_mult,
            "is_surge": bool(surge_mult >= 1.5)
        }

        if is_live_kafka:
            producer.send(KAFKA_TOPIC_RIDE_REQUESTS, value=event)
        else:
            # Standalone simulation output
            surge_tag = " [SURGE > 1.5x]" if event["is_surge"] else ""
            if count < 10 or count % 20 == 0:
                print(f"[STREAM EVENT {count+1:03d}] Zone: {event['pickup_community_area']:02d} | Fare: ${event['fare']:.2f} | Multiplier: {event['surge_multiplier']}x{surge_tag}")

        count += 1
        if delay_sec > 0:
            time.sleep(delay_sec)

    if is_live_kafka:
        producer.flush()
        print(f"[] Successfully published {count} ride-request events to Kafka topic '{KAFKA_TOPIC_RIDE_REQUESTS}'.")
    else:
        print(f"[] Simulated {count} ride-request streaming events successfully (dry-run / standalone mode).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream Chicago Taxi ride requests to Kafka")
    parser.add_argument("--csv", type=str, default=str(RAW_DATA_DIR / "chicago_taxi_trips_12mo.csv"))
    parser.add_argument("--events", type=int, default=50, help="Total events to stream")
    parser.add_argument("--delay", type=float, default=0.02, help="Delay between events in seconds")
    parser.add_argument("--dry-run", action="store_true", help="Run without connecting to Kafka broker")
    args = parser.parse_args()

    stream_ride_events(Path(args.csv), max_events=args.events, delay_sec=args.delay, dry_run=args.dry_run)
