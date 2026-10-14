"""
Kafka Producer for Continuous Ride-Request Event Streaming.
Streams real-time ride request events to Kafka topic 'ride-requests'.
Supports both live Kafka cluster connection and standalone dry-run simulation mode.
"""
import sys
import os
import json
import time
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
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

def is_kafka_broker_online(bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS, timeout: float = 0.5) -> bool:
    try:
        import socket
        host, port = bootstrap_servers.split(":")
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except Exception:
        return False

def create_kafka_producer():
    """Initializes KafkaProducer with JSON serialization and quick connection check."""
    if not KAFKA_AVAILABLE or not is_kafka_broker_online():
        print("[*] Live Kafka broker offline. Running in standalone streaming mode.")
        return None
    try:
        producer = KafkaProducer(
            bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            api_version=(2, 5, 0)
        )
        print(f"[] Connected to Kafka Broker at {KAFKA_BOOTSTRAP_SERVERS}")
        return producer
    except Exception as e:
        print(f"[*] Live Kafka broker offline ({e}). Running in standalone streaming mode.")
        return None

def compute_surge_multiplier(fare, miles, seconds):
    base_cost = BASE_FLAG_DROP + (miles * PER_MILE_RATE) + ((seconds / 60.0) * PER_MINUTE_RATE)
    if base_cost <= 0:
        return 1.0
    return round(fare / base_cost, 2)

def stream_ride_events(csv_path: Path, max_events: int = 100, delay_sec: float = 0.05, dry_run: bool = False):
    """Reads trips CSV and produces continuous ride-request events."""
    if not csv_path.exists():
        print(f"[!] Raw data file not found at {csv_path}. Run scripts/download_data.py first.")
        return

    df = pd.read_csv(csv_path)
    print(f"[*] Loaded {len(df)} historical trips from {csv_path.name}")
    print(f"[*] Starting dynamic event stream to topic '{KAFKA_TOPIC_RIDE_REQUESTS}' (events={max_events}, delay={delay_sec}s)...")

    producer = None if dry_run else create_kafka_producer()
    is_live_kafka = producer is not None

    count = 0
    # Stream in continuous batches
    sample_indices = df.sample(min(len(df), max_events)).index

    for idx in sample_indices:
        row = df.loc[idx]
        pickup_val = row.get("pickup_community_area")
        pickup_zone = int(pickup_val) if pd.notna(pickup_val) else 32

        dropoff_val = row.get("dropoff_community_area")
        dropoff_zone = int(dropoff_val) if pd.notna(dropoff_val) else 8

        fare_val = row.get("fare")
        fare = float(fare_val) if pd.notna(fare_val) else 14.50

        miles_val = row.get("trip_miles")
        miles = float(miles_val) if pd.notna(miles_val) else 3.0

        seconds_val = row.get("trip_seconds")
        seconds = float(seconds_val) if pd.notna(seconds_val) else 750.0

        surge_mult = compute_surge_multiplier(fare, miles, seconds)

        event = {
            "event_type": "RIDE_REQUEST",
            "trip_id": str(row.get("trip_id", f"TRIP_{idx:07d}")),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pickup_community_area": pickup_zone,
            "dropoff_community_area": dropoff_zone,
            "trip_miles": miles,
            "trip_seconds": seconds,
            "fare": fare,
            "surge_multiplier": surge_mult,
            "is_surge": bool(surge_mult >= 1.5)
        }

        if is_live_kafka:
            try:
                producer.send(KAFKA_TOPIC_RIDE_REQUESTS, value=event)
            except Exception:
                pass
        
        surge_tag = " [SURGE >= 1.5x]" if event["is_surge"] else ""
        if count < 10 or count % 15 == 0 or event["is_surge"]:
            print(f"[STREAM EVENT {count+1:03d}] Zone: {event['pickup_community_area']:02d} -> {event['dropoff_community_area']:02d} | Dist: {event['trip_miles']}mi | Fare: ${event['fare']:.2f} ({event['surge_multiplier']}x){surge_tag}")

        count += 1
        if count >= max_events:
            break
        if delay_sec > 0:
            time.sleep(delay_sec)

    if is_live_kafka:
        producer.flush()
        print(f"[] Published {count} ride-request events to Kafka topic '{KAFKA_TOPIC_RIDE_REQUESTS}'.")
    else:
        print(f"[] Streamed {count} ride-request events successfully (simulation / fallback mode).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream Chicago Taxi ride requests to Kafka")
    parser.add_argument("--csv", type=str, default=str(RAW_DATA_DIR / "chicago_taxi_trips_12mo.csv"))
    parser.add_argument("--events", type=int, default=50, help="Total events to stream")
    parser.add_argument("--delay", type=float, default=0.03, help="Delay between events in seconds")
    parser.add_argument("--dry-run", action="store_true", help="Force standalone mode without Kafka broker")
    args = parser.parse_args()

    stream_ride_events(Path(args.csv), max_events=args.events, delay_sec=args.delay, dry_run=args.dry_run)
