"""
Kafka Consumer for Ride Requests and HDFS Staging Sink.
Consumes real-time ride request events from Kafka topic 'ride-requests'
and batches them into HDFS-structured partitioned storage (e.g. /data/raw/taxi_trips/year=YYYY/month=MM/).
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
    KAFKA_CONSUMER_GROUP,
    RAW_DATA_DIR
)

try:
    from kafka import KafkaConsumer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

class HdfsBatchSink:
    """Buffers streaming events and flushes to partitioned HDFS / local storage."""
    def __init__(self, base_dir: Path = RAW_DATA_DIR, batch_size: int = 100):
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.buffer = []

    def add_event(self, event: dict):
        self.buffer.append(event)
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        df = pd.DataFrame(self.buffer)
        
        # Partition by year/month
        df["dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["year"] = df["dt"].dt.year.fillna(2023).astype(int)
        df["month"] = df["dt"].dt.strftime("%m").fillna("01")

        for (yr, mo), group in df.groupby(["year", "month"]):
            part_dir = self.base_dir / f"year={yr}" / f"month={mo}"
            part_dir.mkdir(parents=True, exist_ok=True)
            batch_file = part_dir / f"stream_batch_{int(time.time() * 1000)}.csv"
            clean_cols = [c for c in group.columns if c not in ["dt", "year", "month"]]
            group[clean_cols].to_csv(batch_file, index=False)
            print(f"[HDFS SINK] Flushed {len(group)} records to -> {batch_file.relative_to(self.base_dir.parent)}")

        self.buffer.clear()

def consume_and_sink(max_messages: int = 50, timeout_ms: int = 5000):
    """Consumes ride events from Kafka and writes to storage sink."""
    if not KAFKA_AVAILABLE:
        print("[!] kafka-python not installed. Running simulated sink.")
        return

    sink = HdfsBatchSink(RAW_DATA_DIR, batch_size=25)
    try:
        consumer = KafkaConsumer(
            KAFKA_TOPIC_RIDE_REQUESTS,
            bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            group_id=KAFKA_CONSUMER_GROUP,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            consumer_timeout_ms=timeout_ms
        )
        print(f"[*] Subscribed to Kafka topic '{KAFKA_TOPIC_RIDE_REQUESTS}'. Listening for incoming rides...")
        
        count = 0
        for msg in consumer:
            event = msg.value
            sink.add_event(event)
            count += 1
            if count >= max_messages:
                break

        sink.flush()
        print(f"[] Consumer completed. Total records processed and staged to HDFS: {count}")
    except Exception as e:
        print(f"[!] Kafka connection failed ({e}). (Ensure Kafka broker is running or use mock mode).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consume ride requests from Kafka to HDFS")
    parser.add_argument("--max-messages", type=int, default=100)
    args = parser.parse_args()
    consume_and_sink(max_messages=args.max_messages)
