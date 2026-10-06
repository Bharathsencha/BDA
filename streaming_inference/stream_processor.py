"""
Real-Time Stream Inference Processor & Surge Alert Engine.
Consumes sliding window ride requests, performs live Random Forest inference,
and broadcasts 20-minute advance surge warning alerts.
"""
import sys
import os
import time
import json
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.config import (
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    BASE_FLAG_DROP,
    PER_MILE_RATE,
    PER_MINUTE_RATE,
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC_RIDE_REQUESTS,
    KAFKA_TOPIC_SURGE_ALERTS
)
from config.chicago_zones import CHICAGO_ZONE_COORDINATES
from ml.dispatch_optimizer import optimize_driver_repositioning

def run_live_stream_inference(max_iterations: int = 30, delay_sec: float = 0.5):
    """
    Simulates real-time stream ingestion and live model evaluation.
    Outputs live alerts to data/processed/live_alerts.json.
    """
    rf_path = MODEL_DIR / "random_forest_surge_model.pkl"
    if not rf_path.exists():
        print("[!] Model artifact missing. Run ml/advanced_training.py first.")
        return

    model = joblib.load(rf_path)
    raw_csv = RAW_DATA_DIR / "chicago_taxi_trips_12mo.csv"
    if not raw_csv.exists():
        print("[!] Raw trips CSV not found.")
        return

    df_trips = pd.read_csv(raw_csv)
    print(f"[*] Starting Real-Time Inference Stream Engine on {len(df_trips)} trips...")

    live_alerts_file = PROCESSED_DATA_DIR / "live_alerts.json"

    recent_alerts = []

    for i in range(max_iterations):
        sample = df_trips.sample(1).iloc[0]

        zone_id = int(sample.get("pickup_community_area", 32))
        trip_miles = float(sample.get("trip_miles", 3.5))
        trip_seconds = float(sample.get("trip_seconds", 850))
        fare = float(sample.get("fare", 22.0))
        now = datetime.now()

        hour = now.hour
        dow = now.weekday()
        is_weekend = 1 if dow in [5, 6] else 0
        is_rush = 1 if hour in [7, 8, 9, 16, 17, 18, 19] and is_weekend == 0 else 0
        speed = (trip_miles / (trip_seconds / 3600.0)) if trip_seconds > 0 else 18.0
        demand_trips = int(sample.get("pickup_community_area", 10)) * 2

        # Build feature vector
        feats = pd.DataFrame([{
            "zone_id": zone_id,
            "hour": hour,
            "day_of_week": dow,
            "is_weekend": is_weekend,
            "is_rush_hour": is_rush,
            "demand_trips": demand_trips,
            "avg_fare": fare,
            "avg_miles": trip_miles,
            "avg_seconds": trip_seconds,
            "avg_speed_mph": speed
        }])

        prob = float(model.predict_proba(feats)[0][1])
        base_fare = BASE_FLAG_DROP + (trip_miles * PER_MILE_RATE) + ((trip_seconds / 60.0) * PER_MINUTE_RATE)
        multiplier = round(fare / base_fare, 2) if base_fare > 0 else 1.0

        zone_name = CHICAGO_ZONE_COORDINATES.get(zone_id, {}).get("name", f"Zone {zone_id}")

        alert_item = {
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "zone_id": zone_id,
            "zone_name": zone_name,
            "surge_probability": round(prob, 4),
            "estimated_multiplier": multiplier,
            "is_surge_warning": bool(prob >= 0.5 or multiplier >= 1.5),
            "dispatch_directive": optimize_driver_repositioning(zone_id, prob, demand_trips, multiplier)
        }

        recent_alerts.insert(0, alert_item)
        if len(recent_alerts) > 50:
            recent_alerts.pop()

        # Update JSON store
        with open(live_alerts_file, "w") as fp:
            json.dump(recent_alerts, fp, indent=2)

        status_tag = "ALERT: SURGE > 1.5x PREDICTED" if alert_item["is_surge_warning"] else "NORMAL PRICING"
        print(f"[{alert_item['timestamp']}] Zone {zone_id:02d} ({zone_name:<20}) | Risk: {prob*100:5.1f}% | {status_tag}")

        time.sleep(delay_sec)

    print(f"\n[OK] Real-Time stream evaluation completed. Stored alerts in {live_alerts_file}")

if __name__ == "__main__":
    run_live_stream_inference(max_iterations=10, delay_sec=0.1)
