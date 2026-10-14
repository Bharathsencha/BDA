#!/usr/bin/env python3
"""
Hadoop Streaming Mapper: Spatio-Temporal Event Extractor.
Reads raw Chicago Taxi CSV records from standard input.
Emits pickup and drop-off events mapped to 15-minute time windows:
  Key  : zone_id \t time_bucket
  Value: event_type (P=Pickup, D=Dropoff) , trip_miles , trip_seconds , surge_multiplier
"""
import sys
import os
from datetime import datetime

# Regulated Chicago Taxi Base Rate Constants
BASE_FLAG_DROP = 3.25
PER_MILE_RATE = 2.25
PER_MINUTE_RATE = 0.40

def parse_time_bucket(timestamp_str):
    """Floors timestamp to nearest 15-minute interval for spatio-temporal windowing."""
    clean_ts = timestamp_str.replace("T", " ").split(".")[0].strip()
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%m/%d/%Y %I:%M:%S %p",
        "%m/%d/%Y %H:%M"
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(clean_ts, fmt)
            minute_bucket = (dt.minute // 15) * 15
            bucket_dt = dt.replace(minute=minute_bucket, second=0, microsecond=0)
            return bucket_dt.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            continue
    return None

def main():
    for line in sys.stdin:
        line = line.strip()
        if not line or line.startswith("trip_id") or line.startswith("Trip ID"):
            continue

        parts = line.split(",")
        if len(parts) < 8:
            continue

        try:
            # 0: trip_id, 1: trip_start_timestamp, 3: trip_seconds, 4: trip_miles,
            # 5: pickup_community_area, 6: dropoff_community_area, 7: fare
            timestamp_str = parts[1].strip('"').strip()
            trip_seconds = float(parts[3].strip('"').strip())
            trip_miles = float(parts[4].strip('"').strip())
            pickup_zone_str = parts[5].strip('"').strip()
            dropoff_zone_str = parts[6].strip('"').strip() if len(parts) > 6 else ""
            fare = float(parts[7].strip('"').strip())

            if not pickup_zone_str or trip_miles <= 0 or fare <= 0 or trip_seconds <= 0:
                continue

            pickup_zone = int(float(pickup_zone_str))
            if pickup_zone < 1 or pickup_zone > 77:
                continue

            time_bucket = parse_time_bucket(timestamp_str)
            if not time_bucket:
                continue

            # Compute surge multiplier relative to regulated base fare
            expected_base = BASE_FLAG_DROP + (trip_miles * PER_MILE_RATE) + ((trip_seconds / 60.0) * PER_MINUTE_RATE)
            surge_mult = round(fare / expected_base, 2) if expected_base > 0 else 1.0

            # Emit Pickup event for pickup zone:
            # Key: zone_id \t time_bucket | Value: P,miles,seconds,surge_mult
            print(f"{pickup_zone}\t{time_bucket}\tP,{trip_miles:.2f},{trip_seconds:.0f},{surge_mult:.2f}")

            # Emit Dropoff event for dropoff zone (fleet supply arrival):
            if dropoff_zone_str:
                try:
                    dropoff_zone = int(float(dropoff_zone_str))
                    if 1 <= dropoff_zone <= 77:
                        print(f"{dropoff_zone}\t{time_bucket}\tD,{trip_miles:.2f},{trip_seconds:.0f},1.00")
                except ValueError:
                    pass

        except (ValueError, IndexError):
            continue

if __name__ == "__main__":
    main()
