#!/usr/bin/env python3
"""
Hadoop Streaming Reducer: Spatio-Temporal Feature Aggregator.
Groups incoming mapped records by (zone_id, time_bucket) and computes demand,
average fare, trip duration, speed, and binary surge classification target (is_surge_above_1_5x).
"""
import sys
from datetime import datetime

SURGE_THRESHOLD = 1.5

def emit_record(current_key, trip_count, sum_fare, sum_miles, sum_seconds, sum_surge):
    if trip_count == 0 or not current_key:
        return

    zone_id, time_bucket = current_key.split("\t")
    
    avg_fare = sum_fare / trip_count
    avg_miles = sum_miles / trip_count
    avg_seconds = sum_seconds / trip_count
    avg_surge = sum_surge / trip_count

    total_hours = sum_seconds / 3600.0
    avg_speed_mph = (sum_miles / total_hours) if total_hours > 0 else 0.0

    # Extract temporal features from time_bucket (e.g. 2023-05-14 17:15)
    try:
        dt = datetime.strptime(time_bucket, "%Y-%m-%d %H:%M")
        hour = dt.hour
        day_of_week = dt.weekday()
        is_weekend = 1 if day_of_week in [5, 6] else 0
        is_rush_hour = 1 if hour in [7, 8, 9, 16, 17, 18, 19] and is_weekend == 0 else 0
    except ValueError:
        hour, day_of_week, is_weekend, is_rush_hour = 0, 0, 0, 0

    # Binary Classification Target for 20-min forward prediction
    is_surge_above_1_5x = 1 if avg_surge >= SURGE_THRESHOLD else 0

    # Output CSV format:
    # zone_id,time_bucket,hour,day_of_week,is_weekend,is_rush_hour,demand_trips,avg_fare,avg_miles,avg_seconds,avg_speed_mph,avg_surge_multiplier,is_surge_above_1_5x
    print(f"{zone_id},{time_bucket},{hour},{day_of_week},{is_weekend},{is_rush_hour},{trip_count},{avg_fare:.2f},{avg_miles:.2f},{avg_seconds:.1f},{avg_speed_mph:.2f},{avg_surge:.2f},{is_surge_above_1_5x}")

def main():
    current_key = None
    trip_count = 0
    sum_fare = 0.0
    sum_miles = 0.0
    sum_seconds = 0.0
    sum_surge = 0.0

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        parts = line.split("\t")
        if len(parts) < 3:
            continue

        zone_id = parts[0]
        time_bucket = parts[1]
        val_str = parts[2]
        key = f"{zone_id}\t{time_bucket}"

        try:
            val_parts = val_str.split(",")
            fare = float(val_parts[0])
            miles = float(val_parts[1])
            seconds = float(val_parts[2])
            surge = float(val_parts[3])
        except (ValueError, IndexError):
            continue

        if current_key == key:
            trip_count += 1
            sum_fare += fare
            sum_miles += miles
            sum_seconds += seconds
            sum_surge += surge
        else:
            if current_key:
                emit_record(current_key, trip_count, sum_fare, sum_miles, sum_seconds, sum_surge)
            
            current_key = key
            trip_count = 1
            sum_fare = fare
            sum_miles = miles
            sum_seconds = seconds
            sum_surge = surge

    # Emit final group
    if current_key and trip_count > 0:
        emit_record(current_key, trip_count, sum_fare, sum_miles, sum_seconds, sum_surge)

if __name__ == "__main__":
    main()
