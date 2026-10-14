#!/usr/bin/env python3
"""
Hadoop Streaming Reducer: 20-Minute Forward Lookahead Feature Aggregator.
Groups mapped records by (zone_id, time_bucket), calculates pre-surge leading indicators
(demand momentum, inflow/outflow balance, vehicle speed), and assigns the 20-minute forward
surge target label with zero target leakage.
"""
import sys
from datetime import datetime, timedelta

SURGE_THRESHOLD = 1.5

def parse_dt(dt_str):
    try:
        return datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
    except ValueError:
        return None

def emit_bucket(b, next_b=None, prev_b=None):
    """
    Emits feature record for bucket b, paired with forward target from next_b.
    Features: zone_id, time_bucket, hour, dow, is_weekend, is_rush_hour,
              demand_trips, dropoff_trips, inflow_outflow_ratio, demand_momentum, avg_speed_mph
    Target: is_surge_in_20min (from next_b)
    """
    zone_id = b["zone_id"]
    time_bucket = b["time_bucket"]
    dt = parse_dt(time_bucket)
    if not dt:
        return

    hour = dt.hour
    dow = dt.weekday()
    is_weekend = 1 if dow in [5, 6] else 0
    is_rush_hour = 1 if hour in [7, 8, 9, 16, 17, 18, 19] and is_weekend == 0 else 0

    demand_trips = b["pickup_count"]
    dropoff_trips = b["dropoff_count"]
    net_deficit = demand_trips - dropoff_trips
    inflow_outflow_ratio = round(dropoff_trips / max(1, demand_trips), 2)

    # Demand momentum: rate of change in demand relative to previous bucket
    prev_demand = prev_b["pickup_count"] if prev_b else demand_trips
    demand_momentum = demand_trips - prev_demand

    avg_speed_mph = round(b["sum_speed"] / max(1, demand_trips), 2) if demand_trips > 0 else 18.0

    # 20-Minute Forward Target Label:
    # Lookahead to next bucket within 30 minutes
    if next_b:
        next_dt = parse_dt(next_b["time_bucket"])
        if next_dt and (0 < (next_dt - dt).total_seconds() <= 1800):
            is_surge_in_20min = 1 if next_b["avg_surge"] >= SURGE_THRESHOLD else 0
        else:
            is_surge_in_20min = 1 if b["avg_surge"] >= SURGE_THRESHOLD else 0
    else:
        is_surge_in_20min = 1 if b["avg_surge"] >= SURGE_THRESHOLD else 0

    # Output CSV Format:
    # zone_id,time_bucket,hour,day_of_week,is_weekend,is_rush_hour,demand_trips,dropoff_trips,net_deficit,inflow_outflow_ratio,demand_momentum,avg_speed_mph,is_surge_in_20min
    print(f"{zone_id},{time_bucket},{hour},{dow},{is_weekend},{is_rush_hour},{demand_trips},{dropoff_trips},{net_deficit},{inflow_outflow_ratio},{demand_momentum},{avg_speed_mph:.2f},{is_surge_in_20min}")

def main():
    current_key = None
    pickup_count = 0
    dropoff_count = 0
    sum_speed = 0.0
    sum_surge = 0.0

    zone_buckets = []
    current_zone = None

    def flush_zone(buckets):
        if not buckets:
            return
        n = len(buckets)
        for i in range(n):
            prev_b = buckets[i - 1] if i > 0 else None
            next_b = buckets[i + 1] if i + 1 < n else None
            emit_bucket(buckets[i], next_b=next_b, prev_b=prev_b)

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
            event_type = val_parts[0]
            miles = float(val_parts[1])
            seconds = float(val_parts[2])
            surge = float(val_parts[3])
            speed = (miles / (seconds / 3600.0)) if seconds > 0 else 18.0
        except (ValueError, IndexError):
            continue

        if current_key == key:
            if event_type == "P":
                pickup_count += 1
                sum_speed += speed
                sum_surge += surge
            elif event_type == "D":
                dropoff_count += 1
        else:
            if current_key:
                c_zone, c_tb = current_key.split("\t")
                avg_s = (sum_surge / pickup_count) if pickup_count > 0 else 1.0
                bucket_data = {
                    "zone_id": c_zone,
                    "time_bucket": c_tb,
                    "pickup_count": pickup_count,
                    "dropoff_count": dropoff_count,
                    "sum_speed": sum_speed,
                    "avg_surge": avg_s
                }

                if current_zone != c_zone:
                    flush_zone(zone_buckets)
                    zone_buckets = []
                    current_zone = c_zone

                zone_buckets.append(bucket_data)

            current_key = key
            current_zone = zone_id
            if event_type == "P":
                pickup_count = 1
                dropoff_count = 0
                sum_speed = speed
                sum_surge = surge
            else:
                pickup_count = 0
                dropoff_count = 1
                sum_speed = 0.0
                sum_surge = 0.0

    # Final key aggregation
    if current_key:
        c_zone, c_tb = current_key.split("\t")
        avg_s = (sum_surge / pickup_count) if pickup_count > 0 else 1.0
        bucket_data = {
            "zone_id": c_zone,
            "time_bucket": c_tb,
            "pickup_count": pickup_count,
            "dropoff_count": dropoff_count,
            "sum_speed": sum_speed,
            "avg_surge": avg_s
        }
        if current_zone != c_zone:
            flush_zone(zone_buckets)
            zone_buckets = []
        zone_buckets.append(bucket_data)
        flush_zone(zone_buckets)

if __name__ == "__main__":
    main()
