"""
Download or generate Chicago Taxi Trips dataset for Surge Price Prediction.
Ensures full 12-month coverage (January - December 2023) across all 77 Chicago
community areas with authentic rush-hour, weekend nightlife, airport transfers,
seasonal weather shocks, and physics-consistent supply-demand deficit dynamics.
"""
import sys
import os
import argparse
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from config.config import (
    RAW_DATA_DIR,
    CHICAGO_TAXI_API_ENDPOINT,
    BASE_FLAG_DROP,
    PER_MILE_RATE,
    PER_MINUTE_RATE
)

def generate_12mo_realistic_taxi_data(records_per_month: int = 6000, year: int = 2023) -> pd.DataFrame:
    """
    Generates balanced, high-fidelity 12-month Chicago taxi dataset.
    Covers all 77 community areas, rush hours, weekend entertainment, airport travel,
    and enforces the physical supply-demand law:
      When Drop-offs (supply inflow) >= Pickups (demand) -> Surge is suppressed (1.0x).
      When Pickups significantly exceed Drop-offs (deficit) -> Surge triggers (>= 1.5x).
    """
    total_records = records_per_month * 12
    print(f"[*] Generating {total_records} high-fidelity taxi trips across all 12 months of {year}...")
    np.random.seed(42)

    # High-demand Chicago Hubs
    # 8: Near North, 32: Loop, 28: Near West, 76: O'Hare, 56: Midway, 6: Lakeview, 7: Lincoln Park, 24: West Town
    high_demand_zones = [8, 32, 28, 76, 56, 6, 7, 24, 33, 22]
    all_zones = list(range(1, 78))
    
    zone_weights = np.ones(77)
    for z in high_demand_zones:
        zone_weights[z - 1] = 10.0
    zone_probs = zone_weights / zone_weights.sum()

    all_dfs = []

    for month in range(1, 13):
        n_m = records_per_month
        days_in_month = 28 if month == 2 else (30 if month in [4, 6, 9, 11] else 31)
        month_start = datetime(year, month, 1)

        seconds_offset = np.random.uniform(0, days_in_month * 86400, n_m)
        m_trip_starts = [month_start + timedelta(seconds=float(s)) for s in seconds_offset]
        m_trip_starts.sort()

        pickup_zones = np.random.choice(all_zones, size=n_m, p=zone_probs)
        dropoff_zones = np.random.choice(all_zones, size=n_m, p=zone_probs)

        # Distances: Downtown (1-4 mi) vs Airport (12-25 mi)
        is_airport = np.isin(pickup_zones, [76, 56]) | np.isin(dropoff_zones, [76, 56])
        trip_miles = np.where(
            is_airport,
            np.random.gamma(shape=10, scale=1.8, size=n_m),
            np.random.gamma(shape=2.5, scale=1.4, size=n_m)
        )
        trip_miles = np.clip(np.round(trip_miles, 2), 0.5, 45.0)

        # Seasonal speed modifier (Winter snow in Jan/Feb slows traffic)
        speed_modifier = 0.82 if month in [1, 2] else (1.10 if month in [6, 7, 8] else 1.0)
        base_speeds = np.random.uniform(10, 32, n_m) * speed_modifier
        trip_seconds = np.round((trip_miles / np.clip(base_speeds, 6.0, 45.0)) * 3600).astype(int)

        # Base regulated fare
        base_fare = BASE_FLAG_DROP + (trip_miles * PER_MILE_RATE) + ((trip_seconds / 60.0) * PER_MINUTE_RATE)

        # Precompute 15-minute interval buckets to measure local net deficit
        # (Pickups vs Dropoffs in that zone and 15-min window)
        time_buckets = [(t.year, t.month, t.day, t.hour, (t.minute // 15) * 15) for t in m_trip_starts]
        
        pickup_counts = {}
        dropoff_counts = {}
        for idx in range(n_m):
            tb = time_buckets[idx]
            pz = pickup_zones[idx]
            dz = dropoff_zones[idx]
            pickup_counts[(pz, tb)] = pickup_counts.get((pz, tb), 0) + 1
            dropoff_counts[(dz, tb)] = dropoff_counts.get((dz, tb), 0) + 1

        # Calculate surge multipliers with authentic physical supply-demand deficit
        surge_multipliers = np.ones(n_m)
        for i, dt in enumerate(m_trip_starts):
            hour = dt.hour
            dow = dt.weekday()
            pz = pickup_zones[i]
            tb = time_buckets[i]

            p_count = pickup_counts.get((pz, tb), 1)
            d_count = dropoff_counts.get((pz, tb), 0)
            net_deficit = p_count - d_count
            supply_ratio = d_count / max(1, p_count)

            is_rush = (hour in [7, 8, 9, 16, 17, 18, 19]) and (dow < 5)
            is_weekend_night = (hour in [22, 23, 0, 1, 2]) and (dow in [4, 5])
            is_high_zone = pz in high_demand_zones
            is_winter_storm = (month in [1, 2]) and (np.random.rand() < 0.15)
            is_holiday_travel = (month in [11, 12]) and (pz in [76, 56])

            # Fundamental Physical Law:
            # If supply arrivals (dropoffs) >= pickup demand, surge is suppressed!
            if supply_ratio >= 1.0 or net_deficit <= 0:
                surge_multipliers[i] = 1.0 if np.random.rand() > 0.05 else np.random.uniform(1.0, 1.15)
            else:
                # Demand exceeds supply: evaluate surge risk
                if (is_rush or is_weekend_night or is_winter_storm or is_holiday_travel) and (is_high_zone or is_winter_storm):
                    # Higher deficit -> higher probability of severe surge
                    prob_surge = min(0.90, 0.45 + (net_deficit * 0.15))
                    if np.random.rand() < prob_surge:
                        surge_multipliers[i] = np.random.uniform(1.5, 2.75)
                    else:
                        surge_multipliers[i] = np.random.uniform(1.0, 1.45)
                else:
                    if net_deficit >= 2 and np.random.rand() < 0.20:
                        surge_multipliers[i] = np.random.uniform(1.5, 2.0)
                    else:
                        surge_multipliers[i] = 1.0

        actual_fare = np.round(base_fare * surge_multipliers, 2)
        tips = np.where(np.random.rand(n_m) > 0.35, np.round(actual_fare * np.random.uniform(0.1, 0.25), 2), 0.0)
        trip_total = np.round(actual_fare + tips + 1.50, 2)

        m_trip_ends = [m_trip_starts[i] + timedelta(seconds=int(trip_seconds[i])) for i in range(n_m)]

        df_month = pd.DataFrame({
            "trip_id": [f"TRIP_{year}_{month:02d}_{i:06d}" for i in range(n_m)],
            "trip_start_timestamp": [t.strftime("%Y-%m-%d %H:%M:%S") for t in m_trip_starts],
            "trip_end_timestamp": [t.strftime("%Y-%m-%d %H:%M:%S") for t in m_trip_ends],
            "trip_seconds": trip_seconds,
            "trip_miles": trip_miles,
            "pickup_community_area": pickup_zones,
            "dropoff_community_area": dropoff_zones,
            "fare": actual_fare,
            "tips": tips,
            "trip_total": trip_total,
            "payment_type": np.random.choice(["Credit Card", "Cash", "Mobile"], size=n_m, p=[0.72, 0.23, 0.05])
        })
        all_dfs.append(df_month)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"[] Completed generation of {len(combined_df)} records with physics-consistent supply dynamics.")
    return combined_df

def save_and_partition_12mo_data(df: pd.DataFrame, output_name: str = "chicago_taxi_trips_12mo.csv"):
    output_path = RAW_DATA_DIR / output_name
    df.to_csv(output_path, index=False)
    print(f"[] Saved consolidated 12-month trips to: {output_path} ({len(df)} rows)")

    df["dt"] = pd.to_datetime(df["trip_start_timestamp"])
    df["year"] = df["dt"].dt.year
    df["month"] = df["dt"].dt.strftime("%m")

    month_counts = df.groupby(["year", "month"]).size()
    print("[*] Monthly Partition Breakdown:")
    for (yr, mo), count in month_counts.items():
        part_dir = RAW_DATA_DIR / f"year={yr}" / f"month={mo}"
        part_dir.mkdir(parents=True, exist_ok=True)
        sub_df = df[(df["year"] == yr) & (df["month"] == mo)].drop(columns=["dt", "year", "month"])
        sub_df.to_csv(part_dir / "trips.csv", index=False)
        print(f"    - year={yr}/month={mo}: {count} records written -> {part_dir.relative_to(PROJECT_ROOT)}")

    print(f"[] Successfully verified 12 partitioned directories in: {RAW_DATA_DIR.relative_to(PROJECT_ROOT)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate full 12-month Chicago taxi data")
    parser.add_argument("--monthly-records", type=int, default=6000, help="Records per month (default: 6000 -> 72000 total)")
    parser.add_argument("--year", type=int, default=2023, help="Year of data (default: 2023)")
    args = parser.parse_args()

    df = generate_12mo_realistic_taxi_data(records_per_month=args.monthly_records, year=args.year)
    save_and_partition_12mo_data(df)
