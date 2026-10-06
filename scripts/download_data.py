"""
Download or generate Chicago Taxi Trips dataset for Surge Price Prediction.
Fetches real data from City of Chicago Open Data Portal or generates high-fidelity
realistic trip logs covering all 77 community areas with peak rush-hour demand patterns.
"""
import sys
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.config import (
    RAW_DATA_DIR,
    CHICAGO_TAXI_API_ENDPOINT,
    BASE_FLAG_DROP,
    PER_MILE_RATE,
    PER_MINUTE_RATE
)

def fetch_chicago_portal_data(limit: int = 20000) -> pd.DataFrame:
    """Fetch real Chicago taxi trip data from City of Chicago API."""
    print(f"[*] Attempting to fetch {limit} records from City of Chicago Open Data Portal...")
    params = {
        "$limit": limit,
        "$where": "trip_miles > 0 AND fare > 0 AND trip_seconds > 60 AND pickup_community_area IS NOT NULL",
        "$order": "trip_start_timestamp DESC"
    }
    try:
        response = requests.get(CHICAGO_TAXI_API_ENDPOINT, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            if len(data) > 0:
                df = pd.DataFrame(data)
                # Normalize column names
                rename_map = {
                    "trip_id": "trip_id",
                    "trip_start_timestamp": "trip_start_timestamp",
                    "trip_end_timestamp": "trip_end_timestamp",
                    "trip_seconds": "trip_seconds",
                    "trip_miles": "trip_miles",
                    "pickup_community_area": "pickup_community_area",
                    "dropoff_community_area": "dropoff_community_area",
                    "fare": "fare",
                    "tips": "tips",
                    "trip_total": "trip_total",
                    "payment_type": "payment_type"
                }
                cols_present = [c for c in rename_map.keys() if c in df.columns]
                df = df[cols_present]
                
                # Convert types
                numeric_cols = ["trip_seconds", "trip_miles", "pickup_community_area", "dropoff_community_area", "fare", "trip_total"]
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                
                df.dropna(subset=["pickup_community_area", "trip_miles", "fare", "trip_start_timestamp"], inplace=True)
                df["pickup_community_area"] = df["pickup_community_area"].astype(int)
                df["dropoff_community_area"] = df["dropoff_community_area"].fillna(df["pickup_community_area"]).astype(int)
                print(f"[] Successfully retrieved {len(df)} real trip records from Chicago Open Data Portal.")
                return df
    except Exception as e:
        print(f"[!] Live API request failed or timed out ({e}). Switching to high-fidelity generator.")
    return pd.DataFrame()

def generate_realistic_taxi_data(n_records: int = 50000, start_date: str = "2023-01-01") -> pd.DataFrame:
    """
    Generate high-fidelity Chicago Taxi Trip data mimicking 12 months of demand,
    including rush hour peaks (8-10 AM, 5-8 PM), weekend nightlife, airport transfers (O'Hare #76, Midway #56, Loop #8, Near North #32),
    and simulated supply-demand surge events.
    """
    print(f"[*] Generating {n_records} realistic Chicago taxi trip records...")
    np.random.seed(42)
    
    # High-demand zones in Chicago:
    # 8: Near North Side, 32: Loop, 28: Near West Side, 76: O'Hare, 56: Midway, 6: Lakeview, 7: Lincoln Park
    high_demand_zones = [8, 32, 28, 76, 56, 6, 7, 24, 33, 22]
    all_zones = list(range(1, 78))
    
    # Probabilities for pickup zones (skewed towards downtown & airports)
    zone_weights = np.ones(77)
    for z in high_demand_zones:
        zone_weights[z - 1] = 12.0
    zone_probs = zone_weights / zone_weights.sum()

    base_time = datetime.strptime(start_date, "%Y-%m-%d")
    
    # Timestamps randomly distributed over 12 months (365 days)
    random_seconds = np.random.uniform(0, 365 * 86400, n_records)
    trip_starts = [base_time + timedelta(seconds=float(s)) for s in random_seconds]
    trip_starts.sort()
    
    pickup_zones = np.random.choice(all_zones, size=n_records, p=zone_probs)
    dropoff_zones = np.random.choice(all_zones, size=n_records, p=zone_probs)
    
    # Trip miles & seconds
    # Short trips downtown (1-4 miles), longer trips to airports (12-25 miles)
    is_airport = np.isin(pickup_zones, [76, 56]) | np.isin(dropoff_zones, [76, 56])
    trip_miles = np.where(
        is_airport,
        np.random.gamma(shape=10, scale=1.8, size=n_records),
        np.random.gamma(shape=2.5, scale=1.4, size=n_records)
    )
    trip_miles = np.clip(np.round(trip_miles, 2), 0.5, 45.0)
    
    # Average speed 12-30 mph
    speeds_mph = np.random.uniform(10, 32, n_records)
    trip_seconds = np.round((trip_miles / speeds_mph) * 3600).astype(int)
    
    # Base fare calculation
    base_fare = BASE_FLAG_DROP + (trip_miles * PER_MILE_RATE) + ((trip_seconds / 60.0) * PER_MINUTE_RATE)
    
    # Surge Multiplier simulation:
    # Demand spikes during rush hour (7-9 AM, 16-20 PM) and Friday/Saturday late nights in high-demand zones
    surge_multipliers = np.ones(n_records)
    for i, start_dt in enumerate(trip_starts):
        hour = start_dt.hour
        dow = start_dt.weekday() # 0=Mon, 5=Sat, 6=Sun
        p_zone = pickup_zones[i]
        
        is_rush = (hour in [7, 8, 9, 16, 17, 18, 19]) and (dow < 5)
        is_weekend_night = (hour in [22, 23, 0, 1, 2]) and (dow in [4, 5])
        is_high_zone = p_zone in high_demand_zones
        
        if (is_rush or is_weekend_night) and is_high_zone:
            # 35% probability of surge > 1.5x during peak demand
            if np.random.rand() < 0.35:
                surge_multipliers[i] = np.random.uniform(1.5, 2.8)
            else:
                surge_multipliers[i] = np.random.uniform(1.0, 1.45)
        else:
            if np.random.rand() < 0.05:
                surge_multipliers[i] = np.random.uniform(1.2, 1.8)
            else:
                surge_multipliers[i] = 1.0

    actual_fare = np.round(base_fare * surge_multipliers, 2)
    tips = np.where(np.random.rand(n_records) > 0.3, np.round(actual_fare * np.random.uniform(0.1, 0.25), 2), 0.0)
    trip_total = np.round(actual_fare + tips + 1.50, 2) # +1.50 fee/tax
    
    trip_ends = [trip_starts[i] + timedelta(seconds=int(trip_seconds[i])) for i in range(n_records)]
    
    df = pd.DataFrame({
        "trip_id": [f"TRIP_{i:07d}" for i in range(n_records)],
        "trip_start_timestamp": [t.strftime("%Y-%m-%d %H:%M:%S") for t in trip_starts],
        "trip_end_timestamp": [t.strftime("%Y-%m-%d %H:%M:%S") for t in trip_ends],
        "trip_seconds": trip_seconds,
        "trip_miles": trip_miles,
        "pickup_community_area": pickup_zones,
        "dropoff_community_area": dropoff_zones,
        "fare": actual_fare,
        "tips": tips,
        "trip_total": trip_total,
        "payment_type": np.random.choice(["Credit Card", "Cash", "Mobile"], size=n_records, p=[0.7, 0.25, 0.05])
    })
    
    print(f"[] Generated {len(df)} synthetic Chicago trip records.")
    return df

def save_and_partition_data(df: pd.DataFrame, output_name: str = "chicago_taxi_trips_12mo.csv"):
    """Saves raw CSV and prepares year/month partition folders for HDFS loading."""
    output_path = RAW_DATA_DIR / output_name
    df.to_csv(output_path, index=False)
    print(f"[] Saved consolidated raw trips to: {output_path} ({len(df)} rows)")
    
    # Also partition into year/month directories
    df["dt"] = pd.to_datetime(df["trip_start_timestamp"])
    df["year"] = df["dt"].dt.year
    df["month"] = df["dt"].dt.strftime("%m")
    
    for (yr, mo), group in df.groupby(["year", "month"]):
        part_dir = RAW_DATA_DIR / f"year={yr}" / f"month={mo}"
        part_dir.mkdir(parents=True, exist_ok=True)
        group.drop(columns=["dt", "year", "month"]).to_csv(part_dir / "trips.csv", index=False)
    
    print(f"[] Partitioned data across 12 months in: {RAW_DATA_DIR}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download or generate Chicago taxi data")
    parser.add_argument("--limit", type=int, default=30000, help="Number of records to fetch/generate")
    parser.add_argument("--synthetic-only", action="store_true", help="Force synthetic generation")
    args = parser.parse_args()
    
    df = pd.DataFrame()
    if not args.synthetic_only:
        df = fetch_chicago_portal_data(limit=args.limit)
        
    if df.empty or len(df) < 5000:
        df = generate_realistic_taxi_data(n_records=args.limit)
        
    save_and_partition_data(df)
