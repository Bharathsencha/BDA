#!/bin/bash
# ==============================================================================
# Script to run MapReduce locally via standard Unix pipeline simulation
# Input : data/raw/chicago_taxi_trips_12mo.csv
# Output: data/processed/features_labels.csv
# ==============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_CSV="$PROJECT_DIR/data/raw/chicago_taxi_trips_12mo.csv"
OUTPUT_CSV="$PROJECT_DIR/data/processed/features_labels.csv"

echo "================================================================="
echo "Running Local MapReduce Pipeline for 20-Min Advance Feature Extraction"
echo "================================================================="

if [ ! -f "$INPUT_CSV" ]; then
    echo "[*] Input 12-month data not found. Downloading/generating dataset first..."
    "$PROJECT_DIR/venv/bin/python" "$PROJECT_DIR/scripts/download_data.py" --monthly-records 6000
fi

mkdir -p "$PROJECT_DIR/data/processed"

# Updated leak-free schema with 20-minute forward lookahead target
echo "zone_id,time_bucket,hour,day_of_week,is_weekend,is_rush_hour,demand_trips,dropoff_trips,inflow_outflow_ratio,demand_momentum,avg_speed_mph,is_surge_in_20min" > "$OUTPUT_CSV"

# Pipeline: CSV -> Mapper -> Sort (Shuffle) -> Reducer -> Output CSV
cat "$INPUT_CSV" | "$PROJECT_DIR/mapreduce/mapper.py" | sort -k1,1n -k2,2 | "$PROJECT_DIR/mapreduce/reducer.py" >> "$OUTPUT_CSV"

TOTAL_ROWS=$(wc -l < "$OUTPUT_CSV")
echo "[] MapReduce complete! Processed $TOTAL_ROWS feature vectors."
echo "[] Saved output to: $OUTPUT_CSV"
echo "-----------------------------------------------------------------"
echo "First 5 feature rows:"
head -n 6 "$OUTPUT_CSV"
echo "================================================================="
