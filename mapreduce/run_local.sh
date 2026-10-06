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
echo "Running Local MapReduce Pipeline for Surge Feature Extraction"
echo "================================================================="

if [ ! -f "$INPUT_CSV" ]; then
    echo "[!] Input data not found. Downloading/generating dataset first..."
    "$PROJECT_DIR/venv/bin/python" "$PROJECT_DIR/scripts/download_data.py" --limit 25000
fi

mkdir -p "$PROJECT_DIR/data/processed"

echo "zone_id,time_bucket,hour,day_of_week,is_weekend,is_rush_hour,demand_trips,avg_fare,avg_miles,avg_seconds,avg_speed_mph,avg_surge_multiplier,is_surge_above_1_5x" > "$OUTPUT_CSV"

# Pipeline: CSV -> Mapper -> Sort (Shuffle) -> Reducer -> Output CSV
cat "$INPUT_CSV" | "$PROJECT_DIR/mapreduce/mapper.py" | sort -k1,1 -k2,2 | "$PROJECT_DIR/mapreduce/reducer.py" >> "$OUTPUT_CSV"

TOTAL_ROWS=$(wc -l < "$OUTPUT_CSV")
echo "[] MapReduce complete! Processed $TOTAL_ROWS feature vectors."
echo "[] Saved output to: $OUTPUT_CSV"
echo "-----------------------------------------------------------------"
echo "First 5 feature rows:"
head -n 6 "$OUTPUT_CSV"
echo "================================================================="
