"""
Master Groundwork Demo: Runs the foundational 30% pipeline end-to-end.
1. Downloads / generates 12-month Chicago taxi dataset.
2. Replays ride-request events (Kafka simulation).
3. Executes MapReduce feature aggregation pipeline.
4. Trains Decision Tree & Random Forest models.
"""
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON_BIN = PROJECT_ROOT / "venv" / "bin" / "python"

def run_step(title, command):
    print("\n" + "="*70)
    print(f" >>> STEP: {title}")
    print("="*70)
    res = subprocess.run(command, shell=True, cwd=PROJECT_ROOT)
    if res.returncode != 0:
        print(f"[!] Error occurred during: {title}")
        sys.exit(res.returncode)

def main():
    print("**********************************************************************")
    print("  RIDE-HAILING SURGE PRICE PREDICTION - 30% GROUNDWORK PIPELINE")
    print("**********************************************************************")

    # Step 1: Download / Generate Data
    run_step("Data Ingestion (Chicago Taxi Dataset)", f"{PYTHON_BIN} scripts/download_data.py --limit 30000")

    # Step 2: Stream Kafka Ride Request Events
    run_step("Kafka Ride-Request Streaming Simulation", f"{PYTHON_BIN} kafka/producer.py --events 25 --dry-run")

    # Step 3: Run MapReduce Feature Aggregator
    run_step("MapReduce Spatio-Temporal Feature Aggregation", "bash mapreduce/run_local.sh")

    # Step 4: Train Decision Tree & Random Forest ML Classifiers
    run_step("Train Decision Tree & Random Forest Models", f"{PYTHON_BIN} ml/train_baseline.py")

    print("\n" + "*"*70)
    print("   30% FOUNDATIONAL GROUNDWORK COMPLETED SUCCESSFULLY!")
    print("  - Java MapReduce classes ready in: src/main/java/com/bda/surge/")
    print("  - Maven pom.xml ready for Hadoop cluster builds")
    print("  - Kafka Producer & Consumer ready in: kafka/")
    print("  - Streaming MapReduce scripts ready in: mapreduce/")
    print("  - ML Baseline Trained & Evaluated in: ml/")
    print("**********************************************************************\n")

if __name__ == "__main__":
    main()
