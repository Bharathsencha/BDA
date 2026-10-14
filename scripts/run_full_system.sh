#!/bin/bash
# ==============================================================================
# Master Execution and Launch Script for Ride-Hailing Surge Prediction System
# ==============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "================================================================="
echo " Launching Ride-Hailing Surge Prediction System Pipeline"
echo "================================================================="

# 1. Execute end-to-end data, MapReduce, and ML pipeline
"$PROJECT_DIR/venv/bin/python" "$PROJECT_DIR/scripts/run_groundwork_demo.py"

echo ""
echo "================================================================="
echo " System Ready. To launch interactive presentation dashboard:"
echo " ./venv/bin/streamlit run app/dashboard.py"
echo "================================================================="
