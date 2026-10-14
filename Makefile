.PHONY: all run clean

PYTHON = ./venv/bin/python
STREAMLIT = ./venv/bin/streamlit

# Default target: executes complete end-to-end pipeline
all:
	@echo "Executing complete end-to-end pipeline..."
	@bash scripts/run_full_system.sh

# Launches interactive presentation dashboard
run:
	@echo "Launching predictive dispatch control center..."
	@$(STREAMLIT) run app/dashboard.py

# Cleans build artifacts, temporary files, and Python cache
clean:
	@echo "Cleaning temporary files, build artifacts, and cache..."
	@rm -rf target/ 2>/dev/null || docker run --rm -v "$$(pwd)":/app -w /app alpine rm -rf /app/target 2>/dev/null || true
	@rm -rf dependency-reduced-pom.xml
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@rm -f data/processed/live_alerts.json
	@echo "Clean completed."
