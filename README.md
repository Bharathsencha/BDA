# Ride-Hailing Surge Price Prediction (Big Data Analytics)

An end-to-end Big Data system to predict ride-hailing surge multipliers (>= 1.5x) **20 minutes in advance** across Chicago's 77 community zones to enable proactive driver pre-positioning and prevent passenger price shocks.

---

## 1. Problem Statement & Objective

* **Current Challenge**: Surge multipliers in ride-hailing platforms (Uber, Lyft) are applied reactively only *after* demand spikes, leaving drivers unprepared and passengers facing sudden price shocks.
* **Objective**: Stream live ride-request events via **Apache Kafka**, store 12 months of historical trip and fare data on **Hadoop HDFS**, execute spatio-temporal feature aggregation with a true 20-minute forward lookahead via **Java MapReduce**, and train **Decision Tree & Random Forest Classifiers** on non-leaked pre-surge signals to predict impending surges.
* **Dataset**: Chicago Taxi Trips Dataset (City of Chicago Open Data Portal, 77 Community Areas, 12 Calendar Months).

---

## 2. Tech Stack & Frameworks Used

| Layer | Technology / Framework | Purpose |
| :--- | :--- | :--- |
| **Distributed Storage** | **Apache Hadoop HDFS 3.x** | Cold storage for 12 months of historical trip records partitioned by `year=YYYY/month=MM/`. |
| **Event Streaming** | **Apache Kafka & Zookeeper** | Real-time ingestion and message brokering for `ride-requests` and `surge-alerts` topics. |
| **Distributed Processing** | **Native Java MapReduce (Hadoop)** | High-throughput batch feature extraction (`SurgeMapper.java`, `SurgeReducer.java`, `SurgeDriver.java`). |
| **Streaming ETL** | **Python Hadoop Streaming** | Lightweight Unix pipeline simulation and cluster streaming scripts (`mapper.py`, `reducer.py`). |
| **Machine Learning** | **Scikit-Learn & Joblib** | Random Forest, Decision Tree, Gradient Boosting, and Logistic Regression models. |
| **Data Engineering** | **Pandas & NumPy** | In-memory data manipulation, timestamp windowing, and statistical feature engineering. |
| **Spatial Analytics** | **Haversine Metric & Adjacency Graph** | Chicago 77 Community Area centroid coordinate mapping and neighboring zone spillover modeling. |
| **Optimization Engine** | **Fleet Supply Solver** | Mathematical driver repositioning algorithm to compute vehicle transfer quotas and transit ETAs. |
| **Web Dashboard** | **Streamlit** | Real-time predictive dispatch center and interactive control interface. |
| **Data Visualization** | **Plotly Express & Graph Objects** | Interactive Mapbox geospatial maps, gauge risk meters, ROC curves, and feature importance charts. |
| **Containerization** | **Docker & Docker Compose** | Multi-container orchestration (`docker-compose.yml`) for Kafka, Zookeeper, and Maven compilation. |

---

## 3. Project Directory Structure

```
bda/
├── Makefile                          # Simple commands: make, make run, make clean
├── docker-compose.yml                # Docker Compose for Apache Kafka and Zookeeper
├── pom.xml                           # Maven build configuration for Java Hadoop MapReduce
├── src/main/java/com/bda/surge/
│   ├── SurgeMapper.java              # Java Hadoop Mapper for taxi trip parsing & rate computation
│   ├── SurgeReducer.java             # Java Hadoop Reducer for spatio-temporal feature aggregation
│   └── SurgeDriver.java              # Java Hadoop ToolRunner entry point
├── config/
│   ├── config.py                     # Centralized configuration (Kafka, HDFS, Chicago base rates)
│   └── chicago_zones.py              # Coordinates (lat/lon) and adjacency for Chicago's 77 zones
├── data/
│   ├── raw/                          # 12-Month Partitioned raw trips (year=YYYY/month=MM/)
│   └── processed/                    # Extracted spatio-temporal features, labels, and live alerts
├── kafka/
│   ├── producer.py                   # Streams continuous ride-request events to Kafka
│   └── consumer_hdfs.py              # Consumes stream and stages partitioned batches to HDFS
├── mapreduce/
│   ├── mapper.py                     # Python Hadoop Streaming Mapper (Pickups & Dropoffs)
│   ├── reducer.py                    # Python Hadoop Streaming Reducer (20-Min Forward Target)
│   ├── run_local.sh                  # Standalone Unix pipe MapReduce runner
│   └── run_hadoop.sh                 # Cluster MapReduce runner (Java JAR & Streaming)
├── ml/
│   ├── train_baseline.py             # Decision Tree & Random Forest baseline training
│   ├── advanced_training.py          # 4-Model benchmark suite (ROC-AUC, PR Curves, Evaluation JSON)
│   ├── dispatch_optimizer.py         # Driver repositioning and fleet supply optimization solver
│   └── models/                       # Serialized model artifacts (.pkl) and evaluation metrics
├── streaming_inference/
│   └── stream_processor.py           # Real-time streaming inference worker and alert generator
├── app/
│   └── dashboard.py                  # Streamlit 4-Tab Predictive Dispatch Control Center
├── scripts/
│   ├── download_data.py              # 12-Month Chicago data generator & Socrata portal fetcher
│   ├── stage_hdfs.sh                 # HDFS partitioned data loader script
│   ├── run_groundwork_demo.py        # Master end-to-end Python pipeline runner
│   └── run_full_system.sh            # Full system bash launcher
├── README.md
└── requirements.txt
```

---

## 4. Machine Learning Benchmark Results

Trained on **131,450 spatio-temporal feature instances** across all 12 calendar months and all 77 Chicago community areas. Evaluated on an **out-of-time temporal test set (October - December 2023 | 32,873 instances)** with **zero target leakage** (no `avg_fare` or contemporaneous surge multipliers in the feature space):

| Model Architecture | ROC-AUC Score | F1 Score (0.50) | Recall (Standard 0.50) | Recall (Tuned Cutoff 0.35) | Precision (0.50) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Ensemble)** | **0.8124** | **0.32** | **65.19%** | **73.24%** | **21.23%** |
| **Decision Tree (Baseline)** | **0.8194** | **0.32** | **66.72%** | **69.77%** | **21.15%** |
| **Gradient Boosting** | **0.8629** | **0.13** | **7.10%** | **49.53%** | **55.33%** |
| **Logistic Regression** | **0.7463** | **0.22** | **56.68%** | **76.71%** | **13.97%** |

*Key Operational Takeaways*:
* Tuning the decision threshold to 0.35 captures **70% to 76% of all impending surges 20 minutes prior to their onset**, giving the dispatch engine sufficient lead time to mobilize surrounding idle fleets.
* Gradient Boosting yields the highest raw discrimination (ROC-AUC 0.8629), while Random Forest delivers balanced precision and recall across high-density commercial hubs and residential zones.
* Physical supply-demand deficit modeling ensures that surplus incoming drop-offs consistently suppress surge pricing risk, aligning algorithmic predictions with transportation market dynamics.

### Feature Importance Ranking (Pre-Surge Operational Signals)
1. **`is_rush_hour`** (32.43%) - Commuter peak hours (7-9 AM, 4-7 PM weekdays)
2. **`hour`** (14.54%) - Diurnal transportation activity cycles
3. **`day_of_week`** (10.61%) - Commuting rhythms vs weekend entertainment trips
4. **`zone_id`** (10.55%) - Spatial hub baseline (commercial business district vs residential)
5. **`net_deficit`** (9.14%) - Local demand deficit (`pickups - dropoffs`)
6. **`inflow_outflow_ratio`** (7.65%) - Ratio of incoming vehicle supply to ride requests
7. **`avg_speed_mph`** (4.55%) - Real-time traffic congestion index
8. **`dropoff_trips`** (4.06%) - Incoming vehicle arrivals supplying available drivers
9. **`demand_trips`** (3.31%) - Immediate 15-minute pickup demand count
10. **`is_weekend`** (1.89%) - Weekend night leisure indicator
11. **`demand_momentum`** (1.27%) - Demand acceleration rate (delta trips / delta t)

---

## 5. Interactive Operations Dashboard

The web interface (`app/dashboard.py`) provides a four-tab executive control center:
1. **Live Predictive Dispatch & Map**: Real-time 20-minute forward surge warnings, risk probability gauge meter, physical supply-demand deficit balance, and an interactive 77-community zone risk map.
2. **Fleet Supply & Rebalancing Optimizer**: Automated solver calculating vehicle transfer quotas from adjacent donor zones, transit ETAs, and incentive bonuses to preemptively balance supply deficits.
3. **Multi-Model Benchmark Lab**: Comparative analysis showing model ROC curves, threshold tuning impacts, confusion matrices, and feature importance rankings.
4. **Kafka Real-Time Stream Monitor**: Real-time event streaming telemetry, live batch simulation triggers, and an event ticker logging 20-minute forward surge alerts emitted to Kafka.

---

## 6. Execution Instructions

### Quick Start via Makefile
```bash
# Execute complete end-to-end pipeline (Data Ingestion -> MapReduce -> ML Training -> Streaming):
make

# Launch the interactive operations dashboard:
make run

# Clean build artifacts, temporary files, and Python cache:
make clean
```

### Direct Script Execution
```bash
# Run complete end-to-end pipeline via bash:
bash scripts/run_full_system.sh

# Or directly via Python orchestrator:
./venv/bin/python scripts/run_groundwork_demo.py

# Launch Streamlit dashboard directly:
./venv/bin/streamlit run app/dashboard.py
```
Open dashboard in browser at: `http://localhost:8501`

### Apache Kafka & Zookeeper via Docker
```bash
# Start background Kafka broker:
docker compose up -d

# Check container status:
docker compose ps

# Stream ride-request events to live Kafka topic:
./venv/bin/python kafka/producer.py --events 100 --delay 0.05
```

### Native Java MapReduce on Hadoop
```bash
# Compile and package Shaded Uber JAR via Maven:
docker run --rm -v "$(pwd)":/app -w /app maven:3.9-eclipse-temurin-11 mvn clean package -DskipTests

# Submit to Hadoop Cluster:
hadoop jar target/surge-price-prediction-mapreduce-1.0-SNAPSHOT.jar \
    com.bda.surge.SurgeDriver \
    /data/raw/taxi_trips \
    /data/processed/surge_features
```
