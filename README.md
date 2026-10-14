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
│   └── dashboard.py                  # Advanced Streamlit Predictive Dispatch Control Center
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

Trained on **131,250 spatio-temporal feature instances** across all 12 calendar months and all 77 Chicago community areas. Evaluated on an **out-of-time temporal test set (October - December 2023 | 32,846 instances)** with **zero target leakage** (no `avg_fare` or contemporaneous surge multipliers in the feature space):

| Model Architecture | ROC-AUC Score | F1 Score (0.50) | Recall (Standard 0.50) | Recall (Tuned Cutoff 0.35) | Precision (0.50) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree (Baseline)** | **0.8325** | **0.23** | **70.06%** | **91.80%** | **13.00%** |
| **Random Forest (Ensemble)** | **0.8290** | **0.23** | **67.63%** | **88.37%** | **14.00%** |
| **Gradient Boosting** | **0.8497** | **0.05** | **2.70%** | **15.20%** | **28.00%** |
| **Logistic Regression** | **0.7661** | **0.18** | **64.74%** | **89.10%** | **10.50%** |

*Key Operational Takeaway*: By tuning the classification threshold to $\tau = 0.35$, the Decision Tree and Random Forest classifiers achieve **$\ge 88-92\%$ Recall**, ensuring the dispatch engine detects impending demand surges with 20 minutes of advance notice so idle vehicles can be proactively repositioned.

### Feature Importance Ranking (Pre-Surge Operational Signals)
1. **`is_rush_hour`** (24.03%) - High-frequency commuting windows (7-9 AM, 4-7 PM)
2. **`demand_trips`** (19.74%) - Absolute pickup count in current 15-minute window
3. **`zone_id`** (14.76%) - Commercial vs residential spatial characteristics
4. **`hour`** (11.10%) - Diurnal transit patterns
5. **`day_of_week`** (8.37%) - Weekday commuting vs weekend entertainment
6. **`avg_speed_mph`** (7.99%) - Congestion index (traffic slowdowns precede surge)
7. **`inflow_outflow_ratio`** (5.96%) - Vehicle supply balance (arrivals vs departures)
8. **`dropoff_trips`** (4.05%) - Drop-off volume supplying available drivers
9. **`demand_momentum`** (2.92%) - Demand acceleration ($\Delta \text{trips} / \Delta t$)
10. **`is_weekend`** (1.08%) - Weekend night activity flag

---

## 5. Execution Instructions

### 1. Run Complete End-to-End Pipeline
Executes data verification, HDFS partition checks, MapReduce feature extraction, multi-model training, and real-time streaming evaluation:
```bash
bash scripts/run_full_system.sh
```
Or directly via Python:
```bash
./venv/bin/python scripts/run_groundwork_demo.py
```

### 2. Launch the Interactive Operations Dashboard
```bash
./venv/bin/streamlit run app/dashboard.py
```
Open in browser at: `http://localhost:8501`

### 3. Start Apache Kafka & Zookeeper via Docker
```bash
# Start background Kafka broker:
docker compose up -d

# Check container status:
docker compose ps

# Stream ride-request events to live Kafka topic:
./venv/bin/python kafka/producer.py --events 100 --delay 0.05
```

### 4. Build and Run Native Java MapReduce on Hadoop
```bash
# Compile and package Shaded Uber JAR via Maven:
docker run --rm -v "$(pwd)":/app -w /app maven:3.9-eclipse-temurin-11 mvn clean package -DskipTests

# Submit to Hadoop Cluster:
hadoop jar target/surge-price-prediction-mapreduce-1.0-SNAPSHOT.jar \
    com.bda.surge.SurgeDriver \
    /data/raw/taxi_trips \
    /data/processed/surge_features
```
