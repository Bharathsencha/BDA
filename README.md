# Ride-Hailing Surge Price Prediction (Big Data Analytics)

An end-to-end Big Data system to predict ride-hailing surge multipliers (>1.5x) **20 minutes in advance** across Chicago's 77 community zones to enable proactive driver pre-positioning and prevent passenger price shocks.

---

## 1. Problem Statement & Objective

* **Current Challenge**: Surge multipliers in ride-hailing platforms (Uber, Lyft) are applied reactively only *after* demand spikes, leaving drivers unprepared and passengers facing sudden price shocks.
* **Objective**: Stream live ride-request events via **Apache Kafka**, store 12 months of historical trip and fare data on **Hadoop HDFS**, execute spatio-temporal feature aggregation via **Java MapReduce**, and train **Ensemble Classifiers** to predict 20 minutes in advance which zone-hour combinations will require a surge multiplier above 1.5x.
* **Dataset**: Chicago Taxi Trips Dataset (City of Chicago Open Data Portal, 77 Community Areas).

---

## 2. Tech Stack & Frameworks Used

| Layer | Technology / Framework | Purpose |
| :--- | :--- | :--- |
| **Distributed Storage** | **Apache Hadoop HDFS 3.x** | Cold storage for 12 months of historical trip records partitioned by `year=YYYY/month=MM/`. |
| **Event Streaming** | **Apache Kafka & Zookeeper** | Real-time ingestion and message brokering for `ride-requests` and `surge-alerts` topics. |
| **Distributed Processing** | **Native Java MapReduce (Hadoop)** | High-throughput batch feature extraction (`SurgeMapper.java`, `SurgeReducer.java`, `SurgeDriver.java`). |
| **Streaming ETL** | **Python Hadoop Streaming** | Lightweight Unix pipeline simulation and cluster streaming scripts (`mapper.py`, `reducer.py`). |
| **Machine Learning** | **Scikit-Learn & Joblib** | Random Forest, Gradient Boosting, Decision Tree, and Logistic Regression classification models. |
| **Data Engineering** | **Pandas & NumPy** | In-memory data manipulation, timestamp normalization, and statistical feature engineering. |
| **Spatial Analytics** | **Haversine Metric & Adjacency Graph** | Chicago 77 Community Area centroid coordinate mapping and neighboring zone spillover modeling. |
| **Optimization Engine** | **Fleet Supply Solver** | Mathematical driver repositioning algorithm to compute vehicle transfer quotas and transit ETAs. |
| **Web Dashboard** | **Streamlit** | Real-time predictive dispatch center and interactive control interface. |
| **Data Visualization** | **Plotly Express & Graph Objects** | Interactive Mapbox geospatial maps, gauge risk meters, ROC curves, and feature importance charts. |
| **Build & Environment** | **Apache Maven & Python 3.12 (venv)** | Java dependency management (`pom.xml`) and isolated Python environment. |

---

## 3. Project Directory Structure

```
bda/
 pom.xml                               # Maven build configuration for Java Hadoop MapReduce
 src/main/java/com/bda/surge/
    SurgeMapper.java                  # Java Hadoop Mapper for taxi trip parsing & rate computation
    SurgeReducer.java                 # Java Hadoop Reducer for spatio-temporal feature aggregation
    SurgeDriver.java                  # Java Hadoop ToolRunner entry point
 config/
    config.py                         # Centralized configuration (Kafka, HDFS, Chicago base rates)
    chicago_zones.py                  # Coordinates (lat/lon) and adjacency for Chicago's 77 zones
 data/
    raw/                              # Partitioned raw trips (year=YYYY/month=MM/)
    processed/                        # Extracted spatio-temporal features, labels, and live alerts
 kafka/
    producer.py                       # Streams ride-request events to Kafka
    consumer_hdfs.py                  # Consumes stream and stages partitioned batches to HDFS
 mapreduce/
    mapper.py                         # Python Hadoop Streaming Mapper
    reducer.py                        # Python Hadoop Streaming Reducer
    run_local.sh                      # Standalone Unix pipe MapReduce runner
    run_hadoop.sh                     # Cluster MapReduce runner (Java JAR & Streaming)
 ml/
    train_baseline.py                 # Decision Tree & Random Forest baseline training
    advanced_training.py              # 4-Model benchmark suite (ROC-AUC, PR Curves, Evaluation JSON)
    dispatch_optimizer.py             # Driver repositioning and fleet supply optimization solver
    models/                           # Serialized model artifacts (.pkl) and evaluation metrics
 streaming_inference/
    stream_processor.py               # Real-time streaming inference worker and alert generator
 app/
    dashboard.py                      # Advanced Streamlit Predictive Dispatch Control Center
 scripts/
    download_data.py                  # Chicago Open Data Portal fetcher & 12-mo generator
    run_groundwork_demo.py            # Master end-to-end execution demo
 README.md
 requirements.txt
```

---

## 4. Machine Learning Benchmark Results

Trained on 5,870 spatio-temporal feature vectors extracted across Chicago's 77 zones:

| Model Architecture | ROC-AUC Score | F1 Score | Recall (Surge >= 1.5x) | Precision |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest (Ensemble)** | **0.9962** | **0.8814** | **92.86%** | **83.87%** |
| **Gradient Boosting** | **0.9937** | **0.8750** | **87.50%** | **87.50%** |
| **Decision Tree** | **0.9690** | **0.8000** | **92.86%** | **70.27%** |
| **Logistic Regression** | **0.9975** | **0.7285** | **98.21%** | **57.89%** |

*Note: High Recall is prioritized to ensure the dispatch engine never misses critical surge events, allowing idle vehicles to be pre-positioned in advance.*

---

## 5. Quick Start Instructions

### 1. Run Complete End-to-End Groundwork Pipeline
```bash
./venv/bin/python scripts/run_groundwork_demo.py
```

### 2. Run Advanced Multi-Model Benchmark Suite
```bash
./venv/bin/python ml/advanced_training.py
```

### 3. Launch the Interactive Dashboard GUI
```bash
./venv/bin/streamlit run app/dashboard.py
```
Open in browser at: `http://localhost:8501`

### 4. Build and Run Native Java MapReduce on Hadoop
```bash
# Compile and package JAR via Maven:
mvn clean package

# Submit to Hadoop Cluster:
hadoop jar target/surge-price-prediction-mapreduce-1.0-SNAPSHOT.jar \
    com.bda.surge.SurgeDriver \
    /data/raw/taxi_trips \
    /data/processed/surge_features
```
