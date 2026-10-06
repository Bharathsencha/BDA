# Ride-Hailing Surge Price Prediction (Big Data Analytics)

An end-to-end Big Data system to predict ride-hailing surge multipliers (>1.5x) **20 minutes in advance** across Chicago's 77 community zones.

---

## 1. System Architecture & Groundwork (30%)

```
bda/
├── pom.xml                               # Maven build configuration for Java Hadoop MapReduce
├── src/main/java/com/bda/surge/
│   ├── SurgeMapper.java                  # Java Hadoop Mapper for taxi trip parsing & rate computation
│   ├── SurgeReducer.java                 # Java Hadoop Reducer for spatio-temporal feature aggregation
│   └── SurgeDriver.java                  # Java Hadoop ToolRunner entry point
├── config/
│   └── config.py                         # Centralized configuration (Kafka, HDFS, Chicago base rates)
├── data/
│   ├── raw/                              # Partitioned raw trips (year=YYYY/month=MM/)
│   └── processed/                        # Extracted spatio-temporal features & labels
├── kafka/
│   ├── producer.py                       # Streams ride-request events to Kafka
│   └── consumer_hdfs.py                  # Consumes stream and stages partitioned batches to HDFS
├── mapreduce/
│   ├── mapper.py                         # Python Hadoop Streaming Mapper
│   ├── reducer.py                        # Python Hadoop Streaming Reducer
│   ├── run_local.sh                      # Standalone Unix pipe MapReduce runner
│   └── run_hadoop.sh                     # Cluster MapReduce runner (Java JAR & Streaming)
├── ml/
│   ├── train_baseline.py                 # Decision Tree & Random Forest classifiers
│   └── models/                           # Serialized model artifacts (.pkl)
├── scripts/
│   ├── download_data.py                  # Chicago Open Data Portal fetcher & 12-mo generator
│   └── run_groundwork_demo.py            # Master end-to-end 30% execution demo
├── README.md
└── requirements.txt
```

---

## 2. Quick Start

### 1. Run Complete End-to-End Groundwork Demo
```bash
./venv/bin/python scripts/run_groundwork_demo.py
```

### 2. Run MapReduce Locally (Unix Pipes)
```bash
bash mapreduce/run_local.sh
```

### 3. Java Hadoop MapReduce Setup
Build and run on a Hadoop cluster:
```bash
# Compile and package JAR via Maven:
mvn clean package

# Submit to Hadoop:
hadoop jar target/surge-price-prediction-mapreduce-1.0-SNAPSHOT.jar \
    com.bda.surge.SurgeDriver \
    /data/raw/taxi_trips \
    /data/processed/surge_features
```

### 4. Train Decision Tree & Random Forest Models
```bash
./venv/bin/python ml/train_baseline.py
```
