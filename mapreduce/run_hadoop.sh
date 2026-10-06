#!/bin/bash
# ==============================================================================
# Hadoop Cluster Execution Runner for Java MapReduce & Streaming
# ==============================================================================

set -e

HADOOP_HOME=${HADOOP_HOME:-/usr/local/hadoop}
HDFS_INPUT="/data/raw/taxi_trips"
HDFS_OUTPUT="/data/processed/surge_features"

echo "================================================================="
echo " Hadoop Cluster MapReduce Runner"
echo "================================================================="

# Mode 1: Native Java MapReduce Execution
run_java_mapreduce() {
    echo "[*] Step 1: Uploading raw dataset to HDFS..."
    hdfs dfs -mkdir -p $HDFS_INPUT
    hdfs dfs -put -f data/raw/chicago_taxi_trips_12mo.csv $HDFS_INPUT/

    echo "[*] Step 2: Removing existing output directory if present..."
    hdfs dfs -rm -r -f $HDFS_OUTPUT

    echo "[*] Step 3: Executing Java MapReduce Driver..."
    hadoop jar target/surge-price-prediction-mapreduce-1.0-SNAPSHOT.jar \
        com.bda.surge.SurgeDriver \
        $HDFS_INPUT \
        $HDFS_OUTPUT

    echo "[*] Step 4: Inspecting HDFS Output..."
    hdfs dfs -ls $HDFS_OUTPUT
    hdfs dfs -cat $HDFS_OUTPUT/part-r-00000 | head -n 20
}

# Mode 2: Python Hadoop Streaming Execution
run_streaming_mapreduce() {
    STREAMING_JAR=$(find $HADOOP_HOME -name "hadoop-streaming*.jar" | head -n 1)
    if [ -z "$STREAMING_JAR" ]; then
        STREAMING_JAR="/usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-*.jar"
    fi

    echo "[*] Executing Hadoop Streaming with Python Mapper & Reducer..."
    hadoop jar $STREAMING_JAR \
        -files mapreduce/mapper.py,mapreduce/reducer.py \
        -mapper "python3 mapper.py" \
        -reducer "python3 reducer.py" \
        -input $HDFS_INPUT \
        -output $HDFS_OUTPUT
}

echo "Select execution mode:"
echo "1) Java MapReduce (compiled JAR)"
echo "2) Hadoop Streaming (Python)"
read -p "Enter choice [1 or 2]: " choice

case $choice in
    1) run_java_mapreduce ;;
    2) run_streaming_mapreduce ;;
    *) echo "Invalid choice. Exiting." ;;
esac
