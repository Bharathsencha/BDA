package com.bda.surge;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

/**
 * SurgeReducer:
 * Aggregates all ride trips for a specific (Zone, 15-Minute Window).
 * Computes demand pressure, vehicle inflow/outflow balance, velocity index,
 * and outputs the leak-free classification vector.
 */
public class SurgeReducer extends Reducer<Text, Text, Text, Text> {

    private static final double SURGE_THRESHOLD = 1.5;
    private final Text outValue = new Text();

    @Override
    protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        int pickupCount = 0;
        int dropoffCount = 0;
        double sumSpeed = 0.0;
        double sumSurgeMultiplier = 0.0;

        for (Text val : values) {
            String[] parts = val.toString().split(",");
            if (parts.length < 4) {
                continue;
            }

            try {
                String eventType = parts[0];
                double miles = Double.parseDouble(parts[1]);
                double seconds = Double.parseDouble(parts[2]);
                double surge = Double.parseDouble(parts[3]);
                double speed = (seconds > 0) ? (miles / (seconds / 3600.0)) : 18.0;

                if ("P".equals(eventType)) {
                    pickupCount++;
                    sumSpeed += speed;
                    sumSurgeMultiplier += surge;
                } else if ("D".equals(eventType)) {
                    dropoffCount++;
                }
            } catch (NumberFormatException ignored) {
            }
        }

        if (pickupCount == 0 && dropoffCount == 0) {
            return;
        }

        double avgSpeedMph = (pickupCount > 0) ? (sumSpeed / pickupCount) : 18.0;
        double avgSurge = (pickupCount > 0) ? (sumSurgeMultiplier / pickupCount) : 1.0;
        double inflowOutflowRatio = (double) dropoffCount / Math.max(1, pickupCount);
        inflowOutflowRatio = Math.round(inflowOutflowRatio * 100.0) / 100.0;

        // Binary Classification Target: 1 if surge multiplier >= 1.5, else 0
        int isSurgeAbove15x = (avgSurge >= SURGE_THRESHOLD) ? 1 : 0;

        // Output CSV Line Format:
        // demand_trips,dropoff_trips,inflow_outflow_ratio,avg_speed_mph,is_surge_above_1_5x
        String result = String.format("%d,%d,%.2f,%.2f,%d",
                pickupCount,
                dropoffCount,
                inflowOutflowRatio,
                avgSpeedMph,
                isSurgeAbove15x
        );

        outValue.set(result);
        context.write(key, outValue);
    }
}
