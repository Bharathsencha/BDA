package com.bda.surge;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

/**
 * SurgeReducer:
 * Aggregates all ride trips for a specific (Zone, 15-Minute Window).
 * Computes demand pressure (trip count), velocity, average surge multiplier,
 * and produces the binary classification target (is_surge_above_1_5x).
 */
public class SurgeReducer extends Reducer<Text, Text, Text, Text> {

    private static final double SURGE_THRESHOLD = 1.5;
    private final Text outValue = new Text();

    @Override
    protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        int tripCount = 0;
        double sumFare = 0.0;
        double sumMiles = 0.0;
        double sumSeconds = 0.0;
        double sumSurgeMultiplier = 0.0;

        for (Text val : values) {
            String[] parts = val.toString().split(",");
            if (parts.length < 4) {
                continue;
            }

            try {
                double fare = Double.parseDouble(parts[0]);
                double miles = Double.parseDouble(parts[1]);
                double seconds = Double.parseDouble(parts[2]);
                double surge = Double.parseDouble(parts[3]);

                sumFare += fare;
                sumMiles += miles;
                sumSeconds += seconds;
                sumSurgeMultiplier += surge;
                tripCount++;
            } catch (NumberFormatException ignored) {
            }
        }

        if (tripCount == 0) {
            return;
        }

        double avgFare = sumFare / tripCount;
        double avgMiles = sumMiles / tripCount;
        double avgSeconds = sumSeconds / tripCount;
        double avgSurgeMultiplier = sumSurgeMultiplier / tripCount;

        // Speed in miles per hour (avoid divide by zero)
        double totalHours = sumSeconds / 3600.0;
        double avgSpeedMph = (totalHours > 0) ? (sumMiles / totalHours) : 0.0;

        // Target Label: 1 if surge multiplier >= 1.5, else 0
        int isSurgeAbove15x = (avgSurgeMultiplier >= SURGE_THRESHOLD) ? 1 : 0;

        // Output CSV Line Format:
        // demand_trips,avg_fare,avg_miles,avg_seconds,avg_speed_mph,avg_surge_multiplier,is_surge_above_1_5x
        String result = String.format("%d,%.2f,%.2f,%.1f,%.2f,%.2f,%d",
                tripCount,
                avgFare,
                avgMiles,
                avgSeconds,
                avgSpeedMph,
                avgSurgeMultiplier,
                isSurgeAbove15x
        );

        outValue.set(result);
        context.write(key, outValue);
    }
}
