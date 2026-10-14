package com.bda.surge;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;

/**
 * SurgeMapper:
 * Parses raw Chicago Taxi Trip records from CSV.
 * Emits spatio-temporal events (Pickups and Dropoffs) keyed by (ZoneID \t TimeBucket).
 * Value: EventType (P or D), TripMiles, TripSeconds, SurgeMultiplier
 */
public class SurgeMapper extends Mapper<LongWritable, Text, Text, Text> {

    // Regulated Chicago Taxi Base Rate Constants
    private static final double BASE_FLAG_DROP = 3.25;
    private static final double PER_MILE_RATE = 2.25;
    private static final double PER_MINUTE_RATE = 0.40;

    private final Text outKey = new Text();
    private final Text outValue = new Text();

    private static final DateTimeFormatter[] DATE_FORMATTERS = new DateTimeFormatter[]{
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"),
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm"),
            DateTimeFormatter.ofPattern("MM/dd/yyyy hh:mm:ss a"),
            DateTimeFormatter.ofPattern("MM/dd/yyyy HH:mm")
    };

    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String line = value.toString().trim();
        if (line.isEmpty() || line.startsWith("trip_id") || line.startsWith("Trip ID")) {
            return;
        }

        String[] tokens = line.split(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)");
        if (tokens.length < 8) {
            return;
        }

        try {
            // Field mappings based on Chicago Taxi CSV
            // 0: trip_id, 1: trip_start_timestamp, 3: trip_seconds, 4: trip_miles,
            // 5: pickup_community_area, 6: dropoff_community_area, 7: fare
            String timestampStr = tokens[1].replace("\"", "").trim();
            double tripSeconds = Double.parseDouble(tokens[3].replace("\"", "").trim());
            double tripMiles = Double.parseDouble(tokens[4].replace("\"", "").trim());
            String pickupAreaStr = tokens[5].replace("\"", "").trim();
            String dropoffAreaStr = (tokens.length > 6) ? tokens[6].replace("\"", "").trim() : "";
            double fare = Double.parseDouble(tokens[7].replace("\"", "").trim());

            if (pickupAreaStr.isEmpty() || tripMiles <= 0 || fare <= 0 || tripSeconds <= 0) {
                return;
            }

            int pickupZone = (int) Double.parseDouble(pickupAreaStr);
            if (pickupZone < 1 || pickupZone > 77) {
                return;
            }

            String timeBucket = extractTimeBucket(timestampStr);
            if (timeBucket == null) {
                return;
            }

            // Expected base fare under regulated rates
            double expectedBaseFare = BASE_FLAG_DROP + (tripMiles * PER_MILE_RATE) + ((tripSeconds / 60.0) * PER_MINUTE_RATE);
            double surgeMultiplier = (expectedBaseFare > 0) ? (fare / expectedBaseFare) : 1.0;
            surgeMultiplier = Math.round(surgeMultiplier * 100.0) / 100.0;

            // 1. Emit Pickup event (demand pressure)
            outKey.set(String.format("%d\t%s", pickupZone, timeBucket));
            outValue.set(String.format("P,%.2f,%.0f,%.2f", tripMiles, tripSeconds, surgeMultiplier));
            context.write(outKey, outValue);

            // 2. Emit Dropoff event (fleet supply arrival)
            if (!dropoffAreaStr.isEmpty()) {
                try {
                    int dropoffZone = (int) Double.parseDouble(dropoffAreaStr);
                    if (dropoffZone >= 1 && dropoffZone <= 77) {
                        outKey.set(String.format("%d\t%s", dropoffZone, timeBucket));
                        outValue.set(String.format("D,%.2f,%.0f,1.00", tripMiles, tripSeconds));
                        context.write(outKey, outValue);
                    }
                } catch (NumberFormatException ignored) {
                }
            }

        } catch (NumberFormatException | ArrayIndexOutOfBoundsException ignored) {
        }
    }

    private String extractTimeBucket(String timestampStr) {
        String cleanTs = timestampStr.replace("T", " ").trim();
        if (cleanTs.contains(".")) {
            cleanTs = cleanTs.substring(0, cleanTs.indexOf('.'));
        }

        for (DateTimeFormatter dtf : DATE_FORMATTERS) {
            try {
                LocalDateTime ldt = LocalDateTime.parse(cleanTs, dtf);
                int minute = ldt.getMinute();
                int flooredMinute = (minute / 15) * 15;
                LocalDateTime bucketTime = ldt.withMinute(flooredMinute).withSecond(0).withNano(0);
                return bucketTime.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm"));
            } catch (DateTimeParseException ignored) {
            }
        }
        return null;
    }
}
