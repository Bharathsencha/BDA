package com.bda.surge;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

/**
 * SurgeDriver:
 * Main Driver class for Hadoop MapReduce Surge Price Prediction Job.
 * Configures and triggers distributed feature extraction across Chicago Taxi trip logs.
 *
 * Usage:
 *   hadoop jar surge-price-prediction.jar com.bda.surge.SurgeDriver <input_path> <output_path>
 */
public class SurgeDriver extends Configured implements Tool {

    @Override
    public int run(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: SurgeDriver <input_path> <output_path>");
            System.err.println("Example: hadoop jar surge.jar com.bda.surge.SurgeDriver /data/raw/taxi_trips /data/processed/surge_features");
            return -1;
        }

        Configuration conf = getConf();
        Job job = Job.getInstance(conf, "Chicago Taxi Surge Price Prediction Feature Aggregator");

        job.setJarByClass(SurgeDriver.class);

        // Set Mapper and Reducer
        job.setMapperClass(SurgeMapper.class);
        job.setReducerClass(SurgeReducer.class);

        // Set Key and Value Output Classes
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        // Set Input and Output Formats
        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        // Input and Output Paths (HDFS or Local)
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        System.out.println("===============================================================");
        System.out.println("Starting Hadoop MapReduce Job: Surge Price Feature Aggregator");
        System.out.println("Input Path : " + args[0]);
        System.out.println("Output Path: " + args[1]);
        System.out.println("===============================================================");

        boolean success = job.waitForCompletion(true);
        return success ? 0 : 1;
    }

    public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new Configuration(), new SurgeDriver(), args);
        System.exit(exitCode);
    }
}
