package org.minerva.spark;

import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import scala.Tuple2;

public final class JavaWordCount {
	private static final Pattern SPACE = Pattern.compile(" ");

	public static void main(String[] args) throws Exception {

		if (args.length < 1) {
			System.err.println("Usage: JavaWordCount <file> <master>");
			System.exit(1);
		}

		SparkConf sparkConf = new SparkConf().setAppName("JavaWordCount");

		sparkConf.setMaster(sparkConf.get("master", "local"));

		JavaSparkContext ctx = new JavaSparkContext(sparkConf);
		JavaRDD<String> lines = ctx.textFile(args[0], 1);

		JavaRDD<String> words = lines.flatMap(x -> (List<String>) (Arrays
				.asList(SPACE.split(x))));

		JavaPairRDD<String, Integer> ones = words
				.mapToPair(x -> new Tuple2<String, Integer>(x, 1));

		JavaPairRDD<String, Integer> counts = ones.reduceByKey((Integer i1,
				Integer i2) -> i1 + i2);

		List<Tuple2<String, Integer>> output = counts.collect();

		System.out.println(output);

		ctx.stop();
		ctx.close();
	}
}
