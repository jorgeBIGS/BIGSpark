/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.minerva.spark;

import java.util.HashMap;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.mllib.util.MLUtils;

import scala.Tuple2;

public final class JavaRandomForest {

	private static RandomForestModel model;

	/**
	 * Note: This example illustrates binary classification. For information on
	 * multiclass classification, please refer to the JavaDecisionTree.java
	 * example.
	 */
	private static JavaPairRDD<Double, Double> getPredictions(
			JavaRDD<LabeledPoint> trainingData, JavaRDD<LabeledPoint> testData) {
		// Train a RandomForest model.
		// Empty categoricalFeaturesInfo indicates all features are continuous.
		Integer numClasses = 2;
		HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
		Integer numTrees = 3; // Use more in practice.
		String featureSubsetStrategy = "auto"; // Let the algorithm choose.
		String impurity = "gini";
		Integer maxDepth = 4;
		Integer maxBins = 32;
		Integer seed = 12345;

		model = RandomForest.trainClassifier(trainingData, numClasses,
				categoricalFeaturesInfo, numTrees, featureSubsetStrategy,
				impurity, maxDepth, maxBins, seed);

		// Evaluate model on test instances and compute test error
		JavaPairRDD<Double, Double> predictionAndLabel = testData
				.mapToPair(p -> new Tuple2<Double, Double>(model.predict(p
						.features()), p.label()));

		return predictionAndLabel;
	}

	private static void testClassification(JavaRDD<LabeledPoint> trainingData,
			JavaRDD<LabeledPoint> testData) {
		JavaPairRDD<Double, Double> predictionAndLabel = getPredictions(
				trainingData, testData);
		Double testErr = 1.0
				* predictionAndLabel.filter(pl -> pl._1().equals(pl._2()))
						.count() / testData.count();
		System.out.println("Test Error: " + testErr);
		System.out.println("Learned classification forest model:\n"
				+ model.toDebugString());
	}

	private static void testRegression(JavaRDD<LabeledPoint> trainingData,
			JavaRDD<LabeledPoint> testData) {
		JavaPairRDD<Double, Double> predictionAndLabel = getPredictions(
				trainingData, testData);

		Double testMSE = predictionAndLabel.map(
				pl -> Math.pow(pl._1() - pl._2(), 2)).reduce((a, b) -> a + b)
				/ testData.count();
		System.out.println("Test Mean Squared Error: " + testMSE);
		System.out.println("Learned regression forest model:\n"
				+ model.toDebugString());
	}

	public static void main(String[] args) {
		SparkConf sparkConf = new SparkConf()
				.setAppName("JavaRandomForestExample");
		sparkConf.setMaster(sparkConf.get("master", "local"));
		JavaSparkContext sc = new JavaSparkContext(sparkConf);

		// Load and parse the data file.
		String datapath = args[1];
		JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(sc.sc(), datapath)
				.toJavaRDD();
		// Split the data into training and test sets (30% held out for testing)
		JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[] { 0.7,
				0.3 });
		JavaRDD<LabeledPoint> trainingData = splits[0];
		JavaRDD<LabeledPoint> testData = splits[1];

		System.out
				.println("\nRunning example of classification using RandomForest\n");
		testClassification(trainingData, testData);

		System.out
				.println("\nRunning example of regression using RandomForest\n");
		testRegression(trainingData, testData);
		sc.stop();
		sc.close();
	}
}
