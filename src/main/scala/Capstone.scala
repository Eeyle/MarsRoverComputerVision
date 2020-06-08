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

// scalastyle:off println

import org.apache.spark.sql.{SparkSession, SQLContext, Column, Row};
import org.apache.spark.sql.functions.{col, udf, monotonically_increasing_id};
import org.apache.spark.sql.types._; // the datatypes used when making a custom schema

import org.apache.spark.ml.image.ImageSchema;
import org.apache.spark.ml.linalg.{Vector, Vectors};
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType;

import org.apache.spark.ml.classification.MultilayerPerceptronClassifier;
import org.apache.spark.ml.classification.{LinearSVC, RandomForestClassifier};
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;

import org.apache.spark.ml.feature.VectorAssembler;//{StringIndexer, VectorAssembler, OneHotEncoderEstimator};
//import org.apache.spark.ml.Pipeline;
// import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

//import org.platanios.tensorflow.api._;
//import org.platanios.tensorflow.api.tf; // yeah this can be cleaned up
//import org.platanios.tensorflow.api.data._;

//import java.nio.ByteBuffer;

object Capstone {
  def main(args: Array[String]) {
    val spark = SparkSession
      .builder
      .appName("Capstone")
      .config("spark.master", "local[*]")
//      .config("spark.master", "spark://ee:7077")
//      .config("spark.master", "spark://192.168.1.2:7077")
      .getOrCreate();

//	import spark.implicits._; // some helpful conversions
//
//	var new_labels = spark
//		.read
//		.format("csv")
//		.option("inferSchema", true)
//		.option("header", true)
//		.load("data/train-calibrated-shuffled.csv");
//
//	var labels = Seq(
//		(0, 0),
//	   	(1, 0),
//	   	(2, 0),
//	   	(3, 0),
//	   	(4, 0),
//	   	(5, 0),
//	   	(6, 0),
//	   	(7, 0),
//	   	(8, 0),
//	   	(9, 0),
//	   	(10, 1),
//	   	(11, 1),
//	   	(12, 1),
//	   	(13, 1),
//	   	(14, 1),
//	   	(15, 1),
//	   	(16, 1),
//	   	(17, 1),
//	   	(18, 1),
//	   	(19, 1)
//	).toDF("id", "thelabel");
//
//	// convert Array[Byte] to Vector for use as features
//	val binaryToShortUDF = udf(
//		(image: Array[Byte]) => {	
//			Vectors.dense(
//				image.map(
//					// convert unsigned byte to signed byte and minmax scale
//					x => (x & 0xFF) / 255.0
//				)
//			)
//		},
//		VectorType // custom spark sql type created by spark.ml.linalg.SQLTypes
//	);
//	spark.udf.register("binaryToShort", binaryToShortUDF);
//
//
//	val imageDir = "data/test/"; 
//	var images = ImageSchema.readImages(
//		path = imageDir,
//		sparkSession = spark,
//		recursive = false,
//		numPartitions = 1,
//		dropImageFailures = false,
//		sampleRatio = 1.0,
//		seed = 2
//	).map(
//		image => {
//			var row = image.getAs[Row](0);
//			ImageSchema.getData(row)
//		}
//	)
//
//
//	var asdf = images.select(
//			binaryToShortUDF(col("value")) as "features"
//		).withColumn("id", monotonically_increasing_id)
//		.join(labels, "id")
//		.withColumnRenamed("thelabel", "label");
//
////	var asdf = spark.createDataFrame(
////		images.select(
////			binaryToShortUDF(col("value")) as "features"
////		).rdd,
////		StructType(Array(StructField("features", VectorType, false))) // the dataframe is an Array[Vector] at heart
////	)
////		.withColumn("id", monotonically_increasing_id)
////		.join(labels, "id")
////		.withColumnRenamed("thelabel", "label");
////	var the_df = images//asdf
////		.withColumn("id", monotonically_increasing_id)
////		.join(labels, "id")
////		.withColumnRenamed("thelabel", "label");
//
//

	var trainImages = spark.read
		.format("csv")
		.option("inferSchema", true)
		.option("header", false)
		.load("data/processed_csv/train_images.csv")
		.withColumn("label", col("_c4096").cast(IntegerType));

	var valImages = spark.read
		.format("csv")
		.option("inferSchema", true)
		.option("header", false)
		.load("data/processed_csv/val_images.csv")
		.withColumn("label", col("_c4096").cast(IntegerType));

	var testImages = spark.read
		.format("csv")
		.option("inferSchema", true)
		.option("header", false)
		.load("data/processed_csv/test_images.csv")
		.withColumn("label", col("_c4096").cast(IntegerType));


	var assembler = new VectorAssembler()
		.setInputCols(Array.range(0, 4096).map(x => s"_c$x"))
		.setOutputCol("features");

	trainImages = assembler transform trainImages;
	valImages = assembler transform valImages;
	testImages = assembler transform testImages;

  	// Initializing model
  	var evaluator = new MulticlassClassificationEvaluator()
		.setMetricName("accuracy");

//    val layers = Array[Int](4096, 1024, 16, 2); // input features, hidden layers, output classes
//    var nn = new MultilayerPerceptronClassifier()
//    	.setLayers(layers)
//    	.setBlockSize(128)
//    	.setSeed(2L)
//    	.setMaxIter(10);

//	var rf = new RandomForestClassifier()
//		.setFeatureSubsetStrategy("sqrt")
//		.setMaxDepth(7)
//		.setNumTrees(128)
//		.setSeed(2L);
	
	var svc = new LinearSVC()
		.setMaxIter(25)
		.setRegParam(0.01);

  	// Modeling
//  	var nnModel = nn fit trainImages;
//  	var predictions = nnModel transform valImages;

//	var rfModel = rf fit trainImages;
//	var predictions = rfModel transform valImages;

	var svcModel = svc fit trainImages;
	var predictions = svcModel transform valImages;

  	println(s"Evaluation ${evaluator evaluate predictions}");


    // ---
	println("\n-----\nDone!\n-----\n");

//	trainImages show;

//	asdf printSchema;
//	asdf show;
//	var image = (asdf takeAsList 20) get 0;
//	println(image getClass);

//	images show;
//	var image = (images takeAsList 20) get 0;
//	println(image);
//	println(image get 0);


    spark.stop();
  }
}
