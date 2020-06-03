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
import org.apache.spark.ml.classification.LinearSVC;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;

//import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, OneHotEncoderEstimator};
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
      //.config("spark.master", "spark://ee:7077")
//      .config("spark.master", "spark://192.168.1.2:7077")
      .getOrCreate();

	import spark.implicits._; // some helpful conversions


	val imageDir = "data/test/"; 
	var images = ImageSchema.readImages(
		path = imageDir,
		sparkSession = spark,
		recursive = false,
		numPartitions = 1,
		dropImageFailures = false,
		sampleRatio = 1.0,
		seed = 2
	)

	var image_data = images
		.map(
			image => {
				var row = image.getAs[Row](0);
				ImageSchema.getData(row);	
			}
		)


	val binaryToShort = udf(
		(image: Array[Byte]) => {	
			Vectors.dense(
				image.map(
					x => (x & 0xFF) / 255.0
				)
			)
		},
		VectorType
	);
	spark.udf.register("binaryToShort", binaryToShort);

	var asdf = spark.createDataFrame(
		image_data.select(
			binaryToShort(col("value")) as "features"
		).rdd,
		StructType(Array(StructField("features", VectorType, false)))
	);


	var labels = Seq(
		(0, 0),
		(1, 0),
		(2, 0),
		(3, 0),
		(4, 0),
		(5, 0),
		(6, 0),
		(7, 0),
		(8, 0),
		(9, 0),
		(10, 1),
		(11, 1),
		(12, 1),
		(13, 1),
		(14, 1),
		(15, 1),
		(16, 1),
		(17, 1),
		(18, 1),
		(19, 1)
	).toDF("id", "thelabel");

	var the_df = asdf
		.withColumn("id", monotonically_increasing_id)
		.join(labels, "id")
		.withColumnRenamed("thelabel", "label");


  	// Initializing model
  	var Array(train, test) = the_df randomSplit (Array(0.8, 0.2), 2);
  	var evaluator = new MulticlassClassificationEvaluator()
		.setMetricName("accuracy");

	asdf = spark.emptyDataFrame;
//	the_df = spark.emptyDataFrame;
	image_data = spark.emptyDataset[Array[Byte]];
	images = spark.emptyDataFrame;

    val layers = Array[Int](4096, 1024, 16, 2); // input features, hidden layers, output classes
    var nn = new MultilayerPerceptronClassifier()
    	.setLayers(layers)
    	.setBlockSize(128)
    	.setSeed(2L)
    	.setMaxIter(10);


  	// Modeling
  	var nnModel = nn fit the_df;
  	var predictions = nnModel transform the_df;
  	println(s"Evaluation ${evaluator evaluate predictions}");

    // ---
	println("\n-----\nDone!\n-----\n");

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
