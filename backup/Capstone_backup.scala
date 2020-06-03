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

import scala.math.random

import org.apache.spark.sql.{SparkSession, SQLContext}

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, OneHotEncoderEstimator}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

/** Computes an approximation to pi */
object Capstone {
  def main(args: Array[String]) {
    val spark = SparkSession
      .builder
      .appName("Capstone")
      //.config("spark.master", "local[*]")
      //.config("spark.master", "spark://ee:7077")
      .config("spark.master", "spark://192.168.1.2:7077")
      // debug configurations that I'm scared to delete
      //.config("spark.sql.autoBroadcastJoinThreshold", -1)
      //.config("spark.executor.memory", "6g")
      //.config("spark.sql.broadcastTimeout", 30000)//3000)
      //.config("spark.task.maxDirectResultSize", "6g")
      .getOrCreate()

    // ---
    // Reading data
    val df = spark.sqlContext
		  .read.format("csv")
		  .option("header", true)
		  .option("inferSchema", true)
      .load("data/Heart.csv")

    // Indexing data
    var labelIndexer = new StringIndexer()
      .setInputCol("AHD")
      .setOutputCol("label")
  	var chestPainIndexer = new StringIndexer()
      .setInputCol("ChestPain")
  		.setOutputCol("ChestPainIndexed")
  	var thalIndexer = new StringIndexer()
  		.setInputCol("Thal")
  		.setOutputCol("ThalIndexed")
  	var ohc = new OneHotEncoderEstimator()
  		.setInputCols(Array(chestPainIndexer.getOutputCol, thalIndexer.getOutputCol))
  		.setOutputCols(Array("ChestPainVec", "ThalVec"))

  	// Assembling features
  	var features = Array.concat(
  		df.columns.slice(1, df.columns.length - 1),
  		ohc.getOutputCols
  	)
  	features = features filter {x => x != chestPainIndexer.getInputCol && x != thalIndexer.getInputCol}
  	var assembler = new VectorAssembler()
  		.setInputCols(features)
  		.setOutputCol("features")

  	// Initializing model
  	var Array(train, test) = df randomSplit (Array(0.8, 0.2), 42)
  	var lr = new LogisticRegression()
  	var evaluator = new MulticlassClassificationEvaluator()
    		.setMetricName("accuracy")

    // NN setup
    // val layers = Array[Int](65536, 1024, 16, 2) // input features, hidden layers, output classes
    // var nn = new new MultilayerPerceptronClassifier()
    //   .setLayers(layers)
    //   .setBlockSize(128)
    //   .setSeed(1234L)
    //   .setMaxIter(100)

  	// Pipeline setup
  	var pipe = new Pipeline()
		.setStages(Array(
			labelIndexer,
			chestPainIndexer,
			thalIndexer,
			ohc,
			assembler,
			lr
		))

  	var paramGrid = new ParamGridBuilder()
  		.addGrid(lr.elasticNetParam, Array(0.1))
  		.addGrid(lr.regParam, Array(0.1))
  		.build()

  	var cv = new CrossValidator()
  		.setEstimator(pipe)
  		.setEstimatorParamMaps(paramGrid)
  		.setEvaluator(evaluator)

  	// Modeling
  	var cvModel = cv fit train
  	var predictions = cvModel transform test
  	println(s"Evaluation ${evaluator evaluate predictions}")
  	println(cvModel getEstimatorParamMaps (cvModel.avgMetrics indexOf (cvModel.avgMetrics max)))

    // ---
    spark.stop()
  }
}
