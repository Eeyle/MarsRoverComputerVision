
import org.apache.spark.sql.{SparkSession, SQLContext} // , Column, Row};
import org.apache.spark.sql.functions.col; // {col, udf, monotonically_increasing_id};
import org.apache.spark.sql.types._; // the datatypes used when making a custom schema or when casting in sql selects

//import org.apache.spark.ml.image.ImageSchema;
//import org.apache.spark.ml.linalg.{Vector, Vectors};
//import org.apache.spark.ml.linalg.SQLDataTypes.VectorType; // custom spark.sql.types Type for Vector classes

import org.apache.spark.ml.classification.MultilayerPerceptronClassifier;
import org.apache.spark.ml.classification.{LinearSVC, RandomForestClassifier};
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;

import org.apache.spark.ml.feature.VectorAssembler; 
//import org.apache.spark.ml.Pipeline;
//import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

// Scala implementation of Tensorflow API version 0.3 community-maintained
// would have loved to implement a CNN but too many issues arose from the unfinished and community-maintained api
//import org.platanios.tensorflow.api._;
//import org.platanios.tensorflow.api.tf; 
//import org.platanios.tensorflow.api.data._;


object Capstone {
  def main(args: Array[String]) {
    val spark = SparkSession
      .builder
      .appName("Capstone")
      .config("spark.master", "local[*]")
//      .config("spark.master", "spark://ee:7077") // 
//      .config("spark.master", "spark://192.168.1.2:7077") // IP of Spark Master
      .getOrCreate();

	// --== Loading Data ==--
	// Image sets were converted to csv files for simple loading and distribution with Spark
	// the final column _c4096 is the label
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

	// Each image is 64x64 greyscale leading to 4096 features each
	var assembler = new VectorAssembler()
		.setInputCols(Array.range(0, 4096).map(x => s"_c$x"))
		.setOutputCol("features");

	trainImages = assembler transform trainImages;
	valImages = assembler transform valImages;
	testImages = assembler transform testImages;

  	// --== Modeling ==--
  	var evaluator = new MulticlassClassificationEvaluator()
		.setMetricName("accuracy");

//	// MLP NN
//    val layers = Array[Int](4096, 1024, 16, 2); // input features, hidden layers, output classes
//    var nn = new MultilayerPerceptronClassifier()
//    	.setLayers(layers)
//    	.setBlockSize(128)
//    	.setSeed(2L)
//    	.setMaxIter(10);

	// Random Forest
	var rf = new RandomForestClassifier()
		.setFeatureSubsetStrategy("sqrt")
		.setMaxDepth(16)
		.setNumTrees(256)
		.setSeed(2L);
	
//	// SVM
//	var svc = new LinearSVC()
//		.setMaxIter(25)
//		.setRegParam(0.01);

  	//  Modeling
//  	var nnModel = nn fit trainImages;
//  	var predictions = nnModel transform valImages;

	var rfModel = rf fit trainImages;
	var predictions = rfModel transform valImages;

//	var svcModel = svc fit trainImages;
//	var predictions = svcModel transform valImages;

  	println(s"Evaluation ${evaluator evaluate predictions}");

	println("\n-----\nDone!\n-----\n");
    spark.stop();
  }
}

// -- deprecated attempt --
// /* I attempted to use Spark's built-in image loading system to load images directly from file into Spark. I was able to process the data correctly with enough effort, but the solution was clunky at best and used far too much memory. Instead, image data was processed in Python in data/processing/csv_conversion.ipynb, converting each set of images to a single csv file which is ready to be loaded by Spark. */
//
//	import spark.implicits._; // some helpful type conversions
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
// // It was these few next line that were particularly difficult, clunky, and not robust to expansion.
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
