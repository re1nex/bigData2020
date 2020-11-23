import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassifier, GBTClassifier}
import org.apache.spark.ml.feature._
import org.apache.spark.mllib.feature.Stemmer
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType

val spark = org.apache.spark.sql.SparkSession.builder
  .master("local")
  .appName("Task2")
  .getOrCreate;
val train = spark
  .read
  .option("header", "true")
  .option("inferSchema", "true")
  .csv("src/main/resources/train.csv")
  .filter(col("text").isNotNull)
  .filter(col("target").isNotNull)
  .select("id", "text", "target")
  .withColumnRenamed("target", "label")

val test = spark
  .read
  .option("header", "true")
  .option("inferSchema", "true")
  .csv("src/main/resources/test.csv")
  .filter(col("text").isNotNull)
  .filter(col("id").isNotNull)
  .select("id", "text")

val sample = spark
  .read
  .option("header", "true")
  .option("inferSchema", "true")
  .csv("src/main/resources/sample_submission.csv")

val regexTokenizer = new RegexTokenizer()
  .setInputCol("text")
  .setOutputCol("token")
  .setPattern("[\\W]")
  .setToLowercase(true)

val remover = new StopWordsRemover()
  .setInputCol("token")
  .setOutputCol("filtered")

val stemmer = new Stemmer()
  .setInputCol("filtered")
  .setOutputCol("stemmed")
  .setLanguage("English")

val hashingTF = new HashingTF()
  .setInputCol("stemmed")
  .setOutputCol("rowFeatures")
  .setNumFeatures(10000)

val idf = new IDF()
  .setInputCol("rowFeatures")
  .setOutputCol("features")

val labelIndexer = new StringIndexer()
  .setInputCol("label")
  .setOutputCol("indexedLabel")

val gbt = new GBTClassifier()
  .setLabelCol("label")
  .setFeaturesCol("features")
  .setPredictionCol("target")
  .setMaxIter(25)

val pipeline = new Pipeline()
  .setStages(Array(regexTokenizer, remover, stemmer, hashingTF, idf, labelIndexer, gbt))

var result = pipeline.fit(train)
  .transform(test)

result = result
  .select("id", "target")
  .withColumn("target", result("target").cast(IntegerType))

result = result.join(sample, sample.col("id").equalTo(result.col("id")), "right")
  .select(sample.col("id"), when(result.col("id").isNull, lit(0)).otherwise(col("target")).as("target"))

result.write.option("header", "true").option("inferSchema", "true").csv("src/main/resources/result.csv")