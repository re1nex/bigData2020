
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DataTypes, IntegerType, StructType}


val spark = org.apache.spark.sql.SparkSession.builder
  .master("local")
  .appName("Task3")
  .getOrCreate

spark.sparkContext.setLogLevel("OFF")

val input = spark.readStream
  .format("socket")
  .option("host", "localhost")
  .option("port", 8000)
  .load()

val schema = new StructType()
  .add("id", IntegerType)
  .add("text", DataTypes.StringType);

val inputJson =
  input.withColumn("json", from_json(col("value"), schema))
    .select("json.*")
    .filter(col("text").isNotNull)
    .filter(col("id").isNotNull)
    .select(col("id"), col("text"))


val model = PipelineModel.read.load("model/")
val result = model.transform(inputJson.select(col("id"), col("text")))
  .select(col("id"), col("target").as("target").cast(DataTypes.IntegerType))

result
  .repartition(1)
  .writeStream
  .outputMode("append")
  .format("com.databricks.spark.csv")
  .option("header", "true")
  .option("path", "data")
  .option("checkpointLocation", "checkpoint")
  .start()
  .awaitTermination()
