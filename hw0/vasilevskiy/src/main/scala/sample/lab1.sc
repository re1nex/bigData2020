
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._


val spark = org.apache.spark.sql.SparkSession.builder
  .master("local")
  .appName("Task1")
  .getOrCreate;

var df = spark
  .read
  .option("header", "true")
  .option("quote", "\"")
  .option("mode", "DROPMALFORMED")
  .option("escape", "\"")
  .option("inferSchema", "true")
  .csv("src/main/resources/AB_NYC_2019.csv");

df = df
  .withColumn("price", df("price").cast("Integer"))
  .withColumn("latitude", df("latitude").cast("Double"))
  .withColumn("longitude", df("longitude").cast("Double"))
  .withColumn("minimum_nights", df("minimum_nights").cast("Integer"))
  .withColumn("number_of_reviews", df("number_of_reviews").cast("Integer"))


//1
df.createOrReplaceTempView("df")
spark.sql("select room_type, mean(price) as mean from df group by room_type ").show()
spark.sql("select room_type, percentile_approx(price, 0.5) as median from df group by room_type ").show()
spark.sql("select room_type, variance(price) as var from df group by room_type ").show()
df.groupBy("room_type", "price")
  .count()
  .withColumn("row_number", row_number().over(Window.partitionBy("room_type").orderBy(desc("count"))))
  .select("room_type", "price")
  .where(col("row_number") === 1)
  .show()

//2
df.orderBy("price").show(1)
df.orderBy(col("price").desc).show(1)

//3
df.agg(corr("price", "minimum_nights").as("corr(price,minimum_nights)"),
  corr("price", "number_of_reviews").as("corr(price,number_of_reviews)"))
  .show()

//4
val encodeGeoHash = (lat: Double, lng: Double, precision: Int) => {
  val base32 = "0123456789bcdefghjkmnpqrstuvwxyz"
  var (minLat, maxLat) = (-90.0, 90.0)
  var (minLng, maxLng) = (-180.0, 180.0)
  val bits = List(16, 8, 4, 2, 1)

  (0 until precision).map { p => {
    base32 apply (0 until 5).map { i => {
      if (((5 * p) + i) % 2 == 0) {
        val mid = (minLng + maxLng) / 2.0
        if (lng > mid) {
          minLng = mid
          bits(i)
        } else {
          maxLng = mid
          0
        }
      } else {
        val mid = (minLat + maxLat) / 2.0
        if (lat > mid) {
          minLat = mid
          bits(i)
        } else {
          maxLat = mid
          0
        }
      }
    }
    }.reduceLeft((a, b) => a | b)
  }
  }.mkString("")
}

val geoHash_udf = udf(encodeGeoHash)

df
  .withColumn("geoHash", geoHash_udf(col("latitude"), col("longitude"), lit(5)))
  .groupBy("geoHash")
  .agg(
    avg("price").as("avg_price")
  )
  .orderBy(desc("avg_price"))
  .show(1)