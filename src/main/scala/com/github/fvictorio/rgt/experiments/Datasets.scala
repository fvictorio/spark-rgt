package com.github.fvictorio.rgt.experiments

import com.github.fvictorio.rgt.RGTInputNode
import org.apache.spark.ml.linalg.{Vector,Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{SparkSession, functions => f}

object Datasets {
  def loadEmnist(spark: SparkSession, path: String, partitions: Int = 100, randomMin: Int = 0, randomMax: Int = 8): (RDD[(Long, RGTInputNode)], RDD[(Long, Long)]) = {
    import spark.implicits._

    val dfRaw = spark.read.format("csv")
      .option("header", false)
      .load(path)

    val dfLabeled = dfRaw
      .map(row => {
        val label = row.getString(0).toLong
        val head = row.getString(1).toDouble
        val tail = (2 to 784).map(i => row.getString(i).toDouble)
        val features = Vectors.dense(head, tail: _*)

        (label, features)
      })
      .withColumn("id", f.monotonically_increasing_id())
      .toDF("correctLabel", "features", "id")
      .select("id", "features", "correctLabel")

    val extractLabel = f.udf((label: Long, features: Vector) => {
      val hashCode = features.toArray.toSeq.hashCode
      val modulo = (hashCode % 100 + 100) % 100
      if (randomMin <= modulo && modulo < randomMax) {
        Some(label)
      } else {
        None
      }
    })

    val df = dfLabeled
      .withColumn("label", extractLabel($"correctLabel", $"features"))
      .repartition(partitions)
      .persist

    val rddWithCorrectLabel = df
      .select("id", "correctLabel")
      .map(row => (row.getLong(0), row.getLong(1)))
      .rdd

    rddWithCorrectLabel.count

    val rdd: RDD[(Long, RGTInputNode)] = df
      .select("id", "features", "label")
      .map(row => {
        val labelAny = row.get(2)
        val label: Option[Long] = if (labelAny == null) None else Some(labelAny.asInstanceOf[Long])
        (row.getLong(0), RGTInputNode(row.getAs[Vector](1), label))
      })
      .rdd

    (rdd, rddWithCorrectLabel)
  }

  def loadIris(spark: SparkSession, path: String = this.getClass.getResource("/iris.csv").getPath, randomMin: Int = 0, randomMax: Int = 8): (RDD[(Long, RGTInputNode)], RDD[(Long, Long)]) = {
    import spark.implicits._

    val dfRaw = spark.read.format("csv")
      .option("header", true)
      .load(path)

    val extractCorrectLabel = f.udf((label: String) => {
      if (label == "Iris-setosa")
        0L
      else if (label == "Iris-versicolor")
        1L
      else {
        2L
      }
    })

    val extractLabel = f.udf((features: Vector, label: Long) => {
      val hashCode = features.toArray.toSeq.hashCode
      val modulo = (hashCode % 100 + 100) % 100
      if (randomMin <= modulo && modulo < randomMax) {
        Some(label)
      } else {
        None
      }
    })

    val extractFeatures = f.udf((x1: String, x2: String, x3: String, x4: String) => Vectors.dense(x1.toDouble, x2.toDouble, x3.toDouble, x4.toDouble))

    val df = dfRaw
      .withColumn("id", f.monotonically_increasing_id)
      .withColumn("correctLabel", extractCorrectLabel($"Species"))
      .withColumn("features", extractFeatures($"SepalLength", $"SepalWidth", $"PetalLength", $"PetalWidth"))
      .withColumn("label", extractLabel($"features", $"correctLabel"))
      .drop("Species")
      .drop("SepalLength")
      .drop("SepalWidth")
      .drop("PetalLength")
      .drop("PetalWidth")

    val rdd: RDD[(Long, RGTInputNode)] = df
      .select("id", "features", "label")
      .map(row => {
        val labelAny = row.get(2)
        val label: Option[Long] = if (labelAny == null) None else Some(labelAny.asInstanceOf[Long])
        (row.getLong(0), RGTInputNode(row.getAs[Vector](1), label))
      })
      .rdd

    val rddWithCorrectLabel = df
      .select("id", "correctLabel")
      .map(row => (row.getLong(0), row.getLong(1)))
      .rdd

    (rdd, rddWithCorrectLabel)
  }
}
