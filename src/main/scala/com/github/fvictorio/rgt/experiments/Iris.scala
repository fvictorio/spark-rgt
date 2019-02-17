package com.github.fvictorio.rgt.experiments

import com.github.fvictorio.rgt._
import org.apache.spark.sql.SparkSession

object Iris {
  def main(args: Array[String]) {
    val spark = SparkSession.builder.appName("Iris")
      .master("local[*]")
      .getOrCreate()
    val sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    sc.setCheckpointDir("/tmp")

    val path = if (args.length > 0)
      args(0)
    else
      this.getClass.getResource("/iris.csv").getPath

    val (rdd, rddWithCorrectLabel) = Datasets.loadIris(spark, path)

    val rgt = new RGT(maxIter = 5L, noNeighbors = 10, divisionStepMaxIter = 1000)

    val (result, _) = rgt.transform(rdd)

    val resultWithCorrectLabel = result
      .join(rddWithCorrectLabel)
      .map(_._2)

    val successRate = Helpers.computeSuccessRate(resultWithCorrectLabel)

    println(successRate)
  }
}
