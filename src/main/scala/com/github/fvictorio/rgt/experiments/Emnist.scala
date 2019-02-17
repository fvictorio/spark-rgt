package com.github.fvictorio.rgt.experiments

import com.github.fvictorio.rgt._
import com.github.fvictorio.rgt.Helpers.parseArgs
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{SparkSession, functions => f}
import org.apache.log4j.{LogManager, Logger}
import org.apache.spark.rdd.RDD

object Emnist {
  val logger: Logger = LogManager.getLogger(this.getClass)

  def main(args: Array[String]) {
    val argsMap = parseArgs(args.toList, Map(
      "weightFeatures" -> "true",
      "partitions" -> "4",
      "maxIter" -> "20",
      "noNeighbors" -> "10",
      "divisionStepMaxIter" -> "1000",
      "divisionStepMaxIterWithoutChanges" -> "10",
      "checkpointDir" -> "/tmp",
      "randomMin" -> "0",
      "randomMax" -> "8",
      "graphConstructionMaxIter" -> "5",
      "graphDivisionEarlyTermination" -> "0.01",
      "relieffM" -> "50",
      "relieffK" -> "3",
      "graphConstructionBucketsPerInstance" -> "4",
      "graphConstructionSampleRate" -> "1.0"
    ))

    logger.info(s"Using args ${argsMap.toSeq.sortBy(_._1).mkString(", ")}")

    val spark = SparkSession.builder
      .appName("EMNIST")
       .master("local[*]")
      .getOrCreate()
    val sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    sc.setCheckpointDir(argsMap("checkpointDir"))

    val path = argsMap.getOrElse("dataset", this.getClass.getResource("/emnist-1k.csv").getPath)
    val partitions = argsMap("partitions").toInt
    val randomMin = argsMap("randomMin").toInt
    val randomMax = argsMap("randomMax").toInt

    val (rdd, rddWithCorrectLabel) = Datasets.loadEmnist(spark, path, randomMin = randomMin, randomMax = randomMax)

    val rgt = new RGT(
      weightFeatures = argsMap("weightFeatures").toBoolean,
      maxIter = argsMap("maxIter").toLong,
      noNeighbors = argsMap("noNeighbors").toInt,
      divisionStepMaxIter = argsMap("divisionStepMaxIter").toInt,
      divisionStepMaxIterWithoutChanges = argsMap("divisionStepMaxIterWithoutChanges").toInt,
      rddPartitions = partitions,
      graphConstructionMaxIter = argsMap("graphConstructionMaxIter").toInt,
      graphDivisionEarlyTermination = argsMap("graphDivisionEarlyTermination").toDouble,
      relieffM = argsMap("relieffM").toLong,
      relieffK = argsMap("relieffK").toLong,
      graphConstructionBucketsPerInstance = argsMap("graphConstructionBucketsPerInstance").toInt,
      graphConstructionSampleRate = argsMap("graphConstructionSampleRate").toDouble
    )

    val (result, _) = rgt.transform(rdd)

    val resultWithCorrectLabel = result
      .join(rddWithCorrectLabel)
      .map(_._2)

    val successRate = Helpers.computeSuccessRate(resultWithCorrectLabel)

    println(successRate)
  }
}
