package com.github.fvictorio.rgt

import java.util
import java.util.function.Consumer

import breeze.linalg.DenseVector
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.api.java.function.PairFlatMapFunction
import org.apache.spark.sql.{DataFrame, functions => f}
import org.apache.log4j.{LogManager, Logger}
import org.apache.spark.rdd.RDD

import scala.collection.mutable

object Helpers {
  def multiply(v1: Vector, v2: Vector): Vector = {
    assert(v1.size == v2.size, "Cannot multiply vectors of different size")

    val multiplied = v1.toArray.zip(v2.toArray).map { case (x, y) => x * y}

    val result = Vectors.dense(multiplied.head, multiplied.tail : _*)

    result
  }

  def subtract(v1: Vector, v2: Vector): Vector = {
    assert(v1.size == v2.size, "Cannot subtract vectors of different size")

    val bv1 = new DenseVector(v1.toArray)
    val bv2 = new DenseVector(v2.toArray)

    Vectors.dense((bv1 - bv2).toArray)
  }

  def ones(size: Int): Vector = {
    val zeros = Vectors.zeros(size)
    val onesArray = zeros.toArray.map(_ => 1.0)
    val ones = Vectors.dense(onesArray.head, onesArray.tail : _*)

    ones
  }

  def computeSuccessRate(rdd: RDD[(RGTOutputNode, Long)]): Double = {
    val N = rdd.filter(_._1.label.isEmpty).count

    val success = rdd
        .filter{case (node, correctLabel) => node.label.isEmpty && node.prediction == correctLabel}
      .count

    success.toDouble / N
  }

  def computeGM(rdd: RDD[(RGTOutputNode, Long)]): Double = {
    val (tp, fp, tn, fn) = rdd
      .map[(Long, Long, Long, Long)]{case (node, correctLabel) => {
        if (correctLabel == 0 && node.prediction == 0)
          (0L, 0L, 1L, 0L)
        else if (correctLabel == 0 && node.prediction == 1)
          (0L, 1L, 0L, 0L)
        else if (correctLabel == 1 && node.prediction == 0)
          (0L, 0L, 0L, 1L)
        else if (correctLabel == 1 && node.prediction == 1)
          (1L, 0L, 0L, 0L)
        else {
          assert(false, "computeGM only works in two-classes problems")
          (0L, 0L, 0L, 0L)
        }
      }}
      .reduce{case ((tp1, fp1, tn1, fn1), (tp2, fp2, tn2, fn2)) => (tp1 + tp2, fp1 + fp2, tn1 + tn2, fn1 + fn2)}

    val sensitivity = tp.toDouble / (tp + fn)
    val specificity = tn.toDouble / (tn + fp)

    Math.sqrt(sensitivity * specificity)
  }


  def computeLabelToM(labels: Seq[Long]): Map[Long, Double] = {
    val labelToM = labels
      .distinct
      .sorted
      .zipWithIndex
      .map { case (label, index) => (label, 2.0 * ((index + 1) % 2) - 1.0) }
      .toMap

    labelToM
  }

  def computePartitionAndLabelToM(rdd: RDD[(Long, RGTNodeWithNeighbors)]): Map[(Long, Long), Double] = {
    val labelsPerPartition = rdd
      .map(node => (node._2.partition, node._2.label, node._2.finished))
      .filter{case (_, label, finished) => label.nonEmpty && !finished}
      .distinct
      .collect
      .groupBy(_._1)
      .mapValues(_.map(_._2.get))

    val partitionAndLabelToM = labelsPerPartition
      .mapValues(computeLabelToM(_))
      .mapValues(_.toSeq)
      .toSeq
      .flatMap{ case (partition, labelAndM) => labelAndM.map(x => ((partition, x._1), x._2))}
      .toMap

    partitionAndLabelToM
  }

  def printMap[T](map: mutable.Map[Long, T]): String = {
    printMap(map.toMap)
  }
  def printMap[T](map: Map[Long, T]): String = {
    val sb = new mutable.StringBuilder()
    map.toSeq.sortBy(_._1).foreach{case(partition, value) => {
      sb.append(s"${partition}: ${value}\n")
    }}

    sb.mkString
  }

  def parseArgs(args: List[String], argsMap: Map[String, String]): Map[String, String] = {
    args match {
      case Nil => argsMap
      case "--dataset" :: value :: tail => parseArgs(tail, argsMap ++ Map("dataset" -> value))
      case "--partitions" :: value :: tail => parseArgs(tail, argsMap ++ Map("partitions" -> value))
      case "--max-iter" :: value :: tail => parseArgs(tail, argsMap ++ Map("maxIter" -> value))
      case "--weight-features" :: value :: tail => parseArgs(tail, argsMap ++ Map("weightFeatures" -> value))
      case "--no-neighbors" :: value :: tail => parseArgs(tail, argsMap ++ Map("noNeighbors" -> value))
      case "--division-step-max-iter" :: value :: tail => parseArgs(tail, argsMap ++ Map("divisionStepMaxIter" -> value))
      case "--division-step-max-iter-without-changes" :: value :: tail => parseArgs(tail, argsMap ++ Map("divisionStepMaxIterWithoutChanges" -> value))
      case "--checkpoint-dir" :: value :: tail => parseArgs(tail, argsMap ++ Map("checkpointDir" -> value))
      case "--random-min" :: value :: tail => parseArgs(tail, argsMap ++ Map("randomMin" -> value))
      case "--random-max" :: value :: tail => parseArgs(tail, argsMap ++ Map("randomMax" -> value))
      case "--graph-construction-max-iter" :: value :: tail => parseArgs(tail, argsMap ++ Map("graphConstructionMaxIter" -> value))
      case "--graph-construction-buckets-per-instance" :: value :: tail => parseArgs(tail, argsMap ++ Map("graphConstructionBucketsPerInstance" -> value))
      case "--graph-construction-sample-rate" :: value :: tail => parseArgs(tail, argsMap ++ Map("graphConstructionSampleRate" -> value))
      case "--graph-division-early-termination" :: value :: tail => parseArgs(tail, argsMap ++ Map("graphDivisionEarlyTermination" -> value))
      case "--gini-impurity" :: value :: tail => parseArgs(tail, argsMap ++ Map("giniImpurity" -> value))
      case "--relieff-m" :: value :: tail => parseArgs(tail, argsMap ++ Map("relieffM" -> value))
      case "--relieff-k" :: value :: tail => parseArgs(tail, argsMap ++ Map("relieffK" -> value))
      case option :: _ => println(s"Unknown option '${option}")
                          sys.exit(1)
    }
  }

  def dumpLabelCount(rdd: RDD[(Long, RGTNode)]): String = {
    val labeledPerPartition = rdd
      .values
      .filter(_.label.nonEmpty)
      .map(node => ((node.partition, node.label.get, node.finished), 1))
      .reduceByKey(_+_)
      .map{case((partition, label, finished), count) => ((partition, finished), label, count)}
      .collect
      .groupBy(_._1)
      .mapValues(a => a.map(x => (x._2, x._3)))
      .mapValues(a => a.sortBy(_._1).mkString(","))

    val nonLabeledPerPartition = rdd
      .values
      .filter(_.label.isEmpty)
      .map(node => ((node.partition, node.finished), 1))
      .reduceByKey(_+_)
      .collect
      .toMap
      .mapValues(_.toString)

    val countPerLabel = (labeledPerPartition.toSeq ++ nonLabeledPerPartition.toSeq)
      .groupBy(_._1)
      .mapValues(_.map(_._2).mkString(","))

    val mapStr = countPerLabel.toString()

    s"{(partition, finished) -> [(label, count), ..., unlabeledCount]}: ${mapStr}"
  }

  //noinspection ConvertExpressionToSAM
//  implicit def toPairFlatMap[T,K,V](f: T => Seq[(K, V)]): PairFlatMapFunction[T, K, V] = {
//    new PairFlatMapFunction[T,K,V] {
//      override def call(t: T): util.Iterator[(K, V)] = {
//        import scala.collection.JavaConverters._
//
//        f(t).iterator.asJava
//      }
//    }
//  }

  def setDFName(df: DataFrame, name: String): Unit = {
    df.createOrReplaceTempView(name)
    df.sparkSession.sqlContext.cacheTable(name)
    df.persist
  }

  def checkSkewness[T](rdd: RDD[T]): Double = {
    val counts = rdd.mapPartitionsWithIndex((x, it) => Seq((x, it.size.toDouble)).iterator).collect

    val maxPartition = counts.map(_._2).max
    val minPartition = counts.map(_._2).min

    maxPartition / minPartition
  }

  def countByPartition[T](rdd: RDD[T]): Array[Int] = {
    rdd.mapPartitions(iter => Iterator(iter.length)).collect()
  }

  //noinspection ConvertExpressionToSAM
  implicit def toConsumer[A](function: A => Unit): Consumer[A] = new Consumer[A]() {
    override def accept(arg: A): Unit = function.apply(arg)
  }

  def roundUp(x: Double): Long = Math.ceil(x).toLong

  def roundDown(x: Double): Long = Math.floor(x).toLong
}
