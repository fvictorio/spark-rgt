package com.github.fvictorio.rgt

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.log4j.{LogManager, Logger}
import org.apache.spark.rdd.RDD

import scala.util.Random

private case class ReliefFInstance(features: Vector, label: Long)

object ReliefF {
  val logger: Logger = LogManager.getLogger(this.getClass)

  def distance(v1: Vector, v2: Vector, min: Array[Double], max: Array[Double]): Double = {
    Helpers.subtract(v1, v2).toArray.toSeq.zipWithIndex.map{case(x,i) => Math.abs(x) / (max(i) - min(i)) }.sum
  }

  private def computeNearestHits(instance: ReliefFInstance, k: Long, instances: Seq[ReliefFInstance], min: Array[Double], max: Array[Double]): Seq[ReliefFInstance] = {
    val result = instances
      .filter(_.label == instance.label)
      .filter(_ ne instance)
      .sortBy(i => distance(i.features, instance.features, min, max))
      .take(k.toInt)

    result
  }

  private def computeNearestMisses(instance: ReliefFInstance, k: Long, instances: Seq[ReliefFInstance], label: Long, min: Array[Double], max: Array[Double]): Seq[ReliefFInstance] = {
    val result = instances
      .filter(_.label == label)
      .sortBy(i => distance(i.features, instance.features, min, max))
      .take(k.toInt)

    result
  }

  def diff(i: Int, v1: Vector, v2: Vector, minPerAttribute: Array[Double], maxPerAttribute: Array[Double]): Double =  {
    val minI = minPerAttribute(i)
    val maxI = maxPerAttribute(i)
    Math.abs(v1(i) - v2(i)) / (maxI - minI)
  }

  def priorProb(instances: Seq[ReliefFInstance], label: Long): Double = {
    instances.count(_.label == label).toDouble / instances.length
  }

  def repeat[T](seq: Seq[T]): Stream[T] = {
    Stream.continually(seq.toStream).flatten
  }

  private def getWeights(instances: Seq[ReliefFInstance], m: Long, k: Long, r: Random, bias: Double): Vector = {
    val featuresLength = instances.head.features.size

    val labelsMap = instances.groupBy(_.label).mapValues(_.map(_.features))
    val labels = labelsMap.keys.toSeq

    assert(labels.toSet == instances.map(_.label).distinct.toSet)

    val randomInstances = r.shuffle(instances).take(m.toInt)

    val weights = Array.fill(featuresLength)(0.0)

    val priorProbs = labels.map(label => (label, priorProb(instances, label))).toMap

    val minPerAttribute = instances
      .map(_.features.toArray)
      .reduce((x, y) => {
        x.zip(y).map(pair => Math.min(pair._1, pair._2))
      })
    val maxPerAttribute = instances
      .map(_.features.toArray)
      .reduce((x, y) => {
        x.zip(y).map(pair => Math.max(pair._1, pair._2))
      })

    for (instance <- randomInstances) {
      val nearestHits = computeNearestHits(instance, k, instances, minPerAttribute, maxPerAttribute)
      val noHits = nearestHits.length
      val nearestMisses = labels.filter(_ != instance.label).map(label => (label, computeNearestMisses(instance, k, instances, label, minPerAttribute, maxPerAttribute))).toMap
      val noMissesAvg = nearestMisses.values.map(_.length).sum / nearestMisses.values.toList.length

      (0 until featuresLength).foreach(a => {
        val sumHits = if (noHits > 0)
          nearestHits.map(hit => diff(a, instance.features, hit.features, minPerAttribute, maxPerAttribute)).sum / (m * noHits)
        else
          0.0

        val sumMisses = if (noMissesAvg > 0)
          labels
            .filter(_ != instance.label)
            .map(label => {
              (priorProbs(label) / (1 - priorProbs(instance.label))) * nearestMisses(label).map(miss => diff(a, instance.features, miss.features, minPerAttribute, maxPerAttribute)).sum
            }).sum / (m * noMissesAvg)
        else
          0.0

        weights(a) = weights(a) - sumHits + sumMisses
      })
    }

    val trimmedWeights = weights.map(x => if (x.isNaN) 0.0 else x)


//    println(trimmedWeights.mkString(","))
    val maxWeight = trimmedWeights.max

    val scaledWeights = trimmedWeights.map(_ / maxWeight)
//    println(scaledWeights.mkString(","))

    val unbiasedWeights = if (scaledWeights.exists(_ > -bias)) {
      scaledWeights.map(w => (w + bias) / (1 + bias)).map(w => if (w < 0) 0.0 else w)
    } else {
      scaledWeights
    }


    Vectors.dense(unbiasedWeights)

  }

  def getWeights(rdd: RDD[(Long, RGTNode)], m: Long, k: Long, r: Random = new Random(), bias: Double = 0.1): Map[Long, Vector] = {
    logger.info(s"[ReliefF] Create partition map")
    val labeledInstances: Seq[(Long, Vector, Long)] = rdd
      .filter(_._2.label.nonEmpty)
      .collect
      .map(node => (node._2.partition, node._2.features, node._2.label.get))

    val partitionMap = labeledInstances.groupBy(_._1)
        .map {
          case (key, value) => (key, value.map(x => ReliefFInstance(x._2, x._3)))
        }


    val result = partitionMap
      .map {
        case (partition, instances) => {
          logger.debug(s"[ReliefF] Compute weights for partition $partition")
          (partition, getWeights(instances, Math.min(m, instances.length), k, r, bias))
        }
      }

    result
  }
}
