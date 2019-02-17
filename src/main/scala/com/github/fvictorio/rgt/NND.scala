package com.github.fvictorio.rgt

import com.github.fvictorio.rgt.Helpers.roundUp
import org.apache.log4j.{LogManager, Logger}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

import scala.util.Random

trait AttributesToString {
  override def toString(): String = {
    val VertexAttributes(a1, a2, a3) = this.asInstanceOf[VertexAttributes]
    val a1Str = if (a1 == null) "_" else a1 match {
      case SetAttributeB => "B"
      case SetAttributeR => "R"
      case SetAttributeT => "T"
    }
    val a2Str = if (a2 == null) "_" else a2 match {
      case IncSearchAttributeO => "O"
      case IncSearchAttributeN => "N"
    }
    val a3Str = if (a3 == null) "_" else a3 match {
      case SamplingAttributeS => "S"
      case SamplingAttributeU => "U"
    }

    s"Attrs(${a1Str}, ${a2Str}, ${a3Str})"
  }
}

trait VertexToString {
  override def toString(): String = {
    this match {
      case KeyVertexWithoutFeatures(id) => s"Key(${id})"
      case KeyVertexWithFeatures(id, _, _) => s"Key(${id})"
      case ValueVertexWithoutFeatures(id, similarity, attributes) => s"Value(${id}, ${similarity.formatted("%1.2f")}, ${attributes})"
      case ValueVertexWithFeatures(id, similarity, attributes, _, _) => s"Value(${id}, ${similarity.formatted("%1.2f")}, ${attributes})"
    }
  }
}

sealed trait KeyVertex
case class KeyVertexWithoutFeatures(id: Long) extends KeyVertex with VertexToString
case class KeyVertexWithFeatures(id: Long, features: Vector, partition: Long) extends KeyVertex with VertexToString

sealed trait SetAttribute
case object SetAttributeB extends SetAttribute
case object SetAttributeR extends SetAttribute
case object SetAttributeT extends SetAttribute

sealed trait IncSearchAttribute
case object IncSearchAttributeO extends IncSearchAttribute
case object IncSearchAttributeN extends IncSearchAttribute

sealed trait SamplingAttribute
case object SamplingAttributeS extends SamplingAttribute
case object SamplingAttributeU extends SamplingAttribute

sealed trait ValueVertex
case class VertexAttributes(a1: SetAttribute, a2: IncSearchAttribute, a3: SamplingAttribute) extends AttributesToString
case class ValueVertexWithoutFeatures(id: Long, similarity: Double, attributes: VertexAttributes) extends ValueVertex with VertexToString
case class ValueVertexWithFeatures(id: Long, similarity: Double, attributes: VertexAttributes, features: Vector, partition: Long) extends ValueVertex with VertexToString

object NND {
  val logger: Logger = LogManager.getLogger(this.getClass)

  def similarity(v1: KeyVertexWithFeatures, v2: KeyVertexWithFeatures): Double =
    if (v1.partition == v2.partition)
      1.0 / (1.0 + Vectors.norm(Helpers.subtract(v1.features, v2.features), 2.0))
    else
      Double.NegativeInfinity

  def similarity(v1: ValueVertexWithFeatures, v2: ValueVertexWithFeatures): Double =
    if (v1.partition == v2.partition)
      1.0 / (1.0 + Vectors.norm(Helpers.subtract(v1.features, v2.features), 2.0))
    else
      Double.NegativeInfinity

  def buildGraph(rdd: RDD[(Long, RGTNode)], noNeighbors: Int, maxIterations: Int, earlyTermination: Double, sampleRate: Double, bucketsPerInstance: Int): RDD[(Long, RGTNodeWithNeighbors)] = {
    logger.debug(s"[NND] Building graph (noNeighbors: ${noNeighbors}, maxIterations: ${maxIterations}, earlyTermination: ${earlyTermination}, sampleRate: ${sampleRate})")

    val sc = rdd.sparkContext

    // initialGraph
    val nodes = rdd
      .filter(!_._2.finished)
      .map{case (id, node) => (id, node.features, node.partition)}
      .map {
        case (id, features, partition) => KeyVertexWithFeatures(id, features, partition)
      }

    val countPerPartition = rdd
      .map{case (_, node) => (node.partition, 1)}
      .reduceByKey(_ + _)
      .collect
      .toMap

    val maxNumberOfBuckets = roundUp(countPerPartition.values.max.toDouble * bucketsPerInstance / (2 * noNeighbors))

    val N = nodes.count()

    val randomized = nodes.flatMap(node => {
      val hashCode = node.features.toArray.toSeq.hashCode
      val r = new Random(hashCode)
      val numberOfBuckets = roundUp(countPerPartition(node.partition).toDouble * bucketsPerInstance / (2 * noNeighbors))
      r.shuffle(1 to numberOfBuckets.toInt).take(bucketsPerInstance).map(bucket => (maxNumberOfBuckets * node.partition + bucket, node))
//      (1 to 10).map(_ => (numberOfBuckets * node.partition + Random.nextInt(numberOfBuckets), node))
    })

    val randomWithNeighbors = randomized.groupByKey.flatMap { case (_, nodesIter) => {
      val nodes = nodesIter.toList.distinct
      nodes.map(node => {
        val neighbors: Seq[ValueVertexWithoutFeatures] = (1 to noNeighbors).map(_ => nodes(Random.nextInt(nodes.length))).map(n => ValueVertexWithoutFeatures(n.id, similarity(node, n), VertexAttributes(SetAttributeB, null, null)))
        (node, neighbors.filter(_.id != node.id))
      }).iterator
    }
    }

    var graph: RDD[(KeyVertexWithFeatures, Seq[ValueVertexWithoutFeatures])] = randomWithNeighbors.reduceByKey((n1: Seq[ValueVertexWithoutFeatures], n2: Seq[ValueVertexWithoutFeatures]) => {
      (n1 ++ n2)
        .sortBy(-_.similarity)
        .distinct
        .take(noNeighbors)
    })

    var finished = false
    var i = 0
    while (!finished && i < maxIterations) {
      logger.debug(s"[NND] Iteration: ${i}")
      val updatedAccum = sc.longAccumulator

      // createRevVertices
      val graphWithRevVertices = graph
        .flatMap{case (vAsterisk, bv) => {
          val selfVertex = (KeyVertexWithoutFeatures(vAsterisk.id), ValueVertexWithFeatures(vAsterisk.id, Double.NegativeInfinity, VertexAttributes(null, null, null), vAsterisk.features, vAsterisk.partition))

          val otherVertexes = bv.map(neighbor => {
            val attrs = VertexAttributes(SetAttributeR, neighbor.attributes.a2, null)
            (KeyVertexWithoutFeatures(neighbor.id), ValueVertexWithoutFeatures(vAsterisk.id, neighbor.similarity, attrs))
          })

          selfVertex.asInstanceOf[(KeyVertexWithoutFeatures, ValueVertex)] +: otherVertexes.asInstanceOf[Seq[(KeyVertexWithoutFeatures, ValueVertex)]]
        }}
        .groupByKey
        .map{case (_, valuesIt) => {
          val values = valuesIt.toSeq

          assert(values.count(_.isInstanceOf[ValueVertexWithFeatures]) == 1)

          val v = values.find(_.isInstanceOf[ValueVertexWithFeatures]).get.asInstanceOf[ValueVertexWithFeatures]

          val rv = values.filter(_.isInstanceOf[ValueVertexWithoutFeatures]).map(_.asInstanceOf[ValueVertexWithoutFeatures])

          (v, rv)
        }}

      // createKVertices
      val graphWithKVertices = graphWithRevVertices
        .flatMap{case (v, rv) => {
          val self = (KeyVertexWithoutFeatures(v.id), v.copy(similarity = Double.NegativeInfinity, attributes = VertexAttributes(null, null, null)))

          val sampleSize = Math.max(Math.ceil(sampleRate * rv.size).toInt, 1) // take at least one element

          val rsv = Random.shuffle(rv).take(sampleSize)

          val others = rv.map(u => {
            val sampled = rsv.contains(u)
            val samplingAttribute = if (sampled) SamplingAttributeS else SamplingAttributeU
            (KeyVertexWithoutFeatures(u.id), ValueVertexWithFeatures(v.id, u.similarity, VertexAttributes(SetAttributeB, u.attributes.a2, samplingAttribute), v.features, v.partition))
          })

          self +: others
        }}
        .groupByKey
        .map{case (_, valuesIt) => {
          val values = valuesIt.toSeq

          assert(values.count(_.attributes.a1 != SetAttributeB) == 1)

          val v = values.find(_.attributes.a1 != SetAttributeB).get

          val bv = values.filter(_.attributes.a1 == SetAttributeB)

          (KeyVertexWithFeatures(v.id, v.features, v.partition), bv)
        }}

      // createAdjVertices
      val graphWithAdjVertices = graphWithKVertices
        .flatMap{case(v, bv) => {
          val emit1 = (KeyVertexWithoutFeatures(v.id), ValueVertexWithFeatures(v.id, Double.NegativeInfinity, VertexAttributes(null, null, null), v.features, v.partition))
          val emitOthers1 = bv.map(u => (KeyVertexWithoutFeatures(v.id), u))

          val emitOthers2 = bv
            .filter(_.attributes.a3 == SamplingAttributeS)
            .map(u => {
              val attrs = VertexAttributes(SetAttributeR, u.attributes.a2, null)
              (KeyVertexWithoutFeatures(u.id), ValueVertexWithFeatures(v.id, u.similarity, attrs, v.features, v.partition))
            })

          emit1 +: (emitOthers1 ++ emitOthers2)
        }}
        .groupByKey
        .map{case (_, valuesIt) => {
          val values = valuesIt.toSeq

          assert(values.count(_.attributes == VertexAttributes(null, null, null)) == 1)

          val v = values.find(_.attributes == VertexAttributes(null, null, null)).get

          val bvAndRv = values.filter(_.attributes != VertexAttributes(null, null, null))

          (KeyVertexWithFeatures(v.id, v.features, v.partition), bvAndRv)
        }}

      // updateKVertices
      val newGraph = graphWithAdjVertices
        .flatMap{case(v, asv) => {
          val bv = asv.filter(_.attributes.a1 == SetAttributeB)

          val emit1 = (KeyVertexWithoutFeatures(v.id), ValueVertexWithFeatures(v.id, Double.NegativeInfinity, VertexAttributes(null, null, null), v.features, v.partition).asInstanceOf[ValueVertex])

          val emitBv = bv.map(u => {
            (KeyVertexWithoutFeatures(v.id), ValueVertexWithoutFeatures(u.id, u.similarity, u.attributes).asInstanceOf[ValueVertex])
          })

          val emitAsv = asv.flatMap(u => {
            asv.filter(_.id != u.id).map(p => {
              (KeyVertexWithoutFeatures(u.id), ValueVertexWithoutFeatures(p.id, similarity(u, p), VertexAttributes(SetAttributeT, IncSearchAttributeN, null)).asInstanceOf[ValueVertex])
            })
          })

          emit1 +: (emitBv ++ emitAsv)
        }}
        .groupByKey
        .map{case (_, valuesIt) => {
          val values = valuesIt.toSeq

          assert(values.count(_.isInstanceOf[ValueVertexWithFeatures]) == 1)

          val v = values.find(_.isInstanceOf[ValueVertexWithFeatures]).get.asInstanceOf[ValueVertexWithFeatures]

          val bv = values.filter(_.isInstanceOf[ValueVertexWithoutFeatures])
            .map(_.asInstanceOf[ValueVertexWithoutFeatures])
            .groupBy(_.id)
            .map{case(id, nodes) => {
              if (nodes.length == 1)
                nodes.head
              else
                nodes.reduce((n1, n2) =>
                  if (n1.attributes.a1 == SetAttributeB)
                    n1
                  else
                    n2
                )
            }}
            .toSeq
            .sortBy(-_.similarity)
            .take(noNeighbors)

          val updated = bv.count(_.attributes.a1 != SetAttributeB)

          updatedAccum.add(updated)

          (KeyVertexWithFeatures(v.id, v.features, v.partition), bv.map(u => ValueVertexWithoutFeatures(u.id, u.similarity, VertexAttributes(SetAttributeB, null, null))))
        }}

      newGraph.persist()
      newGraph.count()
      graph.unpersist()

      graph = newGraph

      i += 1

      val percentageUpdated = updatedAccum.value.toDouble / (noNeighbors * N.toDouble)
      logger.debug(s"[NND] Percentage updated: ${percentageUpdated}")
      if (percentageUpdated <= earlyTermination) {
        finished = true
      }
    }

    val graphRDD = graph.map { case (v, bv) => {
      val neighborListSeq = bv.map(neighbor => {
        (neighbor.id, neighbor.similarity)
      })

      (v.id, neighborListSeq)
    }
    }

    val result = rdd
      .leftOuterJoin(graphRDD)
      .mapValues{case (node, neighbors) => {
        RGTNodeWithNeighbors(node.features, node.label, node.partition, node.finished, neighbors.orNull)
      }}

    graph.unpersist()

    result
  }
}
