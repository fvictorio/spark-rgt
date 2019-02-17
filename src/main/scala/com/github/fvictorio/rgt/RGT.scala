package com.github.fvictorio.rgt

import org.apache.spark.ml.linalg.Vector
import Helpers._
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import org.apache.spark.util.LongAccumulator
import org.apache.log4j.{LogManager, Logger}
import org.apache.spark.HashPartitioner

import scala.collection.mutable
import scala.io.StdIn

case class RGTInputNode(features: Vector, label: Option[Long])

case class RGTOutputNode(features: Vector, label: Option[Long], prediction: Long)

case class RGTNode(features: Vector, label: Option[Long], partition: Long, finished: Boolean)

case class RGTNodeWithNeighbors(features: Vector, label: Option[Long], partition: Long, finished: Boolean, neighbors: Seq[(Long, Double)])

case class RGTNodeDivisionStep(label: Option[Long], partition: Long, neighbors: Seq[(Long, Double)], M: Double = 0.0, D: Double = 0.0, L: Double = 0.0, grad: Double = 0.0, newM: Double = 0.0, divisionFinished: Boolean = false)

class RGT(
           val giniImpurity: Double = 0.1,
           val noNeighbors: Int = 5,
           val maxIter: Long = 20,
           val divisionStepMaxIter: Long = 1000,
           val divisionStepMaxIterWithoutChanges: Long = 10,
           val weightFeatures: Boolean = true,
           val relieffM: Long = Long.MaxValue,
           val relieffK: Long = 3L,
           val nndMaxIterations: Int = 5,
           val rddPartitions: Int = 4,
           val graphConstructionMaxIter: Int = 5,
           val graphConstructionEarlyTermination: Double = 0.001,
           val graphDivisionEarlyTermination: Double = 0.005,
           val graphConstructionBucketsPerInstance: Int = 2,
           val graphConstructionSampleRate: Double = 1.0
         ) {

  val logger: Logger = LogManager.getLogger(this.getClass)
  var i = 0L

  def transform(input: RDD[(Long, RGTInputNode)]): (RDD[(Long, RGTOutputNode)], Long) = {
    logger.info(s"[RGT] === Begin ===")
    assert({
      logger.info("[RGT] Assertions are enabled")
      true
    })

    val labelsCount = input
      .filter{case (_, node) => node.label.nonEmpty}
      .map{case (_, node) => node.label.get}
      .countByValue
      .toMap
    val labeledCount = labelsCount.values.sum
    val numberOfClasses = labelsCount.keys.size

    val labelsWeights = labelsCount.mapValues(x => labeledCount / (numberOfClasses * x.toDouble)).map(identity)

    logger.info(s"[RGT] Labels count: ${labelsCount.toList.sortBy(_._1).mkString(", ")}")
    logger.info(s"[RGT] Labels weights: ${labelsWeights.toList.sortBy(_._1).mkString(", ")}")

    // Initialize the dataframe with dummy columns
    var result = input.mapValues(inputNode => RGTNode(inputNode.features, inputNode.label, 0L, false))
      .setName("resultWithPartitionAndFinished")

    result = RGT.computeFinished(result, labelsWeights, this.giniImpurity, this.noNeighbors)
      .setName("resultWithComputedFinished")

    val initialCount = result.count

    // Main loop
    i = 0L
    while (!RGT.isFinished(result) && i < maxIter) {
      logger.info(s"[RGT] Iteration $i")

      // log finished partitions
      logger.debug({
        val finishedMap = result
          .values
          .map(node => (node.partition, node.finished))
          .distinct
          .collect
          .sortBy(_._1)
          .mkString(",")
        s"[RGT] Finished partitions: $finishedMap"
      })

      // log labeled instances
      logger.debug({
        val labelCount = Helpers.dumpLabelCount(result)
        s"[RGT] Label count: ${labelCount}"
      })

      // count per partition
      logger.debug({
        val countPerPartition = result
          .map{case (_, node) => (node.partition, 1)}
          .reduceByKey(_ + _)
          .collect
        s"[RGT] Count per partition: ${countPerPartition.sortBy(_._1).mkString(", ")}"
      })

      // weight features
      logger.info(s"[RGT] BEGIN Weighting features (partitions size: ${result.partitions.length})")
      logger.trace(s"[RGT] Partitions: ${countByPartition(result).mkString(", ")}")
      val datasetWithWeightedFeatures = weightFeatures(result)
        .setName(s"datasetWithWeightedFeatures-${i}")
      datasetWithWeightedFeatures.checkpoint()
      assert({
        datasetWithWeightedFeatures.count == initialCount
      }, "Count of dataframe changed")
      datasetWithWeightedFeatures.count
      logger.info(s"[RGT] END Weighting features")

      // build graph
      logger.info(s"[RGT] BEGIN Graph construction (partitions size: ${datasetWithWeightedFeatures.partitions.length})")
      logger.debug(s"[RGT] Partitions: ${countByPartition(datasetWithWeightedFeatures).mkString(", ")}")
      val datasetWithNeighbors = NND.buildGraph(datasetWithWeightedFeatures, this.noNeighbors,
        this.graphConstructionMaxIter,
        this.graphConstructionEarlyTermination,
        this.graphConstructionSampleRate,
        this.graphConstructionBucketsPerInstance
      )
        .repartition(rddPartitions)
        .setName(s"datasetWithNeighbors-${i}")
      datasetWithNeighbors.checkpoint()
      assert(datasetWithNeighbors.count == initialCount, "Count of dataframe changed")
      datasetWithWeightedFeatures.unpersist()
      logger.info(s"[RGT] END Graph construction")

      // divide graph
      logger.info(s"[RGT] BEGIN Graph division (partitions size: ${datasetWithNeighbors.partitions.length})")
      logger.debug(s"[RGT] Partitions: ${countByPartition(datasetWithNeighbors).mkString(", ")}")
      val datasetWithPartitions = divideGraph(datasetWithNeighbors, initialCount)
        .setName(s"datasetWithPartitions-${i}")
      datasetWithPartitions.checkpoint()
      assert(datasetWithPartitions.count == initialCount, "Count of dataframe changed")
      datasetWithNeighbors.unpersist()
      logger.info(s"[RGT] END Graph division")

      result = datasetWithPartitions.mapValues(RGT.dropNeighbors)
        .setName(s"datasetWithPartitionsDropNeighbors-${i}")

      // log labeled instances
      logger.debug({
        val labelCount = Helpers.dumpLabelCount(result)
        s"[RGT] Label count: ${labelCount}"
      })

      result = RGT.computeFinished(result, labelsWeights, this.giniImpurity, this.noNeighbors)
        .partitionBy(new HashPartitioner(rddPartitions))
        .setName(s"resultComputeFinished-${i}")
      i += 1

      assert(result.count == initialCount, "Count of dataframe changed")

      datasetWithPartitions.unpersist()
    }

    logger.info({
      val maxIterationsReached = i == maxIter
      val message = if (maxIterationsReached) "Max number of iterations reached" else "All partitions are finished"
      s"[RGT] Finished: $message"
    })

    // Assert that it finished because it reached the maximum number of iterations
    // or because all rows are finished
    assert(i == maxIter || (result.filter(!_._2.finished).count == 0))

    // Assign predicted labels in each finished partition
    val output = RGT.assignLabels(result)
      .setName("output")

    logger.info("[RGT] === End ===")

    // Return resulting dataframe
    (output, i)
  }

  def weightFeatures(rdd: RDD[(Long, RGTNode)]): RDD[(Long, RGTNode)] = {
    var result: RDD[(Long, RGTNode)] = null
    if (weightFeatures) {
      val weightsPerPartition = ReliefF.getWeights(rdd.filter(!_._2.finished), relieffM, relieffK)

      logger.info(s"[ReliefF] BEGIN Apply weights")
      val broadcastWeights = rdd.sparkContext.broadcast(weightsPerPartition)

      result = rdd.mapValues(node => {
        val weights = broadcastWeights.value.getOrElse(node.partition, ones(node.features.size))
        val newFeatures = multiply(node.features, weights)

        node.copy(features = newFeatures)
      })
      logger.info(s"[ReliefF] END Apply weights")
    } else {
      result = rdd
    }

    result
  }

  def divideGraph(_rdd: RDD[(Long, RGTNodeWithNeighbors)], initialCount: Long): RDD[(Long, RGTNodeWithNeighbors)] = {
    val sc = _rdd.sparkContext
    val rdd = _rdd.filter(!_._2.finished)

    // Convert dataframe to graph
    val vertices: RDD[(Long, RGT.DivisionStepNode)] = rdd
      .map { case (id, node) => (id, node.label, node.partition) }
      .map { case (id, label, partition) => {
        (id, RGT.DivisionStepNode(label, partition))
      }
      }
      .setName(s"vertices-${i}")

    val edges = rdd.map { case (id, node) => (id, node.neighbors) }
      .flatMap { case (id, neighbors) => {
        neighbors.flatMap {
          case (neighborId, similarity) => Seq(Edge(id, neighborId, similarity), Edge(neighborId, id, similarity))
        }
      }
      }
      .setName(s"edges-${i}")

    val graph = Graph(vertices, edges)

    // Divide graph
    val partitionAndLabelToM = computePartitionAndLabelToM(rdd)

    logger.debug(s"partitionAndLabelToM: ${partitionAndLabelToM}")

    val newGraph = RGT.divideGraph(graph, partitionAndLabelToM, divisionStepMaxIter, divisionStepMaxIterWithoutChanges, graphDivisionEarlyTermination, noNeighbors, initialCount)
    graph.unpersist()

    newGraph.vertices.setName(s"newGraph-vertices-${i}")
    newGraph.edges.setName(s"newGraph-edges-${i}")

    // Update partitions
    val currentMaxPartition = _rdd.map(_._2.partition).max
    val broadcastCurrentMaxPartition = sc.broadcast(currentMaxPartition)
    val rddWithNewPartitions = newGraph.vertices.map {
      case (id, RGT.DivisionStepNode(_, partition, _, m, _, _, _, _)) => {
        val newPartition = if (m < 0)
          partition
        else
          partition + broadcastCurrentMaxPartition.value + 1

        (id.toLong, newPartition)
      }
    }

    newGraph.unpersist()

    val result = _rdd
      .leftOuterJoin(rddWithNewPartitions)
      .mapValues { case (node, newPartition) => node.copy(partition = newPartition.getOrElse(node.partition)) }

    result
  }
}

object RGT {
  val logger: Logger = LogManager.getLogger(this.getClass)

  def addNeighbors(node: RGTNode, neighbors: Seq[(Long, Double)]): RGTNodeWithNeighbors =
    RGTNodeWithNeighbors(node.features, node.label, node.partition, node.finished, neighbors)

  def dropNeighbors(node: RGTNodeWithNeighbors): RGTNode =
    RGTNode(node.features, node.label, node.partition, node.finished)

  def addPrediction(node: RGTNode, prediction: Long): RGTOutputNode =
    RGTOutputNode(node.features, node.label, prediction)

  def isFinished(rdd: RDD[(Long, RGTNode)]): Boolean = {
    rdd.filter(!_._2.finished).count == 0
  }

  def computeFinished(rdd: RDD[(Long, RGTNode)], labelsWeights: Map[Long, Double], maxGiniImpurity: Double, noNeighbors: Int): RDD[(Long, RGTNode)] = {
    val nullPerPartition = rdd
      .filter { case (_, node) => !node.finished && node.label.isEmpty }
      .groupBy(_._2.partition)
      .mapValues(_.size)
      .collect
      .toMap

    val countPerPartition = rdd
      .filter(node => !node._2.finished && node._2.label.nonEmpty)
      .groupBy(_._2.partition)
      .mapValues(_.size)
      .collect
      .toMap


    val weightedCountPerLabel = rdd
      .filter(node => !node._2.finished && node._2.label.nonEmpty)
      .groupBy(node => (node._2.partition, node._2.label.get))
      .mapValues(_.size.toDouble)
      .map{case ((partition, label), count) => ((partition, label), count * labelsWeights(label))}
      .collect
      .toMap

    val weightedCountPerPartition = weightedCountPerLabel
      .toList
      .map{case ((partition, _), count) => (partition, count)}
      .groupBy(_._1)
      .mapValues(_.map(_._2).sum)

    val giniPerPartition = weightedCountPerLabel
      .transform { case ((partition, _), count) =>
        math.pow(count / weightedCountPerPartition(partition), 2)
      }
      .toList
      .map { case ((partition, label), ratioSquared) => {
        (partition, label, ratioSquared)
      }
      }
      .groupBy(_._1)
      .mapValues(x => x.map(_._3).sum)

    val isFinished = giniPerPartition
      .transform { case (partition, gini) => {
        val nullCount = nullPerPartition.getOrElse(partition, 0)
        val partitionCount = countPerPartition.getOrElse(partition, 0)

        if (nullCount == 0)
          true
        else if ((partitionCount + nullCount) < 2 * noNeighbors)
          true
        else
          (1 - gini) < maxGiniImpurity
      }
      }

    rdd
      .mapValues(node => {
        val newFinished = if (node.finished) node.finished else isFinished(node.partition)
        node.copy(finished = newFinished)
      })
  }

  def assignLabels(rdd: RDD[(Long, RGTNode)]): RDD[(Long, RGTOutputNode)] = {
    def mostRepeating(xs: Array[Long]): Long = {
      xs
        .map((_, 1))
        .groupBy(_._1)
        .mapValues(_.map(_._2).sum)
        .toSeq
        .sortWith((a, b) => {
          if (a._2 != b._2)
            a._2 > b._2
          else
            a._1 <= b._1
        })
        .head._1
    }

    val labelPerPartition = rdd
      .filter(_._2.label.nonEmpty)
      .map(node => (node._2.partition, node._2.label.get))
      .collect
      .groupBy(_._1)
      .mapValues(_.map(_._2))
      .mapValues(mostRepeating)
      .map(identity) // needed to make the map serializable, since mapValues is lazy (see https://stackoverflow.com/questions/32900862/map-can-not-be-serializable-in-scala)

    rdd.mapValues(node => RGT.addPrediction(node, labelPerPartition(node.partition)))
  }

  case class DivisionStepNode(
                               label: Option[Long],
                               partition: Long,
                               finished: Boolean = false,
                               M: Double = 0.0,
                               D: Double = 0.0,
                               L: Double = 0.0,
                               grad: Double = 0.0,
                               newM: Double = 0.0
                             )

  def divideGraph(initialGraph: Graph[DivisionStepNode, Double], partitionAndLabelToM: Map[(Long, Long), Double], maxIter: Long, maxIterWithoutChanges: Long, graphDivisionEarlyTermination: Double, noNeighbors: Int, initialCount: Long): Graph[DivisionStepNode, Double] = {
    val sc = initialGraph.vertices.sparkContext

    logger.debug(s"[Graph division] Dividing graph (maxIter: ${maxIter}, graphDivisionEarlyTermination: ${graphDivisionEarlyTermination})")

    // Compute initial M
    val graphWithM = initialGraph.mapVertices((_, node) => {
      val partition = node.partition
      val label = node.label

      val M = if (label.nonEmpty) {
        partitionAndLabelToM((partition, label.get))
      } else {
        0.0
      }

      node.copy(M = M)
    })

    // Compute D[i]
    val verticesWithD = graphWithM.aggregateMessages[Double](
      triplet => {
        triplet.sendToDst(triplet.attr)
      },
      (a, b) => a + b
    )
    var graph = graphWithM.joinVertices(verticesWithD)(
      (_, node, D) => node.copy(D = D)
    )
    val countPerPartition = graphWithM.vertices.map { case (_, node) => (node.partition, 1) }.reduceByKey(_+_).collect.toMap

    val partitionsIds = graphWithM.vertices.map { case (_, node) => node.partition }.distinct.collect
    val iterationsWithoutSignChanges = mutable.Map(partitionsIds.map(partition => (partition, 0L)): _*)
    var accumulatorPerPartition: Map[Long, LongAccumulator] = partitionsIds.map(p => (p, sc.longAccumulator)).toMap

    var iter = 0L
    while (iterationsWithoutSignChanges.values.exists(_ < maxIterWithoutChanges) && iter < maxIter) {
      logger.info(s"[Graph division] Iteration: $iter")
      logger.info(s"Number of iterations without sign changes:")
      logger.debug(printMap(iterationsWithoutSignChanges.map{case (partition, iterations) => (partition, (iterations, accumulatorPerPartition(partition).value))}))
      iter += 1
      accumulatorPerPartition = partitionsIds.map(p => (p, sc.longAccumulator)).toMap
      var newGraph = computeNextM(graph, accumulatorPerPartition)
      graph.unpersist()
      graph = newGraph
      graph.checkpoint()
      graph.vertices.count()


      // Count number of nodes whose M changed its sign
      accumulatorPerPartition.foreach { case (partition, accumulator) => {
        val partitionCount = countPerPartition(partition)
        val ratioChanged = accumulator.value.toDouble / partitionCount
//        val minChanged = Math.min(Math.floor(countPerPartition(partition) * graphDivisionEarlyTermination), noNeighbors)
//        if (accumulator.value > roundDown(graphDivisionEarlyTermination * partitionCount / initialCount)) {
        if (ratioChanged > graphDivisionEarlyTermination) {
          iterationsWithoutSignChanges(partition) = 0
        } else {
          iterationsWithoutSignChanges(partition) += 1
        }
      }
      }

      newGraph = graph.mapVertices { case (_, node) => node.copy(M = node.newM, finished = iterationsWithoutSignChanges(node.partition) >= maxIterWithoutChanges) }
      graph.unpersist()
      graph = newGraph
    }

    graphWithM.unpersist()
    verticesWithD.unpersist()

    graph
  }

  def computeNextM(graph: Graph[DivisionStepNode, Double], accumulatorPerPartition: Map[Long, LongAccumulator]): Graph[DivisionStepNode, Double] = {
    // Get spark context
    val sc = graph.vertices.sparkContext

    // Compute L[i]
    val verticesWithL = graph.aggregateMessages[Double](
      triplet => {
        if (!triplet.srcAttr.finished) {
          val Mk = triplet.srcAttr.M
          val Di = triplet.dstAttr.D
          val Dk = triplet.srcAttr.D
          val Aik = triplet.attr

          triplet.sendToDst(Mk * Aik / Math.sqrt(Di * Dk))
        }
      },
      (a, b) => a + b
    )
    val graphWithMAndDAndL = graph.joinVertices(verticesWithL)(
      (_, node, Lsubexpr) => {
        val M = node.M
        val L = M - Lsubexpr
        node.copy(L = L)
      }
    )
    verticesWithL.unpersist()

    // Compute M2 and L2
    val M2L2: Map[Long, (Double, Double)] = graph.vertices
      .map { case (_, node) => (node.partition, (Math.pow(node.M, 2), node.M * node.L)) }
      .reduceByKey { case (a, b) => (a._1 + b._1, a._2 + b._2) }
      .map { case (partition, (m2, l2)) => (partition, (1.0 / m2, l2)) }
      .collect
      .toMap

    val broadcastM2L2 = sc.broadcast(M2L2)

    // Compute gradient
    val resultWithoutNormalization = graphWithMAndDAndL.mapVertices((_, node) => {
      val M = node.M
      val L = node.L
      val partition = node.partition

      val (m2, l2) = broadcastM2L2.value(partition)

      val grad = 2 * (m2 * L - Math.pow(m2, 2) * l2 * M)

      val newM = M - grad
      node.copy(grad = grad, newM = newM)
    })
    graphWithMAndDAndL.unpersist()

    // Compute M averages
    val mAverages = resultWithoutNormalization
      .vertices
      .map { case (_, node) => (node.partition, (node.label, node.newM)) }
      .aggregateByKey((0, 0.0, 0, 0.0))((acc, x) => {
        if (x._1.isEmpty)
          (acc._1 + 1, acc._2 + x._2, acc._3, acc._4)
        else
          (acc._1, acc._2, acc._3 + 1, acc._4 + x._2)
      }, (acc1, acc2) => {
        (acc1._1 + acc2._1, acc1._2 + acc2._2, acc1._3 + acc2._3, acc1._4 + acc2._4)
      })
      .collect
      .toMap
      .transform { case (_, (nonLabeledCount, nonLabeledSum, labeledCount, labeledSum)) =>
        (nonLabeledSum / nonLabeledCount, labeledSum / labeledCount)
      }

    val nonLabeledMAverage = mAverages.transform((_, value) => value._1)
    val labeledMAverage = mAverages.transform((_, value) => value._2)

    val broadcastAverageMLabeled = sc.broadcast(labeledMAverage)
    val broadcastAverageMNonLabeled = sc.broadcast(nonLabeledMAverage)

    val result = resultWithoutNormalization.mapVertices((_, node) => {
      val newNode = if (node.finished) {
        node.copy(newM = node.M)
      } else if (node.label.isEmpty) {
        node.copy(newM = node.newM - broadcastAverageMNonLabeled.value(node.partition))
      } else {
        node.copy(newM = node.newM - broadcastAverageMLabeled.value(node.partition))
      }

      if (newNode.M.signum != newNode.newM.signum) {
        accumulatorPerPartition(newNode.partition).add(1)
      }

      newNode
    })

    resultWithoutNormalization.unpersist()

    result
  }

  def prompt(): Unit = {
    printf(">>> ")
    StdIn.readChar()
  }
}
