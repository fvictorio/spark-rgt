package com.github.fvictorio.rgt

import com.github.fvictorio.nnd
import org.apache.spark.rdd.RDD

object NND {
  def buildGraph(rdd: RDD[(Long, RGTNode)], noNeighbors: Int, maxIterations: Int, earlyTermination: Double, sampleRate: Double, bucketsPerInstance: Int): RDD[(Long, RGTNodeWithNeighbors)] = {
    val input = rdd.map{case (id, node) => (id, nnd.Node(node.features, node.label, node.partition, node.finished))}
    val output = nnd.NND.buildGraph(input, noNeighbors, maxIterations, earlyTermination, sampleRate, bucketsPerInstance)
    output.map{case (id, node) => (id, RGTNodeWithNeighbors(node.features, node.label, node.partition, node.finished, node.neighbors))}
  }
}
