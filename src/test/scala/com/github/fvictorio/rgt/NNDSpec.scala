package com.github.fvictorio.rgt

import com.github.fvictorio.rgt.experiments.Datasets
import org.apache.spark.SparkConf
import org.apache.spark.sql.{Row, SparkSession}
import org.scalatest._

class NNDSpec extends UnitTest with Matchers {
  "NND" should "compute 95% of the same edges as the iris saved example" in {
    val conf = new SparkConf()
      .setMaster("local[*]")
      .setAppName("EMNIST")
    val spark = SparkSession.builder.config(conf).getOrCreate()
    val sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    sc.setCheckpointDir("/tmp")

    val rdd = Datasets.loadIris(spark)._1
      .mapValues(node => RGTNode(node.features, node.label, 0L, false))

    val graph = NND.buildGraph(rdd, 5, 5, 0.001, 1.0, 2)
      .mapValues(_.neighbors)
      .persist

    val fixture = spark.read.parquet(this.getClass.getResource("/iris-graph").getPath)
      .select("id", "neighbors")
      .rdd
      .map(row => {
        val id = row.getLong(0)
        val neighbors = row.getAs[Seq[Row]](1).map(r => (r.getLong(0), r.getDouble(1)))

        (id, neighbors)
      })

    val compareNeighbors: (Seq[(Long, Double)], Seq[(Long, Double)]) => Double = (neighborsRows: Seq[(Long, Double)], correctNeighborsRows: Seq[(Long, Double)]) => {
      val neighbors = neighborsRows.map(_._1).toSet
      val correctNeighbors = correctNeighborsRows.map(_._1).toSet

      neighbors.count(neighbor => correctNeighbors.contains(neighbor)) / neighbors.size.toDouble
    }

    val (accSum, count) = graph.join(fixture)
      .values
      .map{case (n1, n2) => (compareNeighbors(n1, n2), 1)}
      .reduce((a, b) => (a._1 + b._1, a._2 + b._2))

    val acc = accSum / count

    println(acc)
    acc should be > 0.95
  }

  it should "compute 95% of the same edges as the emnist saved example" in {
    val conf = new SparkConf()
      .setMaster("local[*]")
      .setAppName("EMNIST")
    val spark = SparkSession.builder.config(conf).getOrCreate()
    val sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    sc.setCheckpointDir("/tmp")

    val rdd = Datasets.loadEmnist(spark, this.getClass.getResource("/emnist-2k.csv").getPath)._1
      .mapValues(node => RGTNode(node.features, node.label, 0L, false))
      .repartition(4)

    val graph = NND.buildGraph(rdd, 5, 5, 0.001, 1.0, 2)
      .mapValues(_.neighbors)
      .persist

    val fixture = spark.read.parquet(this.getClass.getResource("/emnist-graph").getPath)
      .select("id", "neighbors")
      .rdd
      .map(row => {
        val id = row.getLong(0)
        val neighbors = row.getAs[Seq[Row]](1).map(r => (r.getLong(0), r.getDouble(1)))

        (id, neighbors)
      })

    val compareNeighbors: (Seq[(Long, Double)], Seq[(Long, Double)]) => Double = (neighborsRows: Seq[(Long, Double)], correctNeighborsRows: Seq[(Long, Double)]) => {
      val neighbors = neighborsRows.map(_._1).toSet
      val correctNeighbors = correctNeighborsRows.map(_._1).toSet

      neighbors.count(neighbor => correctNeighbors.contains(neighbor)) / neighbors.size.toDouble
    }

    val (accSum, count) = graph.join(fixture)
      .values
      .map{case (n1, n2) => (compareNeighbors(n1, n2), 1)}
      .reduce((a, b) => (a._1 + b._1, a._2 + b._2))

    val acc = accSum / count

    println(acc)
    acc should be > 0.95
  }
}
