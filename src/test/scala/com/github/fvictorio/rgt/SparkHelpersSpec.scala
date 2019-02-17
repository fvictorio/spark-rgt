package com.github.fvictorio.rgt

import org.apache.spark.ml.linalg.Vectors
import org.scalatest._

class SparkHelpersSpec extends SparkUnitTest with Matchers {
  "partitionAndLabelToM" should "give alternate M values" in {
    val rdd = sc.parallelize(Seq(
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(0L), 0L, false, Seq())),
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(1L), 0L, false, Seq())),
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(2L), 0L, false, Seq()))
    ))

    val partitionAndLabelToM = Helpers.computePartitionAndLabelToM(rdd)
    assert(partitionAndLabelToM.size == 3)
    assert(partitionAndLabelToM((0L, 0L)) == 1.0)
    assert(partitionAndLabelToM((0L, 1L)) == -1.0)
    assert(partitionAndLabelToM((0L, 2L)) == 1.0)
  }

  it should "give the same result if it's not sorted" in {
    val rdd = sc.parallelize(Seq(
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(1L), 0L, false, Seq())),
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(0L), 0L, false, Seq())),
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(2L), 0L, false, Seq()))
    ))

    val partitionAndLabelToM = Helpers.computePartitionAndLabelToM(rdd)
    assert(partitionAndLabelToM.size == 3)
    assert(partitionAndLabelToM((0L, 0L)) == 1.0)
    assert(partitionAndLabelToM((0L, 1L)) == -1.0)
    assert(partitionAndLabelToM((0L, 2L)) == 1.0)
  }

  it should "give the same result if there are repeated labels" in {
    val rdd = sc.parallelize(Seq(
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(1L), 0L, false, Seq())),
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(1L), 0L, false, Seq())),
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(2L), 0L, false, Seq())),
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(1L), 0L, false, Seq())),
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(2L), 0L, false, Seq())),
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(2L), 0L, false, Seq())),
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(0L), 0L, false, Seq())),
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(1L), 0L, false, Seq())),
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(0L), 0L, false, Seq())),
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(1L), 0L, false, Seq())),
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(2L), 0L, false, Seq())),
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(2L), 0L, false, Seq())),
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(1L), 0L, false, Seq())),
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(0L), 0L, false, Seq())),
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(1L), 0L, false, Seq())),
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(0L), 0L, false, Seq())),
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(0L), 0L, false, Seq())),
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(0L), 0L, false, Seq())),
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(1L), 0L, false, Seq())),
        (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(1L), 0L, false, Seq()))
    ))

    val partitionAndLabelToM = Helpers.computePartitionAndLabelToM(rdd)
    assert(partitionAndLabelToM.size == 3)
    assert(partitionAndLabelToM((0L, 0L)) == 1.0)
    assert(partitionAndLabelToM((0L, 1L)) == -1.0)
    assert(partitionAndLabelToM((0L, 2L)) == 1.0)
  }

  it should "work with repeated labels" in {
    val rdd = sc.parallelize(Seq(
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(0L), 0L, false, Seq())),
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(1L), 0L, false, Seq())),
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(1L), 0L, false, Seq())),
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(2L), 0L, false, Seq())),
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(3L), 0L, false, Seq())),
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(2L), 0L, false, Seq()))
    ))

    val partitionAndLabelToM = Helpers.computePartitionAndLabelToM(rdd)
    assert(partitionAndLabelToM.size == 4)
    assert(partitionAndLabelToM((0L, 0L)) == 1.0)
    assert(partitionAndLabelToM((0L, 1L)) == -1.0)
    assert(partitionAndLabelToM((0L, 2L)) == 1.0)
    assert(partitionAndLabelToM((0L, 3L)) == -1.0)
  }

  it should "ignore finished partitions" in {
    val rdd = sc.parallelize(Seq(
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(0L), 0L, false, Seq())),
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(1L), 0L, true, Seq())),
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(2L), 0L, false, Seq())),
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(3L), 0L, false, Seq())),
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(4L), 0L, true, Seq()))
    ))

    val partitionAndLabelToM = Helpers.computePartitionAndLabelToM(rdd)
    assert(partitionAndLabelToM.size == 3)
    assert(partitionAndLabelToM((0L, 0L)) == 1.0)
    assert(partitionAndLabelToM((0L, 2L)) == -1.0)
    assert(partitionAndLabelToM((0L, 3L)) == 1.0)
  }

  it should "compute M values independently in each partition" in {
    val rdd = sc.parallelize(Seq(
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(0L), 0L, false, Seq())),
      (1L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(1L), 1L, false, Seq())),
      (1L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(2L), 1L, false, Seq())),
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(3L), 0L, false, Seq())),
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(4L), 0L, false, Seq())),
      (1L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(5L), 1L, false, Seq()))
    ))

    val partitionAndLabelToM = Helpers.computePartitionAndLabelToM(rdd)
    assert(partitionAndLabelToM.size == 6)

    println(partitionAndLabelToM.toList.mkString(","))

    assert(partitionAndLabelToM((0L, 0L)) == 1.0)
    assert(partitionAndLabelToM((0L, 3L)) == -1.0)
    assert(partitionAndLabelToM((0L, 4L)) == 1.0)

    assert(partitionAndLabelToM((1L, 1L)) == 1.0)
    assert(partitionAndLabelToM((1L, 2L)) == -1.0)
    assert(partitionAndLabelToM((1L, 5L)) == 1.0)
  }

  it should "compute M values if the same label is in different partitions" in {
    val rdd = sc.parallelize(Seq(
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(0L), 0L, false, Seq())),
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(1L), 0L, false, Seq())),
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(3L), 0L, false, Seq())),
      (0L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(4L), 0L, false, Seq())),
      (1L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(1L), 1L, false, Seq())),
      (1L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(2L), 1L, false, Seq())),
      (1L, RGTNodeWithNeighbors(Vectors.dense(0.0), Some(5L), 1L, false, Seq()))
    ))

    val partitionAndLabelToM = Helpers.computePartitionAndLabelToM(rdd)
    assert(partitionAndLabelToM.size == 7)

    assert(partitionAndLabelToM((0L, 0L)) == 1.0)
    assert(partitionAndLabelToM((0L, 1L)) == -1.0)
    assert(partitionAndLabelToM((0L, 3L)) == 1.0)
    assert(partitionAndLabelToM((0L, 4L)) == -1.0)

    assert(partitionAndLabelToM((1L, 1L)) == 1.0)
    assert(partitionAndLabelToM((1L, 2L)) == -1.0)
    assert(partitionAndLabelToM((1L, 5L)) == 1.0)
  }
}
