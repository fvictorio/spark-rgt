package com.github.fvictorio.rgt

import org.scalatest._
import org.scalatest.prop.TableDrivenPropertyChecks
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.ml.linalg.Vectors


class RGTComputeFinishedSpec extends SparkUnitTest with Matchers with TableDrivenPropertyChecks {
  case class Instance(id: Long, label: Option[Long], partition: Long, finished: Boolean)

  val examples = Table(
    ("input", "expected", "maxGiniImpurity", "noNeighbors"),

    // Single partition, one labeled instance
    (
      Seq(
        Instance(id=0L, label=Some(0), partition=0, finished=false),
        Instance(id=1L, label=None,    partition=0, finished=false),
        Instance(id=2L, label=None,    partition=0, finished=false)
      ),
      Seq(
        Instance(id=0L, label=Some(0L), partition=0L, finished=true),
        Instance(id=1L, label=None,    partition=0L, finished=true),
        Instance(id=2L, label=None,    partition=0L, finished=true)
      ),
      0.1,
      1
    )
    ,

    // Single partition, no unlabeled instances
    (
      Seq(
        Instance(id=0L, label=Some(0), partition=0, finished=false),
        Instance(id=1L, label=Some(0), partition=0, finished=false),
        Instance(id=2L, label=Some(0), partition=0, finished=false)
      ),
      Seq(
        Instance(id=0L, label=Some(0), partition=0L, finished=true),
        Instance(id=1L, label=Some(0), partition=0L, finished=true),
        Instance(id=2L, label=Some(0), partition=0L, finished=true)
      ),
      0.1,
      1
    )
    ,

    // Single partition, two labeled instances
    (
      Seq(
        Instance(id=0L, label=Some(0), partition=0, finished=false),
        Instance(id=1L, label=Some(1), partition=0, finished=false),
        Instance(id=2L, label=None,    partition=0, finished=false)
      ),
      Seq(
        Instance(id=0L, label=Some(0), partition=0, finished=false),
        Instance(id=1L, label=Some(1), partition=0, finished=false),
        Instance(id=2L, label=None,    partition=0, finished=false)
      ),
      0.1,
      1
    )
    ,

    // Single partition, all instances are labeled with the same value
    (
      Seq(
        Instance(id=0L, label=Some(0), partition=0, finished=false),
        Instance(id=1L, label=Some(0), partition=0, finished=false),
        Instance(id=2L, label=Some(0), partition=0, finished=false),
        Instance(id=3L, label=Some(1), partition=0, finished=false),
        Instance(id=4L, label=None, partition=0, finished=false)
      ),
      Seq(
        Instance(id=0L, label=Some(0), partition=0, finished=false),
        Instance(id=1L, label=Some(0), partition=0, finished=false),
        Instance(id=2L, label=Some(0), partition=0, finished=false),
        Instance(id=3L, label=Some(1), partition=0, finished=false),
        Instance(id=4L, label=None, partition=0, finished=false)
      ),
      0.1,
      1
    )
    ,

    // Two partitions, one will finish
    (
      Seq(
        Instance(id=0L, label=Some(0), partition=0, finished=false),
        Instance(id=1L, label=Some(0), partition=0, finished=false),
        Instance(id=2L, label=None,    partition=0, finished=false),
        Instance(id=3L, label=Some(0), partition=1, finished=false),
        Instance(id=4L, label=Some(1), partition=1, finished=false),
        Instance(id=5L, label=None,    partition=1, finished=false)
      ),
      Seq(
        Instance(id=0L, label=Some(0), partition=0, finished=true),
        Instance(id=1L, label=Some(0), partition=0, finished=true),
        Instance(id=2L, label=None,    partition=0, finished=true),
        Instance(id=3L, label=Some(0), partition=1, finished=false),
        Instance(id=4L, label=Some(1), partition=1, finished=false),
        Instance(id=5L, label=None,    partition=1, finished=false)
      ),
      0.1,
      1
    )
    ,
    // Two partitions, one is already finished
    (
      Seq(
        Instance(id=0L, label=Some(0), partition=0, finished=false),
        Instance(id=1L, label=Some(0), partition=0, finished=false),
        Instance(id=2L, label=None,    partition=0, finished=false),
        Instance(id=3L, label=Some(0), partition=1, finished=true),
        Instance(id=4L, label=Some(1), partition=1, finished=true),
        Instance(id=5L, label=None,    partition=1, finished=true)
      ),
      Seq(
        Instance(id=0L, label=Some(0), partition=0, finished=true),
        Instance(id=1L, label=Some(0), partition=0, finished=true),
        Instance(id=2L, label=None,    partition=0, finished=true),
        Instance(id=3L, label=Some(0), partition=1, finished=true),
        Instance(id=4L, label=Some(1), partition=1, finished=true),
        Instance(id=5L, label=None,    partition=1, finished=true)
      ),
      0.1,
      1
    )
    ,

    // Two partitions, both will finish
    (
      Seq(
        Instance(id=0L, label=Some(0), partition=0, finished=false),
        Instance(id=1L, label=Some(0), partition=0, finished=false),
        Instance(id=2L, label=None,    partition=0, finished=false),
        Instance(id=3L, label=Some(1), partition=1, finished=false),
        Instance(id=4L, label=Some(1), partition=1, finished=false),
        Instance(id=5L, label=None,    partition=1, finished=false)
      ),
      Seq(
        Instance(id=0L, label=Some(0), partition=0, finished=true),
        Instance(id=1L, label=Some(0), partition=0, finished=true),
        Instance(id=2L, label=None,    partition=0, finished=true),
        Instance(id=3L, label=Some(1), partition=1, finished=true),
        Instance(id=4L, label=Some(1), partition=1, finished=true),
        Instance(id=5L, label=None,    partition=1, finished=true)
      ),
      0.1,
      1
    )
    ,

    // A partition with two labels, but with significantly more of one
    (
      Seq(
        Instance(id=0L, label=Some(0), partition=0, finished=false),
        Instance(id=1L, label=Some(1), partition=0, finished=false),
        Instance(id=2L, label=Some(1), partition=0, finished=false),
        Instance(id=3L, label=Some(1), partition=0, finished=false),
        Instance(id=4L, label=Some(1), partition=0, finished=false),
        Instance(id=5L, label=Some(1), partition=0, finished=false),
        Instance(id=6L, label=Some(1), partition=0, finished=false),
        Instance(id=7L, label=Some(1), partition=0, finished=false),
        Instance(id=8L, label=Some(1), partition=0, finished=false),
        Instance(id=9L, label=None, partition=0, finished=false)
      ),
      Seq(
        Instance(id=0L, label=Some(0), partition=0, finished=true),
        Instance(id=1L, label=Some(1), partition=0, finished=true),
        Instance(id=2L, label=Some(1), partition=0, finished=true),
        Instance(id=3L, label=Some(1), partition=0, finished=true),
        Instance(id=4L, label=Some(1), partition=0, finished=true),
        Instance(id=5L, label=Some(1), partition=0, finished=true),
        Instance(id=6L, label=Some(1), partition=0, finished=true),
        Instance(id=7L, label=Some(1), partition=0, finished=true),
        Instance(id=8L, label=Some(1), partition=0, finished=true),
        Instance(id=9L, label=None, partition=0, finished=true)
      ),
      0.2,
      1
    )
    ,

    // A partition with two labels, with *almost* significantly more of one
    (
      Seq(
        Instance(id=0L, label=Some(0), partition=0, finished=false),
        Instance(id=1L, label=Some(1), partition=0, finished=false),
        Instance(id=2L, label=Some(1), partition=0, finished=false),
        Instance(id=3L, label=Some(1), partition=0, finished=false),
        Instance(id=4L, label=Some(1), partition=0, finished=false),
        Instance(id=5L, label=Some(1), partition=0, finished=false),
        Instance(id=6L, label=Some(1), partition=0, finished=false),
        Instance(id=7L, label=Some(1), partition=0, finished=false),
        Instance(id=8L, label=Some(1), partition=0, finished=false),
        Instance(id=9L, label=None, partition=0, finished=false)
      ),
      Seq(
        Instance(id=0L, label=Some(0), partition=0, finished=false),
        Instance(id=1L, label=Some(1), partition=0, finished=false),
        Instance(id=2L, label=Some(1), partition=0, finished=false),
        Instance(id=3L, label=Some(1), partition=0, finished=false),
        Instance(id=4L, label=Some(1), partition=0, finished=false),
        Instance(id=5L, label=Some(1), partition=0, finished=false),
        Instance(id=6L, label=Some(1), partition=0, finished=false),
        Instance(id=7L, label=Some(1), partition=0, finished=false),
        Instance(id=8L, label=Some(1), partition=0, finished=false),
        Instance(id=9L, label=None, partition=0, finished=false)
      ),
      0.15,
      1
    )
    ,

    // Single partition with less instances than number of neighbors
    (
      Seq(
        Instance(id=0L, label=Some(0), partition=0, finished=false),
        Instance(id=1L, label=Some(1), partition=0, finished=false),
        Instance(id=2L, label=None, partition=0, finished=false)
      ),
      Seq(
        Instance(id=0L, label=Some(0), partition=0, finished=true),
        Instance(id=1L, label=Some(1), partition=0, finished=true),
        Instance(id=2L, label=None, partition=0, finished=true)
      ),
      0.1,
      3
    )
    ,

    // Single partition with all instances labeled
    (
      Seq(
        Instance(id=0L, label=Some(0), partition=0, finished=false),
        Instance(id=1L, label=Some(1), partition=0, finished=false),
        Instance(id=2L, label=Some(2), partition=0, finished=false)
      ),
      Seq(
        Instance(id=0L, label=Some(0), partition=0, finished=true),
        Instance(id=1L, label=Some(1), partition=0, finished=true),
        Instance(id=2L, label=Some(2), partition=0, finished=true)
      ),
      0.1,
      2
    )
  )

  "RGT.computeFinished" should "work for the given examples" in {
    forAll (examples) { (input, expected, maxGiniImpurity, noNeighbors) => {
      val inputRdd = sc.parallelize(input.map(x => (x.partition, RGTNode(Vectors.dense(0.0), x.label, x.partition, finished = true))))
      val expectedRdd = sc.parallelize(expected.map(x => (x.partition,RGTNode(Vectors.dense(0.0), x.label, x.partition, finished = true))))

      // TODO: This is duplicated from RGT's code; it should be in a helper and tested
      val labelsCount = inputRdd
        .filter{case (_, node) => node.label.nonEmpty}
        .map{case (_, node) => node.label.get}
        .countByValue
        .toMap
      val labeledCount = labelsCount.values.sum
      val numberOfClasses = labelsCount.keys.size

      val labelsWeights = labelsCount.mapValues(x => labeledCount / (numberOfClasses * x.toDouble)).map(identity)

      val result = RGT.computeFinished(inputRdd, labelsWeights, maxGiniImpurity, noNeighbors)

      assertRDDEquals(result, expectedRdd)
    }}
  }
}
