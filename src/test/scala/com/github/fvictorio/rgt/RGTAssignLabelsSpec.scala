package com.github.fvictorio.rgt

import org.apache.spark.ml.linalg.Vectors
import org.scalatest._
import org.scalatest.prop.TableDrivenPropertyChecks

class RGTAssignLabelsSpec extends SparkUnitTest with Matchers with TableDrivenPropertyChecks {
  case class Instance(id: Long, label: Option[Long], partition: Long, prediction: Long = -1)

  val examples = Table(
    ("input", "expected"),

    // Single partition, one labeled instance
    (
      Seq(
        Instance(id=0L, label=Some(0), partition=0),
        Instance(id=1L, label=None,    partition=0),
        Instance(id=2L, label=None,    partition=0)
      ),
      Seq(
        Instance(id=0L, label=Some(0L), partition=0L, prediction=0),
        Instance(id=1L, label=None,     partition=0L, prediction=0),
        Instance(id=2L, label=None,     partition=0L, prediction=0)
      )
    ),

    // Single partition, two labeled instance with the same label
    (
      Seq(
        Instance(id=0L, label=Some(0), partition=0),
        Instance(id=1L, label=Some(0), partition=0),
        Instance(id=2L, label=None,    partition=0)
      ),
      Seq(
        Instance(id=0L, label=Some(0), partition=0L, prediction=0),
        Instance(id=1L, label=Some(0), partition=0L, prediction=0),
        Instance(id=2L, label=None,    partition=0L, prediction=0)
      )
    ),

    // Single partition, all instances labeled with the same label
    (
      Seq(
        Instance(id=0L, label=Some(0), partition=0),
        Instance(id=1L, label=Some(0), partition=0),
        Instance(id=2L, label=Some(0), partition=0)
      ),
      Seq(
        Instance(id=0L, label=Some(0L), partition=0L, prediction=0),
        Instance(id=1L, label=Some(0L),     partition=0L, prediction=0),
        Instance(id=2L, label=Some(0L),     partition=0L, prediction=0)
      )
    ),

    // Single partition, two labeled instance with different labels
    (
      Seq(
        Instance(id=0L, label=Some(0), partition=0),
        Instance(id=1L, label=Some(1), partition=0),
        Instance(id=2L, label=None,    partition=0)
      ),
      Seq(
        Instance(id=0L, label=Some(0), partition=0L, prediction=0),
        Instance(id=1L, label=Some(1), partition=0L, prediction=0),
        Instance(id=2L, label=None,    partition=0L, prediction=0)
      )
    ),

    // Single partition, more instances labeled with one label than the other
    (
      Seq(
        Instance(id=0L, label=Some(0), partition=0),
        Instance(id=1L, label=None,    partition=0),
        Instance(id=2L, label=None,    partition=0),
        Instance(id=3L, label=Some(1), partition=0),
        Instance(id=4L, label=None,    partition=0),
        Instance(id=5L, label=Some(0),    partition=0)
      ),
      Seq(
        Instance(id=0L, label=Some(0), partition=0, prediction=0),
        Instance(id=1L, label=None,    partition=0, prediction=0),
        Instance(id=2L, label=None,    partition=0, prediction=0),
        Instance(id=3L, label=Some(1), partition=0, prediction=0),
        Instance(id=4L, label=None,    partition=0, prediction=0),
        Instance(id=5L, label=Some(0), partition=0, prediction=0)
      )
    ),

    // Two partitions, one labeled instance in each one
    (
      Seq(
        Instance(id=0L, label=Some(0), partition=0),
        Instance(id=1L, label=None,    partition=0),
        Instance(id=2L, label=None,    partition=0),
        Instance(id=3L, label=Some(1), partition=1),
        Instance(id=4L, label=None,    partition=1),
        Instance(id=5L, label=None,    partition=1)
      ),
      Seq(
        Instance(id=0L, label=Some(0), partition=0, prediction=0),
        Instance(id=1L, label=None,    partition=0, prediction=0),
        Instance(id=2L, label=None,    partition=0, prediction=0),
        Instance(id=3L, label=Some(1), partition=1, prediction=1),
        Instance(id=4L, label=None,    partition=1, prediction=1),
        Instance(id=5L, label=None,    partition=1, prediction=1)
      )
    )
  )

  "RGT.assignLabels" should "work for the given examples" in {
    forAll (examples) { (input, expected) => {
      val inputRdd = sc.parallelize(input.map(x => (x.partition, RGTNode(Vectors.dense(0.0), x.label, x.partition, finished = true))))
      val expectedRdd = sc.parallelize(expected.map(x => (x.partition, RGTOutputNode(Vectors.dense(0.0), x.label, x.prediction))))

      val result = RGT.assignLabels(inputRdd)

      assertRDDEquals(result, expectedRdd)
    }}
  }
}
