package com.github.fvictorio.rgt

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{functions => f}
import org.scalactic.{Equality, TolerantNumerics}
import org.scalatest._
import com.github.fvictorio.rgt.experiments.Datasets

import scala.util.Random

class RelieffSpec extends SparkUnitTest with Matchers {
  val iris = Seq(
    (0L, RGTNode(Vectors.dense(5.1,3.5,1.4,0.2),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(4.9,3.0,1.4,0.2),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(4.7,3.2,1.3,0.2),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(4.6,3.1,1.5,0.2),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.0,3.6,1.4,0.2),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.4,3.9,1.7,0.4),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(4.6,3.4,1.4,0.3),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.0,3.4,1.5,0.2),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(4.4,2.9,1.4,0.2),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(4.9,3.1,1.5,0.1),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.4,3.7,1.5,0.2),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(4.8,3.4,1.6,0.2),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(4.8,3.0,1.4,0.1),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(4.3,3.0,1.1,0.1),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.8,4.0,1.2,0.2),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.7,4.4,1.5,0.4),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.4,3.9,1.3,0.4),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.1,3.5,1.4,0.3),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.7,3.8,1.7,0.3),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.1,3.8,1.5,0.3),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.4,3.4,1.7,0.2),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.1,3.7,1.5,0.4),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(4.6,3.6,1.0,0.2),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.1,3.3,1.7,0.5),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(4.8,3.4,1.9,0.2),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.0,3.0,1.6,0.2),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.0,3.4,1.6,0.4),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.2,3.5,1.5,0.2),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.2,3.4,1.4,0.2),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(4.7,3.2,1.6,0.2),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(4.8,3.1,1.6,0.2),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.4,3.4,1.5,0.4),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.2,4.1,1.5,0.1),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.5,4.2,1.4,0.2),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(4.9,3.1,1.5,0.2),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.0,3.2,1.2,0.2),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.5,3.5,1.3,0.2),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(4.9,3.6,1.4,0.1),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(4.4,3.0,1.3,0.2),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.1,3.4,1.5,0.2),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.0,3.5,1.3,0.3),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(4.5,2.3,1.3,0.3),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(4.4,3.2,1.3,0.2),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.0,3.5,1.6,0.6),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.1,3.8,1.9,0.4),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(4.8,3.0,1.4,0.3),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.1,3.8,1.6,0.2),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(4.6,3.2,1.4,0.2),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.3,3.7,1.5,0.2),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.0,3.3,1.4,0.2),Some(0L),0L,false)),
    (0L, RGTNode(Vectors.dense(7.0,3.2,4.7,1.4),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.4,3.2,4.5,1.5),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.9,3.1,4.9,1.5),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.5,2.3,4.0,1.3),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.5,2.8,4.6,1.5),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.7,2.8,4.5,1.3),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.3,3.3,4.7,1.6),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(4.9,2.4,3.3,1.0),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.6,2.9,4.6,1.3),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.2,2.7,3.9,1.4),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.0,2.0,3.5,1.0),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.9,3.0,4.2,1.5),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.0,2.2,4.0,1.0),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.1,2.9,4.7,1.4),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.6,2.9,3.6,1.3),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.7,3.1,4.4,1.4),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.6,3.0,4.5,1.5),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.8,2.7,4.1,1.0),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.2,2.2,4.5,1.5),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.6,2.5,3.9,1.1),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.9,3.2,4.8,1.8),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.1,2.8,4.0,1.3),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.3,2.5,4.9,1.5),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.1,2.8,4.7,1.2),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.4,2.9,4.3,1.3),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.6,3.0,4.4,1.4),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.8,2.8,4.8,1.4),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.7,3.0,5.0,1.7),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.0,2.9,4.5,1.5),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.7,2.6,3.5,1.0),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.5,2.4,3.8,1.1),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.5,2.4,3.7,1.0),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.8,2.7,3.9,1.2),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.0,2.7,5.1,1.6),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.4,3.0,4.5,1.5),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.0,3.4,4.5,1.6),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.7,3.1,4.7,1.5),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.3,2.3,4.4,1.3),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.6,3.0,4.1,1.3),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.5,2.5,4.0,1.3),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.5,2.6,4.4,1.2),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.1,3.0,4.6,1.4),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.8,2.6,4.0,1.2),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.0,2.3,3.3,1.0),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.6,2.7,4.2,1.3),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.7,3.0,4.2,1.2),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.7,2.9,4.2,1.3),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.2,2.9,4.3,1.3),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.1,2.5,3.0,1.1),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.7,2.8,4.1,1.3),Some(1L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.3,3.3,6.0,2.5),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.8,2.7,5.1,1.9),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(7.1,3.0,5.9,2.1),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.3,2.9,5.6,1.8),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.5,3.0,5.8,2.2),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(7.6,3.0,6.6,2.1),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(4.9,2.5,4.5,1.7),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(7.3,2.9,6.3,1.8),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.7,2.5,5.8,1.8),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(7.2,3.6,6.1,2.5),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.5,3.2,5.1,2.0),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.4,2.7,5.3,1.9),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.8,3.0,5.5,2.1),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.7,2.5,5.0,2.0),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.8,2.8,5.1,2.4),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.4,3.2,5.3,2.3),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.5,3.0,5.5,1.8),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(7.7,3.8,6.7,2.2),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(7.7,2.6,6.9,2.3),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.0,2.2,5.0,1.5),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.9,3.2,5.7,2.3),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.6,2.8,4.9,2.0),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(7.7,2.8,6.7,2.0),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.3,2.7,4.9,1.8),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.7,3.3,5.7,2.1),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(7.2,3.2,6.0,1.8),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.2,2.8,4.8,1.8),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.1,3.0,4.9,1.8),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.4,2.8,5.6,2.1),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(7.2,3.0,5.8,1.6),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(7.4,2.8,6.1,1.9),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(7.9,3.8,6.4,2.0),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.4,2.8,5.6,2.2),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.3,2.8,5.1,1.5),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.1,2.6,5.6,1.4),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(7.7,3.0,6.1,2.3),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.3,3.4,5.6,2.4),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.4,3.1,5.5,1.8),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.0,3.0,4.8,1.8),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.9,3.1,5.4,2.1),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.7,3.1,5.6,2.4),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.9,3.1,5.1,2.3),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.8,2.7,5.1,1.9),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.8,3.2,5.9,2.3),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.7,3.3,5.7,2.5),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.7,3.0,5.2,2.3),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.3,2.5,5.0,1.9),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.5,3.0,5.2,2.0),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(6.2,3.4,5.4,2.3),Some(2L),0L,false)),
    (0L, RGTNode(Vectors.dense(5.9,3.0,5.1,1.8),Some(2L),0L,false))
  )

  "ReliefF" should "work for m=1 and k=2" in {
    implicit val doubleEq: Equality[Double] = TolerantNumerics.tolerantDoubleEquality(1e-5f)

    val rdd: RDD[(Long, RGTNode)] = sc.parallelize(Seq(
      (0L, RGTNode(Vectors.dense(-1.0, 0.0), Some(0L), 0L, false)),
      (0L, RGTNode(Vectors.dense( 0.0, 0.0), Some(0L), 0L, false)),
      (0L, RGTNode(Vectors.dense( 1.0, 0.0), Some(0L), 0L, false)),
      (0L, RGTNode(Vectors.dense(-1.0, 1.0), Some(1L), 0L, false)),
      (0L, RGTNode(Vectors.dense( 0.0, 1.0), Some(1L), 0L, false)),
      (0L, RGTNode(Vectors.dense( 1.0, 1.0), Some(1L), 0L, false)),
      (0L, RGTNode(Vectors.dense(-1.0, 2.0), Some(2L), 0L, false)),
      (0L, RGTNode(Vectors.dense( 0.0, 2.0), Some(2L), 0L, false)),
      (0L, RGTNode(Vectors.dense( 1.0, 2.0), Some(2L), 0L, false))
    ))

    val result = ReliefF.getWeights(rdd, 9, 1, new Random(1))

    assert(result(0)(0) === 0.0)
    assert(result(0)(1) === 1.0)
  }

  it should "work for iris with all instances labeled (k = 1)" in {
    implicit val doubleEq: Equality[Double] = TolerantNumerics.tolerantDoubleEquality(1e-2f)

    val rdd = sc.parallelize(iris)

    val result = ReliefF.getWeights(rdd, 150, 1, new Random(1))

    println(result.mkString(","))

    assert(result(0)(0) === 0.4280122)
    assert(result(0)(1) === 0.4870958)
    assert(result(0)(2) === 0.9579036)
    assert(result(0)(3) === 1.0)
  }

  it should "work for iris with all instances labeled (k = 10)" in {
    implicit val doubleEq: Equality[Double] = TolerantNumerics.tolerantDoubleEquality(1e-2f)

    val rdd = sc.parallelize(iris)

    val result = ReliefF.getWeights(rdd, 150, 10, new Random(1))

    println(result.mkString(","))

    assert(result(0)(0) === 0.4294832)
    assert(result(0)(1) === 0.3873913)
    assert(result(0)(2) === 0.960238)
    assert(result(0)(3) === 1.0)
  }

}
