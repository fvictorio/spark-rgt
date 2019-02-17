package com.github.fvictorio.rgt

import org.apache.spark.ml.linalg.Vectors
import org.scalatest._

class RGTIsFinishedSpec extends SparkUnitTest with Matchers {
  "RGT.isFinished" should "return true if column `finished` has all true values" in {
    val rdd = sc.parallelize(Seq(true, true, true).map(finished =>
      (0L, RGTNode(Vectors.dense(0.0), Some(0L), 0L, finished))
    ))

    val result = RGT.isFinished(rdd)

    result should be(true)
  }

  it should "return false if some value of column `finished` is false" in {
    val rdd = sc.parallelize(Seq(false, true, true).map(finished =>
      (0L, RGTNode(Vectors.dense(0.0), Some(0L), 0L, finished))
    ))

    val result = RGT.isFinished(rdd)

    result should be(false)
  }

  it should "return false if all values of column `finished` are false" in {
    val rdd = sc.parallelize(Seq(false, false, false).map(finished =>
      (0L, RGTNode(Vectors.dense(0.0), Some(0L), 0L, finished))
    ))

    val result = RGT.isFinished(rdd)

    result should be(false)
  }
}
