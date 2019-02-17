package com.github.fvictorio.rgt

import Helpers._
import org.apache.spark.ml.linalg.Vectors
import org.scalatest._

class HelpersSpec extends UnitTest with Matchers {
  "multiply" should "multiply two vectors" in {
    val v1 = Vectors.dense(2.0, 3.0)
    val v2 = Vectors.dense(-1.0, 1.5)

    val result = multiply(v1, v2)

    assert(result == Vectors.dense(-2.0, 4.5))
  }

  it should "fail if the vectors have different size" in {
    val v1 = Vectors.dense(2.0, 3.0)
    val v2 = Vectors.dense(-1.0)

    an [AssertionError] should be thrownBy multiply(v1, v2)
  }

  "subtract" should "subtract two vectors" in {
    val v1 = Vectors.dense(2.0, 3.0)
    val v2 = Vectors.dense(-1.0, 1.5)

    val result = subtract(v1, v2)

    assert(result == Vectors.dense(3.0, 1.5))
  }

  it should "fail if the vectors have different size" in {
    val v1 = Vectors.dense(2.0, 3.0)
    val v2 = Vectors.dense(-1.0)

    an [AssertionError] should be thrownBy subtract(v1, v2)
  }

  "ones" should "return a vector made of ones" in {
    val result = ones(3)
    assert(result == Vectors.dense(1.0, 1.0, 1.0))
  }
}
