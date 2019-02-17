package com.github.fvictorio.rgt

import com.holdenkarau.spark.testing.{SharedSparkContext,RDDComparisons}

abstract class SparkUnitTest extends UnitTest with RDDComparisons with SharedSparkContext {
  override def beforeAll(): Unit = {
    super.beforeAll()
    sc.setLogLevel("error")
  }
}
