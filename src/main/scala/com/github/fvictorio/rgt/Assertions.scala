package com.github.fvictorio.rgt

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.DataType

import scala.annotation.elidable

object Assertions {

  @elidable(elidable.ASSERTION)
  def hasColumn(df: DataFrame, col: String, withType: DataType): Unit = {
    val schema = df.schema

    assert(schema.fieldNames.contains(col), s"The dataframe should have a column `$col`")

    val idIndex = schema.fieldIndex(col)
    val idField = schema.fields(idIndex)
    assert(
      idField.dataType == withType,
      s"The `$col` column should be of type `$withType`, found ${idField.dataType} instead"
    )
  }

  @elidable(elidable.ASSERTION)
  def columnIsUnique(df: DataFrame, col: String): Unit = {
    val count = df.count()
    val distinctCount = df.select(col).distinct.count

    assert(count == distinctCount, s"Values in column '$col' are not unique")
  }

  @elidable(elidable.ASSERTION)
  def hasColumns(df: DataFrame, cols: Seq[String]): Unit = {
    val dfCols = df.columns

    assert(cols.toSet.subsetOf(dfCols.toSet), s"Minimum columns expected ${cols.mkString("[", ", ", "]")}, got ${dfCols.mkString("[", ", ", "]")}")
  }
}
