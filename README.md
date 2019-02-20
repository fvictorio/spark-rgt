# spark-rgt

This is an Apache Spark implementation of RGT, a semi-supervised learning algorithm.

## Installation

Add the library to your `build.sbt` file:

```
libraryDependencies += "default" % "spark-rgt_2.11" % "0.0.1"
```

## Usage

Import the library, create a new instance of RGT, and pass a dataset:

```scala
import com.github.fvictorio.rgt.RGT

val rgt = new RGT()

val input = sc.parallelize(Seq(
    
))

val result = rgt.transform(input);
```

Of course, you'll probably want to load a dataset from something like HDFS. See the [`experiments`](src/main/scala/com/github/fvictorio/rgt/experiments/) for some examples that show how to transform a CSV into the required format.
