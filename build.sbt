name := "spark-rgt"

version := "0.1"

scalaVersion := "2.11.8"

parallelExecution in Test := false
fork in Test := true
javaOptions ++= Seq("-Xms512M", "-Xmx2048M", "-XX:MaxPermSize=2048M", "-XX:+CMSClassUnloadingEnabled")

libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.2.1" % "provided"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.2.1" % "provided"
libraryDependencies += "org.apache.spark" %% "spark-hive" % "2.2.1" % "provided"

libraryDependencies += "org.scalactic" %% "scalactic" % "3.0.4"
libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.4" % "test"
libraryDependencies += "com.holdenkarau" %% "spark-testing-base" % "2.2.0_0.8.0" % "test"

test in assembly := {}

// uncomment to compile without assertions
// scalacOptions := Seq("-Xdisable-assertions")
