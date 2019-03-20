name := "spark-rgt"

version := "0.0.1"

scalaVersion := "2.11.8"

lazy val buildSettings = Seq(
  organization        := "com.github.fvictorio",
  version             := "0.0.1"
)

licenses += ("MIT", url("http://opensource.org/licenses/MIT"))

parallelExecution in Test := false
fork in Test := true
javaOptions ++= Seq("-Xms512M", "-Xmx2048M", "-XX:MaxPermSize=2048M", "-XX:+CMSClassUnloadingEnabled")

resolvers += "jitpack" at "https://jitpack.io"

libraryDependencies += "com.github.fvictorio" % "spark-nnd" % "9dcd149141c3a6732e3112e919a9e154274fb48c"

libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.2.1" % "provided"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.2.1" % "provided"
libraryDependencies += "org.apache.spark" %% "spark-hive" % "2.2.1" % "provided"

libraryDependencies += "org.scalactic" %% "scalactic" % "3.0.4"
libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.4" % "test"
libraryDependencies += "com.holdenkarau" %% "spark-testing-base" % "2.2.0_0.8.0" % "test"

test in assembly := {}

// uncomment to compile without assertions
// scalacOptions := Seq("-Xdisable-assertions")
