
lazy val supportedScalaVersions = "2.11.12"

ThisBuild / name 	 := "Capstone"
ThisBuild / organization := "com.eeyle"
ThisBuild / version 	 := "0.1"
ThisBuild / scalaVersion := "2.11.12"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.4.5"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.4.5"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.4.5"
