import java.io._
import utils._
import solver._
import org.apache.log4j.Logger
import org.apache.log4j.Level
import breeze.linalg._
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import scala.collection.mutable.{ArrayBuffer,Map}


object driver {
	
	def main(args: Array[String]) {
			
		// start spark context
		val conf = new SparkConf().setAppName("dfr")
			.set("spark.kryoserializer.buffer.mb","512")
			.set("spark.kryoserializer.buffer.max.mb","1024")

		val options = args.map { arg =>
			arg.dropWhile(_ == '-').split('=') match {
				case Array(opt, v) => (opt -> v)
				case Array(opt) => (opt -> "")
				case _ => throw new IllegalArgumentException("Invalid argument: "+arg)
			}
		}.toMap

	       val rootLogger = Logger.getRootLogger()
               rootLogger.setLevel(Level.ERROR)


		val sc = new SparkContext(conf)	

		
		// read in general inputs
		val inputDirectory = options.getOrElse("inputDirectory","")
		val testDirectory = options.getOrElse("testDirectory","")
		val format = options.getOrElse("format","sparse")
		val printIter = options.getOrElse("printIter","1").toInt
		
		// algorithmic parameters
		val numIterations = options.getOrElse("numIterations","100").toInt
		val numObservations = options.getOrElse("numObservations","0").toInt
		val numFeatures = options.getOrElse("numFeatures","0").toInt
		val numFeatureSplits = options.getOrElse("numFeatureSplits","1").toInt
		val regularizationValue = options.getOrElse("regularizationValue","0").toDouble
		val stepsizeConstant = options.getOrElse("stepsizeConstant","0.0001").toDouble
		val method = options.getOrElse("method","")
		val update = options.getOrElse("update","svrg")
		val parameterCombination = options.getOrElse("paramComb","")
		val numPartitions = options.getOrElse("numPartitions","4").toInt
		val miniBatchFraction = options.getOrElse("miniBatchFraction","1.0").toDouble

		// print out inputs
		println("inputDirectory:        " + inputDirectory)
		println("testDirectory:         " + testDirectory)
		val parsed = inputDirectory.split("/")
		val filename = parsed(parsed.size-1)
		println("format:                " + format)
		println("numPartitions:         " + numPartitions)
		println("printIter:             " + printIter)
		println("numIterations:         " + numIterations)
		println("numObservations:       " + numObservations)	
		println("numFeatures:           " + numFeatures)	
		println("numFeatureSplits:      " + numFeatureSplits)	
		println("regularizationValue:   " + regularizationValue)	
		println("miniBatch Fraction:    " + miniBatchFraction)
		println("stepsizeConstant:      " + stepsizeConstant)
		println("update:		       " + update)		
		println("method:		       " + method)		

		// read in and format data
		val start = System.nanoTime

	
		val data = {
			   if (method.toLowerCase == "oradisa" || method.toLowerCase == "rapsa" || method.toLowerCase == "gradientdescent") {OptUtils.readLibsvmData(sc,inputDirectory,numPartitions,numFeatures).mapPartitions(x => Iterator(x.toArray)).cache() }				      								
			   else {null} 
		}
		
		val n = data.count.toInt

		val testData = {
				if (testDirectory !="") {OptUtils.readLibsvmData(sc,testDirectory,5,numFeatures).cache() }
				else {null}
		}

 
		println("Reading Time: "+(System.nanoTime - start)/Math.pow(10,9))		


		if (method.toLowerCase == "oradisa") {	
	
			val (w,lossFunctionHistory) = oRADiSA.runOpt(data,testData,numIterations,numObservations,numFeatures,regularizationValue,stepsizeConstant,miniBatchFraction,update,printIter,format)
			//val pw = new PrintWriter(new File(filename+"_regParam "+regularizationValue+"_numPartitions "+numPartitions+"_method "+method+" _update "+update+" constant "+stepsizeConstant.toString))
			//pw.write(lossFunctionHistory.toArray.mkString("\n"))
			//pw.close()
			
		}
		else if (method.toLowerCase == "gradientdescent") {
			val (w,lossFunctionHistory) = GradientDescent.runOpt(data,testData,numIterations,numObservations,numFeatures,regularizationValue,stepsizeConstant,miniBatchFraction,update,printIter,format)
			//val pw = new PrintWriter(new File(filename+"_regParam "+regularizationValue+"_numPartitions "+numPartitions+"_method "+method))
			//pw.write(lossFunctionHistory.toArray.mkString("\n"))
			//pw.close()
		}

		println("The algorithm has finished running")
		data.unpersist()
		sc.stop()
	}
}
