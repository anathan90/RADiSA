package solver

import org.apache.spark.SparkContext
import breeze.linalg._
import breeze.linalg.{DenseVector,Vector,SparseVector}
import com.github.fommil.netlib.BLAS
//import org.apache.spark.mllib.regression.LabeledPoint
//import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import scala.collection.mutable.ArrayBuffer
import utils._

object GradientDescent {

	def runOpt(
		data: RDD[Array[LabeledPoint]],
		testData: RDD[LabeledPoint],
		iterations: Int,
		numObs: Int,
		numFeatures: Int,
		regularizationVal: Double,
		stepsizeConstant: Double,
		miniBatch: Double,
		update: String,
		printIter: Int,
		format: String): (Vector[Double],scala.collection.mutable.ArrayBuffer[String]) = { 

		val numPartitions = data.partitions.length.toInt	
		var r = new scala.util.Random
			
		var w = Vector.zeros[Double](numFeatures)
		
		val primalLoss = OptUtils.computeHingeLoss(data,w,numObs,regularizationVal)
		val testAccuracy= {
				if (testData != null) { OptUtils.testAccuracy(testData,w,numObs)}
				else {0000.00000}
		}
		println("Iteration \t LossFunction \t ||grad|| \t TestAccuracy \t Time")
		printf("%.5f \t %.5f \t %.7s \t %.7s \t %.5f \n",0.0,primalLoss,"--------",testAccuracy,0.0)				

		val lossFunctionHistory = new ArrayBuffer[String](iterations)
		lossFunctionHistory.append("Iter# \t Loss \t Time")
		lossFunctionHistory.append("0	 \t 1.0    \t 0.0")		
			
		var iterCounter = 1

		while (iterCounter <= iterations ){
			 				
			val start = System.nanoTime

			val globalGrad = OptUtils.computeHingeGradient(data,w,numObs,regularizationVal,miniBatch)
			
			w -= (stepsizeConstant/(1.0 + 1.0*Math.sqrt(iterCounter.toDouble-1.0))) *  globalGrad
			val startOpt = System.nanoTime	

			if (iterCounter % printIter == 0) {
				//calculate primalLoss
				val elapsedTime = (System.nanoTime - start)/Math.pow(10,9)
				val primalLoss = OptUtils.computeHingeLoss(data,w,numObs,regularizationVal)	
			
				val testAccuracy= {
					if (testData != null) { OptUtils.testAccuracy(testData,w,numObs)}
					else {0000.00000}
				}
				printf("%.5f \t %.5f \t %.5f \t %.5f \t %.5f \n",iterCounter.toDouble,primalLoss,globalGrad.norm(2),testAccuracy,elapsedTime)				
	
				lossFunctionHistory.append(iterCounter.toString+"   \t "+f"$primalLoss%3.7f"+" \t "+f"$elapsedTime%3.7f")
			}
			iterCounter = iterCounter + 1
		}
		(w,lossFunctionHistory)
	}

}
