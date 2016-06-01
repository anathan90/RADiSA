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

object oRADiSA {

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
		var featureID = (0 to numPartitions-1).toList
		var globalGrad = Vector.zeros[Double](1)	
		
	
		while (iterCounter <= iterations ){
			 				
			val start = System.nanoTime

			if (update.toLowerCase == "svrg" & iterCounter > 0) {
				globalGrad = OptUtils.computeHingeGradient(data,w,numObs,regularizationVal,1.0)
			}
			
			featureID = util.Random.shuffle(featureID)
	
			val startOpt = System.nanoTime
			w = data.mapPartitionsWithIndex(updatePartition(_,_,w,featureID,globalGrad,iterCounter,numObs,numPartitions,numFeatures,regularizationVal,stepsizeConstant,miniBatch,update)).treeAggregate(Vector.zeros[Double](numFeatures))((acc,value) => (acc+value),(acc1,acc2) => (acc1+acc2))
			
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


	private def updatePartition(
		partitionID: Int,
		iter: Iterator[Array[LabeledPoint]],
		bcPrimal: Vector[Double],
		featureID: List[Int],
		globalGrad: Vector[Double],
		globalIter: Int,
		numObs: Int,
		numPartitions: Int,
		numFeatures: Int,
		regularizationVal: Double,
		stepsizeConstant: Double,
		miniBatch: Double,
		update: String): Iterator[Vector[Double]] = { //Iterator[(Int,Vector[Double])] = {

			val dataArr = iter.next
			val partitionFeatureID = featureID(partitionID)

			var first = (Math.floor(numFeatures / numPartitions.toDouble) * (partitionFeatureID)).toInt
			var last = (Math.floor(numFeatures / numPartitions.toDouble) * (partitionFeatureID + 1) - 1.0).toInt
			if (partitionFeatureID == (numPartitions - 1)) {
				last = last + (numFeatures % numPartitions).toInt
			}

			//println(partitionID+") first "+first+" last "+last)			

			var wLocal = bcPrimal(first to last)
			var grad = Vector.zeros[Double](last-first+1)
			var gradRefMinusMu = Vector.zeros[Double](1)

			if (update.toLowerCase == "svrg" & globalIter > 0) {
				gradRefMinusMu =  wLocal * regularizationVal - globalGrad(first to last)
			}

			var r = new scala.util.Random		

			val nLocal = dataArr.size - 1			

			//println("Batch size: "+Math.floor(nLocal * miniBatch).toInt)

			for (i <- 0 to Math.floor(nLocal * miniBatch).toInt) {
				
				if (update.toLowerCase == "svrg" & globalIter > 0) {
					grad = wLocal * regularizationVal 
					wLocal -= (stepsizeConstant/(1.0 + 1.0*Math.sqrt(globalIter.toDouble-1.0))) * (grad - gradRefMinusMu) 
				}
				else {
					// randomly select local example
					val idx = r.nextInt(nLocal) + 1
					val X = new SparseVector(Array.empty,Array.empty[Double],(last-first+1))
					val currentPoint = dataArr(idx)
					val Y = currentPoint.label
					grad = wLocal * regularizationVal  

					// Perform 1 Pass SGD
					if ((currentPoint.features.index.indexWhere(_ == last) - currentPoint.features.index.indexWhere(_ == first)) > 1) {
						X := currentPoint.features(first to last)
						if (Y * X.dot(wLocal) < 1.0) {
							grad += X * (Y * (-1.0))	
						}
					}			
					
					if (update.toLowerCase == "sgd") wLocal -= (stepsizeConstant/(1.0+globalIter.toDouble)) * grad 						
					else wLocal -= (stepsizeConstant/(1.0+1.0*(globalIter.toDouble-1.0))) * grad //wLocal -= (stepsizeConstant) * grad
				}
			}
			var wFinal = Vector.zeros[Double](numFeatures)
			wFinal(first to last) := wLocal
			Iterator(wFinal) 				 
	}	
}
