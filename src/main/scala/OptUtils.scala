package utils

import org.apache.spark.SparkContext
import breeze.linalg._
import breeze.linalg.{DenseVector,Vector,SparseVector}
import com.github.fommil.netlib.BLAS
import breeze.numerics._
import org.apache.spark.rdd.RDD
import org.apache.spark.broadcast.Broadcast
import scala.collection.mutable.{ArrayBuffer,Map}

object OptUtils {


	def readLibsvmData(sc: SparkContext, path: String, numPartitions: Int, numFeatures: Int): RDD[LabeledPoint] = {
		val data = sc.textFile(path).repartition(numPartitions)//repartition(numPartitions)
		val bcNumFeatures = sc.broadcast(numFeatures)

		val indexArray = new ArrayBuffer[Int](numPartitions*2)
		for (i<-0 to numPartitions-1){	
			var first = (Math.floor(numFeatures / numPartitions.toDouble) * (i)).toInt
			indexArray += first
			var last = (Math.floor(numFeatures / numPartitions.toDouble) * (i + 1) - 1.0).toInt
			if (i == (numPartitions - 1)) {
				last = last + (numFeatures % numPartitions).toInt
			}
			indexArray += last
		}

		val bcIndexArray = sc.broadcast(indexArray)
 
		val formatData = data.map{x =>
			val parts = x.split(" ")
			val featureRA = parts.slice(1,parts.size+1).map(_.split(":") match { case Array(i,j) => (i.toInt-1,j.toDouble)})
			val features = new SparseVector(featureRA.map(x=>x._1),featureRA.map(x=>x._2), bcNumFeatures.value)
			for (i<-0 to (bcIndexArray.value.size-1)){
				if (features(bcIndexArray.value(i)) == 0.0) features(bcIndexArray.value(i)) = 1.0E-16
			}
			LabeledPoint(parts(0).toDouble,features)
		}
		formatData
	}

	//get number of observations per partition
	def getNumObsPerPartition(data: RDD[Array[LabeledPoint]]): Map[Int,(Int,Int)] = {
		var info = data.mapPartitions{ iter =>
			val dataArr = iter.next
			val partitionID = dataArr(0).label
			Iterator((partitionID.toInt,dataArr.size-1))
		}.collect

		info = info.distinct	
			
		val hMap = Map[Int,(Int,Int)]()
	
		for (i<-0 to info.size-1) {
			val pId = info(i)._1
			val obs = info(i)._2 
			if (pId %10 == 0) hMap += (pId -> (0,obs-1))
			else {
				val pIdPrev = pId - 1
				val lower = hMap(pIdPrev)._2 + 1
				val upper = lower + obs -1
				hMap += (pId -> (lower,upper))
			}
		}

		hMap
	}

	//get number of observations per partition
	def getFeatureMap(data: RDD[Array[LabeledPoint]]): Map[Int,Int] = {
		var info = data.mapPartitions{ iter =>
			val dataArr = iter.next
			val minFeature = min(dataArr(0).features).toInt
			Iterator(minFeature)
		}.collect

		info = info.distinct	
			
		val hMap = Map[Int,Int]()
	
		for (i<-0 to info.size-1) {
			val key = info(i)
			val value = i
			hMap += (key -> value)
		}
		hMap
	}



	//compute Hinge Loss when data is partitioned across observations	
	def computeHingeLoss(data: RDD[Array[LabeledPoint]],
		bcPrimal: Vector[Double],
		numObs: Int,
		regularizationVal: Double): Double = {
			
		val partialLoss = data.mapPartitions{ iter =>
			val dataArr = iter.next
			var localLoss = 0.0
			for (i<-0 to dataArr.size - 1) {
				val currentPoint = dataArr(i)
				val Y = currentPoint.label
				val X = currentPoint.features
				localLoss += Math.max(1 - Y * (X.dot(bcPrimal)),0.0)
			}
			Iterator(localLoss)
		}.reduce(_+_)	
		
		(1/numObs.toDouble) * partialLoss + (0.5 * regularizationVal * Math.pow(bcPrimal.norm(2),2))
	}

	//compute Hinge Loss gradient when data is partitioned across observations	
	def computeHingeGradient(data: RDD[Array[LabeledPoint]],
		bcPrimal: Vector[Double],
		numObs: Int,
		regularizationVal: Double,
		miniBatchFraction: Double): Vector[Double] = {
			
		val partialGradient = data.mapPartitions{ iter =>
			val dataArr = iter.next
			var grad = Vector.zeros[Double](bcPrimal.size)
			val nLocal = dataArr.size -1
			for (i<-0 to Math.floor(nLocal * miniBatchFraction).toInt) {
				val currentPoint = dataArr(i)
				val Y = currentPoint.label
				val X = currentPoint.features
				if (Y * X.dot(bcPrimal) < 1.0) {
					grad += X * (Y * (-1.0))
				}
			}
			Iterator(grad)
		}.treeAggregate(Vector.zeros[Double](bcPrimal.size))((acc,value) => (acc + value),(acc1,acc2) => (acc1+acc2))
		
		(1/(Math.floor(numObs.toDouble * miniBatchFraction))) * partialGradient + (regularizationVal * bcPrimal)
	}


	def testAccuracy(testData: RDD[LabeledPoint],
		bcPrimal: Vector[Double],
		numObs: Int): Double = {
		
		(numObs - testData.map(x => if ((x.features).dot(bcPrimal) * (x.label) > 0.0) 0.0 else 1.0).reduce(_ + _)) / numObs
			
	}
 

	
}
