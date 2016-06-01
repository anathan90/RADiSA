package utils

import breeze.linalg.{SparseVector, Vector, DenseVector}

case class LabeledPoint(val label: Double, val features: SparseVector[Double])
