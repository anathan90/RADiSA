# RADiSA

RADiSA is an optimization method for large scale machine learning. The current implementation assumes that the observations are distributed, but not the features. However, each partition only operates on a subset of features in each iteration. 

This code implements the following algorithms:
 -- RADiSA
 -- Gradient Descent (and mini-batch SGD)

The present code trains hinge-loss SVM for binary classification. 

# Getting Started

To compile the code:

```
sbt compile; sbt package
```

To run the code:

```
./onlyRun
```

