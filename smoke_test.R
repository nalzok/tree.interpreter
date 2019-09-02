library(MASS)
library(ranger)
library(tree.interpreter)

# Classification
set.seed(42L)
rf <- ranger(Species ~ ., iris)
rfHierPred <- annotateHierarchicalPrediction(rf, iris[, -5])
str(rfHierPred$forest$hierarchical.predictions)
rfHierPred$forest$hierarchical.predictions[[1]]

# Regression
set.seed(42L)
data(Boston)
rf <- ranger(medv ~ ., Boston)
rfHierPred <- annotateHierarchicalPrediction(rf, Boston[, -14])
str(rfHierPred$forest$hierarchical.predictions)
rfHierPred$forest$hierarchical.predictions[[1]]
