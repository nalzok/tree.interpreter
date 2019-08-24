library(MASS)
library(ranger)
library(tree.interpreter)

set.seed(42L)
data(Boston)
rf <- ranger(medv ~ ., Boston)
arf <- annotateHierarchicalPrediction(rf, Boston[, -14])
str(arf$forest$node.sizes)
