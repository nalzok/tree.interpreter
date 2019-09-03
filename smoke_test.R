library(MASS)
library(ranger)
library(tree.interpreter)

# Classification
set.seed(42L)
rf <- ranger(Species ~ ., iris, keep.inbag = TRUE)
delta.node.resp <- deltaNodeResponse(rf, iris[, -5], iris[, 5])
delta.node.resp$delta.node.resp[1]

# Regression
set.seed(42L)
data(Boston)
rf <- ranger(medv ~ ., Boston, keep.inbag = TRUE)
delta.node.resp <- deltaNodeResponse(rf, Boston[, -14], Boston[, 14])
delta.node.resp$delta.node.resp[1]
