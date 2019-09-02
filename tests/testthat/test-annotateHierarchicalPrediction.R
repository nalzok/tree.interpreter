library(MASS)
library(ranger)

test_that('Annotating regression trees worksr', {
  set.seed(42L)
  rf <- ranger(medv ~ ., Boston, keep.inbag = TRUE)
  rf <- annotateHierarchicalPrediction(rf, Boston[, -14], Boston[, 14])
  expect_equal(length(rf$forest$hierarchical.predictions), rf$num.trees)
  expect_equal(colnames(rf$forest$hierarchical.predictions[[1]]),
              'Response')
})

test_that('Annotating classification trees works', {
  set.seed(42L)
  rf <- ranger(Species ~ ., iris, keep.inbag = TRUE)
  rf <- annotateHierarchicalPrediction(rf, iris[, -5], iris[, 5])
  expect_equal(length(rf$forest$hierarchical.predictions), rf$num.trees)
  expect_equal(colnames(rf$forest$hierarchical.predictions[[1]]),
               levels(iris$Species))
})

