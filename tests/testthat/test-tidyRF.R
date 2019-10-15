library(ranger)
library(randomForest)
library(MASS)

test_that('tidyRF works for ranger & classification tree', {
  set.seed(42L)
  rf <- ranger(Species ~ ., iris, keep.inbag = TRUE)
  tidy.RF <- tidyRF(rf, iris[, -5], iris[, 5])
  expect_false(any(sapply(tidy.RF$node.resp, anyNA)))
  expect_equal(length(tidy.RF$node.resp), rf$num.trees)
  expect_equal(colnames(tidy.RF$node.resp[[1]]),
               levels(iris$Species))
})

test_that('tidyRF works for randomForest & classification tree', {
  set.seed(42L)
  rf <- randomForest(Species ~ ., iris, keep.inbag = TRUE)
  tidy.RF <- tidyRF(rf, iris[, -5], iris[, 5])
  expect_false(any(sapply(tidy.RF$node.resp, anyNA)))
  expect_equal(length(tidy.RF$node.resp), rf$ntree)
  expect_equal(colnames(tidy.RF$node.resp[[1]]),
               levels(iris$Species))
})

test_that('tidyRF works for ranger & regression tree', {
  set.seed(42L)
  rf <- ranger(medv ~ ., Boston, keep.inbag = TRUE)
  tidy.RF <- tidyRF(rf, Boston[, -14], Boston[, 14])
  expect_false(any(sapply(tidy.RF$node.resp, anyNA)))
  expect_equal(length(tidy.RF$node.resp), rf$num.trees)
  expect_equal(colnames(tidy.RF$node.resp[[1]]),
              'Response')
})

test_that('tidyRF works for randomForest & regression tree', {
  set.seed(42L)
  rf <- randomForest(medv ~ ., Boston, keep.inbag = TRUE)
  tidy.RF <- tidyRF(rf, Boston[, -14], Boston[, 14])
  expect_false(any(sapply(tidy.RF$node.resp, anyNA)))
  expect_equal(length(tidy.RF$node.resp), rf$ntree)
  expect_equal(colnames(tidy.RF$node.resp[[1]]),
              'Response')
})

