library(ranger)
library(randomForest)
library(MASS)

test_that('deltaNodeResponse works for ranger & classification tree', {
  set.seed(42L)
  rf <- ranger(Species ~ ., iris, keep.inbag = TRUE)
  delta.node.resp <- deltaNodeResponse(rf, iris[, -5], iris[, 5])
  expect_false(any(sapply(delta.node.resp$delta.node.resp, anyNA)))
  expect_equal(length(delta.node.resp$delta.node.resp), rf$num.trees)
  expect_equal(colnames(delta.node.resp$delta.node.resp[[1]]),
               levels(iris$Species))
})

test_that('deltaNodeResponse works for randomForest & classification tree', {
  set.seed(42L)
  rf <- randomForest(Species ~ ., iris, keep.inbag = TRUE)
  delta.node.resp <- deltaNodeResponse(rf, iris[, -5], iris[, 5])
  expect_false(any(sapply(delta.node.resp$delta.node.resp, anyNA)))
  expect_equal(length(delta.node.resp$delta.node.resp), rf$ntree)
  expect_equal(colnames(delta.node.resp$delta.node.resp[[1]]),
               levels(iris$Species))
})

test_that('deltaNodeResponse works for ranger & regression tree', {
  set.seed(42L)
  rf <- ranger(medv ~ ., Boston, keep.inbag = TRUE)
  delta.node.resp <- deltaNodeResponse(rf, Boston[, -14], Boston[, 14])
  expect_false(any(sapply(delta.node.resp$delta.node.resp, anyNA)))
  expect_equal(length(delta.node.resp$delta.node.resp), rf$num.trees)
  expect_equal(colnames(delta.node.resp$delta.node.resp[[1]]),
              'Response')
})

test_that('deltaNodeResponse works for randomForest & regression tree', {
  set.seed(42L)
  rf <- randomForest(medv ~ ., Boston, keep.inbag = TRUE)
  delta.node.resp <- deltaNodeResponse(rf, Boston[, -14], Boston[, 14])
  expect_false(any(sapply(delta.node.resp$delta.node.resp, anyNA)))
  expect_equal(length(delta.node.resp$delta.node.resp), rf$ntree)
  expect_equal(colnames(delta.node.resp$delta.node.resp[[1]]),
              'Response')
})

