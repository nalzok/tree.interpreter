library(ranger)

test_that('deltaNodeResponse works for classification tree', {
  set.seed(42L)
  rf <- ranger(Species ~ ., iris, keep.inbag = TRUE)
  delta.node.resp <- deltaNodeResponse(rf, iris[, -5], iris[, 5])
  expect_equal(length(delta.node.resp$delta.node.resp), rf$num.trees)
  expect_equal(rownames(delta.node.resp$delta.node.resp[[1]]),
               levels(iris$Species))
})

library(MASS)

test_that('deltaNodeResponse works for regression tree', {
  set.seed(42L)
  rf <- ranger(medv ~ ., Boston, keep.inbag = TRUE)
  delta.node.resp <- deltaNodeResponse(rf, Boston[, -14], Boston[, 14])
  expect_equal(length(delta.node.resp$delta.node.resp), rf$num.trees)
  expect_equal(rownames(delta.node.resp$delta.node.resp[[1]]),
              'Response')
})

