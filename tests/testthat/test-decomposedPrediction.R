library(ranger)

test_that('decomposedPrediction works for classification tree', {
  set.seed(42L)
  trainID <- sample(150, 120)
  rf <- ranger(Species ~ ., iris[trainID, ], keep.inbag = TRUE)
  delta.node.resp <- deltaNodeResponse(rf, iris[trainID, -5], iris[trainID, 5])
  decomposed.prediction <-
      decomposedPrediction(delta.node.resp, iris[-trainID, -5])
  expect_equal(length(decomposed.prediction), 30)
  expect_equal(rownames(decomposed.prediction[[1]]),
               levels(iris$Species))
  expect_equal(colnames(decomposed.prediction[[1]]),
               rf$forest$independent.variable.names)
  bias <- attr(decomposed.prediction, 'bias')
  expect_equal(rownames(bias), levels(iris$Species))
  expect_equal(colnames(bias), 'Bias')
})

library(MASS)

test_that('decomposedPrediction works for regression tree', {
  set.seed(42L)
  trainID <- sample(506, 400)
  rf <- ranger(medv ~ ., Boston[trainID, ], keep.inbag = TRUE)
  delta.node.resp <-
      deltaNodeResponse(rf, Boston[trainID, -14], Boston[trainID, 14])
  decomposed.prediction <-
      decomposedPrediction(delta.node.resp, Boston[-trainID, -14])
  expect_equal(length(decomposed.prediction), 106)
  expect_equal(rownames(decomposed.prediction[[1]]),
               'Response')
  expect_equal(colnames(decomposed.prediction[[1]]),
               rf$forest$independent.variable.names)
  bias <- attr(decomposed.prediction, 'bias')
  expect_equal(rownames(bias), 'Response')
  expect_equal(colnames(bias), 'Bias')
  expect_equal(sapply(1:106, function(x)
                      rowSums(decomposed.prediction[[x]]) + bias),
               predict(rf, Boston[-trainID, -14])$predictions)
})

