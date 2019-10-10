library(ranger)

test_that('featureContribution works for classification tree', {
  set.seed(42L)
  trainID <- sample(150, 120)
  rf <- ranger(Species ~ ., iris[trainID, ], keep.inbag = TRUE)
  delta.node.resp <- deltaNodeResponse(rf, iris[trainID, -5], iris[trainID, 5])

  feature.contribution <-
      featureContribution(delta.node.resp, iris[-trainID, -5])
  expect_equal(length(feature.contribution), 30)
  expect_equal(rownames(feature.contribution[[1]]),
               rf$forest$independent.variable.names)
  expect_equal(colnames(feature.contribution[[1]]),
               levels(iris$Species))
})

test_that('trainsetBias works for classification tree', {
  set.seed(42L)
  trainID <- sample(150, 120)
  rf <- ranger(Species ~ ., iris[trainID, ], keep.inbag = TRUE)
  delta.node.resp <- deltaNodeResponse(rf, iris[trainID, -5], iris[trainID, 5])

  trainset.bias <- trainsetBias(delta.node.resp)
  expect_equal(rownames(trainset.bias), 'Bias')
  expect_equal(colnames(trainset.bias), levels(iris$Species))
})

library(MASS)

test_that('featureContribution works for regression tree', {
  set.seed(42L)
  trainID <- sample(506, 400)
  rf <- ranger(medv ~ ., Boston[trainID, ], keep.inbag = TRUE)
  delta.node.resp <-
      deltaNodeResponse(rf, Boston[trainID, -14], Boston[trainID, 14])

  feature.contribution <-
      featureContribution(delta.node.resp, Boston[-trainID, -14])
  expect_equal(length(feature.contribution), 106)
  expect_equal(rownames(feature.contribution[[1]]),
               rf$forest$independent.variable.names)
  expect_equal(colnames(feature.contribution[[1]]),
               'Response')

  trainset.bias <- trainsetBias(delta.node.resp)
  expect_equal(sapply(1:106, function(x)
                      colSums(feature.contribution[[x]]) + trainset.bias),
               predict(rf, Boston[-trainID, -14])$predictions)
})

test_that('trainsetBias works for regression tree', {
  set.seed(42L)
  trainID <- sample(506, 400)
  rf <- ranger(medv ~ ., Boston[trainID, ], keep.inbag = TRUE)
  delta.node.resp <-
      deltaNodeResponse(rf, Boston[trainID, -14], Boston[trainID, 14])

  trainset.bias <- trainsetBias(delta.node.resp)
  expect_equal(rownames(trainset.bias), 'Bias')
  expect_equal(colnames(trainset.bias), 'Response')
})
