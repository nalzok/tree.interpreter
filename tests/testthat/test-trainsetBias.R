library(ranger)
library(randomForest)

test_that('trainsetBias works for ranger & classification tree', {
  set.seed(42L)
  trainID <- sample(150, 120)
  rf <- ranger(Species ~ ., iris[trainID, ], keep.inbag = TRUE)
  tidy.RF <- tidyRF(rf, iris[trainID, -5], iris[trainID, 5])

  trainset.bias <- trainsetBias(tidy.RF)
  expect_equal(dim(trainset.bias), c(1, 3))
  expect_equal(dimnames(trainset.bias),
               list('Bias', levels(iris$Species)))
})

test_that('trainsetBias works for randomForest & classification tree', {
  set.seed(42L)
  trainID <- sample(150, 120)
  rf <- randomForest(Species ~ ., iris[trainID, ], keep.inbag = TRUE)
  tidy.RF <- tidyRF(rf, iris[trainID, -5], iris[trainID, 5])

  trainset.bias <- trainsetBias(tidy.RF)
  expect_equal(dim(trainset.bias), c(1, 3))
  expect_equal(dimnames(trainset.bias),
               list('Bias', levels(iris$Species)))
})

test_that('trainsetBias works for ranger & regression tree', {
  set.seed(42L)
  trainID <- sample(506, 400)
  rf <- ranger(medv ~ ., Boston[trainID, ], keep.inbag = TRUE)
  tidy.RF <-
      tidyRF(rf, Boston[trainID, -14], Boston[trainID, 14])

  trainset.bias <- trainsetBias(tidy.RF)
  expect_equal(dim(trainset.bias), c(1, 1))
  expect_equal(dimnames(trainset.bias),
               list('Bias', 'Response'))
})

test_that('trainsetBias works for randomForest & regression tree', {
  set.seed(42L)
  trainID <- sample(506, 400)
  rf <- randomForest(medv ~ ., Boston[trainID, ], keep.inbag = TRUE)
  tidy.RF <-
      tidyRF(rf, Boston[trainID, -14], Boston[trainID, 14])

  trainset.bias <- trainsetBias(tidy.RF)
  expect_equal(dim(trainset.bias), c(1, 1))
  expect_equal(dimnames(trainset.bias),
               list('Bias', 'Response'))
})

