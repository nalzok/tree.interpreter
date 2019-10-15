library(ranger)
library(randomForest)
library(MASS)

##########################################
# TODO: add tests for keep.inbag = FALSE #
##########################################

test_that('featureContrib works for ranger & classification tree', {
  set.seed(42L)
  trainID <- sample(150, 120)
  rf <- ranger(Species ~ ., iris[trainID, ], keep.inbag = TRUE)
  tidy.RF <- tidyRF(rf, iris[trainID, -5], iris[trainID, 5])

  feature.contrib <-
      featureContrib(tidy.RF, iris[-trainID, -5])
  expect_equal(length(feature.contrib), 30)
  expect_equal(rownames(feature.contrib[[1]]),
               names(iris[trainID, -5]))
  expect_equal(colnames(feature.contrib[[1]]),
               levels(iris$Species))
})

test_that('featureContrib works for randomForest & classification tree', {
  set.seed(42L)
  trainID <- sample(150, 120)
  rf <- randomForest(Species ~ ., iris[trainID, ], keep.inbag = TRUE)
  tidy.RF <- tidyRF(rf, iris[trainID, -5], iris[trainID, 5])

  feature.contrib <-
      featureContrib(tidy.RF, iris[-trainID, -5])
  expect_equal(length(feature.contrib), 30)
  expect_equal(rownames(feature.contrib[[1]]),
               names(iris[trainID, -5]))
  expect_equal(colnames(feature.contrib[[1]]),
               levels(iris$Species))
})

test_that('featureContrib works for ranger & regression tree', {
  set.seed(42L)
  trainID <- sample(506, 400)
  rf <- ranger(medv ~ ., Boston[trainID, ], keep.inbag = TRUE)
  tidy.RF <-
      tidyRF(rf, Boston[trainID, -14], Boston[trainID, 14])

  feature.contrib <-
      featureContrib(tidy.RF, Boston[-trainID, -14])
  expect_equal(length(feature.contrib), 106)
  expect_equal(rownames(feature.contrib[[1]]),
               names(Boston[trainID, -14]))
  expect_equal(colnames(feature.contrib[[1]]),
               'Response')

  trainset.bias <- trainsetBias(tidy.RF)
  expect_equal(sapply(1:106, function(x)
                      colSums(feature.contrib[[x]]) + trainset.bias),
               predict(rf, Boston[-trainID, -14])$predictions)
})

test_that('featureContrib works for randomForest & regression tree', {
  set.seed(42L)
  trainID <- sample(506, 400)
  rf <- randomForest(medv ~ ., Boston[trainID, ], keep.inbag = TRUE)
  tidy.RF <-
      tidyRF(rf, Boston[trainID, -14], Boston[trainID, 14])

  feature.contrib <-
      featureContrib(tidy.RF, Boston[-trainID, -14])
  expect_equal(length(feature.contrib), 106)
  expect_equal(rownames(feature.contrib[[1]]),
               names(Boston[trainID, -14]))
  expect_equal(colnames(feature.contrib[[1]]),
               'Response')

  trainset.bias <- trainsetBias(tidy.RF)
  expect_equal(sapply(1:106, function(x)
                      colSums(feature.contrib[[x]]) + trainset.bias),
               unname(predict(rf, Boston[-trainID, -14])))
})

test_that('trainsetBias works for ranger & classification tree', {
  set.seed(42L)
  trainID <- sample(150, 120)
  rf <- ranger(Species ~ ., iris[trainID, ], keep.inbag = TRUE)
  tidy.RF <- tidyRF(rf, iris[trainID, -5], iris[trainID, 5])

  trainset.bias <- trainsetBias(tidy.RF)
  expect_equal(rownames(trainset.bias), 'Bias')
  expect_equal(colnames(trainset.bias), levels(iris$Species))
})

test_that('trainsetBias works for randomForest & classification tree', {
  set.seed(42L)
  trainID <- sample(150, 120)
  rf <- randomForest(Species ~ ., iris[trainID, ], keep.inbag = TRUE)
  tidy.RF <- tidyRF(rf, iris[trainID, -5], iris[trainID, 5])

  trainset.bias <- trainsetBias(tidy.RF)
  expect_equal(rownames(trainset.bias), 'Bias')
  expect_equal(colnames(trainset.bias), levels(iris$Species))
})

test_that('trainsetBias works for ranger & regression tree', {
  set.seed(42L)
  trainID <- sample(506, 400)
  rf <- ranger(medv ~ ., Boston[trainID, ], keep.inbag = TRUE)
  tidy.RF <-
      tidyRF(rf, Boston[trainID, -14], Boston[trainID, 14])

  trainset.bias <- trainsetBias(tidy.RF)
  expect_equal(rownames(trainset.bias), 'Bias')
  expect_equal(colnames(trainset.bias), 'Response')
})

test_that('trainsetBias works for randomForest & regression tree', {
  set.seed(42L)
  trainID <- sample(506, 400)
  rf <- randomForest(medv ~ ., Boston[trainID, ], keep.inbag = TRUE)
  tidy.RF <-
      tidyRF(rf, Boston[trainID, -14], Boston[trainID, 14])

  trainset.bias <- trainsetBias(tidy.RF)
  expect_equal(rownames(trainset.bias), 'Bias')
  expect_equal(colnames(trainset.bias), 'Response')
})

test_that('featureContrib retains order of features for ranger', {
  set.seed(42L)
  dummy <- data.frame(var1=1:100, var2=rnorm(100), var3=42L, var4=-(1:100))
  rf <- ranger(var4 ~ var2 + var1, dummy, keep.inbag = TRUE)
  tidy.RF <- tidyRF(rf, dummy[, -(3:4)], dummy[, 4])

  feature.contrib <-
      featureContrib(tidy.RF, dummy[, -(3:4)])
  expect_equal(rownames(feature.contrib[[1]]), c('var1', 'var2'))
})

test_that('featureContrib retains order of features for randomForest', {
  set.seed(42L)
  dummy <- data.frame(var1=1:100, var2=rnorm(100), var3=42L, var4=-(1:100))
  rf <- randomForest(var4 ~ var2 + var1, dummy, keep.inbag = TRUE)
  tidy.RF <- tidyRF(rf, dummy[, -(3:4)], dummy[, 4])

  feature.contrib <-
      featureContrib(tidy.RF, dummy[, -(3:4)])
  expect_equal(rownames(feature.contrib[[1]]), c('var1', 'var2'))
})
