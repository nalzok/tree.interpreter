library(ranger)
library(randomForest)
library(MASS)

test_that('featureContrib works for ranger & regression tree', {
  set.seed(42L)
  trainID <- sample(506, 400)
  rf <- ranger(medv ~ ., Boston[trainID, ], keep.inbag = TRUE)
  tidy.RF <- tidyRF(rf, Boston[trainID, -14], Boston[trainID, 14])

  feature.contrib <- featureContrib(tidy.RF, Boston[-trainID, -14])
  expect_equal(dim(feature.contrib), c(13, 1, 106))

  trainset.bias <- trainsetBias(tidy.RF)
  expect_equal(as.vector(apply(feature.contrib, c(2, 3), sum)) +
               as.vector(trainset.bias),
               predict(rf, Boston[-trainID, -14])$predictions)
})

test_that('featureContrib works for randomForest & regression tree', {
  set.seed(42L)
  trainID <- sample(506, 400)
  rf <- randomForest(medv ~ ., Boston[trainID, ], keep.inbag = TRUE)
  tidy.RF <- tidyRF(rf, Boston[trainID, -14], Boston[trainID, 14])

  feature.contrib <- featureContrib(tidy.RF, Boston[-trainID, -14])
  expect_equal(dim(feature.contrib), c(13, 1, 106))

  trainset.bias <- trainsetBias(tidy.RF)
  expect_equal(as.vector(apply(feature.contrib, c(2, 3), sum)) +
               as.vector(trainset.bias),
               unname(predict(rf, Boston[-trainID, -14])))
})

test_that('featureContrib retains order of features for ranger', {
  set.seed(42L)
  dummy <- data.frame(var1=1:100, var2=rnorm(100), var3=42L, var4=-(1:100))
  rf <- ranger(var4 ~ var2 + var1, dummy, keep.inbag = TRUE)
  tidy.RF <- tidyRF(rf, dummy[, -(3:4)], dummy[, 4])

  feature.contrib <- featureContrib(tidy.RF, dummy[, -(3:4)])
  expect_equal(dimnames(feature.contrib),
               list(c('var1', 'var2'), 'Response', as.character(1:100)))
})

test_that('featureContrib retains order of features for randomForest', {
  set.seed(42L)
  dummy <- data.frame(var1=1:100, var2=rnorm(100), var3=42L, var4=-(1:100))
  rf <- randomForest(var4 ~ var2 + var1, dummy, keep.inbag = TRUE)
  tidy.RF <- tidyRF(rf, dummy[, -(3:4)], dummy[, 4])

  feature.contrib <- featureContrib(tidy.RF, dummy[, -(3:4)])
  expect_equal(dimnames(feature.contrib),
               list(c('var1', 'var2'), 'Response', as.character(1:100)))
})

