library(ranger)
library(randomForest)

test_that('featureContrib works for ranger & regression tree', {
  set.seed(42L)
  trainID <- sample(32, 25)
  rfobj <- ranger(mpg ~ ., mtcars[trainID, ], keep.inbag = TRUE)
  tidy.RF <- tidyRF(rfobj, mtcars[trainID, -1], mtcars[trainID, 1])

  feature.contrib <- featureContrib(tidy.RF, mtcars[-trainID, -1])
  expect_equal(dim(feature.contrib), c(10, 1, 7))

  trainset.bias <- trainsetBias(tidy.RF)
  expect_equal(as.vector(apply(feature.contrib, c(2, 3), sum)) +
               as.vector(trainset.bias),
               predict(rfobj, mtcars[-trainID, -1])$predictions)
})

test_that('featureContrib works for randomForest & regression tree', {
  set.seed(42L)
  trainID <- sample(32, 25)
  rfobj <- randomForest(mpg ~ ., mtcars[trainID, ], keep.inbag = TRUE)
  tidy.RF <- tidyRF(rfobj, mtcars[trainID, -1], mtcars[trainID, 1])

  feature.contrib <- featureContrib(tidy.RF, mtcars[-trainID, -1])
  expect_equal(dim(feature.contrib), c(10, 1, 7))

  trainset.bias <- trainsetBias(tidy.RF)
  expect_equal(as.vector(apply(feature.contrib, c(2, 3), sum)) +
               as.vector(trainset.bias),
               unname(predict(rfobj, mtcars[-trainID, -1])))
})

test_that('featureContrib retains order of features for ranger', {
  set.seed(42L)
  dummy <- data.frame(var1=1:100, var2=rnorm(100), var3=42L, var4=-(1:100))
  rfobj <- ranger(var4 ~ var2 + var1, dummy, keep.inbag = TRUE)
  tidy.RF <- tidyRF(rfobj, dummy[, -(3:4)], dummy[, 4])

  feature.contrib <- featureContrib(tidy.RF, dummy[, -(3:4)])
  expect_equal(dimnames(feature.contrib),
               list(c('var1', 'var2'), 'Response', as.character(1:100)))
})

test_that('featureContrib retains order of features for randomForest', {
  set.seed(42L)
  dummy <- data.frame(var1=1:100, var2=rnorm(100), var3=42L, var4=-(1:100))
  rfobj <- randomForest(var4 ~ var2 + var1, dummy, keep.inbag = TRUE)
  tidy.RF <- tidyRF(rfobj, dummy[, -(3:4)], dummy[, 4])

  feature.contrib <- featureContrib(tidy.RF, dummy[, -(3:4)])
  expect_equal(dimnames(feature.contrib),
               list(c('var1', 'var2'), 'Response', as.character(1:100)))
})
