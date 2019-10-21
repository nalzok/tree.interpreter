library(ranger)
library(randomForest)

test_that('featureContrib works for ranger & classification tree', {
  set.seed(42L)
  trainID <- sample(150, 120)
  rfobj <- ranger(Species ~ ., iris[trainID, ], keep.inbag = TRUE)
  tidy.RF <- tidyRF(rfobj, iris[trainID, -5], iris[trainID, 5])

  feature.contrib <- featureContrib(tidy.RF, iris[-trainID, -5])
  expect_equal(dim(feature.contrib), c(4, 3, 30))
  expect_equal(dimnames(feature.contrib),
               list(names(iris[, -5]),
                    levels(iris$Species),
                    as.character(1:30)))
})

test_that('featureContrib works for randomForest & classification tree', {
  set.seed(42L)
  trainID <- sample(150, 120)
  rfobj <- randomForest(Species ~ ., iris[trainID, ], keep.inbag = TRUE)
  tidy.RF <- tidyRF(rfobj, iris[trainID, -5], iris[trainID, 5])

  feature.contrib <- featureContrib(tidy.RF, iris[-trainID, -5])
  expect_equal(dim(feature.contrib), c(4, 3, 30))
  expect_equal(dimnames(feature.contrib),
               list(names(iris[, -5]),
                    levels(iris$Species),
                    as.character(1:30)))
})
