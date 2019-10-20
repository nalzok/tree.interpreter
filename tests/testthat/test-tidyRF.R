library(ranger)
library(randomForest)
library(MASS)

expected_names <- c("num.classes", "num.trees", "feature.names",
                    "inbag.counts", "left.children", "right.children",
                    "split.variables", "split.values", "node.sizes",
                    "node.resp", "delta.node.resp.left",
                    "delta.node.resp.right")

test_that('tidyRF works for ranger & classification tree', {
  set.seed(42L)
  rf <- ranger(Species ~ ., iris, keep.inbag = TRUE)

  tidy.RF <- tidyRF(rf, iris[, -5], iris[, 5])
  expect_true('tidyRF' %in% class(tidy.RF))
  expect_equal(names(tidy.RF), expected_names)
  expect_gte(min(sapply(tidy.RF$node.sizes, min)), 1)
  expect_false(any(sapply(tidy.RF$node.resp, anyNA)))
  expect_false(any(sapply(tidy.RF$delta.node.resp.left, anyNA)))
  expect_false(any(sapply(tidy.RF$delta.node.resp.right, anyNA)))
  expect_equal(length(tidy.RF$node.resp), rf$num.trees)
  expect_equal(colnames(tidy.RF$node.resp[[1]]),
               levels(iris$Species))
})

test_that('tidyRF works for randomForest & classification tree', {
  set.seed(42L)
  rf <- randomForest(Species ~ ., iris, keep.inbag = TRUE)

  tidy.RF <- tidyRF(rf, iris[, -5], iris[, 5])
  expect_true('tidyRF' %in% class(tidy.RF))
  expect_equal(names(tidy.RF), expected_names)
  expect_gte(min(sapply(tidy.RF$node.sizes, min)), 1)
  expect_false(any(sapply(tidy.RF$node.resp, anyNA)))
  expect_false(any(sapply(tidy.RF$delta.node.resp.left, anyNA)))
  expect_false(any(sapply(tidy.RF$delta.node.resp.right, anyNA)))
  expect_equal(length(tidy.RF$node.resp), rf$ntree)
  expect_equal(colnames(tidy.RF$node.resp[[1]]),
               levels(iris$Species))
})

test_that('tidyRF works for ranger & regression tree', {
  set.seed(42L)
  rf <- ranger(medv ~ ., Boston, keep.inbag = TRUE)

  tidy.RF <- tidyRF(rf, Boston[, -14], Boston[, 14])
  expect_true('tidyRF' %in% class(tidy.RF))
  expect_equal(names(tidy.RF), expected_names)
  expect_gte(min(sapply(tidy.RF$node.sizes, min)), 1)
  expect_false(any(sapply(tidy.RF$node.resp, anyNA)))
  expect_false(any(sapply(tidy.RF$delta.node.resp.left, anyNA)))
  expect_false(any(sapply(tidy.RF$delta.node.resp.right, anyNA)))
  expect_equal(length(tidy.RF$node.resp), rf$num.trees)
  expect_equal(colnames(tidy.RF$node.resp[[1]]),
              'Response')
})

test_that('tidyRF works for randomForest & regression tree', {
  set.seed(42L)
  rf <- randomForest(medv ~ ., Boston, keep.inbag = TRUE)

  tidy.RF <- tidyRF(rf, Boston[, -14], Boston[, 14])
  expect_true('tidyRF' %in% class(tidy.RF))
  expect_equal(names(tidy.RF), expected_names)
  expect_gte(min(sapply(tidy.RF$node.sizes, min)), 0)   # randomForest bug
  expect_false(any(sapply(tidy.RF$node.resp, anyNA)))
  expect_false(any(sapply(tidy.RF$delta.node.resp.left, anyNA)))
  expect_false(any(sapply(tidy.RF$delta.node.resp.right, anyNA)))
  expect_equal(length(tidy.RF$node.resp), rf$ntree)
  expect_equal(colnames(tidy.RF$node.resp[[1]]),
              'Response')
})

test_that('tidyRF works for ranger when keep.inbag = FALSE', {
  set.seed(42L)
  rf <- ranger(Species ~ ., iris, keep.inbag = FALSE)

  expect_warning(tidy.RF <- tidyRF(rf, iris[, -5], iris[, 5]),
                 'keep.inbag = FALSE, using all observations')
  expect_true('tidyRF' %in% class(tidy.RF))
  expect_equal(names(tidy.RF), expected_names)
  expect_gte(min(sapply(tidy.RF$node.sizes, min)), 1)
  expect_false(any(sapply(tidy.RF$node.resp, anyNA)))
  expect_false(any(sapply(tidy.RF$delta.node.resp.left, anyNA)))
  expect_false(any(sapply(tidy.RF$delta.node.resp.right, anyNA)))
  expect_equal(length(tidy.RF$node.resp), rf$num.trees)
  expect_equal(colnames(tidy.RF$node.resp[[1]]),
               levels(iris$Species))
})

test_that('tidyRF works for randomForest when keep.inbag = FALSE', {
  set.seed(42L)
  rf <- randomForest(medv ~ ., Boston, keep.inbag = FALSE)

  expect_warning(tidy.RF <- tidyRF(rf, Boston[, -14], Boston[, 14]),
                 'keep.inbag = FALSE, using all observations')
  expect_true('tidyRF' %in% class(tidy.RF))
  expect_equal(names(tidy.RF), expected_names)
  expect_gte(min(sapply(tidy.RF$node.sizes, min)), 0)   # randomForest bug
  expect_false(any(sapply(tidy.RF$node.resp, anyNA)))
  expect_false(any(sapply(tidy.RF$delta.node.resp.left, anyNA)))
  expect_false(any(sapply(tidy.RF$delta.node.resp.right, anyNA)))
  expect_equal(length(tidy.RF$node.resp), rf$ntree)
  expect_equal(colnames(tidy.RF$node.resp[[1]]),
              'Response')
})

