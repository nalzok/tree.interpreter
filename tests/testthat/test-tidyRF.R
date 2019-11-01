library(ranger)
library(randomForest)

expected_names <- c("num.trees", "feature.names", "num.classes", "class.names",
                    "inbag.counts", "left.children", "right.children",
                    "split.variables", "split.values", "node.sizes",
                    "node.resp", "delta.node.resp.left",
                    "delta.node.resp.right")

test_that('tidyRF works for ranger & classification tree', {
  set.seed(42L)
  rfobj <- ranger(Species ~ ., iris, keep.inbag = TRUE)

  tidy.RF <- tidyRF(rfobj, iris[, -5], iris[, 5])
  expect_true('tidyRF' %in% class(tidy.RF))
  expect_equal(names(tidy.RF), expected_names)
  expect_gte(min(sapply(tidy.RF$node.sizes, min)), 1)
  expect_false(any(sapply(tidy.RF$node.resp, anyNA)))
  expect_false(any(sapply(tidy.RF$delta.node.resp.left, anyNA)))
  expect_false(any(sapply(tidy.RF$delta.node.resp.right, anyNA)))
  expect_equal(length(tidy.RF$node.resp), rfobj$num.trees)
  expect_equal(tidy.RF$class.names, levels(iris$Species))
  expect_equal(colnames(tidy.RF$node.resp[[1]]),
               levels(iris$Species))
})

test_that('tidyRF works for randomForest & classification tree', {
  set.seed(42L)
  rfobj <- randomForest(Species ~ ., iris, keep.inbag = TRUE)

  tidy.RF <- tidyRF(rfobj, iris[, -5], iris[, 5])
  expect_true('tidyRF' %in% class(tidy.RF))
  expect_equal(names(tidy.RF), expected_names)
  expect_gte(min(sapply(tidy.RF$node.sizes, min)), 1)
  expect_false(any(sapply(tidy.RF$node.resp, anyNA)))
  expect_false(any(sapply(tidy.RF$delta.node.resp.left, anyNA)))
  expect_false(any(sapply(tidy.RF$delta.node.resp.right, anyNA)))
  expect_equal(length(tidy.RF$node.resp), rfobj$ntree)
  expect_equal(tidy.RF$class.names, levels(iris$Species))
  expect_equal(colnames(tidy.RF$node.resp[[1]]),
               levels(iris$Species))
})

test_that('tidyRF works for ranger & regression tree', {
  set.seed(42L)
  rfobj <- ranger(mpg ~ ., mtcars, keep.inbag = TRUE)

  tidy.RF <- tidyRF(rfobj, mtcars[, -1], mtcars[, 1])
  expect_true('tidyRF' %in% class(tidy.RF))
  expect_equal(names(tidy.RF), expected_names)
  expect_gte(min(sapply(tidy.RF$node.sizes, min)), 1)
  expect_false(any(sapply(tidy.RF$node.resp, anyNA)))
  expect_false(any(sapply(tidy.RF$delta.node.resp.left, anyNA)))
  expect_false(any(sapply(tidy.RF$delta.node.resp.right, anyNA)))
  expect_equal(length(tidy.RF$node.resp), rfobj$num.trees)
  expect_equal(tidy.RF$class.names, 'Response')
  expect_equal(colnames(tidy.RF$node.resp[[1]]),
              'Response')
})

test_that('tidyRF works for randomForest & regression tree', {
  set.seed(42L)
  rfobj <- randomForest(mpg ~ ., mtcars, keep.inbag = TRUE)

  tidy.RF <- tidyRF(rfobj, mtcars[, -1], mtcars[, 1])
  expect_true('tidyRF' %in% class(tidy.RF))
  expect_equal(names(tidy.RF), expected_names)
  expect_gte(min(sapply(tidy.RF$node.sizes, min)), 0)   # randomForest bug
  expect_false(any(sapply(tidy.RF$node.resp, anyNA)))
  expect_false(any(sapply(tidy.RF$delta.node.resp.left, anyNA)))
  expect_false(any(sapply(tidy.RF$delta.node.resp.right, anyNA)))
  expect_equal(length(tidy.RF$node.resp), rfobj$ntree)
  expect_equal(tidy.RF$class.names, 'Response')
  expect_equal(colnames(tidy.RF$node.resp[[1]]),
              'Response')
})

test_that('tidyRF works for ranger when keep.inbag = FALSE', {
  set.seed(42L)
  rfobj <- ranger(Species ~ ., iris, keep.inbag = FALSE)

  expect_warning(tidy.RF <- tidyRF(rfobj, iris[, -5], iris[, 5]),
                 'keep.inbag = FALSE; all samples will be considered in-bag.')
  expect_true('tidyRF' %in% class(tidy.RF))
  expect_equal(names(tidy.RF), expected_names)
  expect_gte(min(sapply(tidy.RF$node.sizes, min)), 1)
  expect_false(any(sapply(tidy.RF$node.resp, anyNA)))
  expect_false(any(sapply(tidy.RF$delta.node.resp.left, anyNA)))
  expect_false(any(sapply(tidy.RF$delta.node.resp.right, anyNA)))
  expect_equal(length(tidy.RF$node.resp), rfobj$num.trees)
  expect_equal(tidy.RF$class.names, levels(iris$Species))
  expect_equal(colnames(tidy.RF$node.resp[[1]]),
               levels(iris$Species))
})

test_that('tidyRF works for randomForest when keep.inbag = FALSE', {
  set.seed(42L)
  rfobj <- randomForest(mpg ~ ., mtcars, keep.inbag = FALSE)

  expect_warning(tidy.RF <- tidyRF(rfobj, mtcars[, -1], mtcars[, 1]),
                 'keep.inbag = FALSE; all samples will be considered in-bag.')
  expect_true('tidyRF' %in% class(tidy.RF))
  expect_equal(names(tidy.RF), expected_names)
  expect_gte(min(sapply(tidy.RF$node.sizes, min)), 0)   # randomForest bug
  expect_false(any(sapply(tidy.RF$node.resp, anyNA)))
  expect_false(any(sapply(tidy.RF$delta.node.resp.left, anyNA)))
  expect_false(any(sapply(tidy.RF$delta.node.resp.right, anyNA)))
  expect_equal(length(tidy.RF$node.resp), rfobj$ntree)
  expect_equal(tidy.RF$class.names, 'Response')
  expect_equal(colnames(tidy.RF$node.resp[[1]]),
              'Response')
})
