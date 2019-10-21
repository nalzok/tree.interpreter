library(ranger)
library(randomForest)

test_that('MDI works for ranger & classification tree', {
  set.seed(42L)
  rfobj <- ranger(Species ~ ., iris,
                  keep.inbag = TRUE, importance = 'impurity')
  tidy.RF <- tidyRF(rfobj, iris[, -5], iris[, 5])

  iris.MDI <- MDI(tidy.RF, iris[, -5], iris[, 5])
  expect_equal(dim(iris.MDI), c(ncol(iris) - 1, nlevels(iris$Species)))
  expect_equal(dimnames(iris.MDI),
               list(names(iris[, -5]),
                    levels(iris$Species)))
  expect_equal(as.vector(rowSums(iris.MDI)),
               as.vector(ranger::importance(rfobj) /
                         sum(tidy.RF$inbag.counts[[1]])))
})

test_that('MDI works for randomForest & classification tree', {
  set.seed(42L)
  rfobj <- randomForest(Species ~ ., iris,
                        keep.inbag = TRUE, importance = TRUE)
  tidy.RF <- tidyRF(rfobj, iris[, -5], iris[, 5])

  iris.MDI <- MDI(tidy.RF, iris[, -5], iris[, 5])
  expect_equal(dim(iris.MDI), c(ncol(iris) - 1, nlevels(iris$Species)))
  expect_equal(dimnames(iris.MDI),
               list(names(iris[, -5]),
                    levels(iris$Species)))
  expect_equal(as.vector(rowSums(iris.MDI)),
               as.vector(importance(rfobj)[, 'MeanDecreaseGini'] /
                         sum(tidy.RF$inbag.counts[[1]])))
})

test_that('MDIoob works for ranger & classification tree', {
  set.seed(42L)
  rfobj <- ranger(Species ~ ., iris,
                  keep.inbag = TRUE, importance = 'impurity')
  tidy.RF <- tidyRF(rfobj, iris[, -5], iris[, 5])

  iris.MDIoob <- MDIoob(tidy.RF, iris[, -5], iris[, 5])
  expect_equal(dim(iris.MDIoob), c(ncol(iris) - 1, nlevels(iris$Species)))
  expect_equal(dimnames(iris.MDIoob),
               list(names(iris[, -5]),
                    levels(iris$Species)))
})

test_that('MDIoob works for randomForest & classification tree', {
  set.seed(42L)
  rfobj <- randomForest(Species ~ ., iris,
                        keep.inbag = TRUE, importance = TRUE)
  tidy.RF <- tidyRF(rfobj, iris[, -5], iris[, 5])

  iris.MDIoob <- MDIoob(tidy.RF, iris[, -5], iris[, 5])
  expect_equal(dim(iris.MDIoob), c(ncol(iris) - 1, nlevels(iris$Species)))
  expect_equal(dimnames(iris.MDIoob),
               list(names(iris[, -5]),
                    levels(iris$Species)))
})
