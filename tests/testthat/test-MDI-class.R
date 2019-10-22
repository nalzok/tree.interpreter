library(ranger)
library(randomForest)

test_that('MDI works for ranger & classification tree', {
  set.seed(42L)
  rfobj <- ranger(Species ~ ., iris,
                  keep.inbag = TRUE, importance = 'impurity')
  tidy.RF <- tidyRF(rfobj, iris[, -5], iris[, 5])

  iris.MDITree <- MDITree(tidy.RF, 1, iris[, -5], iris[, 5])
  expect_equal(dim(iris.MDITree), c(ncol(iris) - 1, nlevels(iris$Species)))
  expect_equal(dimnames(iris.MDITree),
               list(names(iris[, -5]),
                    levels(iris$Species)))

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

  iris.MDITree <- MDITree(tidy.RF, 1, iris[, -5], iris[, 5])
  expect_equal(dim(iris.MDITree), c(ncol(iris) - 1, nlevels(iris$Species)))
  expect_equal(dimnames(iris.MDITree),
               list(names(iris[, -5]),
                    levels(iris$Species)))

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
  rfobj <- ranger(Species ~ ., iris, keep.inbag = TRUE)
  tidy.RF <- tidyRF(rfobj, iris[, -5], iris[, 5])

  iris.MDIoobTree <- MDIoobTree(tidy.RF, 1, iris[, -5], iris[, 5])
  expect_equal(dim(iris.MDIoobTree), c(ncol(iris) - 1, nlevels(iris$Species)))
  expect_equal(dimnames(iris.MDIoobTree),
               list(names(iris[, -5]),
                    levels(iris$Species)))

  iris.MDIoob <- MDIoob(tidy.RF, iris[, -5], iris[, 5])
  expect_equal(dim(iris.MDIoob), c(ncol(iris) - 1, nlevels(iris$Species)))
  expect_equal(dimnames(iris.MDIoob),
               list(names(iris[, -5]),
                    levels(iris$Species)))
})

test_that('MDIoob works for randomForest & classification tree', {
  set.seed(42L)
  rfobj <- randomForest(Species ~ ., iris, keep.inbag = TRUE)
  tidy.RF <- tidyRF(rfobj, iris[, -5], iris[, 5])

  iris.MDIoobTree <- MDIoobTree(tidy.RF, 1, iris[, -5], iris[, 5])
  expect_equal(dim(iris.MDIoobTree), c(ncol(iris) - 1, nlevels(iris$Species)))
  expect_equal(dimnames(iris.MDIoobTree),
               list(names(iris[, -5]),
                    levels(iris$Species)))

  iris.MDIoob <- MDIoob(tidy.RF, iris[, -5], iris[, 5])
  expect_equal(dim(iris.MDIoob), c(ncol(iris) - 1, nlevels(iris$Species)))
  expect_equal(dimnames(iris.MDIoob),
               list(names(iris[, -5]),
                    levels(iris$Species)))
})

test_that(paste('MDIoob emits error for ranger & classification tree',
                'when keep.inbag = FALSE'), {
  set.seed(42L)
  rfobj <- ranger(Species ~ ., iris)
  expect_warning(tidy.RF <- tidyRF(rfobj, iris[, -5], iris[, 5]),
                 'keep.inbag = FALSE; all samples will be considered in-bag.')

  expect_error(MDIoobTree(tidy.RF, 1, iris[, -5], iris[, 5]),
               'No out-of-bag data available.')
  expect_error(MDIoob(tidy.RF, iris[, -5], iris[, 5]),
               'No out-of-bag data available.')
})

test_that(paste('MDIoob emits error for randomForest & classification tree',
                'when keep.inbag = FALSE'), {
  set.seed(42L)
  rfobj <- randomForest(Species ~ ., iris)
  expect_warning(tidy.RF <- tidyRF(rfobj, iris[, -5], iris[, 5]),
                 'keep.inbag = FALSE; all samples will be considered in-bag.')

  expect_error(MDIoobTree(tidy.RF, 1, iris[, -5], iris[, 5]),
               'No out-of-bag data available.')
  expect_error(MDIoob(tidy.RF, iris[, -5], iris[, 5]),
               'No out-of-bag data available.')
})
