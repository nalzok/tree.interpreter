library(ranger)
library(randomForest)

test_that('MDI works for ranger & regression tree', {
  set.seed(42L)
  rfobj <- ranger(mpg ~ ., mtcars,
                  keep.inbag = TRUE, importance = 'impurity')
  tidy.RF <- tidyRF(rfobj, mtcars[, -1], mtcars[, 1])

  mtcars.MDITree <- MDITree(tidy.RF, 1, mtcars[, -1], mtcars[, 1])
  expect_equal(dim(mtcars.MDITree), c(ncol(mtcars) - 1, 1))
  expect_equal(dimnames(mtcars.MDITree),
               list(names(mtcars[, -1]),
                    'Response'))

  mtcars.MDI <- MDI(tidy.RF, mtcars[, -1], mtcars[, 1])
  expect_equal(dim(mtcars.MDI), c(ncol(mtcars) - 1, 1))
  expect_equal(dimnames(mtcars.MDI),
               list(names(mtcars[, -1]),
                    'Response'))
  expect_equal(as.vector(rowSums(mtcars.MDI)),
               as.vector(ranger::importance(rfobj) /
                         sum(tidy.RF$inbag.counts[[1]])))
})

test_that('MDI works for randomForest & regression tree', {
  set.seed(42L)
  rfobj <- randomForest(mpg ~ ., mtcars,
                        keep.inbag = TRUE, importance = TRUE)
  tidy.RF <- tidyRF(rfobj, mtcars[, -1], mtcars[, 1])

  mtcars.MDITree <- MDITree(tidy.RF, 1, mtcars[, -1], mtcars[, 1])
  expect_equal(dim(mtcars.MDITree), c(ncol(mtcars) - 1, 1))
  expect_equal(dimnames(mtcars.MDITree),
               list(names(mtcars[, -1]),
                    'Response'))

  mtcars.MDI <- MDI(tidy.RF, mtcars[, -1], mtcars[, 1])
  expect_equal(dim(mtcars.MDI), c(ncol(mtcars) - 1, 1))
  expect_equal(dimnames(mtcars.MDI),
               list(names(mtcars[, -1]),
                    'Response'))
  expect_equal(as.vector(rowSums(mtcars.MDI)),
               as.vector(importance(rfobj)[, 'IncNodePurity'] /
                         sum(tidy.RF$inbag.counts[[1]])),
               tolerance = 1e-6)
})

test_that('MDIoob works for ranger & regression tree', {
  set.seed(42L)
  rfobj <- ranger(mpg ~ ., mtcars, keep.inbag = TRUE)
  tidy.RF <- tidyRF(rfobj, mtcars[, -1], mtcars[, 1])

  mtcars.MDIoobTree <- MDIoobTree(tidy.RF, 1, mtcars[, -1], mtcars[, 1])
  expect_equal(dim(mtcars.MDIoobTree), c(ncol(mtcars) - 1, 1))
  expect_equal(dimnames(mtcars.MDIoobTree),
               list(names(mtcars[, -1]),
                    'Response'))

  mtcars.MDIoob <- MDIoob(tidy.RF, mtcars[, -1], mtcars[, 1])
  expect_equal(dim(mtcars.MDIoob), c(ncol(mtcars) - 1, 1))
  expect_equal(dimnames(mtcars.MDIoob),
               list(names(mtcars[, -1]),
                    'Response'))
})

test_that('MDIoob works for randomForest & regression tree', {
  set.seed(42L)
  rfobj <- randomForest(mpg ~ ., mtcars, keep.inbag = TRUE)
  tidy.RF <- tidyRF(rfobj, mtcars[, -1], mtcars[, 1])

  mtcars.MDIoobTree <- MDIoobTree(tidy.RF, 1, mtcars[, -1], mtcars[, 1])
  expect_equal(dim(mtcars.MDIoobTree), c(ncol(mtcars) - 1, 1))
  expect_equal(dimnames(mtcars.MDIoobTree),
               list(names(mtcars[, -1]),
                    'Response'))

  mtcars.MDIoob <- MDIoob(tidy.RF, mtcars[, -1], mtcars[, 1])
  expect_equal(dim(mtcars.MDIoob), c(ncol(mtcars) - 1, 1))
  expect_equal(dimnames(mtcars.MDIoob),
               list(names(mtcars[, -1]),
                    'Response'))
})

test_that(paste('MDIoob emits error for ranger & regression tree',
                'when keep.inbag = FALSE'), {
  set.seed(42L)
  rfobj <- ranger(mpg ~ ., mtcars)
  expect_warning(tidy.RF <- tidyRF(rfobj, mtcars[, -1], mtcars[, 1]),
                 'keep.inbag = FALSE; all samples will be considered in-bag.')

  expect_error(MDIoobTree(tidy.RF, 1, mtcars[, -1], mtcars[, 1]),
               'No out-of-bag data available.')
  expect_error(MDIoob(tidy.RF, mtcars[, -1], mtcars[, 1]),
               'No out-of-bag data available.')
})

test_that(paste('MDIoob emits error for randomForest & regression tree',
                'when keep.inbag = FALSE'), {
  set.seed(42L)
  rfobj <- randomForest(mpg ~ ., mtcars)
  expect_warning(tidy.RF <- tidyRF(rfobj, mtcars[, -1], mtcars[, 1]),
                 'keep.inbag = FALSE; all samples will be considered in-bag.')

  expect_error(MDIoobTree(tidy.RF, 1, mtcars[, -1], mtcars[, 1]),
               'No out-of-bag data available.')
  expect_error(MDIoob(tidy.RF, mtcars[, -1], mtcars[, 1]),
               'No out-of-bag data available.')
})
