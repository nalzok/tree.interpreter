# tree.interpreter
<!-- badges: start -->
[![Travis build status](https://travis-ci.org/nalzok/tree.interpreter.svg?branch=master)](https://travis-ci.org/nalzok/tree.interpreter)
[![AppVeyor build status](https://ci.appveyor.com/api/projects/status/github/nalzok/tree.interpreter?branch=master&svg=true)](https://ci.appveyor.com/project/nalzok/tree.interpreter)
[![Codecov test coverage](https://codecov.io/gh/nalzok/tree.interpreter/branch/master/graph/badge.svg)](https://codecov.io/gh/nalzok/tree.interpreter?branch=master)
<!-- badges: end -->

An R implementation of the [treeinterpreter][treeinterpreter] package on PyPI.
Each prediction is decomposed as (bias + feature\_1\_contribution + ... +
feature\_n\_contribution). This decomposition is then used to calculate the
MDI and MDI-oob feature importance measures for both trees and forests.

## Installation

Before I publish it on CRAN, you can conveniently install it with

```r
devtools::install_github('nalzok/tree.interpreter')
```

## Usage

For example, you can calculate the state-of-the-art MDI-oob feature importance
measure for **ranger**. See `vignette('MDI', package='tree.interpreter')` for
more information.

```r
library(ranger)
library(tree.interpreter)

set.seed(42L)
rfobj <- ranger(mpg ~ ., mtcars, keep.inbag = TRUE)
tidy.RF <- tidyRF(rfobj, mtcars[, -1], mtcars[, 1])
mtcars.MDIoob <- MDIoob(tidy.RF, mtcars[, -1], mtcars[, 1])
mtcars.MDIoob
```

## References

This package companies the paper [A Debiased MDI Feature Importance Measure for
Random Forests][debiased].


  [treeinterpreter]: https://pypi.org/project/treeinterpreter/
  [debiased]: https://arxiv.org/abs/1906.10845
