# tree.interpreter
<!-- badges: start -->
[![Travis build status](https://travis-ci.org/nalzok/tree.interpreter.svg?branch=master)](https://travis-ci.org/nalzok/tree.interpreter)
[![AppVeyor build status](https://ci.appveyor.com/api/projects/status/github/nalzok/tree.interpreter?branch=master&svg=true)](https://ci.appveyor.com/project/nalzok/tree.interpreter)
[![Codecov test coverage](https://codecov.io/gh/nalzok/tree.interpreter/branch/master/graph/badge.svg)](https://codecov.io/gh/nalzok/tree.interpreter?branch=master)
<!-- badges: end -->

An R re-implementation of the [treeinterpreter][treeinterpreter] package on
PyPI. Each prediction can be decomposed as 'prediction = bias +
feature\_1\_contribution + ... + feature\_n\_contribution'. This decomposition
is then used to calculate the Mean Decrease Impurity (MDI) and Mean Decrease
Impurity using out-of-bag samples (MDI-oob) feature importance measures based
on the work of Li et al. (2019) <arXiv:1906.10845>.

## Installation

To install the CRAN version, run

```r
install.packages('tree.interpreter')
```

To install the latest development version, run

```r
devtools::install_github('nalzok/tree.interpreter')
```

macOS users might want to follow the set up instructions by [The Coatless
Professor][coatless] to minimize operational headaches and maximize
computational performance.

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
  [coatless]: https://thecoatlessprofessor.com/programming/cpp/r-compiler-tools-for-rcpp-on-macos/
