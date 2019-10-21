# tree.interpreter
<!-- badges: start -->
[![Travis build status](https://travis-ci.org/nalzok/tree.interpreter.svg?branch=master)](https://travis-ci.org/nalzok/tree.interpreter)
[![Codecov test coverage](https://codecov.io/gh/nalzok/tree.interpreter/branch/master/graph/badge.svg)](https://codecov.io/gh/nalzok/tree.interpreter?branch=master)
[![AppVeyor build status](https://ci.appveyor.com/api/projects/status/github/nalzok/tree.interpreter?branch=master&svg=true)](https://ci.appveyor.com/project/nalzok/tree.interpreter)
<!-- badges: end -->

An R implementation of the [treeinterpreter][treeinterpreter] package on PyPI.
Each prediction is decomposed as (bias + feature\_1\_contribution + ... +
feature\_n\_contribution). This decomposition can then be used as a measure of
feature importance.

Before I publish it on CRAN, you can conveniently install it with

```r
devtools::install_github('nalzok/tree.interpreter')
```

  [treeinterpreter]: https://pypi.org/project/treeinterpreter/
