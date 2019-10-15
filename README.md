# tree.interpreter

An R implementation of the [treeinterpreter][treeinterpreter] package on PyPI.
Each prediction is decomposed as (bias + feature\_1\_contribution + ... +
feature\_n\_contribution). This decomposition can then be used as a measure of
feature importance.

Before I publish it on CRAN, you can conveniently install it with

```r
devtools::install_github('nalzok/tree.interpreter')
```

  [treeinterpreter]: https://pypi.org/project/treeinterpreter/
