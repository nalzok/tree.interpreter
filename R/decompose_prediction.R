#' Annotate Hierarchical Prediction
#'
#' @export
#'
#' @useDynLib tree.interpreter
#' @importFrom Rcpp sourceCpp
annotateHierarchicalPrediction <- function(x, ...) {
    UseMethod('annotateHierarchicalPrediction')
}

#' @export
annotateHierarchicalPrediction.randomForest <- function(rf, oldX) {
    annotateHierarchicalPrediction_randomForest(rf, oldX)
}

#' @export
annotateHierarchicalPrediction.ranger <- function(rf, oldX) {
    annotateHierarchicalPrediction_ranger(rf, oldX)
}
