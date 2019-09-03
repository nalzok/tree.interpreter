#' Decompose Prediction
#'
#' @export
#'
#' @useDynLib tree.interpreter
#' @importFrom Rcpp sourceCpp
decomposePrediction <- function(rf, testX) {
    UseMethod('deltaNodeResponse')
}

#' @export
decomposePrediction.randomForest <- function(rf, testX) {
    decomposePredictionCpp_randomForest(rf, testX)
}

#' @export
decomposePrediction.ranger <- function(rf, testX) {
    decomposePredictionCpp_ranger(rf, testX)
}
