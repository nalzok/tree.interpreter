#' Decomposed Prediction
#'
#' @export
#'
#' @useDynLib tree.interpreter
#' @importFrom Rcpp sourceCpp
decomposedPrediction <- function(rf, testX) {
    decomposedPredictionCpp(rf, testX)
}
