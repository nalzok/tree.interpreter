#' Feature Contribution
#'
#' @export
#'
#' @useDynLib tree.interpreter
#' @importFrom Rcpp sourceCpp
featureContribution <- function(delta.node.responses, testX) {
    featureContributionCpp(delta.node.responses, testX)
}

#' Bias
#'
#' @export
trainsetBias <- function(delta.node.responses) {
    trainsetBiasCpp(delta.node.responses)
}
