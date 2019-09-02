#' Annotate Hierarchical Prediction
#'
#' @export
#'
#' @useDynLib tree.interpreter
#' @importFrom Rcpp sourceCpp
annotateHierarchicalPrediction <- function(rf, oldX) {

    if (any(class(rf) == 'hier.pred.annotated')) {
        return(rf)
    } else if (is.null(oldX)) {
        stop('Please pass in features in the _training_ set as oldX.')
    }

    UseMethod('annotateHierarchicalPrediction')
}

#' @export
annotateHierarchicalPrediction.randomForest <- function(rf, oldX) {
    rf <- annotateHierarchicalPredictionCpp_randomForest(rf, oldX)
    class(rf) <- c(class(rf), 'hier.pred.annotated')
    return(rf)
}

#' @export
annotateHierarchicalPrediction.ranger <- function(rf, oldX) {
    rf <- annotateHierarchicalPredictionCpp_ranger(rf, oldX)
    class(rf) <- c(class(rf), 'hier.pred.annotated')
    return(rf)
}

