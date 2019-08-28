#' Annotate Hierarchical Prediction
#'
#' @export
#'
#' @useDynLib tree.interpreter
#' @importFrom Rcpp sourceCpp
annotateHierarchicalPrediction <- function(x, ...) {

    if (any(class(rf) == 'hier.pred.annotated')) {
        return(rf)
    }

    UseMethod('annotateHierarchicalPrediction')
}

#' @export
annotateHierarchicalPrediction.randomForest <- function(rf, oldX=NULL) {
    rf <- annotateNodeSize(rf, oldX)
    rf <- annotateHierarchicalPredictionCpp_randomForest(rf)
    class(rf) <- c(class(rf), 'hier.pred.annotated')
    return(rf)
}

#' @export
annotateHierarchicalPrediction.ranger <- function(rf, oldX=NULL) {
    rf <- annotateNodeSize(rf, oldX)
    rf <- annotateHierarchicalPredictionCpp_ranger(rf)
    class(rf) <- c(class(rf), 'hier.pred.annotated')
    return(rf)
}
        
#' Annotate Node Size
#'
#' @export
annotateNodeSize <- function(rf, oldX) {

    if (any(class(rf) == 'node.size.annotated')) {
        return(rf)
    } else if (is.null(oldX)) {
        stop('Please pass in features in the training set as oldX.')
    }

    UseMethod('annotateNodeSize')
}

#' @export
annotateNodeSize.randomForest <- function(rf, oldX) {
    rf <- annotateNodeSizeCpp_randomForest(rf, oldX)
    class(rf) <- c(class(rf), 'node.size.annotated')
    return(rf)
}

#' @export
annotateNodeSize.ranger <- function(rf, oldX) {
    rf <- annotateNodeSizeCpp_ranger(rf, oldX)
    class(rf) <- c(class(rf), 'node.size.annotated')
    return(rf)
}
