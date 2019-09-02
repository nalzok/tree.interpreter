#' Annotate Hierarchical Prediction
#'
#' @export
#'
#' @useDynLib tree.interpreter
#' @importFrom Rcpp sourceCpp
annotateHierarchicalPrediction <- function(rf, trainX, trainY) {

    if (any(class(rf) == 'hier.pred.annotated')) {
        return(rf)
    }

    UseMethod('annotateHierarchicalPrediction')
}

#' @export
annotateHierarchicalPrediction.randomForest <- function(rf, trainX, trainY) {
    inbag.counts <- NULL
    rf <- annotateHierarchicalPredictionCpp_randomForest(rf,
                                                         trainX,
                                                         trainY,
                                                         inbag.counts)
    class(rf) <- c(class(rf), 'hier.pred.annotated')
    return(rf)
}

#' @export
annotateHierarchicalPrediction.ranger <- function(rf, trainX, trainY) {
    inbag.counts <- rf$inbag.counts
    if (is.null(inbag.counts)) {
        warning('keep.inbag = FALSE, using all observations')
        inbag.counts <- replicate(rf$num.trees, rep(1, nrow(oldX)),
                                  simplify=FALSE)
    }
    rf <- annotateHierarchicalPredictionCpp_ranger(rf,
                                                   trainX,
                                                   trainY,
                                                   inbag.counts)

    class(rf) <- c(class(rf), 'hier.pred.annotated')
    return(rf)
}

