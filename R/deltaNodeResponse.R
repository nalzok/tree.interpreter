#' Calculate Delta Node Response
#'
#' @export
deltaNodeResponse <- function(rf, trainX, trainY) {
    UseMethod('deltaNodeResponse')
}

#' @export
deltaNodeResponse.randomForest <- function(rf, trainX, trainY) {
    inbag.counts <- NULL
    deltaNodeResponseCpp_randomForest(rf, trainX, trainY, inbag.counts)
}

#' @export
deltaNodeResponse.ranger <- function(rf, trainX, trainY) {
    inbag.counts <- rf$inbag.counts
    if (is.null(inbag.counts)) {
        warning('keep.inbag = FALSE, using all observations')
        inbag.counts <- replicate(rf$num.trees, rep(1, nrow(oldX)),
                                  simplify=FALSE)
    }

    deltaNodeResponseCpp_ranger(rf, trainX, trainY, inbag.counts)
}
