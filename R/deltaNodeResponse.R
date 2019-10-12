#' Calculate Delta Node Response
#'
#' @export
deltaNodeResponse <- function(rf, trainX, trainY) {
    UseMethod('deltaNodeResponse')
}

#' @export
deltaNodeResponse.randomForest <- function(rf, trainX, trainY) {
    if (is.null(rf$inbag)) {
        warning('keep.inbag = FALSE, using all observations')
        inbag.counts <- replicate(rf$num.trees, rep(1, nrow(trainX)),
                                  simplify=FALSE)
    } else {
        inbag.counts <- split(rf$inbag,
                              rep(1:ncol(rf$inbag), each = nrow(rf$inbag)))
    }

    deltaNodeResponseCpp_randomForest(rf, trainX, trainY, inbag.counts)
}

#' @export
deltaNodeResponse.ranger <- function(rf, trainX, trainY) {
    inbag.counts <- rf$inbag.counts
    if (is.null(inbag.counts)) {
        warning('keep.inbag = FALSE, using all observations')
        inbag.counts <- replicate(rf$num.trees, rep(1, nrow(trainX)),
                                  simplify=FALSE)
    }

    deltaNodeResponseCpp_ranger(rf, trainX, trainY, inbag.counts)
}
