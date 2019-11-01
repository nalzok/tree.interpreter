#' Tidy Random Forest
#'
#' Converts random forest objects from various libraries into a common
#' structure, i.e. a ``tidy'' random forest, calculating absent auxiliary
#' information on demand, in order to provide a uniform interface for other
#' \code{tree.interpreter} functions. Note that the output is of a private
#' format, and is subject to change.
#'
#' @param rfobj A random forest object. Currently, supported libraries include
#'   \code{randomForest} and \code{ranger}.
#' @param trainX A data frame. Train set features.
#' @param trainY A data frame. Train set responses.
#'
#' @return A list of class \code{tidyRF}, with entries:
#'   \describe{
#'     \item{\code{num.trees}}{An integer. Number of trees.}
#'     \item{\code{feature.names}}{A vector. Names of features.}
#'     \item{\code{num.classes}}{An integer. For classification trees, number
#'       of classes of the response. For regression trees, this field will
#'       always be 1.}
#'     \item{\code{class.names}}{A vector. Names of response classes.}
#'     \item{\code{inbag.counts}}{A list. For each tree, a vector of the number
#'        of times the observations are in-bag in the trees.}
#'     \item{\code{left.children}}{A list. For each tree, a vector of the left
#'       children of its nodes.}
#'     \item{\code{right.children}}{A list. For each tree, a vector of the
#'       right children of its nodes.}
#'     \item{\code{split.features}}{A list. For each tree, a vector of the
#'       indices of features used at its nodes. Indices start from 0. A value
#'       of 0 means the node is terminal. This does not cause ambiguity,
#'       because the root node will never be a child of other nodes.}
#'     \item{\code{split.values}}{A list. For each tree, a vector of the
#'       values of features used at its nodes.}
#'     \item{\code{node.sizes}}{A list. For each tree, a vector of the sizes
#'       of its nodes.}
#'     \item{\code{node.resp}}{A list. For each tree, a vector of the responses
#'       of its nodes.}
#'     \item{\code{delta.node.resp.left}}{A list. For each tree, a vector of
#'       the difference between the responses of the left children of its nodes
#'       and themselves.}
#'     \item{\code{delta.node.resp.right}}{A list. For each tree, a vector of
#'       the difference between the responses of the right children of its
#'       nodes and themselves.}
#'   }
#'
#' @examples
#' library(ranger)
#' rfobj <- ranger(Species ~ ., iris, keep.inbag=TRUE)
#' tidy.RF <- tidyRF(rfobj, iris[, -5], iris[, 5])
#' str(tidy.RF, max.level=1)
#'
#' @export
tidyRF <- function(rfobj, trainX, trainY) {
    UseMethod('tidyRF')
}

#' @export
tidyRF.randomForest <- function(rfobj, trainX, trainY) {
    if (!is.null(rfobj$inbag)) {
        inbag.counts <- split(rfobj$inbag,
                              rep(1:ncol(rfobj$inbag),
                                  each = nrow(rfobj$inbag)))
    } else {
        warning('keep.inbag = FALSE; all samples will be considered in-bag.')
        inbag.counts <- replicate(rfobj$ntree, rep(1, nrow(trainX)),
                                  simplify=FALSE)
    }

    result <- tidyRFCpp_randomForest(rfobj, trainX, trainY, inbag.counts)
    class(result) <- 'tidyRF'
    return(result)
}

#' @export
tidyRF.ranger <- function(rfobj, trainX, trainY) {
    inbag.counts <- rfobj$inbag.counts
    if (is.null(inbag.counts)) {
        warning('keep.inbag = FALSE; all samples will be considered in-bag.')
        inbag.counts <- replicate(rfobj$num.trees, rep(1, nrow(trainX)),
                                  simplify=FALSE)
    }

    result <- tidyRFCpp_ranger(rfobj, trainX, trainY, inbag.counts)
    class(result) <- 'tidyRF'
    return(result)
}
