#' Mean Decrease in Impurity
#'
#' Calculate the MDI feature importance measure.
#'
#' MDI stands for Mean Decrease in Impurity. It is a widely adopted measure of
#' feature importance in random forests. In this package, we calculate MDI with
#' a new analytical expression derived by Li et al. (See references)
#'
#' See \code{vignette('MDI', package='tree.interpreter')} for more context.
#'
#' @param tidy.RF A tidy random forest. The random forest to calculate MDI
#'   from.
#' @param tree An integer. The index of the tree to look at.
#' @param trainX A data frame. Train set features, such that the \code{T}th
#'   tree is trained with \code{X[tidy.RF$inbag.counts[[T]], ]}.
#' @param trainY A data frame. Train set responses, such that the \code{T}th
#'   tree is trained with \code{Y[tidy.RF$inbag.counts[[T]], ]}.
#'
#' @return A matrix. The content depends on the type of the response.
#'   \itemize{
#'     \item Regression: A P-by-1 matrix, where P is the number of features in
#'       \code{X}. The pth row contains the MDI of feature p.
#'     \item Classification: A P-by-D matrix, where P is the number of features
#'       in \code{X} and D is the number of response classes. The dth column of
#'       the pth row contains the MDI of feature p to class d. You can get the
#'       MDI of each feature by calling \code{rowSums} on the result.
#'  }
#'
#' @describeIn MDI Mean decrease in impurity within a single tree
#'
#' @references A Debiased MDI Feature Importance Measure for Random Forests
#'   \url{https://arxiv.org/abs/1906.10845}
#' @seealso \code{\link{MDIoob}}
#' @seealso \code{vignette('MDI', package='tree.interpreter')}
#'
#' @examples
#' library(ranger)
#' rfobj <- ranger(Species ~ ., iris, keep.inbag=TRUE)
#' tidy.RF <- tidyRF(rfobj, iris[, -5], iris[, 5])
#' MDITree(tidy.RF, 1, iris[, -5], iris[, 5])
#' MDI(tidy.RF, iris[, -5], iris[, 5])
#'
#' @export
MDITree <- function(tidy.RF, tree, trainX, trainY) {
  if (tidy.RF$num.classes > 1) {
    # One-hot encode classification responses
    onehot <- matrix(0, nrow=length(trainY), ncol=tidy.RF$num.classes)
    indice <- matrix(c(1:length(trainY), trainY[1:length(trainY)]), ncol = 2)
    onehot[indice] <- 1
    trainY <- onehot
  } else {
    trainY <- matrix(trainY)
  }

  inbag.counts <- tidy.RF$inbag.counts[[tree]]
  inbag.indices <- as.logical(inbag.counts)
  X.inbag <- trainX[inbag.indices, ]
  Y.inbag <- t(trainY[inbag.indices, ])
  ftk.y <- featureContribTree(tidy.RF, tree, X.inbag) *
    rep(Y.inbag, each=ncol(X.inbag)) *
    rep(inbag.counts[inbag.indices],
        each = tidy.RF$num.classes * ncol(X.inbag))
  apply(ftk.y, 1:2, sum) / sum(inbag.counts)
}

#' @describeIn MDI Mean decrease in impurity within the whole forest
#' @export
MDI <- function(tidy.RF, trainX, trainY) {
  Reduce(`+`, lapply(1:tidy.RF$num.trees,
                     function(tree)
                         MDITree(tidy.RF, tree, trainX, trainY))) /
  tidy.RF$num.trees
}
