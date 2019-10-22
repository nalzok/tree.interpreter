#' Debiased Mean Decrease in Impurity
#'
#' Calculate the MDI-oob feature importance measure.
#'
#' It has long been known that MDI incorrectly assigns high importance to noisy
#' features, leading to systematic bias in feature selection. To address this
#' issue, Li et al. proposed a debiased MDI feature importance measure using
#' out-of-bag samples, called MDI-oob, which has achieved state-of-the-art
#' performance in feature selection for both simulated and real data.
#'
#' See \code{vignette('MDI', package='tree.interpreter')} for more context.
#'
#' @param tidy.RF A tidy random forest. The random forest to calculate MDI-oob
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
#'       \code{X}. The pth row contains the MDI-oob of feature p.
#'     \item Classification: A P-by-D matrix, where P is the number of features
#'       in \code{X} and D is the number of response classes. The dth column of
#'       the pth row contains the MDI-oob of feature p to class d. You can get
#'       the MDI-oob of each feature by calling \code{rowSums} on the result.
#'  }
#'
#' @describeIn MDIoob Debiased mean decrease in impurity within a single tree
#'
#' @references A Debiased MDI Feature Importance Measure for Random Forests
#'   \url{https://arxiv.org/abs/1906.10845}
#' @seealso \code{\link{MDI}}
#' @seealso \code{vignette('MDI', package='tree.interpreter')}
#'
#' @examples
#' library(ranger)
#' rfobj <- ranger(Species ~ ., iris, keep.inbag=TRUE)
#' tidy.RF <- tidyRF(rfobj, iris[, -5], iris[, 5])
#' MDIoobTree(tidy.RF, 1, iris[, -5], iris[, 5])
#' MDIoob(tidy.RF, iris[, -5], iris[, 5])
#'
#' @export
MDIoobTree <- function(tidy.RF, tree, trainX, trainY) {
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
  indices.oob <- !as.logical(inbag.counts)
  X.oob <- trainX[indices.oob, ]
  if (!nrow(X.oob)) {
      stop('No out-of-bag data available.')
  }
  Y.oob <- t(trainY[indices.oob, ])
  ftk.y <- featureContribTree(tidy.RF, tree, X.oob) *
    rep(Y.oob, each=ncol(X.oob))
  apply(ftk.y, 1:2, sum) / sum(indices.oob)
}

#' @describeIn MDIoob Debiased mean decrease in impurity within the whole
#'   forest
#' @export
MDIoob <- function(tidy.RF, trainX, trainY) {
  Reduce(`+`, lapply(1:tidy.RF$num.trees,
                     function(tree)
                         MDIoobTree(tidy.RF, tree, trainX, trainY))) /
  tidy.RF$num.trees
}
