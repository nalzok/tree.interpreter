#' Trainset Bias
#'
#' For a tree in the forest, trainset bias is the prediction of its root node,
#' or the unconditional prediction of the tree, or the average response of the
#' samples used to train the tree.
#'
#' For a forest, the trainset bias is simply the average trainset bias across
#' all trees. This is because the prediction of a forest is the average of the
#' predictions of its trees.
#'
#' Together with \code{featureContrib(Tree)}, they can decompose the prediction
#' by feature importance:
#'
#' \deqn{prediction(MODEL, X) =
#'     trainsetBias(MODEL) +
#'     featureContrib_1(MODEL, X) + ... + featureContrib_p(MODEL, X),}
#'
#' where MODEL can be either a tree or a forest.
#'
#' @param tidy.RF A tidy random forest. The random forest to extract train
#'   set bias from.
#' @param tree An integer. The index of the tree to look at.
#'
#' @return A matrix. The content depends the type of the response.
#'   \itemize{
#'     \item Regression: A 1-by-1 matrix. The trainset bias for the prediction
#'       of the response.
#'     \item Classification: A 1-by-D matrix, where D is the number of response
#'       classes. Each column of the matrix stands for the trainset bias for
#'       the prediction of each response class.
#'   }
#'
#' @describeIn trainsetBias Trainset bias within a single tree
#'
#' @references Interpreting random forests
#'   \url{http://blog.datadive.net/interpreting-random-forests/}
#' @references Random forest interpretation with scikit-learn
#'   \url{http://blog.datadive.net/random-forest-interpretation-with-scikit-learn/}
#' @seealso \code{\link{featureContrib}}
#'
#' @examples
#' library(ranger)
#' rfobj <- ranger(Species ~ ., iris, keep.inbag=TRUE)
#' tidy.RF <- tidyRF(rfobj, iris[, -5], iris[, 5])
#' trainsetBiasTree(tidy.RF, 1)
#' trainsetBias(tidy.RF)
#'
#' @export
trainsetBiasTree <- function(tidy.RF, tree) {
    result <- trainsetBiasTreeCpp(tidy.RF, tree - 1)
    result.row.names <- 'Bias'
    result.column.names <- colnames(tidy.RF$delta.node.resp.left[[1]])
    dimnames(result) <- list(result.row.names,
                             result.column.names)
    return(result)
}

#' @describeIn trainsetBias Trainset bias within the whole forest
#' @export
trainsetBias <- function(tidy.RF) {
  Reduce(`+`, lapply(1:tidy.RF$num.trees,
                     function(tree) trainsetBiasTree(tidy.RF, tree))) /
    tidy.RF$num.trees
}
