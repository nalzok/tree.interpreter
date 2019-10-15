#' Feature Contribution
#'
#' Contribution of each feature to the prediction of test data.
#'
#' Recall that each node in a decision tree has a prediction associated with
#' it. For regression trees, it's the average response in that node, whereas
#' in classification trees, it's the frequency of each response class, or the
#' most frequent response class in that node.
#'
#' For an observation and a tree in the forest, the contribution of each
#' feature is the sum of differences between the predictions of nodes which
#' split on it and those of their children, i.e. the sum of changes in node
#' prediction caused by spliting on it.
#'
#' For an observation and a forest, the contribution of each feature is simply
#' the average contribution across all trees. This is because the prediction of
#' a forest is the average of the predictions of its trees.
#'
#' Together with \code{trainsetBias}, it can perform a decomposition of the
#' prediction for each observation by feature importance:
#'
#' prediction(tree, X) = trainsetBias(tree) + featureContrib_1(tree, X) +
#'                 featureContrib_2(tree, X) + ... + featureContrib_p(tree, X)
#' 
#' @param tidy.RF A tidy random forest. The random forest to make predictions
#'   with.
#' @param testX A data frame. Test set features.
#' @return A list. The structure and content depends the type of the forest.
#'   \itemize{
#'     \item Regression: For each observation, a P-by-1 matrix, where P is the
#'       number of features in \code{testX}. Each row of the matrix stands for
#'       the contribution of that feature to the prediction of the response.
#'     \item Classification: For each observation, a P-by-K matrix, where P is
#'       the number of features in \code{testX}, and K is the number of
#'       response classes. Each row of the matrix stands for the contribution
#'       of that feature to the prediction of each response class.
#'   }
#'
#' @references \url{http://blog.datadive.net/interpreting-random-forests/}
#' @references \url{http://blog.datadive.net/random-forest-interpretation-with-scikit-learn/}
#' @seealso \code{\link{trainsetBias}}
#'
#' @export
#'
#' @useDynLib tree.interpreter
#' @importFrom Rcpp sourceCpp
featureContrib <- function(tidy.RF, testX) {
    featureContribCpp(tidy.RF, testX)
}

#' Trainset Bias
#'
#' For a tree in the forest, trainset bias is the prediction of its root node,
#' or the unconditional prediction of the tree, or the average response of its
#' in-bag observations.
#'
#' For a forest, the trainset bias is simply the average trainset bias across
#' all trees. This is because the prediction of a forest is the average of the
#' predictions of its trees.
#'
#' Together with \code{trainsetBias}, it can perform a decomposition of the
#' prediction for each observation by feature importance:
#'
#' prediction(tree, X) = trainsetBias(tree) + featureContrib_1(tree, X) +
#'                 featureContrib_2(tree, X) + ... + featureContrib_p(tree, X)
#'
#' @param tidy.RF A tidy random forest. The random forest to extract train
#'   set bias from.
#' @return A list. The structure and content depends the type of the forest.
#'   \itemize{
#'     \item Regression: A 1-by-1 matrix. The trainset bias for the prediction
#'       of the response.
#'     \item Classification: A 1-by-K matrix, where K is the number of
#'       response classes. Each column of the matrix stands for the trainset
#'       bias for the prediction of each response class.
#'   }
#'
#' @references \url{http://blog.datadive.net/interpreting-random-forests/}
#' @references \url{http://blog.datadive.net/random-forest-interpretation-with-scikit-learn/}
#' @seealso \code{\link{featureContrib}}
#'
#' @export
trainsetBias <- function(tidy.RF) {
    trainsetBiasCpp(tidy.RF)
}
