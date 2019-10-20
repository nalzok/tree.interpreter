#' Feature Contribution
#'
#' Contribution of each feature to the prediction.
#'
#' Recall that each node in a decision tree has a prediction associated with
#' it. For regression trees, it's the average response in that node, whereas
#' in classification trees, it's the frequency of each response class, or the
#' most frequent response class in that node.
#'
#' For a tree in the forest, the contribution of each feature to the prediction
#' of a sample is the sum of differences between the predictions of nodes which
#' split on the feature and those of their children, i.e. the sum of changes in
#' node prediction caused by spliting on the feature. This is the calculated by
#' \code{featureContribTree}.
#'
#' For a forest, the contribution of each feature to the prediction if a sample
#' is the average contribution across all trees in the forest. This is because
#' the prediction of a forest is the average of the predictions of its trees.
#' This is calculated by \code{featureContrib}.
#'
#' Together with \code{trainsetBias(Tree)}, they can decompose the prediction
#' by feature importance:
#'
#' \deqn{prediction(MODEL, X) =
#'     trainsetBias(MODEL) +
#'     featureContrib_1(MODEL, X) + ... + featureContrib_p(MODEL, X),}
#'
#' where MODEL can be either a tree or a forest.
#' 
#' @param tidy.RF A tidy random forest. The random forest to make predictions
#'   with.
#' @param tree An integer. The index of the tree to look at.
#' @param X A data frame. Samples to be predicted.
#' @return A cube (3D array). The content depends on the type of the response.
#'   \itemize{
#'     \item Regression: A P-by-1-by-N array, where P is the number of features
#'       in \code{X}, and N the number of samples in \code{X}. The pth row of
#'       the nth slice stands for the contribution of feature p to the
#'       prediction for response n.
#'     \item Classification: A P-by-D-by-N array, where P is the number of
#'       features in \code{X}, D is the number of response classes, and N is
#'       the number of samples in \code{X}. The pth row of the nth slice stands
#'       for the contribution of feature p to the prediction of each response
#'       class for response n.
#'   }
#'
#' @describeIn featureContrib Feature contribution within a single tree
#'
#' @references Interpreting random forests
#'   \url{http://blog.datadive.net/interpreting-random-forests/}
#' @references Random forest interpretation with scikit-learn
#'   \url{http://blog.datadive.net/random-forest-interpretation-with-scikit-learn/}
#' @seealso \code{\link{trainsetBias}}
#'
#' @export
#'
#' @useDynLib tree.interpreter
#' @importFrom Rcpp sourceCpp
featureContribTree <- function(tidy.RF, tree, X) {
    result <- featureContribTreeCpp(tidy.RF, tree - 1, X)
    result.row.names <- tidy.RF$feature.names
    result.column.names <- colnames(tidy.RF$delta.node.resp.left[[1]])
    result.slice.names <- 1:nrow(X)
    dimnames(result) <- list(result.row.names,
                             result.column.names,
                             result.slice.names)
    return(result)
}

#' @describeIn featureContrib Feature contribution within the whole forest
#' @export
featureContrib <- function(tidy.RF, X) {
  Reduce(`+`, lapply(1:tidy.RF$num.trees,
                     function(tree) featureContribTree(tidy.RF, tree, X))) /
    tidy.RF$num.trees
}


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
