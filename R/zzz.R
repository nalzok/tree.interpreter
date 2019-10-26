#' Random Forest Prediction Decomposition and Feature Importance Measure
#'
#' An R re-implementation of the 'treeinterpreter' package on PyPI.
#' <https://pypi.org/project/treeinterpreter/>. Each prediction can be
#' decomposed as 'prediction = bias + feature_1_contribution + ... +
#' feature_n_contribution'. This decomposition is then used to calculate the
#' Mean Decrease Impurity (MDI) and Mean Decrease Impurity using out-of-bag
#' samples (MDI-oob) feature importance measures based on the work of Li et al.
#' (2019) <arXiv:1906.10845>.
#' 
#' @section \code{tidyRF}:
#' The function \code{tidyRF} can turn a \code{randomForest} or \code{ranger}
#' object into a package-agnostic random forest object. All other functions
#' in this package operate on such a \code{tidyRF} object.
#'
#' @section The \code{featureContrib} and \code{trainsetBias} families:
#' The \code{featureContrib} and \code{trainsetBias} families can decompose the
#' prediction of regression/classification trees/forests into bias and feature
#' contribution components.
#'
#' @section The \code{MDI} and \code{MDIoob} families:
#' The \code{MDI} family can calculate the good old MDI feature importance
#' measure, which unfortunately has some feature selection bias. MDI-oob is a
#' debiased MDI feature importance measure that has achieved state-of-the-art
#' performance in feature selection for both simulated and real data. It can be
#' calculated with functions from the \code{MDIoob} family.
#' 
#' @examples
#' library(ranger)
#' rfobj <- ranger(mpg ~ ., mtcars, keep.inbag = TRUE)
#' tidy.RF <- tidyRF(rfobj, mtcars[, -1], mtcars[, 1])
#' MDIoob(tidy.RF, mtcars[, -1], mtcars[, 1])
#'
#' @docType package
#' @name tree.interpreter
NULL
