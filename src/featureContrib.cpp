#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
arma::cube featureContribTreeCpp(
        const Rcpp::List & tidyRF,
        const int tree,
        const Rcpp::DataFrame & X) {

    const int num_classes = tidyRF["num.classes"];
    const Rcpp::CharacterVector variable_names = tidyRF["feature.names"];
    const Rcpp::List left_children_ensemble = tidyRF["left.children"];
    const Rcpp::List right_children_ensemble = tidyRF["right.children"];
    const Rcpp::List split_variables_ensemble = tidyRF["split.variables"];
    const Rcpp::List split_values_ensemble = tidyRF["split.values"];
    const Rcpp::List delta_node_responses_left_ensemble
        = tidyRF["delta.node.resp.left"];
    const Rcpp::List delta_node_responses_right_ensemble
        = tidyRF["delta.node.resp.right"];

    const arma::ivec left_children = left_children_ensemble[tree];
    const arma::ivec right_children = right_children_ensemble[tree];
    const arma::ivec split_variables = split_variables_ensemble[tree];
    const arma::vec split_values = split_values_ensemble[tree];
    const arma::mat delta_node_responses_left
        = delta_node_responses_left_ensemble[tree];
    const arma::mat delta_node_responses_right
        = delta_node_responses_right_ensemble[tree];

    const int num_X = X.nrows();
    const int num_variables = X.size();

    arma::cube feature_contribution_tree_cube(
            num_variables, num_classes,
            num_X, arma::fill::zeros);

    for (int x = 0; x < num_X; x++) {
        int node_id = 0;
        while (left_children[node_id] || right_children[node_id]) {
            const double split_value = split_values[node_id];
            const int split_variable = split_variables[node_id];
            const arma::vec split_column = X[split_variable];
            const double value = split_column[x];

            if (value <= split_value) {
                feature_contribution_tree_cube.subcube(
                        split_variable, 0, x,
                        split_variable, num_classes - 1, x)
                    += delta_node_responses_left.row(node_id);
                node_id = left_children[node_id];
            } else {
                feature_contribution_tree_cube.subcube(
                        split_variable, 0, x,
                        split_variable, num_classes - 1, x)
                    += delta_node_responses_right.row(node_id);
                node_id = right_children[node_id];
            }
        }
    }

    return feature_contribution_tree_cube;
}

// [[Rcpp::export]]
arma::mat trainsetBiasTreeCpp(
        const Rcpp::List & tidyRF,
        const int tree) {

    const Rcpp::List node_responses_ensemble = tidyRF["node.resp"];
    const arma::mat node_responses = node_responses_ensemble[tree];
    const arma::mat trainset_bias_tree(node_responses.row(0));

    return trainset_bias_tree;
}
