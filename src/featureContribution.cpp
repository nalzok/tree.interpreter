#include <Rcpp.h>

// [[Rcpp::export]]
Rcpp::List featureContributionCpp(
        const Rcpp::List delta_node_responses,
        const Rcpp::DataFrame testX) {

    const int n_classes = delta_node_responses["n.classes"];
    const int num_trees = delta_node_responses["num.trees"];
    const Rcpp::CharacterVector variable_names
        = delta_node_responses["variable.names"];
    const Rcpp::List left_children_ensemble
        = delta_node_responses["left.children"];
    const Rcpp::List right_children_ensemble
        = delta_node_responses["right.children"];
    const Rcpp::List split_variables_ensemble
        = delta_node_responses["split.variables"];
    const Rcpp::List split_values_ensemble
        = delta_node_responses["split.values"];
    const Rcpp::List delta_node_responses_ensemble
        = delta_node_responses["delta.node.resp"];

    const int num_X = testX.nrows();
    const int num_variables = testX.size();
    Rcpp::List feature_contribution_forest_list(num_X);

    for (int x = 0; x < num_X; x++) {
        Rcpp::NumericMatrix feature_contribution_forest(
                num_variables, n_classes);
        Rcpp::rownames(feature_contribution_forest)
            = variable_names;
        Rcpp::colnames(feature_contribution_forest)
            = Rcpp::colnames(delta_node_responses_ensemble[0]);

        for (int tree = 0; tree < num_trees; tree++) {
            const Rcpp::IntegerVector left_children
                = left_children_ensemble[tree];
            const Rcpp::IntegerVector right_children
                = right_children_ensemble[tree];
            const Rcpp::IntegerVector split_variables
                = split_variables_ensemble[tree];
            const Rcpp::NumericVector split_values
                = split_values_ensemble[tree];
            const Rcpp::NumericMatrix delta_node_responses
                = delta_node_responses_ensemble[tree];

            int node_id = 0;
            while (left_children[node_id] || right_children[node_id]) {
                const double split_value = split_values[node_id];
                const int split_variable = split_variables[node_id];
                const Rcpp::NumericVector split_column
                    = testX[split_variable];
                const double value = split_column[x];

                node_id = (value <= split_value) ?
                    left_children[node_id] : right_children[node_id];

                feature_contribution_forest.row(split_variable)
                    = feature_contribution_forest.row(split_variable)
                    + delta_node_responses.row(node_id);
            }
        }

        Rcpp::NumericVector alias_feature_contribution_forest
             = feature_contribution_forest;
        alias_feature_contribution_forest
             = alias_feature_contribution_forest / num_trees;

        feature_contribution_forest_list[x] = feature_contribution_forest;
    }

    return feature_contribution_forest_list;
}

// [[Rcpp::export]]
Rcpp::NumericMatrix trainsetBiasCpp(const Rcpp::List delta_node_responses) {
    const int n_classes = delta_node_responses["n.classes"];
    const int num_trees = delta_node_responses["num.trees"];
    const Rcpp::List delta_node_responses_ensemble
        = delta_node_responses["delta.node.resp"];

    Rcpp::NumericMatrix trainset_bias(1, n_classes);
    Rcpp::rownames(trainset_bias) = Rcpp::CharacterVector::create("Bias");
    Rcpp::colnames(trainset_bias)
        = Rcpp::colnames(delta_node_responses_ensemble[0]);

    for (int tree = 0; tree < num_trees; tree++) {
        const Rcpp::NumericMatrix delta_node_responses
            = delta_node_responses_ensemble[tree];
        trainset_bias += delta_node_responses.row(0);
    }
    Rcpp::NumericVector alias_trainset_bias = trainset_bias;
    alias_trainset_bias = alias_trainset_bias / num_trees;

    return trainset_bias;
}
