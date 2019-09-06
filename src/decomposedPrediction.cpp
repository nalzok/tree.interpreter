#include <Rcpp.h>

// [[Rcpp::export]]
Rcpp::List decomposedPredictionCpp(
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
    Rcpp::List decomposed_prediction_forest_list(num_X);

    for (int x = 0; x < num_X; x++) {
        Rcpp::NumericMatrix decomposed_prediction_forest(
                n_classes, num_variables);
        Rcpp::rownames(decomposed_prediction_forest)
            = Rcpp::rownames(delta_node_responses_ensemble[0]);
        Rcpp::colnames(decomposed_prediction_forest)
            = variable_names;

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

                decomposed_prediction_forest.column(split_variable)
                    = decomposed_prediction_forest.column(split_variable)
                    + delta_node_responses.column(node_id);
            }
        }

        Rcpp::NumericVector alias_decomposed_prediction_forest
            = decomposed_prediction_forest;
        alias_decomposed_prediction_forest
            = alias_decomposed_prediction_forest / num_trees;

        decomposed_prediction_forest_list[x] = decomposed_prediction_forest;
    }


    Rcpp::NumericMatrix bias(n_classes, 1);
    Rcpp::rownames(bias) = Rcpp::rownames(delta_node_responses_ensemble[0]);
    Rcpp::colnames(bias) = Rcpp::CharacterVector::create("Bias");
    for (int tree = 0; tree < num_trees; tree++) {
        const Rcpp::NumericMatrix delta_node_responses
            = delta_node_responses_ensemble[tree];
        bias += delta_node_responses.column(0);
    }
    Rcpp::NumericVector alias_bias = bias;
    alias_bias = alias_bias / num_trees;

    decomposed_prediction_forest_list.attr("bias") = bias;


    return decomposed_prediction_forest_list;
}
