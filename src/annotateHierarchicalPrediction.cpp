#include <cstring>
#include <Rcpp.h>

// [[Rcpp::export]]
Rcpp::List annotateNodeSizeCpp_randomForest(
        const Rcpp::List& rf,
        const Rcpp::DataFrame& oldX) {
    return rf;
}

// [[Rcpp::export]]
Rcpp::List annotateHierarchicalPredictionCpp_randomForest(
        const Rcpp::List& rf) {
    return rf;
}

// [[Rcpp::export]]
Rcpp::List annotateNodeSizeCpp_ranger(
        const Rcpp::List& rf,
        const Rcpp::DataFrame& oldX) {

    if (strcmp(rf["treetype"], "Regression")) {
        Rcpp::stop("Only regression trees are supported at the moment.");
    }

    const int num_trees = rf["num.trees"];
    Rcpp::List forest = rf["forest"];
    Rcpp::List children_forest = forest["child.nodeIDs"];
    Rcpp::List split_var_IDs_forest = forest["split.varIDs"];
    Rcpp::List split_values_forest = forest["split.values"];
    Rcpp::CharacterVector independent_variable_names =
        forest["independent.variable.names"];

    Rcpp::List node_sizes_forest(num_trees);

    for (int tree = 0; tree < num_trees; tree++) {
        Rcpp::List children = children_forest[tree];
        Rcpp::IntegerVector left_children = children[0];
        Rcpp::IntegerVector right_children = children[1];
        Rcpp::IntegerVector split_var_IDs = split_var_IDs_forest[tree];
        Rcpp::NumericVector split_values = split_values_forest[tree];

        Rcpp::IntegerVector node_sizes(split_values.size());
        node_sizes[0] = oldX.nrows();

        for (int x = 0; x < oldX.nrows(); x++) {
            int node_id = 0;
            while (left_children[node_id] || right_children[node_id]) {

                const int split_var_ID = split_var_IDs[node_id] - 1;
                const double split_value = split_values[split_var_ID];

                const std::string independent_variable_name =
                    Rcpp::as<std::string>(
                            independent_variable_names[split_var_ID]
                            );
                const Rcpp::NumericVector split_column =
                    oldX[independent_variable_name];
                const double value = split_column[x];

                if (value <= split_value) {
                    node_id = left_children[node_id];
                } else {
                    node_id = right_children[node_id];
                }

                node_sizes[node_id]++;
            }
        }

        node_sizes_forest[tree] = node_sizes;
    }

    forest["node.sizes"] = node_sizes_forest;
    rf["forest"] = forest;

    return rf;
}

// [[Rcpp::export]]
Rcpp::List annotateHierarchicalPredictionCpp_ranger(
        const Rcpp::List& rf) {
    return rf;
}

/*
// [[Rcpp::export]]
Rcpp::Matrix decomposePredict(
const Rcpp::List annotated_forest,
const Rcpp::Matrix newX) {
return Rcpp::Matrix();
}
*/
