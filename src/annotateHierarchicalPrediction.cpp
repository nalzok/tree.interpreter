#include <cstring>
#include <stack>
#include <Rcpp.h>

// [[Rcpp::export]]
Rcpp::List annotateHierarchicalPredictionCpp_randomForest(
        const Rcpp::List& rf,
        const Rcpp::DataFrame& oldX) {
    return rf;
}


// [[Rcpp::export]]
Rcpp::List annotateHierarchicalPredictionCpp_ranger(
        const Rcpp::List& rf,
        const Rcpp::DataFrame& oldX) {

    const int n_classes = strcmp(rf["treetype"], "Classification")
        ? 1 : Rcpp::as<Rcpp::Dimension>(Rcpp::as<Rcpp::IntegerMatrix>(
                    rf["confusion.matrix"]).attr("dim"))[0];

    const int num_trees = rf["num.trees"];
    const Rcpp::List forest = rf["forest"];
    const Rcpp::List children_forest = forest["child.nodeIDs"];
    const Rcpp::List split_var_IDs_forest = forest["split.varIDs"];
    const Rcpp::List split_values_forest = forest["split.values"];
    const Rcpp::CharacterVector independent_variable_names =
        forest["independent.variable.names"];

    Rcpp::List hierarchical_predictions_forest(num_trees);

    for (int tree = 0; tree < num_trees; tree++) {
        const Rcpp::List children = children_forest[tree];
        const Rcpp::IntegerVector left_children = children[0];
        const Rcpp::IntegerVector right_children = children[1];
        const Rcpp::IntegerVector split_var_IDs = split_var_IDs_forest[tree];
        const Rcpp::NumericVector split_values = split_values_forest[tree];

        const int num_nodes = split_values.size();
        Rcpp::IntegerVector node_sizes(num_nodes);
        node_sizes[0] = oldX.nrows();

        Rcpp::NumericMatrix terminal_responses(num_nodes, n_classes);

        for (int x = 0; x < oldX.nrows(); x++) {
            int node_id = 0;
            while (left_children[node_id] || right_children[node_id]) {

                const int split_var_ID = split_var_IDs[node_id] - 1;
                const double split_value = split_values[node_id];

                const std::string independent_variable_name =
                    Rcpp::as<std::string>(
                            independent_variable_names[split_var_ID]);
                const Rcpp::NumericVector split_column =
                    oldX[independent_variable_name];
                const double value = split_column[x];

                node_id = (value <= split_value) ?
                    left_children[node_id] : right_children[node_id];

                node_sizes[node_id]++;
            }

            Rcpp::NumericVector terminal_response
                = terminal_responses(node_id, Rcpp::_);
            if (n_classes == 1) {
                // Regression
                terminal_responses(node_id, 0) += split_values[node_id];
            } else {
                // Classification
                terminal_responses(node_id, split_values[node_id] - 1)++;
            }
        }

        for (int i = 0; i < num_nodes; i++) {
            for (int j = 0; j < n_classes; j++) {
                terminal_responses(i, j) /= node_sizes[i];
            }
        }

        Rcpp::NumericMatrix hierarchical_predictions(num_nodes, n_classes);
        std::fill(hierarchical_predictions.begin(),
                hierarchical_predictions.end(),
                NA_REAL);

        for (int node = num_nodes - 1; node >= 0; node--) {
            const int left_child = left_children[node];
            const int right_child = right_children[node];
            if (!left_child && !right_child) {
                hierarchical_predictions(node, Rcpp::_) =
                    terminal_responses(node, Rcpp::_);
            } else {
                hierarchical_predictions(node, Rcpp::_) =
                    (hierarchical_predictions(left_child, Rcpp::_)
                     * node_sizes[left_child]
                     + hierarchical_predictions(right_child, Rcpp::_)
                     * node_sizes[right_child])
                    / (node_sizes[left_child] + node_sizes[right_child]);
            }
        }

        hierarchical_predictions_forest[tree] = hierarchical_predictions;
    }

    forest["hierarchical.predictions"] = hierarchical_predictions_forest;
    rf["forest"] = forest;

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
