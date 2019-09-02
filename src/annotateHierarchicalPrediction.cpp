#include <cstring>
#include <stack>
#include <Rcpp.h>

// [[Rcpp::export]]
Rcpp::List annotateHierarchicalPredictionCpp_randomForest(
        const Rcpp::List& rf,
        const Rcpp::DataFrame& trainX,
        const Rcpp::DataFrame& trainY,
        const Rcpp::List& inbag_counts) {
    return rf;
}


// [[Rcpp::export]]
Rcpp::List annotateHierarchicalPredictionCpp_ranger(
        const Rcpp::List& rf,
        const Rcpp::DataFrame& trainX,
        const Rcpp::DataFrame& trainY,
        const Rcpp::List& inbag_counts_ensemble) {

    const int n_classes = strcmp(rf["treetype"], "Classification")
        ? 1 : Rcpp::as<Rcpp::Dimension>(Rcpp::as<Rcpp::IntegerMatrix>(
                    rf["confusion.matrix"]).attr("dim"))[0];

    const int num_trees = rf["num.trees"];
    const Rcpp::List forest = rf["forest"];
    const Rcpp::List children_ensemble = forest["child.nodeIDs"];
    const Rcpp::List split_var_IDs_ensemble = forest["split.varIDs"];
    const Rcpp::List split_values_ensemble = forest["split.values"];
    const Rcpp::CharacterVector independent_variable_names =
        forest["independent.variable.names"];

    // We'll use one and only one of these
    Rcpp::NumericVector numeric_responses = trainY[0];
    Rcpp::NumericVector factor_responses = trainY[0];

    Rcpp::List hierarchical_predictions_ensemble(num_trees);

    for (int tree = 0; tree < num_trees; tree++) {
        const Rcpp::List children = children_ensemble[tree];
        const Rcpp::IntegerVector left_children = children[0];
        const Rcpp::IntegerVector right_children = children[1];
        const Rcpp::IntegerVector split_var_IDs = split_var_IDs_ensemble[tree];
        const Rcpp::NumericVector split_values = split_values_ensemble[tree];
        const Rcpp::IntegerVector inbag_counts = inbag_counts_ensemble[tree];

        const int num_nodes = split_values.size();
        Rcpp::IntegerVector node_sizes(num_nodes);

        Rcpp::NumericMatrix hierarchical_predictions(num_nodes, n_classes);

        for (int x = 0; x < inbag_counts.size(); x++) {
            const int inbag_count = inbag_counts[x];

            if (inbag_count > 0) {
                node_sizes[0] += inbag_count;

                int node_id = 0;
                while (left_children[node_id] || right_children[node_id]) {
                    const int split_var_ID = split_var_IDs[node_id] - 1;
                    const double split_value = split_values[node_id];

                    const std::string independent_variable_name =
                        Rcpp::as<std::string>(
                                independent_variable_names[split_var_ID]);
                    const Rcpp::NumericVector split_column =
                        trainX[independent_variable_name];
                    const double value = split_column[x];

                    node_id = (value <= split_value) ?
                        left_children[node_id] : right_children[node_id];

                    node_sizes[node_id] += inbag_count;
                }

                Rcpp::NumericVector hierarchical_prediction
                    = hierarchical_predictions(node_id, Rcpp::_);
                if (n_classes == 1) {
                    // Regression
                    hierarchical_predictions(node_id, 0)
                        += numeric_responses[x] * inbag_count;
                } else {
                    // Classification
                    hierarchical_predictions(node_id, factor_responses[x] - 1)
                        += inbag_count;
                }
            }
        }

        for (int node = num_nodes - 1; node >= 0; node--) {
            const int left_child = left_children[node];
            const int right_child = right_children[node];
            if (!left_child && !right_child) {
                hierarchical_predictions(node, Rcpp::_) =
                    hierarchical_predictions(node, Rcpp::_) / node_sizes[node];
            } else {
                hierarchical_predictions(node, Rcpp::_)
                    = (hierarchical_predictions(left_child, Rcpp::_)
                    * node_sizes[left_child]
                    + hierarchical_predictions(right_child, Rcpp::_)
                    * node_sizes[right_child])
                    / (node_sizes[left_child] + node_sizes[right_child]);

                hierarchical_predictions(left_child, Rcpp::_)
                    = hierarchical_predictions(left_child, Rcpp::_)
                    - hierarchical_predictions(node, Rcpp::_);

                hierarchical_predictions(right_child, Rcpp::_)
                    = hierarchical_predictions(right_child, Rcpp::_)
                    - hierarchical_predictions(node, Rcpp::_);
            }
        }

        hierarchical_predictions_ensemble[tree] = hierarchical_predictions;
    }

    forest["hierarchical.predictions"] = hierarchical_predictions_ensemble;
    rf["forest"] = forest;

    return rf;
}

/*
// [[Rcpp::export]]
Rcpp::Matrix decomposePredict(
const Rcpp::List rf,
const Rcpp::Matrix newX) {
return Rcpp::Matrix();
}
*/
