#include <cstring>
#include <stack>
#include <Rcpp.h>

// [[Rcpp::export]]
Rcpp::List deltaNodeResponseCpp_randomForest(
        const Rcpp::List& rf,
        const Rcpp::DataFrame& trainX,
        const Rcpp::DataFrame& trainY,
        const Rcpp::List& inbag_counts) {
    return rf;
}

// [[Rcpp::export]]
Rcpp::List deltaNodeResponseCpp_ranger(
        const Rcpp::List& rf,
        const Rcpp::DataFrame& trainX,
        const Rcpp::DataFrame& trainY,
        const Rcpp::List& inbag_counts_ensemble) {

    // We'll use one and only one of these
    const Rcpp::NumericVector numeric_responses = trainY[0];
    const Rcpp::IntegerVector factor_responses = trainY[0];

    const int n_classes = strcmp(rf["treetype"], "Classification")
        ? 1 : Rcpp::as<Rcpp::CharacterVector>(
                factor_responses.attr("levels")).size();

    const int num_trees = rf["num.trees"];
    const Rcpp::List forest = rf["forest"];

    const Rcpp::List children_ensemble = forest["child.nodeIDs"];
    Rcpp::List left_children_ensemble(num_trees);
    Rcpp::List right_children_ensemble(num_trees);
    for (int tree = 0; tree < num_trees; tree++) {
        const Rcpp::List children = children_ensemble[tree];
        left_children_ensemble[tree] = children[0];
        right_children_ensemble[tree] = children[1];
    }

    const Rcpp::List split_var_IDs_ensemble = forest["split.varIDs"];
    const Rcpp::CharacterVector independent_variable_names =
        forest["independent.variable.names"];
    const Rcpp::CharacterVector original_variable_names = trainX.names();
    Rcpp::List split_variables_ensemble(num_trees);
    for (int tree = 0; tree < num_trees; tree++) {
        const Rcpp::IntegerVector split_var_IDs = split_var_IDs_ensemble[tree];
        Rcpp::CharacterVector variable_names(split_var_IDs.size());
        for (int i = 0; i < variable_names.size(); i++) {
            int split_var_ID = split_var_IDs[i];
            variable_names[i] = split_var_ID
                ? independent_variable_names[split_var_ID - 1] : "";
        }
        split_variables_ensemble[tree]
            = Rcpp::match(variable_names, original_variable_names) - 1;
    }
    const Rcpp::List split_values_ensemble = forest["split.values"];

    Rcpp::List delta_node_responses_ensemble(num_trees);

    for (int tree = 0; tree < num_trees; tree++) {
        const Rcpp::IntegerVector left_children
            = left_children_ensemble[tree];
        const Rcpp::IntegerVector right_children
            = right_children_ensemble[tree];
        const Rcpp::IntegerVector split_variables
            = split_variables_ensemble[tree];
        const Rcpp::NumericVector split_values = split_values_ensemble[tree];
        const Rcpp::IntegerVector inbag_counts = inbag_counts_ensemble[tree];

        const int num_nodes = split_values.size();
        Rcpp::IntegerVector node_sizes(num_nodes);

        Rcpp::NumericMatrix delta_node_responses(n_classes, num_nodes);
        Rcpp::rownames(delta_node_responses) = (n_classes == 1)
            ? Rcpp::CharacterVector("Response")
            : factor_responses.attr("levels");
        const Rcpp::IntegerVector colnames = Rcpp::seq_len(num_nodes) - 1;
        Rcpp::colnames(delta_node_responses) = colnames;

        for (int x = 0; x < inbag_counts.size(); x++) {
            const int inbag_count = inbag_counts[x];

            if (inbag_count > 0) {
                node_sizes[0] += inbag_count;

                int node_id = 0;
                while (left_children[node_id] || right_children[node_id]) {
                    const double split_value = split_values[node_id];
                    const int split_variable = split_variables[node_id];
                    const Rcpp::NumericVector split_column
                        = trainX[split_variable];
                    const double value = split_column[x];

                    node_id = (value <= split_value) ?
                        left_children[node_id] : right_children[node_id];

                    node_sizes[node_id] += inbag_count;
                }

                if (n_classes == 1) {
                    // Regression
                    delta_node_responses(0, node_id)
                        += numeric_responses[x] * inbag_count;
                } else {
                    // Classification
                    delta_node_responses(factor_responses[x] - 1, node_id)
                        += inbag_count;
                }
            }
        }

        for (int node = num_nodes - 1; node >= 0; node--) {
            const int left_child = left_children[node];
            const int right_child = right_children[node];
            if (!left_child && !right_child) {
                delta_node_responses.column(node) =
                    delta_node_responses.column(node) / node_sizes[node];
            } else {
                delta_node_responses.column(node)
                    = (delta_node_responses.column(left_child)
                            * node_sizes[left_child]
                            + delta_node_responses.column(right_child)
                            * node_sizes[right_child])
                    / (node_sizes[left_child] + node_sizes[right_child]);

                delta_node_responses.column(left_child)
                    = delta_node_responses.column(left_child)
                    - delta_node_responses.column(node);

                delta_node_responses.column(right_child)
                    = delta_node_responses.column(right_child)
                    - delta_node_responses.column(node);
            }
        }

        delta_node_responses_ensemble[tree] = delta_node_responses;
    }

    return Rcpp::List::create(
            Rcpp::Named("n.classes") = n_classes,
            Rcpp::Named("num.trees") = num_trees,
            Rcpp::Named("variable.names") = independent_variable_names,
            Rcpp::Named("left.children") = left_children_ensemble,
            Rcpp::Named("right.children") = right_children_ensemble,
            Rcpp::Named("split.variables") = split_variables_ensemble,
            Rcpp::Named("split.values") = split_values_ensemble,
            Rcpp::Named("delta.node.resp") = delta_node_responses_ensemble
            );
}
