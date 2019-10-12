#include <cstring>
#include <stack>
#include <Rcpp.h>

Rcpp::List calculate_delta_node_responses_ensemble(
        const Rcpp::DataFrame& trainX,
        const Rcpp::NumericVector& numeric_responses,
        const Rcpp::IntegerVector& factor_responses,
        const Rcpp::List& inbag_counts_ensemble,
        const int n_classes,
        const int num_trees,
        const Rcpp::List& left_children_ensemble,
        const Rcpp::List& right_children_ensemble,
        const Rcpp::List& split_variables_ensemble,
        const Rcpp::List& split_values_ensemble);

// [[Rcpp::export]]
Rcpp::List deltaNodeResponseCpp_randomForest(
        const Rcpp::List& rf,
        const Rcpp::DataFrame& trainX,
        const Rcpp::DataFrame& trainY,
        const Rcpp::List& inbag_counts_ensemble) {

    // We'll use one and only one of these
    const Rcpp::NumericVector numeric_responses = trainY[0];
    const Rcpp::IntegerVector factor_responses = trainY[0];

    const int n_classes = strcmp(rf["type"], "classification")
        ? 1 : Rcpp::as<Rcpp::CharacterVector>(
                factor_responses.attr("levels")).size();

    const int num_trees = rf["ntree"];
    const Rcpp::List forest = rf["forest"];
    const Rcpp::IntegerVector num_nodes = forest["ndbigtree"];

    Rcpp::List left_children_ensemble(num_trees);
    Rcpp::List right_children_ensemble(num_trees);
    if (n_classes == 1) {
        // Regression
        Rcpp::IntegerMatrix left_children_ensemble_matrix
            = forest["leftDaughter"];
        Rcpp::IntegerMatrix right_children_ensemble_matrix
            = forest["rightDaughter"];
        for (int tree = 0; tree < num_trees; tree++) {
            const Rcpp::Range row_range = Rcpp::Range(0, num_nodes[tree] - 1);
            const Rcpp::Range column_range = Rcpp::Range(tree, tree);

            const Rcpp::IntegerMatrix left_children_matrix
                = left_children_ensemble_matrix(row_range, column_range);
            Rcpp::IntegerVector left_children
                = Rcpp::as<Rcpp::IntegerVector>(left_children_matrix) - 1;
            left_children[left_children < 0] = 0;
            left_children_ensemble[tree] = left_children;

            const Rcpp::IntegerMatrix right_children_matrix
                = right_children_ensemble_matrix(row_range, column_range);
            Rcpp::IntegerVector right_children
                = Rcpp::as<Rcpp::IntegerVector>(right_children_matrix) - 1;
            right_children[right_children < 0] = 0;
            right_children_ensemble[tree] = right_children;
        }
    } else {
        // Classification
        const Rcpp::IntegerVector children_ensemble = forest["treemap"];
        Rcpp::Dimension children_ensemble_dim = children_ensemble.attr("dim");
        const int num_nodes_max = children_ensemble_dim[0];
        const Rcpp::IntegerVector::const_iterator base
            = children_ensemble.begin();
        for (int tree = 0; tree < num_trees; tree++) {
            const Rcpp::IntegerVector::const_iterator left_offset
                = base + num_nodes_max*2*tree;
            Rcpp::IntegerVector left_children(left_offset,
                    left_offset + num_nodes[tree]);
            left_children = left_children - 1;
            left_children[left_children < 0] = 0;
            left_children_ensemble[tree] = left_children;

            const Rcpp::IntegerVector::const_iterator right_offset
                = left_offset + num_nodes_max;
            Rcpp::IntegerVector right_children(right_offset,
                    right_offset + num_nodes[tree]);
            right_children = right_children - 1;
            right_children[right_children < 0] = 0;
            right_children_ensemble[tree] = right_children;
        }
    }

    Rcpp::List split_variables_ensemble(num_trees);
    Rcpp::List split_values_ensemble(num_trees);
    Rcpp::IntegerMatrix split_variables_ensemble_matrix = forest["bestvar"];
    Rcpp::NumericMatrix split_values_ensemble_matrix = forest["xbestsplit"];
    for (int tree = 0; tree < num_trees; tree++) {
        const Rcpp::Range row_range = Rcpp::Range(0, num_nodes[tree] - 1);
        const Rcpp::Range column_range = Rcpp::Range(tree, tree);

        const Rcpp::IntegerMatrix split_variables_matrix
            = split_variables_ensemble_matrix(row_range, column_range);
        Rcpp::IntegerVector split_variables
            = Rcpp::as<Rcpp::IntegerVector>(split_variables_matrix) - 1;
        split_variables[split_variables < 0] = 0;
        split_variables_ensemble[tree] = split_variables;

        split_values_ensemble[tree]
            = split_values_ensemble_matrix(row_range, column_range);
    }

    const Rcpp::CharacterVector variable_names = trainX.names();

    const Rcpp::List delta_node_responses_ensemble
        = calculate_delta_node_responses_ensemble(
            trainX,
            numeric_responses,
            factor_responses,
            inbag_counts_ensemble,
            n_classes,
            num_trees,
            left_children_ensemble,
            right_children_ensemble,
            split_variables_ensemble,
            split_values_ensemble
            );

    return Rcpp::List::create(
            Rcpp::Named("n.classes") = n_classes,
            Rcpp::Named("num.trees") = num_trees,
            Rcpp::Named("left.children") = left_children_ensemble,
            Rcpp::Named("right.children") = right_children_ensemble,
            Rcpp::Named("variable.names") = variable_names,
            Rcpp::Named("split.variables") = split_variables_ensemble,
            Rcpp::Named("split.values") = split_values_ensemble,
            Rcpp::Named("delta.node.resp") = delta_node_responses_ensemble
            );
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
    Rcpp::IntegerVector reorder
        = Rcpp::match(independent_variable_names, original_variable_names) - 1;
    reorder.push_front(NA_INTEGER);

    Rcpp::List split_variables_ensemble(num_trees);
    for (int tree = 0; tree < num_trees; tree++) {
        Rcpp::IntegerVector split_var_IDs = split_var_IDs_ensemble[tree];
        split_var_IDs[Rcpp::is_na(split_var_IDs)] = 0;
        split_variables_ensemble[tree] = reorder[split_var_IDs];
    }
    const Rcpp::List split_values_ensemble = forest["split.values"];

    const Rcpp::List delta_node_responses_ensemble
        = calculate_delta_node_responses_ensemble(
            trainX,
            numeric_responses,
            factor_responses,
            inbag_counts_ensemble,
            n_classes,
            num_trees,
            left_children_ensemble,
            right_children_ensemble,
            split_variables_ensemble,
            split_values_ensemble
            );

    return Rcpp::List::create(
            Rcpp::Named("n.classes") = n_classes,
            Rcpp::Named("num.trees") = num_trees,
            Rcpp::Named("left.children") = left_children_ensemble,
            Rcpp::Named("right.children") = right_children_ensemble,
            Rcpp::Named("variable.names") = original_variable_names,
            Rcpp::Named("split.variables") = split_variables_ensemble,
            Rcpp::Named("split.values") = split_values_ensemble,
            Rcpp::Named("delta.node.resp") = delta_node_responses_ensemble
            );
}

Rcpp::List calculate_delta_node_responses_ensemble(
        const Rcpp::DataFrame& trainX,
        const Rcpp::NumericVector& numeric_responses,
        const Rcpp::IntegerVector& factor_responses,
        const Rcpp::List& inbag_counts_ensemble,
        const int n_classes,
        const int num_trees,
        const Rcpp::List& left_children_ensemble,
        const Rcpp::List& right_children_ensemble,
        const Rcpp::List& split_variables_ensemble,
        const Rcpp::List& split_values_ensemble) {

    Rcpp::List delta_node_responses_ensemble(num_trees);

    const Rcpp::CharacterVector colnames = (n_classes == 1)
              ? Rcpp::CharacterVector("Response")
              : factor_responses.attr("levels");

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

        Rcpp::NumericMatrix delta_node_responses(num_nodes, n_classes);
        const Rcpp::IntegerVector rownames = Rcpp::seq_len(num_nodes) - 1;
        Rcpp::rownames(delta_node_responses) = rownames;
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
                    delta_node_responses(node_id, 0)
                        += numeric_responses[x] * inbag_count;
                } else {
                    // Classification
                    delta_node_responses(node_id, factor_responses[x] - 1)
                        += inbag_count;
                }
            }
        }

        for (int node = num_nodes - 1; node >= 0; node--) {
            const int left_child = left_children[node];
            const int right_child = right_children[node];
            if (!left_child && !right_child) {
                if (node_sizes[node] > 0) {
                    delta_node_responses.row(node) =
                        delta_node_responses.row(node) / node_sizes[node];
                }
            } else {
                if (node_sizes[left_child] + node_sizes[right_child] > 0) {
                    delta_node_responses.row(node)
                        = (delta_node_responses.row(left_child)
                                * node_sizes[left_child]
                                + delta_node_responses.row(right_child)
                                * node_sizes[right_child])
                        / (node_sizes[left_child] + node_sizes[right_child]);

                    delta_node_responses.row(left_child)
                        = delta_node_responses.row(left_child)
                        - delta_node_responses.row(node);

                    delta_node_responses.row(right_child)
                        = delta_node_responses.row(right_child)
                        - delta_node_responses.row(node);
                }
            }
        }

        delta_node_responses_ensemble[tree] = delta_node_responses;
    }

    return delta_node_responses_ensemble;
}

