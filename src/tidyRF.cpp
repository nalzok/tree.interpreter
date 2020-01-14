#include <cstring>
#include <Rcpp.h>

Rcpp::List calculate_auxiliary_information(
        const Rcpp::DataFrame & trainX,
        const Rcpp::NumericVector & numeric_responses,
        const Rcpp::IntegerVector & factor_responses,
        const Rcpp::List & inbag_counts_ensemble,
        const int num_classes,
        const int num_trees,
        const Rcpp::List & left_children_ensemble,
        const Rcpp::List & right_children_ensemble,
        const Rcpp::List & split_variables_ensemble,
        const Rcpp::List & split_values_ensemble);

// [[Rcpp::export]]
Rcpp::List tidyRFCpp_randomForest(
        const Rcpp::List & rfobj,
        const Rcpp::DataFrame & trainX,
        const Rcpp::DataFrame & trainY,
        const Rcpp::List & inbag_counts_ensemble) {

    const int num_trees = rfobj["ntree"];
    const Rcpp::List forest = rfobj["forest"];
    const Rcpp::IntegerVector num_nodes = forest["ndbigtree"];

    // We'll use one and only one of these
    const Rcpp::NumericVector numeric_responses = trainY[0];
    const Rcpp::IntegerVector factor_responses = trainY[0];

    const Rcpp::CharacterVector class_names =
        strcmp(rfobj["type"], "classification") == 0
        ? factor_responses.attr("levels")
        : Rcpp::CharacterVector::create("Response");
    const int num_classes = class_names.size();

    Rcpp::List left_children_ensemble(num_trees);
    Rcpp::List right_children_ensemble(num_trees);
    if (num_classes == 1) {
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
    const Rcpp::CharacterVector rf_feature_names
        = Rcpp::rownames(rfobj["importance"]);
    const Rcpp::CharacterVector original_feature_names = trainX.names();
    Rcpp::IntegerVector reorder
        = Rcpp::match(rf_feature_names, original_feature_names) - 1;
    reorder.push_front(NA_INTEGER);
    Rcpp::NumericMatrix split_values_ensemble_matrix = forest["xbestsplit"];
    for (int tree = 0; tree < num_trees; tree++) {
        const Rcpp::Range row_range = Rcpp::Range(0, num_nodes[tree] - 1);
        const Rcpp::Range column_range = Rcpp::Range(tree, tree);

        const Rcpp::IntegerMatrix split_variables_matrix
            = split_variables_ensemble_matrix(row_range, column_range);
        split_variables_ensemble[tree] = reorder[
            Rcpp::as<Rcpp::IntegerVector>(split_variables_matrix)];

        split_values_ensemble[tree]
            = split_values_ensemble_matrix(row_range, column_range);
    }

    const Rcpp::CharacterVector feature_names = trainX.names();

    const Rcpp::List auxiliary_information
        = calculate_auxiliary_information(
                trainX,
                numeric_responses,
                factor_responses,
                inbag_counts_ensemble,
                num_classes,
                num_trees,
                left_children_ensemble,
                right_children_ensemble,
                split_variables_ensemble,
                split_values_ensemble
                );

    return Rcpp::List::create(
            Rcpp::Named("num.trees") = num_trees,
            Rcpp::Named("feature.names") = feature_names,
            Rcpp::Named("num.classes") = num_classes,
            Rcpp::Named("class.names") = class_names,
            Rcpp::Named("inbag.counts") = inbag_counts_ensemble,
            Rcpp::Named("left.children") = left_children_ensemble,
            Rcpp::Named("right.children") = right_children_ensemble,
            Rcpp::Named("split.variables") = split_variables_ensemble,
            Rcpp::Named("split.values") = split_values_ensemble,
            Rcpp::Named("node.sizes") = auxiliary_information[0],
            Rcpp::Named("node.resp") = auxiliary_information[1],
            Rcpp::Named("delta.node.resp.left") = auxiliary_information[2],
            Rcpp::Named("delta.node.resp.right") = auxiliary_information[3]
            );
}

// [[Rcpp::export]]
Rcpp::List tidyRFCpp_ranger(
        const Rcpp::List & rfobj,
        const Rcpp::DataFrame & trainX,
        const Rcpp::DataFrame & trainY,
        const Rcpp::List & inbag_counts_ensemble) {

    const int num_trees = rfobj["num.trees"];
    const Rcpp::List forest = rfobj["forest"];

    // We'll use one and only one of these
    Rcpp::NumericVector numeric_responses = trainY[0];
    Rcpp::IntegerVector factor_responses = trainY[0];

    // I don't even know when `class.values` is NOT `1:nlevels(Y)`. It's added
    // simply because this conversion is carried out by `ranger::ranger` when
    // calculating the predictions of the random forest.
    if (forest.containsElementNamed("class.values")) {
        const Rcpp::IntegerVector class_values = forest["class.values"];
        // Preserve attributes of factor_responses via pointer aliasing
        Rcpp::IntegerVector alias_factor_responses = factor_responses;
        alias_factor_responses
            = Rcpp::match(alias_factor_responses, class_values);
    }

    const Rcpp::CharacterVector class_names =
        strcmp(rfobj["treetype"], "Classification") == 0
        ? factor_responses.attr("levels")
        : Rcpp::CharacterVector::create("Response");
    const int num_classes = class_names.size();

    const Rcpp::List children_ensemble = forest["child.nodeIDs"];
    Rcpp::List left_children_ensemble(num_trees);
    Rcpp::List right_children_ensemble(num_trees);
    for (int tree = 0; tree < num_trees; tree++) {
        const Rcpp::List children = children_ensemble[tree];
        left_children_ensemble[tree] = children[0];
        right_children_ensemble[tree] = children[1];
    }

    const Rcpp::List split_var_IDs_ensemble = forest["split.varIDs"];
    const Rcpp::CharacterVector rf_feature_names =
        forest["independent.variable.names"];
    const Rcpp::CharacterVector original_feature_names = trainX.names();
    Rcpp::IntegerVector reorder
        = Rcpp::match(rf_feature_names, original_feature_names) - 1;
    if (forest.containsElementNamed("dependent.varID")) {
        // For ranger version <0.11.5
        reorder.push_front(NA_INTEGER);
    }

    Rcpp::List split_variables_ensemble(num_trees);
    for (int tree = 0; tree < num_trees; tree++) {
        Rcpp::IntegerVector split_var_IDs = split_var_IDs_ensemble[tree];
        split_variables_ensemble[tree] = reorder[split_var_IDs];
    }
    const Rcpp::List split_values_ensemble = forest["split.values"];

    const Rcpp::List auxiliary_information
        = calculate_auxiliary_information(
                trainX,
                numeric_responses,
                factor_responses,
                inbag_counts_ensemble,
                num_classes,
                num_trees,
                left_children_ensemble,
                right_children_ensemble,
                split_variables_ensemble,
                split_values_ensemble
                );

    return Rcpp::List::create(
            Rcpp::Named("num.trees") = num_trees,
            Rcpp::Named("feature.names") = original_feature_names,
            Rcpp::Named("num.classes") = num_classes,
            Rcpp::Named("class.names") = class_names,
            Rcpp::Named("inbag.counts") = inbag_counts_ensemble,
            Rcpp::Named("left.children") = left_children_ensemble,
            Rcpp::Named("right.children") = right_children_ensemble,
            Rcpp::Named("split.variables") = split_variables_ensemble,
            Rcpp::Named("split.values") = split_values_ensemble,
            Rcpp::Named("node.sizes") = auxiliary_information[0],
            Rcpp::Named("node.resp") = auxiliary_information[1],
            Rcpp::Named("delta.node.resp.left") = auxiliary_information[2],
            Rcpp::Named("delta.node.resp.right") = auxiliary_information[3]
            );
}

Rcpp::List calculate_auxiliary_information(
        const Rcpp::DataFrame & trainX,
        const Rcpp::NumericVector & numeric_responses,
        const Rcpp::IntegerVector & factor_responses,
        const Rcpp::List & inbag_counts_ensemble,
        const int num_classes,
        const int num_trees,
        const Rcpp::List & left_children_ensemble,
        const Rcpp::List & right_children_ensemble,
        const Rcpp::List & split_variables_ensemble,
        const Rcpp::List & split_values_ensemble) {

    Rcpp::List auxiliary_information(4);

    Rcpp::List node_sizes_ensemble(num_trees);
    Rcpp::List node_responses_ensemble(num_trees);
    Rcpp::List delta_node_responses_left_ensemble(num_trees);
    Rcpp::List delta_node_responses_right_ensemble(num_trees);

    const Rcpp::CharacterVector colnames = (num_classes == 1)
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
        const Rcpp::IntegerVector rownames = Rcpp::seq_len(num_nodes) - 1;

        Rcpp::IntegerVector node_sizes(num_nodes);

        Rcpp::NumericMatrix node_responses(num_nodes, num_classes);
        Rcpp::rownames(node_responses) = rownames;
        Rcpp::colnames(node_responses) = colnames;

        Rcpp::NumericMatrix delta_node_responses_left(num_nodes, num_classes);
        Rcpp::rownames(delta_node_responses_left) = rownames;
        Rcpp::colnames(delta_node_responses_left) = colnames;

        Rcpp::NumericMatrix delta_node_responses_right(num_nodes, num_classes);
        Rcpp::rownames(delta_node_responses_right) = rownames;
        Rcpp::colnames(delta_node_responses_right) = colnames;

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

                if (num_classes == 1) {
                    // Regression
                    node_responses(node_id, 0)
                        += numeric_responses[x] * inbag_count;
                } else {
                    // Classification
                    node_responses(node_id, factor_responses[x] - 1)
                        += inbag_count;
                }
            }
        }

        node_responses_ensemble[tree] = node_responses;

        for (int node = num_nodes - 1; node >= 0; node--) {
            const int left_child = left_children[node];
            const int right_child = right_children[node];
            if (!node_sizes[node]) {
                continue;
            } else if (!left_child && !right_child) {
                node_responses.row(node)
                    = node_responses.row(node) / node_sizes[node];
            } else {
                node_responses.row(node)
                    = (node_responses.row(left_child)
                            * node_sizes[left_child]
                            + node_responses.row(right_child)
                            * node_sizes[right_child])
                    / (node_sizes[left_child] + node_sizes[right_child]);

                delta_node_responses_left.row(node)
                    = node_responses.row(left_child)
                    - node_responses.row(node);

                delta_node_responses_right.row(node)
                    = node_responses.row(right_child)
                    - node_responses.row(node);
            }
        }

        node_sizes_ensemble[tree] = node_sizes;
        node_responses_ensemble[tree] = node_responses;
        delta_node_responses_left_ensemble[tree] = delta_node_responses_left;
        delta_node_responses_right_ensemble[tree] = delta_node_responses_right;
    }

    return Rcpp::List::create(
            Rcpp::Named("node.sizes") = node_sizes_ensemble,
            Rcpp::Named("node.resp") = node_responses_ensemble,
            Rcpp::Named("delta.node.resp.left")
            = delta_node_responses_left_ensemble,
            Rcpp::Named("delta.node.resp.right")
            = delta_node_responses_right_ensemble
            );
}

