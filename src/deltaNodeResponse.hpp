Rcpp::List deltaNodeResponseCpp_randomForest(
        const Rcpp::List& rf,
        const Rcpp::DataFrame& trainX,
        const Rcpp::DataFrame& trainY,
        const Rcpp::List& inbag_counts);

Rcpp::List deltaNodeResponseCpp_ranger(
        const Rcpp::List& rf,
        const Rcpp::DataFrame& trainX,
        const Rcpp::DataFrame& trainY,
        const Rcpp::List& inbag_counts_ensemble);

Rcpp::List deltaNodeResponseCppClassification(
        const int n_classes,
        const int num_trees,
        const Rcpp::CharacterVector& independent_variable_names,
        const Rcpp::List& left_children_ensemble,
        const Rcpp::List& right_children_ensemble,
        const Rcpp::List& split_variables_ensemble,
        const Rcpp::List& split_values_ensemble,
        const Rcpp::DataFrame& trainX,
        const Rcpp::IntegerVector& factor_responses,
        const Rcpp::List& inbag_counts_ensemble);

Rcpp::List deltaNodeResponseCppRegression(
        const int n_classes,
        const int num_trees,
        const Rcpp::CharacterVector& independent_variable_names,
        const Rcpp::List& left_children_ensemble,
        const Rcpp::List& right_children_ensemble,
        const Rcpp::List& split_variables_ensemble,
        const Rcpp::List& split_values_ensemble,
        const Rcpp::DataFrame& trainX,
        const Rcpp::NumericVector& numeric_responses,
        const Rcpp::List& inbag_counts_ensemble);

void backpropagateResponse(
        const int num_nodes,
        const Rcpp::IntegerVector& left_children,
        const Rcpp::IntegerVector& right_children,
        const Rcpp::IntegerVector& node_sizes,
        Rcpp::NumericMatrix& delta_node_responses);

