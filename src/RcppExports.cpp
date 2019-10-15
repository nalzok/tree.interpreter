// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

// featureContribCpp
Rcpp::List featureContribCpp(const Rcpp::List tidyRF, const Rcpp::DataFrame testX);
RcppExport SEXP _tree_interpreter_featureContribCpp(SEXP tidyRFSEXP, SEXP testXSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::List >::type tidyRF(tidyRFSEXP);
    Rcpp::traits::input_parameter< const Rcpp::DataFrame >::type testX(testXSEXP);
    rcpp_result_gen = Rcpp::wrap(featureContribCpp(tidyRF, testX));
    return rcpp_result_gen;
END_RCPP
}
// trainsetBiasCpp
Rcpp::NumericMatrix trainsetBiasCpp(const Rcpp::List tidyRF);
RcppExport SEXP _tree_interpreter_trainsetBiasCpp(SEXP tidyRFSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::List >::type tidyRF(tidyRFSEXP);
    rcpp_result_gen = Rcpp::wrap(trainsetBiasCpp(tidyRF));
    return rcpp_result_gen;
END_RCPP
}
// tidyRFCpp_randomForest
Rcpp::List tidyRFCpp_randomForest(const Rcpp::List& rfobj, const Rcpp::DataFrame& trainX, const Rcpp::DataFrame& trainY, const Rcpp::List& inbag_counts_ensemble);
RcppExport SEXP _tree_interpreter_tidyRFCpp_randomForest(SEXP rfobjSEXP, SEXP trainXSEXP, SEXP trainYSEXP, SEXP inbag_counts_ensembleSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::List& >::type rfobj(rfobjSEXP);
    Rcpp::traits::input_parameter< const Rcpp::DataFrame& >::type trainX(trainXSEXP);
    Rcpp::traits::input_parameter< const Rcpp::DataFrame& >::type trainY(trainYSEXP);
    Rcpp::traits::input_parameter< const Rcpp::List& >::type inbag_counts_ensemble(inbag_counts_ensembleSEXP);
    rcpp_result_gen = Rcpp::wrap(tidyRFCpp_randomForest(rfobj, trainX, trainY, inbag_counts_ensemble));
    return rcpp_result_gen;
END_RCPP
}
// tidyRFCpp_ranger
Rcpp::List tidyRFCpp_ranger(const Rcpp::List& rfobj, const Rcpp::DataFrame& trainX, const Rcpp::DataFrame& trainY, const Rcpp::List& inbag_counts_ensemble);
RcppExport SEXP _tree_interpreter_tidyRFCpp_ranger(SEXP rfobjSEXP, SEXP trainXSEXP, SEXP trainYSEXP, SEXP inbag_counts_ensembleSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::List& >::type rfobj(rfobjSEXP);
    Rcpp::traits::input_parameter< const Rcpp::DataFrame& >::type trainX(trainXSEXP);
    Rcpp::traits::input_parameter< const Rcpp::DataFrame& >::type trainY(trainYSEXP);
    Rcpp::traits::input_parameter< const Rcpp::List& >::type inbag_counts_ensemble(inbag_counts_ensembleSEXP);
    rcpp_result_gen = Rcpp::wrap(tidyRFCpp_ranger(rfobj, trainX, trainY, inbag_counts_ensemble));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_tree_interpreter_featureContribCpp", (DL_FUNC) &_tree_interpreter_featureContribCpp, 2},
    {"_tree_interpreter_trainsetBiasCpp", (DL_FUNC) &_tree_interpreter_trainsetBiasCpp, 1},
    {"_tree_interpreter_tidyRFCpp_randomForest", (DL_FUNC) &_tree_interpreter_tidyRFCpp_randomForest, 4},
    {"_tree_interpreter_tidyRFCpp_ranger", (DL_FUNC) &_tree_interpreter_tidyRFCpp_ranger, 4},
    {NULL, NULL, 0}
};

RcppExport void R_init_tree_interpreter(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
