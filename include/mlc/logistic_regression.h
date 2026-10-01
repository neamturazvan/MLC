#ifndef MLC_LOGISTIC_REGRESSION_H
#define MLC_LOGISTIC_REGRESSION_H

#include <mlc/matrix.h>
#include <mlc/dataset.h>

typedef struct {
    MLCMatrix weights;
    MLCMatrix bias;
}MLCLogisticRegression;

MLCStatus mlc_logistic_regression_create(size_t feature_count, MLCLogisticRegression *model);
void mlc_logistic_regression_free(MLCLogisticRegression *model);
MLCStatus mlc_logistic_regression_predict_probabilities(const MLCLogisticRegression *model, const MLCMatrix *features,MLCMatrix *probabilities);
MLCStatus mlc_logistic_regression_predict(const MLCLogisticRegression *model, const MLCMatrix *features, MLCMatrix *classes);
MLCStatus mlc_logistic_regression_fit(MLCLogisticRegression *model, const MLCDataset *dataset, double learning_rate, size_t epoch_count);

#endif //MLC_LOGISTIC_REGRESSION_H