#ifndef MLC_LINEAR_REGRESSION_H
#define MLC_LINEAR_REGRESSION_H
#include <mlc/matrix.h>
#include <mlc/dataset.h>

typedef struct {
    MLCMatrix weights;
    MLCMatrix biases;
}MLCLinearRegression;

MLCStatus mlc_linear_regression_create(size_t feature_count, size_t output_count, MLCLinearRegression *model);
void mlc_linear_regression_free(MLCLinearRegression *model);
MLCStatus mlc_linear_regression_predict(const MLCLinearRegression *model, const MLCMatrix *features, MLCMatrix *predictions);
MLCStatus mlc_linear_regression_fit(MLCLinearRegression *model, const MLCDataset *dataset, double learning_rate, size_t epoch_count);
#endif //MLC_LINEAR_REGRESSION_H