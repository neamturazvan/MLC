#include <mlc/loss.h>
#include <math.h>

MLCStatus mlc_loss_forward(const MLCMatrix *predictions, const MLCMatrix *targets, const MLCLossType loss, double *loss_value) {
    if (predictions == NULL || targets == NULL || loss_value == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (predictions->data == NULL || targets->data == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (predictions->rows == 0 || predictions->columns == 0 || targets->rows == 0 || targets->columns == 0) {
        return MLC_STATUS_INVALID_DIMENSIONS;
    }
    if (predictions->rows != targets->rows || predictions->columns != targets->columns) {
        return MLC_STATUS_DIMENSIONS_MISMATCH;
    }
    switch (loss) {
        case MLC_LOSS_MEAN_SQUARED_ERROR: {
            *loss_value = 0.0;
            for (size_t i = 0; i < predictions->rows * predictions->columns; ++i) {
                *loss_value += (predictions->data[i] - targets->data[i]) * (predictions->data[i] - targets->data[i]);
            }
            *loss_value /= predictions->rows * predictions->columns;
            return MLC_STATUS_SUCCESS;
        }
        case MLC_LOSS_BINARY_CROSS_ENTROPY: {
            *loss_value = 0.0;
            const double eps = 1e-12;
            for (size_t i = 0; i < predictions->rows * predictions->columns; ++i) {
                double p = predictions->data[i];
                if (p < eps) {
                    p = eps;
                }
                if (p > 1.0 - eps) {
                    p = 1.0 - eps;
                }
                *loss_value += -(targets->data[i] * log(p) + (1 - targets->data[i]) * log(1 - p));
            }
            *loss_value /= predictions->rows * predictions->columns;

            return MLC_STATUS_SUCCESS;
        }
        case MLC_LOSS_CATEGORICAL_CROSS_ENTROPY: {
            *loss_value = 0.0;
            const double eps = 1e-12;
            for (size_t i = 0; i < predictions->rows * predictions->columns; ++i) {
                double p = predictions->data[i];
                if (p < eps) {
                    p = eps;
                }
                if (p > 1.0 - eps) {
                    p = 1.0 - eps;
                }
                *loss_value += targets->data[i] * log(p);
            }
            *loss_value /= predictions->rows;
            *loss_value *= -1;
            return MLC_STATUS_SUCCESS;
        }
        default:
            return MLC_STATUS_INVALID_ARGUMENT;
    }
}

MLCStatus mlc_loss_backward(const MLCMatrix *predictions, const MLCMatrix *targets, const MLCLossType loss, MLCMatrix *prediction_gradient) {
    if (predictions == NULL || targets == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (predictions->data == NULL || targets->data == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (predictions->rows == 0 || predictions->columns == 0 || targets->rows == 0 || targets->columns == 0) {
        return MLC_STATUS_INVALID_DIMENSIONS;
    }
    if (predictions->rows != targets->rows || predictions->columns != targets->columns) {
        return MLC_STATUS_DIMENSIONS_MISMATCH;
    }
    if (prediction_gradient->data != NULL || prediction_gradient->rows != 0 || prediction_gradient->columns != 0) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    switch (loss) {
        case MLC_LOSS_MEAN_SQUARED_ERROR:
            MLCStatus create_status = mlc_matrix_create(predictions->rows, predictions->columns, prediction_gradient);
            if (create_status != MLC_STATUS_SUCCESS) {
                return create_status;
            }
            double scale = 2.0 / (predictions->rows * predictions->columns);
            for (size_t i = 0; i < predictions->rows * predictions->columns; ++i) {
                prediction_gradient->data[i] = scale * (predictions->data[i] - targets->data[i]);
            }
            return MLC_STATUS_SUCCESS;
        case MLC_LOSS_BINARY_CROSS_ENTROPY: {
            MLCStatus create_status = mlc_matrix_create(predictions->rows, predictions->columns, prediction_gradient);
            if (create_status != MLC_STATUS_SUCCESS) {
                return create_status;
            }
            const double eps = 1e-12;
            for (size_t i = 0; i < predictions->rows * predictions->columns; ++i) {
                double p = predictions->data[i];
                if (p < eps) {
                    p = eps;
                }
                if (p > 1 - eps) {
                    p = 1 - eps;
                }
                prediction_gradient->data[i] = (p - targets->data[i]) / (predictions->rows * predictions->columns *
                    p * (1 - p));
            }
            return MLC_STATUS_SUCCESS;
        }
        case MLC_LOSS_CATEGORICAL_CROSS_ENTROPY: {
            MLCStatus create_status = mlc_matrix_create(predictions->rows, predictions->columns, prediction_gradient);
            if (create_status != MLC_STATUS_SUCCESS) {
                return create_status;
            }
            const double eps = 1e-12;
            for (size_t i = 0; i < predictions->rows * predictions->columns; ++i) {
                double p = predictions->data[i];
                if (p < eps) {
                    p = eps;
                }
                if (p > 1.0 - eps) {
                    p = 1.0 - eps;
                }
                prediction_gradient->data[i] = -targets->data[i] / (predictions->rows * p);
            }
            return MLC_STATUS_SUCCESS;
        }
        default:
            return MLC_STATUS_INVALID_ARGUMENT;
    }
}

