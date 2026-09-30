#ifndef MLC_LOSS_H
#define MLC_LOSS_H

#include <mlc/matrix.h>

typedef enum {
    MLC_LOSS_MEAN_SQUARED_ERROR,
    MLC_LOSS_BINARY_CROSS_ENTROPY,
    MLC_LOSS_CATEGORICAL_CROSS_ENTROPY
}MLCLossType;

MLCStatus mlc_loss_forward(const MLCMatrix *predictions, const MLCMatrix *targets, const MLCLossType loss, double *loss_value);
MLCStatus mlc_loss_backward(const MLCMatrix *predictions, const MLCMatrix *targets, const MLCLossType loss, MLCMatrix *prediction_gradient);

#endif //MLC_LOSS_H