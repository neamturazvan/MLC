#ifndef MLC_ACTIVATION_H
#define MLC_ACTIVATION_H

#include <mlc/matrix.h>

typedef enum {
    MLC_ACTIVATION_LINEAR,
    MLC_ACTIVATION_RELU,
    MLC_ACTIVATION_SIGMOID,
    MLC_ACTIVATION_SOFTMAX
} MLCActivationType;

MLCStatus mlc_activation_forward(const MLCMatrix *input, MLCActivationType activation, MLCMatrix *output);
MLCStatus mlc_activation_backward(const MLCMatrix *activated_output, const MLCMatrix *upstream_gradient, MLCActivationType activation, MLCMatrix *input_gradient);

#endif