#include <mlc/activation.h>
#include <mlc/matrix.h>
#include <math.h>
#include <float.h>


MLCStatus mlc_activation_forward(const MLCMatrix *input, MLCActivationType activation, MLCMatrix *output) {
    if (input == NULL || output == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (input->data == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (input->rows == 0 || input->columns == 0) {
        return MLC_STATUS_INVALID_DIMENSIONS;
    }
    if (output->data != NULL || output->rows != 0 || output->columns != 0) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    switch (activation) {
        case MLC_ACTIVATION_LINEAR:
            return mlc_matrix_clone(input, output);
        case MLC_ACTIVATION_RELU:{
            MLCStatus create_status = mlc_matrix_create(input->rows, input->columns, output);
            if (create_status != MLC_STATUS_SUCCESS) {
                return create_status;
            }
            for (size_t i = 0; i < input->rows * input->columns; ++i) {
                if (input->data[i] > 0) {
                    output->data[i] = input->data[i];
                }
                else {
                    output->data[i] = 0;
                }
            }
            return MLC_STATUS_SUCCESS;
    }
        case MLC_ACTIVATION_SIGMOID: {
            MLCStatus create_status = mlc_matrix_create(input->rows, input->columns, output);
            if (create_status != MLC_STATUS_SUCCESS) {
                return create_status;
            }
            for (size_t i = 0; i < input->rows * input->columns; ++i) {
                if (input->data[i] > 0) {
                    output->data[i] = 1.0 / (1.0 + exp(-input->data[i]));
                }
                else {
                    output->data[i] = exp(input->data[i]) / (1.0 + exp(input->data[i]));
                }
            }
            return MLC_STATUS_SUCCESS;
        }
        case MLC_ACTIVATION_SOFTMAX: {
            MLCStatus create_status = mlc_matrix_create(input->rows, input->columns, output);
            if (create_status != MLC_STATUS_SUCCESS) {
                return create_status;
            }
            for (size_t i = 0; i < input->rows; ++i) {
                double max_value = input->data[i * input->columns], normalization = 0.0;
                for (size_t j = 0; j < input->columns; ++j) {
                    size_t pos = i * input->columns + j;
                    if (input->data[pos] > max_value) {
                        max_value = input->data[pos];
                    }
                }
                for (size_t j = 0; j < input->columns; ++j) {
                    size_t pos = i * input->columns + j;
                    output->data[pos] = exp(input->data[pos] - max_value);
                    normalization += output->data[pos];
                }
                for (size_t j = 0; j < input->columns; ++j) {
                    size_t pos = i * input->columns + j;
                    output->data[pos] /= normalization;
                }
            }
            return MLC_STATUS_SUCCESS;
        }
        default:
            return MLC_STATUS_INVALID_ARGUMENT;
    }
}

MLCStatus mlc_activation_backward(const MLCMatrix *activated_output, const MLCMatrix *upstream_gradient, MLCActivationType activation, MLCMatrix *input_gradient) {
    if (activated_output == NULL || upstream_gradient == NULL || input_gradient == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (activated_output->data == NULL || upstream_gradient->data == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (activated_output->rows == 0 || activated_output->columns == 0 || upstream_gradient->rows == 0 || upstream_gradient->columns == 0) {
        return MLC_STATUS_INVALID_DIMENSIONS;
    }
    if (input_gradient->data != NULL || input_gradient->rows != 0 || input_gradient->columns != 0) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (activated_output->rows != upstream_gradient->rows || activated_output->columns != upstream_gradient->columns) {
        return MLC_STATUS_DIMENSIONS_MISMATCH;
    }
    switch (activation) {
        case MLC_ACTIVATION_LINEAR:
            return mlc_matrix_clone(upstream_gradient, input_gradient);
        case MLC_ACTIVATION_RELU: {
            MLCStatus create_status = mlc_matrix_create(upstream_gradient->rows, upstream_gradient->columns, input_gradient);
            if (create_status != MLC_STATUS_SUCCESS) {
                return create_status;
            }
            for (size_t i = 0; i < upstream_gradient->rows * upstream_gradient->columns; ++i) {
                if (activated_output->data[i] > 0) {
                    input_gradient->data[i] = upstream_gradient->data[i];
                }
                else {
                    input_gradient->data[i] = 0;
                }
            }
            return MLC_STATUS_SUCCESS;
        }
        case MLC_ACTIVATION_SIGMOID: {
            MLCStatus create_status = mlc_matrix_create(upstream_gradient->rows, upstream_gradient->columns, input_gradient);
            if (create_status != MLC_STATUS_SUCCESS) {
                return create_status;
            }
            for (size_t i = 0; i < upstream_gradient->rows * upstream_gradient->columns; ++i) {
                input_gradient->data[i] = upstream_gradient->data[i] * activated_output->data[i] * (1.0 - activated_output->data[i]);
            }
            return MLC_STATUS_SUCCESS;
        }

        case MLC_ACTIVATION_SOFTMAX: {
            MLCStatus create_status = mlc_matrix_create(upstream_gradient->rows, upstream_gradient->columns, input_gradient);
            if (create_status != MLC_STATUS_SUCCESS) {
                return create_status;
            }
            for (size_t i = 0; i < upstream_gradient->rows; ++i) {
                double dot_product = 0.0;
                for (size_t j = 0; j < upstream_gradient->columns; ++j) {
                    size_t pos = i * upstream_gradient->columns + j;
                    dot_product += upstream_gradient->data[pos] * activated_output->data[pos];
                }
                for (size_t j = 0; j < upstream_gradient->columns; ++j) {
                    size_t pos = i * upstream_gradient->columns + j;
                    input_gradient->data[pos] = activated_output->data[pos] * (upstream_gradient->data[pos] - dot_product);
                }
            }
            return MLC_STATUS_SUCCESS;
        }
        default:
            return MLC_STATUS_INVALID_ARGUMENT;
    }
}
