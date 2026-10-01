#include <mlc/activation.h>
#include <mlc/logistic_regression.h>
#include <mlc/dataset.h>
#include <math.h>
#include <mlc/loss.h>

MLCStatus mlc_logistic_regression_create(size_t feature_count, MLCLogisticRegression *model) {
    if (model == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (model->weights.data != NULL || model->bias.data != NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (model->weights.rows != 0 || model->weights.columns != 0 || model->bias.rows != 0 || model->bias.columns != 0) {
        return MLC_STATUS_INVALID_DIMENSIONS;
    }
    if (feature_count == 0) {
        return MLC_STATUS_INVALID_DIMENSIONS;
    }
    MLCStatus create_status = mlc_matrix_create( feature_count, 1, &model->weights);
    if (create_status != MLC_STATUS_SUCCESS) {
        return create_status;
    }
    create_status = mlc_matrix_create(1, 1, &model->bias);
    if (create_status != MLC_STATUS_SUCCESS) {
        mlc_matrix_free(&model->weights);
        return create_status;
    }
    return MLC_STATUS_SUCCESS;
}


void mlc_logistic_regression_free(MLCLogisticRegression *model) {
    if (model == NULL) {
        return;
    }
    mlc_matrix_free(&model->weights);
    mlc_matrix_free(&model->bias);
}

MLCStatus mlc_logistic_regression_predict_probabilities(const MLCLogisticRegression *model, const MLCMatrix *features,MLCMatrix *probabilities) {
    if (model == NULL || features == NULL || probabilities == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (model->weights.data == NULL || model->bias.data == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (model->weights.rows == 0 || model->weights.columns == 0 || model->bias.rows == 0 || model->bias.columns == 0) {
        return MLC_STATUS_INVALID_DIMENSIONS;
    }
    if (features->data == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (features->rows == 0 || features->columns == 0) {
        return MLC_STATUS_INVALID_DIMENSIONS;
    }
    if (probabilities->data != NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (probabilities->rows != 0 || probabilities->columns != 0) {
        return MLC_STATUS_INVALID_DIMENSIONS;
    }
    if (model->weights.columns != 1) {
        return MLC_STATUS_INVALID_DIMENSIONS;
    }
    if (model->bias.rows != 1 || model->bias.columns != 1) {
        return MLC_STATUS_INVALID_DIMENSIONS;
    }
    if (features->columns != model->weights.rows) {
        return MLC_STATUS_DIMENSIONS_MISMATCH;
    }
    MLCMatrix logits = {0}, biased_logits = {0};
    MLCStatus status = mlc_matrix_multiply(features, &model->weights, &logits);
    if (status != MLC_STATUS_SUCCESS) {
        return status;
    }
    status = mlc_matrix_add_row_vector(&logits, &model->bias, &biased_logits);
    if (status != MLC_STATUS_SUCCESS) {
        mlc_matrix_free(&logits);
        return status;
    }
    status = mlc_activation_forward(&biased_logits, MLC_ACTIVATION_SIGMOID, probabilities);
    if (status != MLC_STATUS_SUCCESS) {
        mlc_matrix_free(&logits);
        mlc_matrix_free(&biased_logits);
        return status;
    }
    mlc_matrix_free(&logits);
    mlc_matrix_free(&biased_logits);
    return MLC_STATUS_SUCCESS;
}

MLCStatus mlc_logistic_regression_predict(const MLCLogisticRegression *model, const MLCMatrix *features, MLCMatrix *classes) {
    if (classes == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (classes->data != NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (classes->rows != 0 || classes->columns != 0) {
        return MLC_STATUS_INVALID_DIMENSIONS;
    }
    MLCStatus status = mlc_logistic_regression_predict_probabilities(model, features, classes);
    if (status != MLC_STATUS_SUCCESS) {
        return status;
    }
    for (size_t i = 0; i < classes->rows * classes->columns; ++i) {
        double p = classes->data[i];
        if (p >= 0.5) {
            classes->data[i] = 1.0;
        }
        else {
            classes->data[i] = 0.0;
        }
    }
    return MLC_STATUS_SUCCESS;
}

MLCStatus mlc_logistic_regression_fit(MLCLogisticRegression *model, const MLCDataset *dataset, double learning_rate, size_t epoch_count) {
    if (model == NULL || dataset == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (model->weights.data == NULL || model->bias.data == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (model->weights.rows == 0 || model->weights.columns == 0 || model->bias.rows == 0 || model->bias.columns == 0) {
        return MLC_STATUS_INVALID_DIMENSIONS;
    }
    if (dataset->features.data == NULL || dataset->targets.data == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (dataset->features.rows == 0 || dataset->features.columns == 0 || dataset->targets.rows == 0 || dataset->targets.columns == 0) {
        return MLC_STATUS_INVALID_DIMENSIONS;
    }
    if (model->weights.columns != 1) {
        return MLC_STATUS_INVALID_DIMENSIONS;
    }
    if (model->bias.rows != 1 || model->bias.columns != 1) {
        return MLC_STATUS_INVALID_DIMENSIONS;
    }
    if (dataset->features.rows != dataset->targets.rows) {
        return MLC_STATUS_DIMENSIONS_MISMATCH;
    }
    if (dataset->targets.columns != 1) {
        return MLC_STATUS_INVALID_DIMENSIONS;
    }
    if (dataset->features.columns != model->weights.rows) {
        return MLC_STATUS_DIMENSIONS_MISMATCH;
    }
    if (!isfinite(learning_rate) || learning_rate <= 0 || epoch_count == 0) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    MLCMatrix features_transpose = {0};
    MLCStatus status = mlc_matrix_transpose(&dataset->features, &features_transpose);
    if (status != MLC_STATUS_SUCCESS) {
        return status;
    }
    for (size_t i = 0; i < epoch_count; ++i) {
        MLCMatrix probabilities = {0}, probability_gradient = {0}, logit_gradient = {0}, weight_gradient = {0}, bias_gradient = {0};
        status = mlc_logistic_regression_predict_probabilities(model, &dataset->features, &probabilities);
        if (status != MLC_STATUS_SUCCESS) {
            mlc_matrix_free(&features_transpose);
            return status;
        }
        status = mlc_loss_backward(&probabilities, &dataset->targets, MLC_LOSS_BINARY_CROSS_ENTROPY, &probability_gradient);
        if (status != MLC_STATUS_SUCCESS) {
            mlc_matrix_free(&features_transpose);
            mlc_matrix_free(&probabilities);
            return status;
        }
        status = mlc_activation_backward(&probabilities, &probability_gradient, MLC_ACTIVATION_SIGMOID, &logit_gradient);
        if (status != MLC_STATUS_SUCCESS) {
            mlc_matrix_free(&features_transpose);
            mlc_matrix_free(&probabilities);
            mlc_matrix_free(&probability_gradient);
            return status;
        }
        status = mlc_matrix_multiply(&features_transpose, &logit_gradient, &weight_gradient);
        if (status != MLC_STATUS_SUCCESS) {
            mlc_matrix_free(&features_transpose);
            mlc_matrix_free(&probabilities);
            mlc_matrix_free(&probability_gradient);
            mlc_matrix_free(&logit_gradient);
            return status;
        }
        status = mlc_matrix_sum_rows(&logit_gradient, &bias_gradient);
        if (status != MLC_STATUS_SUCCESS) {
            mlc_matrix_free(&features_transpose);
            mlc_matrix_free(&probabilities);
            mlc_matrix_free(&probability_gradient);
            mlc_matrix_free(&logit_gradient);
            mlc_matrix_free(&weight_gradient);
            return status;
        }
        for (size_t j = 0; j < model->weights.rows * model->weights.columns; ++j) {
            model->weights.data[j] = model->weights.data[j] - learning_rate * weight_gradient.data[j];
        }
        model->bias.data[0] = model->bias.data[0] - learning_rate * bias_gradient.data[0];
        mlc_matrix_free(&probabilities);
        mlc_matrix_free(&probability_gradient);
        mlc_matrix_free(&logit_gradient);
        mlc_matrix_free(&weight_gradient);
        mlc_matrix_free(&bias_gradient);
    }
    mlc_matrix_free(&features_transpose);
    return MLC_STATUS_SUCCESS;
}
