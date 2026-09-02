#include <mlc/linear_regression.h>
#include <math.h>

MLCStatus mlc_linear_regression_create(size_t feature_count, size_t output_count, MLCLinearRegression *model) {
    if (model == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (feature_count == 0 || output_count == 0) {
        return MLC_STATUS_INVALID_DIMENSIONS;
    }
    if (model->weights.data != NULL || model->weights.rows != 0 || model->weights.columns != 0
        || model->biases.data != NULL || model->biases.rows != 0 || model->biases.columns != 0) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    MLCStatus create_status = mlc_matrix_create(feature_count, output_count, &model->weights);
    if (create_status != MLC_STATUS_SUCCESS) {
        return create_status;
    }
    create_status = mlc_matrix_create(1, output_count, &model->biases);
    if (create_status != MLC_STATUS_SUCCESS) {
        mlc_matrix_free(&model->weights);
        return create_status;
    }
    return MLC_STATUS_SUCCESS;
}

void mlc_linear_regression_free(MLCLinearRegression *model) {
    if (model == NULL) {
        return;
    }
    mlc_matrix_free(&model->weights);
    mlc_matrix_free(&model->biases);
}

MLCStatus mlc_linear_regression_predict(const MLCLinearRegression *model, const MLCMatrix *features, MLCMatrix *predictions) {
    if (model == NULL || features == NULL  || predictions == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (model->weights.data == NULL || model->weights.rows == 0 || model->weights.columns == 0
        || model->biases.data == NULL || model->biases.rows == 0 || model->biases.columns == 0) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (features->data == NULL || features->rows == 0 || features->columns == 0) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (predictions->data != NULL || predictions->rows != 0 || predictions->columns != 0) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (model->biases.rows != 1) {
        return MLC_STATUS_INVALID_DIMENSIONS;
    }
    if (model->biases.columns != model->weights.columns) {
        return MLC_STATUS_INVALID_DIMENSIONS;
    }
    if (features->columns != model->weights.rows) {
        return MLC_STATUS_DIMENSIONS_MISMATCH;
    }
    MLCMatrix linear_output = {0};
    MLCStatus status = mlc_matrix_multiply(features, &model->weights, &linear_output);
    if (status != MLC_STATUS_SUCCESS) {
        return status;
    }
    status = mlc_matrix_add_row_vector(&linear_output, &model->biases, predictions);
    mlc_matrix_free(&linear_output);
    if (status != MLC_STATUS_SUCCESS) {
        mlc_matrix_free(predictions);
        return status;
    }
    return MLC_STATUS_SUCCESS;
}

MLCStatus mlc_linear_regression_fit(MLCLinearRegression *model, const MLCDataset *dataset, double learning_rate, size_t epoch_count) {
    if (model == NULL || dataset == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (model->weights.data == NULL || model->weights.rows == 0 || model->weights.columns == 0
        || model->biases.data == NULL || model->biases.rows == 0 || model->biases.columns == 0) {
        return MLC_STATUS_INVALID_ARGUMENT;
        }
    if (dataset->features.data == NULL || dataset->targets.data == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (dataset->features.rows == 0 || dataset->features.columns == 0 || dataset->targets.rows == 0 || dataset->targets.columns == 0) {
        return MLC_STATUS_INVALID_DIMENSIONS;
    }
    if (dataset->features.rows != dataset->targets.rows) {
        return MLC_STATUS_DIMENSIONS_MISMATCH;
    }
    if (dataset->features.columns != model->weights.rows) {
        return MLC_STATUS_DIMENSIONS_MISMATCH;
    }
    if (dataset->targets.columns != model->biases.columns) {
        return MLC_STATUS_DIMENSIONS_MISMATCH;
    }
    if (model->biases.rows != 1) {
        return MLC_STATUS_INVALID_DIMENSIONS;
    }
    if (model->weights.columns != model->biases.columns) {
        return MLC_STATUS_INVALID_DIMENSIONS;
    }
    if (!isfinite(learning_rate) || learning_rate <= 0.0) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (epoch_count == 0) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    size_t N = dataset->features.rows;
    size_t O = dataset->targets.columns;
    double step_scale = (2.0 * learning_rate) / ((double)(N) * (double)(O));
    MLCMatrix X_transpose = {0};
    MLCStatus status = mlc_matrix_transpose(&dataset->features, &X_transpose);
    if (status != MLC_STATUS_SUCCESS) {
        return status;
    }
    for (size_t epoch = 0; epoch < epoch_count; ++epoch) {
        MLCMatrix prediction = {0}, error = {0}, weight_gradients = {0}, bias_gradients = {0};
        status = mlc_linear_regression_predict(model, &dataset->features, &prediction);
        if (status != MLC_STATUS_SUCCESS) {
            return status;
        }
        status = mlc_matrix_subtraction(&prediction, &dataset->targets, &error);
        if (status != MLC_STATUS_SUCCESS) {
            mlc_matrix_free(&X_transpose);
            mlc_matrix_free(&prediction);
            return status;
        }
        status = mlc_matrix_multiply(&X_transpose, &error, &weight_gradients);
        if (status != MLC_STATUS_SUCCESS) {
            mlc_matrix_free(&X_transpose);
            mlc_matrix_free(&prediction);
            mlc_matrix_free(&error);
            return status;
        }
        status = mlc_matrix_sum_rows(&error, &bias_gradients);
        if (status != MLC_STATUS_SUCCESS) {
            mlc_matrix_free(&X_transpose);
            mlc_matrix_free(&prediction);
            mlc_matrix_free(&error);
            mlc_matrix_free(&weight_gradients);
            return status;
        }
        for (size_t i = 0; i < model->weights.rows * model->weights.columns; ++i) {
            model->weights.data[i] = model->weights.data[i] - step_scale * weight_gradients.data[i];
        }
        for (size_t i = 0; i < model->biases.rows * model->biases.columns; ++i) {
            model->biases.data[i] = model->biases.data[i] - step_scale * bias_gradients.data[i];
        }
        mlc_matrix_free(&prediction);
        mlc_matrix_free(&error);
        mlc_matrix_free(&weight_gradients);
        mlc_matrix_free(&bias_gradients);
    }
    mlc_matrix_free(&X_transpose);
    return MLC_STATUS_SUCCESS;
}


