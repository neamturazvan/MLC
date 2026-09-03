#include <mlc/linear_regression.h>
#include <assert.h>
#include <math.h>

static void test_linear_regression_create_and_free() {
    MLCLinearRegression test_model = {0};
    assert(mlc_linear_regression_create(2,3, &test_model) == MLC_STATUS_SUCCESS);
    assert(test_model.weights.rows == 2 && test_model.weights.columns == 3);
    assert(test_model.biases.rows == 1 && test_model.biases.columns == 3);
    for (size_t i = 0; i < test_model.weights.rows * test_model.weights.columns; ++i) {
        assert(test_model.weights.data[i] == 0.0);
    }
    for (size_t i = 0; i < test_model.biases.rows * test_model.biases.columns; ++i) {
        assert(test_model.biases.data[i] == 0.0);
    }
    mlc_linear_regression_free(&test_model);
    assert(test_model.weights.data == NULL && test_model.weights.data == NULL);
    assert(test_model.weights.rows == 0 && test_model.weights.columns == 0 && test_model.biases.rows == 0 && test_model.biases.columns == 0);
    mlc_linear_regression_free(&test_model);
}


static void test_linear_regression_predict() {
    MLCLinearRegression test_model = {0};
    MLCMatrix features = {0}, prediction = {0};
    assert(mlc_matrix_create(2,2,&features) == MLC_STATUS_SUCCESS);
    mlc_matrix_set(&features, 0,0, 1);
    mlc_matrix_set(&features, 0,1, 2);
    mlc_matrix_set(&features, 1,0, 3);
    mlc_matrix_set(&features, 1,1, 4);
    assert(mlc_linear_regression_create(2,2, &test_model) == MLC_STATUS_SUCCESS);
    mlc_matrix_set(&test_model.weights, 0,0,2);
    mlc_matrix_set(&test_model.weights, 0,1,-1);
    mlc_matrix_set(&test_model.weights, 1,0,0.5);
    mlc_matrix_set(&test_model.weights, 1,1,3);
    mlc_matrix_set(&test_model.biases, 0, 0 ,1);
    mlc_matrix_set(&test_model.biases, 0, 1 ,-2);
    assert(mlc_linear_regression_predict(&test_model, &features, &prediction) == MLC_STATUS_SUCCESS);
    assert(prediction.rows == 2 && prediction.columns == 2);
    assert(fabs(prediction.data[0] - 4.0) < 1e-9);
    assert(fabs(prediction.data[1] - 3.0) < 1e-9);
    assert(fabs(prediction.data[2] - 9.0) < 1e-9);
    assert(fabs(prediction.data[3] - 7.0) < 1e-9);
    mlc_linear_regression_free(&test_model);
    mlc_matrix_free(&features);
    mlc_matrix_free(&prediction);
}

static void test_linear_regression_fit() {
    MLCLinearRegression test_model = {0};
    MLCDataset test_dataset = {0};
    MLCMatrix test_prediction = {0}, test_feature = {0};
    assert(mlc_dataset_load_csv("linear_regression_2x+1.csv", 1, false, &test_dataset) == MLC_STATUS_SUCCESS);
    assert(mlc_linear_regression_create(1,1,&test_model) == MLC_STATUS_SUCCESS);
    assert(mlc_linear_regression_fit(&test_model, &test_dataset, 0.1, 200) == MLC_STATUS_SUCCESS);
    assert(fabs(test_model.weights.data[0] - 2.0) < 1e-6);
    assert(fabs(test_model.biases.data[0] - 1.0) < 1e-6);
    assert(mlc_matrix_create(1,1,&test_feature) == MLC_STATUS_SUCCESS);
    mlc_matrix_set(&test_feature,0,0, 6.0);
    assert(mlc_linear_regression_predict(&test_model, &test_feature, &test_prediction) == MLC_STATUS_SUCCESS);
    assert(fabs(test_prediction.data[0] - 13.0) < 1e-6);
    mlc_linear_regression_free(&test_model);
    mlc_dataset_free(&test_dataset);
    mlc_matrix_free(&test_prediction);
    mlc_matrix_free(&test_feature);

}

static void test_linear_regression_fit_multi_output() {
    MLCLinearRegression test_model = {0};
    MLCDataset test_dataset = {0};
    MLCMatrix test_prediction = {0}, test_feature = {0};
    assert(mlc_dataset_load_csv("multi_output_training.csv", 2, false, &test_dataset) == MLC_STATUS_SUCCESS);
    assert(mlc_linear_regression_create(2,2,&test_model) == MLC_STATUS_SUCCESS);
    assert(mlc_linear_regression_fit(&test_model, &test_dataset, 0.1, 500) == MLC_STATUS_SUCCESS);
    assert(fabs(test_model.weights.data[0] - 2.0) < 1e-6);
    assert(fabs(test_model.weights.data[1] + 1.0) < 1e-6);
    assert(fabs(test_model.weights.data[2] - 3.0) < 1e-6);
    assert(fabs(test_model.weights.data[3] - 4.0) < 1e-6);
    assert(fabs(test_model.biases.data[0] - 1.0) < 1e-6);
    assert(fabs(test_model.biases.data[1] + 2.0) < 1e-6);
    assert(mlc_matrix_create(3,2, &test_feature) == MLC_STATUS_SUCCESS);
    mlc_matrix_set(&test_feature, 0, 0,2.0);
    mlc_matrix_set(&test_feature, 0, 1,3.0);
    mlc_matrix_set(&test_feature, 1, 0,-2.0);
    mlc_matrix_set(&test_feature, 1, 1,4.0);
    mlc_matrix_set(&test_feature, 2, 0,3.0);
    mlc_matrix_set(&test_feature, 2, 1,-2.0);
    assert(mlc_linear_regression_predict(&test_model, &test_feature, &test_prediction) == MLC_STATUS_SUCCESS);
    assert(fabs(test_prediction.data[0] - 14.0) < 1e-6);
    assert(fabs(test_prediction.data[1] - 8.0) < 1e-6);
    assert(fabs(test_prediction.data[2] - 9.0) < 1e-6);
    assert(fabs(test_prediction.data[3] - 16.0) < 1e-6);
    assert(fabs(test_prediction.data[4] - 1.0) < 1e-6);
    assert(fabs(test_prediction.data[5] - -13.0) < 1e-6);
}

static void test_linear_regression_invalid_arguments(void) {
    MLCLinearRegression model = {0};

    assert(mlc_linear_regression_create(0, 1, &model) == MLC_STATUS_INVALID_DIMENSIONS);
    assert(mlc_linear_regression_create(1, 0, &model) == MLC_STATUS_INVALID_DIMENSIONS);
    assert(mlc_linear_regression_create(1, 1, NULL) == MLC_STATUS_INVALID_ARGUMENT);
    assert(mlc_linear_regression_create(2, 2, &model) == MLC_STATUS_SUCCESS);
    assert(mlc_linear_regression_create(2, 2, &model) == MLC_STATUS_INVALID_ARGUMENT);

    MLCMatrix valid_features = {0};
    MLCMatrix wrong_features = {0};
    MLCMatrix predictions = {0};
    MLCMatrix nonempty_predictions = {0};

    assert(mlc_matrix_create(1, 2, &valid_features) == MLC_STATUS_SUCCESS);
    assert(mlc_matrix_create(1, 3, &wrong_features) == MLC_STATUS_SUCCESS);
    assert(mlc_linear_regression_predict(&model, &wrong_features, &predictions) == MLC_STATUS_DIMENSIONS_MISMATCH);
    assert(mlc_matrix_create(1, 2, &nonempty_predictions) == MLC_STATUS_SUCCESS);
    assert(mlc_linear_regression_predict(&model, &valid_features, &nonempty_predictions) == MLC_STATUS_INVALID_ARGUMENT);
    assert(mlc_linear_regression_predict(NULL, &valid_features, &predictions) == MLC_STATUS_INVALID_ARGUMENT);
    assert(mlc_linear_regression_predict(&model, NULL, &predictions) == MLC_STATUS_INVALID_ARGUMENT);
    assert(mlc_linear_regression_predict(&model, &valid_features, NULL) == MLC_STATUS_INVALID_ARGUMENT);

    MLCDataset row_mismatch_dataset = {0};

    assert(mlc_matrix_create(2, 2, &row_mismatch_dataset.features) == MLC_STATUS_SUCCESS);
    assert(mlc_matrix_create(3, 2, &row_mismatch_dataset.targets) == MLC_STATUS_SUCCESS);
    assert(mlc_linear_regression_fit(&model, &row_mismatch_dataset, 0.1, 1) == MLC_STATUS_DIMENSIONS_MISMATCH);

    MLCDataset wrong_feature_count_dataset = {0};

    assert(mlc_dataset_create(2, 3, 2, &wrong_feature_count_dataset) == MLC_STATUS_SUCCESS);
    assert(mlc_linear_regression_fit(&model, &wrong_feature_count_dataset, 0.1, 1) == MLC_STATUS_DIMENSIONS_MISMATCH);

    MLCDataset wrong_output_count_dataset = {0};

    assert(mlc_dataset_create(2, 2, 1, &wrong_output_count_dataset) == MLC_STATUS_SUCCESS);
    assert(mlc_linear_regression_fit(&model, &wrong_output_count_dataset, 0.1, 1) == MLC_STATUS_DIMENSIONS_MISMATCH);

    MLCDataset valid_dataset = {0};

    assert(mlc_dataset_create(2, 2, 2, &valid_dataset) == MLC_STATUS_SUCCESS);
    assert(mlc_linear_regression_fit(&model, &valid_dataset, 0.0, 1) == MLC_STATUS_INVALID_ARGUMENT);
    assert(mlc_linear_regression_fit(&model, &valid_dataset, NAN, 1) == MLC_STATUS_INVALID_ARGUMENT);
    assert(mlc_linear_regression_fit(&model, &valid_dataset, INFINITY, 1) == MLC_STATUS_INVALID_ARGUMENT);
    assert(mlc_linear_regression_fit(&model, &valid_dataset, 0.1, 0) == MLC_STATUS_INVALID_ARGUMENT);
    assert(mlc_linear_regression_fit(NULL, &valid_dataset, 0.1, 1) == MLC_STATUS_INVALID_ARGUMENT);
    assert(mlc_linear_regression_fit(&model, NULL, 0.1, 1) == MLC_STATUS_INVALID_ARGUMENT);

    mlc_matrix_free(&valid_features);
    mlc_matrix_free(&wrong_features);
    mlc_matrix_free(&predictions);
    mlc_matrix_free(&nonempty_predictions);
    mlc_matrix_free(&row_mismatch_dataset.features);
    mlc_matrix_free(&row_mismatch_dataset.targets);
    mlc_dataset_free(&wrong_feature_count_dataset);
    mlc_dataset_free(&wrong_output_count_dataset);
    mlc_dataset_free(&valid_dataset);
    mlc_linear_regression_free(&model);
}

int main(void) {

    test_linear_regression_create_and_free();
    test_linear_regression_predict();
    test_linear_regression_fit();
    test_linear_regression_fit_multi_output();
    test_linear_regression_invalid_arguments();
    return 0;
}