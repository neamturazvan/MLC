#include <mlc/logistic_regression.h>
#include <mlc/loss.h>
#include <math.h>
#include <assert.h>

static void test_logistic_regression_create_and_free(void) {
    MLCLogisticRegression model = {0};

    assert(mlc_logistic_regression_create(3, &model) == MLC_STATUS_SUCCESS);
    assert(model.weights.data != NULL && model.weights.rows == 3 && model.weights.columns == 1);
    assert(model.bias.data != NULL && model.bias.rows == 1 && model.bias.columns == 1);

    for (size_t i = 0; i < model.weights.rows * model.weights.columns; ++i) {
        assert(model.weights.data[i] == 0.0);
    }

    assert(model.bias.data[0] == 0.0);

    mlc_logistic_regression_free(&model);

    assert(model.weights.data == NULL && model.weights.rows == 0 && model.weights.columns == 0);
    assert(model.bias.data == NULL && model.bias.rows == 0 && model.bias.columns == 0);
}

static void test_logistic_regression_create_invalid_inputs(void) {
    MLCLogisticRegression model = {0};

    assert(mlc_logistic_regression_create(0, &model) == MLC_STATUS_INVALID_DIMENSIONS);
    assert(mlc_logistic_regression_create(3, NULL) == MLC_STATUS_INVALID_ARGUMENT);
    assert(mlc_logistic_regression_create(3, &model) == MLC_STATUS_SUCCESS);
    assert(mlc_logistic_regression_create(3, &model) == MLC_STATUS_INVALID_ARGUMENT);

    mlc_logistic_regression_free(&model);
}

static void test_logistic_regression_predict_probabilities(void) {
    MLCLogisticRegression model = {0};
    MLCMatrix features = {0};
    MLCMatrix probabilities = {0};

    assert(mlc_logistic_regression_create(2, &model) == MLC_STATUS_SUCCESS);
    assert(mlc_matrix_create(3, 2, &features) == MLC_STATUS_SUCCESS);

    model.weights.data[0] = 1.0;
    model.weights.data[1] = -1.0;
    model.bias.data[0] = 0.5;

    double feature_values[] = {
        1.0, 0.0,
        0.0, 1.0,
        2.0, 1.0
    };

    double expected_probabilities[] = {
        0.8175744761936437,
        0.3775406687981454,
        0.8175744761936437
    };

    for (size_t i = 0; i < 6; ++i) {
        features.data[i] = feature_values[i];
    }

    assert(mlc_logistic_regression_predict_probabilities(&model, &features, &probabilities) == MLC_STATUS_SUCCESS);
    assert(probabilities.rows == 3 && probabilities.columns == 1);

    for (size_t i = 0; i < 3; ++i) {
        assert(fabs(probabilities.data[i] - expected_probabilities[i]) < 1e-12);
        assert(probabilities.data[i] >= 0.0 && probabilities.data[i] <= 1.0);
    }

    assert(model.weights.data[0] == 1.0 && model.weights.data[1] == -1.0);
    assert(model.bias.data[0] == 0.5);

    for (size_t i = 0; i < 6; ++i) {
        assert(features.data[i] == feature_values[i]);
    }

    mlc_logistic_regression_free(&model);
    mlc_matrix_free(&features);
    mlc_matrix_free(&probabilities);
}

static void test_logistic_regression_predict_probabilities_invalid_inputs(void) {
    MLCLogisticRegression model = {0};
    MLCMatrix features = {0};
    MLCMatrix mismatched_features = {0};
    MLCMatrix probabilities = {0};

    assert(mlc_logistic_regression_create(2, &model) == MLC_STATUS_SUCCESS);
    assert(mlc_matrix_create(1, 2, &features) == MLC_STATUS_SUCCESS);
    assert(mlc_matrix_create(1, 3, &mismatched_features) == MLC_STATUS_SUCCESS);

    assert(mlc_logistic_regression_predict_probabilities(NULL, &features, &probabilities) == MLC_STATUS_INVALID_ARGUMENT);
    assert(mlc_logistic_regression_predict_probabilities(&model, NULL, &probabilities) == MLC_STATUS_INVALID_ARGUMENT);
    assert(mlc_logistic_regression_predict_probabilities(&model, &features, NULL) == MLC_STATUS_INVALID_ARGUMENT);
    assert(mlc_logistic_regression_predict_probabilities(&model, &mismatched_features, &probabilities) == MLC_STATUS_DIMENSIONS_MISMATCH);

    assert(mlc_matrix_create(1, 1, &probabilities) == MLC_STATUS_SUCCESS);
    assert(mlc_logistic_regression_predict_probabilities(&model, &features, &probabilities) == MLC_STATUS_INVALID_ARGUMENT);

    mlc_logistic_regression_free(&model);
    mlc_matrix_free(&features);
    mlc_matrix_free(&mismatched_features);
    mlc_matrix_free(&probabilities);
}

static void test_logistic_regression_predict(void) {
    MLCLogisticRegression model = {0};
    MLCMatrix features = {0};
    MLCMatrix classes = {0};

    assert(mlc_logistic_regression_create(2, &model) == MLC_STATUS_SUCCESS);
    assert(mlc_matrix_create(3, 2, &features) == MLC_STATUS_SUCCESS);

    model.weights.data[0] = 1.0;
    model.weights.data[1] = -1.0;
    model.bias.data[0] = 0.5;

    double feature_values[] = {
         1.0, 0.0,
         0.0, 1.0,
        -0.5, 0.0
    };

    double expected_classes[] = {
        1.0,
        0.0,
        1.0
    };

    for (size_t i = 0; i < 6; ++i) {
        features.data[i] = feature_values[i];
    }

    assert(mlc_logistic_regression_predict(&model, &features, &classes) == MLC_STATUS_SUCCESS);
    assert(classes.rows == 3 && classes.columns == 1);

    for (size_t i = 0; i < 3; ++i) {
        assert(classes.data[i] == expected_classes[i]);
    }

    assert(model.weights.data[0] == 1.0 && model.weights.data[1] == -1.0);
    assert(model.bias.data[0] == 0.5);

    mlc_logistic_regression_free(&model);
    mlc_matrix_free(&features);
    mlc_matrix_free(&classes);
}

static void test_logistic_regression_predict_invalid_inputs(void) {
    MLCLogisticRegression model = {0};
    MLCMatrix features = {0};
    MLCMatrix mismatched_features = {0};
    MLCMatrix classes = {0};

    assert(mlc_logistic_regression_create(2, &model) == MLC_STATUS_SUCCESS);
    assert(mlc_matrix_create(1, 2, &features) == MLC_STATUS_SUCCESS);
    assert(mlc_matrix_create(1, 3, &mismatched_features) == MLC_STATUS_SUCCESS);

    assert(mlc_logistic_regression_predict(NULL, &features, &classes) == MLC_STATUS_INVALID_ARGUMENT);
    assert(mlc_logistic_regression_predict(&model, NULL, &classes) == MLC_STATUS_INVALID_ARGUMENT);
    assert(mlc_logistic_regression_predict(&model, &features, NULL) == MLC_STATUS_INVALID_ARGUMENT);
    assert(mlc_logistic_regression_predict(&model, &mismatched_features, &classes) == MLC_STATUS_DIMENSIONS_MISMATCH);

    assert(mlc_matrix_create(1, 1, &classes) == MLC_STATUS_SUCCESS);
    assert(mlc_logistic_regression_predict(&model, &features, &classes) == MLC_STATUS_INVALID_ARGUMENT);

    mlc_logistic_regression_free(&model);
    mlc_matrix_free(&features);
    mlc_matrix_free(&mismatched_features);
    mlc_matrix_free(&classes);
}

static void test_logistic_regression_fit(void) {
    MLCDataset dataset = {0};
    MLCLogisticRegression model = {0};
    MLCMatrix initial_probabilities = {0};
    MLCMatrix final_probabilities = {0};
    MLCMatrix classes = {0};

    assert(mlc_dataset_create(6, 1, 1, &dataset) == MLC_STATUS_SUCCESS);
    assert(mlc_logistic_regression_create(1, &model) == MLC_STATUS_SUCCESS);

    double feature_values[] = {
        -3.0,
        -2.0,
        -1.0,
         1.0,
         2.0,
         3.0
    };

    double target_values[] = {
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0
    };

    for (size_t i = 0; i < 6; ++i) {
        dataset.features.data[i] = feature_values[i];
        dataset.targets.data[i] = target_values[i];
    }

    assert(mlc_logistic_regression_predict_probabilities(&model, &dataset.features, &initial_probabilities) == MLC_STATUS_SUCCESS);

    double initial_loss = 0.0;
    assert(mlc_loss_forward(&initial_probabilities, &dataset.targets, MLC_LOSS_BINARY_CROSS_ENTROPY, &initial_loss) == MLC_STATUS_SUCCESS);

    assert(mlc_logistic_regression_fit(&model, &dataset, 0.1, 1000) == MLC_STATUS_SUCCESS);

    assert(mlc_logistic_regression_predict_probabilities(&model, &dataset.features, &final_probabilities) == MLC_STATUS_SUCCESS);

    double final_loss = 0.0;
    assert(mlc_loss_forward(&final_probabilities, &dataset.targets, MLC_LOSS_BINARY_CROSS_ENTROPY, &final_loss) == MLC_STATUS_SUCCESS);

    assert(final_loss < initial_loss);
    assert(final_loss < 0.1);
    assert(model.weights.data[0] > 0.0);

    assert(mlc_logistic_regression_predict(&model, &dataset.features, &classes) == MLC_STATUS_SUCCESS);

    for (size_t i = 0; i < 6; ++i) {
        assert(classes.data[i] == target_values[i]);
    }

    MLCMatrix unseen_features = {0};
    MLCMatrix unseen_probabilities = {0};
    MLCMatrix unseen_classes = {0};

    assert(mlc_matrix_create(2, 1, &unseen_features) == MLC_STATUS_SUCCESS);

    unseen_features.data[0] = -0.5;
    unseen_features.data[1] = 0.5;

    assert(mlc_logistic_regression_predict_probabilities(&model, &unseen_features, &unseen_probabilities) == MLC_STATUS_SUCCESS);
    assert(unseen_probabilities.rows == 2 && unseen_probabilities.columns == 1);
    assert(unseen_probabilities.data[0] < 0.5);
    assert(unseen_probabilities.data[1] > 0.5);

    assert(mlc_logistic_regression_predict(&model, &unseen_features, &unseen_classes) == MLC_STATUS_SUCCESS);
    assert(unseen_classes.data[0] == 0.0);
    assert(unseen_classes.data[1] == 1.0);

    mlc_matrix_free(&unseen_features);
    mlc_matrix_free(&unseen_probabilities);
    mlc_matrix_free(&unseen_classes);

    mlc_matrix_free(&initial_probabilities);
    mlc_matrix_free(&final_probabilities);
    mlc_matrix_free(&classes);
    mlc_logistic_regression_free(&model);
    mlc_dataset_free(&dataset);
}

static void test_logistic_regression_fit_invalid_inputs(void) {
    MLCLogisticRegression model = {0};
    MLCLogisticRegression empty_model = {0};
    MLCDataset dataset = {0};
    MLCDataset feature_mismatch = {0};
    MLCDataset target_column_mismatch = {0};
    MLCDataset row_mismatch = {0};

    assert(mlc_logistic_regression_create(1, &model) == MLC_STATUS_SUCCESS);
    assert(mlc_dataset_create(2, 1, 1, &dataset) == MLC_STATUS_SUCCESS);
    assert(mlc_dataset_create(2, 2, 1, &feature_mismatch) == MLC_STATUS_SUCCESS);
    assert(mlc_dataset_create(2, 1, 2, &target_column_mismatch) == MLC_STATUS_SUCCESS);

    assert(mlc_matrix_create(2, 1, &row_mismatch.features) == MLC_STATUS_SUCCESS);
    assert(mlc_matrix_create(3, 1, &row_mismatch.targets) == MLC_STATUS_SUCCESS);

    assert(mlc_logistic_regression_fit(NULL, &dataset, 0.1, 10) == MLC_STATUS_INVALID_ARGUMENT);
    assert(mlc_logistic_regression_fit(&model, NULL, 0.1, 10) == MLC_STATUS_INVALID_ARGUMENT);
    assert(mlc_logistic_regression_fit(&empty_model, &dataset, 0.1, 10) == MLC_STATUS_INVALID_ARGUMENT);

    assert(mlc_logistic_regression_fit(&model, &dataset, 0.0, 10) == MLC_STATUS_INVALID_ARGUMENT);
    assert(mlc_logistic_regression_fit(&model, &dataset, -0.1, 10) == MLC_STATUS_INVALID_ARGUMENT);
    assert(mlc_logistic_regression_fit(&model, &dataset, NAN, 10) == MLC_STATUS_INVALID_ARGUMENT);
    assert(mlc_logistic_regression_fit(&model, &dataset, INFINITY, 10) == MLC_STATUS_INVALID_ARGUMENT);
    assert(mlc_logistic_regression_fit(&model, &dataset, 0.1, 0) == MLC_STATUS_INVALID_ARGUMENT);

    assert(mlc_logistic_regression_fit(&model, &feature_mismatch, 0.1, 10) == MLC_STATUS_DIMENSIONS_MISMATCH);
    assert(mlc_logistic_regression_fit(&model, &target_column_mismatch, 0.1, 10) == MLC_STATUS_INVALID_DIMENSIONS);
    assert(mlc_logistic_regression_fit(&model, &row_mismatch, 0.1, 10) == MLC_STATUS_DIMENSIONS_MISMATCH);

    mlc_logistic_regression_free(&model);
    mlc_dataset_free(&dataset);
    mlc_dataset_free(&feature_mismatch);
    mlc_dataset_free(&target_column_mismatch);
    mlc_dataset_free(&row_mismatch);
}

int main(void) {
    test_logistic_regression_create_and_free();
    test_logistic_regression_create_invalid_inputs();
    test_logistic_regression_predict_probabilities();
    test_logistic_regression_predict_probabilities_invalid_inputs();
    test_logistic_regression_predict();
    test_logistic_regression_predict_invalid_inputs();
    test_logistic_regression_fit();
    test_logistic_regression_fit_invalid_inputs();
    return 0;
}