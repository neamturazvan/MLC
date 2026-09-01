#include <mlc/dataset.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>

static void test_dataset_create_and_free() {
    MLCDataset test_dataset = {0};
    assert(mlc_dataset_create(3,2,1, &test_dataset) == MLC_STATUS_SUCCESS);
    assert(test_dataset.features.rows == 3 && test_dataset.features.columns == 2);
    assert(test_dataset.targets.rows == 3 && test_dataset.targets.columns == 1);
    assert(test_dataset.features.data != NULL && test_dataset.targets.data != NULL);
    mlc_dataset_free(&test_dataset);
    assert(test_dataset.features.rows == 0 && test_dataset.features.columns == 0 && test_dataset.features.data == NULL);
    assert(test_dataset.targets.rows == 0 && test_dataset.targets.columns == 0 && test_dataset.targets.data == NULL);
    mlc_dataset_free(&test_dataset);
}

static void test_dataset_load_csv_with_header() {
    MLCDataset test_dataset = {0};
    MLCStatus load_status = mlc_dataset_load_csv("valid_with_header.csv", 1,true, &test_dataset);
    assert(load_status == MLC_STATUS_SUCCESS);
    assert(test_dataset.features.rows == 3 && test_dataset.features.columns == 2);
    assert(test_dataset.targets.rows == 3 && test_dataset.targets.columns == 1);
    assert(test_dataset.features.data[0] == 1.5);
    assert(test_dataset.features.data[1] == -2.0);
    assert(test_dataset.features.data[2] == 3.0);
    assert(test_dataset.features.data[3] == 4.5);
    assert(test_dataset.features.data[4] == -1.25);
    assert(test_dataset.features.data[5] == 2.5);
    assert(test_dataset.targets.data[0] == 0);
    assert(test_dataset.targets.data[1] == 1);
    assert(test_dataset.targets.data[2] == 0);
    mlc_dataset_free(&test_dataset);
}

static void test_dataset_load_without_header() {
    MLCDataset test_dataset = {0};
    MLCStatus load_status = mlc_dataset_load_csv("valid_without_header.csv", 2, false, &test_dataset);
    assert(load_status == MLC_STATUS_SUCCESS);
    assert(test_dataset.features.rows == 2 && test_dataset.features.columns == 2);
    assert(test_dataset.targets.rows == 2 && test_dataset.targets.columns == 2);
    assert(test_dataset.features.data[0] == 0.125);
    assert(test_dataset.features.data[1] == 4.5);
    assert(test_dataset.features.data[2] == -3.0);
    assert(test_dataset.features.data[3] == 2.5);
    assert(test_dataset.targets.data[0] == -2.0);
    assert(test_dataset.targets.data[1] == 1.0);
    assert(test_dataset.targets.data[2] == 0.25);
    assert(test_dataset.targets.data[3] == 0.0);
    mlc_dataset_free(&test_dataset);
}

static void test_dataset_load_csv_argument_and_file_errors() {
    MLCDataset test_dataset = {0};
    assert(mlc_dataset_load_csv(NULL, 2, true, &test_dataset) == MLC_STATUS_INVALID_ARGUMENT);
    assert(mlc_dataset_load_csv("valid_with_header.csv", 2, true, NULL) == MLC_STATUS_INVALID_ARGUMENT);
    assert(mlc_dataset_load_csv("valid_with_header.csv", 0, true, &test_dataset) == MLC_STATUS_INVALID_DIMENSIONS);
    assert(mlc_dataset_load_csv("test_that_does_not_exist.csv", 2, true, &test_dataset) == MLC_STATUS_FILE_ERROR);
    mlc_dataset_create(2,3,1,&test_dataset);
    assert(mlc_dataset_load_csv("valid_with_header.csv", 2, true, &test_dataset) == MLC_STATUS_INVALID_ARGUMENT);
    assert(test_dataset.features.data != NULL && test_dataset.targets.data != NULL);
    mlc_dataset_free(&test_dataset);
}

static void test_dataset_load_csv_invalid_numbers() {
    MLCDataset test_dataset = {0};
    assert(mlc_dataset_load_csv("invalid_text.csv", 1, true, &test_dataset) == MLC_STATUS_PARSE_ERROR);
    assert(test_dataset.features.rows == 0 && test_dataset.features.columns == 0 && test_dataset.features.data == NULL
        && test_dataset.targets.rows == 0 && test_dataset.targets.columns == 0 && test_dataset.targets.data == NULL);
    mlc_dataset_free(&test_dataset);
    assert(mlc_dataset_load_csv("invalid_range.csv", 1, true, &test_dataset) == MLC_STATUS_PARSE_ERROR);
    assert(test_dataset.features.rows == 0 && test_dataset.features.columns == 0 && test_dataset.features.data == NULL
        && test_dataset.targets.rows == 0 && test_dataset.targets.columns == 0 && test_dataset.targets.data == NULL);
    mlc_dataset_free(&test_dataset);
    assert(mlc_dataset_load_csv("invalid_nonfinite.csv", 1, true, &test_dataset) == MLC_STATUS_PARSE_ERROR);
    assert(test_dataset.features.rows == 0 && test_dataset.features.columns == 0 && test_dataset.features.data == NULL
        && test_dataset.targets.rows == 0 && test_dataset.targets.columns == 0 && test_dataset.targets.data == NULL);
    mlc_dataset_free(&test_dataset);
}

static void test_dataset_load_csv_malformed_rows() {
    MLCDataset test_dataset = {0};
    assert(mlc_dataset_load_csv("invalid_inconsistent_columns.csv", 1, true, &test_dataset) == MLC_STATUS_PARSE_ERROR);
    assert(test_dataset.features.rows == 0 && test_dataset.features.columns == 0 && test_dataset.features.data == NULL
        && test_dataset.targets.rows == 0 && test_dataset.targets.columns == 0 && test_dataset.targets.data == NULL);
    mlc_dataset_free(&test_dataset);
    assert(mlc_dataset_load_csv("invalid_empty_field.csv", 1, true, &test_dataset) == MLC_STATUS_PARSE_ERROR);
    assert(test_dataset.features.rows == 0 && test_dataset.features.columns == 0 && test_dataset.features.data == NULL
        && test_dataset.targets.rows == 0 && test_dataset.targets.columns == 0 && test_dataset.targets.data == NULL);
    mlc_dataset_free(&test_dataset);
    assert(mlc_dataset_load_csv("invalid_blank_row.csv", 1, true, &test_dataset) == MLC_STATUS_PARSE_ERROR);
    assert(test_dataset.features.rows == 0 && test_dataset.features.columns == 0 && test_dataset.features.data == NULL
        && test_dataset.targets.rows == 0 && test_dataset.targets.columns == 0 && test_dataset.targets.data == NULL);
    mlc_dataset_free(&test_dataset);
}

static void test_dataset_load_csv_invalid_shapes() {
    MLCDataset test_dataset = {0};
    assert(mlc_dataset_load_csv("empty.csv", 1, true, &test_dataset) == MLC_STATUS_PARSE_ERROR);
    assert(mlc_dataset_load_csv("header_only.csv", 1,true, &test_dataset) == MLC_STATUS_PARSE_ERROR);
    assert(mlc_dataset_load_csv("valid_with_header.csv", 3, true, &test_dataset) == MLC_STATUS_INVALID_DIMENSIONS);
}

static void test_dataset_load_csv_overlong_row() {
    MLCDataset test_dataset = {0};
    const char *filepath = "generated_overlong.csv";
    FILE *fptr = fopen(filepath, "w+");
    for (size_t i = 0; i < 2047; ++i) {
        fprintf(fptr, "1,");
    }
    fprintf(fptr, "1\n");
    remove(filepath);
    fclose(fptr);
    assert(mlc_dataset_load_csv(filepath, 1, false, &test_dataset) == MLC_STATUS_PARSE_ERROR);

}

static void test_dataset_reproducibility_and_pairing() {
    MLCDataset test_dataset1 = {0}, test_dataset2 = {0};
    assert(mlc_dataset_load_csv("shuffle_test.csv", 1, false, &test_dataset1) == MLC_STATUS_SUCCESS);
    assert(mlc_dataset_load_csv("shuffle_test.csv", 1, false, &test_dataset2) == MLC_STATUS_SUCCESS);
    MLCStatus shuffle_status1 = mlc_dataset_shuffle(&test_dataset1, 42);
    MLCStatus shuffle_status2 = mlc_dataset_shuffle(&test_dataset2, 42);
    assert(shuffle_status1 == MLC_STATUS_SUCCESS && shuffle_status2 == MLC_STATUS_SUCCESS);
    for (size_t i = 0; i < test_dataset1.features.rows * test_dataset1.features.columns; ++i) {
        assert(test_dataset1.features.data[i] == test_dataset2.features.data[i]);
    }
    for (size_t i = 0; i < test_dataset1.targets.rows * test_dataset1.targets.columns; ++i) {
        assert(test_dataset1.targets.data[i] == test_dataset2.targets.data[i]);
    }
    bool seen[5] = {false};
    for (size_t i = 0; i < test_dataset1.features.rows; ++i) {
        size_t index_features = i * test_dataset1.features.columns;
        size_t index_targets = i * test_dataset2.targets.columns;
        assert(test_dataset1.features.data[index_features] + 100 == test_dataset1.features.data[index_features + 1]);
        assert(test_dataset1.features.data[index_features] + 1000 == test_dataset1.targets.data[index_targets]);
        assert(test_dataset1.features.data[index_features] >= 0 && test_dataset1.features.data[index_features] <= 4);
        seen[(size_t)test_dataset1.features.data[index_features]] = true;
    }
    for (size_t i = 0; i < 5; ++i) {
        assert(seen[i] == true);
    }
    mlc_dataset_free(&test_dataset1);
    mlc_dataset_free(&test_dataset2);
}

static void test_dataset_shuffle_validation() {
    MLCDataset test_dataset = {0};
    assert(mlc_dataset_shuffle(NULL, 42) == MLC_STATUS_INVALID_ARGUMENT);
    assert(mlc_dataset_shuffle(&test_dataset, 42) == MLC_STATUS_INVALID_ARGUMENT);
    mlc_matrix_create(2,3,&test_dataset.features);
    mlc_matrix_create(3,3,&test_dataset.targets);
    assert(mlc_dataset_shuffle(&test_dataset, 42) == MLC_STATUS_DIMENSIONS_MISMATCH);
    mlc_dataset_free(&test_dataset);
    mlc_matrix_create(1,4,&test_dataset.features);
    mlc_matrix_create(1,1,&test_dataset.targets);
    assert(mlc_dataset_shuffle(&test_dataset, 42) == MLC_STATUS_SUCCESS);
}

static void test_dataset_train_test_split() {
    MLCDataset original = {0}, training = {0}, testing = {0};
    assert(mlc_dataset_load_csv("shuffle_test.csv", 1, false, &original) == MLC_STATUS_SUCCESS);
    assert(mlc_dataset_train_test_split(&original, 0.7, &training, &testing) == MLC_STATUS_SUCCESS);
    assert(training.features.rows == 3 && training.features.columns == 2 && training.targets.rows == 3 && training.targets.columns == 1);
    assert(testing.features.rows == 2 && testing.features.columns == 2 && testing.targets.rows == 2 && testing.targets.columns == 1);
    for (size_t i = 0; i < training.features.rows; ++i) {
        for (size_t j = 0; j < training.features.columns; ++j) {
            size_t index = i * training.features.columns + j;
            assert(training.features.data[index] == original.features.data[index]);
        }
        for (size_t j = 0; j < training.targets.columns; ++j) {
            size_t index = i * training.targets.columns + j;
            assert(training.targets.data[index] == original.targets.data[index]);
        }
    }
    for (size_t i = 0; i < testing.features.rows; ++i) {
        for (size_t j = 0; j < testing.features.columns; ++j) {
            size_t index = i * testing.features.columns + j;
            assert(testing.features.data[index] == original.features.data[index + training.features.rows * training.features.columns]);
        }
        for (size_t j = 0; j < testing.targets.columns; ++j) {
            size_t index = i * testing.targets.columns + j;
            assert(testing.targets.data[index] == original.targets.data[index + training.targets.rows * training.targets.columns]);
        }
    }
    mlc_dataset_free(&original);
    mlc_dataset_free(&training);
    mlc_dataset_free(&testing);
}

static void test_dataset_train_test_split_invalid_arguments() {
    MLCDataset original = {0}, training = {0}, testing = {0};
    assert(mlc_dataset_load_csv("shuffle_test.csv", 1, false, &original) == MLC_STATUS_SUCCESS);
    assert(mlc_dataset_train_test_split(NULL, 0.4, &training, &testing) == MLC_STATUS_INVALID_ARGUMENT);
    assert(mlc_dataset_train_test_split(&original, 0.4, NULL, &testing) == MLC_STATUS_INVALID_ARGUMENT);
    assert(mlc_dataset_train_test_split(&original, 0.4, &training, NULL) == MLC_STATUS_INVALID_ARGUMENT);
    assert(mlc_dataset_train_test_split(&original, 0.0, &training, &testing) == MLC_STATUS_INVALID_ARGUMENT);
    assert(mlc_dataset_train_test_split(&original, 1.0, &training, &testing) == MLC_STATUS_INVALID_ARGUMENT);
    assert(mlc_dataset_train_test_split(&original, -0.4, &training, &testing) == MLC_STATUS_INVALID_ARGUMENT);
    assert(mlc_dataset_train_test_split(&original, 2, &training, &testing) == MLC_STATUS_INVALID_ARGUMENT);
    assert(mlc_dataset_train_test_split(&original, NAN, &training, &testing) == MLC_STATUS_INVALID_DIMENSIONS);
    mlc_dataset_free(&original);
}

static void test_dataset_train_test_split_invalid_state() {
    MLCDataset original = {0}, training = {0}, testing = {0};
    assert(mlc_dataset_train_test_split(&original, 0.5, &training, &testing) == MLC_STATUS_INVALID_ARGUMENT);
    mlc_dataset_load_csv("shuffle_test.csv", 1, false, &original);
    original.targets.rows = 23;
    assert(mlc_dataset_train_test_split(&original, 0.5, &training, &testing) == MLC_STATUS_DIMENSIONS_MISMATCH);
    mlc_dataset_load_csv("shuffle_test.csv", 1, false, &training);
    assert(mlc_dataset_train_test_split(&original, 0.5, &training, &testing) == MLC_STATUS_INVALID_ARGUMENT);
    mlc_dataset_free(&training);
    mlc_dataset_load_csv("shuffle_test.csv", 1, false, &testing);
    assert(mlc_dataset_train_test_split(&original, 0.5, &training, &testing) == MLC_STATUS_INVALID_ARGUMENT);
    mlc_dataset_free(&testing);
    mlc_dataset_free(&original);
    mlc_dataset_load_csv("one_sample.csv", 1, false, &original);
    assert(mlc_dataset_train_test_split(&original, 0.5, &training, &testing) == MLC_STATUS_INVALID_DIMENSIONS);
}

static void test_standard_scaler_fit() {
    MLCDataset dataset = {0};
    MLCStandardScaler scaler = {0};
    assert(mlc_dataset_load_csv("shuffle_test.csv", 1, false, &dataset) == MLC_STATUS_SUCCESS);
    assert(mlc_standard_scaler_fit(&dataset, &scaler) == MLC_STATUS_SUCCESS);
    assert(scaler.means.rows == 1 && scaler.means.columns == 2);
    assert(scaler.standard_deviations.rows == 1 && scaler.standard_deviations.columns == 2);
    assert(scaler.means.data[0] == 2);
    assert(scaler.means.data[1] == 102);
    assert(fabs(scaler.standard_deviations.data[0] - sqrt(2)) < 1e-9);
    assert(fabs(scaler.standard_deviations.data[1] - sqrt(2)) < 1e-9);
    mlc_dataset_free(&dataset);
    mlc_matrix_free(&scaler.means);
    mlc_matrix_free(&scaler.standard_deviations);

}

static void test_standard_scaler_transform() {
    MLCDataset dataset = {0}, training = {0}, testing = {0};
    MLCStandardScaler scaler = {0};
    assert(mlc_dataset_load_csv("shuffle_test.csv", 1, false, &dataset) == MLC_STATUS_SUCCESS);
    assert(mlc_dataset_train_test_split(&dataset, 0.6, &training, &testing) == MLC_STATUS_SUCCESS);
    assert(mlc_standard_scaler_fit(&training, &scaler) == MLC_STATUS_SUCCESS);
    assert(mlc_standard_scaler_transform(&scaler, &training) == MLC_STATUS_SUCCESS);
    assert(mlc_standard_scaler_transform(&scaler, &testing) == MLC_STATUS_SUCCESS);
    assert(fabs(training.features.data[0] + 1.224744871) < 1e-9);
    assert(fabs(training.features.data[1] + 1.224744871) < 1e-9);
    assert(fabs(training.features.data[2]) < 1e-9);
    assert(fabs(training.features.data[3]) < 1e-9);
    assert(fabs(training.features.data[4] - 1.224744871) < 1e-9);
    assert(fabs(training.features.data[5] - 1.224744871) < 1e-9);
    assert(fabs(testing.features.data[0] - 2.449489743) < 1e-9);
    assert(fabs(testing.features.data[1] - 2.449489743) < 1e-9);
    assert(fabs(testing.features.data[2] - 3.674234614) < 1e-9);
    assert(fabs(testing.features.data[3] - 3.674234614) < 1e-9);
}

static void test_standard_scaler_edge_cases() {
    MLCDataset dataset = {0};
    MLCStandardScaler scaler = {0};
    assert(mlc_dataset_load_csv("edge_cases_standard_scaler.csv", 1, false, &dataset) == MLC_STATUS_SUCCESS);
    assert(mlc_standard_scaler_fit(&dataset, &scaler) == MLC_STATUS_SUCCESS);
    assert(scaler.means.data[0] == 5);
    assert(scaler.standard_deviations.data[0] == 0);
    assert(mlc_standard_scaler_transform(&scaler, &dataset) == MLC_STATUS_SUCCESS);
    assert(dataset.features.data[0] == 0);
    assert(dataset.features.data[2] == 0);
    assert(dataset.features.data[4] == 0);
    assert(fabs(dataset.features.data[1] + 1.224744871) < 1e-9);
    assert(fabs(dataset.features.data[3]) < 1e-9);
    assert(fabs(dataset.features.data[5] - 1.224744871) < 1e-9);
    mlc_matrix_free(&scaler.means);
    mlc_matrix_free(&scaler.standard_deviations);
    mlc_dataset_free(&dataset);
    assert(mlc_standard_scaler_fit(NULL, &scaler) == MLC_STATUS_INVALID_ARGUMENT);
    assert(mlc_standard_scaler_fit(&dataset, NULL) == MLC_STATUS_INVALID_ARGUMENT);
    assert(mlc_dataset_load_csv("edge_cases_standard_scaler.csv", 1, false, &dataset) == MLC_STATUS_SUCCESS);
    assert(mlc_standard_scaler_fit(&dataset, &scaler) == MLC_STATUS_SUCCESS);
    assert(mlc_standard_scaler_fit(&dataset, &scaler) == MLC_STATUS_INVALID_ARGUMENT);
    assert(mlc_standard_scaler_transform(NULL, &dataset) == MLC_STATUS_INVALID_ARGUMENT);
    assert(mlc_standard_scaler_transform(&scaler, NULL) == MLC_STATUS_INVALID_ARGUMENT);
    mlc_matrix_free(&scaler.means);
    mlc_matrix_free(&scaler.standard_deviations);
    assert(mlc_standard_scaler_transform(&scaler, &dataset) == MLC_STATUS_INVALID_ARGUMENT);
    assert(mlc_standard_scaler_fit(&dataset, &scaler) == MLC_STATUS_SUCCESS);
    MLCDataset different_column_dataset = {0};
    assert(mlc_dataset_load_csv("valid_without_header.csv", 1, false, &different_column_dataset) == MLC_STATUS_SUCCESS);
    assert(mlc_standard_scaler_transform(&scaler, &different_column_dataset) == MLC_STATUS_DIMENSIONS_MISMATCH);

}

int main(void) {
    test_dataset_create_and_free();
    test_dataset_load_csv_with_header();
    test_dataset_load_without_header();
    test_dataset_load_csv_argument_and_file_errors();
    test_dataset_load_csv_invalid_numbers();
    test_dataset_load_csv_malformed_rows();
    test_dataset_load_csv_invalid_shapes();
    test_dataset_load_csv_overlong_row();
    test_dataset_reproducibility_and_pairing();
    test_dataset_shuffle_validation();
    test_dataset_train_test_split();
    test_dataset_train_test_split_invalid_arguments();
    test_dataset_train_test_split_invalid_state();
    test_standard_scaler_fit();
    test_standard_scaler_transform();
    test_standard_scaler_edge_cases();
    test_standard_scaler_edge_cases();
}
