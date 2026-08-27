#include <mlc/dataset.h>
#include <assert.h>
#include <stdio.h>

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

int main(void) {
    test_dataset_create_and_free();
    test_dataset_load_csv_with_header();
    test_dataset_load_without_header();
    test_dataset_load_csv_argument_and_file_errors();
    test_dataset_load_csv_invalid_numbers();
    test_dataset_load_csv_malformed_rows();
    test_dataset_load_csv_invalid_shapes();
    test_dataset_load_csv_overlong_row();
}
