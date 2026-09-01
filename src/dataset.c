#include <mlc/dataset.h>
#include <stdio.h>
#include <errno.h>
#include <ctype.h>
#include <math.h>

#define MLC_CSV_LINE_CAPACITY 4096

MLCStatus mlc_dataset_create(size_t sample_count, size_t feature_count, size_t target_count, MLCDataset *output_dataset) {
    if (output_dataset == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (sample_count == 0 || feature_count == 0 || target_count == 0) {
        return MLC_STATUS_INVALID_DIMENSIONS;
    }
    if (output_dataset->features.data != NULL || output_dataset->features.columns != 0 || output_dataset->features.rows != 0 ||
        output_dataset->targets.data != NULL || output_dataset->targets.columns != 0 || output_dataset->targets.rows != 0) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    MLCStatus create_features_status = mlc_matrix_create(sample_count, feature_count, &output_dataset->features);
    if (create_features_status != MLC_STATUS_SUCCESS) {
        return create_features_status;
    }
    MLCStatus create_targets_status = mlc_matrix_create(sample_count, target_count, &output_dataset->targets);
    if (create_targets_status != MLC_STATUS_SUCCESS) {
        mlc_matrix_free(&output_dataset->features);
        return create_targets_status;
    }
    return MLC_STATUS_SUCCESS;
}

void mlc_dataset_free(MLCDataset *dataset) {
    if (dataset == NULL) {
        return;
    }
    mlc_matrix_free(&dataset->features);
    mlc_matrix_free(&dataset->targets);
}

MLCStatus mlc_dataset_load_csv(const char *filepath, size_t target_count, bool has_header, MLCDataset *output_dataset) {
    if (filepath == NULL || output_dataset == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (target_count == 0) {
        return MLC_STATUS_INVALID_DIMENSIONS;
    }
    if (output_dataset->features.data != NULL || output_dataset->features.columns != 0 || output_dataset->features.rows != 0 ||
        output_dataset->targets.data != NULL || output_dataset->targets.columns != 0 || output_dataset->targets.rows != 0) {
        return MLC_STATUS_INVALID_ARGUMENT;
        }
    FILE *file = fopen(filepath, "r");
    if (file == NULL) {
        return MLC_STATUS_FILE_ERROR;
    }
    char line[MLC_CSV_LINE_CAPACITY];
    size_t sample_count = 0;
    size_t feature_count = 0;
    size_t total_column_count = 0;
    if (!has_header) {
        sample_count++;
    }
    if (fgets(line, sizeof(line), file) == NULL) {
        fclose(file);
        if (ferror(file)) {
            return MLC_STATUS_FILE_ERROR;
        }
        return MLC_STATUS_PARSE_ERROR;
    }
    if (strchr(line, '\n') == NULL && !feof(file)) {
        fclose(file);
        return MLC_STATUS_PARSE_ERROR;
    }
    size_t commas_count = 0;
    for (size_t i = 0; line[i] != '\0'; ++i) {
        if (line[i] == ',') {
            commas_count++;
        }
    }
    total_column_count = commas_count + 1;
    while (fgets(line, sizeof(line), file) != NULL) {
        if (strchr(line, '\n') == NULL && !feof(file)) {
            fclose(file);
            return MLC_STATUS_PARSE_ERROR;
        }
        sample_count++;
        commas_count = 0;
        for (size_t i = 0; line[i] != '\0'; ++i) {
            if (line[i] == ',') {
                commas_count++;
            }
        }
        if (commas_count + 1 != total_column_count) {
            fclose(file);
            return MLC_STATUS_PARSE_ERROR;
        }
    }
    if (ferror(file)) {
        fclose(file);
        return MLC_STATUS_FILE_ERROR;
    }
    if (total_column_count <= target_count) {
        fclose(file);
        return MLC_STATUS_INVALID_DIMENSIONS;
    }
    if (sample_count == 0) {
        fclose(file);
        return MLC_STATUS_PARSE_ERROR;
    }
    feature_count = total_column_count - target_count;
    MLCStatus create_status = mlc_dataset_create(sample_count, feature_count, target_count, output_dataset);
    if (create_status != MLC_STATUS_SUCCESS) {
        fclose(file);
        return create_status;
    }
    rewind(file);
    if (has_header) {
        fgets(line, sizeof(line),file);
    }
    size_t row_index = 0;
    size_t column_index;
    while (fgets(line,sizeof(line),file) != NULL) {
        if (strchr(line, '\n') == NULL && !feof(file)) {
            fclose(file);
            return MLC_STATUS_PARSE_ERROR;
        }
        char line_copy[MLC_CSV_LINE_CAPACITY];
        column_index = 0;
        strcpy(line_copy, line);
        char *token = strtok(line_copy, ",");
        while (token != NULL) {
            char *end_pointer;
            errno = 0;
            double parsed_number = strtod(token, &end_pointer);
            if (end_pointer == token) {
                fclose(file);
                mlc_dataset_free(output_dataset);
                return MLC_STATUS_PARSE_ERROR;
            }
            while (isspace((unsigned char)*end_pointer)) {
                end_pointer++;
            }
            if (*end_pointer != '\0') {
                fclose(file);
                mlc_dataset_free(output_dataset);
                return MLC_STATUS_PARSE_ERROR;
            }
            if (errno == ERANGE) {
                fclose(file);
                mlc_dataset_free(output_dataset);
                return MLC_STATUS_PARSE_ERROR;
            }
            if (!isfinite(parsed_number)) {
                fclose(file);
                mlc_dataset_free(output_dataset);
                return MLC_STATUS_PARSE_ERROR;
            }
            if (column_index < feature_count) {
                MLCStatus set_status = mlc_matrix_set(&output_dataset->features, row_index, column_index, parsed_number);
                if (set_status != MLC_STATUS_SUCCESS) {
                    fclose(file);
                    mlc_dataset_free(output_dataset);
                    return set_status;
                }
            }
            else {
                MLCStatus set_status = mlc_matrix_set(&output_dataset->targets, row_index, column_index - feature_count, parsed_number);
                if (set_status != MLC_STATUS_SUCCESS) {
                    fclose(file);
                    mlc_dataset_free(output_dataset);
                    return set_status;
                }
            }
            column_index++;
            token = strtok(NULL, ",");
        }
        if (column_index != total_column_count) {
            fclose(file);
            mlc_dataset_free(output_dataset);
            return MLC_STATUS_PARSE_ERROR;
        }
        row_index++;
    }
    if (row_index != sample_count) {
        fclose(file);
        mlc_dataset_free(output_dataset);
        return MLC_STATUS_PARSE_ERROR;
    }
    if (ferror(file)) {
        fclose(file);
        mlc_dataset_free(output_dataset);
        return MLC_STATUS_FILE_ERROR;
    }
    fclose(file);
    return MLC_STATUS_SUCCESS;
}

MLCStatus mlc_dataset_shuffle(MLCDataset *dataset, uint32_t seed) {
    if (dataset == NULL) {
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
    if (dataset->features.rows == 1) {
        return MLC_STATUS_SUCCESS;
    }
    uint32_t state = seed;
    size_t row_count = dataset->features.rows;
    for (size_t i = row_count - 1; i > 0; --i) {
        state = state * 1664525u + 1013904223u;
        size_t j = state % (i + 1);
        for (size_t k = 0; k < dataset->features.columns; ++k) {
            size_t index_i = i * dataset->features.columns + k;
            size_t index_j = j * dataset->features.columns + k;
            double aux = dataset->features.data[index_i];
            dataset->features.data[index_i] = dataset->features.data[index_j];
            dataset->features.data[index_j] = aux;
        }
        for (size_t k = 0; k < dataset->targets.columns; ++k) {
            size_t index_i = i * dataset->targets.columns + k;
            size_t index_j = j * dataset->targets.columns + k;
            double aux = dataset->targets.data[index_i];
            dataset->targets.data[index_i] = dataset->targets.data[index_j];
            dataset->targets.data[index_j] = aux;
        }
    }
    return MLC_STATUS_SUCCESS;
}

MLCStatus mlc_dataset_train_test_split(const MLCDataset *dataset, double training_ratio, MLCDataset *training_dataset, MLCDataset *testing_dataset) {
    if (dataset == NULL || training_dataset == NULL || testing_dataset == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (dataset->features.data == NULL || dataset->targets.data == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (dataset->features.rows == 0 || dataset->features.columns == 0 || dataset->targets.rows == 0 || dataset->targets.columns == 0) {
        return MLC_STATUS_INVALID_DIMENSIONS;
    }
    if (training_dataset->features.data != NULL || training_dataset->features.columns != 0 || training_dataset->features.rows != 0 ||
        training_dataset->targets.data != NULL || training_dataset->targets.columns != 0 || training_dataset->targets.rows != 0) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (testing_dataset->features.data != NULL || testing_dataset->features.columns != 0 || testing_dataset->features.rows != 0 ||
        testing_dataset->targets.data != NULL || testing_dataset->targets.columns != 0 || testing_dataset->targets.rows != 0) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (training_ratio <= 0.0 || training_ratio >= 1.0) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (dataset->features.rows != dataset->targets.rows) {
        return MLC_STATUS_DIMENSIONS_MISMATCH;
    }
    size_t training_sample_count = (size_t)(training_ratio * dataset->features.rows);
    size_t testing_sample_count = dataset->features.rows - training_sample_count;
    if (training_sample_count == 0 || testing_sample_count == 0) {
        return MLC_STATUS_INVALID_DIMENSIONS;
    }
    MLCStatus create_status = mlc_dataset_create(training_sample_count, dataset->features.columns, dataset->targets.columns, training_dataset);
    if (create_status != MLC_STATUS_SUCCESS) {
        return create_status;
    }
    create_status = mlc_dataset_create(testing_sample_count, dataset->features.columns, dataset->targets.columns, testing_dataset);
    if (create_status != MLC_STATUS_SUCCESS) {
        mlc_dataset_free(training_dataset);
        return create_status;
    }
    memcpy(training_dataset->features.data, dataset->features.data, training_sample_count * dataset->features.columns * sizeof(double));
    memcpy(training_dataset->targets.data, dataset->targets.data, training_sample_count * dataset->targets.columns * sizeof(double));

    memcpy(testing_dataset->features.data, dataset->features.data + training_sample_count * dataset->features.columns, testing_sample_count * dataset->features.columns * sizeof(double));
    memcpy(testing_dataset->targets.data, dataset->targets.data + training_sample_count * dataset->targets.columns, testing_sample_count * dataset->targets.columns * sizeof(double));
    return MLC_STATUS_SUCCESS;
}


MLCStatus mlc_standard_scaler_fit(const MLCDataset *dataset, MLCStandardScaler *scaler) {
    if (dataset == NULL || scaler == NULL) {
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
    if (scaler->means.data != NULL || scaler->means.rows != 0 || scaler->means.columns != 0 ||
        scaler->standard_deviations.data != NULL || scaler->standard_deviations.rows != 0 || scaler->standard_deviations.columns != 0) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    MLCStatus create_status = mlc_matrix_create(1, dataset->features.columns, &scaler->means);
    if (create_status != MLC_STATUS_SUCCESS) {
        return create_status;
    }
    create_status = mlc_matrix_create(1, dataset->features.columns, &scaler->standard_deviations);
    if (create_status != MLC_STATUS_SUCCESS) {
        mlc_matrix_free(&scaler->means);
        return create_status;
    }
    for (size_t i = 0; i < dataset->features.columns; ++i) {
        double sum = 0;
        for (size_t j = 0; j < dataset->features.rows; ++j) {
            size_t index = j * dataset->features.columns + i;
            sum += dataset->features.data[index];
        }
        scaler->means.data[i] = sum / (double)dataset->features.rows;
    }
    for (size_t i = 0; i < dataset->features.columns; ++i) {
        double square_differences_sum = 0;
        for (size_t j = 0; j < dataset->features.rows; ++j) {
            size_t index = j * dataset->features.columns + i;
            double difference = dataset->features.data[index] - scaler->means.data[i];
            square_differences_sum += difference * difference;
        }
        double variance = square_differences_sum / (double)dataset->features.rows;
        double standard_deviation = sqrt(variance);
        scaler->standard_deviations.data[i] = standard_deviation;
    }
    return MLC_STATUS_SUCCESS;
}

void mlc_standard_scaler_free(MLCStandardScaler *scaler) {
    if (scaler == NULL) {
        return;
    }
    mlc_matrix_free(&scaler->means);
    mlc_matrix_free(&scaler->standard_deviations);
}

MLCStatus mlc_standard_scaler_transform(const MLCStandardScaler *scaler, MLCDataset *dataset) {
    if (dataset == NULL || scaler == NULL) {
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
    if (scaler->means.data == NULL || scaler->standard_deviations.data == NULL){
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (scaler->means.rows != 1 || scaler->standard_deviations.rows != 1) {
        return MLC_STATUS_INVALID_DIMENSIONS;
    }
    if (scaler->means.columns == 0 || scaler->standard_deviations.columns == 0) {
        return MLC_STATUS_INVALID_DIMENSIONS;
    }
    if (scaler->means.columns != scaler->standard_deviations.columns) {
        return MLC_STATUS_DIMENSIONS_MISMATCH;
    }
    if (scaler->means.columns != dataset->features.columns) {
        return MLC_STATUS_DIMENSIONS_MISMATCH;
    }
    for (size_t i = 0; i < dataset->features.rows; ++i) {
        for (size_t j = 0; j < dataset->features.columns; ++j) {
            size_t index = i * dataset->features.columns + j;
            double centered = dataset->features.data[index] - scaler->means.data[j];
            if (scaler->standard_deviations.data[j] == 0) {
                dataset->features.data[index] = centered;
            }
            else {
                dataset->features.data[index] = centered / scaler->standard_deviations.data[j];
            }
        }
    }
    return MLC_STATUS_SUCCESS;
}
