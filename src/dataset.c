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
    if (strchr(line, '\n') == NULL) {
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

