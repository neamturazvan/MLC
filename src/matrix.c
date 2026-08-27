#include <mlc/matrix.h>


MLCStatus mlc_matrix_create(size_t rows, size_t columns, MLCMatrix *matrix) {
    if (matrix == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (matrix->data != NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (rows == 0 || columns == 0) {
        return MLC_STATUS_INVALID_DIMENSIONS;
    }
    if (rows > SIZE_MAX / columns) {
        return MLC_STATUS_INVALID_DIMENSIONS;
    }
    size_t element_count = rows * columns;
    matrix->data = calloc(element_count, sizeof(double));
    if (matrix->data == NULL) {
        return MLC_STATUS_MEMORY_ALLOCATION_FAILURE;
    }
    matrix->rows = rows;
    matrix->columns = columns;
    return MLC_STATUS_SUCCESS;
}

void mlc_matrix_free(MLCMatrix *matrix) {
    if (matrix == NULL) {
        return;
    }
    matrix->rows = 0;
    matrix->columns = 0;
    free(matrix->data);
    matrix->data = NULL;
}

MLCStatus mlc_matrix_get(const MLCMatrix *matrix, size_t row, size_t column, double *out_value) {
    if (matrix == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (matrix->data == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (row >= matrix->rows || column >= matrix->columns) {
        return MLC_STATUS_OUT_OF_BOUNDS_INDEX;
    }
    if (out_value == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    *out_value = matrix->data[row * matrix->columns + column];
    return MLC_STATUS_SUCCESS;
}

MLCStatus mlc_matrix_set(MLCMatrix *matrix, size_t row, size_t column, double new_value) {
    if (matrix == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (matrix->data == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (row >= matrix->rows || column >= matrix->columns) {
        return MLC_STATUS_OUT_OF_BOUNDS_INDEX;
    }
    matrix->data[row * matrix->columns + column] = new_value;
    return MLC_STATUS_SUCCESS;
}

MLCStatus mlc_matrix_fill(MLCMatrix *matrix, double fill_value) {
    if (matrix == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (matrix->data == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    size_t matrix_size = matrix->rows * matrix->columns;
    for (size_t i = 0; i < matrix_size; ++i) {
        matrix->data[i] = fill_value;
    }
    return MLC_STATUS_SUCCESS;
}

MLCStatus mlc_matrix_clone(const MLCMatrix *source_matrix, MLCMatrix *destination_matrix) {
    if (source_matrix == NULL || source_matrix->data == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (destination_matrix == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (destination_matrix->data != NULL || destination_matrix->rows != 0 || destination_matrix->columns != 0) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    MLCStatus create_status = mlc_matrix_create(source_matrix->rows, source_matrix->columns, destination_matrix);
    if (create_status != 0) {
        return create_status;
    }
    size_t matrix_size = source_matrix->rows * source_matrix->columns;
    memcpy(destination_matrix->data, source_matrix->data, matrix_size * sizeof(double));
    return MLC_STATUS_SUCCESS;
}

MLCStatus mlc_matrix_add(const MLCMatrix *matrix1, const MLCMatrix *matrix2, MLCMatrix *output_matrix) {
    if (matrix1 == NULL || matrix2 == NULL || output_matrix == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (matrix1->data == NULL || matrix2->data == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (output_matrix->data != NULL || output_matrix->rows != 0 || output_matrix->columns != 0) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (matrix1->rows != matrix2->rows || matrix1->columns != matrix2->columns) {
        return MLC_STATUS_DIMENSIONS_MISMATCH;
    }
    MLCStatus create_status = mlc_matrix_create(matrix1->rows, matrix1->columns, output_matrix);
    if (create_status != 0) {
        return create_status;
    }
    size_t matrix_size = matrix1->rows * matrix1->columns;
    for (size_t i = 0; i < matrix_size; ++i) {
        output_matrix->data[i] = matrix1->data[i] + matrix2->data[i];
    }
    return MLC_STATUS_SUCCESS;
}

MLCStatus mlc_matrix_subtraction(const MLCMatrix *matrix1, const MLCMatrix *matrix2, MLCMatrix *output_matrix) {
    if (matrix1 == NULL || matrix2 == NULL || output_matrix == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (matrix1->data == NULL || matrix2->data == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (output_matrix->data != NULL || output_matrix->rows != 0 || output_matrix->columns != 0) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (matrix1->rows != matrix2->rows || matrix1->columns != matrix2->columns) {
        return MLC_STATUS_DIMENSIONS_MISMATCH;
    }
    MLCStatus create_status = mlc_matrix_create(matrix1->rows, matrix1->columns, output_matrix);
    if (create_status != 0) {
        return create_status;
    }
    size_t matrix_size = matrix1->rows * matrix1->columns;
    for (size_t i = 0; i < matrix_size; ++i) {
        output_matrix->data[i] = matrix1->data[i] - matrix2->data[i];
    }
    return MLC_STATUS_SUCCESS;
}

MLCStatus mlc_matrix_scale(const MLCMatrix *matrix, double scalar, MLCMatrix *output_matrix) {
    if (matrix == NULL || output_matrix == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (matrix->data == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (output_matrix->data != NULL || output_matrix->rows != 0 || output_matrix->columns != 0) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    size_t matrix_size = matrix->rows * matrix->columns;
    MLCStatus create_status = mlc_matrix_create(matrix->rows, matrix->columns, output_matrix);
    if (create_status != 0) {
        return create_status;
    }
    for (size_t i = 0; i < matrix_size; ++i) {
        output_matrix->data[i] = matrix->data[i] * scalar;
    }
    return MLC_STATUS_SUCCESS;
}


MLCStatus mlc_matrix_hadamard(const MLCMatrix *matrix1, const MLCMatrix *matrix2, MLCMatrix *output_matrix) {
    if (matrix1 == NULL || matrix2 == NULL || output_matrix == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (matrix1->data == NULL || matrix2->data == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (output_matrix->data != NULL || output_matrix->rows != 0 || output_matrix->columns != 0) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (matrix1->rows != matrix2->rows || matrix1->columns != matrix2->columns) {
        return MLC_STATUS_DIMENSIONS_MISMATCH;
    }
    size_t matrix_size = matrix1->rows * matrix1->columns;
    MLCStatus create_status = mlc_matrix_create(matrix1->rows, matrix1->columns, output_matrix);
    if (create_status != 0) {
        return create_status;
    }
    for (size_t i = 0; i < matrix_size; ++i) {
        output_matrix->data[i] = matrix1->data[i] * matrix2->data[i];
    }
    return MLC_STATUS_SUCCESS;
}

MLCStatus mlc_matrix_transpose(const MLCMatrix *matrix, MLCMatrix *output_matrix) {
    if (matrix == NULL || output_matrix == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (matrix->data == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (output_matrix->data != NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    size_t matrix_size = matrix->rows * matrix->columns;
    MLCStatus create_status = mlc_matrix_create(matrix->columns, matrix->rows, output_matrix);
    if (create_status != 0) {
        return create_status;
    }
    size_t k = 0;
    for (size_t i = 0; i < matrix->columns; ++i) {
        for (size_t j = i; j < matrix_size; j += matrix->columns) {
            output_matrix->data[k++] = matrix->data[j];
        }
    }
    return MLC_STATUS_SUCCESS;
}

MLCStatus mlc_matrix_multiply(const MLCMatrix *matrix1, const MLCMatrix *matrix2, MLCMatrix *output_matrix) {
    if (matrix1 == NULL || matrix2 == NULL || output_matrix == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (matrix1->data == NULL || matrix2->data == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (output_matrix->data != NULL || output_matrix->rows != 0 || output_matrix->columns != 0) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (matrix1->columns != matrix2->rows) {
        return MLC_STATUS_DIMENSIONS_MISMATCH;
    }
    MLCStatus create_status = mlc_matrix_create(matrix1->rows, matrix2->columns, output_matrix);
    if (create_status != 0) {
        return create_status;
    }
    size_t output_matrix_size = 0;
    for (size_t i = 0; i < matrix1->rows; ++i) {
        for (size_t j = 0; j < matrix2->columns; ++j) {
            double sum = 0;
            for (size_t k = 0; k < matrix1->columns; ++k) {
                size_t index1 = i * matrix1->columns + k;
                size_t index2 = k * matrix2->columns + j;
                sum += matrix1->data[index1] * matrix2->data[index2];
            }
            output_matrix->data[output_matrix_size++] = sum;
        }
    }
    return MLC_STATUS_SUCCESS;
}

MLCStatus mlc_matrix_add_row_vector(const MLCMatrix *matrix, const MLCMatrix *vector, MLCMatrix *output_matrix) {
    if (matrix == NULL || vector == NULL || output_matrix == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (matrix->data == NULL || vector->data == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (output_matrix->data != NULL || output_matrix->rows != 0 || output_matrix->columns != 0) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (vector->rows != 1 || vector->columns != matrix->columns) {
        return MLC_STATUS_DIMENSIONS_MISMATCH;
    }
    MLCStatus create_status = mlc_matrix_create(matrix->rows, matrix->columns, output_matrix);
    if (create_status != 0) {
        return create_status;
    }
    size_t matrix_size = matrix->rows * matrix->columns;
    size_t vector_index = 0;
    for (size_t i = 0; i < matrix_size; ++i) {
        if (vector_index == vector->columns) {
            vector_index = 0;
        }
        output_matrix->data[i] = matrix->data[i] + vector->data[vector_index++];
    }
    return MLC_STATUS_SUCCESS;
}

MLCStatus mlc_matrix_sum_rows(const MLCMatrix *matrix, MLCMatrix *output_matrix) {
    if (matrix == NULL || output_matrix == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (matrix->data == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (output_matrix->data != NULL || output_matrix->rows != 0 || output_matrix->columns != 0) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    MLCStatus create_status = mlc_matrix_create(1, matrix->columns, output_matrix);
    if (create_status != 0) {
        return create_status;
    }
    for (size_t j = 0; j < matrix->columns; ++j) {
        double sum = 0;
        for (size_t i = 0; i < matrix->rows; ++i) {
            size_t index = i * matrix->columns + j;
            sum += matrix->data[index];
        }
        output_matrix->data[j] = sum;
    }
    return MLC_STATUS_SUCCESS;
}

MLCStatus mlc_matrix_apply(const MLCMatrix *matrix, MLCUnaryFunction function, MLCMatrix *output_matrix) {
    if (matrix == NULL || output_matrix == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (matrix->data == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (output_matrix->data != NULL || output_matrix->rows != 0 || output_matrix->columns != 0) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (function == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    size_t matrix_size = matrix->rows * matrix->columns;
    MLCStatus create_status = mlc_matrix_create(matrix->rows, matrix->columns, output_matrix);
    if (create_status != 0) {
        return create_status;
    }
    for (size_t i = 0; i < matrix_size; ++i) {
        output_matrix->data[i] = function(matrix->data[i]);
    }
    return MLC_STATUS_SUCCESS;
}

MLCStatus mlc_matrix_fill_random_uniform(MLCMatrix *matrix, double minimum, double maximum, uint32_t seed) {
    if (matrix == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (matrix->data == NULL) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    if (minimum >= maximum) {
        return MLC_STATUS_INVALID_ARGUMENT;
    }
    uint32_t state = seed;
    size_t matrix_size = matrix->rows * matrix->columns;
    for (size_t i = 0; i < matrix_size; ++i) {
        state = state * 1664525u + 1013904223u;
        double unit_value = state / 4294967296.0;
        double value = minimum + unit_value * (maximum - minimum);
        matrix->data[i] = value;
    }
    return MLC_STATUS_SUCCESS;
}