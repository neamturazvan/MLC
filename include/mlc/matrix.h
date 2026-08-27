#ifndef MLC_MATRIX_H
#define MLC_MATRIX_H
#include <mlc/status.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>


typedef struct {
    size_t rows;
    size_t columns;
    double *data;
} MLCMatrix;

typedef double (*MLCUnaryFunction)(double);

MLCStatus mlc_matrix_create(size_t rows, size_t columns, MLCMatrix *matrix);
void mlc_matrix_free(MLCMatrix *matrix);
MLCStatus mlc_matrix_get(const MLCMatrix *matrix, size_t row, size_t column, double *out_value);
MLCStatus mlc_matrix_set(MLCMatrix *matrix, size_t row, size_t column, double new_value);
MLCStatus mlc_matrix_fill(MLCMatrix *matrix, double fill_value);
MLCStatus mlc_matrix_clone(const MLCMatrix *source_matrix, MLCMatrix *destination_matrix);
MLCStatus mlc_matrix_add(const MLCMatrix *matrix1, const MLCMatrix *matrix2, MLCMatrix *output_matrix);
MLCStatus mlc_matrix_subtraction(const MLCMatrix *matrix1, const MLCMatrix *matrix2, MLCMatrix *output_matrix);
MLCStatus mlc_matrix_scale(const MLCMatrix *matrix, double scalar, MLCMatrix *output_matrix);
MLCStatus mlc_matrix_hadamard(const MLCMatrix *matrix1, const MLCMatrix *matrix2, MLCMatrix *output_matrix);
MLCStatus mlc_matrix_transpose(const MLCMatrix *matrix, MLCMatrix *output_matrix);
MLCStatus mlc_matrix_multiply(const MLCMatrix *matrix1, const MLCMatrix *matrix2, MLCMatrix *output_matrix);
MLCStatus mlc_matrix_add_row_vector(const MLCMatrix *matrix, const MLCMatrix *vector, MLCMatrix *output_matrix);
MLCStatus mlc_matrix_sum_rows(const MLCMatrix *matrix, MLCMatrix *output_matrix);
MLCStatus mlc_matrix_apply(const MLCMatrix *matrix, MLCUnaryFunction function, MLCMatrix *output_matrix);
MLCStatus mlc_matrix_fill_random_uniform(MLCMatrix *matrix, double minimum, double maximum, uint32_t seed);

#endif //MLC_MATRIX_H