#include <mlc/matrix.h>
#include <assert.h>


static void test_matrix_create_and_free(void) {
    MLCMatrix test_matrix = {0};
    MLCStatus create_status = mlc_matrix_create(2, 3, &test_matrix);
    assert(create_status == MLC_STATUS_SUCCESS);
    assert(test_matrix.rows == 2);
    assert(test_matrix.columns == 3);
    assert(test_matrix.data != NULL);
    for (size_t i = 0; i < test_matrix.rows * test_matrix.columns; ++i)
        assert(test_matrix.data[i] == 0.0);
    mlc_matrix_free(&test_matrix);
    assert(test_matrix.rows == 0);
    assert(test_matrix.columns == 0);
    assert(test_matrix.data == NULL);
    mlc_matrix_free(&test_matrix);
}

static void test_matrix_create_rejects_zero_dimensions() {
    MLCMatrix test_matrix1 = {0}, test_matrix2 = {0};
    MLCStatus create_status1 = mlc_matrix_create(0, 3, &test_matrix1);
    MLCStatus create_status2 = mlc_matrix_create(3, 0, &test_matrix2);
    assert(create_status1 == MLC_STATUS_INVALID_DIMENSIONS);
    assert(create_status2 == MLC_STATUS_INVALID_DIMENSIONS);
    mlc_matrix_free(&test_matrix1);
    mlc_matrix_free(&test_matrix2);
}

static void test_matrix_null_pointer() {
    MLCStatus create_status = mlc_matrix_create(2, 3, NULL);
    assert(create_status == MLC_STATUS_INVALID_ARGUMENT);
    mlc_matrix_free(NULL);
}

static void test_matrix_create_rejects_overflow() {
    MLCMatrix test_matrix = {0};
    MLCStatus create_status = mlc_matrix_create(SIZE_MAX, 2, &test_matrix);
    assert(create_status == MLC_STATUS_INVALID_DIMENSIONS);
}

static void test_matrix_create_rejects_allocated_matrix() {
    MLCMatrix test_matrix = {0};
    mlc_matrix_create(2,3, &test_matrix);
    MLCStatus create_status = mlc_matrix_create(4, 5, &test_matrix);
    assert(create_status == MLC_STATUS_INVALID_ARGUMENT);
    mlc_matrix_free(&test_matrix);
}

static void test_matrix_get_and_set() {
    MLCMatrix test_matrix = {0};
    mlc_matrix_create(2,3, &test_matrix);
    MLCStatus set_status = mlc_matrix_set(&test_matrix,1, 2, 7.5);
    assert(set_status == MLC_STATUS_SUCCESS);
    double output_value;
    mlc_matrix_get(&test_matrix, 1, 2, &output_value);
    assert(output_value == 7.5);
    mlc_matrix_get(&test_matrix, 0, 0, &output_value);
    assert(output_value == 0.0);
    mlc_matrix_free(&test_matrix);
}

static void test_matrix_access_rejects_invalid_arguments() {
    double output_value;
    assert(mlc_matrix_set(NULL,2, 3, 3.4) == MLC_STATUS_INVALID_ARGUMENT);
    assert(mlc_matrix_get(NULL, 2, 3, &output_value) == MLC_STATUS_INVALID_ARGUMENT);
    MLCMatrix test_matrix = {0};
    assert(mlc_matrix_set(&test_matrix, 3, 4, 3.5) == MLC_STATUS_INVALID_ARGUMENT);
    assert(mlc_matrix_get(&test_matrix, 4,5, &output_value) == MLC_STATUS_INVALID_ARGUMENT);
    mlc_matrix_create(2,3, &test_matrix);
    assert(mlc_matrix_get(&test_matrix, 1,2, NULL) == MLC_STATUS_INVALID_ARGUMENT);
    mlc_matrix_free(&test_matrix);
}

static void test_matrix_access_rejects_out_of_bounds_indices() {
    MLCMatrix test_matrix = {0};
    double output_value;
    mlc_matrix_create(2,3,&test_matrix);
    assert(mlc_matrix_get(&test_matrix, 2, 0, &output_value) == MLC_STATUS_OUT_OF_BOUNDS_INDEX);
    assert(mlc_matrix_set(&test_matrix, 2, 0 ,3.4) == MLC_STATUS_OUT_OF_BOUNDS_INDEX);
    assert(mlc_matrix_get(&test_matrix, 0, 4, &output_value) == MLC_STATUS_OUT_OF_BOUNDS_INDEX);
    assert(mlc_matrix_set(&test_matrix, 0, 4, 3.4) == MLC_STATUS_OUT_OF_BOUNDS_INDEX);
    mlc_matrix_free(&test_matrix);
}

static void test_matrix_successful_fill() {
    MLCMatrix test_matrix = {0};
    mlc_matrix_create(2,3,&test_matrix);
    MLCStatus fill_status = mlc_matrix_fill(&test_matrix, 4.5);
    size_t matrix_size = test_matrix.rows * test_matrix.columns;
    for (size_t i = 0; i < matrix_size; i++) {
        assert(test_matrix.data[i] == 4.5);
    }
    mlc_matrix_free(&test_matrix);
}

static void test_matrix_invalid_matrix_fill() {
    assert(mlc_matrix_fill(NULL, 3.4) == MLC_STATUS_INVALID_ARGUMENT);
    MLCMatrix test_matrix = {0};
    assert(mlc_matrix_fill(&test_matrix, 2.4) == MLC_STATUS_INVALID_ARGUMENT);
}

static void test_matrix_successful_deep_clone() {
    MLCMatrix source_matrix = {0}, destination_matrix = {0};
    mlc_matrix_create(2,3, &source_matrix);
    mlc_matrix_fill(&source_matrix, 3.4);
    MLCStatus clone_status = mlc_matrix_clone(&source_matrix, &destination_matrix);
    assert(clone_status == MLC_STATUS_SUCCESS);
    assert(source_matrix.data != destination_matrix.data);
    size_t matrix_size = source_matrix.rows * source_matrix.columns;
    for (size_t i = 0; i < matrix_size; ++i) {
        assert(source_matrix.data[i] == destination_matrix.data[i]);
    }
    destination_matrix.data[3] = 32;
    assert(source_matrix.data[3] != 32);
    mlc_matrix_free(&source_matrix);
    mlc_matrix_free(&destination_matrix);
}

static void test_matrix_invalid_clone() {
    MLCMatrix source_matrix = {0}, destination_matrix = {0};
    mlc_matrix_create(2,3, &source_matrix);
    assert(mlc_matrix_clone(NULL, &destination_matrix) == MLC_STATUS_INVALID_ARGUMENT);
    assert(mlc_matrix_clone(&source_matrix, NULL) == MLC_STATUS_INVALID_ARGUMENT);
    mlc_matrix_free(&source_matrix);
    assert(mlc_matrix_clone(&source_matrix, &destination_matrix) == MLC_STATUS_INVALID_ARGUMENT);
    mlc_matrix_create(2,3, &destination_matrix);
    mlc_matrix_create(2,3,&source_matrix);
    assert(mlc_matrix_clone(&source_matrix, &destination_matrix) == MLC_STATUS_INVALID_ARGUMENT);
    mlc_matrix_free(&source_matrix);
    mlc_matrix_free(&destination_matrix);
}

static void test_matrix_successful_addition() {
    MLCMatrix matrix1 = {0}, matrix2 = {0}, output_matrix = {0};
    mlc_matrix_create(2,2,&matrix1);
    mlc_matrix_create(2,2,&matrix2);
    mlc_matrix_set(&matrix1, 0, 0, 1);
    mlc_matrix_set(&matrix1, 0, 1, 2);
    mlc_matrix_set(&matrix1, 1, 0, 3);
    mlc_matrix_set(&matrix1, 1, 1, 4);
    mlc_matrix_set(&matrix2, 0 , 0, 5);
    mlc_matrix_set(&matrix2, 0 , 1, 6);
    mlc_matrix_set(&matrix2, 1 , 0, 7);
    mlc_matrix_set(&matrix2, 1 , 1, 8);
    MLCStatus add_status = mlc_matrix_add(&matrix1, &matrix2, &output_matrix);
    assert(add_status == MLC_STATUS_SUCCESS);
    assert(output_matrix.rows == matrix1.rows && output_matrix.columns == matrix1.columns);
    double elem1, elem2, elem3, elem4;
    mlc_matrix_get(&output_matrix, 0 , 0, &elem1);
    mlc_matrix_get(&output_matrix, 0 , 1, &elem2);
    mlc_matrix_get(&output_matrix, 1 , 0, &elem3);
    mlc_matrix_get(&output_matrix, 1 , 1, &elem4);
    assert(elem1 == 6);
    assert(elem2 == 8);
    assert(elem3 == 10);
    assert(elem4 == 12);
    mlc_matrix_free(&matrix1);
    mlc_matrix_free(&matrix2);
    mlc_matrix_free(&output_matrix);
}

static void test_matrix_dimensions_mismatch() {
    MLCMatrix matrix1 = {0}, matrix2 = {0}, output_matrix = {0};
    mlc_matrix_create(2,4, &matrix1);
    mlc_matrix_create(2,3,&matrix2);
    assert(mlc_matrix_add(&matrix1, &matrix2, &output_matrix) == MLC_STATUS_DIMENSIONS_MISMATCH);
    mlc_matrix_free(&matrix1);
    mlc_matrix_free(&matrix2);
    mlc_matrix_free(&output_matrix);
}

static void test_matrix_scale() {
    MLCMatrix test_matrix = {0}, output_matrix = {0};
    mlc_matrix_create(2,2, &test_matrix);
    mlc_matrix_fill(&test_matrix, 4);
    mlc_matrix_scale(&test_matrix, -2, &output_matrix);
    size_t matrix_size = test_matrix.rows * test_matrix.columns;
    for (size_t i = 0; i < matrix_size; ++i) {
        assert(output_matrix.data[i] == -8);
    }
    mlc_matrix_free(&test_matrix);
    mlc_matrix_free(&output_matrix);
}

static void test_matrix_scale_invalid_input() {
    MLCMatrix input_matrix = {0}, output_matrix = {0};
    assert(mlc_matrix_scale(NULL, 3, &output_matrix) == MLC_STATUS_INVALID_ARGUMENT);
    mlc_matrix_create(2,3,&input_matrix);
    assert(mlc_matrix_scale(&input_matrix, 3, NULL) == MLC_STATUS_INVALID_ARGUMENT);
    mlc_matrix_free(&input_matrix);
    assert(mlc_matrix_scale(&input_matrix, 3, &output_matrix) == MLC_STATUS_INVALID_ARGUMENT);
    mlc_matrix_create(2,3,&output_matrix);
    assert(mlc_matrix_scale(&input_matrix, 3, &output_matrix) == MLC_STATUS_INVALID_ARGUMENT);
    mlc_matrix_free(&input_matrix);
    mlc_matrix_free(&output_matrix);
}

static void test_matrix_hadamard() {
    MLCMatrix test_matrix1 = {0}, test_matrix2 = {0}, output_matrix = {0};
    mlc_matrix_create(1,2, &test_matrix1);
    mlc_matrix_create(1,2, &test_matrix2);
    mlc_matrix_set(&test_matrix1, 0 , 0 , 1);
    mlc_matrix_set(&test_matrix1, 0 , 1 , 2);
    mlc_matrix_set(&test_matrix2, 0 , 0 , 5);
    mlc_matrix_set(&test_matrix2, 0 , 1 , 6);
    mlc_matrix_hadamard(&test_matrix1, &test_matrix2, &output_matrix);
    assert(output_matrix.data[0] == 5 && output_matrix.data[1] == 12);
    mlc_matrix_free(&test_matrix1);
    mlc_matrix_free(&test_matrix2);
    mlc_matrix_free(&output_matrix);
}

static void test_matrix_hadamard_dimension_mismatch() {
    MLCMatrix test_matrix1 = {0}, test_matrix2 = {0}, output_matrix = {0};
    mlc_matrix_create(1,1, &test_matrix1);
    mlc_matrix_create(1,2, &test_matrix2);
    mlc_matrix_set(&test_matrix1, 0 , 0 , 1);
    mlc_matrix_set(&test_matrix2, 0 , 0 , 5);
    mlc_matrix_set(&test_matrix2, 0 , 1 , 6);
    MLCStatus hadamard_status = mlc_matrix_hadamard(&test_matrix1, &test_matrix2, &output_matrix);
    assert(hadamard_status == MLC_STATUS_DIMENSIONS_MISMATCH);
    mlc_matrix_free(&test_matrix1);
    mlc_matrix_free(&test_matrix2);
}

static void test_matrix_transpose() {
    MLCMatrix test_matrix = {0}, output_matrix = {0};
    mlc_matrix_create(2,3, &test_matrix);
    mlc_matrix_set(&test_matrix, 0, 0, 1);
    mlc_matrix_set(&test_matrix, 0, 1, 2);
    mlc_matrix_set(&test_matrix, 0, 2, 3);
    mlc_matrix_set(&test_matrix, 1, 0, 4);
    mlc_matrix_set(&test_matrix, 1, 1, 5);
    mlc_matrix_set(&test_matrix, 1, 2, 6);
    mlc_matrix_transpose(&test_matrix, &output_matrix);
    assert(output_matrix.columns == test_matrix.rows && output_matrix.rows == test_matrix.columns);
    assert(output_matrix.data[0] == 1);
    assert(output_matrix.data[1] == 4);
    assert(output_matrix.data[2] == 2);
    assert(output_matrix.data[3] == 5);
    assert(output_matrix.data[4] == 3);
    assert(output_matrix.data[5] == 6);
    mlc_matrix_free(&test_matrix);
    mlc_matrix_free(&output_matrix);
}

static void test_matrix_transpose_invalid_argument() {
    MLCMatrix test_matrix = {0};
    mlc_matrix_create(1,2,&test_matrix);
    assert(mlc_matrix_transpose(&test_matrix, NULL) == MLC_STATUS_INVALID_ARGUMENT);
}

static void test_matrix_multiply() {
    MLCMatrix test_matrix1 = {0}, test_matrix2 = {0}, output_matrix = {0};
    mlc_matrix_create(2,3, &test_matrix1);
    mlc_matrix_create(3,2,&test_matrix2);
    mlc_matrix_set(&test_matrix1, 0, 0, 1);
    mlc_matrix_set(&test_matrix1, 0, 1, 2);
    mlc_matrix_set(&test_matrix1, 0, 2, 3);
    mlc_matrix_set(&test_matrix1, 1, 0, 4);
    mlc_matrix_set(&test_matrix1, 1, 1, 5);
    mlc_matrix_set(&test_matrix1, 1, 2, 6);
    mlc_matrix_set(&test_matrix2, 0 ,0, 7);
    mlc_matrix_set(&test_matrix2, 0 ,1, 8);
    mlc_matrix_set(&test_matrix2, 1 ,0, 9);
    mlc_matrix_set(&test_matrix2, 1 ,1, 10);
    mlc_matrix_set(&test_matrix2, 2 ,0, 11);
    mlc_matrix_set(&test_matrix2, 2 ,1, 12);
    mlc_matrix_multiply(&test_matrix1, &test_matrix2, &output_matrix);
    assert(output_matrix.rows == test_matrix1.rows && output_matrix.columns == test_matrix2.columns);
    assert(output_matrix.data[0] == 58 && output_matrix.data[1] == 64 && output_matrix.data[2] == 139 && output_matrix.data[3] == 154);
    mlc_matrix_free(&test_matrix1);
    mlc_matrix_free(&test_matrix2);
    mlc_matrix_free(&output_matrix);
}

static void test_matrix_multiply_dimension_mismatch() {
    MLCMatrix test_matrix1 = {0}, test_matrix2 = {0}, output_matrix = {0};
    mlc_matrix_create(2,3, &test_matrix1);
    mlc_matrix_create(2,2,&test_matrix2);
    MLCStatus multiply_status = mlc_matrix_multiply(&test_matrix1, &test_matrix2, &output_matrix);
    assert(multiply_status == MLC_STATUS_DIMENSIONS_MISMATCH);
    assert(output_matrix.data == NULL);
}

static void test_matrix_add_row_vector() {
    MLCMatrix test_matrix = {0}, test_vector = {0}, output_matrix = {0};
    mlc_matrix_create(2,3,&test_matrix);
    mlc_matrix_create(1, 3, &test_vector);
    mlc_matrix_set(&test_matrix, 0, 0, 1);
    mlc_matrix_set(&test_matrix, 0, 1, 2);
    mlc_matrix_set(&test_matrix, 0, 2, 3);
    mlc_matrix_set(&test_matrix, 1, 0, 4);
    mlc_matrix_set(&test_matrix, 1, 1,5);
    mlc_matrix_set(&test_matrix, 1, 2, 6);
    mlc_matrix_set(&test_vector,0, 0, 10);
    mlc_matrix_set(&test_vector,0, 1, 20);
    mlc_matrix_set(&test_vector,0, 2, 30);
    mlc_matrix_add_row_vector(&test_matrix, &test_vector, &output_matrix);
    assert(output_matrix.data[0] == 11);
    assert(output_matrix.data[1] == 22);
    assert(output_matrix.data[2] == 33);
    assert(output_matrix.data[3] == 14);
    assert(output_matrix.data[4] == 25);
    assert(output_matrix.data[5] == 36);
    mlc_matrix_free(&test_matrix);
    mlc_matrix_free(&test_vector);
    mlc_matrix_free(&output_matrix);
}

static void test_matrix_add_row_vector_dimensions_mismatch() {
    MLCMatrix test_matrix = {0}, test_vector = {0}, output_matrix = {0};
    mlc_matrix_create(2,3,&test_matrix);
    mlc_matrix_create(1,2,&test_vector);
    assert(mlc_matrix_add_row_vector(&test_matrix, &test_vector, &output_matrix) == MLC_STATUS_DIMENSIONS_MISMATCH);
}

static void test_matrix_sum_rows() {
    MLCMatrix test_matrix = {0}, output_matrix = {0};
    mlc_matrix_create(2,3,&test_matrix);
    mlc_matrix_set(&test_matrix, 0, 0, 1);
    mlc_matrix_set(&test_matrix, 0, 1, 2);
    mlc_matrix_set(&test_matrix, 0, 2, 3);
    mlc_matrix_set(&test_matrix, 1, 0, 4);
    mlc_matrix_set(&test_matrix, 1, 1,5);
    mlc_matrix_set(&test_matrix, 1, 2, 6);
    mlc_matrix_sum_rows(&test_matrix, &output_matrix);
    assert(output_matrix.data[0] == 5);
    assert(output_matrix.data[1] == 7);
    assert(output_matrix.data[2] == 9);
    mlc_matrix_free(&test_matrix);
    mlc_matrix_free(&output_matrix);
}

static double square(double value) {
    return value * value;
}

static void test_matrix_apply() {
    MLCMatrix test_matrix = {0}, output_matrix = {0};
    mlc_matrix_create(2,2,&test_matrix);
    mlc_matrix_set(&test_matrix, 0,0,-2);
    mlc_matrix_set(&test_matrix, 0,1,3);
    mlc_matrix_set(&test_matrix, 1,0,0);
    mlc_matrix_set(&test_matrix, 1,1,-4);
    mlc_matrix_apply(&test_matrix, square, &output_matrix);
    assert(output_matrix.data[0] == 4);
    assert(output_matrix.data[1] == 9);
    assert(output_matrix.data[2] == 0);
    assert(output_matrix.data[3] == 16);
}

static void test_matrix_apply_null_function() {
    MLCMatrix test_matrix = {0}, output_matrix = {0};
    mlc_matrix_create(2,2,&test_matrix);
    assert(mlc_matrix_apply(&test_matrix, NULL, &output_matrix) == MLC_STATUS_INVALID_ARGUMENT);
}

static void test_matrix_fill_random_uniform() {
    MLCMatrix test_matrix1 = {0}, test_matrix2 = {0};
    mlc_matrix_create(2,3,&test_matrix1);
    mlc_matrix_create(2,3,&test_matrix2);
    mlc_matrix_fill_random_uniform(&test_matrix1, -0.5,0.5,32);
    mlc_matrix_fill_random_uniform(&test_matrix2, -0.5,0.5,32);
    for (size_t i = 0; i < test_matrix1.rows * test_matrix1.columns; ++i) {
        assert(test_matrix1.data[i] == test_matrix2.data[i]);
        assert(test_matrix1.data[i] >= -0.5 && test_matrix1.data[i] < 0.5);
    }
    assert(mlc_matrix_fill_random_uniform(&test_matrix1, 0.5, -0.5, 32) == MLC_STATUS_INVALID_ARGUMENT);
}

int main(void) {
    test_matrix_create_and_free();
    test_matrix_create_rejects_zero_dimensions();
    test_matrix_null_pointer();
    test_matrix_create_rejects_overflow();
    test_matrix_create_rejects_allocated_matrix();
    test_matrix_get_and_set();
    test_matrix_access_rejects_invalid_arguments();
    test_matrix_access_rejects_out_of_bounds_indices();
    test_matrix_successful_fill();
    test_matrix_invalid_matrix_fill();
    test_matrix_successful_deep_clone();
    test_matrix_invalid_clone();
    test_matrix_successful_addition();
    test_matrix_dimensions_mismatch();
    test_matrix_scale();
    test_matrix_scale_invalid_input();
    test_matrix_hadamard();
    test_matrix_hadamard_dimension_mismatch();
    test_matrix_transpose();
    test_matrix_transpose_invalid_argument();
    test_matrix_multiply();
    test_matrix_multiply_dimension_mismatch();
    test_matrix_add_row_vector();
    test_matrix_add_row_vector_dimensions_mismatch();
    test_matrix_sum_rows();
    test_matrix_apply();
    test_matrix_apply_null_function();
    test_matrix_fill_random_uniform();
    return 0;
}
