#include <mlc/activation.h>
#include <mlc/matrix.h>
#include <assert.h>
#include <math.h>

static void test_forward_activation_linear() {
    MLCMatrix input = {0};
    MLCMatrix output = {0};
    assert(mlc_matrix_create(2, 3, &input) == MLC_STATUS_SUCCESS);
    const double values[] = {
        -2.0,  0.0, 3.0,
         4.0, -5.0, 1.0
    };
    for (size_t i = 0; i < 6; ++i) {
        input.data[i] = values[i];
    }
    assert(mlc_activation_forward(&input, MLC_ACTIVATION_LINEAR, &output) == MLC_STATUS_SUCCESS);
    assert(output.rows == 2);
    assert(output.columns == 3);
    assert(output.data != input.data);

    for (size_t i = 0; i < 6; ++i) {
        assert(output.data[i] == values[i]);
    }

    output.data[0] = 100.0;
    assert(input.data[0] == -2.0);

    mlc_matrix_free(&input);
    mlc_matrix_free(&output);
}

static void test_backward_activation_linear() {
    MLCMatrix activated_output = {0};
    MLCMatrix upstream_gradient = {0};
    MLCMatrix input_gradient = {0};
    const double activated_values[] = {
        -2.0,  0.0, 3.0,
         4.0, -5.0, 1.0
    };
    const double upstream_values[] = {
        0.5, -1.0,  2.0,
        3.0,  0.25, -4.0
   };
   assert(mlc_matrix_create(2,3,&activated_output) == MLC_STATUS_SUCCESS);
   assert(mlc_matrix_create(2,3, &upstream_gradient) == MLC_STATUS_SUCCESS);
   for (size_t i = 0; i < 6; ++i) {
       activated_output.data[i] = activated_values[i];
       upstream_gradient.data[i] = upstream_values[i];
   }
   assert(mlc_activation_backward(&activated_output, &upstream_gradient, MLC_ACTIVATION_LINEAR, &input_gradient) == MLC_STATUS_SUCCESS);
    assert(input_gradient.rows == 2);
    assert(input_gradient.columns == 3);
    assert(input_gradient.data != upstream_gradient.data);

    for (size_t i = 0; i < 6; ++i) {
        assert(input_gradient.data[i] == upstream_values[i]);
    }

    input_gradient.data[0] = 100.0;
    assert(upstream_gradient.data[0] == 0.5);

    mlc_matrix_free(&activated_output);
    mlc_matrix_free(&upstream_gradient);
    mlc_matrix_free(&input_gradient);
}

static void test_forward_activation_relu() {
    MLCMatrix input = {0};
    MLCMatrix output = {0};

    const double input_values[] = {
        -2.0,  0.0, 3.0,
         4.0, -5.0, 1.0
    };

    const double expected_values[] = {
        0.0, 0.0, 3.0,
        4.0, 0.0, 1.0
    };
    assert(mlc_matrix_create(2, 3, &input) == MLC_STATUS_SUCCESS);
    for (size_t i = 0; i < 6; ++i) {
        input.data[i] = input_values[i];
    }
    assert(mlc_activation_forward(&input, MLC_ACTIVATION_RELU, &output) == MLC_STATUS_SUCCESS);
    for (size_t i = 0; i < 6; ++i) {
        assert(output.data[i] == expected_values[i]);
    }
    assert(output.rows == 2);
    assert(output.columns == 3);
    assert(output.data != input.data);
    output.data[0] = 100.0;
    assert(input.data[0] == -2.0);

    mlc_matrix_free(&input);
    mlc_matrix_free(&output);
}

static void test_backward_activation_relu() {
    MLCMatrix activated_output = {0};
    MLCMatrix upstream_gradient = {0};
    MLCMatrix input_gradient = {0};

    const double activated_values[] = {
        0.0, 0.0, 3.0,
        4.0, 0.0, 1.0
    };
    const double upstream_values[] = {
        5.0, -4.0, 2.0,
       -3.0,  7.0, 0.5
   };
    const double expected_values[] = {
        0.0, 0.0, 2.0,
       -3.0, 0.0, 0.5
   };
    assert(mlc_matrix_create(2, 3, &activated_output) == MLC_STATUS_SUCCESS);
    assert(mlc_matrix_create(2, 3, &upstream_gradient) == MLC_STATUS_SUCCESS);

    for (size_t i = 0; i < 6; ++i) {
        activated_output.data[i] = activated_values[i];
        upstream_gradient.data[i] = upstream_values[i];
    }
    assert(mlc_activation_backward(&activated_output, &upstream_gradient, MLC_ACTIVATION_RELU, &input_gradient) == MLC_STATUS_SUCCESS);
    for (size_t i = 0; i < upstream_gradient.rows * upstream_gradient.columns; ++i) {
        assert(input_gradient.data[i] == expected_values[i]);
    }
    assert(input_gradient.rows == upstream_gradient.rows);
    assert(input_gradient.columns == upstream_gradient.columns);
    assert(input_gradient.data != upstream_gradient.data);
    input_gradient.data[0] = 100.0;
    assert(upstream_gradient.data[0] == 5.0);

    mlc_matrix_free(&activated_output);
    mlc_matrix_free(&upstream_gradient);
    mlc_matrix_free(&input_gradient);
}

static void test_forward_activation_sigmoid() {
    MLCMatrix input = {0};
    MLCMatrix output = {0};

    const double input_values[] = {
        -1000.0,
        -2.0,
        0.0,
        2.0,
        1000.0
    };

    const double expected_values[] = {
        0.0,
        0.11920292202211755,
        0.5,
        0.8807970779778823,
        1.0
    };
    assert(mlc_matrix_create(1, 5, &input) == MLC_STATUS_SUCCESS);
    for (size_t i = 0; i < 5; ++i) {
        input.data[i] = input_values[i];
    }
    assert(mlc_activation_forward(&input, MLC_ACTIVATION_SIGMOID, &output) == MLC_STATUS_SUCCESS);
    for (size_t i = 0; i < input.rows * input.columns; ++i) {
        assert(fabs(output.data[i] - expected_values[i]) < 1e-12);
    }
    assert(output.rows == input.rows);
    assert(output.columns == input.columns);
    assert(output.data != input.data);
    output.data[0] = 100.0;
    assert(input.data[0] == -1000.0);

    mlc_matrix_free(&input);
    mlc_matrix_free(&output);

}

static void test_backward_activation_sigmoid() {
    MLCMatrix activated_output = {0};
    MLCMatrix upstream_gradient = {0};
    MLCMatrix input_gradient = {0};

    const double activated_values[] = {
        0.2, 0.5, 0.8
    };

    const double upstream_values[] = {
        5.0, -4.0, 2.0
    };

    const double expected_values[] = {
        0.8, -1.0, 0.32
    };

    assert(mlc_matrix_create(1, 3, &activated_output) == MLC_STATUS_SUCCESS);
    assert(mlc_matrix_create(1, 3, &upstream_gradient) == MLC_STATUS_SUCCESS);

    for (size_t i = 0; i < 3; ++i) {
        activated_output.data[i] = activated_values[i];
        upstream_gradient.data[i] = upstream_values[i];
    }

    assert(mlc_activation_backward(
        &activated_output,
        &upstream_gradient,
        MLC_ACTIVATION_SIGMOID,
        &input_gradient
    ) == MLC_STATUS_SUCCESS);

    assert(input_gradient.rows == 1);
    assert(input_gradient.columns == 3);
    assert(input_gradient.data != upstream_gradient.data);

    for (size_t i = 0; i < 3; ++i) {
        assert(fabs(input_gradient.data[i] - expected_values[i]) < 1e-12);
    }

    mlc_matrix_free(&activated_output);
    mlc_matrix_free(&upstream_gradient);
    mlc_matrix_free(&input_gradient);
}

static void test_forward_activation_softmax() {
    MLCMatrix input = {0};
    MLCMatrix output = {0};

    assert(mlc_matrix_create(2, 3, &input) == MLC_STATUS_SUCCESS);

    input.data[0] = 2.0;
    input.data[1] = 1.0;
    input.data[2] = 0.0;
    input.data[3] = 1000.0;
    input.data[4] = 1001.0;
    input.data[5] = 999.0;

    assert(mlc_activation_forward(&input, MLC_ACTIVATION_SOFTMAX, &output) == MLC_STATUS_SUCCESS);

    double expected[] = {
        0.6652409557748218,
        0.2447284710547976,
        0.0900305731703805,
        0.2447284710547976,
        0.6652409557748218,
        0.0900305731703805
    };

    assert(output.rows == 2 && output.columns == 3);

    for (size_t i = 0; i < output.rows * output.columns; ++i) {
        assert(isfinite(output.data[i]));
        assert(fabs(output.data[i] - expected[i]) < 1e-12);
    }

    for (size_t i = 0; i < output.rows; ++i) {
        double row_sum = 0.0;

        for (size_t j = 0; j < output.columns; ++j) {
            row_sum += output.data[i * output.columns + j];
        }

        assert(fabs(row_sum - 1.0) < 1e-12);
    }

    assert(input.data[0] == 2.0 && input.data[1] == 1.0 && input.data[2] == 0.0);
    assert(input.data[3] == 1000.0 && input.data[4] == 1001.0 && input.data[5] == 999.0);

    mlc_matrix_free(&input);
    mlc_matrix_free(&output);
}

static void test_backward_activation_softmax() {
    MLCMatrix activated_output = {0};
    MLCMatrix upstream_gradient = {0};
    MLCMatrix input_gradient = {0};
    assert(mlc_matrix_create(2, 3, &activated_output) == MLC_STATUS_SUCCESS);
    assert(mlc_matrix_create(2, 3, &upstream_gradient) == MLC_STATUS_SUCCESS);
    double activated_values[] = {
        0.2, 0.3, 0.5,
        0.6, 0.1, 0.3
    };

    double upstream_values[] = {
        1.0, 2.0, 3.0,
       -1.0, 4.0, 2.0
    };

    double expected[] = {
        -0.26, -0.09, 0.35,
        -0.84,  0.36, 0.48
    };
    for (size_t i = 0; i < 6; ++i) {
        activated_output.data[i] = activated_values[i];
        upstream_gradient.data[i] = upstream_values[i];
    }
    assert(mlc_activation_backward(&activated_output, &upstream_gradient, MLC_ACTIVATION_SOFTMAX, &input_gradient) == MLC_STATUS_SUCCESS);
    assert(input_gradient.rows == 2 && input_gradient.columns == 3);

    for (size_t i = 0; i < 6; ++i) {
        assert(fabs(input_gradient.data[i] - expected[i]) < 1e-12);
    }
    for (size_t i = 0; i < input_gradient.rows; ++i) {
        double row_sum = 0.0;

        for (size_t j = 0; j < input_gradient.columns; ++j) {
            row_sum += input_gradient.data[i * input_gradient.columns + j];
        }

        assert(fabs(row_sum) < 1e-12);
    }

    mlc_matrix_free(&activated_output);
    mlc_matrix_free(&upstream_gradient);
    mlc_matrix_free(&input_gradient);
}

int main(void) {
    test_forward_activation_linear();
    test_backward_activation_linear();
    test_forward_activation_relu();
    test_backward_activation_relu();
    test_forward_activation_sigmoid();
    test_backward_activation_sigmoid();
    test_forward_activation_softmax();
    test_backward_activation_softmax();
    return 0;
}