#include <mlc/loss.h>
#include <math.h>
#include <assert.h>

static void test_loss_forward_mse() {
    MLCMatrix predictions = {0};
    MLCMatrix targets = {0};
    double loss_value;
    assert(mlc_matrix_create(2, 2, &predictions) == MLC_STATUS_SUCCESS);
    assert(mlc_matrix_create(2, 2, &targets) == MLC_STATUS_SUCCESS);

    double prediction_values[] = {
        1.0, 3.0,
        2.0, 8.0
    };

    double target_values[] = {
        2.0, 1.0,
        2.0, 4.0
    };
    for (size_t i = 0; i < 4; ++i) {
        predictions.data[i] = prediction_values[i];
        targets.data[i] = target_values[i];
    }
    assert(mlc_loss_forward(&predictions, &targets, MLC_LOSS_MEAN_SQUARED_ERROR, &loss_value) == MLC_STATUS_SUCCESS);
    assert(fabs(loss_value - 5.25) < 1e-12);
    for (size_t i = 0; i < 4; ++i) {
        assert(predictions.data[i] == prediction_values[i]);
        assert(targets.data[i] == target_values[i]);
    }

    mlc_matrix_free(&predictions);
    mlc_matrix_free(&targets);
}

static void test_loss_backward_mse() {
    MLCMatrix predictions = {0};
    MLCMatrix targets = {0};
    MLCMatrix prediction_gradient = {0};
    assert(mlc_matrix_create(2, 2, &predictions) == MLC_STATUS_SUCCESS);
    assert(mlc_matrix_create(2, 2, &targets) == MLC_STATUS_SUCCESS);

    double prediction_values[] = {
        1.0, 3.0,
        2.0, 8.0
    };

    double target_values[] = {
        2.0, 1.0,
        2.0, 4.0
    };

    double expected_gradient[] = {
        -0.5, 1.0,
         0.0, 2.0
    };
    for (size_t i = 0; i < 4; ++i) {
        predictions.data[i] = prediction_values[i];
        targets.data[i] = target_values[i];
    }
    assert(mlc_loss_backward(&predictions, &targets, MLC_LOSS_MEAN_SQUARED_ERROR, &prediction_gradient) == MLC_STATUS_SUCCESS);
    for (size_t i = 0; i < 4; ++i) {
        assert(fabs(prediction_gradient.data[i] - expected_gradient[i]) < 1e-12);
    }

    mlc_matrix_free(&predictions);
    mlc_matrix_free(&targets);
    mlc_matrix_free(&prediction_gradient);
}

static void test_loss_forward_bce() {
    MLCMatrix predictions = {0};
    MLCMatrix targets = {0};
    assert(mlc_matrix_create(2, 1, &predictions) == MLC_STATUS_SUCCESS);
    assert(mlc_matrix_create(2, 1, &targets) == MLC_STATUS_SUCCESS);
    predictions.data[0] = 0.8;
    predictions.data[1] = 0.3;
    targets.data[0] = 1.0;
    targets.data[1] = 0.0;
    double loss_value = 0.0;
    assert(mlc_loss_forward(&predictions, &targets, MLC_LOSS_BINARY_CROSS_ENTROPY, &loss_value) == MLC_STATUS_SUCCESS);
    assert(fabs(loss_value - 0.2899092476264711) < 1e-12);
    assert(predictions.data[0] == 0.8 && predictions.data[1] == 0.3);
    assert(targets.data[0] == 1.0 && targets.data[1] == 0.0);

    mlc_matrix_free(&predictions);
    mlc_matrix_free(&targets);
}

static void test_loss_backward_bce() {
    MLCMatrix predictions = {0};
    MLCMatrix targets = {0};
    MLCMatrix prediction_gradient = {0};
    assert(mlc_matrix_create(2, 1, &predictions) == MLC_STATUS_SUCCESS);
    assert(mlc_matrix_create(2, 1, &targets) == MLC_STATUS_SUCCESS);
    predictions.data[0] = 0.8;
    predictions.data[1] = 0.3;
    targets.data[0] = 1.0;
    targets.data[1] = 0.0;

    double expected_gradient[] = {
        -0.625,
         0.7142857142857143
    };

    assert(mlc_loss_backward(&predictions, &targets, MLC_LOSS_BINARY_CROSS_ENTROPY, &prediction_gradient) == MLC_STATUS_SUCCESS);
    assert(prediction_gradient.rows == 2 && prediction_gradient.columns == 1);
    for (size_t i = 0; i < 2; ++i) {
        assert(fabs(prediction_gradient.data[i] - expected_gradient[i]) < 1e-12);
    }
    assert(predictions.data[0] == 0.8 && predictions.data[1] == 0.3);
    assert(targets.data[0] == 1.0 && targets.data[1] == 0.0);

    mlc_matrix_free(&predictions);
    mlc_matrix_free(&targets);
    mlc_matrix_free(&prediction_gradient);
}

static void test_loss_forward_cce(void) {
    MLCMatrix predictions = {0};
    MLCMatrix targets = {0};
    assert(mlc_matrix_create(2, 3, &predictions) == MLC_STATUS_SUCCESS);
    assert(mlc_matrix_create(2, 3, &targets) == MLC_STATUS_SUCCESS);
    double prediction_values[] = {
        0.1, 0.6, 0.3,
        0.2, 0.3, 0.5
    };

    double target_values[] = {
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    };

    for (size_t i = 0; i < 6; ++i) {
        predictions.data[i] = prediction_values[i];
        targets.data[i] = target_values[i];
    }

    double loss_value = 0.0;
    assert(mlc_loss_forward(&predictions, &targets, MLC_LOSS_CATEGORICAL_CROSS_ENTROPY, &loss_value) == MLC_STATUS_SUCCESS);
    assert(fabs(loss_value - 0.6019864021629680) < 1e-12);
    for (size_t i = 0; i < 6; ++i) {
        assert(predictions.data[i] == prediction_values[i]);
        assert(targets.data[i] == target_values[i]);
    }

    mlc_matrix_free(&predictions);
    mlc_matrix_free(&targets);
}

static void test_loss_backward_cce(void) {
    MLCMatrix predictions = {0};
    MLCMatrix targets = {0};
    MLCMatrix prediction_gradient = {0};
    assert(mlc_matrix_create(2, 3, &predictions) == MLC_STATUS_SUCCESS);
    assert(mlc_matrix_create(2, 3, &targets) == MLC_STATUS_SUCCESS);
    double prediction_values[] = {
        0.1, 0.6, 0.3,
        0.2, 0.3, 0.5
    };

    double target_values[] = {
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    };

    double expected_gradient[] = {
        0.0, -0.8333333333333333, 0.0,
        0.0,  0.0,                -1.0
    };

    for (size_t i = 0; i < 6; ++i) {
        predictions.data[i] = prediction_values[i];
        targets.data[i] = target_values[i];
    }
    assert(mlc_loss_backward(&predictions, &targets, MLC_LOSS_CATEGORICAL_CROSS_ENTROPY, &prediction_gradient) == MLC_STATUS_SUCCESS);
    assert(prediction_gradient.rows == 2 && prediction_gradient.columns == 3);
    for (size_t i = 0; i < 6; ++i) {
        assert(fabs(prediction_gradient.data[i] - expected_gradient[i]) < 1e-12);
    }
    for (size_t i = 0; i < 6; ++i) {
        assert(predictions.data[i] == prediction_values[i]);
        assert(targets.data[i] == target_values[i]);
    }

    mlc_matrix_free(&predictions);
    mlc_matrix_free(&targets);
    mlc_matrix_free(&prediction_gradient);
}

int main(void) {
    test_loss_forward_mse();
    test_loss_backward_mse();
    test_loss_forward_bce();
    test_loss_backward_bce();
    test_loss_forward_cce();
    test_loss_backward_cce();
    return 0;
}