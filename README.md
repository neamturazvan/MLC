MLC is a machine-learning foundations library written from scratch in C11. The current public `main` branch implements dense matrix operations, dataset preprocessing utilities and multi-output linear regression behind a reusable C API.

> **Project status:** Work in progress. Matrix operations, dataset preprocessing and multi-output linear regression are implemented and tested. Classification models and configurable feed-forward neural networks are currently planned.
## Implemented features

### Matrix operations

- Heap-backed dense matrices using `double` values
- Creation, cleanup, bounds-checked access, filling and deep cloning
- Addition, subtraction and scalar multiplication
- Hadamard product and matrix multiplication
- Transposition, row-vector addition and row reduction
- Element-wise unary-function application
- Reproducible uniform random initialisation using an explicit seed

### Dataset preprocessing

- Separate feature and target matrices
- CSV loading with optional header handling
- Validation for malformed rows, invalid numeric fields and non-finite values
- Deterministic paired shuffling of features and targets
- Train/test splitting
- Standard-scaler fitting and transformation

### Linear regression

- Single-output and multi-output regression
- Full-batch gradient-descent training
- Configurable learning rate and epoch count
- Separate model creation, training, prediction and cleanup
- Zero-initialized weight and bias matrices
- Validation for incompatible feature, target and model dimensions

### Engineering

- C11 static library with public headers under `include/mlc`
- CMake build with optional examples and CTest integration
- Explicit status codes for invalid arguments, dimension mismatches, allocation failures, file errors and parse errors
- Matrix, dataset and linear regression test executables

## Build

Requirements:

- A C11-compatible compiler
- CMake 4.1 or newer, matching the current `CMakeLists.txt`

```sh
cmake -S . -B build -DMLC_BUILD_EXAMPLES=ON -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

## Minimal example

```c
#include <mlc/matrix.h>
#include <stdio.h>

int main(void) {
    MLCMatrix a = {0};
    MLCMatrix b = {0};
    MLCMatrix result = {0};

    if (mlc_matrix_create(2, 2, &a) != MLC_STATUS_SUCCESS ||
        mlc_matrix_create(2, 2, &b) != MLC_STATUS_SUCCESS) {
        mlc_matrix_free(&a);
        mlc_matrix_free(&b);
        return 1;
    }

    mlc_matrix_fill(&a, 2.0);
    mlc_matrix_fill(&b, 3.0);

    if (mlc_matrix_add(&a, &b, &result) == MLC_STATUS_SUCCESS) {
        double value = 0.0;
        mlc_matrix_get(&result, 0, 0, &value);
        printf("result[0][0] = %.1f\n", value);
    }

    mlc_matrix_free(&a);
    mlc_matrix_free(&b);
    mlc_matrix_free(&result);
    return 0;
}
```

## Repository structure

```text
MLC/
├── include/mlc/       Public headers
├── src/               Library implementation
├── tests/             Matrix and dataset tests plus CSV fixtures
├── examples/          Example programs
└── CMakeLists.txt      Top-level build configuration
```

## Current limitations

- The CSV reader handles simple comma-separated numeric data; quoted fields are not supported.
- CSV rows are currently limited to 4096 characters.
- The example program is still a placeholder and should be replaced with a working end-to-end example.
- Logistic Regression and neural-network APIs are not yet implemented on the public branch.

## Planned work

- logistic regression and classification metrics
- Configurable feed-forward neural networks
- Training metrics and model-evaluation utilities
- Expanded examples and documentation
- Continuous-integration builds across multiple compilers

