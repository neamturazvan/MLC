add_executable(test_matrix
        test_matrix.c
)

add_executable(
        test_dataset
        test_dataset.c
)

target_link_libraries(
        test_matrix
        PRIVATE
            MLC
)

target_link_libraries(
        test_dataset
        PRIVATE
            MLC
)

add_test(
        NAME matrix_tests
        COMMAND test_matrix
)

add_test(
        NAME dataset_tests
        COMMAND test_dataset
)

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/data/valid_with_header.csv
        ${CMAKE_CURRENT_BINARY_DIR}/valid_with_header.csv
        COPYONLY
)

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/data/valid_without_header.csv
        ${CMAKE_CURRENT_BINARY_DIR}/valid_without_header.csv
        COPYONLY
)

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/data/invalid_text.csv
        ${CMAKE_CURRENT_BINARY_DIR}/invalid_text.csv
        COPYONLY
)

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/data/invalid_nonfinite.csv
        ${CMAKE_CURRENT_BINARY_DIR}/invalid_nonfinite.csv
        COPYONLY
)

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/data/invalid_range.csv
        ${CMAKE_CURRENT_BINARY_DIR}/invalid_range.csv
        COPYONLY
)

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/data/invalid_inconsistent_columns.csv
        ${CMAKE_CURRENT_BINARY_DIR}/invalid_inconsistent_columns.csv
        COPYONLY
)

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/data/invalid_empty_field.csv
        ${CMAKE_CURRENT_BINARY_DIR}/invalid_empty_field.csv
        COPYONLY
)

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/data/invalid_blank_row.csv
        ${CMAKE_CURRENT_BINARY_DIR}/invalid_blank_row.csv
        COPYONLY
)

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/data/empty.csv
        ${CMAKE_CURRENT_BINARY_DIR}/empty.csv
        COPYONLY
)

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/data/header_only.csv
        ${CMAKE_CURRENT_BINARY_DIR}/header_only.csv
        COPYONLY
)

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/data/shuffle_test.csv
        ${CMAKE_CURRENT_BINARY_DIR}/shuffle_test.csv
        COPYONLY
)

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/data/one_sample.csv
        ${CMAKE_CURRENT_BINARY_DIR}/one_sample.csv
        COPYONLY
)

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/data/edge_cases_standard_scaler.csv
        ${CMAKE_CURRENT_BINARY_DIR}/edge_cases_standard_scaler.csv
        COPYONLY
)

