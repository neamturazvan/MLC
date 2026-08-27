add_executable(matrix_example
        matrix_example.c
)

target_link_libraries(matrix_example
        PRIVATE
        MLC
)
