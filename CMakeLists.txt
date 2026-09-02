cmake_minimum_required(VERSION 4.1)
project(
        MLC
        VERSION 0.1.0
        DESCRIPTION "A machine-learning library written in C"
        LANGUAGES C
)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_library(MLC STATIC
        src/matrix.c
        include/mlc/status.h
        include/mlc/matrix.h
        include/mlc/dataset.h
        src/dataset.c
        include/mlc/linear_regression.h
        src/linear_regression.c
)

target_include_directories(MLC
        PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

if(MSVC)
    target_compile_options(MLC PRIVATE /W4)
else()
    target_compile_options(MLC PRIVATE -Wall -Wextra -Wpedantic)
endif()

option(MLC_BUILD_EXAMPLES "Build MLC example programs" ON)

if(MLC_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

include(CTest)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
