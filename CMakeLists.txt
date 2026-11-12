
cmake_minimum_required(VERSION 3.18)

project(QWECPP
    VERSION 0.1.0
    LANGUAGES C
)

# C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Compiler warnings
add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

# Core library
add_library(qwecpp_core STATIC
    src/core/dtype.c
    src/core/tensor.c
)

target_include_directories(qwecpp_core
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

# Unit tests
enable_testing()

add_executable(test_tensor
    tests/unit/test_tensor.c
)

target_link_libraries(test_tensor
    PRIVATE qwecpp_core
)

add_test(
    NAME tensor_test
    COMMAND test_tensor
)
