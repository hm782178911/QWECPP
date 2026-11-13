
cmake_minimum_required(VERSION 3.18)

project(QWECPP
    VERSION 0.1.0
    LANGUAGES C
)

# 可执行文件输出到 build/bin/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# 静态库（.a）输出到 build/lib/
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# 共享库（.so）输出到 build/lib/
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

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

# add_library(qwecpp_core_static STATIC
#     src/core/dtype.c
#     src/core/tensor.c
# )

# add_library(qwecpp_core_shared SHARED
#     src/core/dtype.c
#     src/core/tensor.c
# )


target_include_directories(qwecpp_core
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

# target_include_directories(qwecpp_core_static
#     PUBLIC
#         ${PROJECT_SOURCE_DIR}/include
# )

# target_include_directories(qwecpp_core_shared
#     PUBLIC
#         ${PROJECT_SOURCE_DIR}/include
# )

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
