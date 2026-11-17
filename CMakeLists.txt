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

# 使用标准 C11，不启用编译器特有语言扩展。
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# 警告选项只传给 C/C++ 编译器，避免 NVCC 误读这些参数。
add_compile_options(
    $<$<COMPILE_LANGUAGE:C,CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:C,CXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:C,CXX>:-Wpedantic>
)

# 核心数据类型与 Tensor 元数据，不依赖 CUDA。
add_library(qwecpp_core STATIC
    src/core/dtype.c
    src/core/tensor.c
)

target_include_directories(qwecpp_core
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

# 基础单元测试。
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

# 随项目提供 cJSON 1.7.19（MIT），构建时不需要联网下载。
add_library(qwecpp_json STATIC third_party/cJSON/cJSON.c)
target_include_directories(qwecpp_json PUBLIC ${PROJECT_SOURCE_DIR}/third_party/cJSON)

add_library(qwecpp_loader STATIC src/loader/safetensors.c)
target_link_libraries(qwecpp_loader PUBLIC qwecpp_core PRIVATE qwecpp_json m)

add_executable(qwecpp_inspect tools/qwecpp_inspect.c)
target_link_libraries(qwecpp_inspect PRIVATE qwecpp_loader)

add_executable(test_reader tests/loader/test_reader.c)
target_link_libraries(test_reader PRIVATE qwecpp_loader)

find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME safetensors_test
    COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/loader/test_safetensors.py
            $<TARGET_FILE:qwecpp_inspect> $<TARGET_FILE:test_reader>)

# CUDA 可选：关闭时仍可编译并测试纯 C 基础模块。
option(QWECPP_ENABLE_CUDA "Build Qwen3 CUDA forward" OFF)
if(QWECPP_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(qwecpp_cuda STATIC
        src/backend/cuda/transfer.cu
        src/operators/tensor_ops.cu
        kernels/core/convert.cu
        kernels/embedding/embedding.cu
        kernels/normalization/rmsnorm.cu
        kernels/positional/rope.cu
        kernels/attention/attention.cu
        kernels/activation/elementwise.cu
        src/models/qwen3.cu)
    set_target_properties(qwecpp_cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_include_directories(qwecpp_cuda PUBLIC ${PROJECT_SOURCE_DIR}/include PRIVATE ${PROJECT_SOURCE_DIR}/kernels)
    target_link_libraries(qwecpp_cuda PUBLIC qwecpp_loader CUDA::cudart CUDA::cublas PRIVATE qwecpp_json)
    add_executable(qwecpp_forward tools/qwecpp_forward.cu)
    set_target_properties(qwecpp_forward PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_link_libraries(qwecpp_forward PRIVATE qwecpp_cuda)
    add_executable(test_cuda_tensor tests/operators/test_cuda_tensor.cu)
    set_target_properties(test_cuda_tensor PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_link_libraries(test_cuda_tensor PRIVATE qwecpp_cuda)
    add_test(NAME cuda_tensor_test COMMAND test_cuda_tensor)
    add_executable(test_transfer tests/operators/test_transfer.cu)
    set_target_properties(test_transfer PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_link_libraries(test_transfer PRIVATE qwecpp_cuda)
    add_test(NAME cuda_transfer_test COMMAND test_transfer)
    set_tests_properties(cuda_transfer_test PROPERTIES SKIP_RETURN_CODE 77)
endif()

option(QWECPP_ENABLE_CHAT "Build native tokenizer and terminal chat" ON)
if(QWECPP_ENABLE_CUDA AND QWECPP_ENABLE_CHAT)
    enable_language(CXX)
    find_path(PCRE2_INCLUDE_DIR pcre2.h REQUIRED)
    find_library(PCRE2_LIBRARY NAMES pcre2-8 REQUIRED)
    find_package(ICU REQUIRED COMPONENTS uc)
    add_library(qwecpp_tokenizer STATIC src/runtime/tokenizer.cpp)
    target_compile_features(qwecpp_tokenizer PUBLIC cxx_std_17)
    target_include_directories(qwecpp_tokenizer PUBLIC ${PROJECT_SOURCE_DIR}/include PRIVATE ${PCRE2_INCLUDE_DIR})
    target_link_libraries(qwecpp_tokenizer PUBLIC qwecpp_json ${PCRE2_LIBRARY} ICU::uc)
    add_executable(qwecpp_tokenize tools/qwecpp_tokenize.cpp)
    target_link_libraries(qwecpp_tokenize PRIVATE qwecpp_tokenizer)
    add_executable(qwecpp_chat tools/qwecpp_chat.cu)
    set_target_properties(qwecpp_chat PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_link_libraries(qwecpp_chat PRIVATE qwecpp_cuda qwecpp_tokenizer)
endif()
