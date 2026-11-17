#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <string>
namespace qwe {
/**
 * @brief 将CUDA Runtime错误码转换为 C++ 异常。
 * @param[in] e 待检查的 API 返回状态。
 * @return 成功状态下正常返回；失败时抛 std::runtime_error。
 * @note 不同步 GPU；异步执行错误可能在后续同步或拷贝时报告。
 */
inline void check(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}

/**
 * @brief 将cuBLAS错误码转换为 C++ 异常。
 * @param[in] e 待检查的 API 返回状态。
 * @return 成功状态下正常返回；失败时抛 std::runtime_error。
 * @note 不同步 GPU；异步执行错误可能在后续同步或拷贝时报告。
 */
inline void check(cublasStatus_t e) {
    if (e != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS error " + std::to_string(e));
}
} // namespace qwe
