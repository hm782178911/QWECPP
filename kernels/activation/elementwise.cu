#include "cuda_kernels.cuh"
#include "backend/cuda/error.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
namespace qwe::kernels {

// 逐元素运算：SwiGLU 计算 SiLU(gate) × up；残差连接执行 x += y。

/**
 * @brief GPU kernel：原地执行 gate = SiLU(gate) × up。
 * @param[in,out] g 含 n 个 FP32 元素的门控分支显存。
 * @param[in] u 含 n 个 FP32 元素的升维分支显存。
 * @param[in] n 元素总数。
 * @return 无；结果写入输出或原地参数。
 * @note 由算子层预先校验维度、类型、设备、布局和缓冲区大小；本层不分配显存。
 * @note 这是设备函数，必须通过 CUDA kernel launch 调用。
 */
__global__ void silu_kernel(float *g, const float *u, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        g[i] = (g[i] / (1 + expf(-g[i]))) * u[i];
}

/**
 * @brief GPU kernel：执行逐元素残差相加 x += y。
 * @param[in,out] x 含 n 个 FP32 元素的显存，原地写回结果。
 * @param[in] y 含 n 个 FP32 元素的显存。
 * @param[in] n 元素总数。
 * @return 无；结果写入输出或原地参数。
 * @note 由算子层预先校验维度、类型、设备、布局和缓冲区大小；本层不分配显存。
 * @note 这是设备函数，必须通过 CUDA kernel launch 调用。
 */
__global__ void add_kernel(float *x, const float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] += y[i];
}

/**
 * @brief 启动封装：原地执行 gate = SiLU(gate) × up。
 * @param[in,out] g 含 n 个 FP32 元素的门控分支显存。
 * @param[in] u 含 n 个 FP32 元素的升维分支显存。
 * @param[in] n 元素总数。
 * @return 无；结果写入输出或原地参数。
 * @note 由算子层预先校验维度、类型、设备、布局和缓冲区大小；本层不分配显存。
 * @note 仅提交默认 stream 上的工作，返回不代表 GPU 已执行完成。
 */
void silu_mul(float *g, const float *u, int n) {
    silu_kernel<<<(n + 255) / 256, 256>>>(g, u, n);
    check(cudaGetLastError());
}

/**
 * @brief 启动封装：执行逐元素残差相加 x += y。
 * @param[in,out] x 含 n 个 FP32 元素的显存，原地写回结果。
 * @param[in] y 含 n 个 FP32 元素的显存。
 * @param[in] n 元素总数。
 * @return 无；结果写入输出或原地参数。
 * @note 由算子层预先校验维度、类型、设备、布局和缓冲区大小；本层不分配显存。
 * @note 仅提交默认 stream 上的工作，返回不代表 GPU 已执行完成。
 */
void add(float *x, const float *y, int n) {
    add_kernel<<<(n + 255) / 256, 256>>>(x, y, n);
    check(cudaGetLastError());
}
} // namespace qwe::kernels
