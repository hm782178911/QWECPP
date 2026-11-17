#include "cuda_kernels.cuh"
#include "backend/cuda/error.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
namespace qwe::kernels {

// RMSNorm：每个 block 处理一行，先归约平方和，再乘归一化系数和权重。

/**
 * @brief GPU kernel：按行执行 RMSNorm 与逐通道缩放。
 * @param[out] out FP32 显存输出，与输入同形，允许完全原地。
 * @param[in] in 连续 FP32 行数据，每行 width 个元素。
 * @param[in] weight 长度为 width 的 FP32 缩放权重。
 * @param[in] width 每行元素数。
 * @param[in] eps 正数稳定项，输出为 x/sqrt(mean(x²)+eps)×weight。
 * @return 无；结果写入输出或原地参数。
 * @note 由算子层预先校验维度、类型、设备、布局和缓冲区大小；本层不分配显存。
 * @note 这是设备函数，必须通过 CUDA kernel launch 调用。
 * @note 行数由 gridDim.x 决定，blockDim.x 必须为 256。
 */
__global__ void
norm_kernel(float *out, const float *in, const float *weight, int width, float eps) {
    __shared__ float sums[256];
    int row = blockIdx.x, t = threadIdx.x;
    float total = 0;
    for (int j = t; j < width; j += blockDim.x) {
        float x = in[(size_t)row * width + j];
        total += x * x;
    }
    sums[t] = total;
    __syncthreads();
    for (int s = 128; s; s /= 2) {
        if (t < s)
            sums[t] += sums[t + s];
        __syncthreads();
    }
    float scale = rsqrtf(sums[0] / width + eps);
    for (int j = t; j < width; j += blockDim.x)
        out[(size_t)row * width + j] = in[(size_t)row * width + j] * scale * weight[j];
}

/**
 * @brief 启动封装：按行执行 RMSNorm 与逐通道缩放。
 * @param[out] o FP32 显存输出 [rows, d]，允许完全原地。
 * @param[in] x 连续 FP32 输入 [rows, d]。
 * @param[in] w 长度为 d 的 FP32 缩放权重。
 * @param[in] rows 行数，每个 block 处理一行。
 * @param[in] d 每行元素数。
 * @param[in] eps 正数稳定项，输出为 x/sqrt(mean(x²)+eps)×weight。
 * @return 无；结果写入输出或原地参数。
 * @note 由算子层预先校验维度、类型、设备、布局和缓冲区大小；本层不分配显存。
 * @note 仅提交默认 stream 上的工作，返回不代表 GPU 已执行完成。
 */
void norm(float *o, const float *x, const float *w, int rows, int d, float eps) {
    norm_kernel<<<rows, 256>>>(o, x, w, d, eps);
    check(cudaGetLastError());
}
} // namespace qwe::kernels
