#include "cuda_kernels.cuh"
#include "backend/cuda/error.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
namespace qwe::kernels {

// RoPE：每个线程处理同一个 head 内前后半区的一对元素；位置来自 token 行号。

/**
 * @brief GPU kernel：对前后半区成对元素原地施加 RoPE。
 * @param[in,out] x 连续 FP32 张量 [序列长度, head数, head宽度]。
 * @param[in] heads head 数量。
 * @param[in] dim 每个 head 的宽度，必须为偶数。
 * @param[in] theta 正数频率基数；第 j 对的角度为 position × theta^(-2j/d)。
 * @param[in] n 序列长度，位置从 0 到 n-1。
 * @return 无；结果写入输出或原地参数。
 * @note 由算子层预先校验维度、类型、设备、布局和缓冲区大小；本层不分配显存。
 * @note 这是设备函数，必须通过 CUDA kernel launch 调用。
 */
__global__ void rope_kernel(float *x, int heads, int dim, float theta, int n, int offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * heads * (dim / 2))
        return;
    int j = i % (dim / 2), row = i / (dim / 2), position = offset + row / heads;
    float angle = position * powf(theta, -2.0f * j / dim), s, c;
    sincosf(angle, &s, &c);
    float a = x[(size_t)row * dim + j], b = x[(size_t)row * dim + j + dim / 2];
    x[(size_t)row * dim + j] = a * c - b * s;
    x[(size_t)row * dim + j + dim / 2] = b * c + a * s;
}

/**
 * @brief 启动封装：对前后半区成对元素原地施加 RoPE。
 * @param[in,out] x 连续 FP32 张量 [序列长度, head数, head宽度]。
 * @param[in] n 序列长度，位置从 0 到 n-1。
 * @param[in] h head 数量。
 * @param[in] d 每个 head 的宽度，必须为偶数。
 * @param[in] theta 正数频率基数；第 j 对的角度为 position × theta^(-2j/d)。
 * @return 无；结果写入输出或原地参数。
 * @note 由算子层预先校验维度、类型、设备、布局和缓冲区大小；本层不分配显存。
 * @note 仅提交默认 stream 上的工作，返回不代表 GPU 已执行完成。
 */
void rope(float *x, int n, int h, int d, float theta, int position_offset) {
    int count = n * h * (d / 2);
    rope_kernel<<<(count + 255) / 256, 256>>>(x, h, d, theta, n, position_offset);
    check(cudaGetLastError());
}
} // namespace qwe::kernels
