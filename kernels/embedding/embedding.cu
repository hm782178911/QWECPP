#include "cuda_kernels.cuh"
#include "backend/cuda/error.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
namespace qwe::kernels {

// Embedding 查表：每个线程写一个输出元素，按 token ID 读取权重表的一行。

/**
 * @brief GPU kernel：按 token ID 从词向量表查出各位置的向量。
 * @param[out] out FP32 显存输出，连续布局 [序列长度, 隐藏宽度]。
 * @param[in] w 连续 FP32 词向量表 [词表大小, d]。
 * @param[in] ids INT32 显存 ID 数组；调用方保证每个 ID 在词表内。
 * @param[in] n 序列长度。
 * @param[in] d 隐藏宽度。
 * @return 无；结果写入输出或原地参数。
 * @note 由算子层预先校验维度、类型、设备、布局和缓冲区大小；本层不分配显存。
 * @note 这是设备函数，必须通过 CUDA kernel launch 调用。
 */
__global__ void embedding_kernel(float *out, const float *w, const int *ids, int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n * d)
        out[i] = w[(size_t)ids[i / d] * d + i % d];
}

/**
 * @brief 启动封装：按 token ID 从词向量表查出各位置的向量。
 * @param[out] o FP32 显存输出，连续布局 [n, d]。
 * @param[in] w 连续 FP32 词向量表 [词表大小, d]。
 * @param[in] ids INT32 显存 ID 数组；调用方保证每个 ID 在词表内。
 * @param[in] n 序列长度。
 * @param[in] d 隐藏宽度。
 * @return 无；结果写入输出或原地参数。
 * @note 由算子层预先校验维度、类型、设备、布局和缓冲区大小；本层不分配显存。
 * @note 仅提交默认 stream 上的工作，返回不代表 GPU 已执行完成。
 */
void embedding(float *o, const float *w, const int *ids, int n, int d) {
    embedding_kernel<<<(n * d + 255) / 256, 256>>>(o, w, ids, n, d);
    check(cudaGetLastError());
}
} // namespace qwe::kernels
