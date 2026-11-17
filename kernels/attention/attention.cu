#include "cuda_kernels.cuh"
#include "backend/cuda/error.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
namespace qwe::kernels {

// 基础因果 GQA：每个 block 处理一个 Query 位置和 head，多个 Q head 共享 KV head。
// 当前最多 8192 个位置；共享内存保存 Softmax 分数，不读取未来 token。

/**
 * @brief GPU kernel：计算缩放点积、因果 Softmax 和对 V 的加权求和。
 * @param[out] out 连续 FP32 输出 [序列长度, Q头数, head宽度]。
 * @param[in] q 连续 FP32 Query，布局 [序列长度, Q头数, head宽度]。
 * @param[in] k 连续 FP32 Key，布局 [序列长度, KV头数, head宽度]。
 * @param[in] v 连续 FP32 Value，形状与 k 相同。
 * @param[in] n 序列长度，取值 1～8192。
 * @param[in] h Query head 数，必须是 kh 的整数倍。
 * @param[in] kh Key/Value head 数。
 * @param[in] d 每个 head 的宽度。
 * @return 无；结果写入输出或原地参数。
 * @note 由算子层预先校验维度、类型、设备、布局和缓冲区大小；本层不分配显存。
 * @note 这是设备函数，必须通过 CUDA kernel launch 调用。
 * @note 每个 block 对应一个 Query 位置/head，blockDim.x 必须为 256。
 */
__global__ void attention_kernel(
    float *out, const float *q, const float *k, const float *v, int n, int h, int kh, int d) {
    extern __shared__ float scores[];
    int t = threadIdx.x, head = blockIdx.x % h, pos = blockIdx.x / h, kv = head / (h / kh);

    // 只计算 [0, pos] 的 Q·K 分数；同组 Query head 使用相同的 KV head。
    for (int s = t; s <= pos; s += blockDim.x) {
        float dot = 0;
        for (int j = 0; j < d; ++j)
            dot += q[((size_t)pos * h + head) * d + j] * k[((size_t)s * kh + kv) * d + j];
        scores[s] = dot * rsqrtf((float)d);
    }
    __syncthreads();

    // 参考实现由一个线程计算稳定 Softmax：先减最大值，再指数归一化。
    if (t == 0) {
        float m = scores[0];
        for (int j = 1; j <= pos; ++j)
            m = fmaxf(m, scores[j]);
        float sum = 0;
        for (int j = 0; j <= pos; ++j) {
            scores[j] = expf(scores[j] - m);
            sum += scores[j];
        }
        for (int j = 0; j <= pos; ++j)
            scores[j] /= sum;
    }
    __syncthreads();

    // 每个线程负责若干输出通道，对历史 V 按注意力概率加权求和。
    for (int j = t; j < d; j += blockDim.x) {
        float sum = 0;
        for (int s = 0; s <= pos; ++s)
            sum += scores[s] * v[((size_t)s * kh + kv) * d + j];
        out[((size_t)pos * h + head) * d + j] = sum;
    }
}

/**
 * @brief 启动封装：计算缩放点积、因果 Softmax 和对 V 的加权求和。
 * @param[out] o 连续 FP32 输出 [n, h, d]。
 * @param[in] q 连续 FP32 Query，布局 [序列长度, Q头数, head宽度]。
 * @param[in] k 连续 FP32 Key，布局 [序列长度, KV头数, head宽度]。
 * @param[in] v 连续 FP32 Value，形状与 k 相同。
 * @param[in] n 序列长度，取值 1～8192。
 * @param[in] h Query head 数，必须是 kh 的整数倍。
 * @param[in] kh Key/Value head 数。
 * @param[in] d 每个 head 的宽度。
 * @return 无；结果写入输出或原地参数。
 * @note 由算子层预先校验维度、类型、设备、布局和缓冲区大小；本层不分配显存。
 * @note 仅提交默认 stream 上的工作，返回不代表 GPU 已执行完成。
 */
void attention(
    float *o, const float *q, const float *k, const float *v, int n, int h, int kh, int d) {
    if (n < 1 || n > 8192 || h % kh)
        throw std::runtime_error("unsupported attention dimensions");
    attention_kernel<<<n * h, 256, (size_t)n*sizeof(float)>>>(o, q, k, v, n, h, kh, d);
    check(cudaGetLastError());
}
} // namespace qwe::kernels
