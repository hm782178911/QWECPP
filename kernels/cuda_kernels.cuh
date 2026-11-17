#pragma once
#include <cstddef>
// 内部 kernel 启动接口：只接收裸指针和必要尺寸，不暴露给模型层。
namespace qwe::kernels {
/**
 * @brief 启动封装：将输入元素转换为 FP32。
 * @param[out] out FP32 显存输出，至少容纳 n 个元素。
 * @param[in] in 输入显存，包含 n 个 dtype 类型的元素。
 * @param[in] n 元素总数，不是字节数。
 * @param[in] dtype 项目 DType 枚举值：0=FP32、1=FP16、2=BF16。
 * @return 无；结果写入输出或原地参数。
 * @note 由算子层预先校验维度、类型、设备、布局和缓冲区大小；本层不分配显存。
 * @note 仅提交默认 stream 上的工作，返回不代表 GPU 已执行完成。
 */
void convert(float *out, const void *in, size_t n, int dtype);

/**
 * @brief 启动封装：按 token ID 从词向量表查出各位置的向量。
 * @param[out] out FP32 显存输出，连续布局 [序列长度, 隐藏宽度]。
 * @param[in] weight 连续 FP32 词向量表 [词表大小, width]。
 * @param[in] ids INT32 显存 ID 数组；调用方保证每个 ID 在词表内。
 * @param[in] count 序列长度。
 * @param[in] width 隐藏宽度。
 * @return 无；结果写入输出或原地参数。
 * @note 由算子层预先校验维度、类型、设备、布局和缓冲区大小；本层不分配显存。
 * @note 仅提交默认 stream 上的工作，返回不代表 GPU 已执行完成。
 */
void embedding(float *out, const float *weight, const int *ids, int count, int width);

/**
 * @brief 启动封装：按行执行 RMSNorm 与逐通道缩放。
 * @param[out] out FP32 显存输出，与输入同形，允许完全原地。
 * @param[in] in 连续 FP32 行数据，每行 width 个元素。
 * @param[in] weight 长度为 width 的 FP32 缩放权重。
 * @param[in] rows 行数，每个 block 处理一行。
 * @param[in] width 每行元素数。
 * @param[in] eps 正数稳定项，输出为 x/sqrt(mean(x²)+eps)×weight。
 * @return 无；结果写入输出或原地参数。
 * @note 由算子层预先校验维度、类型、设备、布局和缓冲区大小；本层不分配显存。
 * @note 仅提交默认 stream 上的工作，返回不代表 GPU 已执行完成。
 */
void norm(float *out, const float *in, const float *weight, int rows, int width, float eps);

/**
 * @brief 启动封装：对前后半区成对元素原地施加 RoPE。
 * @param[in,out] x 连续 FP32 张量 [序列长度, head数, head宽度]。
 * @param[in] tokens 序列长度，也是位置编号范围。
 * @param[in] heads head 数量。
 * @param[in] dim 每个 head 的宽度，必须为偶数。
 * @param[in] theta 正数频率基数；第 j 对的角度为 position × theta^(-2j/d)。
 * @return 无；结果写入输出或原地参数。
 * @note 由算子层预先校验维度、类型、设备、布局和缓冲区大小；本层不分配显存。
 * @note 仅提交默认 stream 上的工作，返回不代表 GPU 已执行完成。
 */
void rope(float *x, int tokens, int heads, int dim, float theta);

/**
 * @brief 启动封装：计算缩放点积、因果 Softmax 和对 V 的加权求和。
 * @param[out] out 连续 FP32 输出 [序列长度, Q头数, head宽度]。
 * @param[in] q 连续 FP32 Query，布局 [序列长度, Q头数, head宽度]。
 * @param[in] k 连续 FP32 Key，布局 [序列长度, KV头数, head宽度]。
 * @param[in] v 连续 FP32 Value，形状与 k 相同。
 * @param[in] tokens 序列长度，取值 1～8192。
 * @param[in] heads Query head 数，必须是 KV head 数的整数倍。
 * @param[in] kvheads Key/Value head 数。
 * @param[in] dim 每个 head 的宽度。
 * @return 无；结果写入输出或原地参数。
 * @note 由算子层预先校验维度、类型、设备、布局和缓冲区大小；本层不分配显存。
 * @note 仅提交默认 stream 上的工作，返回不代表 GPU 已执行完成。
 */
void attention(float *out,
               const float *q,
               const float *k,
               const float *v,
               int tokens,
               int heads,
               int kvheads,
               int dim);

/**
 * @brief 启动封装：原地执行 gate = SiLU(gate) × up。
 * @param[in,out] gate 含 n 个 FP32 元素的门控分支显存。
 * @param[in] up 含 n 个 FP32 元素的升维分支显存。
 * @param[in] n 元素总数。
 * @return 无；结果写入输出或原地参数。
 * @note 由算子层预先校验维度、类型、设备、布局和缓冲区大小；本层不分配显存。
 * @note 仅提交默认 stream 上的工作，返回不代表 GPU 已执行完成。
 */
void silu_mul(float *gate, const float *up, int n);

/**
 * @brief 启动封装：执行逐元素残差相加 x += y。
 * @param[in,out] x 含 n 个 FP32 元素的显存，原地写回结果。
 * @param[in] y 含 n 个 FP32 元素的显存。
 * @param[in] n 元素总数。
 * @return 无；结果写入输出或原地参数。
 * @note 由算子层预先校验维度、类型、设备、布局和缓冲区大小；本层不分配显存。
 * @note 仅提交默认 stream 上的工作，返回不代表 GPU 已执行完成。
 */
void add(float *x, const float *y, int n);
} // namespace qwe::kernels
