#pragma once
#include "core/tensor.h"
#include <cublas_v2.h>
namespace qwe {
/* 默认接收连续布局的 FP32 CUDA Tensor；convert 输入另支持 BF16/FP16，
 * Embedding 的 token ID 使用 INT32，调用方须先检查 ID 没有超出词表。
 * 当前统一使用默认 CUDA stream；Linear 会拒绝绑定其他 stream 的句柄。
 * RMSNorm、RoPE、silu_mul 和 add 支持指定的原地计算，不保证任意重叠视图安全。 */

/**
 * @brief 转换元素类型并保持形状不变。
 * @param[out] out 已分配的 FP32 CUDA Tensor，与输入同形。
 * @param[in] in 连续 FP32/FP16/BF16 CUDA Tensor。
 * @return 无返回值；结果通过输出参数写入。
 * @note 不分配输出内存；异步提交到默认 stream。
 */
void convert(Tensor &out, const Tensor &in);

/**
 * @brief 根据 token ID 从词向量表中查出隐藏状态。
 * @param[out] out 已分配的 FP32 输出，形状 [n, h]。
 * @param[in] weight FP32 词向量表，形状 [词表大小, h]。
 * @param[in] ids INT32 CUDA Tensor，形状 [n]，每个 ID 必须在词表范围内。
 * @return 无返回值；结果通过输出参数写入。
 * @note ID 范围由调用方校验；本函数检查类型、布局、设备和形状。
 */
void embedding(Tensor &out, const Tensor &weight, const Tensor &ids);

/**
 * @brief 对最后一维执行 RMSNorm 并乘以逐通道权重。
 * @param[out] out 与 in 同形的 FP32 输出，可与 in 完全原地共享。
 * @param[in] in 连续 FP32 CUDA Tensor。
 * @param[in] weight 一维 FP32 权重，长度等于输入最后一维。
 * @param[in] eps 正数稳定项，结果为 in / sqrt(mean(in²)+eps) × weight。
 * @return 无返回值；结果通过输出参数写入。
 */
void norm(Tensor &out, const Tensor &in, const Tensor &weight, float eps);

/**
 * @brief 对 Query 或 Key 原地施加旋转位置编码。
 * @param[in,out] x FP32 Tensor，形状 [n, heads, d]，d 必须为偶数。
 * @param[in] theta 正数 RoPE 频率基数；位置从 0 到 n-1。
 * @return 无返回值；结果通过输出参数写入。
 * @note 返回后 x 的数值被修改，形状和内存地址不变。
 */
void rope(Tensor &x, float theta, int position_offset = 0);

/**
 * @brief 计算不读取未来位置的分组查询注意力。
 * @param[out] out FP32 输出，形状 [n, Q头数, d]，不能与输入共享起始地址。
 * @param[in] q Query，形状 [n, Q头数, d]。
 * @param[in] k Key，形状 [n, KV头数, d]。
 * @param[in] v Value，形状必须与 k 相同。
 * @return 无返回值；结果通过输出参数写入。
 * @note n≤8192；Q头数必须是 KV头数的整数倍；不复制或缓存历史 K/V。
 */
void attention(Tensor &out, const Tensor &q, const Tensor &k, const Tensor &v);

/**
 * @brief 执行 SwiGLU 的激活与门控乘法。
 * @param[in,out] gate FP32 Tensor，原地更新为 SiLU(gate) × up。
 * @param[in] up 与 gate 同形的 FP32 Tensor。
 * @return 无返回值；结果通过输出参数写入。
 */
void silu_mul(Tensor &gate, const Tensor &up);

/**
 * @brief 执行逐元素残差相加。
 * @param[in,out] x FP32 Tensor，原地更新为 x+y。
 * @param[in] y 与 x 同形的 FP32 Tensor。
 * @return 无返回值；结果通过输出参数写入。
 */
void add(Tensor &x, const Tensor &y);

/**
 * @brief 执行不带 bias 的线性投影 out = in × weightᵀ。
 * @param[in] handle 绑定默认 CUDA stream 的有效 cuBLAS 句柄。
 * @param[out] out FP32 输出，形状 [n, 输出宽度]，不能与输入/权重同起始地址。
 * @param[in] in FP32 输入，形状 [n, 输入宽度]。
 * @param[in] weight FP32 权重，形状 [输出宽度, 输入宽度]。
 * @return 无返回值；结果通过输出参数写入。
 * @note 使用 cuBLAS，不自行分配输出；元数据或 stream 不支持时抛异常。
 */
void linear(cublasHandle_t handle, Tensor &out, const Tensor &in, const Tensor &weight);
} // namespace qwe

namespace qwe {
// q is suffix [queries,Qheads,dim], k/v contain [offset+queries,KVheads,dim].
void attention_cached(Tensor &out,const Tensor &q,const Tensor &k,const Tensor &v,int offset);
}
