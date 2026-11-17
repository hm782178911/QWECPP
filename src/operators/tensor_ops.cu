#include "operators/tensor_ops.cuh"
#include "backend/cuda/error.cuh"
#include "cuda_kernels.cuh"
#include <climits>
#include <cmath>
namespace qwe {
/**
 * @brief 校验一个前置条件。
 * @param[in] ok 条件是否满足。
 * @param[in] message 条件失败时的异常信息。
 * @return 无；ok=false 时抛 std::runtime_error。
 */
static void require(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}

/**
 * @brief 检查 Tensor 能否安全交给当前 CUDA 算子。
 * @param[in] t 待校验的 Tensor 元数据。
 * @param[in] type 要求的数据类型，默认 FP32。
 * @return 无；类型、设备、形状、尺寸或连续步长不合法时抛异常。
 */
static void validate(const Tensor &t, DType type = DTYPE_FP32) {
    require(t.data && t.dtype == type && t.device == TENSOR_CUDA,
            "expected CUDA Tensor with supported dtype");
    int current;
    check(cudaGetDevice(&current));
    require(t.device_id == current, "Tensor is on another CUDA device");
    Tensor expected{};
    require(tensor_init(&expected, type, t.ndim, t.shape) == 0, "invalid Tensor metadata");
    require(t.numel == expected.numel && t.nbytes == expected.nbytes && t.numel <= INT_MAX - 255,
            "invalid Tensor size or exceeds kernel limits");
    for (int i = 0; i < t.ndim; ++i)
        require(t.stride[i] == expected.stride[i], "operator requires contiguous Tensor");
}

/**
 * @brief 判断两个 Tensor 的维数和各维长度是否一致。
 * @param[in] a 第一个 Tensor。
 * @param[in] b 第二个 Tensor。
 * @return 同形返回 true，否则 false；不比较 dtype、设备或数据。
 */
static bool same_shape(const Tensor &a, const Tensor &b) {
    if (a.ndim != b.ndim)
        return false;
    for (int i = 0; i < a.ndim; ++i)
        if (a.shape[i] != b.shape[i])
            return false;
    return true;
}

/**
 * @brief 提取已校验 FP32 Tensor 的显存地址。
 * @param[in] t 调用方已确认其 dtype 为 FP32 的 Tensor。
 * @return 借用的 float*；不执行内存分配或主机/设备传输。
 */
static float *ptr(const Tensor &t) {
    return static_cast<float *>(t.data);
}

/**
 * @brief 转换元素类型并保持形状不变。
 * @param[out] out 已分配的 FP32 CUDA Tensor，与输入同形。
 * @param[in] in 连续 FP32/FP16/BF16 CUDA Tensor。
 * @return 无返回值；结果通过输出参数写入。
 * @note 不分配输出内存；异步提交到默认 stream。
 */
void convert(Tensor &out, const Tensor &in) {
    validate(out);
    require(in.dtype == DTYPE_FP32 || in.dtype == DTYPE_FP16 || in.dtype == DTYPE_BF16,
            "unsupported conversion dtype");
    validate(in, in.dtype);
    require(same_shape(out, in), "convert shape mismatch");
    kernels::convert(ptr(out), in.data, in.numel, (int)in.dtype);
}

/**
 * @brief 根据 token ID 从词向量表中查出隐藏状态。
 * @param[out] out 已分配的 FP32 输出，形状 [n, h]。
 * @param[in] w FP32 词向量表，形状 [词表大小, h]。
 * @param[in] ids INT32 CUDA Tensor，形状 [n]，每个 ID 必须在词表范围内。
 * @return 无返回值；结果通过输出参数写入。
 * @note ID 范围由调用方校验；本函数检查类型、布局、设备和形状。
 */
void embedding(Tensor &out, const Tensor &w, const Tensor &ids) {
    validate(out);
    validate(w);
    validate(ids, DTYPE_INT32);
    require(w.ndim == 2 && ids.ndim == 1 && out.ndim == 2 && out.shape[0] == ids.shape[0] &&
                out.shape[1] == w.shape[1],
            "embedding shape mismatch");
    kernels::embedding(
        ptr(out), ptr(w), static_cast<const int *>(ids.data), (int)ids.numel, (int)w.shape[1]);
}

/**
 * @brief 对最后一维执行 RMSNorm 并乘以逐通道权重。
 * @param[out] out 与 in 同形的 FP32 输出，可与 in 完全原地共享。
 * @param[in] in 连续 FP32 CUDA Tensor。
 * @param[in] w 一维 FP32 权重，长度等于输入最后一维。
 * @param[in] eps 正数稳定项，结果为 in / sqrt(mean(in²)+eps) × weight。
 * @return 无返回值；结果通过输出参数写入。
 */
void norm(Tensor &out, const Tensor &in, const Tensor &w, float eps) {
    validate(out);
    validate(in);
    validate(w);
    require(same_shape(out, in) && w.ndim == 1 && w.shape[0] == in.shape[in.ndim - 1],
            "RMSNorm shape mismatch");
    require(std::isfinite(eps) && eps > 0, "invalid RMSNorm epsilon");
    int width = (int)w.numel;
    kernels::norm(ptr(out), ptr(in), ptr(w), (int)(in.numel / width), width, eps);
}

/**
 * @brief 对 Query 或 Key 原地施加旋转位置编码。
 * @param[in,out] x FP32 Tensor，形状 [n, heads, d]，d 必须为偶数。
 * @param[in] theta 正数 RoPE 频率基数；位置从 0 到 n-1。
 * @return 无返回值；结果通过输出参数写入。
 * @note 返回后 x 的数值被修改，形状和内存地址不变。
 */
void rope(Tensor &x, float theta) {
    validate(x);
    require(x.ndim == 3 && x.shape[2] % 2 == 0, "RoPE requires [tokens, heads, even head_dim]");
    require(std::isfinite(theta) && theta > 0, "invalid RoPE theta");
    kernels::rope(ptr(x), (int)x.shape[0], (int)x.shape[1], (int)x.shape[2], theta);
}

/**
 * @brief 计算不读取未来位置的分组查询注意力。
 * @param[out] out FP32 输出，形状 [n, Q头数, d]，不能与输入共享起始地址。
 * @param[in] q Query，形状 [n, Q头数, d]。
 * @param[in] k Key，形状 [n, KV头数, d]。
 * @param[in] v Value，形状必须与 k 相同。
 * @return 无返回值；结果通过输出参数写入。
 * @note n≤8192；Q头数必须是 KV头数的整数倍；不复制或缓存历史 K/V。
 */
void attention(Tensor &out, const Tensor &q, const Tensor &k, const Tensor &v) {
    validate(out);
    validate(q);
    validate(k);
    validate(v);
    require(q.ndim == 3 && k.ndim == 3 && same_shape(out, q) && same_shape(k, v),
            "attention rank/shape mismatch");
    require(q.shape[0] == k.shape[0] && q.shape[2] == k.shape[2] && q.shape[1] % k.shape[1] == 0 &&
                q.shape[0] <= 8192,
            "unsupported GQA shape");
    require(out.data != q.data && out.data != k.data && out.data != v.data,
            "attention output cannot alias inputs");
    kernels::attention(ptr(out),
                       ptr(q),
                       ptr(k),
                       ptr(v),
                       (int)q.shape[0],
                       (int)q.shape[1],
                       (int)k.shape[1],
                       (int)q.shape[2]);
}

/**
 * @brief 执行 SwiGLU 的激活与门控乘法。
 * @param[in,out] gate FP32 Tensor，原地更新为 SiLU(gate) × up。
 * @param[in] up 与 gate 同形的 FP32 Tensor。
 * @return 无返回值；结果通过输出参数写入。
 */
void silu_mul(Tensor &gate, const Tensor &up) {
    validate(gate);
    validate(up);
    require(same_shape(gate, up), "SwiGLU shape mismatch");
    kernels::silu_mul(ptr(gate), ptr(up), (int)gate.numel);
}

/**
 * @brief 执行逐元素残差相加。
 * @param[in,out] x FP32 Tensor，原地更新为 x+y。
 * @param[in] y 与 x 同形的 FP32 Tensor。
 * @return 无返回值；结果通过输出参数写入。
 */
void add(Tensor &x, const Tensor &y) {
    validate(x);
    validate(y);
    require(same_shape(x, y), "residual shape mismatch");
    kernels::add(ptr(x), ptr(y), (int)x.numel);
}

/**
 * @brief 执行不带 bias 的线性投影 out = in × weightᵀ。
 * @param[in] handle 绑定默认 CUDA stream 的有效 cuBLAS 句柄。
 * @param[out] out FP32 输出，形状 [n, 输出宽度]，不能与输入/权重同起始地址。
 * @param[in] in FP32 输入，形状 [n, 输入宽度]。
 * @param[in] w FP32 权重，形状 [输出宽度, 输入宽度]。
 * @return 无返回值；结果通过输出参数写入。
 * @note 使用 cuBLAS，不自行分配输出；元数据或 stream 不支持时抛异常。
 */
void linear(cublasHandle_t handle, Tensor &out, const Tensor &in, const Tensor &w) {
    validate(out);
    validate(in);
    validate(w);
    require(in.ndim == 2 && w.ndim == 2 && out.ndim == 2 && in.shape[1] == w.shape[1] &&
                out.shape[0] == in.shape[0] && out.shape[1] == w.shape[0],
            "Linear shape mismatch");
    require(out.data != in.data && out.data != w.data, "Linear output cannot alias inputs");
    cudaStream_t stream;
    check(cublasGetStream(handle, &stream));
    require(stream == nullptr || stream == cudaStreamLegacy,
            "operators currently require the default CUDA stream");
    float a = 1, b = 0;
    check(cublasSgemm(handle,
                      CUBLAS_OP_T,
                      CUBLAS_OP_N,
                      (int)w.shape[0],
                      (int)in.shape[0],
                      (int)in.shape[1],
                      &a,
                      ptr(w),
                      (int)w.shape[1],
                      ptr(in),
                      (int)in.shape[1],
                      &b,
                      ptr(out),
                      (int)out.shape[1]));
}
} // namespace qwe
