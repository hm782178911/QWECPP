#include "cuda_kernels.cuh"
#include "backend/cuda/error.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
namespace qwe::kernels {

// 类型转换：每个线程读取一个 FP16/BF16 元素并写出 FP32；不改变形状。

/**
 * @brief GPU kernel：将输入元素转换为 FP32。
 * @param[out] out FP32 显存输出，至少容纳 n 个元素。
 * @param[in] in 输入显存，包含 n 个 dtype 类型的元素。
 * @param[in] n 元素总数，不是字节数。
 * @param[in] dtype 项目 DType 枚举值：0=FP32、1=FP16、2=BF16。
 * @return 无；结果写入输出或原地参数。
 * @note 由算子层预先校验维度、类型、设备、布局和缓冲区大小；本层不分配显存。
 * @note 这是设备函数，必须通过 CUDA kernel launch 调用。
 */
__global__ void convert_kernel(float *out, const void *in, size_t n, int dtype) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = dtype == 2 ? __bfloat162float(((const __nv_bfloat16 *)in)[i])
                            : __half2float(((const __half *)in)[i]);
}

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
void convert(float *out, const void *in, size_t n, int dtype) {
    if (dtype == 0)
        check(cudaMemcpy(out, in, n * sizeof(float), cudaMemcpyDeviceToDevice));
    else {
        convert_kernel<<<(n + 255) / 256, 256>>>(out, in, n, dtype);
        check(cudaGetLastError());
    }
}
} // namespace qwe::kernels
