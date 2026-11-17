#pragma once
#include "core/tensor.hpp"
#include "backend/cuda/error.cuh"
#include "backend/cuda/device_guard.cuh"
namespace qwe {
/* 管理 cudaMalloc/cudaFree 生命周期；tensor 只是该显存的非拥有型视图。
 * 禁止复制，避免重复释放；移动时转移所有权，并清空原对象的视图。 */
class DeviceTensor {
  public:
    Tensor tensor;

    /**
     * @brief 在当前 CUDA 设备上分配张量显存。
     * @param[in] dtype 显存数据的元素类型。
     * @param[in] shape 张量形状，决定分配字节数。
     * @return 构造对象，tensor 记录显存地址、形状和当前设备编号。
     * @note 显存内容未初始化；对象销毁时释放，禁止复制以避免重复释放。
     */
    explicit DeviceTensor(DType dtype, std::initializer_list<int64_t> shape)
        : tensor(make_tensor(dtype, shape)) {
        check(cudaGetDevice(&tensor.device_id));
        tensor.device = TENSOR_CUDA;
        check(cudaMalloc(&tensor.data, tensor.nbytes));
    }

    /**
     * @brief 释放本对象持有的显存。
     * @return 无；此后引用这块显存的 Tensor 视图全部失效。
     */
    ~DeviceTensor() {
        // 析构时可能已经切换到另一阶段，必须回到分配显存的设备再释放。
        if (tensor.data) {
            int previous = 0;
            if (cudaGetDevice(&previous) == cudaSuccess &&
                cudaSetDevice(tensor.device_id) == cudaSuccess) {
                cudaFree(tensor.data);
                cudaSetDevice(previous);
            }
        }
    }
    DeviceTensor(const DeviceTensor &) = delete;
    DeviceTensor &operator=(const DeviceTensor &) = delete;

    /**
     * @brief 转移显存所有权，不复制显存内容。
     * @param[in,out] x 被移动对象；移动后其 Tensor 元数据被清空。
     * @return 构造新的显存所有者。
     */
    DeviceTensor(DeviceTensor &&x) noexcept : tensor(x.tensor) {
        tensor_destroy(&x.tensor);
    }
};
} // namespace qwe
