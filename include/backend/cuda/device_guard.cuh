#pragma once
#include "backend/cuda/error.cuh"
namespace qwe {
/**
 * @brief 临时选择一个 CUDA 设备，离开作用域时恢复原设备。
 * @param[in] device 目标设备编号。
 * @return 构造设备作用域；CUDA API 失败时抛异常。
 */
class DeviceGuard {
    int previous;

  public:
    explicit DeviceGuard(int device) {
        check(cudaGetDevice(&previous));
        check(cudaSetDevice(device));
    }
    /**
     * @brief 恢复进入作用域前的 CUDA 设备。
     * @return 无；析构不抛异常。
     */
    ~DeviceGuard() {
        cudaSetDevice(previous);
    }
    DeviceGuard(const DeviceGuard &) = delete;
    DeviceGuard &operator=(const DeviceGuard &) = delete;
};
} // namespace qwe
