#include "backend/cuda/transfer.cuh"
#include "backend/cuda/device_guard.cuh"
#include <memory>
#include <map>
#include <mutex>
#include <cstring>
#include <iostream>
#include <chrono>
namespace qwe {
namespace {
struct Pinned {
    void *data = nullptr;
    /**
     * @brief 分配用于跨卡中转或完整性比较的固定页主机内存。
     * @param[in] bytes 分配字节数。
     * @return 构造内存所有者；分配失败时抛异常。
     */
    explicit Pinned(size_t bytes) {
        check(cudaMallocHost(&data, bytes));
    }
    /**
     * @brief 释放本对象持有的固定页主机内存。
     * @return 无；此前借用的 data 指针失效。
     */
    ~Pinned() {
        if (data)
            cudaFreeHost(data);
    }
};
/**
 * @brief 校验传输对象的设备位置、大小和连续布局。
 * @param[in] t 要检查的 Tensor。
 * @return 无；元数据不合法时抛异常。
 */
void validate(const Tensor &t) {
    Tensor expected{};
    if (!t.data || t.device != TENSOR_CUDA || t.device_id < 0 ||
        tensor_init(&expected, t.dtype, t.ndim, t.shape) || expected.numel != t.numel ||
        expected.nbytes != t.nbytes)
        throw std::runtime_error("invalid transfer Tensor");
    for (int i = 0; i < t.ndim; ++i)
        if (t.stride[i] != expected.stride[i])
            throw std::runtime_error("transfer requires contiguous Tensor");
}
} // namespace
void transfer(Tensor &out, const Tensor &in, bool force_host, TransferStats *stats) {
    using Clock=std::chrono::steady_clock;
    struct Timer {
        TransferStats *stats;
        Clock::time_point start;
        ~Timer() { if(stats) stats->total_ms += std::chrono::duration<double,std::milli>(Clock::now()-start).count(); }
    } timer{stats, stats ? Clock::now() : Clock::time_point{}};
    validate(out);
    validate(in);
    if (out.dtype != in.dtype || out.ndim != in.ndim)
        throw std::runtime_error("transfer dtype/rank mismatch");
    for (int i = 0; i < in.ndim; ++i)
        if (out.shape[i] != in.shape[i])
            throw std::runtime_error("transfer shape mismatch");
    if(stats) { ++stats->calls; stats->bytes += in.nbytes; }
    auto wait_start=stats ? Clock::now() : Clock::time_point{};
    // 第一版显式完成源阶段，避免把跨设备默认 stream 的隐式次序当成同步保证。
    {
        DeviceGuard source(in.device_id);
        check(cudaDeviceSynchronize());
    }
    if(stats) stats->source_wait_ms += std::chrono::duration<double,std::milli>(Clock::now()-wait_start).count();
    DeviceGuard destination(out.device_id);
    if (in.device_id == out.device_id) {
        check(cudaMemcpy(out.data, in.data, in.nbytes, cudaMemcpyDeviceToDevice));
        return;
    }
    // 可访问标志不等于实际链路数据正确。每个方向首次使用时核对真实字节，
    // 发现静默错误后，本进程不再使用该方向的 P2P。
    static std::mutex route_mutex;
    static std::map<std::pair<int, int>, bool> verified_routes;
    std::lock_guard<std::mutex> lock(route_mutex);
    auto key = std::make_pair(in.device_id, out.device_id);
    auto known = verified_routes.find(key);
    int accessible = 0;
    if (!force_host && (known == verified_routes.end() || known->second))
        check(cudaDeviceCanAccessPeer(&accessible, out.device_id, in.device_id));
    if (accessible) {
        auto status = cudaDeviceEnablePeerAccess(in.device_id, 0);
        if (status == cudaErrorPeerAccessAlreadyEnabled)
            cudaGetLastError();
        else if (status != cudaSuccess) {
            cudaGetLastError();
            accessible = 0;
        }
    }
    if (accessible) {
        auto copy_status =
            cudaMemcpyPeer(out.data, out.device_id, in.data, in.device_id, in.nbytes);
        if (copy_status != cudaSuccess) {
            // 致命执行错误不能靠换传输通道修复；其他复制错误禁用此方向并回退。
            if (copy_status == cudaErrorIllegalAddress || copy_status == cudaErrorAssert ||
                copy_status == cudaErrorLaunchFailure || copy_status == cudaErrorLaunchTimeout ||
                copy_status == cudaErrorECCUncorrectable)
                check(copy_status);
            cudaGetLastError();
            verified_routes[key] = false;
            accessible = 0;
            std::cerr << "P2P copy failed; using pinned host transfer: "
                      << cudaGetErrorString(copy_status) << "\n";
        }
    }
    if (accessible) {
        bool used_host = false;
        check(cudaDeviceSynchronize());
        if (known == verified_routes.end()) {
            Pinned source_bytes(in.nbytes), target_bytes(in.nbytes);
            {
                DeviceGuard source(in.device_id);
                check(cudaMemcpy(source_bytes.data, in.data, in.nbytes, cudaMemcpyDeviceToHost));
            }
            check(cudaMemcpy(target_bytes.data, out.data, out.nbytes, cudaMemcpyDeviceToHost));
            bool correct = std::memcmp(source_bytes.data, target_bytes.data, in.nbytes) == 0;
            verified_routes[key] = correct;
            if (!correct) {
                used_host = true;
                std::cerr << "P2P integrity check failed for GPU " << in.device_id << " -> "
                          << out.device_id << "; using pinned host transfer.\n";
                check(cudaMemcpy(out.data, source_bytes.data, in.nbytes, cudaMemcpyHostToDevice));
            }
        }
        if(stats) { if(used_host) ++stats->host_calls; else ++stats->p2p_calls; }
        return;
    }
    if(stats) ++stats->host_calls;
    Pinned host(in.nbytes);
    {
        DeviceGuard source(in.device_id);
        check(cudaMemcpy(host.data, in.data, in.nbytes, cudaMemcpyDeviceToHost));
    }
    check(cudaMemcpy(out.data, host.data, in.nbytes, cudaMemcpyHostToDevice));
}
} // namespace qwe
