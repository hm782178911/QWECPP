#pragma once
#include "core/tensor.h"
namespace qwe {
struct TransferStats {
    unsigned long long calls=0, bytes=0, p2p_calls=0, host_calls=0;
    double total_ms=0, source_wait_ms=0;
};
/**
 * @brief 将同形、同类型的连续 CUDA Tensor 从一个设备传到另一个设备。
 * @param[out] out 已分配的目标 Tensor。
 * @param[in] in 源 Tensor，不修改其数值。
 * @param[in] force_host true 强制使用固定页主机内存中转，供验证/诊断使用。
 * @return 无；返回前目标数据已可供下一阶段读取。
 * @note 优先启用可用的 P2P；不能直接访问时自动中转；恢复调用方当前设备。
 */
void transfer(Tensor &out, const Tensor &in, bool force_host = false, TransferStats *stats = nullptr);
} // namespace qwe
