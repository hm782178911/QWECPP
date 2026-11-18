#pragma once
#include <string>
#include <vector>
#include <memory>
#include "core/tensor.hpp"
#include "backend/cuda/transfer.cuh"
namespace qwe {
/** 单卡保持原行为；两卡模式按层连续划分两个阶段，阶段间传递隐藏状态。 */
struct Qwen3Options {
    std::vector<int> devices;         // 空列表表示使用当前设备；当前支持一个或两个不同设备。
    int max_context = 2048; // 当前引擎支持 1..8192，再取模型位置上限。
    bool kv_cache = false; // last_only 前向复用本实例的精确前缀 K/V。 // Optional last-position incremental inference.
    bool profile = false; // 启用通信诊断计时。
    bool tensor_parallel = false;    // 两卡张量并行；不与 split_layer 混用。
    int split_layer = -1;             // 第二阶段的起始层，-1 表示均分；仅两卡有效。
    bool force_host_transfer = false; // 强制通过固定页主机内存中转隐藏状态。
};
class Qwen3 {
    struct Impl;
    std::unique_ptr<Impl> impl;

  public:
    /**
     * @brief 读取配置、校验各层权重并将模型加载到当前 GPU。
     * @param[in] directory 包含 config.json 和单文件或带索引分片权重的模型目录。
     * @param[in] options 单卡或两卡分层执行选项。
     * @return 构造可复用的模型实例；资源在实例销毁前持续有效。
     * @note 文件、形状、类型或设备操作失败时抛异常；不执行文本分词。
     */
    explicit Qwen3(const std::string &directory, const Qwen3Options &options = {});

    /**
     * @brief 释放模型持有的显存、cuBLAS 句柄和权重文件映射。
     * @return 无；引用这些资源的外部视图随之失效。
     */
    ~Qwen3();

    /**
     * @brief 执行单条序列的完整 Qwen3 因果前向推理。
     * @param[in] tokens 完整上下文的 token ID，长度为 1～context_limit()，每个 ID 必须在词表内。
     * @param[in] last_only true 只投影最后位置；false 返回所有输入位置。
     * @return 拥有 CPU 内存的 HostTensor；FP32 行优先，形状为 [last_only?1:n, vocab]。
     * @note 不修改 tokens；默认重算完整序列；kv_cache 开启且 last_only=true 时复用精确前缀，仅处理新增后缀。
     * @note GPU 计算完成后复制结果到 CPU；返回对象销毁时自动释放其 CPU 内存。
     */
    HostTensor forward(const std::vector<int> &tokens, bool last_only = false);

    /**
     * @brief 查询当前实现允许输入的最大 token 数。
     * @return min(max_context, 模型配置的 max_position_embeddings)，单位为 token。
     */
    int context_limit() const;
    TransferStats last_transfer_stats() const;
    // 使前缀失效，下一次重新 prefill；保留已分配容量供复用。
    void reset_cache();
    size_t kv_cache_bytes() const;
};

/**
 * @brief 创建临时模型并执行一次全位置前向。
 * @param[in] directory 本地模型目录。
 * @param[in] tokens 完整上下文 token ID，长度与取值要求同 Qwen3::forward。
 * @return 形状为 [n, vocab] 的 FP32 CPU HostTensor。
 * @note 每次调用都会重新加载并释放模型；连续生成应复用 Qwen3 实例。
 */
HostTensor qwen3_forward(const std::string &directory, const std::vector<int> &tokens);
} // namespace qwe
