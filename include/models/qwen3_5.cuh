#pragma once
#include "models/qwen3.cuh"
#include "models/qwen3_5_vision.cuh"
namespace qwe {
/** Qwen3.5-0.8B 图文参考前向。单卡FP32、完整上下文重算，不支持缓存/并行。
 * 构造时加载权重并持有显存；forward输入完整token序列，返回拥有CPU内存的logits。
 * last_only=false形状[n,vocab]，true形状[1,vocab]；失败抛异常，调用后恢复原设备。
 */
class Qwen3_5 {
    struct Impl;
    std::unique_ptr<Impl> impl;

  public:
    explicit Qwen3_5(const std::string &, const Qwen3Options & = {});
    ~Qwen3_5();
    HostTensor forward(const std::vector<int> &, bool last_only = false);
    int context_limit() const;
    // 图像预处理由runtime执行；编码结果独立持有CUDA显存，可在本轮生成中重复使用。
    ImageFeatures encode_image(const ImageInput &image);
    HostTensor forward_multimodal(const std::vector<int> &tokens,
                                  const std::vector<ImageFeatures> &images,
                                  bool last_only = false);
};
} // namespace qwe
