#pragma once
#include "runtime/image_input.h"
#include "backend/cuda/device_tensor.cuh"
namespace qwe {
/** 图片编码结果：[grid_h*grid_w/4,1024] CUDA FP32；embeddings持有显存。 */
struct ImageFeatures {
    int grid_h, grid_w;
    std::unique_ptr<DeviceTensor> embeddings;
};
class Qwen3_5Vision {
    struct Impl;
    std::unique_ptr<Impl> impl;

  public:
    explicit Qwen3_5Vision(const std::string &directory);
    ~Qwen3_5Vision();
    ImageFeatures encode(const ImageInput &image);
};
} // namespace qwe
