#pragma once
#include "loader/weight_archive.hpp"
#include "backend/cuda/device_tensor.cuh"
#include "operators/tensor_ops.cuh"
#include <map>
#include <memory>
#include <algorithm>
#include <vector>
namespace qwe {
struct WeightStore {
    WeightArchive archive;
    std::map<std::string, std::unique_ptr<DeviceTensor>> gpu;

    /**
     * @brief 创建按名称管理 GPU 权重的缓存并映射源文件。
     * @param[in] path 源 Safetensors 文件路径。
     * @return 构造权重缓存；显存数据由 load 按需加载。
     */
    explicit WeightStore(const std::string &path) : archive(path) {}

    /**
     * @brief 按名称加载、转换并缓存一份 GPU 权重。
     * @param[in] name Safetensors 中的完整张量名称。
     * @param[in] shape 模型期望的权重形状，必须与文件一致。
     * @return 缓存中 FP32 CUDA Tensor 的借用指针；所有者为 WeightStore。
     * @note 缓存命中直接复用；否则上传数据，必要时从 BF16/FP16 转成 FP32。
     */
    const Tensor *load(const std::string &name,
                       std::initializer_list<int64_t> shape,
                       std::initializer_list<int64_t> storage_shape = {}) {
        // 同一份权重只上传一次；返回的指针在 WeightStore 存活期间有效。
        auto cached = gpu.find(name);
        if (cached != gpu.end())
            return &cached->second->tensor;
        const auto &file = archive.file_for(name);
        const SafeTensorEntry *e = safetensors_find(&file, name.c_str());
        auto file_shape = storage_shape.size() ? storage_shape : shape;
        if (!e || e->ndim != (int)file_shape.size())
            throw std::runtime_error("missing or invalid weight: " + name);
        int i = 0;
        for (auto d : file_shape)
            if (e->shape[i++] != d)
                throw std::runtime_error("weight shape mismatch: " + name);
        size_t numel = 1;
        for (auto d : shape) {
            if (d < 1 || numel > SIZE_MAX / (size_t)d)
                throw std::runtime_error("invalid weight view");
            numel *= d;
        }
        if (numel != e->numel)
            throw std::runtime_error("weight reshape size mismatch: " + name);
        if (e->dtype != DTYPE_FP32 && e->dtype != DTYPE_FP16 && e->dtype != DTYPE_BF16)
            throw std::runtime_error("unsupported weight dtype: " + name);

        // 首版算子统一使用 FP32；BF16/FP16 权重在上传后转换一次。
        auto out = std::make_unique<DeviceTensor>(DTYPE_FP32, shape);
        if (e->dtype == DTYPE_FP32)
            check(cudaMemcpy(out->tensor.data,
                             safetensors_get_data(&file, e),
                             out->tensor.nbytes,
                             cudaMemcpyHostToDevice));
        else {
            DeviceTensor raw(e->dtype, shape);
            check(cudaMemcpy(raw.tensor.data,
                             safetensors_get_data(&file, e),
                             raw.tensor.nbytes,
                             cudaMemcpyHostToDevice));
            convert(out->tensor, raw.tensor);
        }
        const Tensor *result = &out->tensor;
        gpu.emplace(name, std::move(out));
        return result;
    }
    // Pack only this rank's contiguous row/column slice before uploading.
    const Tensor *shard(const std::string &name, int rows, int cols, int axis, int rank) {
        const auto &file = archive.file_for(name);
        const auto *e = safetensors_find(&file, name.c_str());
        if (!e || e->ndim != 2 || e->shape[0] != rows || e->shape[1] != cols)
            throw std::runtime_error("weight shape mismatch: " + name);
        if (e->dtype != DTYPE_FP32 && e->dtype != DTYPE_FP16 && e->dtype != DTYPE_BF16)
            throw std::runtime_error("unsupported weight dtype: " + name);
        int r = axis == 0 ? rows / 2 : rows, c = axis == 1 ? cols / 2 : cols;
        size_t width = dtype_size(e->dtype);
        std::vector<unsigned char> packed((size_t)r * c * width);
        const auto *source = static_cast<const unsigned char *>(safetensors_get_data(&file, e));
        for (int i = 0; i < r; ++i) {
            size_t offset =
                (size_t)(i + (axis == 0 ? rank * r : 0)) * cols + (axis == 1 ? rank * c : 0);
            std::copy_n(
                source + offset * width, (size_t)c * width, packed.data() + (size_t)i * c * width);
        }
        auto out = std::make_unique<DeviceTensor>(DTYPE_FP32, std::initializer_list<int64_t>{r, c});
        DeviceTensor raw(e->dtype, {r, c});
        check(cudaMemcpy(raw.tensor.data, packed.data(), packed.size(), cudaMemcpyHostToDevice));
        if (e->dtype == DTYPE_FP32)
            check(cudaMemcpy(
                out->tensor.data, raw.tensor.data, out->tensor.nbytes, cudaMemcpyDeviceToDevice));
        else
            convert(out->tensor, raw.tensor);
        const Tensor *result = &out->tensor;
        gpu.emplace(name, std::move(out));
        return result;
    }
};

} // namespace qwe
