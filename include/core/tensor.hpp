#pragma once
#include "core/tensor.h"
#include <initializer_list>
#include <memory>
#include <stdexcept>
namespace qwe {
/**
 * @brief 创建只含元数据的连续 Tensor。
 * @param[in] dtype 元素类型。
 * @param[in] shape 1～4 维正整数形状。
 * @return data=NULL 的 CPU Tensor；形状或大小非法时抛异常。
 */
inline Tensor make_tensor(DType dtype, std::initializer_list<int64_t> shape) {
    Tensor t{};
    if (tensor_init(&t, dtype, (int)shape.size(), shape.begin()))
        throw std::runtime_error("invalid Tensor shape/type");
    return t;
}

/**
 * @brief 创建共享原数据的连续形状视图。
 * @param[in] source 元数据有效的连续 Tensor，不修改原对象。
 * @param[in] shape 目标形状，元素数必须保持不变。
 * @return 新的非拥有型 Tensor，data 和设备信息与 source 一致。
 * @note 不复制数据；视图不能比原数据的所有者活得更久。
 */
inline Tensor reshape(const Tensor &source, std::initializer_list<int64_t> shape) {
    Tensor expected{};
    if (tensor_init(&expected, source.dtype, source.ndim, source.shape) ||
        source.numel != expected.numel || source.nbytes != expected.nbytes)
        throw std::runtime_error("invalid source Tensor metadata");
    for (int i = 0; i < source.ndim; ++i)
        if (source.stride[i] != expected.stride[i])
            throw std::runtime_error("reshape requires contiguous Tensor");
    Tensor out = make_tensor(source.dtype, shape);
    if (out.numel != source.numel)
        throw std::runtime_error("reshape changes element count");
    out.data = source.data;
    out.device = source.device;
    out.device_id = source.device_id;
    return out;
}

/**
 * @brief 取得二维张量最后一行的共享视图。
 * @param[in] source 有数据的连续二维 Tensor，形状为 [行数, 列数]。
 * @return 形状为 [1, 列数] 的非拥有型 Tensor；无内存拷贝。
 */
inline Tensor last_row(const Tensor &source) {
    if (source.ndim != 2 || !source.data)
        throw std::runtime_error("last_row requires matrix");
    auto checked = reshape(source, {source.shape[0], source.shape[1]});
    Tensor out = make_tensor(source.dtype, {1, source.shape[1]});
    out.data =
        (unsigned char *)checked.data + (source.numel - out.numel) * dtype_size(source.dtype);
    out.device = source.device;
    out.device_id = source.device_id;
    return out;
}

/* 持有 FP32 CPU 输出内存；形状、类型等信息仍使用项目原有的 C Tensor。 */
class HostTensor {
    std::unique_ptr<float[]> storage;

  public:
    Tensor tensor;

    /**
     * @brief 分配并持有一块 FP32 CPU 数据内存。
     * @param[in] shape 输出数据的形状，要求 1～4 维正整数。
     * @return 构造对象；tensor.data 指向本对象持有的内存。
     * @note 对象销毁时自动释放；外部 Tensor 视图不能单独释放该内存。
     */
    explicit HostTensor(std::initializer_list<int64_t> shape)
        : tensor(make_tensor(DTYPE_FP32, shape)) {
        storage = std::make_unique<float[]>(tensor.numel);
        tensor.data = storage.get();
    }
    HostTensor(const HostTensor &) = delete;
    HostTensor &operator=(const HostTensor &) = delete;

    /**
     * @brief 转移 CPU 内存所有权并清空原对象视图。
     * @param[in,out] other 被移动对象；移动后不再持有原数据。
     * @return 构造新的内存所有者，不复制张量数据。
     */
    HostTensor(HostTensor &&other) noexcept
        : storage(std::move(other.storage)), tensor(other.tensor) {
        tensor_destroy(&other.tensor);
    }

    /**
     * @brief 获取可写的 CPU 数据地址。
     * @return 借用的 float*，有效期不超过本对象。
     */
    float *data() {
        return static_cast<float *>(tensor.data);
    }

    /**
     * @brief 获取只读的 CPU 数据地址。
     * @return 借用的 const float*，调用方不得释放。
     */
    const float *data() const {
        return static_cast<const float *>(tensor.data);
    }

    /**
     * @brief 查询 CPU 输出中的元素总数。
     * @return tensor.numel，不是字节数。
     */
    size_t size() const {
        return tensor.numel;
    }

    /**
     * @brief 取得用于遍历的起始地址。
     * @return 指向第一个 FP32 元素的借用指针。
     */
    float *begin() {
        return data();
    }

    /**
     * @brief 取得用于遍历的结束地址。
     * @return 指向最后元素之后的指针，不可解引用。
     */
    float *end() {
        return data() + size();
    }
};
} // namespace qwe
