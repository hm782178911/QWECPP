#ifndef QWECPP_TENSOR_H
#define QWECPP_TENSOR_H

#include <stddef.h>
#include <stdint.h>

#include "core/dtype.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { TENSOR_CPU = 0, TENSOR_CUDA = 1 } TensorDevice;

/* 非拥有型数据视图：只描述数据，分配和释放由外部内存对象负责。 */
typedef struct {
    void *data; // 数据地址；可能指向 CPU 内存，也可能指向 CUDA 显存。
    DType dtype;

    int ndim;          // 有效维数，当前范围为 1～4。
    int64_t shape[4];  // 各维长度。
    int64_t stride[4]; // 各维步长，单位为元素，而不是字节。

    size_t numel;
    size_t nbytes;
    TensorDevice device; // 数据所在设备，结构体自身仍可位于 CPU。
    int device_id;       // CUDA 设备编号；CPU Tensor 初始化为 -1。
} Tensor;

/**
 * @brief 初始化连续布局的 Tensor 元数据，不分配数据内存。
 * @param[out] tensor 待初始化对象；成功后 data=NULL，默认设备为 CPU。
 * @param[in] dtype 元素类型，决定 nbytes。
 * @param[in] ndim 有效维数，取值 1～4。
 * @param[in] shape 含 ndim 个正整数的形状数组。
 * @return 成功返回 0；空指针、非法维度/类型或大小溢出返回 -1。
 */
int tensor_init(Tensor *tensor, DType dtype, int ndim, const int64_t *shape);

/**
 * @brief 清空 Tensor 元数据，不释放 data 指向的内存。
 * @param[in,out] tensor 待清空对象；允许 NULL。
 * @return 无。
 */
void tensor_destroy(Tensor *tensor);

/**
 * @brief 查询张量的元素总数。
 * @param[in] tensor 目标 Tensor，允许 NULL。
 * @return numel；输入为 NULL 时返回 0。
 */
size_t tensor_numel(const Tensor *tensor);

/**
 * @brief 查询张量数据所占字节数。
 * @param[in] tensor 目标 Tensor，允许 NULL。
 * @return nbytes；输入为 NULL 时返回 0。
 */
size_t tensor_nbytes(const Tensor *tensor);

#ifdef __cplusplus
}
#endif

#endif // QWECPP_TENSOR_H
