
#include <string.h>
#include <limits.h>

#include "core/tensor.h"
#include "core/dtype.h"

/**
 * @brief 初始化连续布局的 Tensor 元数据，不分配数据内存。
 * @param[out] tensor 待初始化对象；成功后 data=NULL，默认设备为 CPU。
 * @param[in] dtype 元素类型，决定 nbytes。
 * @param[in] ndim 有效维数，取值 1～4。
 * @param[in] shape 含 ndim 个正整数的形状数组。
 * @return 成功返回 0；空指针、非法维度/类型或大小溢出返回 -1。
 */
int tensor_init(Tensor *tensor, DType dtype, int ndim, const int64_t *shape) {
    if (tensor == NULL || shape == NULL) {
        return -1;
    }

    if (ndim <= 0 || ndim > 4) {
        return -1;
    }

    size_t element_size = dtype_size(dtype);
    if (element_size == 0) {
        return -1;
    }

    memset(tensor, 0, sizeof(Tensor));

    tensor->device_id = -1;
    tensor->dtype = dtype;
    tensor->ndim = ndim;
    tensor->numel = 1;

    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0) {
            memset(tensor, 0, sizeof(Tensor));
            return -1;
        }

        if ((uint64_t)tensor->numel > SIZE_MAX / (uint64_t)shape[i]) {
            memset(tensor, 0, sizeof(Tensor));
            return -1;
        }

        tensor->shape[i] = shape[i];
        tensor->numel *= (size_t)shape[i];
    }

    // 计算行优先连续布局的步长，单位为元素。
    size_t stride = 1;

    for (int i = ndim - 1; i >= 0; i--) {
        tensor->stride[i] = (int64_t)stride;

        if (i > 0) {
            if (stride > INT64_MAX / (uint64_t)shape[i]) {
                memset(tensor, 0, sizeof(Tensor));
                return -1;
            }
            stride *= (size_t)shape[i];
        }
    }

    if (tensor->numel > SIZE_MAX / element_size) {
        memset(tensor, 0, sizeof(Tensor));
        return -1;
    }

    tensor->nbytes = tensor->numel * element_size;

    return 0;
}

/**
 * @brief 清空 Tensor 元数据，不释放 data 指向的内存。
 * @param[in,out] tensor 待清空对象；允许 NULL。
 * @return 无。
 */
void tensor_destroy(Tensor *tensor) {
    if (tensor == NULL) {
        return;
    }

    // Tensor 不拥有 data，销毁只清空元数据，不释放外部内存。
    memset(tensor, 0, sizeof(Tensor));
}

/**
 * @brief 查询张量的元素总数。
 * @param[in] tensor 目标 Tensor，允许 NULL。
 * @return numel；输入为 NULL 时返回 0。
 */
size_t tensor_numel(const Tensor *tensor) {
    return tensor ? tensor->numel : 0;
}

/**
 * @brief 查询张量数据所占字节数。
 * @param[in] tensor 目标 Tensor，允许 NULL。
 * @return nbytes；输入为 NULL 时返回 0。
 */
size_t tensor_nbytes(const Tensor *tensor) {
    if (tensor == NULL) {
        return 0;
    }

    return tensor->nbytes;
}
