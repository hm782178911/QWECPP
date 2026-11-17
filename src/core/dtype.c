
#include "core/dtype.h"

/**
 * @brief 查询一种数据类型的单元素字节数。
 * @param[in] dtype 数据类型枚举。
 * @return 有效类型返回 1/2/4/8；未知类型返回 0。
 */
size_t dtype_size(DType dtype) {
    switch (dtype) {
    case DTYPE_FP32:
        return 4;
    case DTYPE_FP16:
        return 2;
    case DTYPE_BF16:
        return 2;
    case DTYPE_INT8:
        return 1;
    case DTYPE_INT32:
        return 4;
    case DTYPE_INT64:
        return 8;
    default:
        return 0;
    }
}

/**
 * @brief 查询数据类型的可读名称。
 * @param[in] dtype 数据类型枚举。
 * @return 静态字符串指针；未知类型返回 "UNKNOWN"，调用方不得释放。
 */
const char *dtype_name(DType dtype) {
    switch (dtype) {
    case DTYPE_FP32:
        return "FP32";
    case DTYPE_FP16:
        return "FP16";
    case DTYPE_BF16:
        return "BF16";
    case DTYPE_INT8:
        return "INT8";
    case DTYPE_INT32:
        return "INT32";
    case DTYPE_INT64:
        return "INT64";
    default:
        return "UNKNOWN";
    }
}
