#ifndef QWECPP_DTYPE_H
#define QWECPP_DTYPE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { DTYPE_FP32, DTYPE_FP16, DTYPE_BF16, DTYPE_INT8, DTYPE_INT32, DTYPE_INT64 } DType;

/**
 * @brief 查询一种数据类型的单元素字节数。
 * @param[in] dtype 数据类型枚举。
 * @return 有效类型返回 1/2/4/8；未知类型返回 0。
 */
size_t dtype_size(DType dtype);

/**
 * @brief 查询数据类型的可读名称。
 * @param[in] dtype 数据类型枚举。
 * @return 静态字符串指针；未知类型返回 "UNKNOWN"，调用方不得释放。
 */
const char *dtype_name(DType dtype);

#ifdef __cplusplus
}
#endif

#endif // QWECPP_DTYPE_H
