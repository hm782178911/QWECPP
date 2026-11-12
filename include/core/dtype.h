#ifndef QWECPP_DTYPE_H
#define QWECPP_DTYPE_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    DTYPE_FP32,
    DTYPE_FP16,
    DTYPE_BF16,
    DTYPE_INT8,
    DTYPE_INT32,
    DTYPE_INT64
} DType;

size_t dtype_size(DType dtype);
const char* dtype_name(DType dtype);

#endif // QWECPP_DTYPE_H
