
#include "core/dtype.h"

size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DTYPE_FP32:  return 4;
        case DTYPE_FP16:  return 2;
        case DTYPE_BF16:  return 2;
        case DTYPE_INT8:  return 1;
        case DTYPE_INT32: return 4;
        case DTYPE_INT64: return 8;
        default:          return 0;
    }
}

const char* dtype_name(DType dtype) {
    switch (dtype) {
        case DTYPE_FP32:  return "FP32";
        case DTYPE_FP16:  return "FP16";
        case DTYPE_BF16:  return "BF16";
        case DTYPE_INT8:  return "INT8";
        case DTYPE_INT32: return "INT32";
        case DTYPE_INT64: return "INT64";
        default:          return "UNKNOWN";
    }
}

