
#include <string.h>
#include <limits.h>

#include "core/tensor.h"
#include "core/dtype.h"

int tensor_init(Tensor* tensor, DType dtype, int ndim,
                const int64_t* shape) {
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

    tensor->dtype = dtype;
    tensor->ndim = ndim;
    tensor->numel = 1;

    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0) {
            memset(tensor, 0, sizeof(Tensor));
            return -1;
        }

        if ((uint64_t)tensor->numel >
            SIZE_MAX / (uint64_t)shape[i]) {
            memset(tensor, 0, sizeof(Tensor));
            return -1;
        }

        tensor->shape[i] = shape[i];
        tensor->numel *= (size_t)shape[i];
    }

    // Row-major contiguous strides.
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

void tensor_destroy(Tensor* tensor) {
    if (tensor == NULL) {
        return;
    }

    // data is non-owning in the current design.
    memset(tensor, 0, sizeof(Tensor));
}

size_t tensor_numel(const Tensor* tensor) {
    return tensor ? tensor->numel : 0;
}

size_t tensor_nbytes(const Tensor* tensor) {
    if (tensor == NULL) {
        return 0;
    }

    return tensor->nbytes;
}

