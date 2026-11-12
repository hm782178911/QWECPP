#ifndef QWECPP_TENSOR_H
#define QWECPP_TENSOR_H

#include <stddef.h>
#include <stdint.h>

#include "core/dtype.h"

typedef struct {
    void* data;
    DType dtype;

    int ndim;
    int64_t shape[4];
    int64_t stride[4];

    size_t numel;
    size_t nbytes;
} Tensor;

int tensor_init(
    Tensor* tensor,
    DType dtype,
    int ndim,
    const int64_t* shape
);

void tensor_destroy(Tensor* tensor);

size_t tensor_numel(const Tensor* tensor);
size_t tensor_nbytes(const Tensor* tensor);


#endif // QWECPP_TENSOR_H
