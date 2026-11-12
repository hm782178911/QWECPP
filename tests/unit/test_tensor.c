
#include <stdio.h>
#include <stdint.h>

#include "core/dtype.h"
#include "core/tensor.h"

int main(void) {
    int64_t shape[] = {2, 3, 4};

    Tensor tensor;

    if (tensor_init(&tensor, DTYPE_FP32, 3, shape) != 0) {
        printf("Tensor initialization failed!\n");
        return 1;
    }

    printf("dtype:  %s\n", dtype_name(tensor.dtype));
    printf("ndim:   %d\n", tensor.ndim);

    printf("shape:  ");
    for (int i = 0; i < tensor.ndim; i++) {
        printf("%ld ", (long)tensor.shape[i]);
    }
    printf("\n");

    printf("stride: ");
    for (int i = 0; i < tensor.ndim; i++) {
        printf("%ld ", (long)tensor.stride[i]);
    }
    printf("\n");

    printf("numel:  %zu\n", tensor_numel(&tensor));
    printf("nbytes: %zu\n", tensor_nbytes(&tensor));

    tensor_destroy(&tensor);

    return 0;
}
