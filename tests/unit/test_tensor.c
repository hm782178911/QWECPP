#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "core/tensor.h"
/* CHECK 不受 NDEBUG 影响，Release 构建下仍会执行实际校验。 */
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x);                                \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)
int main(void) {
    Tensor t;
    int64_t shape[] = {2, 3, 4};
    size_t sizes[] = {4, 2, 2, 1, 4, 8};
    for (int d = DTYPE_FP32; d <= DTYPE_INT64; ++d) {
        CHECK(dtype_size((DType)d) == sizes[d]);
        CHECK(strcmp(dtype_name((DType)d), "UNKNOWN") != 0);
        CHECK(tensor_init(&t, (DType)d, 3, shape) == 0);
        CHECK(t.dtype == (DType)d && t.ndim == 3 && t.data == NULL);
        CHECK(t.shape[0] == 2 && t.shape[1] == 3 && t.shape[2] == 4);
        CHECK(t.stride[0] == 12 && t.stride[1] == 4 && t.stride[2] == 1);
        CHECK(tensor_numel(&t) == 24 && tensor_nbytes(&t) == 24 * sizes[d]);
        int borrowed = 1;
        t.data = &borrowed;
        tensor_destroy(&t);
        CHECK(t.data == NULL && t.ndim == 0 && t.numel == 0 && t.nbytes == 0 && borrowed == 1);
        tensor_destroy(&t);
    }
    CHECK(tensor_init(NULL, DTYPE_FP32, 3, shape) != 0);
    CHECK(tensor_init(&t, DTYPE_FP32, 3, NULL) != 0);
    CHECK(tensor_init(&t, DTYPE_FP32, 0, shape) != 0);
    CHECK(tensor_init(&t, DTYPE_FP32, 5, shape) != 0);
    CHECK(tensor_init(&t, (DType)99, 3, shape) != 0);
    int64_t zero[] = {0}, negative[] = {-1}, overflow[] = {INT64_MAX, 3};
    CHECK(tensor_init(&t, DTYPE_FP32, 1, zero) != 0);
    CHECK(tensor_init(&t, DTYPE_FP32, 1, negative) != 0);
    CHECK(tensor_init(&t, DTYPE_FP32, 2, overflow) != 0);
    CHECK(tensor_init(&t, DTYPE_FP32, 1, overflow) != 0);
    CHECK(dtype_size((DType)99) == 0);
    CHECK(strcmp(dtype_name((DType)99), "UNKNOWN") == 0);
    CHECK(tensor_numel(NULL) == 0 && tensor_nbytes(NULL) == 0);
    tensor_destroy(NULL);
    puts("Tensor checks passed");
    return 0;
}
