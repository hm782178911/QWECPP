#include "loader/safetensors.h"
#include <stdio.h>
#include <string.h>
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "line %d: %s\n", __LINE__, #x);                                        \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)
int main(int argc, char **argv) {
    CHECK(argc == 2);
    SafeTensorFile f = {0};
    CHECK(safetensors_open(NULL, argv[1]) != 0);
    CHECK(safetensors_open(&f, NULL) != 0 && f.mapped_data == NULL);
    CHECK(safetensors_open(&f, argv[1]) == 0);
    CHECK(safetensors_count(&f) == 1);
    const SafeTensorEntry *e = safetensors_find(&f, "x");
    CHECK(e && e->dtype == DTYPE_FP32 && e->ndim == 1 && e->shape[0] == 2 && e->numel == 2);
    const void *data = safetensors_get_data(&f, e);
    const unsigned char expected[] = {0, 0, 128, 63, 0, 0, 0, 64};
    CHECK(data && memcmp(data, expected, 8) == 0);
    CHECK(safetensors_find(&f, "missing") == NULL);
    CHECK(safetensors_find(NULL, "x") == NULL && safetensors_find(&f, NULL) == NULL);
    SafeTensorEntry other = {0};
    CHECK(safetensors_get_data(&f, &other) == NULL);
    CHECK(safetensors_get_data(NULL, e) == NULL && safetensors_get_data(&f, NULL) == NULL);
    CHECK(safetensors_open(&f, argv[1]) != 0);
    CHECK(safetensors_get_data(&f, e) == data);
    safetensors_close(&f);
    CHECK(f.mapped_data == NULL && f.entries == NULL && safetensors_count(&f) == 0);
    safetensors_close(&f);
    safetensors_close(NULL);
    CHECK(safetensors_count(NULL) == 0);
    for (int i = 0; i < 32; ++i) {
        CHECK(safetensors_open(&f, argv[1]) == 0);
        safetensors_close(&f);
    }
    return 0;
}
