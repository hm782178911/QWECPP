#define _POSIX_C_SOURCE 200809L
#include "loader/safetensors.h"
#include "cJSON.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include <ctype.h>

/**
 * @brief 将 JSON 数值转换成精确的非负整数。
 * @param[in] v JSON 数值节点。
 * @param[out] out 成功时接收转换后的 uint64_t。
 * @return 成功为 0；小数、负数、非有限值或超过 2^53-1 时为 -1。
 */
static int integer(const cJSON *v, uint64_t *out) {
    if (!cJSON_IsNumber(v) || !isfinite(v->valuedouble) || v->valuedouble < 0 ||
        v->valuedouble > 9007199254740991.0 || floor(v->valuedouble) != v->valuedouble)
        return -1;
    *out = (uint64_t)v->valuedouble;
    return 0;
}

/**
 * @brief 检查一个 JSON 对象是否包含重复键。
 * @param[in] obj 已确认为对象的 cJSON 节点。
 * @return 无重复返回 1，有重复返回 0。
 */
static int unique(const cJSON *obj) {
    for (const cJSON *a = obj->child; a; a = a->next)
        for (const cJSON *b = a->next; b; b = b->next)
            if (strcmp(a->string, b->string) == 0)
                return 0;
    return 1;
}

/**
 * @brief 把 Safetensors 类型名称映射为项目 DType。
 * @param[in] s F32/F16/BF16/I8/I32/I64 之一。
 * @param[out] out 成功时接收对应 DType。
 * @return 支持的类型返回 0；未知类型返回 -1。
 */
static int parse_dtype(const char *s, DType *out) {
    const char *names[] = {"F32", "F16", "BF16", "I8", "I32", "I64"};
    for (int i = 0; i < 6; ++i)
        if (strcmp(s, names[i]) == 0) {
            *out = (DType)i;
            return 0;
        }
    return -1;
}

/**
 * @brief 按数据偏移和长度对两个 Entry 排序。
 * @param[in] a 第一个 SafeTensorEntry 的地址。
 * @param[in] b 第二个 SafeTensorEntry 的地址。
 * @return 小于/等于/大于时分别返回负数/0/正数，供 qsort 使用。
 */
static int order(const void *a, const void *b) {
    const SafeTensorEntry *x = a, *y = b;
    if (x->data_offset != y->data_offset)
        return x->data_offset < y->data_offset ? -1 : 1;
    return (x->data_size > y->data_size) - (x->data_size < y->data_size);
}

/**
 * @brief 解除 mmap 并释放条目名称和元数据。
 * @param[in,out] f 要关闭的对象，可为 NULL；关闭后清零。
 * @return 无。
 * @note 不释放调用方另行复制的数据；借用的 Entry/data 指针全部失效。
 */
void safetensors_close(SafeTensorFile *f) {
    if (!f)
        return;
    for (size_t i = 0; i < f->num_entries; ++i)
        free(f->entries[i].name);
    free(f->entries);
    if (f->mapped_data)
        munmap(f->mapped_data, f->file_size);
    memset(f, 0, sizeof(*f));
}

/**
 * @brief 映射并校验一个 Safetensors 文件。
 * @param[in,out] f 以 {0} 初始化的对象；复用前先 close，失败原因写入 error。
 * @param[in] path 单个权重文件路径，不接受目录。
 * @return 成功为 0，失败为 -1。
 * @note 校验头部、dtype/shape、偏移、字节数及数据区间；不把整份权重复制到内存。
 */
int safetensors_open(SafeTensorFile *f, const char *path) {
    cJSON *root = NULL;
    char *header = NULL;
    const char *why = "invalid Safetensors file";
    if (!f)
        return -1;
    if (f->mapped_data || f->entries) {
        snprintf(f->error, sizeof(f->error), "close file before reopening");
        return -1;
    }
    memset(f, 0, sizeof(*f));
    if (!path) {
        why = "null path";
        goto fail;
    }
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        why = "cannot open file";
        goto fail;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 8 ||
        (uintmax_t)st.st_size > SIZE_MAX) {
        close(fd);
        why = "invalid file size or non-regular file";
        goto fail;
    }
    f->file_size = (size_t)st.st_size;
    f->mapped_data = mmap(NULL, f->file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (f->mapped_data == MAP_FAILED) {
        f->mapped_data = NULL;
        why = "mmap failed";
        goto fail;
    }
    const unsigned char *bytes = f->mapped_data;
    uint64_t length = 0;
    for (int i = 0; i < 8; ++i)
        length |= (uint64_t)bytes[i] << (8 * i);
    if (!length || length > 100000000 || length > f->file_size - 8) {
        why = "invalid header length";
        goto fail;
    }
    f->data_start = 8 + length;
    f->data_size = f->file_size - f->data_start;
    header = malloc((size_t)length + 1);
    if (!header) {
        why = "out of memory";
        goto fail;
    }
    memcpy(header, bytes + 8, (size_t)length);
    header[length] = 0;
    if (header[0] != '{' || memchr(header, 0, (size_t)length)) {
        why = "invalid JSON header";
        goto fail;
    }
    const char *end = NULL;
    root = cJSON_ParseWithLengthOpts(header, (size_t)length + 1, &end, 0);
    if (!cJSON_IsObject(root) || !unique(root)) {
        why = "invalid JSON object or duplicate tensor name";
        goto fail;
    }
    while (end < header + length && *end == ' ')
        ++end;
    if (end != header + length) {
        why = "trailing header content";
        goto fail;
    }
    size_t count = 0;
    for (const cJSON *v = root->child; v; v = v->next)
        if (strcmp(v->string, "__metadata__") != 0)
            ++count;
    if (count > SIZE_MAX / sizeof(*f->entries)) {
        why = "too many tensors";
        goto fail;
    }
    f->entries = calloc(count ? count : 1, sizeof(*f->entries));
    if (!f->entries) {
        why = "out of memory";
        goto fail;
    }
    for (const cJSON *v = root->child; v; v = v->next) {
        if (!cJSON_IsObject(v) || !unique(v)) {
            why = "invalid entry or duplicate field";
            goto fail;
        }
        if (strcmp(v->string, "__metadata__") == 0) {
            for (const cJSON *m = v->child; m; m = m->next)
                if (!cJSON_IsString(m)) {
                    why = "metadata values must be strings";
                    goto fail;
                }
            continue;
        }
        SafeTensorEntry *e = &f->entries[f->num_entries++];
        e->name = strdup(v->string);
        if (!e->name) {
            why = "out of memory";
            goto fail;
        }
        const cJSON *dt = cJSON_GetObjectItemCaseSensitive(v, "dtype");
        const cJSON *shape = cJSON_GetObjectItemCaseSensitive(v, "shape");
        const cJSON *offsets = cJSON_GetObjectItemCaseSensitive(v, "data_offsets");
        if (!cJSON_IsString(dt) || parse_dtype(dt->valuestring, &e->dtype) != 0) {
            why = "unsupported dtype";
            goto fail;
        }
        if (!cJSON_IsArray(shape) || (e->ndim = cJSON_GetArraySize(shape)) > 8) {
            why = "unsupported shape (maximum rank 8)";
            goto fail;
        }
        e->numel = 1;
        int k = 0, empty = 0;
        for (const cJSON *d = shape->child; d; d = d->next) {
            uint64_t dim;
            if (integer(d, &dim) != 0) {
                why = "invalid shape dimension";
                goto fail;
            }
            e->shape[k++] = (int64_t)dim;
            if (!dim)
                empty = 1;
        }
        if (empty)
            e->numel = 0;
        else
            for (int i = 0; i < e->ndim; ++i) {
                uint64_t dim = (uint64_t)e->shape[i];
                if (e->numel > UINT64_MAX / dim) {
                    why = "shape product overflow";
                    goto fail;
                }
                e->numel *= dim;
            }
        uint64_t start, finish;
        if (!cJSON_IsArray(offsets) || cJSON_GetArraySize(offsets) != 2 ||
            integer(cJSON_GetArrayItem(offsets, 0), &start) != 0 ||
            integer(cJSON_GetArrayItem(offsets, 1), &finish) != 0 || start > finish ||
            finish > f->data_size) {
            why = "invalid data offsets";
            goto fail;
        }
        e->data_offset = start;
        e->data_size = finish - start;
        size_t size = dtype_size(e->dtype);
        if (e->numel > UINT64_MAX / size || e->numel * size != e->data_size) {
            why = "shape and byte length mismatch";
            goto fail;
        }
    }
    qsort(f->entries, f->num_entries, sizeof(*f->entries), order);
    uint64_t cursor = 0;
    for (size_t i = 0; i < f->num_entries; ++i) {
        if (f->entries[i].data_offset != cursor) {
            why = "overlap or hole in tensor data";
            goto fail;
        }
        cursor += f->entries[i].data_size;
    }
    if (cursor != f->data_size) {
        why = "unindexed trailing tensor data";
        goto fail;
    }
    cJSON_Delete(root);
    free(header);
    return 0;
fail:
    cJSON_Delete(root);
    free(header);
    safetensors_close(f);
    snprintf(f->error, sizeof(f->error), "%s", why);
    return -1;
}

/**
 * @brief 查询文件中的权重张量数量。
 * @param[in] f 已打开对象，也允许 NULL。
 * @return 张量数；NULL 返回 0。
 */
size_t safetensors_count(const SafeTensorFile *f) {
    return f ? f->num_entries : 0;
}

/**
 * @brief 按名称查找文件中的张量元数据。
 * @param[in] f 已打开对象。
 * @param[in] name 大小写敏感的完整张量名称。
 * @return 借用的 Entry 指针；未找到或参数为 NULL 时返回 NULL。
 * @note 返回值由 file 所有，调用方不得释放。
 */
const SafeTensorEntry *safetensors_find(const SafeTensorFile *f, const char *name) {
    if (!f || !name)
        return NULL;
    for (size_t i = 0; i < f->num_entries; ++i)
        if (strcmp(name, f->entries[i].name) == 0)
            return &f->entries[i];
    return NULL;
}

/**
 * @brief 取得某个 Entry 对应的映射数据地址。
 * @param[in] f 已打开且仍有效的文件映射。
 * @param[in] e 必须是 file 自身持有的 Entry。
 * @return 只读借用指针；参数不合法或 Entry 不属于该文件时返回 NULL。
 * @note 不做 dtype 转换；数据有效期截至文件 close。
 */
const void *safetensors_get_data(const SafeTensorFile *f, const SafeTensorEntry *e) {
    if (!f || !f->mapped_data || !e)
        return NULL;
    /* 通过 Entry 地址确认它属于当前文件，避免把其他映射的偏移用于本文件。 */
    for (size_t i = 0; i < f->num_entries; ++i)
        if (e == &f->entries[i])
            return (const unsigned char *)f->mapped_data + f->data_start + e->data_offset;
    return NULL;
}

/**
 * @brief 向标准输出打印张量名称、类型、形状和汇总。
 * @param[in] f 已校验的文件对象，NULL 时不输出。
 * @return 无；输出写入 stdout。
 */
void safetensors_inspect(const SafeTensorFile *f) {
    if (!f)
        return;
    uint64_t total = 0;
    printf("Tensor Count: %zu\n", f->num_entries);
    for (size_t i = 0; i < f->num_entries; ++i) {
        const SafeTensorEntry *e = &f->entries[i];
        printf("%s\n    dtype: %s\n    shape: [", e->name, dtype_name(e->dtype));
        for (int j = 0; j < e->ndim; ++j)
            printf("%s%" PRId64, j ? ", " : "", e->shape[j]);
        printf("]\n");
        if (total > UINT64_MAX - e->numel) {
            puts("Total Parameters: overflow");
            return;
        }
        total += e->numel;
    }
    printf("Total Parameters: %" PRIu64 "\nTotal Weight Size: %" PRIu64 " bytes\n",
           total,
           f->data_size);
}
