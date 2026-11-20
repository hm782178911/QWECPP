#ifndef QWECPP_SAFETENSORS_H
#define QWECPP_SAFETENSORS_H
#include "core/dtype.h"
#include <stdint.h>
typedef struct {
    char *name;
    DType dtype;
    int ndim;
    // 文件元数据最多8维；计算Tensor仍最多4维。
    int64_t shape[8];
    uint64_t data_offset, data_size, numel;
} SafeTensorEntry;
typedef struct {
    void *mapped_data;
    size_t file_size;
    uint64_t data_start, data_size;
    SafeTensorEntry *entries;
    size_t num_entries;
    char error[256];
} SafeTensorFile;

/**
 * @brief 映射并校验一个 Safetensors 文件。
 * @param[in,out] file 以 {0} 初始化的对象；复用前先 close，失败原因写入 error。
 * @param[in] path 单个权重文件路径，不接受目录。
 * @return 成功为 0，失败为 -1。
 * @note 校验头部、dtype/shape、偏移、字节数及数据区间；不把整份权重复制到内存。
 */
int safetensors_open(SafeTensorFile *file, const char *path);

/**
 * @brief 解除 mmap 并释放条目名称和元数据。
 * @param[in,out] file 要关闭的对象，可为 NULL；关闭后清零。
 * @return 无。
 * @note 不释放调用方另行复制的数据；借用的 Entry/data 指针全部失效。
 */
void safetensors_close(SafeTensorFile *file);

/**
 * @brief 查询文件中的权重张量数量。
 * @param[in] file 已打开对象，也允许 NULL。
 * @return 张量数；NULL 返回 0。
 */
size_t safetensors_count(const SafeTensorFile *file);

/**
 * @brief 按名称查找文件中的张量元数据。
 * @param[in] file 已打开对象。
 * @param[in] name 大小写敏感的完整张量名称。
 * @return 借用的 Entry 指针；未找到或参数为 NULL 时返回 NULL。
 * @note 返回值由 file 所有，调用方不得释放。
 */
const SafeTensorEntry *safetensors_find(const SafeTensorFile *file, const char *name);

/**
 * @brief 取得某个 Entry 对应的映射数据地址。
 * @param[in] file 已打开且仍有效的文件映射。
 * @param[in] entry 必须是 file 自身持有的 Entry。
 * @return 只读借用指针；参数不合法或 Entry 不属于该文件时返回 NULL。
 * @note 不做 dtype 转换；数据有效期截至文件 close。
 */
const void *safetensors_get_data(const SafeTensorFile *file, const SafeTensorEntry *entry);

/**
 * @brief 向标准输出打印张量名称、类型、形状和汇总。
 * @param[in] file 已校验的文件对象，NULL 时不输出。
 * @return 无；输出写入 stdout。
 */
void safetensors_inspect(const SafeTensorFile *file);
#endif
