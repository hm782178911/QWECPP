#include "loader/safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/**
 * @brief 读取并打印一个模型权重文件的元数据。
 * @param[in] argc 命令行参数数量。
 * @param[in] argv --model FILE_OR_DIR；目录下只查找 model.safetensors。
 * @return 成功为 0，错误为 1；检查结果写入 stdout，错误写入 stderr。
 */
int main(int argc, char **argv) {
    if (argc != 3 || strcmp(argv[1], "--model") != 0) {
        fprintf(stderr, "Usage: %s --model <file.safetensors | model-directory>\n", argv[0]);
        return 1;
    }
    const char *path = argv[2];
    char *joined = NULL;
    struct stat st;
    if (stat(path, &st) == 0 && S_ISDIR(st.st_mode)) {
        size_t n = strlen(path);
        joined = malloc(n + 32);
        if (!joined)
            return 1;
        snprintf(joined, n + 32, "%s/model.safetensors", path);
        path = joined;
    }
    SafeTensorFile file = {0};
    if (safetensors_open(&file, path) != 0) {
        fprintf(stderr, "%s: %s (single-file checkpoints only)\n", path, file.error);
        free(joined);
        return 1;
    }
    printf("Model: %s\n", argv[2]);
    safetensors_inspect(&file);
    safetensors_close(&file);
    free(joined);
    return 0;
}
