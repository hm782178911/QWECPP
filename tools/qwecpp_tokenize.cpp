#include "runtime/tokenizer.h"
#include "cJSON.h"
#include <iostream>

/**
 * @brief 检查原生 tokenizer 的编码和解码结果。
 * @param[in] argc 命令行参数数量。
 * @param[in] argv 两个位置参数：模型目录、待编码文本。
 * @return 成功为 0，错误为 1；stdout 输出含 ids 和 decoded 的 JSON。
 */
int main(int argc, char **argv) {
    try {
        if (argc != 3)
            return 1;
        qwe::Tokenizer t(argv[1]);
        auto ids = t.encode(argv[2]);
        auto decoded = t.decode(ids);
        cJSON *root = cJSON_CreateObject(), *array = cJSON_AddArrayToObject(root, "ids");
        for (int id : ids)
            cJSON_AddItemToArray(array, cJSON_CreateNumber(id));
        cJSON_AddStringToObject(root, "decoded", decoded.c_str());
        char *s = cJSON_PrintUnformatted(root);
        if (!s) {
            cJSON_Delete(root);
            return 1;
        }
        std::cout << s << "\n";
        cJSON_free(s);
        cJSON_Delete(root);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
