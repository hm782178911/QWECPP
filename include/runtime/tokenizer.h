#pragma once
#include <memory>
#include <string>
#include <vector>
namespace qwe {
class Tokenizer {
    struct Impl;
    std::unique_ptr<Impl> impl;

  public:

    /**
     * @brief 加载 Qwen3 分词配置、词表、BPE 合并顺序和 Unicode 正则。
     * @param[in] directory 包含 tokenizer.json 的模型目录。
     * @return 构造原生 tokenizer；文件或配置不支持时抛异常。
     */
    explicit Tokenizer(const std::string &directory);

    /**
     * @brief 释放词表、合并表和正则编译资源。
     * @return 无。
     */
    ~Tokenizer();

    /**
     * @brief 把 UTF-8 文本编码为模型 token ID。
     * @param[in] text 输入文本，可包含 added token；普通文本先做 NFC 归一化。
     * @return 按顺序排列的 token ID；空文本返回空列表。
     * @note 不自动添加 BOS/EOS 或聊天模板；文本非法时抛异常。
     */
    std::vector<int> encode(const std::string &text) const;

    /**
     * @brief 把 token ID 序列解码为原始文本字节。
     * @param[in] ids 词表或 added token 中存在的 ID 序列。
     * @param[in] skip_special true 跳过标记为 special 的 token，其他 added token 仍保留。
     * @return 拼接后的字节字符串；未知 ID 抛异常。
     * @note 单个 token 可能含不完整 UTF-8；流式显示需由调用方暂存半个字符。
     */
    std::string decode(const std::vector<int> &ids, bool skip_special = false) const;
};
} // namespace qwe
