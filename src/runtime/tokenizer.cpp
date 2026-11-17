#include "runtime/tokenizer.h"
#include "cJSON.h"
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <climits>
#include <cmath>
#include <unicode/unorm2.h>
#include <unicode/ustring.h>
namespace qwe {
/**
 * @brief 把字节字母表使用的码点编码为 UTF-8。
 * @param[in] c 当前实现使用的 Unicode BMP 码点。
 * @return 该码点的 UTF-8 字节串。
 */
static std::string utf8(unsigned c) {
    if (c < 128)
        return std::string(1, (char)c);
    if (c < 2048)
        return {char(0xc0 | (c >> 6)), char(0x80 | (c & 63))};
    return {char(0xe0 | (c >> 12)), char(0x80 | ((c >> 6) & 63)), char(0x80 | (c & 63))};
}

/**
 * @brief 将 UTF-8 文本转换为 NFC 规范形式。
 * @param[in] text 待规范化的 UTF-8 文本。
 * @return 规范化后的 UTF-8 文本；编码或 ICU 操作失败时抛异常。
 */
static std::string nfc(const std::string &text) {
    if (text.size() > INT_MAX)
        throw std::runtime_error("input too large");
    UErrorCode status = U_ZERO_ERROR;
    int32_t length = 0;
    u_strFromUTF8(nullptr, 0, &length, text.data(), (int32_t)text.size(), &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status))
        throw std::runtime_error("invalid UTF-8 input");
    status = U_ZERO_ERROR;
    std::vector<UChar> input(length + 1);
    u_strFromUTF8(input.data(), input.size(), &length, text.data(), (int32_t)text.size(), &status);
    if (U_FAILURE(status))
        throw std::runtime_error("UTF-8 conversion failed");
    const UNormalizer2 *normalizer = unorm2_getNFCInstance(&status);
    int32_t needed = unorm2_normalize(normalizer, input.data(), length, nullptr, 0, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status))
        throw std::runtime_error("NFC normalization failed");
    status = U_ZERO_ERROR;
    std::vector<UChar> normalized(needed + 1);
    needed = unorm2_normalize(
        normalizer, input.data(), length, normalized.data(), normalized.size(), &status);
    if (U_FAILURE(status))
        throw std::runtime_error("NFC normalization failed");
    int32_t bytes = 0;
    u_strToUTF8(nullptr, 0, &bytes, normalized.data(), needed, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status))
        throw std::runtime_error("UTF-8 conversion failed");
    status = U_ZERO_ERROR;
    std::string out(bytes, '\0');
    u_strToUTF8(out.data(), bytes, nullptr, normalized.data(), needed, &status);
    if (U_FAILURE(status))
        throw std::runtime_error("UTF-8 conversion failed");
    return out;
}

/**
 * @brief 构造一个可用于查找 BPE 合并 rank 的键。
 * @param[in] a 左侧字节级 token 字符串。
 * @param[in] b 右侧字节级 token 字符串。
 * @return 由 NUL 分隔的组合键，仅内部查表使用。
 */
static std::string pair_key(const std::string &a, const std::string &b) {
    return a + std::string(1, '\0') + b;
}

struct Tokenizer::Impl {
    std::array<std::string, 256> alphabet;
    std::unordered_map<std::string, unsigned char> bytes;
    std::unordered_map<std::string, int> vocab, ranks;
    std::unordered_map<int, std::string> inverse, added_ids;
    std::unordered_map<std::string, int> added;
    std::unordered_set<int> special;
    pcre2_code *regex = nullptr;

    /**
     * @brief 释放 tokenizer 内部持有的已编译正则。
     * @return 无；其他容器由各自析构函数释放。
     */
    ~Impl() {
        if (regex)
            pcre2_code_free(regex);
    }

    /**
     * @brief 对一个正则切片执行字节级 BPE。
     * @param[in] text 已归一化的普通文本切片。
     * @param[in,out] out 在末尾追加编码 ID，不清空已有内容。
     * @return 无；编码结果追加到 out。
     */
    void piece(const std::string &text, std::vector<int> &out) const {
        std::vector<std::string> parts;
        for (unsigned char b : text)
            parts.push_back(alphabet[b]);
        while (parts.size() > 1) {
            int best = INT_MAX;
            size_t where = 0;
            for (size_t i = 0; i + 1 < parts.size(); ++i) {
                auto r = ranks.find(pair_key(parts[i], parts[i + 1]));
                if (r != ranks.end() && r->second < best) {
                    best = r->second;
                    where = i;
                }
            }
            if (best == INT_MAX)
                break;
            parts[where] += parts[where + 1];
            parts.erase(parts.begin() + where + 1);
        }
        for (const auto &s : parts) {
            auto it = vocab.find(s);
            if (it == vocab.end())
                throw std::runtime_error("BPE token absent from vocabulary");
            out.push_back(it->second);
        }
    }

    /**
     * @brief 对不含 added token 的文本归一化、切词并编码。
     * @param[in] raw 待处理的 UTF-8 普通文本。
     * @param[in,out] out 在末尾追加 token ID。
     * @return 无；文本或正则匹配非法时抛异常。
     */
    void ordinary(const std::string &raw, std::vector<int> &out) const {
        std::string text = nfc(raw);
        std::unique_ptr<pcre2_match_data, decltype(&pcre2_match_data_free)> match(
            pcre2_match_data_create_from_pattern(regex, nullptr), pcre2_match_data_free);
        if (!match)
            throw std::runtime_error("regex allocation failed");
        size_t pos = 0;
        while (pos < text.size()) {
            int rc = pcre2_match(regex,
                                 (PCRE2_SPTR)text.data(),
                                 text.size(),
                                 pos,
                                 PCRE2_ANCHORED,
                                 match.get(),
                                 nullptr);
            if (rc < 0)
                throw std::runtime_error("invalid UTF-8 or tokenizer regex match failed");
            auto range = pcre2_get_ovector_pointer(match.get());
            if (range[0] != pos || range[1] <= pos)
                throw std::runtime_error("tokenizer regex gap");
            piece(text.substr(pos, range[1] - pos), out);
            pos = range[1];
        }
    }
};

/**
 * @brief 加载 Qwen3 分词配置、词表、BPE 合并顺序和 Unicode 正则。
 * @param[in] directory 包含 tokenizer.json 的模型目录。
 * @return 构造原生 tokenizer；文件或配置不支持时抛异常。
 */
Tokenizer::Tokenizer(const std::string &directory) : impl(std::make_unique<Impl>()) {
    std::ifstream f(directory + "/tokenizer.json");
    if (!f)
        throw std::runtime_error("cannot read tokenizer.json");
    std::string text((std::istreambuf_iterator<char>(f)), {});
    std::unique_ptr<cJSON, decltype(&cJSON_Delete)> root(cJSON_Parse(text.c_str()), cJSON_Delete);
    auto field = [](const cJSON *o, const char *k) {
        return cJSON_GetObjectItemCaseSensitive(o, k);
    };
    auto normalizer = field(field(root.get(), "normalizer"), "type");
    if (!cJSON_IsString(normalizer) || std::string(normalizer->valuestring) != "NFC")
        throw std::runtime_error("expected NFC normalizer");
    auto model = field(root.get(), "model"), type = field(model, "type");
    if (!cJSON_IsString(type) || std::string(type->valuestring) != "BPE")
        throw std::runtime_error("only BPE tokenizer supported");
    auto id = [](const cJSON *v) {
        if (!cJSON_IsNumber(v) || v->valuedouble < 0 || v->valuedouble > INT_MAX ||
            std::floor(v->valuedouble) != v->valuedouble)
            throw std::runtime_error("invalid tokenizer ID");
        return (int)v->valuedouble;
    };
    auto vocab = field(model, "vocab"), merges = field(model, "merges");
    if (!cJSON_IsObject(vocab) || !cJSON_IsArray(merges))
        throw std::runtime_error("missing BPE vocabulary/merges");
    for (auto v = vocab->child; v; v = v->next) {
        int n = id(v);
        impl->vocab.emplace(v->string, n);
        impl->inverse.emplace(n, v->string);
    }
    int rank = 0;
    for (auto m = merges->child; m; m = m->next) {
        std::string a, b;
        if (cJSON_IsArray(m) && cJSON_GetArraySize(m) == 2) {
            auto x = cJSON_GetArrayItem(m, 0), y = cJSON_GetArrayItem(m, 1);
            if (!cJSON_IsString(x) || !cJSON_IsString(y))
                throw std::runtime_error("invalid merge pair");
            a = x->valuestring;
            b = y->valuestring;
        } else if (cJSON_IsString(m)) {
            std::string pair = m->valuestring;
            auto split = pair.find(' ');
            if (split == std::string::npos)
                throw std::runtime_error("invalid merge string");
            a = pair.substr(0, split);
            b = pair.substr(split + 1);
        } else
            throw std::runtime_error("invalid BPE merge");
        impl->ranks.emplace(pair_key(a, b), rank++);
    }

    // GPT-2/Qwen 字节字母表：可见字节直接映射，其余映射到 256 起的码点。
    unsigned next = 256;
    for (unsigned b = 0; b < 256; ++b) {
        bool direct = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174);
        auto s = utf8(direct ? b : next++);
        impl->alphabet[b] = s;
        impl->bytes.emplace(s, (unsigned char)b);
    }
    auto added = field(root.get(), "added_tokens");
    if (!cJSON_IsArray(added))
        throw std::runtime_error("missing added tokens");
    for (auto a = added->child; a; a = a->next) {
        auto content = field(a, "content");
        if (!cJSON_IsString(content) || !*content->valuestring)
            throw std::runtime_error("invalid added token");
        for (const char *flag : {"single_word", "lstrip", "rstrip"})
            if (cJSON_IsTrue(field(a, flag)))
                throw std::runtime_error("unsupported added-token matching flag");
        int n = id(field(a, "id"));
        impl->added.emplace(content->valuestring, n);
        impl->added_ids.emplace(n, content->valuestring);
        if (cJSON_IsTrue(field(a, "special")))
            impl->special.insert(n);
    }
    auto pre = field(root.get(), "pre_tokenizer"), seq = field(pre, "pretokenizers");
    auto split = cJSON_GetArrayItem(seq, 0), byte = cJSON_GetArrayItem(seq, 1);
    auto st = field(split, "type"), bt = field(byte, "type");
    auto pattern = field(field(split, "pattern"), "Regex");
    if (!cJSON_IsString(st) || std::string(st->valuestring) != "Split" || !cJSON_IsString(bt) ||
        std::string(bt->valuestring) != "ByteLevel" || !cJSON_IsString(pattern) ||
        cJSON_IsTrue(field(byte, "add_prefix_space")) || cJSON_IsTrue(field(byte, "use_regex")))
        throw std::runtime_error("unsupported pre-tokenizer (expected Qwen3 Split + ByteLevel)");
    int error;
    PCRE2_SIZE offset;
    impl->regex = pcre2_compile((PCRE2_SPTR)pattern->valuestring,
                                PCRE2_ZERO_TERMINATED,
                                PCRE2_UTF | PCRE2_UCP,
                                &error,
                                &offset,
                                nullptr);
    if (!impl->regex)
        throw std::runtime_error("cannot compile tokenizer regex at byte " +
                                 std::to_string(offset));
}

/**
 * @brief 释放词表、合并表和正则编译资源。
 * @return 无。
 */
Tokenizer::~Tokenizer() = default;

/**
 * @brief 把 UTF-8 文本编码为模型 token ID。
 * @param[in] text 输入文本，可包含 added token；普通文本先做 NFC 归一化。
 * @return 按顺序排列的 token ID；空文本返回空列表。
 * @note 不自动添加 BOS/EOS 或聊天模板；文本非法时抛异常。
 */
std::vector<int> Tokenizer::encode(const std::string &text) const {
    std::vector<int> out;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t first = std::string::npos;
        const std::string *chosen = nullptr;
        int id = -1;
        for (const auto &a : impl->added) {
            auto at = text.find(a.first, pos);
            if (at < first || (at == first && at != std::string::npos && chosen &&
                               a.first.size() > chosen->size())) {
                first = at;
                chosen = &a.first;
                id = a.second;
            }
        }
        if (first == std::string::npos) {
            impl->ordinary(text.substr(pos), out);
            break;
        }
        impl->ordinary(text.substr(pos, first - pos), out);
        out.push_back(id);
        pos = first + chosen->size();
    }
    return out;
}

/**
 * @brief 把 token ID 序列解码为原始文本字节。
 * @param[in] ids 词表或 added token 中存在的 ID 序列。
 * @param[in] skip true 跳过标记为 special 的 token，其他 added token 仍保留。
 * @return 拼接后的字节字符串；未知 ID 抛异常。
 * @note 单个 token 可能含不完整 UTF-8；流式显示需由调用方暂存半个字符。
 */
std::string Tokenizer::decode(const std::vector<int> &ids, bool skip) const {
    std::string result;
    for (int id : ids) {
        if (skip && impl->special.count(id))
            continue;
        auto a = impl->added_ids.find(id);
        if (a != impl->added_ids.end()) {
            result += a->second;
            continue;
        }
        auto v = impl->inverse.find(id);
        if (v == impl->inverse.end())
            throw std::runtime_error("unknown token ID");
        const std::string &s = v->second;
        for (size_t i = 0; i < s.size();) {
            unsigned char c = s[i];
            size_t length = c < 128 ? 1 : c < 224 ? 2 : c < 240 ? 3 : 4;
            auto b = impl->bytes.find(s.substr(i, length));
            if (b == impl->bytes.end())
                throw std::runtime_error("invalid byte-level token");
            result += (char)b->second;
            i += length;
        }
    }
    return result;
}
} // namespace qwe
