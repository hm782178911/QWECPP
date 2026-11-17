#pragma once
#include <string>
#include <vector>
#include <stdexcept>
namespace qwe {
/**
 * @brief 将命令行设备列表解析为设备编号。
 * @param[in] text 例如 "0" 或 "0,1"，不允许空段、负数或重复编号。
 * @return 一个或两个设备编号；非法输入抛异常，设备存在性由模型初始化检查。
 */
inline std::vector<int> parse_devices(const std::string &text) {
    std::vector<int> result;
    size_t start = 0;
    do {
        auto end = text.find(',', start);
        auto part = text.substr(start, end == std::string::npos ? end : end - start);
        if (part.empty() || part.find_first_not_of("0123456789") != std::string::npos)
            throw std::runtime_error("invalid device list");
        size_t used;
        int id = std::stoi(part, &used);
        if (used != part.size() || result.size() >= 2 || (!result.empty() && result[0] == id))
            throw std::runtime_error("expected one or two distinct devices");
        result.push_back(id);
        if (end == std::string::npos)
            break;
        start = end + 1;
    } while (true);
    return result;
}
} // namespace qwe
