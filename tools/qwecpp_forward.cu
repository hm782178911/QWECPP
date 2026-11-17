#include "models/qwen3.cuh"
#include "runtime/device_options.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

/**
 * @brief 执行一次 token ID 到 logits 的前向计算。
 * @param[in] argc 命令行参数数量。
 * @param[in] argv --model DIR --tokens ID,ID [--output FILE]。
 * @return 成功为 0，错误为 1；stdout 显示 top-5，可选写出全位置 FP32 logits。
 */
int main(int argc, char **argv) {
    try {
        qwe::Qwen3Options options;
        std::string model, tokens, output;
        for (int i = 1; i < argc; ++i) {
            std::string key = argv[i];
            if (key == "--help") {
                std::cout
                    << "Usage: qwecpp_forward --model DIR --tokens ID,ID [--output logits.bin] "
                       "[--devices 0,1] [--max-context 2048] [--parallel pp|tp] [--split-layer 14] [--pp-host-transfer]\n";
                return 0;
            }
            if (key == "--pp-host-transfer") {
                options.force_host_transfer = true;
                continue;
            }
            if (i + 1 >= argc)
                throw std::runtime_error("missing option value");
            if (key == "--max-context") {
                std::string value=argv[++i]; size_t used;
                options.max_context=std::stoi(value,&used);
                if(used!=value.size() || options.max_context<1 || options.max_context>8192)
                    throw std::runtime_error("max-context must be 1..8192");
            } else if (key == "--parallel") {
                std::string mode = argv[++i];
                if (mode != "tp" && mode != "pp") throw std::runtime_error("parallel must be tp or pp");
                options.tensor_parallel = mode == "tp";
            } else if (key == "--devices")
                options.devices = qwe::parse_devices(argv[++i]);
            else if (key == "--split-layer") {
                std::string value = argv[++i];
                size_t used;
                options.split_layer = std::stoi(value, &used);
                if (used != value.size() || options.split_layer < 1)
                    throw std::runtime_error("split-layer must be positive");
            } else if (key == "--model")
                model = argv[++i];
            else if (key == "--tokens")
                tokens = argv[++i];
            else if (key == "--output")
                output = argv[++i];
            else
                throw std::runtime_error("unknown option: " + key);
        }
        if (model.empty() || tokens.empty())
            throw std::runtime_error("--model and --tokens required");
        std::vector<int> ids;
        size_t start = 0;
        do {
            auto end = tokens.find(',', start);
            std::string part = tokens.substr(start, end == std::string::npos ? end : end - start);
            if (part.empty() || part.find_first_not_of("0123456789") != std::string::npos)
                throw std::runtime_error("invalid token list");
            size_t used = 0;
            int id = std::stoi(part, &used);
            if (used != part.size())
                throw std::runtime_error("invalid token ID");
            ids.push_back(id);
            if (ids.size() > (size_t)options.max_context)
                throw std::runtime_error("token count exceeds max-context");
            if (end == std::string::npos)
                break;
            start = end + 1;
        } while (true);
        qwe::Qwen3 runtime(model, options);
        auto logits = runtime.forward(ids);
        size_t vocab = logits.size() / ids.size();
        if (!output.empty()) {
            std::ofstream file(output, std::ios::binary);
            file.write(reinterpret_cast<const char *>(logits.data()),
                       logits.size() * sizeof(float));
            file.close();
            if (!file)
                throw std::runtime_error("cannot write logits output");
        }
        std::vector<int> order(vocab);
        std::iota(order.begin(), order.end(), 0);
        auto last = logits.data() + (ids.size() - 1) * vocab;
        size_t count = std::min<size_t>(5, vocab);
        std::partial_sort(order.begin(), order.begin() + count, order.end(), [&](int a, int b) {
            return last[a] > last[b];
        });
        std::cout << "Logits shape: [" << ids.size() << ", " << vocab << "]\nLast token top-"
                  << count << ":\n";
        for (size_t i = 0; i < count; ++i)
            std::cout << order[i] << " " << last[order[i]] << "\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "forward: " << e.what() << "\n";
        return 1;
    }
}
