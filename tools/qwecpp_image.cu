#include "runtime/text_model.cuh"
#include "runtime/tokenizer.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
static void write(const std::string &path, const float *p, size_t count) {
    if (path.empty())
        return;
    std::ofstream f(path, std::ios::binary);
    f.write((const char *)p, count * sizeof(float));
    if (!f)
        throw std::runtime_error("cannot write output");
}
int main(int argc, char **argv) {
    try {
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 >= argc)
                throw std::runtime_error("missing value");
            std::string key = argv[i];
            if (key != "--model" && key != "--image" && key != "--image2" && key != "--prompt" &&
                key != "--patches" && key != "--features" && key != "--output" &&
                key != "--metadata")
                throw std::runtime_error("unknown image option");
            args[key] = argv[i + 1];
        }
        if (args["--model"].empty() || args["--image"].empty())
            throw std::runtime_error("--model and --image required");
        std::vector<qwe::ImageInput> inputs;
        for (auto key : {"--image", "--image2"})
            if (!args[key].empty()) {
                if (std::filesystem::file_size(args[key]) > 8 * 1024 * 1024)
                    throw std::runtime_error("image exceeds 8MiB");
                std::ifstream f(args[key], std::ios::binary);
                std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), {});
                inputs.push_back(qwe::preprocess_image(bytes));
            }
        qwe::Tokenizer tokenizer(args["--model"]);
        qwe::TextModel model(args["--model"], {});
        std::vector<qwe::ImageFeatures> features;
        int patches = 0;
        std::string prompt = "<|im_start|>user\n";
        for (auto &input : inputs) {
            features.push_back(model.encode_image(input));
            patches += input.grid_h * input.grid_w;
            prompt += "<|vision_start|>";
            for (int i = 0; i < input.grid_h * input.grid_w / 4; ++i)
                prompt += "<|image_pad|>";
            prompt += "<|vision_end|>";
        }
        prompt += args["--prompt"] + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        auto ids = tokenizer.encode(prompt);
        auto logits = model.forward_multimodal(ids, features, true);
        write(args["--output"], logits.data(), logits.size());
        if (!args["--patches"].empty()) {
            qwe::HostTensor host({patches, 1536});
            size_t offset = 0;
            for (auto &input : inputs) {
                std::copy_n(input.patches.data(), input.patches.size(), host.data() + offset);
                offset += input.patches.size();
            }
            write(args["--patches"], host.data(), host.size());
        }
        if (!args["--features"].empty()) {
            qwe::HostTensor host({patches / 4, 1024});
            size_t offset = 0;
            for (auto &feature : features) {
                auto &tensor = feature.embeddings->tensor;
                qwe::check(cudaMemcpy(
                    host.data() + offset, tensor.data, tensor.nbytes, cudaMemcpyDeviceToHost));
                offset += tensor.numel;
            }
            write(args["--features"], host.data(), host.size());
        }
        if (!args["--metadata"].empty()) {
            std::ofstream out(args["--metadata"]);
            out << "{\"grid_h\":" << inputs[0].grid_h << ",\"grid_w\":" << inputs[0].grid_w
                << ",\"grids\":[";
            for (size_t i = 0; i < inputs.size(); ++i) {
                if (i)
                    out << ",";
                out << "[1," << inputs[i].grid_h << "," << inputs[i].grid_w << "]";
            }
            out << "],\"tokens\":[";
            for (size_t i = 0; i < ids.size(); ++i) {
                if (i)
                    out << ",";
                out << ids[i];
            }
            out << "]}";
            if (!out)
                throw std::runtime_error("cannot write metadata");
        }
        std::cout << "images=" << inputs.size() << " visual_tokens=" << patches / 4
                  << " next_token="
                  << std::max_element(logits.begin(), logits.end()) - logits.begin() << "\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
