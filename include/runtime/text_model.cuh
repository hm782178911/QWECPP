#pragma once
#include "models/qwen3_5.cuh"
#include <fstream>
#include "cJSON.h"
namespace qwe {
class TextModel {
    std::unique_ptr<Qwen3> q3;
    std::unique_ptr<Qwen3_5> q35;

  public:
    TextModel(const std::string &dir, const Qwen3Options &options) {
        std::ifstream f(dir + "/config.json");
        std::string text((std::istreambuf_iterator<char>(f)), {});
        std::unique_ptr<cJSON, decltype(&cJSON_Delete)> root(cJSON_Parse(text.c_str()),
                                                             cJSON_Delete);
        auto *type = cJSON_GetObjectItemCaseSensitive(root.get(), "model_type");
        if (cJSON_IsString(type) && std::string(type->valuestring) == "qwen3_5")
            q35 = std::make_unique<Qwen3_5>(dir, options);
        else
            q3 = std::make_unique<Qwen3>(dir, options);
    }
    HostTensor forward(const std::vector<int> &ids, bool last = false) {
        return q35 ? q35->forward(ids, last) : q3->forward(ids, last);
    }
    bool supports_vision() const {
        return (bool)q35;
    }
    ImageFeatures encode_image(const ImageInput &image) {
        if (!q35)
            throw std::runtime_error("model does not support images");
        return q35->encode_image(image);
    }
    HostTensor forward_multimodal(const std::vector<int> &tokens,
                                  const std::vector<ImageFeatures> &images,
                                  bool last = true) {
        if (!q35)
            throw std::runtime_error("model does not support images");
        return q35->forward_multimodal(tokens, images, last);
    }
    int context_limit() const {
        return q35 ? q35->context_limit() : q3->context_limit();
    }
    void reset_cache() {
        if (q3)
            q3->reset_cache();
    }
    size_t kv_cache_bytes() const {
        return q3 ? q3->kv_cache_bytes() : 0;
    }
    TransferStats last_transfer_stats() const {
        return q3 ? q3->last_transfer_stats() : TransferStats{};
    }
    TransferStats last_kv_transfer_stats() const {
        return q3 ? q3->last_kv_transfer_stats() : TransferStats{};
    }
};
} // namespace qwe
