#include "models/qwen3_5_vision.cuh"
#include "loader/device_weights.cuh"
#include "operators/vision_ops.cuh"
#include "backend/cuda/device_guard.cuh"
#include <fstream>
#include <iterator>
namespace qwe {
struct Qwen3_5Vision::Impl {
    WeightStore weights;
    int device;
    cublasHandle_t blas = nullptr;
    const Tensor *patch, *patch_bias, *positions;
    struct Layer {
        const Tensor *n1, *b1, *n2, *b2, *qkv, *qb, *proj, *pb, *fc1, *f1b, *fc2, *f2b;
    };
    std::vector<Layer> layers;
    const Tensor *mn, *mb, *mf1, *mf1b, *mf2, *mf2b;
    explicit Impl(const std::string &dir) : weights(dir) {
        check(cudaGetDevice(&device));
        std::ifstream f(dir + "/config.json");
        std::string text((std::istreambuf_iterator<char>(f)), {});
        std::unique_ptr<cJSON, decltype(&cJSON_Delete)> root(cJSON_Parse(text.c_str()),
                                                             cJSON_Delete);
        auto *c = cJSON_GetObjectItemCaseSensitive(root.get(), "vision_config");
        for (auto expected : std::initializer_list<std::pair<const char *, int>>{
                 {"hidden_size", 768},
                 {"depth", 12},
                 {"intermediate_size", 3072},
                 {"num_heads", 12},
                 {"patch_size", 16},
                 {"temporal_patch_size", 2},
                 {"spatial_merge_size", 2},
                 {"in_channels", 3},
                 {"out_hidden_size", 1024},
                 {"num_position_embeddings", 2304}}) {
            auto *v = cJSON_GetObjectItemCaseSensitive(c, expected.first);
            if (!cJSON_IsNumber(v) || v->valuedouble != expected.second)
                throw std::runtime_error("unsupported vision config");
        }
        auto *act = cJSON_GetObjectItemCaseSensitive(c, "hidden_act"),
             *deep = cJSON_GetObjectItemCaseSensitive(c, "deepstack_visual_indexes");
        if (!cJSON_IsString(act) || std::string(act->valuestring) != "gelu_pytorch_tanh" ||
            !cJSON_IsArray(deep) || cJSON_GetArraySize(deep))
            throw std::runtime_error("unsupported vision activation/deepstack");
        check(cublasCreate(&blas));
        try {
            check(cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH));
            std::string base = "model.visual.";
            patch =
                weights.load(base + "patch_embed.proj.weight", {768, 1536}, {768, 3, 2, 16, 16});
            patch_bias = weights.load(base + "patch_embed.proj.bias", {768});
            positions = weights.load(base + "pos_embed.weight", {2304, 768});
            for (int i = 0; i < 12; ++i) {
                std::string b = base + "blocks." + std::to_string(i) + ".";
                layers.push_back({weights.load(b + "norm1.weight", {768}),
                                  weights.load(b + "norm1.bias", {768}),
                                  weights.load(b + "norm2.weight", {768}),
                                  weights.load(b + "norm2.bias", {768}),
                                  weights.load(b + "attn.qkv.weight", {2304, 768}),
                                  weights.load(b + "attn.qkv.bias", {2304}),
                                  weights.load(b + "attn.proj.weight", {768, 768}),
                                  weights.load(b + "attn.proj.bias", {768}),
                                  weights.load(b + "mlp.linear_fc1.weight", {3072, 768}),
                                  weights.load(b + "mlp.linear_fc1.bias", {3072}),
                                  weights.load(b + "mlp.linear_fc2.weight", {768, 3072}),
                                  weights.load(b + "mlp.linear_fc2.bias", {768})});
            }
            mn = weights.load(base + "merger.norm.weight", {768});
            mb = weights.load(base + "merger.norm.bias", {768});
            mf1 = weights.load(base + "merger.linear_fc1.weight", {3072, 3072});
            mf1b = weights.load(base + "merger.linear_fc1.bias", {3072});
            mf2 = weights.load(base + "merger.linear_fc2.weight", {1024, 3072});
            mf2b = weights.load(base + "merger.linear_fc2.bias", {1024});
        } catch (...) {
            cublasDestroy(blas);
            blas = nullptr;
            throw;
        }
    }
    ~Impl() {
        if (blas) {
            DeviceGuard guard(device);
            cublasDestroy(blas);
        }
    }
    ImageFeatures encode(const ImageInput &image) {
        DeviceGuard guard(device);
        int n = image.grid_h * image.grid_w;
        if (n < 4 || n > 1024 || image.grid_h % 2 || image.grid_w % 2 ||
            image.patches.tensor.shape[0] != n || image.patches.tensor.shape[1] != 1536)
            throw std::runtime_error("invalid image patch grid");
        DeviceTensor pixels(DTYPE_FP32, {n, 1536}), x(DTYPE_FP32, {n, 768}),
            normed(DTYPE_FP32, {n, 768}), qkv(DTYPE_FP32, {n, 2304}), attn(DTYPE_FP32, {n, 768}),
            proj(DTYPE_FP32, {n, 768}), mlp(DTYPE_FP32, {n, 3072});
        check(cudaMemcpy(pixels.tensor.data,
                         image.patches.data(),
                         pixels.tensor.nbytes,
                         cudaMemcpyHostToDevice));
        linear(blas, x.tensor, pixels.tensor, *patch);
        vision_position(x.tensor, *patch_bias, *positions, image.grid_h, image.grid_w);
        for (auto &l : layers) {
            layer_norm(normed.tensor, x.tensor, *l.n1, *l.b1);
            linear(blas, qkv.tensor, normed.tensor, *l.qkv);
            bias_rows(qkv.tensor, *l.qb);
            vision_rope(qkv.tensor, image.grid_h, image.grid_w);
            vision_attention(attn.tensor, qkv.tensor);
            linear(blas, proj.tensor, attn.tensor, *l.proj);
            bias_rows(proj.tensor, *l.pb);
            add(x.tensor, proj.tensor);
            layer_norm(normed.tensor, x.tensor, *l.n2, *l.b2);
            linear(blas, mlp.tensor, normed.tensor, *l.fc1);
            bias_rows(mlp.tensor, *l.f1b);
            gelu(mlp.tensor, true);
            linear(blas, proj.tensor, mlp.tensor, *l.fc2);
            bias_rows(proj.tensor, *l.f2b);
            add(x.tensor, proj.tensor);
        }
        layer_norm(normed.tensor, x.tensor, *mn, *mb);
        auto merged = reshape(normed.tensor, {n / 4, 3072});
        DeviceTensor mid(DTYPE_FP32, {n / 4, 3072});
        auto result =
            std::make_unique<DeviceTensor>(DTYPE_FP32, std::initializer_list<int64_t>{n / 4, 1024});
        linear(blas, mid.tensor, merged, *mf1);
        bias_rows(mid.tensor, *mf1b);
        gelu(mid.tensor, false);
        linear(blas, result->tensor, mid.tensor, *mf2);
        bias_rows(result->tensor, *mf2b);
        return {image.grid_h, image.grid_w, std::move(result)};
    }
};
Qwen3_5Vision::Qwen3_5Vision(const std::string &dir) : impl(std::make_unique<Impl>(dir)) {}
Qwen3_5Vision::~Qwen3_5Vision() = default;
ImageFeatures Qwen3_5Vision::encode(const ImageInput &input) {
    return impl->encode(input);
}
} // namespace qwe
