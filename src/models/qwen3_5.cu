#include "models/qwen3_5.cuh"
#include "loader/device_weights.cuh"
#include "operators/qwen35.cuh"
#include "operators/vision_ops.cuh"
#include "backend/cuda/device_guard.cuh"
#include <fstream>
#include <cmath>
#include <iterator>
namespace qwe {
struct Qwen3_5::Impl {
    int device, limit, h, f, heads, kv, d, lh, ld, layers, conv, rotary, vocab;
    float eps, theta;
    std::string directory;
    std::unique_ptr<Qwen3_5Vision> vision;
    int image_token_id = 0;
    WeightStore weights;
    cublasHandle_t blas = nullptr;
    const Tensor *emb, *final_norm;
    struct Layer {
        bool linear;
        const Tensor *pre, *post, *gate, *up, *down, *q, *k, *v, *o, *qn, *kn, *z, *a, *b, *A, *dt,
            *cw, *ln;
    };
    std::vector<Layer> blocks;
    // 普通Norm权重采用1+w；加载后转换一次，门控Norm使用原权重。
    const Tensor *loadnorm(const std::string &name, int dim) {
        auto *w = weights.load(name, {dim});
        DeviceTensor ones(DTYPE_FP32, {dim});
        std::vector<float> v(dim, 1.f);
        check(cudaMemcpy(ones.tensor.data, v.data(), dim * 4, cudaMemcpyHostToDevice));
        auto t = *w;
        add(t, ones.tensor);
        return w;
    }
    Impl(const std::string &dir, const Qwen3Options &opt) : directory(dir), weights(dir) {
        if (opt.devices.size() > 1 || opt.tensor_parallel || opt.prefill_decode ||
            opt.split_layer != -1 || opt.kv_cache)
            throw std::runtime_error(
                "Qwen3.5 first version supports single GPU text only; no KV cache/PP/TP/PD");
        check(cudaGetDevice(&device));
        if (!opt.devices.empty())
            device = opt.devices[0];
        DeviceGuard guard(device);
        std::ifstream in(dir + "/config.json");
        std::string text((std::istreambuf_iterator<char>(in)), {});
        std::unique_ptr<cJSON, decltype(&cJSON_Delete)> root(cJSON_Parse(text.c_str()),
                                                             cJSON_Delete);
        auto *type = cJSON_GetObjectItemCaseSensitive(root.get(), "model_type");
        if (!cJSON_IsString(type) || std::string(type->valuestring) != "qwen3_5")
            throw std::runtime_error("expected Qwen3.5 config");
        auto *image_id = cJSON_GetObjectItemCaseSensitive(root.get(), "image_token_id");
        if (!cJSON_IsNumber(image_id) || image_id->valuedouble < 0 ||
            image_id->valuedouble > 1000000 ||
            floor(image_id->valuedouble) != image_id->valuedouble)
            throw std::runtime_error("invalid image token ID");
        image_token_id = (int)image_id->valuedouble;
        auto *cfg = cJSON_GetObjectItemCaseSensitive(root.get(), "text_config");
        auto require_string = [](const cJSON *object, const char *name, const char *expected) {
            auto *value = cJSON_GetObjectItemCaseSensitive(object, name);
            if (!cJSON_IsString(value) || std::string(value->valuestring) != expected)
                throw std::runtime_error(std::string("unsupported config: ") + name);
        };
        require_string(cfg, "model_type", "qwen3_5_text");
        require_string(cfg, "hidden_act", "silu");
        if (!cJSON_IsTrue(cJSON_GetObjectItemCaseSensitive(root.get(), "tie_word_embeddings")) ||
            !cJSON_IsTrue(cJSON_GetObjectItemCaseSensitive(cfg, "tie_word_embeddings")) ||
            !cJSON_IsTrue(cJSON_GetObjectItemCaseSensitive(cfg, "attn_output_gate")) ||
            !cJSON_IsFalse(cJSON_GetObjectItemCaseSensitive(cfg, "attention_bias")))
            throw std::runtime_error("unsupported Qwen3.5 weight tying/gate/bias");

        auto num = [&](const cJSON *obj, const char *name) {
            auto *v = cJSON_GetObjectItemCaseSensitive(obj, name);
            if (!cJSON_IsNumber(v) || !std::isfinite(v->valuedouble))
                throw std::runtime_error(std::string("invalid config: ") + name);
            return v->valuedouble;
        };
        auto integer = [&](const char *name) {
            double v = num(cfg, name);
            if (v < 1 || v > 1000000 || floor(v) != v)
                throw std::runtime_error("invalid integer dimension");
            return (int)v;
        };
        h = integer("hidden_size");
        f = integer("intermediate_size");
        heads = integer("num_attention_heads");
        kv = integer("num_key_value_heads");
        d = integer("head_dim");
        lh = integer("linear_num_key_heads");
        ld = integer("linear_key_head_dim");
        layers = integer("num_hidden_layers");
        conv = integer("linear_conv_kernel_dim");
        vocab = integer("vocab_size");
        eps = num(cfg, "rms_norm_eps");
        auto *rp = cJSON_GetObjectItemCaseSensitive(cfg, "rope_parameters");
        require_string(rp, "rope_type", "default");
        auto *sections = cJSON_GetObjectItemCaseSensitive(rp, "mrope_section");
        if (!cJSON_IsTrue(cJSON_GetObjectItemCaseSensitive(rp, "mrope_interleaved")) ||
            !cJSON_IsArray(sections) || cJSON_GetArraySize(sections) != 3)
            throw std::runtime_error("unsupported MRoPE layout");
        int expected_sections[] = {11, 11, 10};
        for (int i = 0; i < 3; ++i) {
            auto *value = cJSON_GetArrayItem(sections, i);
            if (!cJSON_IsNumber(value) || value->valuedouble != expected_sections[i])
                throw std::runtime_error("unsupported MRoPE sections");
        }

        theta = num(rp, "rope_theta");
        double rd = d * num(rp, "partial_rotary_factor");
        if (rd != 64)
            throw std::runtime_error("unsupported partial rotary dimension");
        rotary = (int)rd;
        if (opt.max_context < 1 || opt.max_context > 8192)
            throw std::runtime_error("max-context must be 1..8192");
        limit = std::min(opt.max_context, integer("max_position_embeddings"));
        if (h != 1024 || f != 3584 || heads != 8 || kv != 2 || d != 256 || lh != 16 || ld != 128 ||
            layers != 24 || conv != 4 || rotary != 64 || integer("linear_num_value_heads") != lh ||
            integer("linear_value_head_dim") != ld || eps <= 0 || theta <= 0)
            throw std::runtime_error(
                "unsupported Qwen3.5 text configuration (validated 0.8B only)");
        auto *types = cJSON_GetObjectItemCaseSensitive(cfg, "layer_types");
        if (!cJSON_IsArray(types) || cJSON_GetArraySize(types) != layers)
            throw std::runtime_error("invalid layer_types");
        check(cublasCreate(&blas));
        check(cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH));
        try {
            std::string prefix = "model.language_model.";
            emb = weights.load(prefix + "embed_tokens.weight", {vocab, h});
            final_norm = loadnorm(prefix + "norm.weight", h);
            for (int i = 0; i < layers; ++i) {
                auto *type = cJSON_GetArrayItem(types, i);
                if (!cJSON_IsString(type))
                    throw std::runtime_error("invalid layer type");
                std::string kind = type->valuestring;
                if (kind != "linear_attention" && kind != "full_attention")
                    throw std::runtime_error("unsupported layer");
                Layer l{};
                l.linear = kind == "linear_attention";
                std::string base = prefix + "layers." + std::to_string(i) + ".";
                l.pre = loadnorm(base + "input_layernorm.weight", h);
                l.post = loadnorm(base + "post_attention_layernorm.weight", h);
                l.gate = weights.load(base + "mlp.gate_proj.weight", {f, h});
                l.up = weights.load(base + "mlp.up_proj.weight", {f, h});
                l.down = weights.load(base + "mlp.down_proj.weight", {h, f});
                if (l.linear) {
                    base += "linear_attn.";
                    int width = lh * ld;
                    l.q = weights.load(base + "in_proj_qkv.weight", {3 * width, h});
                    l.z = weights.load(base + "in_proj_z.weight", {width, h});
                    l.a = weights.load(base + "in_proj_a.weight", {lh, h});
                    l.b = weights.load(base + "in_proj_b.weight", {lh, h});
                    l.A = weights.load(base + "A_log", {lh});
                    l.dt = weights.load(base + "dt_bias", {lh});
                    l.cw = weights.load(base + "conv1d.weight", {3 * width, 1, conv});
                    l.ln = weights.load(base + "norm.weight", {ld});
                    l.o = weights.load(base + "out_proj.weight", {h, width});
                } else {
                    base += "self_attn.";
                    l.q = weights.load(base + "q_proj.weight", {2 * heads * d, h});
                    l.k = weights.load(base + "k_proj.weight", {kv * d, h});
                    l.v = weights.load(base + "v_proj.weight", {kv * d, h});
                    l.qn = loadnorm(base + "q_norm.weight", d);
                    l.kn = loadnorm(base + "k_norm.weight", d);
                    l.o = weights.load(base + "o_proj.weight", {h, heads * d});
                }
                blocks.push_back(l);
            }
        } catch (...) {
            cublasDestroy(blas);
            blas = nullptr;
            throw;
        }
    }
    ~Impl() {
        DeviceGuard guard(device);
        if (blas)
            cublasDestroy(blas);
    }
    HostTensor forward(const std::vector<int> &tokens,
                       bool last,
                       const std::vector<ImageFeatures> *images = nullptr) {
        if (tokens.empty() || tokens.size() > (size_t)limit)
            throw std::runtime_error("invalid context length");
        for (int id : tokens)
            if (id < 0 || id >= vocab)
                throw std::runtime_error("token outside vocabulary");
        DeviceGuard guard(device);
        int n = tokens.size(), width = lh * ld;
        DeviceTensor ids(DTYPE_INT32, {n}), x(DTYPE_FP32, {n, h}), normalized(DTYPE_FP32, {n, h}),
            projected(DTYPE_FP32, {n, h}), gate(DTYPE_FP32, {n, f}), up(DTYPE_FP32, {n, f});
        check(cudaMemcpy(ids.tensor.data, tokens.data(), n * 4, cudaMemcpyHostToDevice));
        embedding(x.tensor, *emb, ids.tensor);
        std::unique_ptr<DeviceTensor> positions;
        if (images && !images->empty()) {
            std::vector<int> pos(n * 3);
            size_t image_index = 0;
            int current = 0;
            for (int i = 0; i < n;) {
                if (tokens[i] != image_token_id) {
                    for (int axis = 0; axis < 3; ++axis)
                        pos[i * 3 + axis] = current;
                    ++current;
                    ++i;
                    continue;
                }
                if (image_index >= images->size())
                    throw std::runtime_error("image placeholders exceed image count");
                const auto &image = (*images)[image_index++];
                if (image.grid_h < 2 || image.grid_w < 2 || image.grid_h % 2 || image.grid_w % 2 ||
                    image.grid_h * image.grid_w > 1024 || !image.embeddings)
                    throw std::runtime_error("invalid image feature grid");
                int gh = image.grid_h / 2, gw = image.grid_w / 2, count = gh * gw;
                const auto &features = image.embeddings->tensor;
                if (i + count > n || features.device_id != device || features.dtype != DTYPE_FP32 ||
                    features.ndim != 2 || features.shape[0] != count || features.shape[1] != h)
                    throw std::runtime_error("invalid image feature shape/device");
                for (int j = 0; j < count; ++j) {
                    if (tokens[i + j] != image_token_id)
                        throw std::runtime_error("image placeholder length mismatch");
                    pos[(i + j) * 3] = current;
                    pos[(i + j) * 3 + 1] = current + j / gw;
                    pos[(i + j) * 3 + 2] = current + j % gw;
                }
                check(cudaMemcpy((float *)x.tensor.data + (size_t)i * h,
                                 features.data,
                                 features.nbytes,
                                 cudaMemcpyDeviceToDevice));
                current += std::max(gh, gw);
                i += count;
            }
            if (image_index != images->size())
                throw std::runtime_error("unused image features");
            positions =
                std::make_unique<DeviceTensor>(DTYPE_INT32, std::initializer_list<int64_t>{n, 3});
            check(cudaMemcpy(positions->tensor.data,
                             pos.data(),
                             pos.size() * sizeof(int),
                             cudaMemcpyHostToDevice));
        } else if (std::find(tokens.begin(), tokens.end(), image_token_id) != tokens.end())
            throw std::runtime_error("image placeholder requires image input");

        for (const auto &l : blocks) {
            norm(normalized.tensor, x.tensor, *l.pre, eps);
            if (l.linear) {
                DeviceTensor raw(DTYPE_FP32, {n, 3 * width}), mixed(DTYPE_FP32, {n, 3 * width}),
                    z(DTYPE_FP32, {n, width}), a(DTYPE_FP32, {n, lh}), b(DTYPE_FP32, {n, lh}),
                    out(DTYPE_FP32, {n, width}), normout(DTYPE_FP32, {n, width});
                linear(blas, raw.tensor, normalized.tensor, *l.q);
                linear(blas, z.tensor, normalized.tensor, *l.z);
                linear(blas, a.tensor, normalized.tensor, *l.a);
                linear(blas, b.tensor, normalized.tensor, *l.b);
                conv35(mixed.tensor, raw.tensor, *l.cw);
                delta35(out.tensor, mixed.tensor, a.tensor, b.tensor, *l.A, *l.dt);
                auto ov = reshape(out.tensor, {n, lh, ld}),
                     nv = reshape(normout.tensor, {n, lh, ld});
                norm(nv, ov, *l.ln, eps);
                gate35(normout.tensor, z.tensor, true);
                linear(blas, projected.tensor, normout.tensor, *l.o);
            } else {
                DeviceTensor qg(DTYPE_FP32, {n, heads * 2 * d}), q(DTYPE_FP32, {n, heads, d}),
                    g(DTYPE_FP32, {n, heads, d}), k(DTYPE_FP32, {n, kv, d}),
                    v(DTYPE_FP32, {n, kv, d}), out(DTYPE_FP32, {n, heads, d});
                linear(blas, qg.tensor, normalized.tensor, *l.q);
                split_gate35(q.tensor, g.tensor, qg.tensor);
                auto k2 = reshape(k.tensor, {n, kv * d}), v2 = reshape(v.tensor, {n, kv * d});
                linear(blas, k2, normalized.tensor, *l.k);
                linear(blas, v2, normalized.tensor, *l.v);
                norm(q.tensor, q.tensor, *l.qn, eps);
                norm(k.tensor, k.tensor, *l.kn, eps);
                if (positions) {
                    multimodal_rope(q.tensor, positions->tensor, theta);
                    multimodal_rope(k.tensor, positions->tensor, theta);
                } else {
                    partial_rope35(q.tensor, rotary, theta);
                    partial_rope35(k.tensor, rotary, theta);
                }
                attention(out.tensor, q.tensor, k.tensor, v.tensor);
                gate35(out.tensor, g.tensor, false);
                auto flat = reshape(out.tensor, {n, heads * d});
                linear(blas, projected.tensor, flat, *l.o);
            }
            add(x.tensor, projected.tensor);
            norm(normalized.tensor, x.tensor, *l.post, eps);
            linear(blas, gate.tensor, normalized.tensor, *l.gate);
            linear(blas, up.tensor, normalized.tensor, *l.up);
            silu_mul(gate.tensor, up.tensor);
            linear(blas, projected.tensor, gate.tensor, *l.down);
            add(x.tensor, projected.tensor);
        }
        norm(normalized.tensor, x.tensor, *final_norm, eps);
        auto input = last ? last_row(normalized.tensor) : normalized.tensor;
        DeviceTensor logits(DTYPE_FP32, {last ? 1 : n, vocab});
        linear(blas, logits.tensor, input, *emb);
        HostTensor result({last ? 1 : n, vocab});
        check(cudaMemcpy(
            result.data(), logits.tensor.data, result.tensor.nbytes, cudaMemcpyDeviceToHost));
        for (float x : result)
            if (!std::isfinite(x))
                throw std::runtime_error("nonfinite logits");
        return result;
    }
};
Qwen3_5::Qwen3_5(const std::string &dir, const Qwen3Options &opt)
    : impl(std::make_unique<Impl>(dir, opt)) {}
Qwen3_5::~Qwen3_5() = default;
HostTensor Qwen3_5::forward(const std::vector<int> &tokens, bool last) {
    return impl->forward(tokens, last);
}
int Qwen3_5::context_limit() const {
    return impl->limit;
}
ImageFeatures Qwen3_5::encode_image(const ImageInput &image) {
    DeviceGuard guard(impl->device);
    if (!impl->vision)
        impl->vision = std::make_unique<Qwen3_5Vision>(impl->directory);
    return impl->vision->encode(image);
}
HostTensor Qwen3_5::forward_multimodal(const std::vector<int> &tokens,
                                       const std::vector<ImageFeatures> &images,
                                       bool last) {
    return impl->forward(tokens, last, &images);
}
} // namespace qwe
