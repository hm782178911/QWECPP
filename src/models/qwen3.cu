#include "models/qwen3.cuh"
#include "backend/cuda/kv_cache.cuh"
#include "loader/weight_archive.hpp"
#include "operators/tensor_ops.cuh"
#include "backend/cuda/device_tensor.cuh"
#include "backend/cuda/device_guard.cuh"
#include "backend/cuda/transfer.cuh"
extern "C" {
#include "loader/safetensors.h"
}
#include "cJSON.h"
#include <fstream>
#include <iterator>
#include <memory>
#include <map>
#include <cmath>
#include <limits>
#include <algorithm>
namespace qwe {
struct Config {
    int hidden;       // 隐藏状态宽度，例如当前模型的 1024。
    int intermediate; // MLP 中间层宽度。
    int layers;       // Transformer Block 数量。
    int heads;        // Query 头数。
    int kvheads;      // Key/Value 头数；GQA 中可以少于 Query 头数。
    int dim;          // 每个注意力头的宽度，不一定等于 hidden / heads。
    int vocab;        // 词表大小，也是输出 logits 的最后一维。
    int maxpos;       // 模型配置支持的位置上限。
    float eps;        // RMSNorm 防止除零的小常数。
    float theta;      // RoPE 频率基数。
    bool tied;        // 是否让输出投影复用 Embedding 权重。
};

/**
 * @brief 读取 Qwen3 配置并检查基础实现的支持范围。
 * @param[in] path config.json 的完整文件路径。
 * @return 校验后的 Config；文件缺失、字段非法或变体不支持时抛异常。
 */
static Config config(const std::string &path) {
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("cannot read config: " + path);
    std::string text((std::istreambuf_iterator<char>(input)), {});
    std::unique_ptr<cJSON, decltype(&cJSON_Delete)> root(cJSON_Parse(text.c_str()), cJSON_Delete);
    if (!cJSON_IsObject(root.get()))
        throw std::runtime_error("invalid config JSON");
    auto get = [&](const char *key) { return cJSON_GetObjectItemCaseSensitive(root.get(), key); };
    auto number = [&](const char *key) -> double {
        const cJSON *v = get(key);
        if (!cJSON_IsNumber(v) || !std::isfinite(v->valuedouble))
            throw std::runtime_error(std::string("invalid config: ") + key);
        return v->valuedouble;
    };
    auto integer = [&](const char *key) -> int {
        double v = number(key);
        if (v < 1 || v > 2000000 || std::floor(v) != v)
            throw std::runtime_error(std::string("invalid integer: ") + key);
        return (int)v;
    };
    const cJSON *type = get("model_type"), *act = get("hidden_act");
    if (!cJSON_IsString(type) || std::string(type->valuestring) != "qwen3" ||
        !cJSON_IsString(act) || std::string(act->valuestring) != "silu")
        throw std::runtime_error("only Qwen3 SiLU architecture supported");
    for (const char *key : {"attention_bias", "use_sliding_window"}) {
        const cJSON *v = get(key);
        if (v && !cJSON_IsFalse(v))
            throw std::runtime_error(std::string("unsupported config: ") + key);
    }
    if (get("rope_scaling") && !cJSON_IsNull(get("rope_scaling")))
        throw std::runtime_error("scaled RoPE not supported");
    if (!cJSON_IsBool(get("tie_word_embeddings")))
        throw std::runtime_error("missing tie_word_embeddings");
    Config c{integer("hidden_size"),
             integer("intermediate_size"),
             integer("num_hidden_layers"),
             integer("num_attention_heads"),
             integer("num_key_value_heads"),
             integer("head_dim"),
             integer("vocab_size"),
             integer("max_position_embeddings"),
             (float)number("rms_norm_eps"),
             (float)number("rope_theta"),
             cJSON_IsTrue(get("tie_word_embeddings")) != 0};
    if (c.hidden > 32768 || c.intermediate > 131072 || c.layers > 256 || c.heads > 256 ||
        c.kvheads > c.heads || c.heads % c.kvheads || c.dim > 1024 || c.dim % 2 || c.eps <= 0 ||
        !std::isfinite(c.eps) || c.theta <= 0 || !std::isfinite(c.theta))
        throw std::runtime_error("unsupported model dimensions");
    return c;
}

// 模型复用同一个 cuBLAS 句柄，避免每次前向重复创建。
struct Blas {
    cublasHandle_t handle = nullptr;
    int device = -1;

    /**
     * @brief 为当前 GPU 创建一个可复用 cuBLAS 句柄。
     * @return 构造句柄所有者；失败时抛异常。
     */
    Blas() {
        check(cudaGetDevice(&device));
        check(cublasCreate(&handle));
    }

    /**
     * @brief 销毁本对象持有的 cuBLAS 句柄。
     * @return 无。
     */
    ~Blas() {
        if (handle) {
            int previous;
            if (cudaGetDevice(&previous) == cudaSuccess && cudaSetDevice(device) == cudaSuccess) {
                cublasDestroy(handle);
                cudaSetDevice(previous);
            }
        }
    }
};

// 按权重名称缓存显存对象。Layer 只引用 Tensor，显存由这里统一持有。
struct WeightStore {
    WeightArchive archive;
    std::map<std::string, std::unique_ptr<DeviceTensor>> gpu;

    /**
     * @brief 创建按名称管理 GPU 权重的缓存并映射源文件。
     * @param[in] path 源 Safetensors 文件路径。
     * @return 构造权重缓存；显存数据由 load 按需加载。
     */
    explicit WeightStore(const std::string &path) : archive(path) {}

    /**
     * @brief 按名称加载、转换并缓存一份 GPU 权重。
     * @param[in] name Safetensors 中的完整张量名称。
     * @param[in] shape 模型期望的权重形状，必须与文件一致。
     * @return 缓存中 FP32 CUDA Tensor 的借用指针；所有者为 WeightStore。
     * @note 缓存命中直接复用；否则上传数据，必要时从 BF16/FP16 转成 FP32。
     */
    const Tensor *load(const std::string &name, std::initializer_list<int64_t> shape) {
        // 同一份权重只上传一次；返回的指针在 WeightStore 存活期间有效。
        auto cached = gpu.find(name);
        if (cached != gpu.end())
            return &cached->second->tensor;
        const auto &file = archive.file_for(name);
        const SafeTensorEntry *e = safetensors_find(&file, name.c_str());
        if (!e || e->ndim != (int)shape.size())
            throw std::runtime_error("missing or invalid weight: " + name);
        int i = 0;
        for (auto d : shape)
            if (e->shape[i++] != d)
                throw std::runtime_error("weight shape mismatch: " + name);
        if (e->dtype != DTYPE_FP32 && e->dtype != DTYPE_FP16 && e->dtype != DTYPE_BF16)
            throw std::runtime_error("unsupported weight dtype: " + name);

        // 首版算子统一使用 FP32；BF16/FP16 权重在上传后转换一次。
        auto out = std::make_unique<DeviceTensor>(DTYPE_FP32, shape);
        if (e->dtype == DTYPE_FP32)
            check(cudaMemcpy(out->tensor.data,
                             safetensors_get_data(&file, e),
                             out->tensor.nbytes,
                             cudaMemcpyHostToDevice));
        else {
            DeviceTensor raw(e->dtype, shape);
            check(cudaMemcpy(raw.tensor.data,
                             safetensors_get_data(&file, e),
                             raw.tensor.nbytes,
                             cudaMemcpyHostToDevice));
            convert(out->tensor, raw.tensor);
        }
        const Tensor *result = &out->tensor;
        gpu.emplace(name, std::move(out));
        return result;
    }
    // Pack only this rank's contiguous row/column slice before uploading.
    const Tensor *shard(const std::string &name, int rows, int cols, int axis, int rank) {
        const auto &file = archive.file_for(name);
        const auto *e = safetensors_find(&file, name.c_str());
        if (!e || e->ndim != 2 || e->shape[0] != rows || e->shape[1] != cols)
            throw std::runtime_error("weight shape mismatch: " + name);
        if (e->dtype != DTYPE_FP32 && e->dtype != DTYPE_FP16 && e->dtype != DTYPE_BF16)
            throw std::runtime_error("unsupported weight dtype: " + name);
        int r = axis == 0 ? rows / 2 : rows, c = axis == 1 ? cols / 2 : cols;
        size_t width = dtype_size(e->dtype);
        std::vector<unsigned char> packed((size_t)r*c*width);
        const auto *source = static_cast<const unsigned char *>(safetensors_get_data(&file,e));
        for (int i=0;i<r;++i) {
            size_t offset = (size_t)(i + (axis==0 ? rank*r : 0))*cols + (axis==1 ? rank*c : 0);
            std::copy_n(source+offset*width,(size_t)c*width,packed.data()+(size_t)i*c*width);
        }
        auto out=std::make_unique<DeviceTensor>(DTYPE_FP32,std::initializer_list<int64_t>{r,c});
        DeviceTensor raw(e->dtype,{r,c});
        check(cudaMemcpy(raw.tensor.data,packed.data(),packed.size(),cudaMemcpyHostToDevice));
        if(e->dtype==DTYPE_FP32) check(cudaMemcpy(out->tensor.data,raw.tensor.data,out->tensor.nbytes,cudaMemcpyDeviceToDevice));
        else convert(out->tensor,raw.tensor);
        const Tensor *result=&out->tensor;
        gpu.emplace(name,std::move(out));
        return result;
    }

};

struct Layer {
    const Tensor *pre;  // Attention 前的 RMSNorm 权重。
    const Tensor *post; // MLP 前的 RMSNorm 权重。
    const Tensor *q;    // Query 投影权重。
    const Tensor *k;    // Key 投影权重。
    const Tensor *v;    // Value 投影权重。
    const Tensor *o;    // Attention 输出投影权重。
    const Tensor *qn;   // 每个 Query head 的 RMSNorm 权重。
    const Tensor *kn;   // 每个 Key head 的 RMSNorm 权重。
    const Tensor *gate; // SwiGLU 的门控分支。
    const Tensor *up;   // SwiGLU 的升维分支。
    const Tensor *down; // MLP 降维投影。
};

/**
 * @brief 一个连续层阶段；只持有本阶段所需权重，进入阶段时选择对应 GPU。
 * @note 当前阶段顺序执行，尚不重叠调度多个请求；默认 CUDA stream 属于各自设备。
 */
struct Stage {
    int device;
    Config c;
    bool force_host;
    std::unique_ptr<WeightStore> store;
    std::unique_ptr<Blas> blas;
    const Tensor *emb = nullptr;
    const Tensor *head = nullptr;
    const Tensor *output_norm = nullptr;
    std::vector<Layer> layers;
    std::vector<std::unique_ptr<KVCache>> cache;
    void attend(size_t layer,Tensor &out,const Tensor &q,const Tensor &k,const Tensor &v,int offset,int capacity) {
        if(!capacity) { attention(out,q,k,v); return; }
        if(cache.size()<layers.size()) cache.resize(layers.size());
        if(!cache[layer]) cache[layer]=std::make_unique<KVCache>(capacity,(int)k.shape[1],c.dim);
        cache[layer]->append(k,v,offset);
        auto keys=cache[layer]->key_view(offset+(int)q.shape[0]);
        auto values=cache[layer]->value_view(offset+(int)q.shape[0]);
        attention_cached(out,q,keys,values,offset);
    }
    size_t cache_bytes() const { size_t result=0;for(auto &p:cache) if(p) result+=p->bytes();return result; }

    /**
     * @brief 在指定设备加载连续层区间及必要的输入/输出权重。
     * @param[in] directory 模型目录。
     * @param[in] config 已校验配置。
     * @param[in] gpu 当前阶段设备编号。
     * @param[in] begin 第一层编号（包含）。
     * @param[in] end 结束层编号（不包含）。
     * @param[in] host_transfer 是否强制主机中转。
     * @return 构造阶段；权重、句柄由本对象持有。
     */
    Stage(const std::string &directory,
          const Config &config,
          int gpu,
          int begin,
          int end,
          bool host_transfer, int rank = -1)
        : device(gpu), c(config), force_host(host_transfer) {
        DeviceGuard guard(device);
        store = std::make_unique<WeightStore>(directory);
        blas = std::make_unique<Blas>();
        check(cublasSetMathMode(blas->handle, CUBLAS_PEDANTIC_MATH));
        int h = c.hidden, f = c.intermediate, qd = c.heads * c.dim, kd = c.kvheads * c.dim;
        if (begin == 0)
            emb = store->load("model.embed_tokens.weight", {c.vocab, h});
        if (end == c.layers) {
            // tied embedding 在不同 GPU 上各保存同值副本；单卡缓存只分配一次。
            head =
                store->load(c.tied ? "model.embed_tokens.weight" : "lm_head.weight", {c.vocab, h});
            output_norm = store->load("model.norm.weight", {h});
        }
        for (int i = begin; i < end; ++i) {
            std::string base = "model.layers." + std::to_string(i) + ".";
            auto load = [&](const char *name, std::initializer_list<int64_t> shape) {
                if (rank >= 0 && shape.size() == 2) {
                    auto dims=shape.begin(); int rows=dims[0], cols=dims[1];
                    std::string key=name;
                    int axis=(key=="self_attn.o_proj.weight" || key=="mlp.down_proj.weight") ? 1 : 0;
                    return store->shard(base+name,rows,cols,axis,rank);
                }
                return store->load(base + name, shape);
            };
            layers.push_back({load("input_layernorm.weight", {h}),
                              load("post_attention_layernorm.weight", {h}),
                              load("self_attn.q_proj.weight", {qd, h}),
                              load("self_attn.k_proj.weight", {kd, h}),
                              load("self_attn.v_proj.weight", {kd, h}),
                              load("self_attn.o_proj.weight", {h, qd}),
                              load("self_attn.q_norm.weight", {c.dim}),
                              load("self_attn.k_norm.weight", {c.dim}),
                              load("mlp.gate_proj.weight", {f, h}),
                              load("mlp.up_proj.weight", {f, h}),
                              load("mlp.down_proj.weight", {h, f})});
        }
    }

    /**
     * @brief 执行本阶段的 Embedding（若有）及 Transformer 层。
     * @param[in] tokens 已校验的完整 token 序列，用于确定序列长度及首阶段查表。
     * @param[in] incoming 上阶段的隐藏状态；首阶段为 nullptr，其他阶段为 [n,h]。
     * @return 本设备上拥有显存的 [n,h] 隐藏状态；不复制回 CPU。
     */
    std::unique_ptr<DeviceTensor> run(const std::vector<int> &tokens,
                                      const DeviceTensor *incoming, int offset=0, int capacity=0) {
        DeviceGuard guard(device);
        int n = (int)tokens.size(), h = c.hidden, f = c.intermediate, qd = c.heads * c.dim,
            kd = c.kvheads * c.dim;
        auto owned =
            std::make_unique<DeviceTensor>(DTYPE_FP32, std::initializer_list<int64_t>{n, h});
        auto &x = *owned;
        DeviceTensor normed(DTYPE_FP32, {n, h}), q(DTYPE_FP32, {n, qd}), k(DTYPE_FP32, {n, kd}),
            v(DTYPE_FP32, {n, kd}), attn(DTYPE_FP32, {n, qd}), projected(DTYPE_FP32, {n, h}),
            gate(DTYPE_FP32, {n, f}), up(DTYPE_FP32, {n, f});
        if (incoming)
            transfer(x.tensor, incoming->tensor, force_host);
        else {
            DeviceTensor ids(DTYPE_INT32, {n});
            static_assert(sizeof(int) == sizeof(int32_t));
            check(cudaMemcpy(
                ids.tensor.data, tokens.data(), ids.tensor.nbytes, cudaMemcpyHostToDevice));
            embedding(x.tensor, *emb, ids.tensor);
        }
        auto q_heads = reshape(q.tensor, {n, c.heads, c.dim});
        auto k_heads = reshape(k.tensor, {n, c.kvheads, c.dim});
        auto v_heads = reshape(v.tensor, {n, c.kvheads, c.dim});
        auto attn_heads = reshape(attn.tensor, {n, c.heads, c.dim});
        size_t layer_index=0;
        for (const Layer &l : layers) {
            // Attention 分支：预归一化 → Q/K/V 投影 → 按 head 归一化与旋转位置编码。
            norm(normed.tensor, x.tensor, *l.pre, c.eps);
            linear(blas->handle, q.tensor, normed.tensor, *l.q);
            linear(blas->handle, k.tensor, normed.tensor, *l.k);
            linear(blas->handle, v.tensor, normed.tensor, *l.v);
            norm(q_heads, q_heads, *l.qn, c.eps);
            norm(k_heads, k_heads, *l.kn, c.eps);
            rope(q_heads, c.theta, offset);
            rope(k_heads, c.theta, offset);

            // 因果 GQA 只读取当前位置及以前的 K/V，然后投影回 h 并加上残差。
            attend(layer_index++,attn_heads,q_heads,k_heads,v_heads,offset,capacity);
            linear(blas->handle, projected.tensor, attn.tensor, *l.o);
            add(x.tensor, projected.tensor);

            // MLP 分支：RMSNorm → SiLU(gate) × up → down → 残差相加。
            norm(normed.tensor, x.tensor, *l.post, c.eps);
            linear(blas->handle, gate.tensor, normed.tensor, *l.gate);
            linear(blas->handle, up.tensor, normed.tensor, *l.up);
            silu_mul(gate.tensor, up.tensor);
            linear(blas->handle, projected.tensor, gate.tensor, *l.down);
            add(x.tensor, projected.tensor);
        }
        return owned;
    }

    /**
     * @brief 在最后阶段执行最终 RMSNorm 和词表投影。
     * @param[in] hidden 本设备上的 [n,h] 隐藏状态。
     * @param[in] last_only 是否只计算最后位置 logits。
     * @return FP32 CPU Tensor，形状 [last_only?1:n,vocab]，拥有返回数据。
     */
    HostTensor project(const DeviceTensor &hidden, bool last_only) {
        DeviceGuard guard(device);
        int n = (int)hidden.tensor.shape[0];
        DeviceTensor normed(DTYPE_FP32, {n, c.hidden}),
            logits(DTYPE_FP32, {last_only ? 1 : n, c.vocab});
        norm(normed.tensor, hidden.tensor, *output_norm, c.eps);
        auto input = last_only ? last_row(normed.tensor) : normed.tensor;
        linear(blas->handle, logits.tensor, input, *head);
        HostTensor result({last_only ? 1 : n, c.vocab});
        check(cudaMemcpy(
            result.data(), logits.tensor.data, result.tensor.nbytes, cudaMemcpyDeviceToHost));
        for (float value : result)
            if (!std::isfinite(value))
                throw std::runtime_error("non-finite logits");
        return result;
    }
};

struct Qwen3::Impl {
    Config c;
    int context = 2048;
    bool kv_enabled = false;
    std::vector<int> cached_tokens;
    bool tp = false;
    bool profile = false;
    TransferStats transfer_stats;
    std::vector<std::unique_ptr<Stage>> stages;
    /**
     * @brief 校验设备与切层选项，建立一个或两个阶段。
     * @param[in] directory 模型目录。
     * @param[in] options 设备列表、切层边界和传输选项。
     * @return 构造常驻模型；设备或切层非法时在加载权重前抛异常。
     */
    Impl(const std::string &directory, const Qwen3Options &options)
        : c(config(directory + "/config.json")) {
        if(options.max_context < 1 || options.max_context > 8192)
            throw std::runtime_error("max-context must be 1..8192");
        context=std::min(options.max_context,c.maxpos);
        auto devices = options.devices;
        if (devices.empty()) {
            int current;
            check(cudaGetDevice(&current));
            devices.push_back(current);
        }
        int count;
        check(cudaGetDeviceCount(&count));
        if (devices.size() > 2 || (devices.size() == 2 && devices[0] == devices[1]))
            throw std::runtime_error("expected one or two distinct CUDA devices");
        for (int id : devices)
            if (id < 0 || id >= count)
                throw std::runtime_error("CUDA device index out of range");
        kv_enabled = options.kv_cache;
        profile = options.profile;
        tp = options.tensor_parallel;
        if (tp) {
            if (devices.size()!=2 || options.split_layer!=-1)
                throw std::runtime_error("TP requires two GPUs and no split-layer");
            if (c.heads%2 || c.kvheads%2 || c.intermediate%2)
                throw std::runtime_error("TP requires heads, KV heads and intermediate divisible by two");
            for(int rank=0;rank<2;++rank)
                stages.push_back(std::make_unique<Stage>(directory,c,devices[rank],0,c.layers,options.force_host_transfer,rank));
        } else if (devices.size() == 1) {
            if (options.split_layer != -1)
                throw std::runtime_error("split-layer requires two GPUs");
            stages.push_back(std::make_unique<Stage>(
                directory, c, devices[0], 0, c.layers, options.force_host_transfer));
        } else {
            int split = options.split_layer == -1 ? c.layers / 2 : options.split_layer;
            if (split < 1 || split >= c.layers)
                throw std::runtime_error("split-layer must be inside model layer range");
            stages.push_back(std::make_unique<Stage>(
                directory, c, devices[0], 0, split, options.force_host_transfer));
            stages.push_back(std::make_unique<Stage>(
                directory, c, devices[1], split, c.layers, options.force_host_transfer));
        }
    }
    /**
     * @brief 顺序执行各阶段并返回最后阶段的输出。
     * @param[in] tokens 完整上下文 token ID，长度 1～context_limit()，且不超过模型位置限制。
     * @param[in] last_only 是否只返回最后位置 logits。
     * @return 拥有 CPU 内存的 FP32 HostTensor。
     * @note 没有 KV Cache；单请求不产生流水线重叠。
     */

    HostTensor run_tp(const std::vector<int> &tokens, bool last_only, int offset=0, int capacity=0) {
        int n=(int)tokens.size();
        struct Work {
            DeviceTensor x,normed,q,k,v,attn,projected,gate,up,other;
            Work(int n,const Config &c):x(DTYPE_FP32,{n,c.hidden}),normed(DTYPE_FP32,{n,c.hidden}),
              q(DTYPE_FP32,{n,c.heads*c.dim/2}),k(DTYPE_FP32,{n,c.kvheads*c.dim/2}),
              v(DTYPE_FP32,{n,c.kvheads*c.dim/2}),attn(DTYPE_FP32,{n,c.heads*c.dim/2}),
              projected(DTYPE_FP32,{n,c.hidden}),gate(DTYPE_FP32,{n,c.intermediate/2}),
              up(DTYPE_FP32,{n,c.intermediate/2}),other(DTYPE_FP32,{n,c.hidden}) {}
        };
        std::unique_ptr<Work> w[2];
        for(int rank=0;rank<2;++rank) {
            DeviceGuard guard(stages[rank]->device);
            w[rank]=std::make_unique<Work>(n,c);
            DeviceTensor ids(DTYPE_INT32,{n});
            check(cudaMemcpy(ids.tensor.data,tokens.data(),ids.tensor.nbytes,cudaMemcpyHostToDevice));
            embedding(w[rank]->x.tensor,*stages[rank]->emb,ids.tensor);
        }
        for(int layer=0;layer<c.layers;++layer) for(int phase=0;phase<2;++phase) {
            for(int rank=0;rank<2;++rank) {
                auto &stage=*stages[rank]; DeviceGuard guard(stage.device);
                auto &x=w[rank]->x; auto &normed=w[rank]->normed;
                auto &q=w[rank]->q; auto &k=w[rank]->k; auto &v=w[rank]->v;
                auto &attn=w[rank]->attn; auto &projected=w[rank]->projected;
                auto &gate=w[rank]->gate; auto &up=w[rank]->up;
                auto &blas=stage.blas; const auto &l=stage.layers[layer];
                auto q_heads=reshape(q.tensor,{n,c.heads/2,c.dim});
                auto k_heads=reshape(k.tensor,{n,c.kvheads/2,c.dim});
                auto v_heads=reshape(v.tensor,{n,c.kvheads/2,c.dim});
                auto attn_heads=reshape(attn.tensor,{n,c.heads/2,c.dim});
                if(phase==0) {
            norm(normed.tensor, x.tensor, *l.pre, c.eps);
            linear(blas->handle, q.tensor, normed.tensor, *l.q);
            linear(blas->handle, k.tensor, normed.tensor, *l.k);
            linear(blas->handle, v.tensor, normed.tensor, *l.v);
            norm(q_heads, q_heads, *l.qn, c.eps);
            norm(k_heads, k_heads, *l.kn, c.eps);
            rope(q_heads, c.theta, offset);
            rope(k_heads, c.theta, offset);

            // 因果 GQA 只读取当前位置及以前的 K/V，然后投影回 h 并加上残差。
            stage.attend(layer,attn_heads,q_heads,k_heads,v_heads,offset,capacity);
            linear(blas->handle, projected.tensor, attn.tensor, *l.o);
                } else {
            norm(normed.tensor, x.tensor, *l.post, c.eps);
            linear(blas->handle, gate.tensor, normed.tensor, *l.gate);
            linear(blas->handle, up.tensor, normed.tensor, *l.up);
            silu_mul(gate.tensor, up.tensor);
            linear(blas->handle, projected.tensor, gate.tensor, *l.down);
                }
            }
            // Both partial outputs remain intact until each rank has received its peer.
            for(int rank=0;rank<2;++rank)
                transfer(w[rank]->other.tensor,w[1-rank]->projected.tensor,stages[rank]->force_host,profile ? &transfer_stats : nullptr);
            for(int rank=0;rank<2;++rank) {
                DeviceGuard guard(stages[rank]->device);
                add(w[rank]->projected.tensor,w[rank]->other.tensor);
                add(w[rank]->x.tensor,w[rank]->projected.tensor);
            }
        }
        return stages[0]->project(w[0]->x,last_only);
    }

    HostTensor forward(const std::vector<int> &tokens, bool last_only) {
        if (tokens.empty() || tokens.size() > (size_t)context)
            throw std::runtime_error("token count exceeds configured context limit");
        for (int id : tokens)
            if (id < 0 || id >= c.vocab)
                throw std::runtime_error("token ID outside vocabulary");
        transfer_stats = {};
        int offset=0,capacity=kv_enabled && last_only ? context : 0;
        if(capacity && tokens.size()>cached_tokens.size() &&
           std::equal(cached_tokens.begin(),cached_tokens.end(),tokens.begin())) offset=(int)cached_tokens.size();
        std::vector<int> suffix(tokens.begin()+offset,tokens.end());
        try {
            HostTensor result = [&]() {
                if(tp) return run_tp(suffix,last_only,offset,capacity);
                std::unique_ptr<DeviceTensor> hidden;
                for(auto &stage:stages) hidden=stage->run(suffix,hidden.get(),offset,capacity);
                return stages.back()->project(*hidden,last_only);
            }();
            cached_tokens=capacity ? tokens : std::vector<int>{};
            return result;
        } catch(...) { cached_tokens.clear(); throw; }
    }
};

/**
 * @brief 加载单卡或两卡分层模型，权重在实例存活期间常驻各自 GPU。
 * @param[in] directory 包含 config.json 和 model.safetensors 的模型目录。
 * @param[in] options 设备顺序、切层边界及传输选项。
 * @return 构造模型实例；设备、配置、权重或分配失败时抛异常。
 */
Qwen3::Qwen3(const std::string &directory, const Qwen3Options &options)
    : impl(std::make_unique<Impl>(directory, options)) {}
/**
 * @brief 释放所有阶段的权重显存、cuBLAS 句柄及文件映射。
 * @return 无；不要求调用方预先选择某张 GPU。
 */
Qwen3::~Qwen3() = default;
/**
 * @brief 查询当前实现支持的最大输入长度。
 * @return 配置的有效上下文上限，单位为 token。
 */
void Qwen3::reset_cache() { impl->cached_tokens.clear(); }
size_t Qwen3::kv_cache_bytes() const { size_t bytes=0; for(auto &stage:impl->stages) bytes+=stage->cache_bytes(); return bytes; }
TransferStats Qwen3::last_transfer_stats() const { return impl->transfer_stats; }
int Qwen3::context_limit() const {
    return impl->context;
}
/**
 * @brief 复用常驻的单卡/双卡阶段执行完整前向。
 * @param[in] tokens 完整上下文 ID，长度 1～context_limit()，且各 ID 在词表内。
 * @param[in] last_only true 仅返回最后位置；false 返回所有位置。
 * @return FP32 CPU HostTensor，形状 [last_only?1:n, vocab]，持有输出内存。
 * @note 无 KV Cache；双卡按阶段依次执行，返回后恢复调用方设备。
 */
HostTensor Qwen3::forward(const std::vector<int> &tokens, bool last_only) {
    return impl->forward(tokens, last_only);
}
/**
 * @brief 在当前设备上创建临时单卡模型，执行一次完整前向。
 * @param[in] directory 模型目录。
 * @param[in] tokens 完整上下文 token ID。
 * @return FP32 CPU HostTensor，形状 [n, vocab]；临时模型资源随调用结束释放。
 */
HostTensor qwen3_forward(const std::string &directory, const std::vector<int> &tokens) {
    Qwen3 model(directory);
    return model.forward(tokens);
}
} // namespace qwe
