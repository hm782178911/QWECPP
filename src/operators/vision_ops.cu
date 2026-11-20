#include "operators/vision_ops.cuh"
#include "backend/cuda/error.cuh"
#include <stdexcept>
#include <cmath>
namespace qwe::kernels {
void bias_rows(float *, const float *, int, int);
void layer_norm(float *, const float *, const float *, const float *, int, int, float);
void gelu(float *, int, bool);
void vision_position(float *, const float *, const float *, int, int);
void vision_rope(float *, int, int);
void vision_attention(float *, const float *, int);
void multimodal_rope(float *, const int *, int, int, int, float);
} // namespace qwe::kernels
namespace qwe {
static void require(bool ok, const char *s) {
    if (!ok)
        throw std::runtime_error(s);
}
static void validate(const Tensor &t, DType type = DTYPE_FP32) {
    int device;
    check(cudaGetDevice(&device));
    require(t.data && t.ndim > 0 && t.ndim <= 4 && t.dtype == type && t.device == TENSOR_CUDA &&
                t.device_id == device,
            "invalid vision tensor");
    size_t size = 1;
    for (int i = t.ndim - 1; i >= 0; --i) {
        require(t.shape[i] > 0 && t.stride[i] == (int64_t)size, "noncontiguous vision tensor");
        size *= t.shape[i];
    }
    require(size == t.numel && size < 2147480000, "invalid vision size");
}
static float *ptr(const Tensor &t) {
    validate(t);
    return (float *)t.data;
}
void bias_rows(Tensor &x, const Tensor &b) {
    require(x.ndim == 2 && b.ndim == 1 && b.shape[0] == x.shape[1], "bias shape");
    kernels::bias_rows(ptr(x), ptr(b), (int)x.shape[0], (int)x.shape[1]);
}
void layer_norm(Tensor &out, const Tensor &x, const Tensor &w, const Tensor &b, float eps) {
    require(x.ndim == 2 && out.ndim == 2 && out.shape[0] == x.shape[0] &&
                out.shape[1] == x.shape[1] && w.ndim == 1 && b.ndim == 1 &&
                w.shape[0] == x.shape[1] && b.shape[0] == x.shape[1] && std::isfinite(eps) &&
                eps > 0,
            "LayerNorm shape");
    kernels::layer_norm(ptr(out), ptr(x), ptr(w), ptr(b), (int)x.shape[0], (int)x.shape[1], eps);
}
void gelu(Tensor &x, bool approx) {
    kernels::gelu(ptr(x), (int)x.numel, approx);
}
void vision_position(Tensor &x, const Tensor &b, const Tensor &t, int gh, int gw) {
    require(gh >= 2 && gw >= 2 && gh % 2 == 0 && gw % 2 == 0 && gh * gw <= 1024 && x.ndim == 2 &&
                x.shape[0] == gh * gw && x.shape[1] == 768 && b.numel == 768 && t.ndim == 2 &&
                t.shape[0] == 2304 && t.shape[1] == 768,
            "vision position shape");
    kernels::vision_position(ptr(x), ptr(b), ptr(t), gh, gw);
}
void vision_rope(Tensor &x, int gh, int gw) {
    require(x.ndim == 2 && x.shape[0] == gh * gw && x.shape[1] == 2304 && gh > 0 && gw > 0 &&
                gh % 2 == 0 && gw % 2 == 0 && gh * gw <= 1024,
            "vision RoPE shape");
    kernels::vision_rope(ptr(x), (int)x.shape[0], gw);
}
void vision_attention(Tensor &out, const Tensor &x) {
    require(x.ndim == 2 && out.ndim == 2 && x.shape[1] == 2304 && out.shape[0] == x.shape[0] &&
                out.shape[1] == 768 && x.shape[0] <= 1024 && out.data != x.data,
            "vision Attention shape");
    kernels::vision_attention(ptr(out), ptr(x), (int)x.shape[0]);
}
void multimodal_rope(Tensor &x, const Tensor &p, float theta) {
    validate(p, DTYPE_INT32);
    require(x.ndim == 3 && x.shape[2] >= 64 && p.ndim == 2 && p.shape[0] == x.shape[0] &&
                p.shape[1] == 3 && std::isfinite(theta) && theta > 0,
            "MRoPE shape");
    kernels::multimodal_rope(
        ptr(x), (const int *)p.data, (int)x.shape[0], (int)x.shape[1], (int)x.shape[2], theta);
}
} // namespace qwe
