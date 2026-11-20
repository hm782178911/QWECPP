#include "operators/qwen35.cuh"
#include "backend/cuda/error.cuh"
#include "backend/cuda/device_guard.cuh"
#include <stdexcept>
namespace qwe::kernels {
void split_gate35(float *, float *, const float *, int, int);
void gate35(float *, const float *, int, bool);
void partial_rope35(float *, int, int, int, int, float);
void conv35(float *, const float *, const float *, int, int, int);
void delta35(
    float *, const float *, const float *, const float *, const float *, const float *, int, int);
} // namespace qwe::kernels
namespace qwe {
static float *data(const Tensor &t) {
    int device;
    check(cudaGetDevice(&device));
    if (!t.data || t.device != TENSOR_CUDA || t.device_id != device || t.dtype != DTYPE_FP32)
        throw std::runtime_error("invalid Qwen3.5 tensor");
    size_t stride = 1;
    for (int i = t.ndim - 1; i >= 0; --i) {
        if (t.shape[i] < 1 || t.stride[i] != (int64_t)stride)
            throw std::runtime_error("noncontiguous Qwen3.5 tensor");
        stride *= t.shape[i];
    }
    if (stride != t.numel)
        throw std::runtime_error("invalid tensor size");
    return (float *)t.data;
}
void split_gate35(Tensor &q, Tensor &g, const Tensor &x) {
    if (q.ndim != 3 || g.numel != q.numel || x.numel != 2 * q.numel)
        throw std::runtime_error("Q gate shape");
    kernels::split_gate35(data(q), data(g), data(x), (int)q.numel, (int)q.shape[2]);
}
void gate35(Tensor &x, const Tensor &g, bool silu) {
    if (x.numel != g.numel)
        throw std::runtime_error("gate shape");
    kernels::gate35(data(x), data(g), (int)x.numel, silu);
}
void partial_rope35(Tensor &x, int rd, float theta) {
    if (x.ndim != 3 || rd < 2 || rd % 2 || rd > x.shape[2])
        throw std::runtime_error("partial RoPE shape");
    kernels::partial_rope35(data(x), (int)x.shape[0], (int)x.shape[1], (int)x.shape[2], rd, theta);
}
void conv35(Tensor &out, const Tensor &x, const Tensor &w) {
    if (x.ndim != 2 || out.numel != x.numel || w.ndim != 3 || w.shape[0] != x.shape[1] ||
        w.shape[1] != 1)
        throw std::runtime_error("conv shape");
    kernels::conv35(data(out), data(x), data(w), (int)x.shape[0], (int)x.shape[1], (int)w.shape[2]);
}
void delta35(Tensor &out,
             const Tensor &x,
             const Tensor &a,
             const Tensor &b,
             const Tensor &A,
             const Tensor &dt) {
    if (out.ndim != 2 || a.ndim != 2 || x.numel != out.numel * 3 || a.numel != b.numel ||
        a.shape[0] != out.shape[0] || out.shape[1] != a.shape[1] * 128 || A.numel != a.shape[1] ||
        dt.numel != A.numel)
        throw std::runtime_error("delta shape");
    kernels::delta35(data(out),
                     data(x),
                     data(a),
                     data(b),
                     data(A),
                     data(dt),
                     (int)out.shape[0],
                     (int)a.shape[1]);
}
} // namespace qwe
