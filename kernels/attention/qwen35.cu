#include "cuda_kernels.cuh"
#include "backend/cuda/error.cuh"
#include <cmath>
namespace qwe::kernels {
// 分离每个head的Q/gate；布局从[n,h,2*d]变为两个[n,h,d]。
__global__ void split_gate_kernel(float *q, float *g, const float *x, int count, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        int base = (i / d) * 2 * d + i % d;
        q[i] = x[base];
        g[i] = x[base + d];
    }
}
// 原地门控，silu_gate=true用于线性注意力的输出归一化门控。
__global__ void gate35_kernel(float *x, const float *g, int count, bool silu_gate) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        float z = g[i];
        x[i] *= (silu_gate ? z : 1.f) / (1.f + expf(-z));
    }
}
// 纯文本位置三个轴相同，仅旋转head的前rd维。
__global__ void partial_rope_kernel(float *x, int count, int h, int d, int rd, float theta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    int row = i / (rd / 2), j = i % (rd / 2), pos = row / h;
    float sn, cs;
    sincosf(pos * powf(theta, -2.f * j / rd), &sn, &cs);
    float a = x[(size_t)row * d + j], b = x[(size_t)row * d + j + rd / 2];
    x[(size_t)row * d + j] = a * cs - b * sn;
    x[(size_t)row * d + j + rd / 2] = b * cs + a * sn;
}
__global__ void
conv35_kernel(float *out, const float *x, const float *w, int n, int width, int kernel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * width)
        return;
    int t = i / width, c = i % width;
    float sum = 0;
    for (int j = 0; j < kernel; ++j) {
        int pos = t + j - kernel + 1;
        if (pos >= 0)
            sum += x[(size_t)pos * width + c] * w[c * kernel + j];
    }
    out[i] = sum / (1.f + expf(-sum));
}
// 每个block一个head，thread对应V通道，各线程保留该列递推状态。
// 支持本首版检查过的K/V head_dim=128、相同head数量。
__global__ void delta35_kernel(float *out,
                               const float *x,
                               const float *a,
                               const float *b,
                               const float *A,
                               const float *dt,
                               int n,
                               int heads) {
    int head = blockIdx.x, j = threadIdx.x;
    float state[128];
    for (int i = 0; i < 128; ++i)
        state[i] = 0;
    __shared__ float q[128], k[128], decay, beta;
    int width = heads * 128;
    for (int t = 0; t < n; ++t) {
        if (j == 0) {
            float qs = 1e-6f, ks = 1e-6f;
            for (int i = 0; i < 128; ++i) {
                float qv = x[(size_t)t * 3 * width + head * 128 + i],
                      kv = x[(size_t)t * 3 * width + width + head * 128 + i];
                q[i] = qv;
                k[i] = kv;
                qs += qv * qv;
                ks += kv * kv;
            }
            float qscale = rsqrtf(qs) * rsqrtf(128.f), kscale = rsqrtf(ks);
            for (int i = 0; i < 128; ++i) {
                q[i] *= qscale;
                k[i] *= kscale;
            }
            float z = a[t * heads + head] + dt[head];
            float softplus = fmaxf(z, 0.f) + log1pf(expf(-fabsf(z)));
            decay = expf(-expf(A[head]) * softplus);
            beta = 1.f / (1.f + expf(-b[t * heads + head]));
        }
        __syncthreads();
        float memory = 0;
        for (int i = 0; i < 128; ++i) {
            state[i] *= decay;
            memory += state[i] * k[i];
        }
        float delta = (x[(size_t)t * 3 * width + 2 * width + head * 128 + j] - memory) * beta,
              result = 0;
        for (int i = 0; i < 128; ++i) {
            state[i] += k[i] * delta;
            result += state[i] * q[i];
        }
        out[(size_t)t * width + head * 128 + j] = result;
        __syncthreads();
    }
}
void split_gate35(float *q, float *g, const float *x, int count, int d) {
    split_gate_kernel<<<(count + 255) / 256, 256>>>(q, g, x, count, d);
    check(cudaGetLastError());
}
void gate35(float *x, const float *g, int count, bool silu) {
    gate35_kernel<<<(count + 255) / 256, 256>>>(x, g, count, silu);
    check(cudaGetLastError());
}
void partial_rope35(float *x, int n, int h, int d, int rd, float theta) {
    int count = n * h * (rd / 2);
    partial_rope_kernel<<<(count + 255) / 256, 256>>>(x, count, h, d, rd, theta);
    check(cudaGetLastError());
}
void conv35(float *out, const float *x, const float *w, int n, int width, int kernel) {
    conv35_kernel<<<(n * width + 255) / 256, 256>>>(out, x, w, n, width, kernel);
    check(cudaGetLastError());
}
void delta35(float *out,
             const float *x,
             const float *a,
             const float *b,
             const float *A,
             const float *dt,
             int n,
             int heads) {
    delta35_kernel<<<heads, 128>>>(out, x, a, b, A, dt, n, heads);
    check(cudaGetLastError());
}
} // namespace qwe::kernels
