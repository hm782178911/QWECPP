#include "backend/cuda/error.cuh"
#include <cmath>
namespace qwe::kernels {
__global__ void bias_kernel(float *x, const float *b, int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n * d)
        x[i] += b[i % d];
}
__global__ void
ln_kernel(float *out, const float *x, const float *w, const float *b, int d, float eps) {
    __shared__ float sums[256];
    int row = blockIdx.x, t = threadIdx.x;
    float mean = 0;
    for (int j = t; j < d; j += 256)
        mean += x[(size_t)row * d + j];
    sums[t] = mean;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (t < stride)
            sums[t] += sums[t + stride];
        __syncthreads();
    }
    mean = sums[0] / d;
    // 所有warp读完均值后，才能复用共享数组存放方差。
    __syncthreads();
    float var = 0;
    for (int j = t; j < d; j += 256) {
        float z = x[(size_t)row * d + j] - mean;
        var += z * z;
    }
    sums[t] = var;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (t < stride)
            sums[t] += sums[t + stride];
        __syncthreads();
    }
    float scale = rsqrtf(sums[0] / d + eps);
    for (int j = t; j < d; j += 256)
        out[(size_t)row * d + j] = (x[(size_t)row * d + j] - mean) * scale * w[j] + b[j];
}
__global__ void gelu_kernel(float *x, int n, bool approx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float z = x[i];
        x[i] = approx ? .5f * z * (1 + tanhf(.7978845608028654f * (z + .044715f * z * z * z)))
                      : .5f * z * (1 + erff(z * .7071067811865475f));
    }
}
__device__ void coord(int token, int gw, int &y, int &x) {
    int block = token / 4, sub = token % 4;
    y = block / (gw / 2) * 2 + sub / 2;
    x = block % (gw / 2) * 2 + sub % 2;
}
__global__ void pos_kernel(float *x, const float *b, const float *table, int n, int gh, int gw) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 768)
        return;
    int row = i / 768, c = i % 768, y, z;
    coord(row, gw, y, z);
    float fy = y * (47.f / (gh - 1)), fx = z * (47.f / (gw - 1));
    int y0 = (int)fy, x0 = (int)fx, y1 = min(47, y0 + 1), x1 = min(47, x0 + 1);
    float wy = fy - y0, wx = fx - x0;
    x[i] += b[c] + table[(y0 * 48 + x0) * 768 + c] * (1 - wy) * (1 - wx) +
            table[(y0 * 48 + x1) * 768 + c] * (1 - wy) * wx +
            table[(y1 * 48 + x0) * 768 + c] * wy * (1 - wx) +
            table[(y1 * 48 + x1) * 768 + c] * wy * wx;
}
__global__ void vrope_kernel(float *x, int n, int gw) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 2 * 12 * 32)
        return;
    int j = i % 32, head = (i / 32) % 12, part = (i / (32 * 12)) % 2, row = i / (32 * 12 * 2), y, z;
    coord(row, gw, y, z);
    int axis = j / 16;
    float angle = (axis ? z : y) * powf(10000.f, -2.f * (j % 16) / 32);
    float sn, cs;
    sincosf(angle, &sn, &cs);
    size_t idx = (size_t)row * 2304 + part * 768 + head * 64 + j;
    float a = x[idx], b = x[idx + 32];
    x[idx] = a * cs - b * sn;
    x[idx + 32] = b * cs + a * sn;
}
__global__ void vattn_kernel(float *out, const float *x, int n) {
    extern __shared__ float scores[];
    int row = blockIdx.x / 12, head = blockIdx.x % 12, t = threadIdx.x;
    for (int k = t; k < n; k += 256) {
        float sum = 0;
        for (int j = 0; j < 64; ++j)
            sum +=
                x[(size_t)row * 2304 + head * 64 + j] * x[(size_t)k * 2304 + 768 + head * 64 + j];
        scores[k] = sum * .125f;
    }
    __syncthreads();
    if (!t) {
        float maximum = scores[0];
        for (int k = 1; k < n; ++k)
            maximum = fmaxf(maximum, scores[k]);
        float sum = 0;
        for (int k = 0; k < n; ++k) {
            scores[k] = expf(scores[k] - maximum);
            sum += scores[k];
        }
        for (int k = 0; k < n; ++k)
            scores[k] /= sum;
    }
    __syncthreads();
    if (t < 64) {
        float result = 0;
        for (int k = 0; k < n; ++k)
            result += scores[k] * x[(size_t)k * 2304 + 1536 + head * 64 + t];
        out[(size_t)row * 768 + head * 64 + t] = result;
    }
}
__global__ void mrope_kernel(float *x, const int *positions, int n, int heads, int d, float theta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * heads * 32)
        return;
    int j = i % 32, row = i / 32, pos = positions[(row / heads) * 3 + j % 3];
    float sn, cs;
    sincosf(pos * powf(theta, -2.f * j / 64), &sn, &cs);
    size_t index = (size_t)row * d + j;
    float a = x[index], b = x[index + 32];
    x[index] = a * cs - b * sn;
    x[index + 32] = b * cs + a * sn;
}
void bias_rows(float *x, const float *b, int n, int d) {
    bias_kernel<<<(n * d + 255) / 256, 256>>>(x, b, n, d);
    check(cudaGetLastError());
}
void layer_norm(
    float *out, const float *x, const float *w, const float *b, int n, int d, float eps) {
    ln_kernel<<<n, 256>>>(out, x, w, b, d, eps);
    check(cudaGetLastError());
}
void gelu(float *x, int n, bool approx) {
    gelu_kernel<<<(n + 255) / 256, 256>>>(x, n, approx);
    check(cudaGetLastError());
}
void vision_position(float *x, const float *b, const float *t, int gh, int gw) {
    pos_kernel<<<(gh * gw * 768 + 255) / 256, 256>>>(x, b, t, gh * gw, gh, gw);
    check(cudaGetLastError());
}
void vision_rope(float *x, int n, int gw) {
    vrope_kernel<<<(n * 2 * 12 * 32 + 255) / 256, 256>>>(x, n, gw);
    check(cudaGetLastError());
}
void vision_attention(float *out, const float *x, int n) {
    vattn_kernel<<<n * 12, 256, n * sizeof(float)>>>(out, x, n);
    check(cudaGetLastError());
}
void multimodal_rope(float *x, const int *p, int n, int h, int d, float theta) {
    mrope_kernel<<<(n * h * 32 + 255) / 256, 256>>>(x, p, n, h, d, theta);
    check(cudaGetLastError());
}
} // namespace qwe::kernels
