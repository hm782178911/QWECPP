#include "backend/cuda/device_tensor.cuh"
#include "operators/tensor_ops.cuh"
#include <cmath>
#include <iostream>
#include <vector>
#include <climits>
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            std::cerr << "failed: " << #x << "\n";                                                 \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)
template <class F> bool rejects(F f) {
    try {
        f();
        return false;
    } catch (const std::exception &) {
        return true;
    }
}

int main() {
    using namespace qwe;
    DeviceTensor x(DTYPE_FP32, {2, 2}), w(DTYPE_FP32, {2}), out(DTYPE_FP32, {2, 2});
    float values[] = {1, 2, 3, 4}, weights[] = {1, 1};
    check(cudaMemcpy(x.tensor.data, values, sizeof(values), cudaMemcpyHostToDevice));
    check(cudaMemcpy(w.tensor.data, weights, sizeof(weights), cudaMemcpyHostToDevice));
    CHECK(x.tensor.device == TENSOR_CUDA && x.tensor.shape[0] == 2 && x.tensor.nbytes == 16);
    norm(out.tensor, x.tensor, w.tensor, 1e-6f);
    float actual[4];
    check(cudaMemcpy(actual, out.tensor.data, sizeof(actual), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 4; ++i)
        CHECK(std::fabs(actual[i] -
                        values[i] / std::sqrt((values[i / 2 * 2] * values[i / 2 * 2] +
                                               values[i / 2 * 2 + 1] * values[i / 2 * 2 + 1]) /
                                                  2 +
                                              1e-6f)) < 1e-5f);
    auto view = reshape(x.tensor, {1, 2, 2});
    CHECK(view.data == x.tensor.data && view.shape[2] == 2 && view.stride[0] == 4);
    CHECK(rejects([&] { reshape(x.tensor, {3, 2}); }));
    Tensor bad = x.tensor;
    bad.device = TENSOR_CPU;
    CHECK(rejects([&] { norm(out.tensor, bad, w.tensor, 1e-6f); }));
    bad = x.tensor;
    bad.ndim = 5;
    CHECK(rejects([&] { reshape(bad, {2, 2}); }));
    bad = x.tensor;
    bad.device_id = x.tensor.device_id + 1;
    CHECK(rejects([&] { add(out.tensor, bad); }));
    bad = x.tensor;
    bad.stride[0] = 1;
    CHECK(rejects([&] { norm(out.tensor, bad, w.tensor, 1e-6f); }));
    DeviceTensor wrong(DTYPE_FP16, {2, 2});
    CHECK(rejects([&] { add(out.tensor, wrong.tensor); }));
    CHECK(rejects([&] { norm(out.tensor, x.tensor, w.tensor, -1); }));
    Tensor huge = make_tensor(DTYPE_FP32, {INT_MAX});
    huge.data = x.tensor.data;
    huge.device = TENSOR_CUDA;
    huge.device_id = x.tensor.device_id;
    CHECK(rejects([&] { add(huge, huge); }));
    DeviceTensor mat(DTYPE_FP32, {2, 2});
    check(cudaMemcpy(mat.tensor.data, values, sizeof(values), cudaMemcpyHostToDevice));
    cublasHandle_t handle;
    check(cublasCreate(&handle));
    cudaStream_t stream;
    check(cudaStreamCreate(&stream));
    check(cublasSetStream(handle, stream));
    CHECK(rejects([&] { linear(handle, out.tensor, x.tensor, mat.tensor); }));
    check(cublasSetStream(handle, nullptr));
    check(cudaStreamDestroy(stream));
    linear(handle, out.tensor, x.tensor, mat.tensor);
    check(cublasDestroy(handle));
    check(cudaMemcpy(actual, out.tensor.data, sizeof(actual), cudaMemcpyDeviceToHost));
    CHECK(actual[0] == 5 && actual[1] == 11 && actual[2] == 11 && actual[3] == 25);
    DeviceTensor raw(DTYPE_BF16, {2});
    unsigned short bf[] = {0x3f80, 0x4000};
    check(cudaMemcpy(raw.tensor.data, bf, sizeof(bf), cudaMemcpyHostToDevice));
    convert(w.tensor, raw.tensor);
    check(cudaMemcpy(actual, w.tensor.data, 8, cudaMemcpyDeviceToHost));
    CHECK(actual[0] == 1 && actual[1] == 2);
    // Uniform attention has an analytic prefix mean, including beyond thread count.
    {
        const int n=8192;
        DeviceTensor q(DTYPE_FP32,{n,2,2}), k(DTYPE_FP32,{n,1,2}),
            v(DTYPE_FP32,{n,1,2}), result(DTYPE_FP32,{n,2,2});
        check(cudaMemset(q.tensor.data,0,q.tensor.nbytes));
        check(cudaMemset(k.tensor.data,0,k.tensor.nbytes));
        std::vector<float> input(n*2), output(n*4);
        for(int i=0;i<n;++i) { input[i*2]=(float)(i%2); input[i*2+1]=1; }
        check(cudaMemcpy(v.tensor.data,input.data(),v.tensor.nbytes,cudaMemcpyHostToDevice));
        attention(result.tensor,q.tensor,k.tensor,v.tensor);
        check(cudaMemcpy(output.data(),result.tensor.data,result.tensor.nbytes,cudaMemcpyDeviceToHost));
        for(int i=0;i<n;++i) for(int h=0;h<2;++h) {
            CHECK(std::fabs(output[i*4+h*2]-(float)((i+1)/2)/(i+1))<0.001f);
            CHECK(std::fabs(output[i*4+h*2+1]-1)<0.001f);
        }
    }
    HostTensor host({2, 2});
    CHECK(host.tensor.device == TENSOR_CPU && host.tensor.numel == 4);
    std::cout << "Tensor-backed CUDA operators passed\n";
}
