#include "backend/cuda/device_tensor.cuh"
#include "backend/cuda/transfer.cuh"
#include <iostream>
int main() {
    using namespace qwe;
    int devices = 0;
    check(cudaGetDeviceCount(&devices));
    if (devices < 2)
        return 77;
    float input[1024], output[1024];
    for (int i = 0; i < 1024; ++i)
        input[i] = i + 0.25f;
    std::unique_ptr<DeviceTensor> source, target;
    {
        DeviceGuard guard(0);
        source = std::make_unique<DeviceTensor>(DTYPE_FP32, std::initializer_list<int64_t>{1024});
        check(cudaMemcpy(source->tensor.data, input, sizeof(input), cudaMemcpyHostToDevice));
    }
    {
        DeviceGuard guard(1);
        target = std::make_unique<DeviceTensor>(DTYPE_FP32, std::initializer_list<int64_t>{1024});
    }
    for (bool host : {false, false, true}) {
        transfer(target->tensor, source->tensor, host);
        DeviceGuard guard(1);
        check(cudaMemcpy(output, target->tensor.data, sizeof(output), cudaMemcpyDeviceToHost));
        int wrong = 0;
        for (int i = 0; i < 1024; ++i)
            if (input[i] != output[i])
                ++wrong;
        std::cout << "force_host=" << host << " mismatches=" << wrong << " first=" << output[0]
                  << " expected=" << input[0] << "\n";
        if (wrong)
            return 1;
    }
}
