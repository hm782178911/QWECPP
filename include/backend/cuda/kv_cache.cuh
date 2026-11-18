#pragma once
#include "backend/cuda/device_tensor.cuh"
namespace qwe {
class KVCache {
    DeviceTensor keys,values;
    int capacity,heads,dim;
public:
    KVCache(int capacity,int heads,int dim):keys(DTYPE_FP32,{capacity,heads,dim}),
        values(DTYPE_FP32,{capacity,heads,dim}),capacity(capacity),heads(heads),dim(dim) {}
    size_t bytes() const { return keys.tensor.nbytes+values.tensor.nbytes; }
    void append(const Tensor &k,const Tensor &v,int offset) {
        if(offset<0 || offset+k.shape[0]>capacity) throw std::runtime_error("KV cache capacity exceeded");
        size_t start=(size_t)offset*heads*dim*sizeof(float);
        check(cudaMemcpy((char*)keys.tensor.data+start,k.data,k.nbytes,cudaMemcpyDeviceToDevice));
        check(cudaMemcpy((char*)values.tensor.data+start,v.data,v.nbytes,cudaMemcpyDeviceToDevice));
    }
    Tensor key_view(int total) const { auto t=make_tensor(DTYPE_FP32,{total,heads,dim});t.data=keys.tensor.data;t.device=TENSOR_CUDA;t.device_id=keys.tensor.device_id;return t; }
    Tensor value_view(int total) const { auto t=key_view(total);t.data=values.tensor.data;return t; }
};
}
