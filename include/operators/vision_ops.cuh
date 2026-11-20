#pragma once
#include "core/tensor.h"
namespace qwe {
// 各输入为连续CUDA FP32；函数原地/输出修改及维度由接口校验。
void bias_rows(Tensor &x, const Tensor &bias);
void layer_norm(
    Tensor &out, const Tensor &x, const Tensor &weight, const Tensor &bias, float eps = 1e-6f);
void gelu(Tensor &x, bool approximate);
void vision_position(Tensor &x, const Tensor &bias, const Tensor &table, int gh, int gw);
void vision_rope(Tensor &qkv, int gh, int gw);
void vision_attention(Tensor &out, const Tensor &qkv);
void multimodal_rope(Tensor &x, const Tensor &positions, float theta);
} // namespace qwe
