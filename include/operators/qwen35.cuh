#pragma once
#include "core/tensor.h"
namespace qwe {
void split_gate35(Tensor &, Tensor &, const Tensor &);
void gate35(Tensor &, const Tensor &, bool silu);
void partial_rope35(Tensor &, int rotary_dim, float theta);
void conv35(Tensor &, const Tensor &, const Tensor &);
void delta35(
    Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &);
} // namespace qwe
