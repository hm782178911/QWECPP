# Runtime data and CUDA architecture

- `include/core/tensor.h`: shared C, non-owning Tensor metadata: pointer, dtype, rank/shape/stride, sizes, CPU/CUDA location and device ID. CPU metadata is independent of device data. Tensor does not allocate or free memory.
- `include/core/tensor.hpp`: C++ metadata-only contiguous reshape/last-row helpers and FP32 HostTensor ownership for numerical output. No numerical std::vector container is used by the forward path.
- `include/backend/cuda/device_tensor.cuh`: cudaMalloc/cudaFree ownership around the same Tensor. Weights and activations use DeviceTensor; views share allocations and must not outlive owners.
- `include/operators/tensor_ops.cuh`, `src/operators/tensor_ops.cu`: Tensor-based operator interfaces, dtype/shape/layout/device validation, cuBLAS dispatch and kernel selection. Operators currently require the default/legacy CUDA stream; Linear rejects a cuBLAS handle bound to another stream. Element counts are capped below INT_MAX to keep launch arithmetic in range. Initial kernels require contiguous memory and mostly FP32; IDs are INT32 and conversion accepts BF16/F16/F32.
- `kernels/`: private raw-pointer CUDA kernels and launchers, grouped by core conversion, embedding, normalization, positional encoding, attention and activation. Model code never directly launches a kernel.
- `src/models/qwen3.cu`: config/weight mapping, persistent model resources and Transformer composition through Tensor operator APIs. Output is a Tensor-backed HostTensor. Vectors remain appropriate for token IDs, layer lists and conversational history.
- `src/runtime/tokenizer.cpp`, `tools/qwecpp_chat.cu`: native tokenizer and multi-turn greedy CLI, using the persistent model.

Validation: CTest covers C Tensor, loader and Tensor-backed CUDA operator checks. Independent `tests/models/test_forward.py`, `test_tokenizer.py` and `test_chat.py` compare real-model outputs against Transformers. CUDA memcheck verifies operator and persistent-chat allocation behavior. CUDA remains optional, chat dependencies are independently optional. Default context is 2048 tokens, configurable with --max-context up to 8192 and capped by model positions; no KV cache.

## 两卡分层阶段

`Qwen3Options` 配置一个或两个 GPU 及层边界。Stage 只加载自己负责的连续层，Embedding 位于首阶段，最终归一化/词表投影位于末阶段。层内算子与 kernel 不感知 PP。`DeviceGuard` 恢复当前设备；DeviceTensor 与 cuBLAS 所有者在资源所属设备释放。`backend/cuda/transfer` 管理隐藏状态传输、P2P 首次数据校验、方向缓存和固定页内存回退。

当前为单请求顺序阶段执行，没有多请求流水线调度或 KV Cache。测试含两卡正反设备顺序、切层边界、自动/强制主机传输、与单卡及 Transformers 对照。

## TP、权重与聊天状态

两卡 TP 为每张卡保留完整隐藏状态，局部 head/MLP 通道经 O/Down 投影后交换并求和。当前模型文件负责双卡编排，公共 transfer 层负责传输和可选统计；未来应抽取并行 Linear/Context，避免复制模型前向。通信基于每设备默认 stream 和显式同步，不等于持续双卡并发。

WeightArchive 按索引选择并延迟 mmap 分片，保留单文件兼容；WeightStore 拥有转换后的 GPU 权重。工作区目前每次前向重新分配。聊天窗口只管理输入历史，不是模型 Sliding Window Attention；--debug 记录每轮及 TP 逻辑通信数据。详细变更见 [开发记录](../dev-notes/2026-11-17-tp-and-runtime.md)。
