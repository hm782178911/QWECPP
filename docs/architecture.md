# Runtime data and CUDA architecture

- `include/core/tensor.h`: shared C, non-owning Tensor metadata: pointer, dtype, rank/shape/stride, sizes, CPU/CUDA location and device ID. CPU metadata is independent of device data. Tensor does not allocate or free memory.
- `include/core/tensor.hpp`: C++ metadata-only contiguous reshape/last-row helpers and FP32 HostTensor ownership for numerical output. No numerical std::vector container is used by the forward path.
- `include/backend/cuda/device_tensor.cuh`: cudaMalloc/cudaFree ownership around the same Tensor. Weights and activations use DeviceTensor; views share allocations and must not outlive owners.
- `include/operators/tensor_ops.cuh`, `src/operators/tensor_ops.cu`: Tensor-based operator interfaces, dtype/shape/layout/device validation, cuBLAS dispatch and kernel selection. Operators currently require the default/legacy CUDA stream; Linear rejects a cuBLAS handle bound to another stream. Element counts are capped below INT_MAX to keep launch arithmetic in range. Initial kernels require contiguous memory and mostly FP32; IDs are INT32 and conversion accepts BF16/F16/F32.
- `kernels/`: private raw-pointer CUDA kernels and launchers, grouped by core conversion, embedding, normalization, positional encoding, attention and activation. Model code never directly launches a kernel.
- `src/models/qwen3.cu`: config/weight mapping, persistent model resources and Transformer composition through Tensor operator APIs. Output is a Tensor-backed HostTensor. Vectors remain appropriate for token IDs, layer lists and conversational history.
- `src/runtime/tokenizer.cpp`, `tools/qwecpp_chat.cu`: native tokenizer and multi-turn greedy CLI, using the persistent model.

Validation: CTest covers C Tensor, loader and Tensor-backed CUDA operator checks. Independent `tests/models/test_forward.py`, `test_tokenizer.py` and `test_chat.py` compare real-model outputs against Transformers. CUDA memcheck verifies operator and persistent-chat allocation behavior. CUDA remains optional, chat dependencies are independently optional. Default context is 2048 tokens, configurable with --max-context up to 8192 and capped by model positions; optional per-layer KV cache supports suffix decode.

## 两卡分层阶段

`Qwen3Options` 配置一个或两个 GPU 及层边界。Stage 只加载自己负责的连续层，Embedding 位于首阶段，最终归一化/词表投影位于末阶段。层内算子与 kernel 不感知 PP。`DeviceGuard` 恢复当前设备；DeviceTensor 与 cuBLAS 所有者在资源所属设备释放。`backend/cuda/transfer` 管理隐藏状态传输、P2P 首次数据校验、方向缓存和固定页内存回退。

当前为单请求顺序阶段执行，没有多请求流水线调度。KV Cache 已作为可选功能加入，详见 2026-11-18 开发记录。测试含两卡正反设备顺序、切层边界、自动/强制主机传输、与单卡及 Transformers 对照。

## TP、权重与聊天状态

两卡 TP 为每张卡保留完整隐藏状态，局部 head/MLP 通道经 O/Down 投影后交换并求和。当前模型文件负责双卡编排，公共 transfer 层负责传输和可选统计；未来应抽取并行 Linear/Context，避免复制模型前向。通信基于每设备默认 stream 和显式同步，不等于持续双卡并发。

WeightArchive 按索引选择并延迟 mmap 分片，保留单文件兼容；WeightStore 拥有转换后的 GPU 权重。工作区目前每次前向重新分配。聊天窗口只管理输入历史，不是模型 Sliding Window Attention；--debug 记录每轮及 TP 逻辑通信数据。详细变更见 [开发记录](../dev-notes/2026-11-17-tp-and-runtime.md)。

## KV Cache（2026-11-18）

KVCache 在 backend/cuda 持有每层 FP32 K/V；Stage 负责追加并调用缓存 Attention，公共算子校验后缀与完整前缀形状。Qwen3 用精确 token 前缀判定复用，否则重建；每轮聊天 reset 逻辑长度，显存容量保留。TP 只存本卡 KV heads，PP 只存本阶段层。RoPE 接收绝对位置 offset，缓存保存归一化/旋转后的 K。关闭开关和完整 logits 输出保持原计算路径。


## 同机双卡 Prefill / Decode

PD 在同一 `Qwen3` 实例中建立两组 Stage，第一组执行 Prefill 并产生首 token。第一次增量前向通过公共 CUDA transfer 逐层传输有效 FP32 K/V，形状为 `[prefix_length, kvheads, head_dim]`。两组分别拥有缓存，按上下文容量分配。目的端完成全部交接后才执行后缀计算，后续增量前向留在 Decode 组。reset、非追加输入和前向执行异常使逻辑前缀失效，分配保留供复用。完整位置前向在 Prefill 组计算，并使缓存失效。

四卡 PD 的两组使用同一并行布局。PP 按层区间、TP 按 KV head 分片交接；每个 Prefill rank 只向对应的 Decode rank 复制。TP 组内通信与 KV 交接各自记录统计。
