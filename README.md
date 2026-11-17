# QWECPP

以 C/CUDA 为核心的原生 Qwen3 推理引擎。当前支持 batch=1、FP32 计算、原生中文多轮聊天，以及单卡、两卡 PP 和两卡 TP。Python 仅用于独立验证和性能测量，不进入推理链路。

支持本机 Qwen3-0.6B 和 Qwen3-8B。BF16/F16/F32 权重加载后统一转换为 FP32；同时支持单个 `model.safetensors` 和 `model.safetensors.index.json` 索引分片。模型目录须有 `config.json`，聊天另需 tokenizer 文件。

默认上下文 **2048 token**，可配置至 **8192**，受模型位置上限约束。当前**没有 KV Cache**，每个新 token 重算完整上下文。仅支持未缩放 RoPE、无 attention bias、无模型 Sliding Window Attention 的 Qwen3 SiLU 文本架构。聊天历史滑动窗口是另一个独立选项。

## 构建

基础依赖：CMake、C/C++ 编译器。CUDA 路径需要 CUDA Toolkit/cuBLAS；聊天还需要 PCRE2 和 ICU。默认 CUDA 关闭。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
ctest --test-dir build --output-on-failure
```

本机 GPU 为 RTX PRO 6000 Blackwell Max-Q，当前 CUDA/chat 构建示例：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DQWECPP_ENABLE_CUDA=ON -DQWECPP_ENABLE_CHAT=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DPCRE2_INCLUDE_DIR=/home/hzy/miniforge3/envs/isaac_py312/include \
  -DPCRE2_LIBRARY=/home/hzy/miniforge3/envs/isaac_py312/lib/libpcre2-8.so \
  -DICU_ROOT=/home/hzy/miniforge3/envs/isaac_py312
cmake --build build -j8
```

其他设备请调整架构和库路径。`QWECPP_ENABLE_CHAT=OFF` 可只构建 CUDA 前向；`QWECPP_ENABLE_CUDA=OFF` 保留纯 C 功能。本机复用环境库，构建后需保留相应动态库路径。

## 聊天与前向

```bash
# 单卡（省略 --devices 使用当前 CUDA 设备）
./build/bin/qwecpp_chat --model /home/hzy/models/Qwen/Qwen3-0.6B --devices 0

# 8B 两卡 TP
./build/bin/qwecpp_chat --model /home/hzy/models/Qwen/Qwen3-8B \
  --parallel tp --devices 0,1 --max-context 4096 --max-new-tokens 128 --debug

# 两卡 PP，第二阶段从第 14 层开始（0 起编号）
./build/bin/qwecpp_chat --model /home/hzy/models/Qwen/Qwen3-0.6B \
  --parallel pp --devices 0,1 --split-layer 14

# token ID 前向，输出所有位置的 FP32 logits
./build/bin/qwecpp_forward --model /home/hzy/models/Qwen/Qwen3-8B \
  --tokens 9707,11,1879,0 --parallel tp --devices 0,1 \
  --max-context 2048 --output /tmp/qwen3-logits.bin
```

聊天默认非思考、贪心解码，输入一行即生成；`/clear` 清空历史，`/exit`、`/quit` 或 EOF 退出。不执行任意 Jinja 模板，不支持工具调用或多模态。权重在模型实例生命周期内常驻。

|参数|默认值|含义|
|---|---|---|
|`--devices`|当前设备|一个或两个不同设备；支持 `1,0` 反序|
|`--parallel pp\|tp`|pp|两卡时默认按层 PP；TP 必须显式选择并指定两卡|
|`--split-layer N`|均分层|仅两卡 PP；边界须位于模型层范围内部|
|`--max-context N`|2048|1..8192，再取模型位置上限；模板、历史、输入与输出共用|
|`--max-new-tokens N`|128|聊天每轮最多输出 1..256 token，遇 EOS 可提前结束|
|`--sliding-window`|关闭|聊天历史超限时移除最早完整轮次，为输出预留空间|
|`--debug`|关闭|聊天正常退出时向 stderr 打印性能统计|
|`--pp-host-transfer`|关闭|PP/TP 强制固定页主机中转，用于诊断|

未开启滑动窗口时，超限输入被拒绝，不污染历史；开启后按 `min(max-new-tokens, context_limit-1)` 预留空间，当前问题单独仍放不下则拒绝。删除历史会提示，模型会忘记被移除内容。滑动窗口策略后续再优化。

`qwecpp_forward --output` 可省略，输出文件是 row-major `[token_count,vocab_size]` 原生 FP32，无文件头。聊天仅投影最后位置，再由 CPU argmax 选 token。可复用 C++ API 见 `include/models/qwen3.cuh`。

## PP 与 TP

PP 按连续层切两阶段，首阶段持有 Embedding，末阶段持有最终 RMSNorm/LM Head。同一请求顺序执行，没有多请求/微批流水线重叠。

TP 将 Q/K/V、Gate/Up 按输出维分片，O/Down 按输入维分片；Attention/MLP 局部输出各求和一次。每卡持有完整残差状态，Embedding/归一化/词表头复制，最终仅 rank 0 做词表投影。Q heads、KV heads 和 intermediate 必须能被 2 整除；不支持与 split-layer 混用。

两者复用 P2P 优先的 transfer，保留首次完整性校验及主机回退。历史 peer copy 数据异常已通过一次性 `iommu=pt` 启动对照恢复正确性；未改 BIOS 或永久启动配置，重启后需重新确认。仓库保留完整性校验与回退。本次 TP 检查全部走 P2P。当前无 NCCL、PP+TP 组合或显式通信计算重叠，不承诺单请求加速。

## Debug 指标与性能基线

统计在正常退出时统一打印，`/clear` 不清统计。首轮可能包含 CUDA/cuBLAS 惰性初始化，性能比较应预热并固定输入/输出长度。

|字段|口径|
|---|---|
|`load_ms`|tokenizer 与模型初始化耗时|
|`input_tokens` / `output_tokens`|输入长度 / 实际非 EOS 生成数量|
|`dropped_turns`|本轮移除的历史轮数|
|`ttft_ms`|从分词前到首个非 EOS token 选择完成|
|`first_text_ms`|到首段完整 UTF-8 文本输出完成|
|`total_ms` / `output_tok_s`|整轮耗时（含终止 EOS 检查）/ 输出 token 数除以整轮秒数|
|`tpot_ms`|首末非 EOS token 间平均间隔；不足两 token 为 -1|
|`tp_transfer_calls` / `tp_transfer_bytes`|单向传输次数 / 双向逻辑 payload 总字节|
|`tp_p2p_calls` / `tp_host_calls`|最终完成传输的路径次数|
|`tp_transfer_ms`|transfer 调用墙钟时间总和，含同步|
|`tp_source_wait_ms`|其中等待源设备同步的时间|
|`tp_transfer_rest_ms`|扣除源等待后，仍含路由、锁、分配、复制、目标同步和首次校验|
|`tp_transfer_pct`|transfer 调用耗时占整轮耗时比例|

无非 EOS 输出时首 token/首文本延迟为 -1。每次 TP 前向有 `4×层数` 次单向传输，逻辑字节为 `4×层数×上下文长度×hidden_size×sizeof(float)`，包含终止判断的前向。字节不含重试/校验/主机多段实际流量；耗时不能作为纯 DMA 时间或物理带宽，比例也不是纯通信造成的性能损失。单卡/PP 的 TP 字段为零。诊断会引入计时开销。

保留的 [无 KV Cache 基线](benchmarks/baselines/2026-11-17-no-kv/README.md) 含逐次数据、环境、源码哈希/快照。其上下文为 256；复现需显式 `--max-context 256`。`tests/models/record_baseline.py [新目录]` 新建独立归档，拒绝覆盖已有目录。早期 [外部首字符测量](docs/tp-chat-benchmark.json) 与内部 token 计时口径不同，不直接混用。

## 验证

```bash
ctest --test-dir build --output-on-failure
PY=/home/hzy/miniforge3/envs/llm_py312/bin/python
$PY tests/models/test_forward.py build/bin/qwecpp_forward /home/hzy/models/Qwen/Qwen3-0.6B
$PY tests/models/test_tp.py
$PY tests/models/test_chat_debug.py
$PY tests/models/test_tp_debug.py
$PY tests/models/test_sliding_window.py
$PY tests/models/test_long_context.py
$PY tests/models/test_forward.py build/bin/qwecpp_forward /home/hzy/models/Qwen/Qwen3-8B 0.001
$PY tests/models/test_sharded_model.py build/bin/qwecpp_forward /home/hzy/models/Qwen/Qwen3-8B
$PY tests/models/test_long_reference.py /home/hzy/models/Qwen/Qwen3-8B 512
```

另有 `test_pp.py`、`test_tokenizer.py`、`test_chat.py`，参数为可执行文件和模型目录；聊天测试可追加运行选项。部分专项脚本使用本机路径及双卡条件。

0.6B TP 全 logits 使用 atol=3e-4、rtol=1e-4；8B 对 Transformers FP32 使用 atol=1e-3，TP 对单卡使用 atol=2e-3，rtol 均为 1e-4。原 0.6B 容差未放宽。已验证范围、实际误差和局限见 [本次开发记录](dev-notes/2026-11-17-tp-and-runtime.md)。8192 算子测试不代表所有模型在该长度的全 logits 都已验证。

## 项目结构与后续工作

`include/core`：非拥有型 Tensor 元数据；`DeviceTensor`/`HostTensor` 管理数据生命周期。`src/operators` 校验并调度；`kernels` 放 CUDA 实现；`src/models` 组织前向；`src/runtime`/`tools` 负责 tokenizer 与聊天；`include/loader`/`src/loader` 负责权重读取。

- [架构说明](docs/architecture.md)
- [开发记录目录](dev-notes/README.md)
- [今日跨对话总览：PP、P2P 修复与 TP](dev-notes/2026-11-17-overview.md)
- [P2P/主机中转带宽实验源码与原始日志](benchmarks/p2p/2026-11-17/)
- [PP 设计](docs/pp-design.md)、[TP 初始设计](docs/tp-design.md)

下一步重点：抽取公共并行算子/Context、复用工作区、KV Cache、分离 prefill/decode、通信与算子优化，逐项沿用固定基线量化。滑动策略优化暂缓。

Safetensors Reader 支持 F32/F16/BF16/I8/I32/I64、最多 4 维（含标量/空张量），数据借用 mmap，关闭后失效；Qwen3.5 的高维权重不在支持范围。JSON 使用随仓库提供的 cJSON 1.7.19，保留 MIT 许可证；header 上限 100 MB，拒绝超过 double 精确整数范围的维度/偏移。`qwecpp_inspect` 仍检查单个 Safetensors（目录默认 model.safetensors），并非多分片汇总工具。
