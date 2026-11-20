# QWECPP

**Native LLM Inference Engine**

*Build from the Metal Up.*

QWECPP 是一个以 C / CUDA 为核心的 LLM 推理引擎。从 Qwen3-0.6B 出发，逐步实现权重加载、GPU 算子、端到端推理与多 GPU 执行，目前已支持 Qwen3-8B，以及 Qwen3.5-0.8B 单卡图文推理和网页聊天。

项目的核心目标是理解并实现 LLM 推理的底层基础设施。Python 只用于参考验证和性能分析，不进入推理执行链路。

[项目目标](#1-project-goals) · [当前进展](#2-current-progress) · [快速开始](#3-quick-start) · [项目结构](#4-project-structure) · [技术栈](#5-technology-stack) · [验证与性能](#6-validation--benchmarks) · [开发路线](#7-roadmap) · [文档](#8-documentation)

---

## 1. Project Goals

- 使用 C / CUDA C++ 实现原生 LLM 推理。
- 自主实现关键 GPU 算子，以 cuBLAS 完成当前 Linear / GEMM。
- 从单 GPU 扩展到 PP、TP 等多 GPU 执行方式。
- 每项优化都先验证正确性，再用固定基线做定量分析。

## 2. Current Progress

| 模块 | 当前状态 |
| --- | --- |
| Tensor 与内存 | CPU / CUDA 元数据、设备编号、显存和 CPU 输出所有权 |
| 权重加载 | Safetensors 单文件、索引多分片；BF16 / FP16 / FP32 → FP32 |
| 模型 | Qwen3-0.6B、Qwen3-8B；Qwen3.5-0.8B 单卡图文，batch = 1 |
| GPU 算子 | Embedding、RMSNorm、RoPE、因果 GQA、SwiGLU、残差、Linear |
| 原生聊天 | Tokenizer、多轮历史、流式 UTF-8 输出、贪心解码 |
| 多 GPU | 两卡 PP 按层执行、两卡 TP 按 head / MLP 通道切分、同机两卡 PD |
| 上下文 | 默认 **2048 token**，`--max-context` 可配置至 **8192** |
| 网页服务 | 原生 HTTP / SSE、图文上传、浏览器会话保存 |
| KV Cache（Qwen3） | 可选 `--kv-cache`，单卡 / PP / TP；默认关闭 |
| 历史管理 | 可选滑动窗口，默认关闭 |
| 性能诊断 | 首 token / 首文本延迟、TPOT、吞吐、TP 通信统计 |

**当前边界（以下缓存与多卡能力针对 Qwen3）：** 默认不启用 KV Cache；可通过 `--kv-cache` 在每轮生成内复用历史 K/V。每轮仍重新 prefill；PP 暂无多请求流水线重叠，TP 暂无 NCCL 或显式通信计算重叠。不支持 PP + TP 混合，也不承诺单请求加速。

Qwen3 支持范围为 SiLU 文本架构：未缩放 RoPE、无 attention bias、无模型 Sliding Window Attention。聊天历史裁剪与模型的窗口注意力是不同功能。

## 3. Quick Start

### 3.1 构建

基础依赖为 CMake、C/C++ 编译器。GPU 推理需要 CUDA Toolkit / cuBLAS；聊天另需 PCRE2 和 ICU。

纯 C 基础模块：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
ctest --test-dir build --output-on-failure
```

<details>
<summary>本机 CUDA + 原生聊天构建配置</summary>

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

本机 GPU 为 RTX PRO 6000 Blackwell Max-Q。其他环境请调整 CUDA 架构和依赖路径；运行时需能找到相应动态库。CUDA 默认关闭；`QWECPP_ENABLE_CHAT=OFF` 可只构建 GPU 前向。

</details>

### 3.2 开始聊天

**单卡 Qwen3-0.6B：**

```bash
./build/bin/qwecpp_chat --model /home/hzy/models/Qwen/Qwen3-0.6B
```

**两卡 TP，运行 Qwen3-8B：**

```bash
./build/bin/qwecpp_chat \
  --model /home/hzy/models/Qwen/Qwen3-8B \
  --parallel tp --devices 0,1 --max-context 4096 --kv-cache --debug
```

**两卡 PP：**

```bash
./build/bin/qwecpp_chat \
  --model /home/hzy/models/Qwen/Qwen3-0.6B \
  --parallel pp --devices 0,1 --split-layer 14
```

输入一行后回车即可生成回复。`/clear` 清空历史，`/exit` 或 EOF 退出。默认非思考、贪心解码；该终端入口仅处理文本，不支持工具调用。图片使用下方网页服务或图文 CLI。

### Qwen3.5-0.8B（单卡图文）

```bash
./build/bin/qwecpp_chat --model /home/hzy/models/Qwen/Qwen3.5-0.8B --devices 0 --max-new-tokens 32 --debug
./build/bin/qwecpp_forward --model /home/hzy/models/Qwen/Qwen3.5-0.8B --tokens 1,2,3,4
```

按 config.model_type 自动选择独立 qwen3_5 实现。支持 Gated DeltaNet / 门控完整注意力混合层、部分 RoPE、原生文本分词与多轮聊天。支持已验证0.8B配置，FP32完整上下文重算；图片后端已接通，**不支持视频/音频、Qwen3.5 KV/递推状态缓存、PP/TP/PD**，这些组合明确拒绝。现有Qwen3能力不变。文件Reader元数据扩展至8维以读取含视觉权重的检查点，但计算Tensor仍最多4维，视觉权重按需加载，仅文本请求不加载视觉编码器。数值参考显式使用Transformers PyTorch回退路径；不保证与其可选融合kernel逐位一致。

### 3.3 常用参数

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `--devices 0,1` | 当前 CUDA 设备 | 指定一张或两张不同 GPU |
| `--parallel pp` / `--parallel tp` | pp | 两卡执行方式；TP 需显式选择 |
| `--max-context 4096` | 2048 | 输入、模板、历史和回复共用的上下文上限，最高 8192 |
| `--max-new-tokens 128` | 128 | 每轮最多生成的 token 数，范围 1..256 |
| `--sliding-window` | 关闭 | 空间不足时移除最早完整对话轮次 |
| `--kv-cache` | 关闭 | 每轮首次 prefill，后续仅计算新增 token；缓存本地 KV heads |
| `--debug` | 关闭 | 退出时打印每轮性能与 TP 通信指标 |

开启滑动窗口会为输出预留空间，超长的单个问题仍会被拒绝。模型原生位置上限可能高于 8192，但当前 kernel 支持上限为 8192。

更多选项、token ID 前向和输出文件格式见 [命令行参考](docs/usage-reference.md)。

### 可选 KV Cache（Qwen3）

追加 `--kv-cache` 可启用 FP32 K/V 缓存。首次前向执行 prefill；后续对精确前缀只处理新增 token，RoPE 使用绝对位置。TP 缓存本卡 KV heads，PP 缓存本阶段层。聊天每轮重新 prefill，不跨对话轮次复用；clear/滑动历史后的输入重新建立缓存。缓存按 max-context 预留容量、按层首次使用时分配，reset 使逻辑前缀失效但保留显存；关闭模型时释放。完整位置输出 forward(last_only=false) 保持不缓存。

--debug 附带 kv_cache（开关）、kv_cache_bytes（所有 GPU 上已分配缓存总字节，不含权重/工作区）、prefill_ms（本轮首次模型调用墙钟时间）、decode_ms（后续调用累计时间，含终止 EOS 检查）。关闭缓存时后续调用仍是完整重算，decode_ms 只表示生成阶段计时。首次缓存分配开销计入 prefill；对比应预热并使用相同开关之外的参数。

数值测试：`build/bin/test_kv_cache MODEL_DIR [single|pp|tp] [测试上下文上限]`；聊天对照：`python3 tests/models/test_kv_chat.py`。详见 [KV Cache 开发记录](dev-notes/2026-11-18-kv-cache.md)。

### 同机双卡 PD

```bash
./build/bin/qwecpp_chat --model /home/hzy/models/Qwen/Qwen3-0.6B \
  --parallel pd --devices 0,1 --debug
```

设备顺序为 Prefill、Decode。PD 自动启用 KV Cache，两卡各加载一份完整权重，每轮首次增量前向前交接有效 K/V。可用 `--pp-host-transfer` 强制固定页主机中转。PD 要求两张不同的 GPU，不能指定 `--split-layer`。

四卡可用 `--parallel pd-pp --devices 4,5,6,7` 或 `--parallel pd-tp --devices 4,5,6,7`。前两张用于 Prefill，后两张用于 Decode。PP 两侧共用切层边界，可用 `--split-layer` 指定；TP 两侧按相同的 rank 切分。两侧须使用相同布局，按 rank 交接缓存分片。组内通信和 KV 交接都复用 P2P 校验及主机中转。当前为同一进程内的串行单请求执行。

`--debug` 中 `kv_transfer_ms` 为传输调用累计墙钟时间，含同步、路由和首次校验，`kv_transfer_bytes` 为逻辑 K/V 字节数。PD 的 `decode_ms` 扣除传输调用时间，仍含首次接收缓存分配。首 token 计时在交接之前，TPOT 包含首个 token 间隔内的交接。

## 4. Project Structure

```text
QWECPP/
├── include/                # 公共接口
│   ├── core/               # DType、Tensor 与 CPU 输出所有权
│   ├── backend/cuda/       # CUDA 设备、显存与跨卡传输
│   ├── loader/             # Safetensors / 分片权重接口
│   ├── models/             # Qwen3 / Qwen3.5 模型接口
│   ├── operators/          # Tensor 算子接口
│   └── runtime/            # Tokenizer、设备选项
├── src/                    # 对应模块的实现
├── kernels/                # CUDA kernel，按运算类型组织
├── tools/                  # chat、forward、image、server 等入口
├── web/                    # 原生服务的静态聊天网页
├── tests/                  # 单元、算子与模型参考测试
├── benchmarks/             # 模型基线与独立通信实验
├── dev-notes/              # 按日期整理的开发记录
├── docs/                   # 架构、设计与使用参考
├── third_party/            # cJSON、cpp-httplib、stb 源码与许可证
└── CMakeLists.txt
```

模型层组织计算顺序，算子层校验 Tensor 并调度，kernel 层执行 GPU 运算。Tensor 描述数据，DeviceTensor / HostTensor 管理数据内存。详细职责见 [架构说明](docs/architecture.md)。

## 5. Technology Stack

| 技术 | 用途 |
| --- | --- |
| C11 | 基础数据结构、单文件 Safetensors Reader |
| C++17 / CUDA C++ | 模型、资源管理、Tokenizer 与 GPU kernel |
| cuBLAS | 当前 Linear / GEMM 后端 |
| CUDA P2P / pinned memory | 跨卡传输及回退路径 |
| PCRE2 / ICU | 原生分词的正则与 Unicode 处理 |
| CMake / CTest | 构建与基础测试 |
| Python / Transformers | 独立参考、回归与性能分析 |

JSON 使用随仓库提供的 cJSON 1.7.19，保留 MIT 许可证。NCCL 尚未接入。

## 6. Validation & Benchmarks

### 正确性验证

```bash
ctest --test-dir build --output-on-failure

# 本机双卡 TP 对单卡回归
/home/hzy/miniforge3/envs/llm_py312/bin/python tests/models/test_tp.py
```

验证覆盖模型 logits、Tokenizer、多轮聊天、PP / TP、通信、滑动窗口和长上下文。0.6B 与 8B 采用独立数值容差；8192-token 算子测试不等于所有模型在该长度都做过全 logits 验证。

完整命令及容差见 [验证参考](docs/usage-reference.md#验证)，实测误差见 [开发记录](dev-notes/2026-11-17-tp-and-runtime.md)。

### 性能记录

`--debug` 在退出时向 stderr 输出 TTFT、首文本延迟、TPOT、token/s 和 TP 通信计数/耗时。通信计时含同步等待，不能当作纯链路带宽；首轮冷启动应与预热后的结果分开。

- [KV Cache 开关对照](benchmarks/kv-cache/20261118-104036/README.md)：同一版本、同输入，预热后重复测量。
- [无 KV Cache 模型基线](benchmarks/baselines/2026-11-17-no-kv/README.md)：逐次数据、环境、源码快照；旧上下文口径为 256。
- [P2P / 主机中转实验](benchmarks/p2p/2026-11-17/)：独立测试源码和日志。
- [指标口径](docs/usage-reference.md#debug-指标与性能基线)：计时边界与逻辑字节定义。

本机 P2P 曾因 IOMMU 地址转换出现数据错误，一次性 `iommu=pt` 启动对照后恢复。未永久配置，重启后需重新确认；引擎保留完整性校验和主机回退。

## 7. Roadmap

| 阶段 | 目标 | 进展 |
| --- | --- | --- |
| Phase 1 | 基础 Tensor、权重读取 | 已实现，含多分片 |
| Phase 2 | CUDA 后端与内存管理 | 基础实现完成 |
| Phase 3 | 核心 GPU 算子 | 正确性参考实现完成，性能待优化 |
| Phase 4 | Qwen3 端到端推理 | 0.6B / 8B、原生聊天可用 |
| Phase 5 | 多 GPU 执行 | 两卡 PP / TP 基础实现完成 |
| Phase 6 | 推理性能优化 | KV Cache 基础实现完成，继续优化 prefill / decode 与通信 |
| Phase 7 | 更多架构与执行能力 | 后续扩展 |

近期重点是抽取公共并行算子与 ParallelContext、复用工作区，逐项量化 KV Cache、算子和通信优化的收益。滑动窗口策略后续再优化。

## 8. Documentation

- [开发记录索引](dev-notes/README.md)
- [2026-11-18 可选 KV Cache](dev-notes/2026-11-18-kv-cache.md)
- [2026-11-17 跨对话更新总览](dev-notes/2026-11-17-overview.md)
- [TP、诊断、8B 与长上下文详细记录](dev-notes/2026-11-17-tp-and-runtime.md)
- [架构说明](docs/architecture.md)
- [命令行与诊断参考](docs/usage-reference.md)
- [PP 设计](docs/pp-design.md) / [TP 初始设计](docs/tp-design.md)

新增记录：[Qwen3.5-0.8B](dev-notes/2026-11-19-qwen35.md) · [静态图片](dev-notes/2026-11-19-multimodal.md) · [上传与交付检查](dev-notes/2026-11-20-web-upload.md)。

## 网页聊天服务

```bash
./build/bin/qwecpp_server --model /home/hzy/models/Qwen/Qwen3.5-0.8B --host 0.0.0.0 --port 8080 --web-root web
```

打开服务器8080端口即可使用。参考llama.cpp的会话侧栏、流式聊天、设置与统计，直接调用原生引擎。支持浏览器本地会话和停止生成，支持 PNG/JPEG 选择或拖放上传、附件预览与移除、图文问答；图片随会话保存在浏览器 IndexedDB。每个会话累计最多 4 张 / 8 MiB。默认仅监听127.0.0.1；启动示例开放可信局域网，无认证/TLS，不直接面向公网。

[网页服务/API说明](docs/web-server.md) · [开发记录](dev-notes/2026-11-19-web-server.md)

图片后端：[使用与边界](docs/multimodal.md) · [图文验证记录](dev-notes/2026-11-19-multimodal.md)。
