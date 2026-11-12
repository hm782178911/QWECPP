# QWECPP


# QWECPP

**Native LLM Inference Engine**

Build from the Metal Up.

QWECPP 是一个以 C/CUDA 为核心实现的 LLM 推理引擎项目。从 Qwen3-0.6B 开始，逐步实现模型权重加载、GPU 算子、Transformer 推理、KV Cache 以及多 GPU 推理优化。

项目的核心目标是理解并实现 LLM 推理的底层基础设施，而不是简单封装现有推理框架。

## 1. Project Goals

### 核心目标

- 使用 C / CUDA C++ 实现 LLM 推理引擎。
- 尽量自主实现关键 GPU 算子和推理流程。
- 不依赖 PyTorch、Transformers、vLLM 作为核心推理运行时。
- 从单 GPU 推理开始，逐步扩展到多 GPU。
- 持续进行正确性验证和性能优化。

### 开发路线

以 Qwen3-0.6B 作为第一个目标模型，先完成端到端推理，再逐步扩展模型和硬件支持。

| 阶段 | 目标 |
|---|---|
| Phase 1 | 基础数据结构与权重加载 |
| Phase 2 | CUDA Runtime 与 GPU 内存管理 |
| Phase 3 | 基础 CUDA 算子 |
| Phase 4 | Qwen3-0.6B 单卡推理 |
| Phase 5 | Prefill / Decode 性能优化 |
| Phase 6 | 多 GPU 推理 |
| Phase 7 | 其他模型与架构扩展 |

## 2. Technology Stack

| 技术 | 用途 |
|---|---|
| C11 | 核心数据结构、权重加载、Runtime |
| CUDA C++ | GPU Kernel 与设备端计算 |
| CMake | 项目构建 |
| Safetensors | 模型权重读取 |
| cuBLAS | GEMM 性能参考及可选后端 |
| NCCL | 后续多 GPU 通信 |
| Python | Reference、模型检查、测试数据生成与性能分析 |

Python 仅作为开发辅助工具，不进入核心推理执行链路。

## 3. Project Structure

```text
QWECPP/
├── CMakeLists.txt
├── README.md
│
├── include/
│   ├── core/
│   │   ├── dtype.h
│   │   ├── tensor.h
│   │   └── status.h
│   │
│   ├── loader/
│   │   ├── safetensors.h
│   │   └── weight_loader.h
│   │
│   ├── backend/
│   │   └── cuda/
│   │
│   ├── operators/
│   │
│   └── runtime/
│
├── src/
│   ├── core/
│   │   ├── dtype.c
│   │   └── tensor.c
│   │
│   ├── loader/
│   │   ├── safetensors.c
│   │   └── weight_loader.c
│   │
│   ├── backend/
│   │   └── cuda/
│   │
│   ├── operators/
│   │
│   └── runtime/
│
├── kernels/
│   ├── matmul/
│   ├── normalization/
│   ├── positional/
│   ├── attention/
│   └── activation/
│
├── models/
│   └── qwen3/
│
├── tests/
│   ├── unit/
│   ├── loader/
│   ├── operators/
│   └── models/
│
├── benchmarks/
│
├── tools/
│
└── docs/
```

目录职责：

- `core/`：基础数据类型和 Tensor。
- `loader/`：模型文件解析、权重读取与管理。
- `backend/`：CUDA 设备、Stream、内存等硬件资源管理。
- `operators/`：算子接口及调度。
- `kernels/`：具体 CUDA Kernel 实现。
- `models/`：模型结构、模型计算流程及模型专属权重映射。
- `runtime/`：推理执行、KV Cache、采样和生成流程。
- `tests/`：正确性测试。
- `benchmarks/`：性能测试。

## 4. Current Progress

<badge color="success">Phase 1 — In Progress</badge>

### 已完成

| 模块 | 文件 | 状态 |
|---|---|---|
| Data Type | `include/core/dtype.h` | 已完成 |
| Data Type | `src/core/dtype.c` | 已完成 |
| Tensor | `include/core/tensor.h` | 已完成 |
| Tensor | `src/core/tensor.c` | 已完成 |
| CMake | `CMakeLists.txt` | 已完成 |
| Unit Test | `tests/unit/test_tensor.c` | 测试通过 |

### 当前已有功能

**Data Type**

支持以下数据类型：

- FP32
- FP16
- BF16
- INT8
- INT32
- INT64

提供：

```c
size_t dtype_size(DType dtype);
const char* dtype_name(DType dtype);
```

**Tensor**

当前 Tensor 负责管理元数据，不负责数据内存分配。

```c
typedef struct {
    void* data;
    DType dtype;
    int ndim;
    int64_t shape[4];
    int64_t stride[4];
    size_t numel;
    size_t nbytes;
} Tensor;
```

提供：

```c
int tensor_init(Tensor* tensor, DType dtype,
                int ndim, const int64_t* shape);

void tensor_destroy(Tensor* tensor);

size_t tensor_numel(const Tensor* tensor);

size_t tensor_nbytes(const Tensor* tensor);
```

当前单元测试使用 FP32、Shape `[2, 3, 4]`，验证结果：

| 属性 | 结果 |
|---|---|
| ndim | 3 |
| shape | `[2, 3, 4]` |
| stride | `[12, 4, 1]` |
| numel | 24 |
| nbytes | 96 |

### 构建验证

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

当前测试结果：`100% tests passed`。

## 5. Next Task: Safetensors Reader

<badge color="info">Next</badge>

下一步实现 Safetensors Reader。

这是 QWECPP 第一次正式接触模型权重文件，也是后续 Qwen3 权重加载的基础。

### 5.1 文件格式

Safetensors 文件结构：

```text
┌────────────────────────────┐
│ Header Length (8 bytes)    │
├────────────────────────────┤
│ JSON Header                │
│                            │
│ Tensor Name                │
│ Data Type                  │
│ Shape                      │
│ Data Offsets               │
├────────────────────────────┤
│ Tensor Data                │
│                            │
│ Tensor 1                   │
│ Tensor 2                   │
│ Tensor 3                   │
│ ...                        │
└────────────────────────────┘
```

需要注意：

- Header Length 使用 8-byte little-endian 编码。
- Header 是 JSON 格式。
- `data_offsets` 相对于 Tensor Data 区域起点。
- 文件可能包含 `__metadata__`。
- 不同 Tensor 的数据连续存储。

### 5.2 数据结构

初步设计：

```c
typedef struct {
    char* name;
    DType dtype;

    int ndim;
    int64_t shape[4];

    uint64_t data_offset;
    uint64_t data_size;
} SafeTensorEntry;

typedef struct {
    int fd;

    void* mapped_data;
    size_t file_size;

    uint64_t data_start;
    uint64_t data_size;

    SafeTensorEntry* entries;
    size_t num_entries;
} SafeTensorFile;
```

Safetensors Reader 采用 `mmap` 读取文件，避免额外复制整份模型权重。

Entry 中的 `data` 指针属于非拥有型引用，生命周期依赖对应的 `SafeTensorFile`。

### 5.3 第一批 API

```c
int safetensors_open(
    SafeTensorFile* file,
    const char* path
);

void safetensors_close(
    SafeTensorFile* file
);

size_t safetensors_count(
    const SafeTensorFile* file
);

const SafeTensorEntry* safetensors_find(
    const SafeTensorFile* file,
    const char* name
);

const void* safetensors_get_data(
    const SafeTensorFile* file,
    const SafeTensorEntry* entry
);

void safetensors_inspect(
    const SafeTensorFile* file
);
```

### 5.4 实现要求

- [ ] 打开 Safetensors 文件。
- [ ] 使用 mmap 映射文件。
- [ ] 读取并解析 Header Length。
- [ ] 解析 JSON Header。
- [ ] 获取 Tensor 名称、dtype、shape、offset。
- [ ] 正确计算 Tensor 的 numel 和 nbytes。
- [ ] 校验数据区间是否合法。
- [ ] 校验 shape 与 dtype 对应的数据长度。
- [ ] 支持通过名称查找 Tensor。
- [ ] 返回 mmap 中对应 Tensor 的数据指针。
- [ ] 正确释放 mmap、文件描述符和 Entry 内存。
- [ ] 编写独立单元测试。

JSON 解析优先使用成熟的轻量级 C JSON Parser，不手写完整 JSON 解析器。

第一版先支持单个 `.safetensors` 文件，后续再支持多 shard 和 `model.safetensors.index.json`。

### 5.5 验收目标

使用本地 Qwen3-0.6B checkpoint，执行：

```bash
./build/qwecpp_inspect \
    --model /path/to/Qwen3-0.6B
```

能够输出：

```text
Model: Qwen3-0.6B

Tensor Count: xxx

model.embed_tokens.weight
    dtype: BF16
    shape: [151936, 1024]

model.layers.0.self_attn.q_proj.weight
    dtype: BF16
    shape: [2048, 1024]

...

Total Parameters: xxx
Total Weight Size: xxx
```

实际 dtype、shape 和 Tensor 数量以本地 checkpoint 为准。

## 6. Subsequent Development Plan

### Phase 1：基础设施与权重加载

- [x] Data Type
- [x] Tensor
- [x] CMake
- [x] Tensor Unit Test
- [ ] Safetensors Reader
- [ ] WeightStore
- [ ] Qwen3 Config Parser
- [ ] Qwen3 Weight Mapping
- [ ] Host Weight Validation

### Phase 2：CUDA Backend

- [ ] CUDA Context
- [ ] CUDA Device Management
- [ ] CUDA Stream
- [ ] GPU Memory Allocation
- [ ] Host-to-Device Transfer
- [ ] Device Weight Management

### Phase 3：基础算子

- [ ] Linear / GEMM
- [ ] RMSNorm
- [ ] RoPE
- [ ] Softmax
- [ ] Attention
- [ ] SwiGLU
- [ ] Embedding
- [ ] Argmax / Sampling

先实现正确的基础版本，再逐步优化 Kernel。

### Phase 4：Qwen3-0.6B 推理

- [ ] Qwen3 Config
- [ ] Qwen3 Model Structure
- [ ] Transformer Layer
- [ ] GQA
- [ ] Q/K RMSNorm
- [ ] MLP
- [ ] KV Cache
- [ ] Prefill
- [ ] Decode
- [ ] Tokenizer
- [ ] Text Generation

### Phase 5：性能优化

- [ ] GEMM Tiling
- [ ] Tensor Core
- [ ] FlashAttention
- [ ] Kernel Fusion
- [ ] Decode GEMV Optimization
- [ ] CUDA Graph
- [ ] KV Cache Optimization
- [ ] Quantization

### Phase 6：Multi-GPU

- [ ] Multi-GPU Device Management
- [ ] Model Replication
- [ ] Tensor Parallel
- [ ] NCCL Communication
- [ ] P2P Transfer
- [ ] Multi-GPU Benchmark

## 7. Development Principles

1. **Correctness First**：先保证数值正确，再考虑性能。
2. **Build Incrementally**：每完成一个模块，都要能够独立编译和测试。
3. **Explicit Memory Ownership**：明确 Host、Device、Owned、Borrowed 内存的生命周期。
4. **Separate Model and Operator**：模型负责组织计算流程，算子负责计算接口，Kernel 负责具体实现。
5. **Separate Prefill and Decode**：两条推理路径分别设计和优化。
6. **Benchmark Everything**：关键算子和推理流程都建立性能基准。
7. **Avoid Premature Abstraction**：先跑通 Qwen3-0.6B，再根据实际需求抽象通用架构。

## 8. Build

### Requirements

- GCC / G++
- CMake
- CUDA Toolkit（后续 CUDA 阶段需要）

### Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

cmake --build build -j$(nproc)
```

### Test

```bash
ctest --test-dir build --output-on-failure
```

---

**Current Milestone**

完成基础 Tensor 系统，正在进入 Safetensors 权重读取阶段。

**Next Objective**

实现一个独立、可靠的 Safetensors Reader，能够读取并检查真实 Qwen3-0.6B checkpoint 的全部 Tensor 元数据。
