# 2026-11-17：TP 与推理能力扩展

## 范围与基础

在已有单卡 Qwen3、原生 tokenizer/chat 和两卡按层 PP 基础上开展本次工作。PP/P2P 前期实现来自其他工作，不覆盖、回滚这些改动。实际工程位于 p8_hzy:/home/hzy/project/QWECPP；本地工作目录并不是源码镜像。仓库已有大量未提交/未跟踪文件，不能将当前 git diff 全部视为本次新增内容。

## 本次功能

1. 两卡 TP：Q/K/V、Gate/Up 输出分片，O/Down 输入分片。每层 Attention/MLP 各交换局部输出并求和。支持设备反序，保留主机中转诊断。Embedding、归一化与词表头复制，最终词表投影仅 rank 0 执行。
2. --debug：退出时按轮向 stderr 输出加载、首 token、首文本、总耗时、实际输出 token 数、吞吐、TPOT；另有 TP 通信次数、逻辑字节、P2P/host 路径、源同步等待和传输调用墙钟耗时。EOS 检查前向也计入通信；诊断计时不等于纯 DMA 或物理带宽。
3. 输入历史滑动窗口：--sliding-window 默认关闭，删除最早的完整 user/assistant 轮次，为 max-new-tokens 预留空间；拒绝超限单问题时不污染历史。该策略偏保守，后续再优化。
4. 公共 WeightArchive Loader：索引 weight_map 到 Safetensors 分片的按需 mmap，校验索引、文件名及权重形状；兼容原单文件。新增 Qwen3-8B 支持，复用同一 Qwen3 前向。
5. 长上下文：默认从 256 提高到 2048，--max-context 允许 1..8192，并受模型位置上限约束。Attention 分数改用动态共享内存，线程循环覆盖历史位置；保持每 block 256 threads。单轮 max-new-tokens 仍为 1..256，默认 128。

## 代码位置

- src/models/qwen3.cu：TP 编排、分片上传、模型配置/上下文校验。
- include/loader/weight_archive.hpp：单文件/索引分片的 mmap 所有权。
- src/backend/cuda/transfer.cu：P2P/host 传输及可选诊断统计。
- tools/qwecpp_chat.cu：滑动历史、生成和每轮指标。
- kernels/attention/attention.cu：长序列参考 GQA kernel。
- tests/models/：前向、聊天、TP、分片索引、debug、滑动与长上下文测试。

## 数值验证记录

|场景|结果|
|---|---|
|0.6B TP，1/4/32/256 token，两种设备顺序|全 logits atol=3e-4、rtol=1e-4 通过；最大绝对误差约 1.9145e-4|
|8B 单卡对 Transformers FP32，1/4/8/32/256|atol=1e-3、rtol=1e-4 通过；最大绝对误差 8.6403e-4|
|8B PP 对单卡，1/4/32/256|逐位一致|
|8B TP 对单卡，1/4/32/256，正反设备|独立 atol=2e-3、rtol=1e-4；最大绝对误差 1.4191e-3，末位置 argmax 一致|
|0.6B 1024 token 末位置对 Transformers|单卡 1.9073e-5，TP 2.4319e-5 最大绝对误差|
|8B 512 token 末位置对 Transformers|单卡 3.6240e-5，TP 4.1962e-5 最大绝对误差|
|8192 token 均匀 GQA 算子|解析前缀均值检查通过|

原 0.6B 阈值没有放宽。8B 256-token 测试不满足原 3e-4 容差，因此显式使用独立阈值；未承诺所有长度/输入均有相同误差。长输入完整模型测试仅比较末位置，不能等同于所有位置全 logits 验证。TP 多轮聊天与 Transformers 贪心参考一致；滑动历史连续 12 轮、超限恢复、非法索引、debug 计数及字节数验证通过。CTest 4 项通过。

## 性能归档

固定历史基线：[2026-11-17 无 KV Cache](../benchmarks/baselines/2026-11-17-no-kv/README.md)，含逐次结果、stdout/stderr、环境、模型/源码哈希和源码压缩快照。另保留 docs/tp-chat-benchmark.json 的早期外部首字符测量；它采用重新分词计数，不能与内部精确 token 指标混用。

基线使用 0.6B、1 次预热、5 次重复、最大输出 32、上下文 256。短输入单卡/PP/TP 的 TTFT 中位数为 4.29/4.37/6.53 ms；TPOT 为 4.71/4.92/7.02 ms。长输入为 179 token；详情见原归档。未锁 GPU 时钟或隔离其他进程。首轮冷启动独立保留，不能直接与热轮比较。

record_baseline.py 已改为新建时间戳目录，指定已有目录会拒绝覆盖；显式 --max-context 256 保持旧口径。

## 结构限制和后续工作

- 无 KV Cache，每 token 重算完整上下文；尚无 NCCL、显式通信计算重叠或 PP+TP。
- TP 编排仍集中在模型文件且写死两卡，单卡/TP 前向有重复；后续应抽公共并行 Linear、ParallelContext、归约和可复用 Workspace。
- 所有权重转换成 FP32，8B 参数显存约 32 GB；两卡有复制权重，不能简单视为各占一半。
- 8192 是当前 kernel 支持上限，不代表已验证模型原生 40960 上下文；Attention 二次复杂度，长上下文交互仍慢。
- 滑动窗口后续优化暂缓。

## 本次整理

统一根 README 的现状、构建、CLI、诊断与验证；更新 architecture.md；保留有效测试、设计记录和性能快照。检查未发现仓库内无用中间文件，不删除 build、第三方依赖或其他工作产生的未提交代码。
