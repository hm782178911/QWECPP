# KV Cache 开关性能对照（2026-11-18）

同一代码/二进制、FP32、max-context=512、max-new-tokens=32；每组每种输入预热一次后测三次，取中位数。全部配置实际输出32 token，缓存开关的聊天 stdout 完全一致。short=20 输入 token，long=178。每轮 clear 并重新 prefill，不跨轮缓存。硬件 RTX PRO 6000 Blackwell Max-Q 双卡；未隔离负载或锁定时钟。

|模型|模式|输入|无缓存 TPOT ms|缓存 TPOT ms|decode 加速比|无缓存 TTFT ms|缓存 TTFT ms|
|---|---|---|---:|---:|---:|---:|---:|
|Qwen3-0.6B|single|short|4.660|3.240|1.44x|4.357|4.392|
|Qwen3-0.6B|single|long|15.620|4.797|3.26x|13.721|13.764|
|Qwen3-0.6B|tp|short|6.922|4.904|1.41x|6.505|6.658|
|Qwen3-0.6B|tp|long|14.914|5.844|2.55x|14.087|14.442|
|Qwen3-8B|single|short|38.227|20.611|1.85x|34.811|34.456|
|Qwen3-8B|single|long|109.797|21.800|5.04x|103.830|98.804|
|Qwen3-8B|tp|short|26.743|16.436|1.63x|23.163|23.481|
|Qwen3-8B|tp|long|67.457|18.439|3.66x|64.284|64.674|

kv_cache_bytes 是容量分配总和：0.6B 112 MiB、8B 144 MiB（单卡和TP总量相同），不含权重/工作区。warmup包含首次缓存分配；TTFT 是热态数据。性能不包含最终 RMSNorm 单行优化，该项未实施。PP做了数值验证，本组性能仅测单卡/TP。详见逐次 results.json、完整 stdout/stderr、environment.json、source-hashes.json 和源码快照。
