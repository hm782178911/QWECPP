# 原生服务与网页

## 启动

需 CUDA + CHAT 构建（已有 PCRE2/ICU）。HTTP 使用随仓库提供的 cpp-httplib v0.20.1（MIT），构建不联网；Python不进入推理服务链路。

```bash
cmake --build build -j8
./build/bin/qwecpp_server --model /home/hzy/models/Qwen/Qwen3.5-0.8B --host 0.0.0.0 --port 8080 --web-root web
# Qwen3 可选择单卡/PP/TP和缓存；Qwen3.5首版不支持这些多卡/缓存组合。
./build/bin/qwecpp_server --model /home/hzy/models/Qwen/Qwen3-0.6B --parallel tp --devices 0,1 --kv-cache --web-root web
```

默认监听127.0.0.1:8080，--host 0.0.0.0 用于可信局域网访问。当前没有账户/TLS/多用户隔离，不应直接作为公网服务。命令前台运行，Ctrl-C退出；后台启动由操作员管理，尚未安装systemd服务。

## 网页

独立实现web/index.html、style.css、app.js，参考llama.cpp交互布局，未复制其前端代码。会话侧栏、新建/切换/删除、浏览器本地持久化、中文输入、逐段流式回复、安全Markdown子集（标题/列表/粗体/代码）、复制、停止生成和每轮指标。Markdown通过textContent构建DOM，不执行模型HTML/JS，不加载外部CDN。

设置可调整每轮最大输出1..256；当前模型、上下文、KV开关和并行模式是服务启动配置，网页显示但不动态重载。Enter发送，Shift-Enter换行，输入法组词时不发送。会话及图片存储在浏览器 IndexedDB（自动读取旧 localStorage 文本会话），不同设备/浏览器不共享；服务不持久存对话，收到每轮完整历史后reset并prefill。

Qwen3.5能力查询已返回vision=true，图文API已接通；原Qwen3仍vision=false。audio仍false。网页点击输入框下的“＋”选择图片，或将图片拖入输入区域；发送前可预览、移除。只选图片时默认询问“请描述这张图片。”。每个会话历史及待发送图片累计最多4张、压缩数据总计8MiB；超限需移除附件或新建对话。图片尺寸及损坏检查在发送前完成，后端再次独立校验。原Qwen3禁用上传。保存失败会明确提示，图片不上传到第三方存储。图文请求格式和限额见multimodal.md。首版不是OpenAI协议完整实现。

## 接口

- GET /api/capabilities：模型名称、stream/vision/audio、max_context、kv_cache、parallel。
- POST /api/chat：JSON {id,messages:[{role,content}],max_new_tokens}，接受字符串content；Qwen3.5另支持text/image_url内容数组，末消息必须为user。成功返回SSE `data: JSON`，事件type为delta/done/error。
- delta含text；done含finish_reason（eos/length/context/cancelled）和metrics；error含error。指标为input/output_tokens、ttft_ms、tpot_ms、total_ms、output_tok_s、prefill_ms、decode_ms、kv_cache_bytes。
- POST /api/cancel：JSON {id}，按请求ID取消，在模型下一token边界生效，不能中断当前CUDA调用。

一次仅执行一个生成请求；忙时HTTP409，不静默排队。不合法参数/非文本附件/超长上下文返回400；请求体限制16MiB，图片总压缩量限制8MiB。默认同源，非同源Origin拒绝。SSE断连会在下一边界终止并释放互斥锁；模型常驻，不通过子进程解析CLI输出。

计时从原生分词前开始，TTFT到首个非EOS token选择完成；total包含分词/前向/流写入，不是浏览器实际屏幕首字延迟。TPOT为首末非EOS token的平均间隔，不足两个为-1；decode_ms在无缓存模式下仍是后续全量重算时间。结束EOS判断计入总前向时间，不计output_tokens。

## 验证

```bash
python3 tests/server/test_server.py
```

脚本启动独立随机端口服务，验证能力、SSE文本/最终指标、静态页面、非法输入/附件拒绝、忙响应、取消后恢复和外来Origin拒绝，结束回收自建进程。当前测试默认本机Qwen3.5-0.8B权重路径。浏览器已验证模型就绪、中文回复、Markdown、指标、会话和停止入口；移动布局有响应式CSS，尚未做全部设备覆盖。

图文后端详见 [多模态说明](multimodal.md)。

前端纯逻辑测试（开发检查，需要 Node.js；服务运行不依赖 Node）：`node tests/web/test_attachments.mjs`。
