# 2026-11-19：原生HTTP服务与聊天网页

## 范围

基于用户对llama.cpp网页的参考要求，新增原生qwecpp_server与独立静态web/，部署在p8_hzy，模型常驻并直接调用TextModel/Tokenizer。无Python推理代理，无数据库或云端部署。保留此前尚未提交的Qwen3.5实现；本次服务器依赖该TextModel分派。

## 实现

cpp-httplib v0.20.1固定版本源码和MIT许可证随仓库保存。API提供capabilities/chat SSE/cancel，单生成互斥锁、409忙反馈、输入/Origin/请求大小校验、断连回收。线程按模型设备选择CUDA上下文。网页提供会话侧栏、浏览器持久化、常见Markdown子集、流式中文输出、输出长度设置和性能指标；模型输出只通过文本DOM渲染。文本模型禁用附件；Qwen3.5图片上传在后续阶段接入，见2026-11-20记录。参考功能点而非复制llama.cpp的全部产品能力。

## 已验证

- API集成：随机端口进程，能力与静态资源、SSE、最终统计、非法输入/附件、并发409、取消、恢复、跨Origin拒绝通过。
- 浏览器：模型就绪、中文消息、流式回复/Markdown、指标和停止入口；会话写入本地存储。代码块/复制提供实现，未覆盖所有剪贴板权限与移动设备。
- node --check网页JS通过，CUDA构建及4项CTest通过。独立审阅未发现阻塞问题。

## 边界

最初为文本服务，后续补齐静态图文；Qwen3.5仍不支持KV缓存或并行。设置仅动态调整输出token上限，其他参数由启动配置决定。取消在token边界检查，不能抢占一次CUDA forward。无认证和TLS，限可信本地/局域网；未实现多用户调度、动态模型重载或OpenAI完整兼容。基础HTML/JS不支持执行模型生成的HTML，也没有远端脚本依赖。

当前开发预览：p8_hzy上8080端口，日志/tmp/qwecpp-web-server.log，PID记录/tmp/qwecpp-web-server.pid。使用后台进程而非systemd，不保证重启自动恢复。进程状态可能随操作变化，接续开发需实查。

使用和API细节见 [网页服务文档](../docs/web-server.md)。统一交付检查见2026-11-20记录。
