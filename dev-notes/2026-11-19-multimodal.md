# 2026-11-19：Qwen3.5-0.8B 静态图片后端

## 范围与结构

原生C++/CUDA图片+文本推理，保留已有文本服务；不以Python/Transformers代理推理。首版仅Qwen3.5-0.8B、单GPU、FP32，PNG/JPEG静态图片；无视频/音频/PDF、视觉KV/递推缓存、多卡图文。本记录描述后端/API阶段；网页上传随后接入，见2026-11-20记录。

runtime/image_input负责字节解码、尺寸约束、bicubic抗锯齿缩放、RGB归一化和patch打包；models/qwen3_5_vision组织12层视觉Transformer、二维位置插值/RoPE与patch merger；operators/vision_ops校验并调度kernels/attention/vision。语言模型接收视觉特征替换image占位embedding，MRoPE使用T/H/W交错位置。视觉权重首次使用加载并常驻；每请求每图片仅编码一次，生成期间复用特征；语言层仍完整上下文重算。没有引入跨请求图像缓存。

共享WeightStore新增显式storage_shape参数，以严格校验原始5维Conv3D权重后视作二维Linear加载；未放宽其他权重的形状检查，计算Tensor仍最多4维。图片使用STB Image固定commit源码和许可证，未增加运行时下载。

## 输入与资源边界

- PNG/JPEG每张最多8MiB；API整个会话最多4张，总压缩字节最多8MiB，JSON请求体最多16MiB。
- 原始尺寸每轴最多4096、总像素最多1600万、长宽比最多200。
- 32倍数智能缩放，65536..262144像素；最多1024视觉patch，经2x2 merger后最多256图片token/张。该上限是当前参考Attention的工程限制，低于原processor的最高像素配置。
- 接受data:image/png;base64或data:image/jpeg;base64内联URI；校验base64、MIME签名和解码。拒绝HTTP URL、本地文件路径和其他模态，服务不代用户请求外部资源。
- 图片token计入语言上下文；超限在视觉编码前拒绝。容量与有效长度分开处理。
- 还未处理JPEG EXIF自动旋转或全部色彩配置；测试JPEG为标准RGB样本，不能扩称所有编码器/图片均逐像素一致。

## API与诊断

/api/capabilities按当前模型返回vision=true/false，仍audio=false；给出图片数量/字节/预处理像素限额。/api/chat的content可以是字符串，或text与image_url类型的数组；仅user消息允许图片，支持历史中重复携带图片。

最终SSE metrics增加image_count、image_tokens、image_preprocess_ms、vision_ms。TTFT含预处理/首次视觉权重加载/视觉编码/语言prefill；prefill_ms仍只计首次语言模型调用，vision_ms独立计时。取消在视觉图片边界或token边界检查，不能抢占正在执行的CUDA调用。现有纯文本请求和原Qwen3服务行为保留。

## 验证证据

参考：本机Transformers显式PyTorch回退，FP32 eager；关闭matmul与cuDNN的TF32。最初仅关闭matmul TF32时，视觉特征最大差异约4.1e-2；关闭Conv3D参考的cuDNN TF32后吻合，没有放宽数值容差。

|输入|视觉patch|特征最大绝对误差|末位置logits最大绝对误差|
|---|---:|---:|---:|
|256×256 PNG|256|2.12e-5|4.98e-5|
|256×384 PNG|384|1.53e-5|4.98e-5|
|73×97 PNG（上采样）|280|3.27e-5|4.17e-5|
|640×800 PNG（受限下采样）|952|2.24e-4|1.31e-4|
|256×256 JPEG|256|6.32e-5|6.15e-5|
|两图组合|640|通过3e-4容差|1.15e-4|

预处理patch与参考atol=2e-7、rtol=1e-6通过；特征和末位置logits为atol=3e-4、rtol=1e-4，argmax一致。两图检查覆盖独立视觉注意力范围、特征拼接和多图片MRoPE。不是所有位置全logits验证，也未宣称所有真实图片达到同一误差。

独立审查发现视觉LayerNorm均值读取后复用共享数组缺少barrier；已修复并加多warp解析均值/方差测试。聚焦Racecheck为0hazards（0errors/0warnings），256×256图片整条前向Memcheck为0errors。修复后重跑上述图片数值回归通过。

API单图/双图SSE、最终指标、远程URL/畸形base64/MIME不符/图片过多/非支持模态拒绝，以及失败后文本请求恢复通过。原文本参考、多轮聊天、原Qwen3缓存/PD聊天、CUDA4项CTest和纯C2项CTest通过。原始证据见 [验证输出](evidence/2026-11-19-vision/)。

## 使用与后续

[图文API和原生CLI](../docs/multimodal.md)。后续补更多真实JPEG/EXIF/色彩与图片语义样例、再做视觉性能基线和增量状态缓存。本阶段统一交付检查见2026-11-20记录。
