# Qwen3.5 图片＋文本后端

## 启动服务

```bash
cmake --build build -j8
./build/bin/qwecpp_server --model /home/hzy/models/Qwen/Qwen3.5-0.8B --host 0.0.0.0 --port 8080 --web-root web
```

当前仅静态PNG/JPEG、Qwen3.5-0.8B单卡FP32；不使用Python推理。不支持音频/视频、多卡或该模型的KV/递推缓存。原Qwen3文本服务可继续使用原开关，图片请求会明确拒绝。

## 请求

POST /api/chat，沿用SSE返回协议。content为有序数组：

```json
{
  "id": "image-example",
  "messages": [{
    "role": "user",
    "content": [
      {"type":"image_url","image_url":{"url":"data:image/png;base64,实际图片的base64数据"}},
      {"type":"text","text":"描述这张图片。"}
    ]
  }],
  "max_new_tokens": 32
}
```

URL只接受内联data URI，不会抓取远程URL或打开本地路径。PNG用image/png，JPEG用image/jpeg，必须与实际签名一致。整个会话最多4张、总压缩字节8MiB，整个请求16MiB；每轴原始尺寸最多4096且总像素最多1600万。图像缩放到32倍数、65536..262144像素，最多1024patch/256图片token。图片token与历史/文本共同占用max-context，太长会拒绝而非默默截断。

SSE的done.metrics附带image_count、image_tokens、image_preprocess_ms、vision_ms。视觉编码每请求每图一次，特征复用于后续token；语言层仍全量重算。TTFT包括视觉链路，prefill_ms只计语言prefill。取消无法抢占一次正在执行的GPU算子。网页已支持选择/拖放 PNG、JPEG，发送前预览/移除；历史图片会随下一轮请求重新发送，限额按完整会话累计。详见 [网页说明](web-server.md)。

## 原生检查工具

```bash
./build/bin/qwecpp_image --model /home/hzy/models/Qwen/Qwen3.5-0.8B --image /path/image.png --prompt '描述这张图片。'
# 可加 --image2 第二张图；调试输出可选 --patches/--features/--output/--metadata FILE。
```

工具只做一次图文前向，打印下一个token ID；不是多模态聊天CLI。logits输出为最后位置FP32，features是合并后的视觉embedding，metadata保存展开后的token ID和网格。不要把预分配容量当有效图片长度。

## 验证

```bash
PY=/home/hzy/miniforge3/envs/llm_py312/bin/python
$PY tests/models/test_qwen35_vision.py
$PY tests/server/test_vision.py
ctest --test-dir build --output-on-failure
/usr/local/cuda/bin/compute-sanitizer --tool racecheck build/bin/test_cuda_tensor vision-norm
```

参考使用同一像素预算，FP32且同时关闭matmul/cuDNN TF32，语言DeltaNet显式选官方PyTorch回退路径。测试覆盖数种尺寸、上/下采样、RGB JPEG及双图；并非所有图片格式、方向、色彩配置的认证。EXIF自动旋转尚未实现。详情见 [开发记录](../dev-notes/2026-11-19-multimodal.md)。
