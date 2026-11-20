export const imageTypes = new Set(["image/png", "image/jpeg"]);
export function totals(messages) {
  const images = messages
    .filter((m) => m.status !== "error")
    .flatMap((m) => m.attachments || []);
  return {
    count: images.length,
    bytes: images.reduce((sum, a) => sum + a.size, 0),
  };
}
export function validateFiles(files, history, pending, caps) {
  if (!caps?.vision) throw Error("当前模型不支持图片输入。");
  const previous = totals(history),
    selected = totals([{ attachments: pending }]);
  if (previous.count + selected.count + files.length > caps.max_images)
    throw Error(
      `整个会话最多 ${caps.max_images} 张图片，请移除附件或新建对话。`,
    );
  let bytes = previous.bytes + selected.bytes;
  for (const file of files) {
    if (!imageTypes.has(file.type)) throw Error("仅支持 PNG / JPEG 图片。");
    if (!file.size || file.size > caps.max_image_bytes)
      throw Error("单张图片须大于 0 且不超过 8 MiB。");
    bytes += file.size;
  }
  if (bytes > caps.max_image_bytes)
    throw Error("当前会话图片总大小不能超过 8 MiB，请新建对话。");
}
export function isImageDataUrl(url) {
  return (
    typeof url === "string" &&
    /^data:image\/(png|jpeg);base64,[A-Za-z0-9+/]*={0,2}$/.test(url)
  );
}
export function contentFor(message) {
  const images = message.attachments || [];
  if (!images.length) return message.content;
  if (images.some((a) => !isImageDataUrl(a.url)))
    throw Error("会话中的图片数据无效。");
  return [
    ...images.map((a) => ({ type: "image_url", image_url: { url: a.url } })),
    { type: "text", text: message.content },
  ];
}
