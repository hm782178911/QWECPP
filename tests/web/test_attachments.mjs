import {
  validateFiles,
  contentFor,
  isImageDataUrl,
} from "../../web/attachments.mjs";
import assert from "node:assert/strict";
const caps = { vision: true, max_images: 4, max_image_bytes: 8 * 1024 * 1024 };
const file = { type: "image/png", size: 10 },
  image = { size: 10, url: "data:image/png;base64,AAAA" };
validateFiles([file], [], [], caps);
assert.throws(() =>
  validateFiles([{ type: "text/plain", size: 10 }], [], [], caps),
);
assert.throws(() =>
  validateFiles([file], [{ attachments: Array(4).fill(image) }], [], caps),
);
assert.throws(() =>
  validateFiles([{ ...file, size: caps.max_image_bytes }], [], [image], caps),
);
assert.throws(() => validateFiles([file], [], [], { ...caps, vision: false }));
assert.equal(contentFor({ content: "hello" }), "hello");
assert.equal(
  contentFor({ content: "hi", attachments: [image] })[0].type,
  "image_url",
);
assert.equal(isImageDataUrl("https://example.com/a.png"), false);
console.log(
  "attachment format, capability, cumulative limits and API conversion passed",
);
