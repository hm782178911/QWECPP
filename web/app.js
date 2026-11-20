import { validateFiles, contentFor, isImageDataUrl } from "./attachments.mjs";
import { loadConversations, saveConversations } from "./storage.mjs";
("use strict");
function newId() {
  if (globalThis.crypto?.randomUUID) return crypto.randomUUID();
  return (
    Date.now().toString(36) +
    "-" +
    Math.random().toString(36).slice(2) +
    "-" +
    Math.random().toString(36).slice(2)
  );
}
const $ = (s) => document.querySelector(s),
  storageKey = "qwecpp.conversations.v1";
let pending = [],
  uploading = false,
  restoring = true;
let sessions = [],
  active = null,
  caps = null,
  busy = false,
  requestId = null;
let maxTokens = Number(localStorage.getItem("qwecpp.maxTokens")) || 128;
maxTokens = Math.max(1, Math.min(256, maxTokens));
$("#maxTokens").value = maxTokens;
$("#maxTokensValue").textContent = maxTokens + " token";
function save() {
  saveConversations(sessions.slice(0, 50)).catch(() =>
    showError("浏览器无法保存对话或图片，当前内容仅在本页有效；请勿直接刷新。"),
  );
}
function showError(s) {
  $("#error").textContent = s;
  $("#error").hidden = !s;
}
function current() {
  return sessions.find((s) => s.id === active);
}
function fresh() {
  if (busy || uploading || restoring) return;
  pending = [];
  renderAttachments();
  const s = { id: newId(), title: "新对话", messages: [] };
  sessions.unshift(s);
  active = s.id;
  save();
  render();
  $("#input").focus();
}
function renderNav() {
  const nav = $("#sessions");
  nav.replaceChildren();
  for (const s of sessions) {
    const b = document.createElement("button");
    b.className = "session" + (s.id === active ? " active" : "");
    b.textContent = s.title;
    b.title = s.title;
    b.disabled = busy || uploading || restoring;
    b.onclick = () => {
      pending = [];
      renderAttachments();
      active = s.id;
      showError("");
      render();
      document.body.classList.remove("sidebar-open");
    };
    nav.append(b);
  }
}
// 只构造文本DOM，模型输出不能注入HTML；支持常用标题、列表、粗体、行内代码与代码块。
function inline(el, text) {
  for (const part of text.split(/(`[^`]+`|\*\*[^*]+\*\*)/g)) {
    const node = part.startsWith("`")
      ? document.createElement("code")
      : part.startsWith("**")
        ? document.createElement("strong")
        : document.createElement("span");
    node.textContent = part.startsWith("`")
      ? part.slice(1, -1)
      : part.startsWith("**")
        ? part.slice(2, -2)
        : part;
    el.append(node);
  }
}
function markdown(el, text) {
  el.replaceChildren();
  let code = false,
    lines = [],
    list = null;
  function flush() {
    if (!lines.length) return;
    const pre = document.createElement("pre"),
      c = document.createElement("code");
    c.textContent = lines.join("\n");
    pre.append(c);
    el.append(pre);
    lines = [];
  }
  for (const line of text.split("\n")) {
    if (line.startsWith("```")) {
      if (code) flush();
      code = !code;
      list = null;
      continue;
    }
    if (code) {
      lines.push(line);
      continue;
    }
    let m;
    if ((m = line.match(/^(#{1,6})\s+(.*)/))) {
      const h = document.createElement(m[1].length < 3 ? "h2" : "h3");
      inline(h, m[2]);
      el.append(h);
      list = null;
    } else if ((m = line.match(/^\s*(?:[-*]|\d+\.)\s+(.*)/))) {
      if (!list) {
        list = document.createElement("ul");
        el.append(list);
      }
      const li = document.createElement("li");
      inline(li, m[1]);
      list.append(li);
    } else {
      list = null;
      const p = document.createElement("p");
      inline(p, line);
      el.append(p);
    }
  }
  if (code) flush();
}
function render() {
  renderNav();
  const box = $("#messages");
  box.replaceChildren();
  const s = current();
  $("#welcome").hidden = Boolean(s?.messages.length);
  for (const message of s?.messages || []) {
    const row = document.createElement("article");
    row.className = "message " + message.role;
    const av = document.createElement("div");
    av.className = "avatar";
    av.textContent = message.role === "assistant" ? "Q" : "你";
    const body = document.createElement("div");
    body.className = "message-body";
    const role = document.createElement("div");
    role.className = "message-role";
    role.textContent = message.role === "assistant" ? "QWECPP" : "你";
    const content = document.createElement("div");
    content.className = "content";
    if (message.content) markdown(content, message.content);
    else if (busy && message === s.messages.at(-1)) {
      content.className += " thinking";
      content.textContent = "正在思考…";
    }
    const actions = document.createElement("div");
    actions.className = "message-actions";
    if (message.content) {
      const copy = document.createElement("button");
      copy.className = "copy";
      copy.textContent = "复制";
      copy.onclick = async () => {
        try {
          await navigator.clipboard.writeText(message.content);
          copy.textContent = "已复制";
        } catch {
          showError("无法访问剪贴板，请手动选择复制。");
        }
      };
      actions.append(copy);
    }
    if (message.metrics) {
      const m = message.metrics,
        node = document.createElement("span");
      node.className = "metrics";
      node.textContent = `${m.input_tokens} in / ${m.output_tokens} out · TTFT ${m.ttft_ms.toFixed(0)} ms · TPOT ${m.tpot_ms.toFixed(1)} ms · ${m.output_tok_s.toFixed(1)} tok/s`;
      actions.append(node);
    }
    if (message.status === "error") {
      const node = document.createElement("span");
      node.textContent = "未完成 · 未计入后续上下文";
      node.className = "metrics";
      actions.append(node);
    }
    const photos = document.createElement("div");
    photos.className = "message-photos";
    for (const photo of message.attachments || []) {
      if (!isImageDataUrl(photo.url)) continue;
      const image = document.createElement("img");
      image.src = photo.url;
      image.alt = photo.name;
      image.title = photo.name;
      image.loading = "lazy";
      photos.append(image);
    }
    body.append(role, photos, content, actions);
    row.append(av, body);
    box.append(row);
  }
  $("#send").hidden = busy;
  $("#stop").hidden = !busy;
  $("#send").disabled =
    !caps ||
    restoring ||
    uploading ||
    (!$("#input").value.trim() && !pending.length);
  $("#input").disabled = busy;
  $("#newChat").disabled = busy || uploading || restoring;
  $("#attach").disabled = busy || uploading || restoring || !caps?.vision;
  renderAttachments();
  $("#deleteChat").disabled = busy || uploading || restoring;
  $("#thread").scrollTop = $("#thread").scrollHeight;
}
async function send(event) {
  event?.preventDefault();
  const text = $("#input").value.trim();
  if ((!text && !pending.length) || busy || uploading || restoring || !caps)
    return;
  if (!current()) fresh();
  const s = current();
  if (
    pending.length ||
    s.messages.some((m) => m.status !== "error" && m.attachments?.length)
  ) {
    try {
      validateFiles([], s.messages, pending, caps);
    } catch (e) {
      showError(e.message);
      return;
    }
  }
  if (!s.messages.length) s.title = text.slice(0, 28) || "图片问答";
  const user = {
      role: "user",
      content: text || "请描述这张图片。",
      attachments: pending,
    },
    reply = { role: "assistant", content: "" };
  s.messages.push(user, reply);
  pending = [];
  $("#input").value = "";
  requestId = newId();
  busy = true;
  showError("");
  render();
  let finished = false;
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id: requestId,
        messages: s.messages
          .slice(0, -1)
          .filter((m) => m.status !== "error")
          .map((m) => ({ role: m.role, content: contentFor(m) })),
        max_new_tokens: maxTokens,
      }),
    });
    if (!response.ok) {
      let data = await response.json();
      throw Error(data.error || "请求失败");
    }
    const reader = response.body.getReader(),
      decoder = new TextDecoder();
    let pending = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      pending += decoder.decode(value, { stream: true });
      let pos;
      while ((pos = pending.indexOf("\n\n")) !== -1) {
        const block = pending.slice(0, pos);
        pending = pending.slice(pos + 2);
        for (const line of block.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          const data = JSON.parse(line.slice(6));
          if (data.type === "delta") {
            reply.content += data.text;
            render();
          } else if (data.type === "done") {
            finished = true;
            reply.metrics = data.metrics;
            reply.finish_reason = data.finish_reason;
            if (data.finish_reason === "length")
              showError("已达到本轮输出上限，可在设置中调整。");
            if (data.finish_reason === "context")
              showError("已达到上下文上限，请新建对话。");
          } else if (data.type === "error") throw Error(data.error);
        }
      }
    }
    if (!finished) throw Error("连接中断，回复未完成。");
  } catch (e) {
    reply.status = "error";
    user.status = "error";
    pending = user.attachments || [];
    $("#input").value = text;
    showError(e.message);
  } finally {
    busy = false;
    requestId = null;
    save();
    render();
    $("#input").focus();
  }
}
$("#composer").onsubmit = send;
$("#input").oninput = () => {
  $("#send").disabled =
    !caps ||
    restoring ||
    uploading ||
    (!$("#input").value.trim() && !pending.length);
};
$("#input").onkeydown = (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    send();
  }
};
$("#newChat").onclick = fresh;
$("#menu").onclick = () => document.body.classList.toggle("sidebar-open");
$("#settingsButton").onclick = () => $("#settings").showModal();
$("#maxTokens").oninput = (e) => {
  maxTokens = Number(e.target.value);
  $("#maxTokensValue").textContent = maxTokens + " token";
  localStorage.setItem("qwecpp.maxTokens", maxTokens);
};
$("#deleteChat").onclick = () => {
  if (busy || uploading || restoring) return;
  pending = [];
  renderAttachments();
  sessions = sessions.filter((s) => s.id !== active);
  active = sessions[0]?.id || null;
  save();
  $("#settings").close();
  showError("");
  if (!active) fresh();
  else render();
};
$("#stop").onclick = async () => {
  if (!requestId) return;
  $("#stop").disabled = true;
  $("#stop").textContent = "正在停止…";
  try {
    await fetch("/api/cancel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: requestId }),
    });
  } catch (e) {
    showError(e.message);
  } finally {
    $("#stop").disabled = false;
    $("#stop").textContent = "■ 停止生成";
  }
};
document.querySelectorAll("[data-prompt]").forEach(
  (b) =>
    (b.onclick = () => {
      $("#input").value = b.dataset.prompt;
      $("#input").dispatchEvent(new Event("input"));
      $("#input").focus();
    }),
);
document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "k") {
    e.preventDefault();
    fresh();
  }
});
function renderAttachments() {
  const tray = $("#attachments");
  tray.replaceChildren();
  for (let i = 0; i < pending.length; ++i) {
    const photo = pending[i],
      card = document.createElement("div");
    card.className = "attachment";
    const image = document.createElement("img");
    image.src = photo.url;
    image.alt = photo.name;
    const name = document.createElement("span");
    name.textContent = photo.name;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "remove-image";
    remove.setAttribute("aria-label", "移除图片 " + photo.name);
    remove.textContent = "×";
    remove.disabled = busy || uploading;
    remove.onclick = () => {
      pending.splice(i, 1);
      render();
    };
    card.append(image, name, remove);
    tray.append(card);
  }
  tray.hidden = !pending.length;
}
async function addFiles(files) {
  if (busy || uploading || restoring) return;
  files = Array.from(files);
  if (!files.length) return;
  try {
    validateFiles(files, current()?.messages || [], pending, caps);
    uploading = true;
    showError("");
    render();
    const photos = [];
    for (const file of files) {
      const url = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(Error("图片读取失败"));
        reader.readAsDataURL(file);
      });
      if (!isImageDataUrl(url)) throw Error("不支持的图片数据");
      const image = new Image();
      await new Promise((resolve, reject) => {
        image.onload = resolve;
        image.onerror = () => reject(Error("图片损坏或格式不符"));
        image.src = url;
      });
      const w = image.naturalWidth,
        h = image.naturalHeight;
      if (
        !w ||
        !h ||
        w > 4096 ||
        h > 4096 ||
        w * h > 16000000 ||
        Math.max(w, h) > Math.min(w, h) * 200
      )
        throw Error(
          "图片尺寸超限：每边最多4096，总像素最多1600万，长宽比最多200",
        );
      photos.push({ name: file.name, size: file.size, url });
    }
    pending.push(...photos);
  } catch (e) {
    showError(e.message);
  } finally {
    uploading = false;
    $("#imageFiles").value = "";
    render();
  }
}
$("#attach").onclick = () => $("#imageFiles").click();
$("#imageFiles").onchange = (e) => addFiles(e.target.files);
$("#composer").ondragover = (e) => {
  if (e.dataTransfer.types.includes("Files")) e.preventDefault();
};
$("#composer").ondrop = (e) => {
  e.preventDefault();
  addFiles(e.dataTransfer.files);
};
loadConversations(storageKey)
  .then((data) => {
    sessions = data
      .filter((s) => typeof s.id === "string" && Array.isArray(s.messages))
      .slice(0, 50);
    active = sessions[0]?.id || null;
    restoring = false;
    if (!active) fresh();
    else render();
  })
  .catch(() => {
    restoring = false;
    fresh();
    showError("浏览器持久存储不可用，对话仅在当前页面保存。");
  });
fetch("/api/capabilities")
  .then((r) => {
    if (!r.ok) throw Error("服务连接失败");
    return r.json();
  })
  .then((data) => {
    caps = data;
    $("#attach").setAttribute(
      "aria-label",
      data.vision ? "添加图片" : "当前模型不支持图片",
    );
    $("#capability").textContent = data.vision
      ? "支持 PNG / JPEG · 会话最多 4 张 / 8 MiB"
      : "纯文本模式 · 图片暂未开放";
    $("#attach").title = data.vision
      ? "选择 PNG / JPEG 图片"
      : "当前后端仅支持文本";
    $("#model").textContent = data.model;
    $("#engine").textContent =
      `${data.parallel.toUpperCase()} · FP32 · ${data.max_context} CONTEXT`;
    $("#status").replaceChildren();
    const dot = document.createElement("span");
    dot.className = "dot";
    $("#status").append(dot, "模型已就绪");
    const dl = $("#serverInfo");
    for (const [key, value] of [
      ["模型", data.model],
      ["执行模式", data.parallel.toUpperCase()],
      ["上下文", data.max_context + " token"],
      ["KV Cache", data.kv_cache ? "已开启" : "未开启"],
      ["视觉输入", data.vision ? "可用" : "尚未支持"],
    ]) {
      const dt = document.createElement("dt"),
        dd = document.createElement("dd");
      dt.textContent = key;
      dd.textContent = value;
      dl.append(dt, dd);
    }
    render();
  })
  .catch((e) => {
    $("#model").textContent = "服务未连接";
    $("#status").textContent = "离线";
    showError(e.message);
  });
