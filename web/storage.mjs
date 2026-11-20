// 图片随会话保存在IndexedDB，避免base64突破localStorage约5MiB限额。
let connection;
function open() {
  if (!connection)
    connection = new Promise((resolve, reject) => {
      const request = indexedDB.open("qwecpp-chat", 1);
      request.onupgradeneeded = () => request.result.createObjectStore("state");
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
      request.onblocked = () =>
        reject(Error("存储升级被其他网页阻塞，请关闭旧页面后重试。"));
    });
  return connection;
}
export async function loadConversations(legacyKey) {
  const db = await open();
  const value = await new Promise((resolve, reject) => {
    const request = db
      .transaction("state")
      .objectStore("state")
      .get("conversations");
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
  if (Array.isArray(value)) return value;
  try {
    const old = JSON.parse(localStorage.getItem(legacyKey) || "[]");
    return Array.isArray(old) ? old : [];
  } catch {
    return [];
  }
}
export async function saveConversations(sessions) {
  const snapshot = structuredClone(sessions);
  const db = await open();
  await new Promise((resolve, reject) => {
    const tx = db.transaction("state", "readwrite");
    tx.objectStore("state").put(snapshot, "conversations");
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error || Error("会话保存中断"));
  });
}
