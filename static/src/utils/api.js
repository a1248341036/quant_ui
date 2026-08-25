/**
 * API 请求封装 — 从原 index.html L1580-1584 提取
 */
const apiBase = window.location.port === '43120' ? 'http://127.0.0.1:17891' : '';

export async function api(path, opts) {
  const res = await fetch(apiBase + path, { ...opts, credentials: apiBase ? 'omit' : 'include' });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
