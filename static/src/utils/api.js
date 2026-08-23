/**
 * API 请求封装 — 从原 index.html L1580-1584 提取
 */
export async function api(path, opts) {
  const res = await fetch(path, { ...opts, credentials: 'include' });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
