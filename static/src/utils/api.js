/**
 * API 请求封装 — 从原 index.html L1580-1584 提取
 */
const apiBase = window.location.port === '43120' ? 'http://127.0.0.1:17891' : '';

export async function api(path, opts) {
  const res = await fetch(apiBase + path, { ...opts, credentials: apiBase ? 'omit' : 'include' });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// 统一交易参数默认值（从后端 /api/trading-defaults 获取，缓存到模块级）
let _tradingDefaults = null;

export async function getTradingDefaults() {
  if (_tradingDefaults) return _tradingDefaults;
  try {
    _tradingDefaults = await api('/api/trading-defaults');
  } catch {
    _tradingDefaults = {};
  }
  return _tradingDefaults;
}
