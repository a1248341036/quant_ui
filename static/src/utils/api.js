/**
 * API 请求封装 — 从原 index.html L1580-1584 提取
 */
const apiBase = window.location.port === '43120' ? 'http://127.0.0.1:17891' : '';

export async function api(path, opts = {}) {
  // 统一 JSON 语义：带 body 的请求自动补 Content-Type 并序列化对象，
  // 防止漏写 header 时后端把 JSON 字符串当标量（Pydantic 422 model_attributes_type）。
  const headers = { ...opts.headers }
  if (opts.body != null && !Object.keys(headers).some(h => h.toLowerCase() === 'content-type')) {
    headers['Content-Type'] = 'application/json'
    if (typeof opts.body !== 'string') opts.body = JSON.stringify(opts.body)
  }
  const res = await fetch(apiBase + path, { ...opts, headers, credentials: apiBase ? 'omit' : 'include' });
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

// 统一时间窗口默认值（从后端 /api/alphaagent/windows 获取，缓存到模块级）。
// 所有页面统一从这里取 train/val/test/bt 默认日期，不再散落硬编码。
let _windowDefaults = null;

export async function getWindowDefaults() {
  if (_windowDefaults) return _windowDefaults;
  try {
    _windowDefaults = await api('/api/alphaagent/windows?t=' + Date.now());
  } catch {
    _windowDefaults = {};
  }
  return _windowDefaults;
}
