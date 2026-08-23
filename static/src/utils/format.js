/**
 * 格式化工具函数 — 从原 index.html methods 提取
 */

export function today() {
  return new Date().toISOString().slice(0, 10);
}

export function fmt(x, d) {
  if (x == null || x === '') return '-';
  const n = Number(x);
  return Number.isFinite(n)
    ? n.toLocaleString('zh-CN', { minimumFractionDigits: d ?? 2, maximumFractionDigits: d ?? 2 })
    : String(x);
}

export function pct(x) {
  if (x == null || x === '') return '-';
  const n = Number(x);
  return Number.isFinite(n) ? (n * 100).toFixed(2) + '%' : '-';
}

export function sign(x) {
  return x == null ? '' : (x > 0 ? 'pos' : (x < 0 ? 'neg' : ''));
}

export function stock(code, name, names) {
  const nm = name || (names && names[code]) || '';
  return nm ? `${code} ${nm}` : (code || '-');
}

export function num(v) {
  if (v == null || v === '') return '-';
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(4) : String(v);
}

export function fmtAmt(x) {
  if (x == null) return '-';
  const a = Math.abs(x);
  if (a >= 1e8) return (x / 1e8).toFixed(2) + '亿';
  if (a >= 1e4) return (x / 1e4).toFixed(0) + '万';
  return fmt(x, 0);
}

export function fmtVol(x) {
  if (x == null) return '-';
  const a = Math.abs(x);
  if (a >= 1e4) return (x / 1e4).toFixed(1) + '万手';
  return fmt(x, 0) + '手';
}

export function metricText(k, v) {
  return ['总收益', '年化收益', '年化波动', '最大回撤', '卡玛', '胜率'].includes(k)
    ? pct(v)
    : fmt(v);
}

export function numOrNull(x) {
  if (x == null || x === '') return null;
  const n = Number(x);
  return Number.isFinite(n) ? n : null;
}

export function toNum(x) {
  const n = numOrNull(x);
  return n === null ? 0 : n;
}

export function sortCompare(x, y, dir) {
  const nx = numOrNull(x), ny = numOrNull(y);
  if (nx !== null && ny !== null) return (nx - ny) * dir;
  if (x == null && y == null) return 0;
  if (x == null) return 1;
  if (y == null) return -1;
  return String(x).localeCompare(String(y), 'zh-Hans-CN') * dir;
}
