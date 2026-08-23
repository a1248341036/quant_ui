/**
 * ECharts 图表工具 — 从原 index.html L1586-1670 提取
 * 依赖全局 echarts（由 main.js 挂载到 window）
 */

const charts = {};

export function getChartCache() {
  return charts;
}

export function chart(id) {
  const el = document.getElementById(id);
  if (!el) return null;
  if (charts[id]) {
    const bound = charts[id].getDom ? charts[id].getDom() : null;
    if (!document.body.contains(el) || (bound && bound !== el)) {
      // 详情切换后容器可能已被重建：丢弃绑定到旧 DOM 的实例，避免空图/异常
      charts[id].dispose();
      delete charts[id];
    }
  }
  if (!charts[id]) charts[id] = window.echarts.init(el);
  return charts[id];
}

export function renderLine(id, series, height) {
  const c = chart(id);
  if (!c) return;
  c.setOption({
    tooltip: { trigger: 'axis' },
    legend: { textStyle: { color: '#8494b5' }, top: 0 },
    grid: { left: 50, right: 16, top: 30, bottom: 30 },
    xAxis: { type: 'category', data: series[0]?.dates || [], axisLabel: { color: '#8494b5' } },
    yAxis: { type: 'value', axisLabel: { color: '#8494b5' } },
    series: series.map(s => {
      const item = { name: s.name, type: 'line', data: s.values, showSymbol: false, smooth: true, lineStyle: { width: 2, type: s.dash ? 'dashed' : 'solid' } };
      if (s.fill) item.areaStyle = { color: 'rgba(255, 93, 77, 0.16)' };
      return item;
    })
  }, true);
}

export function monthlyReturnCells(points) {
  if (!Array.isArray(points) || points.length < 2) return null;
  const months = {};
  for (const p of points) {
    const date = String(p.date || '').slice(0, 10);
    const value = Number(p.value);
    if (!date || !Number.isFinite(value) || value <= 0) continue;
    const key = date.slice(0, 7);
    if (!months[key]) months[key] = { first: value, last: value };
    else months[key].last = value;
  }
  const keys = Object.keys(months).sort();
  const years = [...new Set(keys.map(k => k.slice(0, 4)))].sort();
  if (!keys.length || !years.length) return null;
  const yearIndex = Object.fromEntries(years.map((y, i) => [y, i]));
  const annual = Object.fromEntries(years.map(y => [y, 1]));
  const data = [];
  for (const key of keys) {
    const ret = months[key].last / months[key].first - 1;
    const year = key.slice(0, 4);
    const month = Number(key.slice(5, 7)) - 1;
    annual[year] *= 1 + ret;
    data.push({ value: [month, yearIndex[year], ret * 100] });
  }
  for (const year of years) {
    data.push({ value: [12, yearIndex[year], (annual[year] - 1) * 100] });
  }
  return { years, data };
}

export function renderMonthlyHeatmap(id, points) {
  const c = chart(id);
  if (!c) return;
  const result = monthlyReturnCells(points);
  if (!result) { c.clear(); return; }
  c.setOption({
    tooltip: { formatter: p => `${result.years[p.value[1]]} · ${p.value[0] === 12 ? '年度' : `${p.value[0] + 1}月`}: <b>${Number(p.value[2]).toFixed(2)}%</b>` },
    grid: { left: 54, right: 82, top: 24, bottom: 48 },
    xAxis: { type: 'category', position: 'top', data: ['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月','年度'], axisLabel: { color: '#8494b5' } },
    yAxis: { type: 'category', data: result.years, inverse: true, axisLabel: { color: '#8494b5' } },
    visualMap: { type: 'piecewise', dimension: 2, pieces: [{ gt: 0, color: '#c94b55', label: '正收益' }, { value: 0, color: '#27344d', label: '0%' }, { lt: 0, color: '#35b779', label: '负收益' }], orient: 'vertical', right: 0, top: 'middle', textStyle: { color: '#8494b5' } },
    series: [{ type: 'heatmap', data: result.data, label: { show: true, color: '#e6ecf7', formatter: p => `${Number(p.value[2]).toFixed(2)}%` }, emphasis: { itemStyle: { shadowBlur: 8, shadowColor: 'rgba(255,255,255,0.35)' } } }]
  }, true);
}

export function renderBar(id, names, values) {
  const c = chart(id);
  if (!c) return;
  c.setOption({
    tooltip: {},
    grid: { left: 50, right: 16, top: 20, bottom: 40 },
    xAxis: { type: 'category', data: names, axisLabel: { color: '#8494b5', rotate: 30 } },
    yAxis: { type: 'value', axisLabel: { color: '#8494b5' } },
    series: [{ type: 'bar', data: values, itemStyle: { color: '#4f8cff' } }]
  }, true);
}

// 初始化全局 resize 监听
if (typeof window !== 'undefined' && !window.__chartResizeInit) {
  window.__chartResizeInit = true;
  window.addEventListener('resize', () => Object.values(charts).forEach(c => c.resize()));
}
