/**
 * 全局共享 store — 跨页面共享的响应式状态和方法
 * 使用 Vue reactive() 创建单例，各组件 import 后直接使用
 *
 * 从原 index.html 根组件 data() 提取共享部分（names, strategies, panelInfo, indices 等）
 * 以及跨页面方法（reloadStrategies, reloadStrategiesForUniverse, benchSeries, calcSeriesMetrics 等）
 */
import { reactive } from 'vue'
import { api } from '../utils/api.js'
import { today } from '../utils/format.js'

export const store = reactive({
  // —— 共享状态 ——
  panelInfo: {},
  names: {},
  indices: [],
  idxCache: {},
  strategies: [],

  // 策略池（跨页面共享）
  pool: {
    full: [], pooled: [], trash: [],
    q: '', grp: '全部', addSel: [], rmSel: [], delSel: [], purgeSel: [],
  },

  // benchInfo（回测/对比/组合页面共用）
  benchInfo: { name: '', metrics: null },

  // meta（数据状态页面用）
  meta: {},

  // 各页面的策略选择状态——ensureSelectedInPool 需要修正这些引用，
  // 所以它们集中放在 store 以便 Pool.vue 跨页面同步
  bt: { strategy: '低换手冷门', long_short: false, short_n: 3, short_rate: 8.6 },
  paper: { form: { strategy: '动量 20 日' } },
  sig: { strategy: '低换手冷门' },
  sweep: { strategy: '双均线多头 5/20' },
  cmp: { strategies: ['低换手冷门', '反转 20 日', '低波动', '动量 20 日'] },

  // 账户
  account: {},

  // 因子实验室评估历史（跨 AlphaAgent / History 共享，localStorage 持久化）
  labHistory: [],
  // History 打开一条因子评估 → 携载入载荷跳转到 AlphaAgent 因子实验室
  labLoadPayload: null,
  // 跨页面跳转请求（History → alphaagent 等），App.vue watch 后消费并清空
  requestTab: '',
  // 聚宽回测运行详情: 当前打开的 run_id（Code 启动 / History 或详情页切换时更新）
  jqRunId: '',

  // 因子实验室评估历史 —— localStorage 持久化（上限 100 条，最新在前）
  LAB_HISTORY_KEY: 'alphaagent_lab_history_v1',
  LAB_HISTORY_LIMIT: 100,

  loadLabHistory() {
    try {
      const raw = localStorage.getItem(this.LAB_HISTORY_KEY)
      if (raw) {
        const arr = JSON.parse(raw)
        this.labHistory = Array.isArray(arr) ? arr : []
        return
      }
    } catch (e) {}
    this.labHistory = []
  },

  saveLabHistory() {
    try {
      localStorage.setItem(this.LAB_HISTORY_KEY, JSON.stringify(this.labHistory.slice(0, this.LAB_HISTORY_LIMIT)))
    } catch (e) {
      // localStorage 满或不可用时静默失败，历史仅保留内存
    }
  },

  pushLabHistory(entry) {
    this.labHistory.unshift(entry)
    if (this.labHistory.length > this.LAB_HISTORY_LIMIT) {
      this.labHistory.length = this.LAB_HISTORY_LIMIT
    }
    this.saveLabHistory()
  },

  deleteLabHistory(id) {
    this.labHistory = this.labHistory.filter(h => h.id !== id)
    this.saveLabHistory()
  },

  clearLabHistory() {
    this.labHistory = []
    this.saveLabHistory()
  },

  // 发起跨页面跳转（History → alphaagent 等）
  goto(tab) {
    this.requestTab = tab
    this.labLoadPayload = null
  },

  // —— 共享方法 ——

  async loadIndices() {
    try {
      const r = await api('/api/data/indices');
      this.indices = r.items || [];
    } catch (e) {}
  },

  async reloadStrategies() {
    try { this.strategies = await api('/api/strategies'); } catch (e) {}
    this.ensureSelectedInPool();
    this.applyBtStrategyDefaults();
  },

  ensureSelectedInPool() {
    const names = this.strategies.map(s => s.name);
    const fallback = names.length ? names[0] : '';
    const fix = (ref, key) => {
      if (ref[key] && !names.includes(ref[key])) ref[key] = fallback;
    };
    fix(this.bt, 'strategy');
    fix(this.paper.form, 'strategy');
    fix(this.sig, 'strategy');
    fix(this.sweep, 'strategy');
    if (this.cmp.strategies && this.cmp.strategies.length) {
      this.cmp.strategies = this.cmp.strategies.filter(n => names.includes(n));
      if (!this.cmp.strategies.length && fallback) this.cmp.strategies = [fallback];
    }
  },

  applyBtStrategyDefaults() {
    const s = this.strategies.find(x => x.name === this.bt.strategy) || {};
    this.bt.long_short = !!s.long_short;
    this.bt.short_n = s.short_n || 3;
    this.bt.short_rate = Math.round(((s.short_cost_rate || 0) * 100) * 10) / 10;
  },

  async reloadStrategiesForUniverse(universe) {
    try { this.strategies = await api('/api/strategies?universe=' + encodeURIComponent(universe)); } catch (e) {}
    this.ensureSelectedInPool();
    this.applyBtStrategyDefaults();
  },

  async loadPoolData() {
    try { const r = await api('/api/strategy-pool/full'); this.pool.full = r.items || []; } catch (e) {}
    try { const r = await api('/api/strategy-pool'); this.pool.pooled = r.items || []; } catch (e) {}
    try { const r = await api('/api/strategy-pool/trash'); this.pool.trash = r.items || []; } catch (e) {}
    this.pool.q = ''; this.pool.grp = '全部'; this.pool.addSel = []; this.pool.rmSel = [];
    this.pool.delSel = [];
  },

  // benchSeries — 从原 L2193-2210 提取
  async benchSeries(result, scope, start, end) {
    if (scope.bench === '等权股票池') {
      const b = result.bench;
      return { name: '等权基准', points: b || [] };
    }
    if (!(this.indices || []).length) await this.loadIndices();
    const idx = (this.indices || []).find(i => i.name === scope.bench);
    if (!idx) return { name: scope.bench, points: [] };
    const key = idx.code + '|' + start + '|' + end;
    if (!this.idxCache[key]) {
      try {
        const r = await api('/api/data/indices/series?code=' + encodeURIComponent(idx.code) + '&start=' + encodeURIComponent(start) + '&end=' + encodeURIComponent(end));
        if (r.ok && r.items) this.idxCache[key] = r.items;
      } catch (e) {}
    }
    const pts = this.idxCache[key];
    return { name: idx.name, points: pts || [] };
  },

  // calcSeriesMetrics — 从原 L2211-2226 提取
  calcSeriesMetrics(points) {
    if (!points || points.length < 2) return null;
    const v = points.map(p => +p.value);
    const n = v.length;
    const total = v[n - 1] / v[0] - 1;
    const years = (n - 1) / 252;
    const annual = years > 0 ? Math.pow(1 + total, 1 / years) - 1 : 0;
    const rets = [];
    for (let i = 1; i < n; i++) rets.push(v[i] / v[i - 1] - 1);
    const mean = rets.reduce((a, b) => a + b, 0) / rets.length;
    const sd = Math.sqrt(rets.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (rets.length - 1 || 1));
    const sharpe = sd > 0 ? mean / sd * Math.sqrt(252) : 0;
    let peak = v[0], mdd = 0;
    for (const x of v) { if (x > peak) peak = x; const dd = x / peak - 1; if (dd < mdd) mdd = dd; }
    return { total, annual, sharpe, mdd };
  },
})
