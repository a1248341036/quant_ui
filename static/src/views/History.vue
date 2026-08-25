<template>
  <section>
    <div class="card">
      <h2>历史回测（PG 归档）</h2>
      <p class="muted">每次回测/对比/扫描自动落库，参数、指标、净值、交易可追溯。</p>
      <div class="row">
        <label class="field"><span>类型</span>
          <select v-model="history.kind">
            <option value="">全部</option>
            <option value="backtest">单策略回测</option>
            <option value="compare">多策略对比</option>
            <option value="sweep">参数扫描</option>
            <option value="sweep_cli">参数扫描(CLI)</option>
          </select>
        </label>
        <label class="field"><span>条数</span>
          <select v-model.number="history.limit">
            <option :value="50">50</option>
            <option :value="100">100</option>
            <option :value="200">200</option>
          </select>
        </label>
        <button class="primary" @click="loadHistory" :disabled="history.loading">刷新</button>
      </div>
      <p class="err left" v-if="history.error">{{history.error}}</p>
      <div class="table-wrap" v-if="history.items && history.items.length">
        <table>
          <thead><tr>
            <th class="sortable" @click="setHistorySort('run_id')">ID<span class="sort-arrow" v-if="history.sortKey==='run_id'">{{history.sortDir==='asc' ? '↑' : '↓'}}</span></th>
            <th class="sortable" @click="setHistorySort('created_at')">时间<span class="sort-arrow" v-if="history.sortKey==='created_at'">{{history.sortDir==='asc' ? '↑' : '↓'}}</span></th>
            <th class="sortable" @click="setHistorySort('kind')">类型<span class="sort-arrow" v-if="history.sortKey==='kind'">{{history.sortDir==='asc' ? '↑' : '↓'}}</span></th>
            <th class="sortable" @click="setHistorySort('title')">标题<span class="sort-arrow" v-if="history.sortKey==='title'">{{history.sortDir==='asc' ? '↑' : '↓'}}</span></th>
            <th class="sortable" @click="setHistorySort('period')">区间<span class="sort-arrow" v-if="history.sortKey==='period'">{{history.sortDir==='asc' ? '↑' : '↓'}}</span></th>
            <th class="sortable" @click="setHistorySort('total_return')">总收益<span class="sort-arrow" v-if="history.sortKey==='total_return'">{{history.sortDir==='asc' ? '↑' : '↓'}}</span></th>
            <th class="sortable" @click="setHistorySort('annual')">年化<span class="sort-arrow" v-if="history.sortKey==='annual'">{{history.sortDir==='asc' ? '↑' : '↓'}}</span></th>
            <th class="sortable" @click="setHistorySort('sharpe')">夏普<span class="sort-arrow" v-if="history.sortKey==='sharpe'">{{history.sortDir==='asc' ? '↑' : '↓'}}</span></th>
            <th class="sortable" @click="setHistorySort('excess_annual')">超额年化<span class="sort-arrow" v-if="history.sortKey==='excess_annual'">{{history.sortDir==='asc' ? '↑' : '↓'}}</span></th>
            <th class="sortable" @click="setHistorySort('excess_sharpe')">超额夏普<span class="sort-arrow" v-if="history.sortKey==='excess_sharpe'">{{history.sortDir==='asc' ? '↑' : '↓'}}</span></th>
            <th class="sortable" @click="setHistorySort('max_drawdown')">最大回撤<span class="sort-arrow" v-if="history.sortKey==='max_drawdown'">{{history.sortDir==='asc' ? '↑' : '↓'}}</span></th>
            <th class="sortable" @click="setHistorySort('data_version')">数据版本<span class="sort-arrow" v-if="history.sortKey==='data_version'">{{history.sortDir==='asc' ? '↑' : '↓'}}</span></th>
            <th></th>
          </tr></thead>
          <tbody>
            <tr v-for="r in historyItems()" :key="r.run_id" class="clickable-row" @click="openHistory(r.run_id)">
              <td>{{r.run_id}}</td>
              <td>{{(r.created_at||'').slice(0,16).replace('T',' ')}}</td>
              <td>{{kindName(r.kind)}}</td>
              <td>{{historyTitle(r)}}</td>
              <td>{{(r.params?.start||'')}}~{{(r.params?.end||'')}}</td>
              <td>{{pct(r.summary?.total_return)}}</td>
              <td>{{pct(r.summary?.annual)}}</td>
              <td>{{num(r.summary?.sharpe)}}</td>
              <td>{{pct(r.summary?.excess_annual)}}</td>
              <td>{{num(r.summary?.excess_sharpe)}}</td>
              <td>{{pct(r.summary?.max_drawdown)}}</td>
              <td>{{r.data_version||''}}</td>
              <td><button class="ghost" @click.stop="openHistory(r.run_id)">详情</button></td>
            </tr>
          </tbody>
        </table>
      </div>
      <div v-if="history.items && !history.items.length" class="empty">暂无归档记录</div>
    </div>

    <template v-if="history.detail">
      <div class="card" id="historyDetail">
        <h2>详情 #{{history.detail.run_id}} <span class="muted">{{history.detail.created_at}}</span></h2>
        <div class="cards" v-if="history.detail.metrics">
          <div class="metric" v-for="(v,k) in history.detail.metrics" :key="k">
            <div class="label">{{k}}</div><div class="value">{{num(v)}}</div>
          </div>
        </div>
        <h3 v-if="history.detail.params">参数</h3>
        <pre class="code-block">{{JSON.stringify(history.detail.params, null, 2)}}</pre>
        <template v-if="isSweepHistory()">
          <template v-if="histSweepRows('nav').length">
            <h3>扫描结果<span class="muted">（{{history.detail.params?.mode || history.detail.kind}}）</span></h3>
            <div class="table-wrap"><table>
              <tr><th v-for="c in histSweepCols(histSweepRows('nav'))" :key="c" class="sortable" @click="setHistSweepSort(c)">{{c}}<span class="sort-arrow">{{histSweepArrow(c)}}</span></th></tr>
              <tr v-for="(r,i) in sortedHistSweepRows('nav')" :key="i">
                <td v-for="c in histSweepCols(histSweepRows('nav'))" :key="c">{{histCell(c, r[c])}}</td>
              </tr>
            </table></div>
          </template>
          <template v-else>
            <h3>扫描结果<span class="muted">（{{history.detail.params?.mode || history.detail.kind}}）</span></h3>
            <div class="table-wrap" v-if="histSummaryRows().length"><table>
              <tr><th>参数/概要</th><th>值</th></tr>
              <tr v-for="(kv,i) in histSummaryRows()" :key="i">
                <td>{{kv[0]}}</td><td>{{histSummaryValue(kv[1])}}</td>
              </tr>
            </table></div>
            <div v-else class="empty">该记录没有可展示的扫描结果（数据缺失）</div>
          </template>
          <template v-if="histSweepRows('bench').length">
            <h3>热力矩阵（均值夏普）</h3>
            <div class="table-wrap"><table>
              <tr><th>short</th><th v-for="c in histHeatCols(histSweepRows('bench'))" :key="c">{{c}}</th></tr>
              <tr v-for="(r,i) in histSweepRows('bench')" :key="i">
                <td>{{r.short}}</td>
                <td v-for="c in histHeatCols(histSweepRows('bench'))" :key="c">{{fmt(r[c.slice(5)], 3)}}</td>
              </tr>
            </table></div>
          </template>
          <template v-else><h3>热力矩阵（均值夏普）</h3><div class="empty">暂无热力矩阵数据（该模式可能不生成）</div></template>
          <template v-if="histSweepRows('trades').length">
            <h3>窗口明细</h3>
            <div class="table-wrap"><table>
              <tr><th v-for="c in histSweepCols(histSweepRows('trades'))" :key="c" class="sortable" @click="setHistSweepSort(c)">{{c}}<span class="sort-arrow">{{histSweepArrow(c)}}</span></th></tr>
              <tr v-for="(r,i) in sortedHistSweepRows('trades')" :key="i">
                <td v-for="c in histSweepCols(histSweepRows('trades'))" :key="c">{{histCell(c, r[c])}}</td>
              </tr>
            </table></div>
          </template>
          <template v-else><h3>窗口明细</h3><div class="empty">暂无窗口明细数据</div></template>
        </template>
        <template v-else>
          <template v-if="histNavSeries().length">
            <h3>净值</h3>
            <div id="histNav" style="height:320px"></div>
          </template>
          <template v-else><h3>净值</h3><div class="empty">暂无净值数据</div></template>
          <template v-if="histDrawSeries().length">
            <h3>回撤（%）</h3>
            <div id="histDraw" class="chart small"></div>
          </template>
          <template v-else><h3>回撤（%）</h3><div class="empty">暂无回撤数据</div></template>
          <div class="grid" style="grid-template-columns:1fr 1fr">
            <div>
              <h3>持仓明细</h3>
              <div class="table-wrap" v-if="histHoldingRows().length"><table><tr><th v-if="hasNameCol(histHoldingRows())">策略</th><th>代码</th><th>名称</th><th>权重%</th><th>价格</th><th>市值</th></tr>
              <tr v-for="(h,i) in histHoldingRows()" :key="i"><td v-if="hasNameCol(histHoldingRows())">{{h.name}}</td><td>{{stock(h.code, h.name)}}</td><td>{{h.name||'-'}}</td><td>{{fmt(h.weight_pct)}}</td><td>{{fmt(h.price)}}</td><td>{{fmt(h.market_value)}}</td></tr></table></div>
              <div v-else class="empty">暂无持仓数据</div>
            </div>
            <div>
              <h3>调仓记录</h3>
              <div class="table-wrap" v-if="histTradeRows().length"><table><tr><th v-if="hasNameCol(histTradeRows())">策略</th><th>日期</th><th>信号日</th><th>持仓数</th><th>换手</th></tr>
              <tr v-for="(t,i) in histTradeRows()" :key="i"><td v-if="hasNameCol(histTradeRows())">{{t.name}}</td><td>{{(t.date||'').slice(0,10)}}</td><td>{{(t.signal_date||'').slice(0,10)}}</td><td>{{t.num_hold}}</td><td>{{fmt(t.turnover)}}</td></tr></table></div>
              <div v-else class="empty">暂无调仓记录</div>
            </div>
          </div>
        </template>
      </div>
    </template>
  </section>
</template>

<script>
import { store } from '../store/index.js'
import { api } from '../utils/api.js'
import { fmt, pct, sign, num, numOrNull, toNum, sortCompare } from '../utils/format.js'
import { renderLine } from '../utils/charts.js'

export default {
  name: 'History',
  data() {
    return {
      store,
      history: { kind: '', limit: 50, items: [], loading: false, error: '', detail: null, sortKey: 'created_at', sortDir: 'desc' },
      histSweepSort: { key: '', dir: 'asc' },
    }
  },
  mounted() {
    this.loadHistory()
  },
  methods: {
    fmt,
    pct,
    sign,
    num,
    numOrNull,
    toNum,
    sortCompare,
    stock(code, name) {
      const nm = name || store.names[code] || '';
      return nm ? `${code} ${nm}` : (code || '-');
    },
    kindName(k) {
      return ({ backtest: '单策略回测', compare: '多策略对比', sweep: '参数扫描', sweep_cli: '参数扫描(CLI)' })[k] || k || '';
    },
    historyTitle(r) {
      const p = r.params || {}, s = r.summary || {};
      if (r.kind === 'compare') return '对比 ' + (p.strategies || []).slice(0, 4).join('、');
      if (r.kind === 'sweep' || r.kind === 'sweep_cli') {
        return '扫描[' + (p.mode || '') + '] ' + (p.mode === 'factor'
          ? (p.strategy || '')
          : ((p.short_list || []).length + 'x' + (p.long_list || []).length));
      }
      if (s.composite) return '组合[' + (s.composite_name || '自定义') + '] ' + (p.universe || '');
      return (p.strategy || '') + ' · ' + (p.universe || '');
    },
    historyItems() {
      const items = (this.history.items || []).slice();
      const key = this.history.sortKey;
      const dir = this.history.sortDir === 'asc' ? 1 : -1;
      const get = this.historySortValue(key);
      return items.sort((a, b) => this.sortCompare(get(a), get(b), dir));
    },
    historySortValue(key) {
      return (r) => {
        const p = r.params || {}, s = r.summary || {};
        switch (key) {
          case 'run_id': return r.run_id;
          case 'created_at': return r.created_at || '';
          case 'kind': return this.kindName(r.kind);
          case 'title': return this.historyTitle(r);
          case 'period': return (p.start || '') + '~' + (p.end || '');
          case 'total_return': return this.numOrNull(s.total_return);
          case 'annual': return this.numOrNull(s.annual);
          case 'sharpe': return this.numOrNull(s.sharpe);
          case 'excess_annual': return this.numOrNull(s.excess_annual);
          case 'excess_sharpe': return this.numOrNull(s.excess_sharpe);
          case 'max_drawdown': return this.numOrNull(s.max_drawdown);
          case 'data_version': return r.data_version || '';
          default: return '';
        }
      };
    },
    setHistorySort(key) {
      if (this.history.sortKey === key) {
        this.history.sortDir = this.history.sortDir === 'asc' ? 'desc' : 'asc';
      } else {
        this.history.sortKey = key;
        this.history.sortDir = 'asc';
      }
    },
    async loadHistory() {
      this.history.loading = true; this.history.error = '';
      try {
        const r = await api('/api/backtest/runs?kind=' + encodeURIComponent(this.history.kind) + '&limit=' + this.history.limit);
        this.history.items = r.items || [];
      } catch (e) { this.history.error = '加载失败: ' + e.message; }
      finally { this.history.loading = false; }
    },
    async openHistory(id) {
      this.history.error = '';
      this.history.detail = null;
      try {
        const r = await api('/api/backtest/runs/' + id);
        if (r.error) { this.history.error = r.error; return; }
        this.history.detail = r;
        this.$nextTick(() => {
          try {
            if (!this.history.detail) return;
            const isSweep = this.isSweepHistory();
            const series = this.histNavSeries();
            // sweep 详情模板没有图表容器，跳过渲染，避免在不存在/被重建的容器上初始化
            if (series.length && !isSweep) renderLine('histNav', series);
            const dd = this.histDrawSeries();
            if (dd.length && !isSweep) renderLine('histDraw', dd, 220);
            const el = document.getElementById('historyDetail');
            if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
          } catch (err) {
            this.history.error = '详情渲染失败: ' + (err && err.message ? err.message : err);
          }
        });
      } catch (e) { this.history.error = '详情失败: ' + e.message; }
    },
    histNavSeries() {
      const d = this.history.detail;
      if (!d) return [];
      const nav = d.nav;
      if (!Array.isArray(nav) || !nav.length) return [];
      if ('points' in nav[0]) {
        return nav.map(it => ({
          name: String(it.name || ''),
          dates: (it.points || []).map(p => p.date),
          values: (it.points || []).map(p => this.toNum(p.value)),
        }));
      }
      if ('value' in nav[0]) {
        return [{ name: '净值', dates: nav.map(p => p.date), values: nav.map(p => this.toNum(p.value)) }];
      }
      return [];
    },
    histDrawSeries() {
      const d = this.history.detail;
      if (!d || !d.drawdown) return [];
      if (Array.isArray(d.drawdown) && d.drawdown.length && 'points' in d.drawdown[0]) {
        return d.drawdown.map(it => ({
          name: String(it.name || ''),
          dates: (it.points || []).map(p => p.date),
          values: (it.points || []).map(p => +(this.toNum(p.value) * 100).toFixed(2)),
        }));
      }
      if (Array.isArray(d.drawdown) && d.drawdown.length && 'value' in d.drawdown[0]) {
        return [{ name: '回撤', dates: d.drawdown.map(p => p.date), values: d.drawdown.map(p => +(this.toNum(p.value) * 100).toFixed(2)) }];
      }
      return [];
    },
    histHoldingRows() {
      const d = this.history.detail;
      if (!d || !d.holdings) return [];
      if (Array.isArray(d.holdings) && d.holdings.length && 'records' in d.holdings[0]) {
        return d.holdings.flatMap(it => (it.records || []).map(r => ({ name: it.name, ...r })));
      }
      return d.holdings;
    },
    histTradeRows() {
      const d = this.history.detail;
      if (!d || !d.trades) return [];
      if (Array.isArray(d.trades) && d.trades.length && 'records' in d.trades[0]) {
        return d.trades.flatMap(it => (it.records || []).map(r => ({ name: it.name, ...r })));
      }
      return d.trades;
    },
    isSweepHistory() {
      const k = this.history.detail && this.history.detail.kind;
      return k === 'sweep' || k === 'sweep_cli';
    },
    histSweepRows(col) {
      const d = this.history.detail;
      if (!d || !Array.isArray(d[col])) return [];
      if (d[col].length && 'records' in d[col][0]) {
        return d[col].flatMap(it => (it.records || []).map(r => ({ name: it.name, ...r })));
      }
      return d[col];
    },
    sortedHistSweepRows(col) {
      const rows = this.histSweepRows(col).slice();
      if (!this.histSweepSort || this.histSweepSort.key !== col) return rows;
      const dir = this.histSweepSort.dir === 'asc' ? 1 : -1;
      return rows.sort((a, b) => this.sortCompare(a[col], b[col], dir));
    },
    setHistSweepSort(col) {
      if (this.histSweepSort.key === col) {
        this.histSweepSort.dir = this.histSweepSort.dir === 'asc' ? 'desc' : 'asc';
      } else {
        this.histSweepSort.key = col;
        this.histSweepSort.dir = 'asc';
      }
    },
    histSweepArrow(col) {
      return this.histSweepSort.key === col ? (this.histSweepSort.dir === 'asc' ? '↑' : '↓') : '';
    },
    histSummaryRows() {
      const d = this.history.detail;
      if (!d) return [];
      const rows = [];
      const push = (prefix, obj) => {
        if (!obj || typeof obj !== 'object') return;
        for (const [k, v] of Object.entries(obj)) {
          if (v === undefined || v === null) continue;
          rows.push([prefix ? prefix + '.' + k : k, v]);
        }
      };
      push('', d.params);
      push('', d.summary);
      return rows;
    },
    histSummaryValue(v) {
      if (typeof v === 'object') {
        const s = JSON.stringify(v);
        return s && s.length > 120 ? s.slice(0, 120) + '…' : s;
      }
      if (typeof v === 'number') return this.fmt(v);
      return v;
    },
    histSweepCols(rows) {
      if (!rows || !rows.length) return [];
      const order = ['short','long','freq','fold','n_windows','n_days','start','end',
                     'train_start','train_end','test_start','test_end',
                     'chosen_top_n','chosen_freq','chosen_short','chosen_long',
                     'trained','mean_sharpe','median_sharpe','std_sharpe','sharpe',
                     'train_sharpe','total','mean_total','median_total','best_total',
                     'worst_total','annual','calmar','mdd','mean_mdd','max_drawdown',
                     'win_rate','end_nav','top_n','capital'];
      const first = rows[0] || {};
      return order.filter(k => first[k] !== undefined)
                  .concat(Object.keys(first).filter(k => !order.includes(k)));
    },
    histHeatCols(rows) {
      if (!rows || !rows.length) return [];
      return Object.keys(rows[0]).filter(k => k !== 'short').map(k => 'long=' + k);
    },
    histCell(key, v) {
      if (v == null) return '-';
      if (['total','annual','mdd','calmar','win_rate'].some(s => key.includes(s))) return this.pct(v);
      if (['sharpe','nav'].some(s => key.includes(s))) return this.fmt(v, 2);
      if (['days','windows'].some(s => key.includes(s))) return this.fmt(v, 0);
      return v;
    },
    hasNameCol(rows) { return rows.length > 0 && 'name' in rows[0]; },
  },
}
</script>
