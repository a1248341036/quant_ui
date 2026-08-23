<template>
  <section>
    <div class="card">
      <h3>个股看板</h3>
      <div class="stock-search">
        <label class="field"><span>搜索（代码 / 名称）</span>
          <input v-model="stockView.q" @input="stockView.open = true" @focus="stockView.open = true"
                 @blur="stockView.open = false" @keydown.enter.prevent="pickStockMatch()"
                 placeholder="如 601728 / 中国电信 / 平安" autocomplete="off">
          <div v-if="stockView.open && stockView.q.trim()" class="stock-drop">
            <template v-if="stockMatches.length">
              <div v-for="m in stockMatches" :key="m.code" class="stock-drop-item"
                   @mousedown.prevent="pickStock(m.code, m.name)">
                <span class="code">{{m.code}}</span><span class="name">{{m.name}}</span>
                <span class="ind" v-if="m.industry">{{m.industry}}</span>
              </div>
            </template>
            <div v-else class="stock-empty">没有匹配的股票</div>
          </div>
        </label>
      </div>
    </div>

    <div v-if="stockView.loading" class="card"><p class="muted">加载中…</p></div>
    <div v-else-if="stockView.error" class="card"><p class="err">{{stockView.error}}</p></div>

    <template v-else-if="stockView.detail">
      <div class="cards">
        <div class="metric">
          <div class="label">{{stockView.detail.name || stockView.detail.code}} 最新收盘</div>
          <div class="value" :class="sign(stockView.detail.latest?.change_pct)">{{fmt(stockView.detail.latest?.close)}}</div>
        </div>
        <div class="metric">
          <div class="label">涨跌幅</div>
          <div class="value" :class="sign(stockView.detail.latest?.change_pct)">{{stockView.detail.latest?.change_pct == null ? '-' : stockView.detail.latest.change_pct + '%'}}</div>
        </div>
        <div class="metric">
          <div class="label">成交额（最新）</div>
          <div class="value">{{fmtAmt(stockView.detail.latest?.amount)}}</div>
        </div>
        <div class="metric">
          <div class="label">换手率（最新）</div>
          <div class="value">{{pct(stockView.detail.latest?.turnover)}}</div>
        </div>
        <div class="metric">
          <div class="label">20日均成交额</div>
          <div class="value">{{fmtAmt(stockView.detail.latest?.am20)}}</div>
        </div>
        <div class="metric">
          <div class="label">数据区间</div>
          <div class="value" style="font-size:15px">{{stockView.detail.start}} ~ {{stockView.detail.end}}</div>
        </div>
      </div>

      <div class="card" style="display:flex;align-items:center;gap:12px;padding:10px 14px">
        <span class="muted">复权口径</span>
        <select class="input" :value="stockView.adj"
                @change="loadStockDetail(stockView.code, $event.target.value)">
          <option value="qfq">前复权（默认）</option>
          <option value="raw">不复权（真实成交价，历史稳定）</option>
          <option value="hfq">后复权（历史价永不漂移）</option>
        </select>
      </div>

      <div class="card">
        <h3>价格与均线（近 {{stockView.detail.history.length}} 个交易日）</h3>
        <div id="stockChart" class="chart"></div>
      </div>

      <div class="card">
        <h3>最近交易日明细</h3>
        <div class="table-wrap"><table>
          <tr><th>日期</th><th>开盘</th><th>最高</th><th>最低</th><th>收盘</th><th>涨跌幅</th><th>成交量</th><th>成交额</th><th>换手率</th><th>20日换手</th><th>20日均额</th></tr>
          <tr v-for="x in stockRows()" :key="x.date">
            <td>{{x.date}}</td><td>{{fmt(x.open)}}</td><td>{{fmt(x.high)}}</td><td>{{fmt(x.low)}}</td><td>{{fmt(x.close)}}</td>
            <td :class="sign(x.change_pct)">{{x.change_pct == null ? '-' : x.change_pct + '%'}}</td>
            <td>{{fmtVol(x.volume)}}</td><td>{{fmtAmt(x.amount)}}</td>
            <td>{{pct(x.turnover)}}</td><td>{{pct(x.turn20)}}</td><td>{{fmtAmt(x.am20)}}</td>
          </tr>
        </table></div>
      </div>
    </template>

    <div v-else class="card"><p class="muted">搜索并选择一只股票开始查看个股行情。</p></div>
  </section>
</template>

<script>
import { store } from '../store/index.js'
import { api } from '../utils/api.js'
import { chart } from '../utils/charts.js'
import { fmt, pct, sign, fmtAmt, fmtVol } from '../utils/format.js'

export default {
  name: 'Stock',
  data() {
    return {
      store,
      stockView: { q: '', code: '', name: '', detail: null, loading: false, error: '', open: false, adj: 'qfq' },
    }
  },
  computed: {
    stockMatches() {
      const q = (this.stockView.q || '').trim().toLowerCase();
      if (!q) return [];
      const out = [];
      for (const [code, name] of Object.entries(this.store.names)) {
        const c = String(code).padStart(6, '0');
        const n = String(name || '');
        if (c.startsWith(q) || n.toLowerCase().includes(q)) {
          out.push({ code: c, name: n });
          if (out.length >= 20) break;
        }
      }
      return out;
    },
  },
  methods: {
    fmt, pct, sign, fmtAmt, fmtVol,
    pickStockMatch() {
      const ms = this.stockMatches;
      if (ms.length) this.pickStock(ms[0].code, ms[0].name);
    },
    pickStock(code, name) {
      this.stockView.code = code;
      this.stockView.name = name || '';
      this.stockView.q = `${code} ${this.stockView.name}`.trim();
      this.stockView.open = false;
      this.loadStockDetail(code);
    },
    async loadStockDetail(code, adj) {
      if (adj) this.stockView.adj = adj;
      this.stockView.loading = true;
      this.stockView.error = '';
      try {
        const r = await api('/api/stock/' + code + '?days=250&adj=' + encodeURIComponent(this.stockView.adj));
        this.stockView.detail = r;
      } catch (e) {
        this.stockView.error = '加载失败: ' + e.message;
      }
      this.stockView.loading = false;
      this.$nextTick(() => this.renderStockChart());
    },
    stockRows() {
      const rows = (this.stockView.detail?.history || []).slice();
      for (let i = rows.length - 1; i >= 0; i--) {
        const prev = rows[i - 1];
        rows[i].change_pct = (prev && prev.close) ? +(((rows[i].close / prev.close) - 1) * 100).toFixed(2) : null;
      }
      return rows.reverse();
    },
    renderStockChart() {
      const d = this.stockView.detail;
      if (!d || !d.history || !d.history.length) return;
      const c = chart('stockChart');
      if (!c) return;
      const dates = d.history.map(x => x.date);
      const kdata = d.history.map(x => [+x.open, +x.close, +x.low, +x.high]);
      const vol = d.history.map(x => +(x.volume || 0));
      const ma = n => {
        const close = d.history.map(x => +x.close);
        const out = [];
        for (let i = 0; i < close.length; i++) {
          if (i < n - 1) { out.push(null); continue; }
          let s = 0;
          for (let j = i - n + 1; j <= i; j++) s += close[j];
          out.push(+(s / n).toFixed(3));
        }
        return out;
      };
      c.setOption({
        tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
        legend: { data: ['K线', 'MA5', 'MA10', 'MA20', '成交量'], textStyle: { color: '#8494b5' }, top: 0 },
        grid: [
          { left: 62, right: 20, top: 34, height: '56%' },
          { left: 62, right: 20, top: '74%', height: '16%' }
        ],
        xAxis: [
          { type: 'category', data: dates, gridIndex: 0, boundaryGap: false, axisLabel: { color: '#8494b5' } },
          { type: 'category', data: dates, gridIndex: 1, boundaryGap: false, axisLabel: { show: false } }
        ],
        yAxis: [
          { gridIndex: 0, scale: true, axisLabel: { color: '#8494b5' } },
          {
            gridIndex: 1, splitLine: { show: false },
            axisLabel: {
              color: '#8494b5',
              formatter: v => {
                const a = Math.abs(v);
                if (a >= 1e8) return (v / 1e8).toFixed(1) + '亿';
                if (a >= 1e4) return (v / 1e4).toFixed(1) + '万';
                return v;
              }
            }
          }
        ],
        dataZoom: [
          { type: 'inside', xAxisIndex: [0, 1], start: 55, end: 100 },
          { type: 'slider', xAxisIndex: [0, 1], bottom: 2, height: 14, borderColor: '#1f2c45', textStyle: { color: '#8494b5' } }
        ],
        series: [
          {
            name: 'K线', type: 'candlestick', data: kdata,
            itemStyle: { color: '#ff5d4d', color0: '#2bb98a', borderColor: '#ff5d4d', borderColor0: '#2bb98a' }
          },
          { name: 'MA5', type: 'line', data: ma(5), showSymbol: false, lineStyle: { width: 1, color: '#e6b450' } },
          { name: 'MA10', type: 'line', data: ma(10), showSymbol: false, lineStyle: { width: 1, color: '#ff5d4d' } },
          { name: 'MA20', type: 'line', data: ma(20), showSymbol: false, lineStyle: { width: 1, color: '#2bb98a' } },
          { name: '成交量', type: 'bar', xAxisIndex: 1, yAxisIndex: 1, data: vol, itemStyle: { color: 'rgba(79,140,255,.35)' } }
        ]
      }, true);
    },
  },
  beforeUnmount() {
    // 离开页面时释放 chart 实例，避免 DOM 复用冲突
    const charts = window.echarts?.getInstanceByDom?.(document.getElementById('stockChart'));
    if (charts) charts.dispose();
  },
}
</script>
