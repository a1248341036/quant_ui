<template>
    <section v-if="tab==='signals'">
      <div class="card">
        <div class="form-grid">
          <label class="field"><span>股票池</span><select v-model="sig.universe"><option>科技TMT</option><option>沪深300+中证500+中证1000</option><option>ETF</option><option>场外基金</option></select></label>
          <label class="field wide"><span>策略</span><strategy-select v-model="sig.strategy" :strategies="strategies" placeholder="选择策略"></strategy-select></label>
          <label class="field"><span>条数</span><select v-model.number="sig.top_n"><option v-for="n in [5,10,15,20,30]" :value="n">{{n}}</option></select></label>
          <label class="field-check"><input type="checkbox" v-model="sig.long_short"> 多空对冲</label>
          <label class="field"><span>空头只数</span><input type="number" v-model.number="sig.short_n" min="1" max="20" :disabled="!sig.long_short"></label>
        </div>
        <div class="form-actions"><button class="primary" @click="loadSignals">刷新</button></div>
        <p class="muted" v-if="sigDate">信号日：{{sigDate}} · 数据截至：{{panelInfo.last_date || '-'}} · 上次刷新：{{meta.last_update || '-'}}</p>
      </div>
      <div class="card"><div id="sigChart" class="chart small"></div></div>
      <div class="card">
        <div class="table-wrap"><table><tr><th>代码</th><th>方向</th><th>因子得分</th><th>收盘价</th><th>换手率</th></tr>
        <tr v-for="(s,i) in signals" :key="i"><td>{{stock(s.code, s.name, names)}}</td><td>{{s.side || '多'}}</td><td>{{fmt(s.score, 4)}}</td><td>{{fmt(s.close)}}</td><td>{{pct(s.turnover)}}</td></tr>
        </table></div>
        <div v-if="!signals.length" class="empty">暂无信号，请先刷新</div>
      </div>
    </section>
</template>

<script>
import { store } from '../store/index.js'
import { fmt, pct, sign, stock } from '../utils/format.js'
import { api } from '../utils/api.js'
import { renderBar } from '../utils/charts.js'

export default {
  name: 'SignalsView',
  data() {
    return {
      tab: 'signals',
      sig: { universe: '科技TMT', strategy: '低换手冷门', top_n: 10, long_short: false, short_n: 3 },
      signals: [],
      sigDate: '',
      sigInfo: {},
    };
  },
  computed: {
    strategies() {
      return store.strategies || [];
    },
    names() {
      return store.names || {};
    },
    panelInfo() {
      return store.panelInfo || {};
    },
    meta() {
      return store.meta || {};
    },
  },
  methods: {
    fmt,
    pct,
    sign,
    stock,
    async loadSignals() {
      const strat = this.strategies.find(x => x.name === this.sig.strategy) || {};
      const q = new URLSearchParams({ universe: this.sig.universe, strategy: this.sig.strategy, top_n: this.sig.top_n });
      const ls = this.sig.long_short || strat.long_short;
      if (ls) { q.set('long_short', 'true'); q.set('short_n', this.sig.short_n || strat.short_n || 3); }
      try {
        const r = await api('/api/signals?' + q.toString());
        if (r.error) { alert(r.error); return; }
        this.signals = r.items; this.sigDate = r.signal_date; this.sigInfo = r;
        this.$nextTick(() => this.renderSignals());
      } catch (e) { alert('信号获取失败: ' + e.message); }
    },
    renderSignals() {
      renderBar('sigChart', this.signals.map(s => this.stock(s.code, s.name, this.names)), this.signals.map(s => +s.score.toFixed(4)));
    },
  },
  mounted() {
    this.loadSignals();
  },
};
</script>
