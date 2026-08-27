<template>
  <section>
    <div class="card">
      <h3>多因子自由组合</h3>
      <p class="muted">勾选因子、设定方向与权重后可直接回测，组合可保存/加载/删除。每个因子先做横截面百分位排名（0~1），按方向翻转后乘以权重求和，再按综合得分降序选股（买高）。</p>
      <div class="form-section">
        <div class="form-section-title">参数</div>
        <div class="form-grid">
          <label class="field"><span>股票池</span><select v-model="comp.universe"><option>科技TMT</option><option>沪深300+中证500+中证1000</option><option>ETF</option><option>场外基金</option></select></label>
          <label class="field"><span>TopN</span><select v-model.number="comp.top_n"><option v-for="n in [1, 2, 3, 5, 8, 10]" :value="n">{{ n }}</option></select></label>
          <label class="field"><span>初始资金</span><input type="number" v-model.number="comp.capital" step="1000"></label>
          <label class="field"><span>频率</span><select v-model="comp.freq"><option value="monthly">月频</option><option value="weekly">周频</option></select></label>
          <label class="field"><span>开始</span><input type="date" v-model="comp.start"></label>
          <label class="field"><span>结束</span><input type="date" v-model="comp.end"></label>
          <label class="field"><span>基准</span>
            <select v-model="comp.bench">
              <option>等权股票池</option>
              <option v-for="i in store.indices" :value="i.name">{{ i.name }}</option>
            </select>
          </label>
        </div>
      </div>
      <div class="form-section">
        <div class="form-section-title">因子配置</div>
        <div class="table-wrap"><table>
          <tr><th>启用</th><th>因子</th><th>说明</th><th>方向</th><th>权重</th></tr>
          <tr v-for="f in factorOptions" :key="f.name">
            <td><input type="checkbox" v-model="comp.factors[f.name].enabled"></td>
            <td>{{ f.label }}</td>
            <td class="muted">{{ f.desc || '-' }}</td>
            <td><select v-model="comp.factors[f.name].dir" :disabled="!comp.factors[f.name].enabled"><option>买高</option><option>买低</option></select></td>
            <td><input type="number" v-model.number="comp.factors[f.name].weight" step="0.1" min="-10" max="10" :disabled="!comp.factors[f.name].enabled"></td>
          </tr>
        </table></div>
      </div>
      <div class="form-section">
        <div class="form-section-title">组合管理</div>
        <div class="form-grid">
          <label class="field"><span>组合名称</span><input v-model="compName" placeholder="如：动量+低波"></label>
          <label class="field"><span>已保存组合</span>
            <select v-model="compPick">
              <option value="" disabled>选择组合…</option>
              <option v-for="c in composites" :key="c.name" :value="c.name">{{ c.name }}</option>
            </select>
          </label>
        </div>
        <div class="form-actions">
          <span class="left err" v-if="compError">{{ compError }}</span>
          <button @click="saveComposite">保存当前配置</button>
          <button @click="loadComposite" :disabled="!compPick">加载</button>
          <button @click="deleteComposite" :disabled="!compPick">删除</button>
          <button class="primary" @click="runCompositeBacktest" :disabled="compRunning">
            <span v-if="compRunning" class="spinner"></span>{{ compRunning ? '计算中…' : '跑组合回测' }}
          </button>
        </div>
      </div>
    </div>

    <div v-if="compResult">
      <div class="cards">
        <div class="metric" v-for="(v, k) in compResult.metrics" :key="k"><div class="label">{{ k }}</div><div class="value" :class="sign(v)">{{ metricText(k, v) }}</div></div>
      </div>
      <div class="cards" v-if="store.benchInfo.metrics">
        <div class="metric"><div class="label">基准 {{ store.benchInfo.name }} 总收益</div><div class="value" :class="sign(store.benchInfo.metrics.total)">{{ pct(store.benchInfo.metrics.total) }}</div></div>
        <div class="metric"><div class="label">基准年化</div><div class="value" :class="sign(store.benchInfo.metrics.annual)">{{ pct(store.benchInfo.metrics.annual) }}</div></div>
        <div class="metric"><div class="label">基准夏普</div><div class="value">{{ fmt(store.benchInfo.metrics.sharpe, 2) }}</div></div>
        <div class="metric"><div class="label">基准最大回撤</div><div class="value" :class="sign(store.benchInfo.metrics.mdd)">{{ pct(store.benchInfo.metrics.mdd) }}</div></div>
      </div>
      <div class="card"><h3>资金曲线（策略 vs 基准）</h3><div id="compEquity" class="chart"></div></div>
      <div class="card"><h3>月度收益热力图</h3><div id="compMonthly" class="chart heatmap"></div><p class="muted">按组合净值计算；首月和末月按实际区间统计，年度列为月度收益复合值。</p></div>
      <div class="card"><h3>回撤</h3><div id="compDraw" class="chart small"></div></div>
      <div class="grid" style="grid-template-columns:1fr 1fr">
        <div class="card"><h3>持仓明细</h3>
          <div style="overflow-x:auto"><table><tr><th>代码</th><th>名称</th><th>权重%</th><th>价格</th><th>市值</th></tr>
          <tr v-for="h in compResult.holdings" :key="h.code"><td>{{ stock(h.code, h.name) }}</td><td>{{ h.name || '-' }}</td><td>{{ fmt(h.weight_pct) }}</td><td>{{ fmt(h.price) }}</td><td>{{ fmt(h.market_value) }}</td></tr></table></div>
        </div>
        <div class="card"><h3>调仓记录</h3>
          <div style="overflow-x:auto"><table><tr><th>日期</th><th>信号日</th><th>持仓数</th><th>换手</th></tr>
          <tr v-for="(t, i) in compResult.trades" :key="i"><td>{{ t.date.slice(0, 10) }}</td><td>{{ t.signal_date.slice(0, 10) }}</td><td>{{ t.num_hold }}</td><td>{{ fmt(t.turnover) }}</td></tr></table></div>
        </div>
      </div>
    </div>
    <div v-else class="card"><div class="empty">设置因子后点击「跑组合回测」查看结果</div></div>

    <div class="card">
      <h3>今日信号</h3>
      <div class="form-actions" style="margin-top:0">
        <button @click="loadCompositeSignals" :disabled="compSignalsLoading">
          <span v-if="compSignalsLoading" class="spinner"></span>{{ compSignalsLoading ? '获取中…' : '按当前组合刷新' }}
        </button>
        <span class="muted" v-if="compSigDate">信号日：{{ compSigDate }} · 数据截至：{{ store.panelInfo.last_date || '-' }}</span>
      </div>
      <div class="table-wrap"><table><tr><th>代码</th><th>名称</th><th>因子得分</th><th>收盘价</th><th>换手率</th></tr>
      <tr v-for="(s, i) in compSignals" :key="i"><td>{{ stock(s.code, s.name) }}</td><td>{{ s.name || '-' }}</td><td>{{ fmt(s.score, 4) }}</td><td>{{ fmt(s.close) }}</td><td>{{ pct(s.turnover) }}</td></tr>
      </table></div>
      <div v-if="!compSignals.length" class="empty">暂无信号，先按当前组合刷新</div>
    </div>
  </section>
</template>

<script>
import { store } from '../store/index.js'
import { fmt, pct, sign, stock, today, metricText } from '../utils/format.js'
import { api } from '../utils/api.js'
import { renderLine, renderMonthlyHeatmap } from '../utils/charts.js'

export default {
  name: 'Composite',
  data() {
    return {
      store,
      factorOptions: [],
      composites: [],
      compName: '',
      compPick: '',
      compError: '',
      compRunning: false,
      compSignals: [],
      compSigDate: '',
      compSignalsLoading: false,
      comp: {
        universe: '科技TMT',
        top_n: 5,
        capital: 100000,
        freq: 'monthly',
        start: '2026-02-02',
        end: today(),
        bench: '沪深300',
        factors: {
          turn20: { enabled: true, dir: '买低', weight: 1 },
          am20: { enabled: true, dir: '买高', weight: 1 },
          mom20: { enabled: true, dir: '买高', weight: 1 },
          mom60: { enabled: false, dir: '买高', weight: 1 },
          vol20: { enabled: true, dir: '买低', weight: 1 },
          ma_cross5_10: { enabled: false, dir: '买高', weight: 1 },
          ma_cross5_20: { enabled: false, dir: '买高', weight: 1 },
          ma_cross10_30: { enabled: false, dir: '买高', weight: 1 },
          ma_cross20_60: { enabled: false, dir: '买高', weight: 1 },
        },
      },
      compResult: null,
    }
  },
  watch: {
    'comp.bench'() {
      if (this.compResult) this.renderCompositeBacktest()
    },
  },
  mounted() {
    this.loadFactors()
    this.loadComposites()
  },
  methods: {
    fmt,
    pct,
    sign,
    metricText,
    stock(code, name) {
      return stock(code, name, store.names)
    },
    async loadFactors() {
      try {
        this.factorOptions = await api('/api/factors')
        for (const f of this.factorOptions) {
          if (!this.comp.factors[f.name]) this.comp.factors[f.name] = { enabled: false, dir: '买高', weight: 1 }
        }
      } catch (e) {}
    },
    async loadComposites() {
      try { this.composites = await api('/api/composites') } catch (e) {}
    },
    collectComposite() {
      const weights = {}, directions = {}
      for (const f of this.factorOptions) {
        const cfg = this.comp.factors[f.name]
        if (cfg && cfg.enabled) {
          weights[f.name] = Number(cfg.weight) || 0
          directions[f.name] = cfg.dir === '买低'
        }
      }
      return { weights, directions }
    },
    async saveComposite() {
      this.compError = ''
      const { weights, directions } = this.collectComposite()
      if (!Object.keys(weights).length) { this.compError = '请至少勾选一个因子'; return }
      if (!this.compName.trim()) { this.compError = '请填组合名称'; return }
      try {
        const r = await api('/api/composites', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: this.compName, weights, directions }),
        })
        if (r.error) { this.compError = r.error; return }
        await this.loadComposites()
        this.compPick = r.name
      } catch (e) {
        this.compError = '保存失败: ' + e.message
      }
    },
    async loadComposite() {
      this.compError = ''
      if (!this.compPick) return
      const item = this.composites.find(c => c.name === this.compPick)
      if (!item) return
      for (const name of Object.keys(this.comp.factors)) {
        this.comp.factors[name].enabled = false
      }
      for (const [name, w] of Object.entries(item.weights || {})) {
        if (!this.comp.factors[name]) this.comp.factors[name] = { enabled: true, dir: '买高', weight: 1 }
        this.comp.factors[name].enabled = true
        this.comp.factors[name].weight = w
        if (item.directions && name in item.directions) {
          this.comp.factors[name].dir = item.directions[name] ? '买低' : '买高'
        }
      }
    },
    async deleteComposite() {
      this.compError = ''
      if (!this.compPick) return
      try {
        await api('/api/composites/' + encodeURIComponent(this.compPick), { method: 'DELETE' })
        this.compPick = ''
        await this.loadComposites()
      } catch (e) {
        this.compError = '删除失败: ' + e.message
      }
    },
    async runCompositeBacktest() {
      this.compError = ''
      const { weights, directions } = this.collectComposite()
      if (!Object.keys(weights).length) { this.compError = '请至少勾选一个因子'; return }
      this.compRunning = true
      try {
        const r = await api('/api/backtest', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            universe: this.comp.universe,
            strategy: '组合策略',
            top_n: this.comp.top_n,
            capital: this.comp.capital,
            freq: this.comp.freq,
            start: this.comp.start,
            end: this.comp.end,
            exclude_kechuang: true,
            affordable: true,
            composite_weights: weights,
            composite_directions: directions,
            composite_name: this.compName || '未命名组合',
          }),
        })
        if (r.error) { this.compError = r.error; return }
        this.compResult = r
        this.$nextTick(async () => this.renderCompositeBacktest())
      } catch (e) {
        this.compError = '回测失败: ' + e.message
      } finally {
        this.compRunning = false
      }
    },
    async renderCompositeBacktest() {
      const nav = this.compResult.nav
      const series = [
        { name: '组合资金', dates: nav.map(x => x.date), values: nav.map(x => +(x.value * this.comp.capital).toFixed(2)) },
      ]
      const bs = await store.benchSeries(this.compResult, this.comp, this.comp.start, this.comp.end)
      if (bs.points.length) {
        series.push({ name: bs.name, dates: bs.points.map(x => x.date), values: bs.points.map(x => +(x.value * this.comp.capital).toFixed(2)), dash: true })
        store.benchInfo = { name: bs.name, metrics: store.calcSeriesMetrics(bs.points) }
      } else {
        store.benchInfo = { name: '', metrics: null }
      }
      renderLine('compEquity', series)
      renderMonthlyHeatmap('compMonthly', nav)
      const dd = this.compResult.drawdown
      renderLine('compDraw', [{ name: '回撤', dates: dd.map(x => x.date), values: dd.map(x => +(x.value * 100).toFixed(2)), fill: true }])
    },
    async loadCompositeSignals() {
      this.compError = ''
      const { weights, directions } = this.collectComposite()
      if (!Object.keys(weights).length) { this.compError = '请至少勾选一个因子'; return }
      this.compSignalsLoading = true
      try {
        const r = await api('/api/signals', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            universe: this.comp.universe,
            strategy: '组合策略',
            top_n: 10,
            composite_weights: weights,
            composite_directions: directions,
          }),
        })
        if (r.error) { this.compError = r.error; return }
        this.compSignals = r.items
        this.compSigDate = r.signal_date
      } catch (e) {
        this.compError = '信号获取失败: ' + e.message
      } finally {
        this.compSignalsLoading = false
      }
    },
  },
}
</script>
