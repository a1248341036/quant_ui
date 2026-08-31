<template>
  <section>
    <template v-if="false">
    <div class="cards">
      <div class="metric"><div class="label">账户总资产</div><div class="value" :class="{ muted: accEmpty }">{{ accEmpty ? '未开户' : fmt(accSummary?.latest_equity) }}</div></div>
      <div class="metric"><div class="label">现金</div><div class="value" :class="{ muted: accEmpty }">{{ accEmpty ? '未开户' : fmt(accSummary?.cash) }}</div></div>
      <div class="metric"><div class="label">持仓市值</div><div class="value" :class="{ muted: accEmpty }">{{ accEmpty ? '未开户' : fmt(accSummary?.market_value) }}</div></div>
      <div class="metric"><div class="label">累计盈亏</div><div class="value" :class="accEmpty ? 'muted' : sign(accSummary?.pnl)">{{ accEmpty ? '未开户' : fmt(accSummary?.pnl) }}</div></div>
      <div class="metric"><div class="label">累计收益率</div><div class="value" :class="accEmpty ? 'muted' : sign(accSummary?.pnl_pct)">{{ accEmpty ? '未开户' : pct(accSummary?.pnl_pct) }}</div></div>
    </div>
    <div class="card"><h3>资金曲线（真实账户）</h3><div id="dashEquity" class="chart"></div></div>
    <div class="card" v-if="accEmpty"><p class="muted">还没有记账数据，去「账户」页录一笔入金和交易后这里会显示真实资金曲线。下方默认展示多策略模拟对比。</p></div>
    </template>

    <!-- 信号 -->
    <div class="card">
      <h3>信号</h3>
      <div class="form-grid">
        <label class="field"><span>股票池</span><select v-model="sig.universe"><option>科技TMT</option><option>沪深300+中证500+中证1000</option><option>ETF</option><option>场外基金</option></select></label>
        <label class="field wide"><span>策略</span><strategy-select v-model="sig.strategy" :strategies="store.strategies" placeholder="选择策略"></strategy-select></label>
        <label class="field"><span>条数</span><select v-model.number="sig.top_n"><option v-for="n in [5,10,15,20,30]" :value="n">{{n}}</option></select></label>
        <label class="field-check"><input type="checkbox" v-model="sig.long_short"> 多空对冲</label>
        <label class="field"><span>空头只数</span><input type="number" v-model.number="sig.short_n" min="1" max="20" :disabled="!sig.long_short"></label>
      </div>
      <div class="form-actions"><button class="primary" @click="loadSignals">刷新</button></div>
      <p class="muted" v-if="sigDate">信号日：{{sigDate}} · 数据截至：{{store.panelInfo.last_date || '-'}}</p>
    </div>
    <div class="card"><div id="sigChart" class="chart small"></div></div>
    <div class="card">
      <div class="table-wrap"><table><tr><th>代码</th><th>方向</th><th>因子得分</th><th>收盘价</th><th>换手率</th></tr>
      <tr v-for="(s,i) in signals" :key="i"><td>{{stock(s.code, s.name)}}</td><td>{{s.side || '多'}}</td><td>{{fmt(s.score, 4)}}</td><td>{{fmt(s.close)}}</td><td>{{pct(s.turnover)}}</td></tr>
      </table></div>
      <div v-if="!signals.length" class="empty">暂无信号，请先刷新</div>
    </div>

    <!-- 隔夜订单单 -->
    <div class="card">
      <h3>隔夜订单单 <span class="muted" style="font-size:12px;font-weight:normal">收盘信号 → 次日开盘竞价成交，挂单后无需看盘</span></h3>
      <div class="form-grid">
        <label class="field"><span>资金</span><input type="number" v-model.number="orders.capital" step="10000"></label>
        <label class="field"><span>单票上限</span><input type="number" v-model.number="orders.max_weight" step="0.05" min="0.01" max="1"></label>
        <label class="field"><span>止损%</span><input type="number" v-model.number="orders.stoploss_pct" step="0.01" min="0.01"></label>
        <label class="field"><span>限价缓冲%</span><input type="number" v-model.number="orders.limit_buffer_pct" step="0.005" min="0"></label>
        <label class="field"><span>持仓来源</span>
          <select v-model="orders.account_id">
            <option :value="null">无持仓（纯新建仓）</option>
            <option v-for="a in paperAccounts" :key="a.id" :value="a.id">{{ a.name }}（模拟盘）</option>
          </select>
        </label>
      </div>
      <div class="form-actions">
        <p class="err left" v-if="ordersError">{{ ordersError }}</p>
        <button class="primary" @click="genOrders" :disabled="ordersRunning"><span v-if="ordersRunning" class="spinner"></span>{{ ordersRunning ? '生成中…' : '生成订单单' }}</button>
      </div>
      <p class="muted" v-if="orderSheet">
        信号日 {{ orderSheet.signal_date }} · 权重 {{ pct(orderSheet.weight) }} · 持仓 {{ orderSheet.n_held }} 只 · 资金 {{ fmt(orderSheet.capital) }}
      </p>
      <template v-if="orderSheet">
        <div v-if="orderSheet.sells.length"><h4 style="margin:10px 0 4px">卖出（晚间挂单）</h4>
          <div class="table-wrap"><table><tr><th>代码</th><th>名称</th><th>股数</th><th>现价</th><th>盈亏</th><th>卖出限价</th><th>原因</th></tr>
          <tr v-for="r in orderSheet.sells" :key="'s'+r.code"><td>{{stock(r.code, r.name)}}</td><td>{{r.name||'-'}}</td><td>{{r.shares}}</td><td>{{fmt(r.close)}}</td><td :class="sign(r.pnl_pct)">{{pct(r.pnl_pct)}}</td><td>{{fmt(r.limit_price)}}</td><td>{{r.reason}}</td></tr></table></div></div>
        <div v-if="orderSheet.buys.length"><h4 style="margin:10px 0 4px">买入（晚间挂单）</h4>
          <div class="table-wrap"><table><tr><th>代码</th><th>名称</th><th>得分</th><th>收盘</th><th>权重</th><th>股数</th><th>预计金额</th><th>买入限价</th><th>备注</th></tr>
          <tr v-for="r in orderSheet.buys" :key="'b'+r.code"><td>{{stock(r.code, r.name)}}</td><td>{{r.name||'-'}}</td><td>{{fmt(r.score,4)}}</td><td>{{fmt(r.close)}}</td><td>{{pct(r.weight)}}</td><td>{{r.shares}}</td><td>{{fmt(r.target_value)}}</td><td>{{fmt(r.limit_price)}}</td><td>{{r.action}}</td></tr></table></div></div>
        <div v-if="orderSheet.holds.length"><h4 style="margin:10px 0 4px">继续持有</h4>
          <div class="table-wrap"><table><tr><th>代码</th><th>名称</th><th>股数</th><th>现价</th><th>权重</th></tr>
          <tr v-for="r in orderSheet.holds" :key="'h'+r.code"><td>{{stock(r.code, r.name)}}</td><td>{{r.name||'-'}}</td><td>{{r.shares}}</td><td>{{fmt(r.close)}}</td><td>{{pct(r.weight)}}</td></tr></table></div></div>
        <div v-if="orderSheet.stoploss.length"><h4 style="margin:10px 0 4px">止损条件单（券商 App 预设，触发价=成本×(1-止损%)）</h4>
          <div class="table-wrap"><table><tr><th>代码</th><th>名称</th><th>股数</th><th>成本</th><th>现价</th><th>盈亏</th><th>触发价</th></tr>
          <tr v-for="r in orderSheet.stoploss" :key="'sl'+r.code"><td>{{stock(r.code, r.name)}}</td><td>{{r.name||'-'}}</td><td>{{r.shares}}</td><td>{{fmt(r.avg_cost)}}</td><td>{{fmt(r.close)}}</td><td :class="sign(r.pnl_pct)">{{pct(r.pnl_pct)}}</td><td>{{fmt(r.trigger_price)}}</td></tr></table></div></div>
        <p class="muted" v-for="(n,i) in orderSheet.notes" :key="'n'+i">· {{n}}</p>
      </template>
      <div v-else class="empty">先刷新信号，再点「生成订单单」</div>
    </div>

    <template v-if="false">
    <div class="card">
      <h3>策略对比（模拟）</h3>
      <div class="form-section">
        <div class="form-section-title">参数</div>
        <div class="form-grid">
          <label class="field"><span>股票池</span><select v-model="cmp.universe"><option>科技TMT</option><option>沪深300+中证500+中证1000</option><option>ETF</option><option>场外基金</option></select></label>
          <label class="field wide"><span>策略（可多选）</span><strategy-select v-model="cmp.strategies" :strategies="store.strategies" multiple :max="6" placeholder="点选策略"></strategy-select></label>
          <label class="field"><span>TopN</span><select v-model.number="cmp.top_n"><option v-for="n in [3, 5, 8, 10]" :value="n">{{ n }}</option></select></label>
          <label class="field"><span>初始资金</span><input type="number" v-model.number="cmp.capital" step="1000"></label>
          <label class="field"><span>回看起点</span><input type="date" v-model="cmp.start_date" :min="store.panelInfo.first_date || ''" :max="cmp.end_date || store.panelInfo.last_date || ''"></label>
          <label class="field"><span>回看终点</span><input type="date" v-model="cmp.end_date" :min="store.panelInfo.first_date || ''" :max="store.panelInfo.last_date || today()"></label>
          <label class="field"><span>基准</span>
            <select v-model="cmp.bench">
              <option>等权股票池</option>
              <option v-for="i in store.indices" :value="i.name">{{ i.name }}</option>
            </select>
          </label>
        </div>
      </div>
      <div class="form-actions">
        <p class="err left" v-if="cmpError">{{ cmpError }}</p>
        <button class="primary" @click="runCompare()" :disabled="cmpRunning"><span v-if="cmpRunning" class="spinner"></span>{{ cmpRunning ? '跑批中…' : '跑对比' }}</button>
      </div>
    </div>

    <div v-if="cmpResult && cmpResult.items && cmpResult.items.length">
      <div class="card"><h3>多策略资金曲线（vs 基准）</h3><div id="cmpChart" class="chart"></div></div>
      <div class="card"><h3>多策略回撤（%）</h3><div id="cmpDraw" class="chart small"></div></div>
      <div class="cards" v-if="store.benchInfo.metrics">
        <div class="metric"><div class="label">基准 {{ store.benchInfo.name }} 总收益</div><div class="value" :class="sign(store.benchInfo.metrics.total)">{{ pct(store.benchInfo.metrics.total) }}</div></div>
        <div class="metric"><div class="label">基准年化</div><div class="value" :class="sign(store.benchInfo.metrics.annual)">{{ pct(store.benchInfo.metrics.annual) }}</div></div>
        <div class="metric"><div class="label">基准夏普</div><div class="value">{{ fmt(store.benchInfo.metrics.sharpe, 2) }}</div></div>
        <div class="metric"><div class="label">基准最大回撤</div><div class="value" :class="sign(store.benchInfo.metrics.mdd)">{{ pct(store.benchInfo.metrics.mdd) }}</div></div>
      </div>
      <div class="card"><h3>策略指标对比</h3>
        <div style="overflow-x:auto"><table>
          <tr><th>策略</th><th>总收益</th><th>年化收益</th><th>年化波动</th><th>夏普</th><th>最大回撤</th><th>卡玛</th><th>胜率</th><th>信号日</th></tr>
          <tr v-for="r in cmpRows()" :key="r['策略']">
            <td>{{ r['策略'] }}</td>
            <td :class="sign(r['总收益'])">{{ pct(r['总收益']) }}</td>
            <td :class="sign(r['年化收益'])">{{ pct(r['年化收益']) }}</td>
            <td>{{ pct(r['年化波动']) }}</td>
            <td>{{ fmt(r['夏普'], 2) }}</td>
            <td :class="sign(r['最大回撤'])">{{ pct(r['最大回撤']) }}</td>
            <td>{{ fmt(r['卡玛'], 2) }}</td>
            <td>{{ pct(r['胜率']) }}</td>
            <td>{{ r['信号日'] }}</td>
          </tr>
        </table></div>
      </div>
      <div class="card">
        <h3>当前持仓（最近一次调仓）</h3>
        <div class="hold-tabs">
          <button v-for="(it, i) in cmpResult.items" :key="'t' + it.name" :class="{ active: cmpHoldIdx === i }" @click="cmpHoldIdx = i">{{ it.name }}</button>
        </div>
        <div v-for="(it, i) in cmpResult.items" v-show="cmpHoldIdx === i" :key="'h' + it.name">
          <div style="overflow-x:auto" v-if="it.holdings && it.holdings.length">
            <table><tr><th>代码</th><th>名称</th><th>权重%</th><th>价格</th><th>市值</th></tr>
            <tr v-for="h in it.holdings" :key="h.code"><td>{{ stock(h.code, h.name) }}</td><td>{{ h.name || '-' }}</td><td>{{ fmt(h.weight_pct) }}</td><td>{{ fmt(h.price) }}</td><td>{{ fmt(h.market_value) }}</td></tr></table>
          </div>
          <p class="muted" v-else>当前空仓</p>
        </div>
      </div>
    </div>
    </template>
  </section>
</template>

<script>
import { store } from '../store/index.js'
import { fmt, pct, sign, stock, today, metricText } from '../utils/format.js'
import { api } from '../utils/api.js'
import { renderLine, renderBar } from '../utils/charts.js'

export default {
  name: 'Dashboard',
  data() {
    return {
      store,
      accSummary: null,
      accItems: [],
      accEmpty: true,
      // 信号
      sig: { universe: '科技TMT', strategy: '低换手冷门', top_n: 10, long_short: false, short_n: 3 },
      signals: [],
      sigDate: '',
      sigInfo: {},
      // 策略对比（已下架）
      cmp: {
        universe: '科技TMT',
        strategies: ['低换手冷门', '反转 20 日', '低波动', '动量 20 日'],
        top_n: 3,
        capital: 100000,
        start_date: '',
        end_date: '',
        bench: '沪深300',
      },
      cmpResult: null,
      cmpRunning: false,
      cmpError: '',
      cmpHoldIdx: 0,
      // 隔夜订单单
      paperAccounts: [],
      orders: { capital: 100000, max_weight: 0.25, stoploss_pct: 0.07, limit_buffer_pct: 0.02, account_id: null },
      orderSheet: null,
      ordersRunning: false,
      ordersError: '',
    }
  },
  watch: {
    'cmp.universe'(universe) {
      store.reloadStrategiesForUniverse(universe)
    },
    'sig.universe'(universe) {
      store.reloadStrategiesForUniverse(universe)
    },
    'cmp.bench'() {
      if (this.cmpResult && this.cmpResult.items && this.cmpResult.items.length) {
        this.renderCompare(this.cmp.start_date || this.cmp.end_date, this.cmp.end_date || store.panelInfo.last_date)
      }
    },
  },
  mounted() {
    // 账户和策略对比已下架
    // this.loadAccount()
    // this.runCompare(true)
    this.loadSignals()
    this.loadPaperAccounts()
  },
  methods: {
    fmt,
    pct,
    sign,
    today,
    metricText,
    stock(code, name) {
      return stock(code, name, store.names)
    },
    // ---- 信号 ----
    async loadSignals() {
      const strat = (store.strategies || []).find(x => x.name === this.sig.strategy) || {};
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
      renderBar('sigChart', this.signals.map(s => this.stock(s.code, s.name)), this.signals.map(s => +s.score.toFixed(4)));
    },
    // ---- 隔夜订单单 ----
    async loadPaperAccounts() {
      try {
        const r = await api('/api/paper/accounts');
        this.paperAccounts = (r.accounts || r || []).filter(a => a && a.id);
      } catch (e) { /* 账户列表失败不阻塞页面 */ }
    },
    async genOrders() {
      if (!this.signals.length) { this.ordersError = '请先刷新信号'; return; }
      this.ordersRunning = true;
      this.ordersError = '';
      try {
        const strat = (store.strategies || []).find(x => x.name === this.sig.strategy) || {};
        const body = {
          universe: this.sig.universe,
          strategy: this.sig.strategy,
          top_n: this.sig.top_n,
          long_short: this.sig.long_short || strat.long_short || false,
          short_n: this.sig.short_n,
          capital: this.orders.capital,
          max_weight: this.orders.max_weight,
          stoploss_pct: this.orders.stoploss_pct,
          limit_buffer_pct: this.orders.limit_buffer_pct,
        };
        if (this.orders.account_id) body.account_id = this.orders.account_id;
        const r = await api('/api/signals/orders', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (r.error) { this.ordersError = r.error; return; }
        this.orderSheet = r;
      } catch (e) {
        this.ordersError = '订单单生成失败: ' + e.message;
      } finally {
        this.ordersRunning = false;
      }
    },
    // ---- 账户（已下架） ----
    async loadAccount() {
      try {
        const eq = await api('/api/ledger/equity')
        this.accSummary = eq.summary
        this.accItems = eq.items || []
        this.accEmpty = eq.items.length === 0
        if (this.accEmpty) {
          this.accSummary = null
        } else {
          this.$nextTick(() => renderLine('dashEquity', this.accountSeries()))
        }
      } catch (e) {
        this.accEmpty = true
      }
    },
    accountSeries() {
      const items = this.accItems
      return [
        { name: '总资产', dates: items.map(x => x.date.slice(0, 10)), values: items.map(x => +x.equity.toFixed(2)) },
        { name: '持仓市值', dates: items.map(x => x.date.slice(0, 10)), values: items.map(x => +x.market_value.toFixed(2)) },
      ]
    },
    // ---- 策略对比（已下架） ----
    async runCompare(silent) {
      if (!store.strategies.length) return
      if (!this.cmp.strategies.length) { if (!silent) alert('请至少选择一个策略'); return }
      this.cmpRunning = true
      this.cmpError = ''
      try {
        let end = this.cmp.end_date || store.panelInfo.last_date || today()
        let start = this.cmp.start_date
        if (!start) {
          const d = new Date(end + 'T00:00:00Z')
          d.setUTCMonth(d.getUTCMonth() - 6)
          start = d.toISOString().slice(0, 10)
        }
        if (start > end) { [start, end] = [end, start] }
        const r = await api('/api/backtest/compare', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            universe: this.cmp.universe,
            strategies: this.cmp.strategies,
            top_n: this.cmp.top_n,
            capital: this.cmp.capital,
            freq: 'monthly',
            start,
            end,
            exclude_kechuang: true,
            affordable: true,
            amount_q: 0.2,
            warmup_days: 400,
          }),
        })
        if (r.error) { this.cmpError = r.error; return }
        this.cmpResult = r
        this.cmpHoldIdx = 0
        this.$nextTick(async () => this.renderCompare(start, end))
      } catch (e) {
        if (!silent) this.cmpError = '对比失败: ' + e.message
      } finally {
        this.cmpRunning = false
      }
    },
    async renderCompare(start, end) {
      const series = this.cmpResult.items.map(it => ({
        name: it.name,
        dates: it.nav.map(x => x.date),
        values: it.nav.map(x => +(x.value * this.cmp.capital).toFixed(2)),
      }))
      const bs = await store.benchSeries(this.cmpResult, this.cmp, start, end)
      if (bs.points.length) {
        series.push({ name: bs.name, dates: bs.points.map(x => x.date), values: bs.points.map(x => +(x.value * this.cmp.capital).toFixed(2)), dash: true })
        store.benchInfo = { name: bs.name, metrics: store.calcSeriesMetrics(bs.points) }
      } else {
        store.benchInfo = { name: '', metrics: null }
      }
      renderLine('cmpChart', series)
      const ddSeries = this.cmpResult.items
        .filter(it => it.drawdown && it.drawdown.length)
        .map(it => ({ name: it.name, dates: it.drawdown.map(x => x.date), values: it.drawdown.map(x => +(x.value * 100).toFixed(2)) }))
      if (ddSeries.length) renderLine('cmpDraw', ddSeries.map(x => ({ ...x, fill: true })), 220)
    },
    cmpRows() {
      const order = ['总收益', '年化收益', '年化波动', '夏普', '最大回撤', '卡玛', '胜率']
      return this.cmpResult.items.map(it => {
        const row = { '策略': it.name }
        for (const k of order) row[k] = it.metrics ? it.metrics[k] : null
        row['信号日'] = it.last_signal_date || '-'
        return row
      })
    },
  },
}
</script>
