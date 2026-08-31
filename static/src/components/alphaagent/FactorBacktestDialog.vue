<template>
  <div>
    <!-- 回测弹窗 -->
    <Teleport to="body">
      <div v-if="configOpen" class="factor-modal-overlay" @click="$emit('close')">
        <div class="factor-modal" @click.stop style="width:min(460px,92vw)">
          <div class="factor-modal-head">
            <strong>因子回测</strong>
            <button class="factor-modal-close" @click="$emit('close')">×</button>
          </div>
          <div class="factor-modal-body">
            <div class="factor-modal-section">
              <label>回测区间</label>
              <div class="lab-bt-dates">
                <input v-model="bt.start" type="date" class="lab-input">
                <span>→</span>
                <input v-model="bt.end" type="date" class="lab-input">
              </div>
            </div>
            <div class="lab-bt-row">
              <div class="factor-modal-section">
                <label>持仓数</label>
                <input v-model.number="bt.topN" type="number" min="1" max="100" class="lab-input">
              </div>
              <div class="factor-modal-section">
                <label>调仓频率</label>
                <select v-model="bt.freq" class="lab-input">
                  <option value="monthly">月调</option>
                  <option value="weekly">周调</option>
                  <option value="daily">日调</option>
                </select>
              </div>
            </div>
            <div class="factor-modal-section">
              <label>股票池</label>
              <select v-model="bt.universe" class="lab-input">
                <option value="全部股票">全部股票</option>
                <option value="科技TMT">科技TMT</option>
                <option value="沪深300+中证500+中证1000">沪深300+中证500+中证1000</option>
              </select>
            </div>
            <div class="lab-bt-row">
              <div class="factor-modal-section">
                <label>初始资金</label>
                <input v-model.number="bt.capital" type="number" min="1000" step="10000" class="lab-input">
              </div>
              <div class="factor-modal-section">
                <label>预热天数</label>
                <select v-model.number="bt.warmupDays" class="lab-input">
                  <option :value="0">关闭</option>
                  <option :value="120">120 天</option>
                  <option :value="400">400 天</option>
                </select>
              </div>
            </div>
            <div class="factor-modal-section">
              <label>排序方向</label>
              <div class="lab-radio-group">
                <label><input type="radio" v-model="bt.ascending" :value="false"> 因子值大→多头</label>
                <label><input type="radio" v-model="bt.ascending" :value="true"> 因子值小→多头</label>
              </div>
            </div>
            <label class="lab-bt-check">
              <input type="checkbox" v-model="bt.excludeKeChuang"> 剔除科创/创业
            </label>
            <div v-if="bt.error" class="lab-save-error">{{ bt.error }}</div>
            <button class="lab-run-btn" :disabled="bt.running" @click="runBacktest">
              {{ bt.running ? '回测中…' : '开始回测' }}
            </button>
          </div>
        </div>
      </div>
    </Teleport>

    <!-- 回测结果弹窗 -->
    <Teleport to="body">
      <div v-if="bt.result" class="factor-modal-overlay" @click="closeResult">
        <div class="factor-modal bt-result-modal" @click.stop style="width:min(900px,94vw)">
          <div class="factor-modal-head">
            <strong>回测结果</strong>
            <button class="factor-modal-close" @click="closeResult">×</button>
          </div>
          <div class="factor-modal-body">
            <p class="lab-bt-config muted">
              {{ bt.result.config?.universe }} · {{ bt.result.config?.n_codes }} 只 · TopN {{ bt.result.config?.top_n }} · {{ bt.result.config?.freq }} · 预热 {{ bt.result.config?.warmup_days }} 天 · 现金整手撮合
            </p>
            <div class="lab-bt-metrics">
              <div class="lab-metric"><span class="lab-metric-label">总收益</span><span class="lab-metric-value">{{ formatBacktestMetric(bt.result.metrics?.['总收益'], 'pct') }}</span></div>
              <div class="lab-metric"><span class="lab-metric-label">年化收益</span><span class="lab-metric-value">{{ formatBacktestMetric(bt.result.metrics?.['年化收益'], 'pct') }}</span></div>
              <div class="lab-metric"><span class="lab-metric-label">夏普比率</span><span class="lab-metric-value">{{ formatBacktestMetric(bt.result.metrics?.['夏普'], 'ratio') }}</span></div>
              <div class="lab-metric"><span class="lab-metric-label">最大回撤</span><span class="lab-metric-value">{{ formatBacktestMetric(bt.result.metrics?.['最大回撤'], 'pct') }}</span></div>
              <div class="lab-metric"><span class="lab-metric-label">超额年化</span><span class="lab-metric-value">{{ formatBacktestMetric(bt.result.metrics?.['超额年化'], 'pct') }}</span></div>
              <div class="lab-metric"><span class="lab-metric-label">超额夏普</span><span class="lab-metric-value">{{ formatBacktestMetric(bt.result.metrics?.['超额夏普'], 'ratio') }}</span></div>
            </div>
            <div class="lab-bt-actions">
              <button class="lab-action-btn export" @click="exportBacktestResult">导出回测结果</button>
            </div>
            <div class="lab-chart-block">
              <div class="lab-chart-title">净值曲线</div>
              <div id="lab-bt-nav" class="lab-chart-box"></div>
            </div>
            <div class="lab-chart-block">
              <div class="lab-chart-title">回撤曲线</div>
              <div id="lab-bt-dd" class="lab-chart-box"></div>
            </div>
            <details class="lab-bt-details" open>
              <summary>最新持仓（{{ bt.result.holdings?.length || 0 }}）</summary>
              <div class="table-wrap">
                <table>
                  <thead><tr><th>代码</th><th>名称</th><th>权重%</th></tr></thead>
                  <tbody>
                    <tr v-for="h in bt.result.holdings || []" :key="h.code">
                      <td>{{ h.code }}</td><td>{{ h.name || '-' }}</td><td>{{ formatMetricValue(h.weight_pct) }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </details>
            <details class="lab-bt-details">
              <summary>调仓记录（{{ bt.result.trades?.length || 0 }}）</summary>
              <div class="table-wrap">
                <table>
                  <thead><tr><th>日期</th><th>信号日</th><th>持仓数</th><th>换手%</th></tr></thead>
                  <tbody>
                    <tr v-for="(t, i) in bt.result.trades || []" :key="i">
                      <td>{{ String(t.date).slice(0, 10) }}</td>
                      <td>{{ String(t.signal_date).slice(0, 10) }}</td>
                      <td>{{ t.num_hold }}</td>
                      <td>{{ t.turnover == null ? '—' : (Number(t.turnover) * 100).toFixed(2) }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </details>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script>
import { api } from '../../utils/api.js'
import { chart, renderLine } from '../../utils/charts.js'
import { agentStore } from '../../store/alphaagent.js'
import { formatMetricValue, formatBacktestMetric } from '../../utils/alphaagent.js'

// 因子回测弹窗 + 结果弹窗：由父级 v-if 控制挂载，expr/factorName 经 props 注入
export default {
  name: 'FactorBacktestDialog',
  props: {
    expr: { type: String, default: '' },
    factorName: { type: String, default: 'expr' },
    // 打开时的默认回测窗口（因子实验室传验证区间；因子库入口传 null 走全局窗口默认）
    defaults: { type: Object, default: null },
  },
  emits: ['close'],
  data() {
    const w = agentStore.windowDefaults || {}
    return {
      // 配置弹窗开关：回测成功后只关配置弹窗，结果弹窗保持展示（对齐原 btDialog/btResult 双状态）
      configOpen: true,
      bt: {
        start: this.defaults?.start || w.bt_start || w.val_start || '2023-01-01',
        end: this.defaults?.end || w.bt_end || w.val_end || '2024-12-31',
        universe: '全部股票',
        excludeKeChuang: false,
        topN: 5,
        freq: 'monthly',
        capital: 100000,
        warmupDays: 400,
        ascending: false,
        running: false,
        error: '',
        result: null,
      },
    }
  },
  methods: {
    formatMetricValue,
    formatBacktestMetric,
    closeResult() {
      this.bt.result = null
      this.$emit('close')
    },
    async runBacktest() {
      this.bt.running = true
      this.bt.error = ''
      const ctrl = new AbortController()
      const timer = setTimeout(() => ctrl.abort(), 600000)
      try {
        let data
        try {
          data = await api('/api/alphaagent/backtest-factor', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              multi_line_expr: this.expr,
              factor_name: this.factorName || 'expr',
              start: this.bt.start,
              end: this.bt.end,
              top_n: this.bt.topN,
              freq: this.bt.freq,
              capital: this.bt.capital,
              ascending: this.bt.ascending,
              universe: this.bt.universe,
              exclude_kechuang: this.bt.excludeKeChuang,
              warmup_days: this.bt.warmupDays,
            }),
            signal: ctrl.signal,
          })
        } finally {
          clearTimeout(timer)
        }
        if (!data.ok) {
          this.bt.error = data.error || data.detail || '回测失败'
        } else {
          this.bt.result = data
          this.configOpen = false
          this.$nextTick(() => this.renderBtCharts(data))
        }
      } catch (e) {
        this.bt.error = (e && e.name === 'AbortError')
          ? '回测超时（600 秒）。可缩短区间、减少股票池或降低调仓频率后重试。'
          : e.message
      } finally {
        this.bt.running = false
      }
    },
    // ── 导出回测结果（与候选库 registry 格式对齐） ──
    exportBacktestResult() {
      if (!this.bt.result) return

      const factorId = this.factorName || 'expr'
      const now = new Date().toISOString()
      const btStart = this.bt.start || 'start'
      const btEnd = this.bt.end || 'end'

      // 构建与候选库一致的 registry 格式，额外包含回测详情
      const payload = {
        exported_at: now,
        library: 'candidate',
        root: 'artifacts/alphaagent/factorzoo/candidate_1d',
        n_factors: 1,
        factors: [{
          factor_id: factorId,
          name: this.factorName || 'expr',
          expr: this.expr,
          label_col: 'label_1d_open_to_open',
          metrics: this.bt.result.metrics,
          annualized_return: this.bt.result.metrics?.['年化收益'],
          annualized_excess_return: this.bt.result.metrics?.['超额年化'],
          sharpe: this.bt.result.metrics?.['夏普'],
          status: 'backtested',
          created_at: now,
        }],
        registry: {
          [factorId]: {
            factor_id: factorId,
            name: this.factorName || 'expr',
            expr: this.expr,
            comment: `回测区间：${btStart} → ${btEnd}`,
            created_at: now,
            source: 'lab_backtest_export',
            panel_path: 'label_1d_open_to_open',
            backtest_period: `${btStart} → ${btEnd}`,
            include_fundamentals: false,
            backtest_config: this.bt.result.config,
            metrics: this.bt.result.metrics,
            bench_metrics: this.bt.result.bench_metrics,
            nav: this.bt.result.nav,
            bench: this.bt.result.bench,
            drawdown: this.bt.result.drawdown,
            holdings: this.bt.result.holdings,
            trades: this.bt.result.trades,
            similarity: null,
            review_verdict: 'pending',
            review_reasons: '',
            promotion_status: 'candidate',
          }
        }
      }

      const text = JSON.stringify(payload, null, 2)
      const blob = new Blob([text], { type: 'application/json;charset=utf-8' })
      const a = document.createElement('a')
      a.href = URL.createObjectURL(blob)
      a.download = `factor_backtest_${factorId}_${btStart}_${btEnd}.json`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(a.href)
    },
    renderBtCharts(result) {
      // 净值曲线
      const navChart = chart('lab-bt-nav')
      if (navChart && result.nav) {
        const series = [{ name: '策略净值', dates: result.nav.map(p => p.date), values: result.nav.map(p => p.value) }]
        if (result.bench?.length) series.push({ name: '基准', dates: result.bench.map(p => p.date), values: result.bench.map(p => p.value), dash: true })
        renderLine('lab-bt-nav', series)
      }
      // 回撤曲线
      const ddChart = chart('lab-bt-dd')
      if (ddChart && result.drawdown) {
        renderLine('lab-bt-dd', [{ name: '回撤', dates: result.drawdown.map(p => p.date), values: result.drawdown.map(p => +(p.value * 100).toFixed(2)), fill: true }])
      }
    },
  },
}
</script>
