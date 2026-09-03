<template>
  <div class="lab-panel">
    <div class="lab-left">
      <h3>因子表达式</h3>
      <textarea v-model="lab.expr" class="lab-editor" rows="14" spellcheck="false" placeholder="输入 DSL 因子表达式…"></textarea>
      <div class="lab-options">
        <label>因子名称 <input v-model="lab.factorName" type="text" placeholder="expr"></label>
        <div class="lab-date-row">
          <label>训练 <input v-model="lab.trainStart" type="date"> → <input v-model="lab.trainEnd" type="date"></label>
          <label>验证 <input v-model="lab.valStart" type="date"> → <input v-model="lab.valEnd" type="date"></label>
        </div>
      </div>
      <button class="lab-run-btn" :disabled="lab.busy || !lab.expr.trim()" @click="runEval">
        {{ lab.busy ? '评估中…' : '评估因子' }}
      </button>
      <div v-if="lab.results" class="lab-actions">
        <button v-if="anyPassed" class="lab-action-btn save" @click="openSaveDialog">保存到因子库</button>
        <button class="lab-action-btn export" @click="exportLabResult" :disabled="!lab.resultsJSON">导出评估结果</button>
        <button class="lab-action-btn bt" @click="openBacktestDialog">回测</button>
      </div>
      <p v-if="lab.error" class="lab-error">{{ lab.error }}</p>
    </div>
    <div class="lab-right">
      <div v-if="!lab.results && !lab.busy" class="lab-empty">
        输入 DSL 表达式后点击评估，将同时在 train_screen / validation / size_neutral_validation 三个 profile 上评估。
      </div>
      <div v-if="lab.busy" class="lab-loading">正在加载 panel 并评估（首次约 30-60 秒）…</div>
      <div v-if="lab.results" class="lab-results">
        <div v-if="anyPassed || lab.resultsJSON" class="lab-results-actions">
          <button v-if="anyPassed" class="lab-action-btn save" @click="openSaveDialog">保存到因子库</button>
          <button class="lab-action-btn export" @click="exportLabResult" :disabled="!lab.resultsJSON">导出评估结果</button>
          <button class="lab-action-btn bt" @click="openBacktestDialog">回测</button>
        </div>
        <div v-for="(result, profileId) in lab.results" :key="profileId" class="lab-result-card" :class="{ok: result.ok, bad: !result.ok}">
          <div class="lab-result-head">
            <strong>{{ profileId }}</strong>
            <span v-if="result.ok" class="lab-pass" :class="{pass: result.passed, fail: !result.passed}">
              {{ result.passed ? '通过' : '未通过' }}
            </span>
            <span v-if="result.date_range" class="lab-daterange">{{ result.date_range.start }} → {{ result.date_range.end }}</span>
          </div>
          <div v-if="!result.ok" class="lab-result-error">{{ result.error }}</div>
          <div v-if="result.ok && result.metrics" class="lab-metrics">
            <template v-for="(m, mid) in result.metrics" :key="mid">
              <div v-if="typeof m === 'number' || m == null" class="lab-metric">
                <span class="lab-metric-label">{{ metricLabel(mid) }}</span>
                <span class="lab-metric-value">{{ formatMetricValue(m) }}</span>
              </div>
            </template>
          </div>
          <!-- 分位组合（纯多头）指标：与因子库"多头年化/超额年化/夏普"列对齐 -->
          <div v-if="result.ok && quantilePortfolioMetrics(result)" class="lab-metrics lab-quantile-metrics">
            <div class="lab-metric">
              <span class="lab-metric-label">多头年化</span>
              <span class="lab-metric-value" :class="{neg: (quantilePortfolioMetrics(result).top_group_annualized_return ?? 0) < 0}">{{ formatMetricValue(quantilePortfolioMetrics(result).top_group_annualized_return) }}</span>
            </div>
            <div class="lab-metric">
              <span class="lab-metric-label">超额年化</span>
              <span class="lab-metric-value" :class="{neg: (quantilePortfolioMetrics(result).top_group_annualized_excess_return ?? 0) < 0}">{{ formatMetricValue(quantilePortfolioMetrics(result).top_group_annualized_excess_return) }}</span>
            </div>
            <div class="lab-metric">
              <span class="lab-metric-label">夏普</span>
              <span class="lab-metric-value" :class="{neg: (quantilePortfolioMetrics(result).top_group_sharpe ?? 0) < 0}">{{ formatMetricValue(quantilePortfolioMetrics(result).top_group_sharpe) }}</span>
            </div>
            <div class="lab-metric">
              <span class="lab-metric-label">最大回撤</span>
              <span class="lab-metric-value" :class="{neg: (quantilePortfolioMetrics(result).top_group_max_drawdown ?? 0) < 0}">{{ formatMetricValue(quantilePortfolioMetrics(result).top_group_max_drawdown) }}</span>
            </div>
          </div>
          <!-- 图表区 -->
          <div v-if="result.ok && result.chart_data" class="lab-charts">
            <div class="lab-chart-block">
              <div class="lab-chart-title">逐日 IC / RankIC</div>
              <div :id="'lab-ic-' + profileId" class="lab-chart-box"></div>
            </div>
            <div class="lab-chart-block">
              <div class="lab-chart-title">累计多空收益</div>
              <div :id="'lab-cum-' + profileId" class="lab-chart-box"></div>
            </div>
            <div class="lab-chart-block">
              <div class="lab-chart-title">月度 IC 热力图</div>
              <div :id="'lab-heat-' + profileId" class="lab-chart-box heatmap-box"></div>
            </div>
          </div>
          <details v-if="result.ok && result.rule_results?.length" class="lab-rules">
            <summary>规则详情 ({{ result.rule_results.filter(r => r.passed).length }}/{{ result.rule_results.length }})</summary>
            <div v-for="(rule, i) in result.rule_results" :key="i" class="lab-rule" :class="{pass: rule.passed, fail: !rule.passed}">
              <span>{{ rule.metric }} {{ rule.op }} {{ rule.expected }}</span>
              <span>{{ rule.actual != null ? formatMetricValue(rule.actual) : '—' }}</span>
              <span>{{ rule.passed ? '✓' : '✗' }}</span>
            </div>
          </details>
        </div>
      </div>
    </div>

    <!-- 保存因子弹窗 -->
    <Teleport to="body">
      <div v-if="lab.saveDialog" class="factor-modal-overlay" @click="lab.saveDialog = false">
        <div class="factor-modal" @click.stop style="width:min(460px,92vw)">
          <div class="factor-modal-head">
            <strong>保存到因子库</strong>
            <button class="factor-modal-close" @click="lab.saveDialog = false">×</button>
          </div>
          <div class="factor-modal-body">
            <div class="factor-modal-section">
              <label>因子名称</label>
              <input v-model="lab.saveName" type="text" class="lab-input" placeholder="factor_name">
            </div>
            <div class="factor-modal-section">
              <label>注释（经济含义/结构说明）</label>
              <textarea v-model="lab.saveComment" class="lab-input" rows="3" placeholder="描述因子的经济直觉和关键结构"></textarea>
            </div>
            <div class="factor-modal-section">
              <label>类别</label>
              <select v-model="lab.saveCategory" class="lab-input lab-cat-select">
                <option v-for="m in agent.researchModeOptions" :key="m.value" :value="m.value">{{ m.label }}</option>
              </select>
            </div>
            <div class="factor-modal-section">
              <label>目标库</label>
              <div class="lab-radio-group">
                <label><input type="radio" v-model="lab.saveLibrary" value="candidate"> 候选因子库</label>
                <label><input type="radio" v-model="lab.saveLibrary" value="production"> 正式因子库</label>
              </div>
            </div>
            <div v-if="lab.saveError" class="lab-save-error">{{ lab.saveError }}</div>
            <button class="lab-run-btn" :disabled="lab.saving" @click="saveFactor">
              {{ lab.saving ? '保存中…' : '确认保存' }}
            </button>
            <div v-if="lab.saveResult" class="lab-save-success">
              已保存到{{ lab.saveResult.library === 'production' ? '正式' : '候选' }}因子库，ID: {{ lab.saveResult.factor_id }}
            </div>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script>
import { api } from '../../utils/api.js'
import { chart } from '../../utils/charts.js'
import { agentStore } from '../../store/alphaagent.js'
import { store } from '../../store/index.js'
import { labSummary, metricLabel, formatMetricValue, quantilePortfolioMetrics } from '../../utils/alphaagent.js'

export default {
  name: 'FactorLab',
  emits: ['open-backtest'],
  data() {
    return {
      agent: agentStore,
      lab: {
        expr: '',
        factorName: 'expr',
        trainStart: '2020-01-01',
        trainEnd: '2022-12-31',
        valStart: '2023-01-01',
        valEnd: '2024-12-31',
        busy: false,
        error: '',
        results: null,
        resultsJSON: '',
        // 保存因子
        saveDialog: false,
        saveName: '',
        saveComment: '',
        saveLibrary: 'candidate',
        saveCategory: 'technical',
        saving: false,
        saveError: '',
        saveResult: null,
      },
    }
  },
  computed: {
    anyPassed() {
      if (!this.lab.results) return false
      return Object.values(this.lab.results).some(r => r.ok && r.passed)
    },
  },
  mounted() {
    this.applyWindowDefaults()
    // 从"历史"tab 打开一条因子评估 → 载入到因子实验室（界面与评估时一致）
    if (store.labLoadPayload) {
      const payload = store.labLoadPayload
      store.labLoadPayload = null
      this.restoreLabFromHistory(payload)
    }
  },
  watch: {
    // 页面加载早于窗口配置返回时，配置到位后再补应用一次（未应用过默认值时才生效）
    'agent.windowDefaults'() {
      if (!this._defaultsApplied) this.applyWindowDefaults()
    },
  },
  methods: {
    metricLabel,
    formatMetricValue,
    quantilePortfolioMetrics,
    // 统一时间窗口默认值来自后端配置中心（store.init 已拉取），仅覆盖初始值
    applyWindowDefaults() {
      const w = agentStore.windowDefaults
      if (!w) return
      this._defaultsApplied = true
      const set = (obj, key, val) => { if (val) obj[key] = val }
      set(this.lab, 'trainStart', w.train_start)
      set(this.lab, 'trainEnd', w.train_end)
      set(this.lab, 'valStart', w.val_start)
      set(this.lab, 'valEnd', w.val_end)
      // 因子实验室「回测」默认窗口：train 起点 ~ 数据源最新日
      set(this.lab, 'btStart', w.bt_start || w.val_start)
      set(this.lab, 'btEnd', w.bt_end || w.val_end)
    },
    async runEval() {
      if (!this.lab.expr.trim() || this.lab.busy) return
      this.lab.busy = true
      this.lab.error = ''
      this.lab.results = null
      const ctrl = new AbortController()
      const timer = setTimeout(() => ctrl.abort(), 600000)
      try {
        const data = await api('/api/alphaagent/eval-factor', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            multi_line_expr: this.lab.expr,
            factor_name: this.lab.factorName || 'expr',
            train_start: this.lab.trainStart,
            train_end: this.lab.trainEnd,
            val_start: this.lab.valStart,
            val_end: this.lab.valEnd,
            all_profiles: true,
          }),
          signal: ctrl.signal,
        })
        clearTimeout(timer)
        this.lab.results = data.results
        // 开启"导出评估结果"按钮：始终可导出最近一次评估（含 error 的 profile 也会附带）
        this.lab.resultsJSON = JSON.stringify({
          exported_at: new Date().toISOString(),
          factor_name: this.lab.factorName || 'expr',
          multi_line_expr: this.lab.expr,
          train_start: this.lab.trainStart,
          train_end: this.lab.trainEnd,
          val_start: this.lab.valStart,
          val_end: this.lab.valEnd,
          results: data.results,
        }, null, 2)
        // 写入评估历史（顶层"历史"tab 可见，可一键恢复到本界面）
        this.pushLabHistory(data.results)
        this.$nextTick(() => this.renderLabCharts())
      } catch (e) {
        clearTimeout(timer)
        this.lab.error = (e && e.name === 'AbortError')
          ? '评估超时（600 秒）。可缩短区间或减少股票池后重试。'
          : e.message
      } finally {
        this.lab.busy = false
      }
    },
    // 评估成功后写入全局历史（顶层"历史"tab 展示；localStorage 持久化）
    pushLabHistory(results) {
      try {
        const summary = labSummary(results)
        const entry = {
          id: 'lab_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8),
          kind: 'factor_eval',
          factorName: this.lab.factorName || 'expr',
          expr: this.lab.expr,
          trainStart: this.lab.trainStart,
          trainEnd: this.lab.trainEnd,
          valStart: this.lab.valStart,
          valEnd: this.lab.valEnd,
          createdAt: new Date().toISOString(),
          summary,
          results,
        }
        store.pushLabHistory(entry)
      } catch (e) {
        // 历史写入失败不影响评估结果展示
      }
    },
    // 从历史记录恢复因子实验室界面（与刚评估完完全一致）
    restoreLabFromHistory(h) {
      if (!h) return
      this.lab.factorName = h.factorName || 'expr'
      this.lab.expr = h.expr || ''
      this.lab.trainStart = h.trainStart || this.lab.trainStart
      this.lab.trainEnd = h.trainEnd || this.lab.trainEnd
      this.lab.valStart = h.valStart || this.lab.valStart
      this.lab.valEnd = h.valEnd || this.lab.valEnd
      this.lab.results = h.results || null
      this.lab.error = ''
      this.lab.resultsJSON = h.results ? JSON.stringify({
        exported_at: h.createdAt,
        factor_name: h.factorName || 'expr',
        multi_line_expr: h.expr || '',
        train_start: h.trainStart,
        train_end: h.trainEnd,
        val_start: h.valStart,
        val_end: h.valEnd,
        results: h.results,
      }, null, 2) : ''
      this.$nextTick(() => this.renderLabCharts())
    },
    openBacktestDialog() {
      this.$emit('open-backtest', {
        expr: this.lab.expr,
        factorName: this.lab.factorName || 'expr',
        valStart: this.lab.valStart || '2023-01-01',
        valEnd: this.lab.valEnd || '2024-12-31',
      })
    },
    async saveFactor() {
      this.lab.saving = true
      this.lab.saveError = ''
      this.lab.saveResult = null
      try {
        const data = await api('/api/alphaagent/factors', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            multi_line_expr: this.lab.expr,
            factor_name: this.lab.saveName || 'expr',
            comment: this.lab.saveComment,
            library: this.lab.saveLibrary,
            category: this.lab.saveCategory,
            train_start: this.lab.trainStart,
            train_end: this.lab.trainEnd,
            val_start: this.lab.valStart,
            val_end: this.lab.valEnd,
          }),
        })
        if (!data.ok) {
          this.lab.saveError = data.error || '保存失败'
        } else {
          this.lab.saveResult = data
        }
      } catch (e) {
        this.lab.saveError = e.message
      } finally {
        this.lab.saving = false
      }
    },
    openSaveDialog() {
      this.lab.saveName = this.lab.factorName || 'expr'
      this.lab.saveComment = ''
      this.lab.saveError = ''
      this.lab.saveResult = null
      this.lab.saveLibrary = 'candidate'
      this.lab.saveCategory = 'technical'
      this.lab.saveDialog = true
    },
    // ── 因子实验室：导出评估结果（与候选库 registry 格式对齐） ──
    exportLabResult() {
      if (!this.lab.results) return

      // 优先使用 validation 结果，其次用 train_screen
      const primaryResult = this.lab.results.validation || this.lab.results.train_screen
      if (!primaryResult || !primaryResult.ok) {
        alert('没有可用的评估结果，请确保至少有一个 profile 评估通过')
        return
      }

      // 与因子库/候选库对齐的夏普、年化字段（取自 quantile_portfolio 纯多头口径）
      const qp = this.quantilePortfolioMetrics(primaryResult) || {}
      const valQp = this.quantilePortfolioMetrics(this.lab.results.validation) || {}

      const factorId = this.lab.factorName || 'expr'
      const now = new Date().toISOString()

      // 构建与候选库一致的 registry 格式
      const payload = {
        exported_at: now,
        library: 'candidate',
        root: 'artifacts/alphaagent/factorzoo/candidate_1d',
        n_factors: 1,
        factors: [{
          factor_id: factorId,
          name: this.lab.factorName || 'expr',
          expr: this.lab.expr,
          label_col: this.lab.labelCol || 'label_1d_open_to_open',
          train_ic: this.lab.results.train_screen?.metrics?.cross_sectional_core?.ic,
          train_icir: this.lab.results.train_screen?.metrics?.cross_sectional_core?.icir,
          val_ic: this.lab.results.validation?.metrics?.cross_sectional_core?.ic,
          val_icir: this.lab.results.validation?.metrics?.cross_sectional_core?.icir,
          // ↓ 与因子库导出列对齐：多头年化/超额年化/夏普/回撤
          annualized_return: qp.top_group_annualized_return,
          annualized_excess_return: qp.top_group_annualized_excess_return,
          sharpe: qp.top_group_sharpe,
          max_drawdown: qp.top_group_max_drawdown,
          val_annualized_return: valQp.top_group_annualized_return,
          val_annualized_excess_return: valQp.top_group_annualized_excess_return,
          val_sharpe: valQp.top_group_sharpe,
          quantile_portfolio: primaryResult.metrics?.quantile_portfolio,
          metrics: primaryResult.metrics,
          status: 'pending_review',
          created_at: now,
        }],
        registry: {
          [factorId]: {
            factor_id: factorId,
            name: this.lab.factorName || 'expr',
            expr: this.lab.expr,
            comment: this.lab.factorName || 'expr',
            created_at: now,
            source: 'lab_export',
            panel_path: this.lab.labelCol || 'label_1d_open_to_open',
            train_period: `${this.lab.trainStart} → ${this.lab.trainEnd}`,
            val_period: `${this.lab.valStart} → ${this.lab.valEnd}`,
            evaluation_results: this.lab.results,
            metrics: primaryResult.metrics,
            quantile_portfolio: primaryResult.metrics?.quantile_portfolio,
            annualized_return: qp.top_group_annualized_return,
            annualized_excess_return: qp.top_group_annualized_excess_return,
            sharpe: qp.top_group_sharpe,
            max_drawdown: qp.top_group_max_drawdown,
            profile: primaryResult.profile,
            passed: primaryResult.passed,
            rule_results: primaryResult.rule_results,
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
      const dateStr = new Date().toISOString().slice(0, 10)
      a.download = `factor_candidate_${factorId}_${dateStr}.json`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(a.href)
    },
    // ── 因子实验室：图表渲染 ──
    renderLabCharts() {
      if (!this.lab.results) return
      for (const [profileId, result] of Object.entries(this.lab.results)) {
        if (!result.ok || !result.chart_data) continue
        const cd = result.chart_data

        // 逐日 IC / RankIC
        const icChart = chart('lab-ic-' + profileId)
        if (icChart) {
          const icSeries = []
          if (cd.daily_ic?.length) icSeries.push({ name: 'IC', dates: cd.daily_ic.map(p => p.date), values: cd.daily_ic.map(p => p.value) })
          if (cd.daily_rank_ic?.length) icSeries.push({ name: 'RankIC', dates: cd.daily_rank_ic.map(p => p.date), values: cd.daily_rank_ic.map(p => p.value) })
          if (icSeries.length) {
            icChart.setOption({
              tooltip: { trigger: 'axis' },
              legend: { textStyle: { color: '#8494b5' }, top: 0 },
              grid: { left: 44, right: 14, top: 28, bottom: 28 },
              xAxis: { type: 'category', data: icSeries[0].dates, axisLabel: { color: '#8494b5', fontSize: 9 } },
              yAxis: { type: 'value', axisLabel: { color: '#8494b5', fontSize: 9 } },
              series: icSeries.map(s => ({ name: s.name, type: 'line', data: s.values, showSymbol: false, smooth: true, lineStyle: { width: 1.5 } })),
            }, true)
          }
        }

        // 累计多空收益
        const cumChart = chart('lab-cum-' + profileId)
        if (cumChart) {
          const cum = cd.cumulative_long_short || []
          if (cum.length) {
            cumChart.setOption({
              tooltip: { trigger: 'axis' },
              grid: { left: 50, right: 14, top: 14, bottom: 28 },
              xAxis: { type: 'category', data: cum.map(p => p.date), axisLabel: { color: '#8494b5', fontSize: 9 } },
              yAxis: { type: 'value', axisLabel: { color: '#8494b5', fontSize: 9, formatter: v => (v * 100).toFixed(0) + '%' } },
              series: [{ type: 'line', data: cum.map(p => p.value), showSymbol: false, smooth: true, lineStyle: { width: 2 }, areaStyle: { color: 'rgba(79,140,255,0.12)' } }],
            }, true)
          }
        }

        // 月度 IC 热力图 — 用 monthly_ic 数据构建累计 IC 曲线点
        const heatChart = chart('lab-heat-' + profileId)
        if (heatChart) {
          const monthly = cd.monthly_ic || []
          if (monthly.length) {
            // 将月度 IC 均值转为 points 数组，复用 renderMonthlyHeatmap 的接口（需要 {date, value} 格式且 value 为累计净值）
            // 但 monthly_ic 是 mean 值不是累计净值，我们改用自定义热力图
            const years = [...new Set(monthly.map(m => m.month.slice(0, 4)))].sort()
            const months = ['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月']
            const yearIdx = Object.fromEntries(years.map((y, i) => [y, i]))
            const heatData = monthly.map(m => ({
              value: [Number(m.month.slice(5, 7)) - 1, yearIdx[m.month.slice(0, 4)], m.mean != null ? m.mean : 0]
            }))
            heatChart.setOption({
              tooltip: { formatter: p => `${years[p.value[1]]} · ${months[p.value[0]]}: <b>${p.value[2] != null ? Number(p.value[2]).toFixed(4) : '—'}</b>` },
              grid: { left: 44, right: 70, top: 18, bottom: 8 },
              xAxis: { type: 'category', position: 'top', data: months, axisLabel: { color: '#8494b5', fontSize: 9 } },
              yAxis: { type: 'category', data: years, inverse: true, axisLabel: { color: '#8494b5', fontSize: 9 } },
              visualMap: { type: 'piecewise', dimension: 2, pieces: [{ gt: 0, color: '#c94b55', label: '正IC' }, { value: 0, color: '#27344d', label: '0' }, { lt: 0, color: '#35b779', label: '负IC' }], orient: 'vertical', right: 0, top: 'middle', textStyle: { color: '#8494b5', fontSize: 9 } },
              series: [{ type: 'heatmap', data: heatData, label: { show: true, color: '#e6ecf7', fontSize: 8, formatter: p => p.value[2] != null ? Number(p.value[2]).toFixed(3) : '—' } }],
            }, true)
          }
        }
      }
    },
  },
}
</script>
