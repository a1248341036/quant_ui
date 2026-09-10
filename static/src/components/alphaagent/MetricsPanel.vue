<template>
  <main class="agent-main research-summary-page">
    <header class="agent-header">
      <div class="agent-title">
        <span class="agent-orb">✦</span>
        <div>
          <h1>整体统计</h1>
          <p class="agent-subtitle">挖掘效率 · 漏斗转化 · 错误与记忆命中 · Reviewer 校准</p>
        </div>
      </div>
      <div class="header-actions">
        <select v-model.number="lastN" class="metrics-last-select" @change="refresh">
          <option :value="10">最近 10 个 run</option>
          <option :value="20">最近 20 个 run</option>
          <option :value="50">最近 50 个 run</option>
          <option :value="0">全部 run</option>
        </select>
        <button class="summary-refresh-btn" :disabled="loading" @click="refresh">刷新</button>
      </div>
    </header>

    <div class="agent-thread">
      <div v-if="error" class="normal-mode-empty">统计不可用：{{ error }}</div>
      <div v-else-if="loading && !data" class="normal-mode-empty">加载统计中…</div>
      <template v-else-if="data">
        <!-- ── 汇总卡片 ── -->
        <div class="summary-panel">
          <div class="summary-panel-head"><h3>汇总（{{ data.summary.n_runs ?? 0 }} 个 run）</h3></div>
          <div class="metrics-cards">
            <div class="metrics-card"><b>{{ fmt(data.summary.total_wall_minutes ?? 0) }}<i>min</i></b><span>总时长</span></div>
            <div class="metrics-card"><b>{{ fmt(data.summary.total_input_k_tokens ?? 0) }}<i>K</i></b><span>输入 tokens</span></div>
            <div class="metrics-card"><b>{{ fmt(data.summary.total_output_k_tokens ?? 0) }}<i>K</i></b><span>输出 tokens</span></div>
            <div class="metrics-card"><b>{{ fmt(data.summary.total_thinking_k_chars ?? 0) }}<i>K</i></b><span>思维链字符</span></div>
            <div class="metrics-card"><b>{{ data.summary.total_eval ?? 0 }}</b><span>因子评估</span></div>
            <div class="metrics-card"><b>{{ data.summary.total_submit ?? 0 }}</b><span>提交</span></div>
            <div class="metrics-card"><b>{{ data.summary.total_stored_candidate ?? 0 }}</b><span>入候选池</span></div>
            <div class="metrics-card"><b>{{ data.summary.total_stored_production ?? 0 }}</b><span>晋升正式库</span></div>
            <div class="metrics-card"><b>{{ data.summary.mean_minutes_per_delivered == null ? '-' : fmt(data.summary.mean_minutes_per_delivered) }}</b><span>min/产出</span></div>
          </div>
        </div>

        <!-- ── 漏斗转化（echarts 漏斗图） ── -->
        <div class="summary-panel">
          <div class="summary-panel-head">
            <h3>漏斗转化</h3>
            <span class="summary-facet-hint" title="每层标注相对上一层的转化率；stage_one/stage_two/engine_gate 为提交后各门槛的通过数。">ⓘ</span>
          </div>
          <div id="metrics-funnel-chart" class="metrics-chart" :style="{ height: funnelHeight + 'px' }"></div>
        </div>

        <!-- ── 数据面 / 算子 成功率（研究记忆库聚合） ── -->
        <div class="summary-panel">
          <div class="summary-panel-head">
            <h3>数据面 / 算子 成功率</h3>
            <div class="metrics-split-controls">
              <select v-model="facetMetric" class="metrics-last-select" @change="renderFacetOperatorCharts">
                <option value="rate">过线率</option>
                <option value="stored_rate">入库率</option>
                <option value="mean_abs_ic">平均 |IC|</option>
              </select>
              <span class="summary-facet-hint" :title="facetHint">ⓘ</span>
            </div>
          </div>
          <div v-if="facetStatsError" class="normal-mode-empty">统计不可用：{{ facetStatsError }}</div>
          <div v-else-if="!facetStats" class="normal-mode-empty">
            统计未就绪——后端需重启以加载新的 /metrics/overview 字段
          </div>
          <template v-else>
            <div class="metrics-split">
              <div class="metrics-split-col">
                <div class="metrics-split-title">按数据面<span>{{ facetScopeText }}</span></div>
                <div id="metrics-facet-chart" class="metrics-chart" :style="{ height: facetChartHeight + 'px' }"></div>
              </div>
              <div class="metrics-split-col">
                <div class="metrics-split-title">按算子<span>{{ opScopeText }}</span></div>
                <div id="metrics-op-chart" class="metrics-chart" :style="{ height: opChartHeight + 'px' }"></div>
              </div>
            </div>
            <p class="metrics-cal-line">
              {{ facetMetricLabel }} 口径：过线 = promising/validated/candidate_approved/production_approved；
              分母为有效尝试（已剔除评估未产出的 eval_error {{ facetStats?.scope?.n_eval_error ?? 0 }} 次）；
              虚线 = 整体基线 {{ fmtMetric(facetStats?.baseline?.[facetMetric]) }}；样本 &lt; {{ minAttempts }} 次的桶不列入。
            </p>
          </template>
        </div>

        <!-- ── 错误与记忆命中（横向条形图，次数标在条上） ── -->
        <div class="summary-panel">
          <div class="summary-panel-head">
            <h3>工具错误分布（跨全部 run 聚合）</h3>
            <span class="summary-facet-hint" title="红色 = 真错误（参数/求值失败）；灰色 = 门槛拒绝（正常流程）。条上数字为次数。">ⓘ</span>
          </div>
          <div v-if="errorRows.length" id="metrics-error-chart" class="metrics-chart" :style="{ height: Math.max(120, errorRows.length * 30 + 40) + 'px' }"></div>
          <div v-else class="normal-mode-empty">无错误记录</div>
          <div class="summary-panel-head" style="margin-top:14px"><h3>记忆 advisory 命中</h3></div>
          <div v-if="advisoryRows.length" id="metrics-advisory-chart" class="metrics-chart" :style="{ height: Math.max(100, advisoryRows.length * 30 + 40) + 'px' }"></div>
          <div v-else class="normal-mode-empty">无 advisory 记录</div>
        </div>

        <!-- ── Reviewer 校准 ── -->
        <div class="summary-panel">
          <div class="summary-panel-head">
            <h3>Reviewer 校准</h3>
            <span class="summary-facet-hint" title="lift = approve 存活率 − revise 存活率。接近 0 或为负 ⇒ Reviewer 意见对晋升没有预测力。">ⓘ</span>
          </div>
          <table class="summary-table">
            <thead><tr><th>review 意见</th><th>n</th><th>晋升</th><th>存活</th><th>gate 死</th><th>s2 死</th><th style="min-width:140px">存活率</th></tr></thead>
            <tbody>
              <tr v-for="(b, verdict) in data.reviewer_calibration.crosstab" :key="verdict">
                <td>{{ verdict }}</td><td>{{ b.n }}</td><td>{{ b.promoted }}</td><td>{{ b.candidate_alive }}</td>
                <td>{{ b.gate_failed }}</td><td>{{ b.stage_two_failed }}</td>
                <td>
                  <div class="metrics-rate-row">
                    <div class="metrics-rate-bar"><i :style="{ width: rateBarWidth(b.alive_rate) }" :class="rateBarClass(b.alive_rate)"></i></div>
                    <span>{{ b.alive_rate == null ? '-' : (b.alive_rate * 100).toFixed(0) + '%' }}</span>
                  </div>
                </td>
              </tr>
              <tr v-if="!Object.keys(data.reviewer_calibration.crosstab).length">
                <td colspan="7" class="normal-mode-empty">尚无入库因子</td>
              </tr>
            </tbody>
          </table>
          <p class="metrics-cal-line">
            lift = {{ fmt(data.reviewer_calibration.calibration.approve_alive_rate ?? 0) }} −
            {{ fmt(data.reviewer_calibration.calibration.revise_alive_rate ?? 0) }} =
            <b>{{ data.reviewer_calibration.calibration.lift ?? '-' }}</b>
          </p>
        </div>

        <!-- ── 每 run 明细（图表固定紧凑高度 + 表格内部滚动，避免按 run 数线性撑高页面） ── -->
        <div class="summary-panel">
          <div class="summary-panel-head">
            <h3>Run 明细（新 → 旧）</h3>
            <span v-if="(data.runs || []).length > runsChartCap" class="summary-facet-hint">
              图表仅显示最近 {{ runsChartCap }} 个 run，完整明细见下表
            </span>
          </div>
          <div id="metrics-runs-chart" class="metrics-chart" :style="{ height: runsChartHeight + 'px' }"></div>
          <div class="summary-table-wrap metrics-run-wrap" style="margin-top:14px">
          <table class="summary-table metrics-run-table">
            <thead>
              <tr><th>run</th><th>时长min</th><th>LLM次</th><th>输入K</th><th>输出K</th><th>缓存率</th>
                  <th>思维链K</th><th>评估</th><th>提交</th><th>入库</th><th>晋升</th><th>错误率</th><th>min/产出</th></tr>
            </thead>
            <tbody>
              <tr v-for="r in data.runs" :key="r.run_id">
                <td class="metrics-run-id" :title="r.run_id">{{ r.run_id.slice(0, 8) }}</td>
                <td>{{ fmt(r.wall_minutes ?? 0) }}</td>
                <td>{{ r.llm_calls ?? 0 }}</td>
                <td>{{ fmt(r.input_k_tokens ?? 0) }}</td>
                <td>{{ fmt(r.output_k_tokens ?? 0) }}</td>
                <td>{{ r.cache_hit_rate ? (r.cache_hit_rate * 100).toFixed(0) + '%' : '-' }}</td>
                <td>{{ fmt(r.thinking_k_chars ?? 0) }}</td>
                <td>{{ (r.n_eval ?? 0) + (r.n_eval_val ?? 0) }}</td>
                <td>{{ r.n_submit ?? 0 }}</td>
                <td>{{ r.stored_candidate ?? 0 }}</td>
                <td>{{ r.stored_production ?? 0 }}</td>
                <td>{{ r.tool_error_rate == null ? '-' : (r.tool_error_rate * 100).toFixed(0) + '%' }}</td>
                <td>{{ r.minutes_per_delivered ?? '-' }}</td>
              </tr>
            </tbody>
          </table>
          </div>
        </div>
      </template>
    </div>
  </main>
</template>

<script>
import { api } from '../../utils/api.js'
import { chart } from '../../utils/charts.js'

const AXIS_LABEL = { color: '#8494b5', fontSize: 10 }

export default {
  name: 'MetricsPanel',
  data() {
    return { data: null, loading: false, error: '', lastN: 0, facetMetric: 'rate' }
  },
  computed: {
    errorRows() { return Object.entries(this.data?.summary?.error_breakdown || {}) },
    advisoryRows() { return Object.entries(this.data?.summary?.advisory_breakdown || {}) },
    facetStats() { return this.data?.facet_operator || null },
    facetStatsError() { return this.facetStats?.error || '' },
    facetRows() { return this.facetStats?.facets || [] },
    opRows() { return this.facetStats?.operators || [] },
    minAttempts() { return this.facetStats?.min_attempts ?? 8 },
    facetChartHeight() { return Math.max(150, this.facetRows.length * 22 + 46) },
    opChartHeight() { return Math.max(150, this.opRows.length * 22 + 46) },
    facetMetricLabel() {
      return ({ rate: '过线率', stored_rate: '入库率', mean_abs_ic: '平均 |IC|' })[this.facetMetric]
    },
    facetScopeText() {
      const s = this.facetStats?.scope
      if (!s) return ''
      const scope = s.filtered_runs ? `最近 ${s.filtered_runs} 个 run` : '全部 run'
      return ` · ${scope} · 有效尝试 ${s.n_valid}`
    },
    opScopeText() {
      const s = this.facetStats?.scope
      if (!s) return ''
      const shown = this.opRows.length
      const total = s.n_buckets_operators || shown
      return total > shown ? ` · 尝试数前 ${shown}/${total} 个算子` : ` · ${shown} 个算子`
    },
    facetHint() {
      return '数据面来自因子表达式的列族标签（跨面因子在每个触及面下各计一次，'
        + '另有「跨面融合」聚合桶）；算子按表达式实际调用去重计数。'
        + '过线率分母剔除 eval_error（面板缺列/超时等"没算出来"的尝试）。'
        + '切换页面上方 run 窗口会同步改变统计范围。'
    },
    funnelHeight() {
      return 60 + 6 * 46
    },
    // Run 明细图固定紧凑高度：不再按 run 数线性增长(50 个 run 曾撑到 1360px)
    runsChartHeight() {
      return 180
    },
    runsChartCap() {
      return 24
    },
  },
  mounted() { this.refresh() },
  methods: {
    fmt(v) {
      const n = Number(v)
      return isNaN(n) ? String(v ?? '-') : Math.abs(n) >= 100 ? n.toFixed(0) : n.toFixed(1)
    },
    isRealError(kind) {
      // 参数/契约违规与求值失败是真问题；Stage/Gate/Blind 类是门槛拒绝（正常流程）
      return /ToolArguments|Eval|Value/.test(kind)
    },
    rateBarWidth(rate) {
      return rate == null ? '0%' : Math.min(100, Math.max(0, rate * 100)).toFixed(0) + '%'
    },
    rateBarClass(rate) {
      if (rate == null) return ''
      return rate >= 0.5 ? 'rate-good' : rate >= 0.25 ? 'rate-mid' : 'rate-bad'
    },
    renderCharts() {
      if (!window.echarts || !this.data) return
      this.renderFunnel()
      this.renderFacetOperatorCharts()
      this.renderErrorChart('metrics-error-chart', this.errorRows, true)
      this.renderErrorChart('metrics-advisory-chart', this.advisoryRows, false)
      this.renderRunsChart()
    },
    fmtMetric(v) {
      if (v == null) return '-'
      return this.facetMetric === 'mean_abs_ic'
        ? Number(v).toFixed(4)
        : (Number(v) * 100).toFixed(1) + '%'
    },
    // ── 数据面 / 算子 成功率（横向条形，虚线 = 整体基线）──
    renderFacetOperatorCharts() {
      if (!window.echarts || !this.data) return
      const stats = this.facetStats
      if (!stats || stats.error) return
      this.renderBreakdownChart('metrics-facet-chart', this.facetRows)
      this.renderBreakdownChart('metrics-op-chart', this.opRows)
    },
    renderBreakdownChart(id, rows) {
      if (!rows.length) return
      const metric = this.facetMetric
      const isPct = metric !== 'mean_abs_ic'
      const val = r => {
        const v = r[metric]
        if (v == null) return 0
        return isPct ? Number(v) * 100 : Number(v)
      }
      const sorted = [...rows].sort((a, b) => val(a) - val(b))
      const c = chart(id)
      if (!c) return
      const baseRaw = this.facetStats?.baseline?.[metric]
      const baseVal = baseRaw == null ? null : (isPct ? Number(baseRaw) * 100 : Number(baseRaw))
      const maxV = Math.max(...sorted.map(val), baseVal || 0, isPct ? 1 : 0.0001)
      const pct = v => (v == null ? '-' : (Number(v) * 100).toFixed(1) + '%')
      c.setOption({
        tooltip: {
          trigger: 'item',
          formatter: p => {
            const r = sorted[p.dataIndex]
            return `${r.name}<br/>过线率 <b>${pct(r.rate)}</b>（${r.positive}/${r.n_valid}）`
              + `<br/>入库率 ${pct(r.stored_rate)}（${r.stored} 个）`
              + `<br/>平均 |IC| ${r.mean_abs_ic == null ? '-' : Number(r.mean_abs_ic).toFixed(4)}`
              + `<br/>尝试 ${r.n} 次（eval_error ${r.n_eval_error}）`
          },
        },
        grid: { left: 100, right: 78, top: 8, bottom: 6 },
        xAxis: {
          type: 'value', max: maxV * 1.18,
          axisLabel: { ...AXIS_LABEL, formatter: v => (isPct ? v.toFixed(0) + '%' : v) },
          splitLine: { lineStyle: { color: '#1c2536' } },
        },
        yAxis: {
          type: 'category', data: sorted.map(r => r.name),
          axisLabel: { ...AXIS_LABEL, width: 94, overflow: 'truncate' },
        },
        series: [{
          type: 'bar', data: sorted.map(val), barMaxWidth: 14,
          itemStyle: {
            borderRadius: [0, 3, 3, 0],
            color: p => this.breakdownColor(val(sorted[p.dataIndex]), maxV, baseVal),
          },
          label: {
            show: true, position: 'right', color: '#c6d2e8', fontSize: 10,
            formatter: p => {
              const r = sorted[p.dataIndex]
              const shown = isPct ? val(r).toFixed(1) + '%' : val(r).toFixed(4)
              return `${shown} (n=${r.n_valid})`
            },
          },
          markLine: baseVal == null ? undefined : {
            silent: true, symbol: 'none',
            lineStyle: { color: '#8ab4d8', type: 'dashed', width: 1 },
            label: {
              formatter: `基线 ${isPct ? baseVal.toFixed(1) + '%' : baseVal.toFixed(4)}`,
              color: '#8ab4d8', fontSize: 9, position: 'insideEndTop',
            },
            data: [{ xAxis: baseVal }],
          },
        }],
      }, true)
    },
    breakdownColor(v, maxV, baseVal) {
      if (!(v > 0)) return '#5a6478'
      const ref = baseVal && baseVal > 0 ? baseVal : maxV
      const ratio = ref > 0 ? v / ref : 1
      if (ratio >= 1.3) return '#4fc3a1'
      if (ratio >= 0.85) return '#4f8cff'
      return '#7d8bab'
    },
    renderFunnel() {
      const s = this.data.summary || {}
      const stages = [
        { name: '评估', value: s.total_eval ?? 0 },
        { name: '提交', value: s.total_submit ?? 0 },
        { name: 'stage_one', value: s.total_stage_one_pass ?? 0 },
        { name: 'stage_two', value: s.total_stage_two_pass ?? 0 },
        { name: 'engine_gate', value: s.total_gate_pass ?? 0 },
        { name: '晋升', value: s.total_stored_production ?? 0 },
      ]
      const c = chart('metrics-funnel-chart')
      if (!c) return
      c.setOption({
        tooltip: {
          trigger: 'item',
          formatter: p => `${p.name}：<b>${p.value}</b>（占评估 ${(s.total_eval ? (p.value / s.total_eval * 100).toFixed(1) : 0)}%）`,
        },
        series: [{
          type: 'funnel',
          left: '8%', right: '18%', top: 8, bottom: 8,
          minSize: '6%',
          sort: 'descending',
          gap: 3,
          label: {
            show: true, position: 'inside', color: '#e6ecf7', fontSize: 11,
            formatter: p => {
              const idx = stages.findIndex(x => x.name === p.name)
              const prev = idx > 0 ? stages[idx - 1].value : null
              const rate = (prev != null && prev > 0) ? ` · ${(p.value / prev * 100).toFixed(0)}%` : ''
              return `${p.name} ${p.value}${rate}`
            },
          },
          itemStyle: { borderColor: '#1c2536', borderWidth: 1 },
          data: stages.map((st, i) => ({
            name: st.name, value: st.value,
            itemStyle: { color: ['#4f8cff', '#5aa2e8', '#4fc3a1', '#a3d977', '#e8c491', '#ef6b73'][i] },
          })),
        }],
      }, true)
    },
    renderErrorChart(id, rows, isError) {
      if (!rows.length) return
      const sorted = [...rows].sort((a, b) => a[1] - b[1])
        .map(([k, v]) => [isError ? k : this.advisoryLabel(k), v])
      const c = chart(id)
      if (!c) return
      c.setOption({
        tooltip: { trigger: 'item', formatter: p => `${p.name}：<b>${p.value}</b> 次` },
        grid: { left: 170, right: 48, top: 6, bottom: 6 },
        xAxis: { type: 'value', axisLabel: { ...AXIS_LABEL }, splitLine: { lineStyle: { color: '#1c2536' } } },
        yAxis: { type: 'category', data: sorted.map(r => r[0]), axisLabel: { ...AXIS_LABEL, width: 160, overflow: 'truncate' } },
        series: [{
          type: 'bar', data: sorted.map(r => r[1]),
          barMaxWidth: 16,
          label: { show: true, position: 'right', color: '#c6d2e8', fontSize: 10 },
          itemStyle: {
            borderRadius: [0, 3, 3, 0],
            color: isError
              ? p => (this.isRealError(p.name) ? '#ef6b73' : '#4a5a78')
              : '#4fc3a1',
          },
        }],
      }, true)
    },
    renderRunsChart() {
      const runsAll = this.data.runs || []
      if (!runsAll.length) return
      // 图表只画最近 runsChartCap 个（柱宽可读）；全量交给下方可滚动表格
      const runs = runsAll.slice(0, this.runsChartCap)
      const names = runs.map(r => r.run_id.slice(0, 6))
      const c = chart('metrics-runs-chart')
      if (!c) return
      c.setOption({
        tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
        legend: { textStyle: { color: '#8494b5', fontSize: 10 }, top: 0 },
        grid: { left: 44, right: 14, top: 26, bottom: 28 },
        xAxis: { type: 'category', data: names, axisLabel: { ...AXIS_LABEL, rotate: 40, interval: 'auto', hideOverlap: true } },
        yAxis: { type: 'value', axisLabel: { ...AXIS_LABEL } },
        series: [
          { name: '评估', type: 'bar', stack: 'm', data: runs.map(r => (r.n_eval ?? 0) + (r.n_eval_val ?? 0)), itemStyle: { color: '#4f8cff' }, barMaxWidth: 14 },
          { name: '提交', type: 'bar', stack: 'm', data: runs.map(r => r.n_submit ?? 0), itemStyle: { color: '#e8c491' }, barMaxWidth: 14 },
          { name: '入库', type: 'bar', stack: 'm', data: runs.map(r => (r.stored_candidate ?? 0) + (r.stored_production ?? 0)), itemStyle: { color: '#4fc3a1' }, barMaxWidth: 14 },
        ],
      }, true)
    },
    advisoryLabel(kind) {
      return ({
        duplicate_known_dead_end: '重复死路结构提醒',
        edit_veto: '意向编辑否决',
        duplicate_prior_result: '正向结构重复',
      })[kind] || kind
    },
    async refresh() {
      this.loading = true
      this.error = ''
      try {
        this.data = await api(`/api/alphaagent/metrics/overview?last=${this.lastN}`)
        this.$nextTick(() => this.renderCharts())
      } catch (e) {
        this.error = String(e.message || e)
      } finally {
        this.loading = false
      }
    },
  },
}
</script>
