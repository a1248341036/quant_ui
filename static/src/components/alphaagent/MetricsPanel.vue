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
          <div class="summary-panel-head"><h3>汇总（{{ data.summary.n_runs }} 个 run）</h3></div>
          <div class="metrics-cards">
            <div class="metrics-card"><b>{{ fmt(data.summary.total_wall_minutes) }}<i>min</i></b><span>总时长</span></div>
            <div class="metrics-card"><b>{{ fmt(data.summary.total_input_k_tokens) }}<i>K</i></b><span>输入 tokens</span></div>
            <div class="metrics-card"><b>{{ fmt(data.summary.total_output_k_tokens) }}<i>K</i></b><span>输出 tokens</span></div>
            <div class="metrics-card"><b>{{ fmt(data.summary.total_thinking_k_chars) }}<i>K</i></b><span>思维链字符</span></div>
            <div class="metrics-card"><b>{{ fmt(data.summary.total_eval) }}</b><span>因子评估</span></div>
            <div class="metrics-card"><b>{{ data.summary.total_stored_candidate }}</b><span>入候选池</span></div>
            <div class="metrics-card"><b>{{ data.summary.total_stored_production }}</b><span>晋升正式库</span></div>
            <div class="metrics-card"><b>{{ data.summary.mean_minutes_per_delivered ?? '-' }}</b><span>min/产出</span></div>
          </div>
        </div>

        <!-- ── 漏斗 ── -->
        <div class="summary-panel">
          <div class="summary-panel-head"><h3>漏斗转化</h3></div>
          <div class="metrics-funnel">
            <div class="metrics-funnel-step"><b>{{ data.summary.total_eval }}</b><span>评估</span></div>
            <i>→</i>
            <div class="metrics-funnel-step"><b>{{ data.summary.total_submit }}</b><span>提交</span></div>
            <i>→</i>
            <div class="metrics-funnel-step"><b>{{ data.summary.total_stage_one_pass }}</b><span>stage_one</span></div>
            <i>→</i>
            <div class="metrics-funnel-step"><b>{{ data.summary.total_stage_two_pass }}</b><span>stage_two</span></div>
            <i>→</i>
            <div class="metrics-funnel-step"><b>{{ data.summary.total_gate_pass }}</b><span>engine_gate</span></div>
            <i>→</i>
            <div class="metrics-funnel-step" :class="{'metrics-funnel-hot': data.summary.total_stored_production > 0}">
              <b>{{ data.summary.total_stored_production }}</b><span>晋升</span>
            </div>
          </div>
        </div>

        <!-- ── 错误与记忆命中 ── -->
        <div class="summary-panel">
          <div class="summary-panel-head"><h3>工具错误分布（跨全部 run 聚合）</h3></div>
          <table v-if="errorRows.length" class="summary-table">
            <thead><tr><th>错误类型</th><th>次数</th></tr></thead>
            <tbody>
              <tr v-for="row in errorRows" :key="row[0]">
                <td><span class="metrics-err-tag" :class="{'metrics-err-real': isRealError(row[0])}">{{ row[0] }}</span></td>
                <td>{{ row[1] }}</td>
              </tr>
            </tbody>
          </table>
          <div v-else class="normal-mode-empty">无错误记录</div>
          <div class="summary-panel-head" style="margin-top:14px"><h3>记忆 advisory 命中</h3></div>
          <table v-if="advisoryRows.length" class="summary-table">
            <thead><tr><th>类型</th><th>次数</th></tr></thead>
            <tbody>
              <tr v-for="row in advisoryRows" :key="row[0]">
                <td>{{ advisoryLabel(row[0]) }}</td><td>{{ row[1] }}</td>
              </tr>
            </tbody>
          </table>
          <div v-else class="normal-mode-empty">无 advisory 记录</div>
        </div>

        <!-- ── Reviewer 校准 ── -->
        <div class="summary-panel">
          <div class="summary-panel-head">
            <h3>Reviewer 校准</h3>
            <span class="summary-facet-hint" title="lift = approve 存活率 − revise 存活率。接近 0 或为负 ⇒ Reviewer 意见对晋升没有预测力。">ⓘ</span>
          </div>
          <table class="summary-table">
            <thead><tr><th>review 意见</th><th>n</th><th>晋升</th><th>存活</th><th>gate 死</th><th>s2 死</th><th>存活率</th></tr></thead>
            <tbody>
              <tr v-for="(b, verdict) in data.reviewer_calibration.crosstab" :key="verdict">
                <td>{{ verdict }}</td><td>{{ b.n }}</td><td>{{ b.promoted }}</td><td>{{ b.candidate_alive }}</td>
                <td>{{ b.gate_failed }}</td><td>{{ b.stage_two_failed }}</td>
                <td>{{ b.alive_rate == null ? '-' : (b.alive_rate * 100).toFixed(0) + '%' }}</td>
              </tr>
              <tr v-if="!Object.keys(data.reviewer_calibration.crosstab).length">
                <td colspan="7" class="normal-mode-empty">尚无入库因子</td>
              </tr>
            </tbody>
          </table>
          <p class="metrics-cal-line">
            lift = {{ fmt(data.reviewer_calibration.calibration.approve_alive_rate) }} −
            {{ fmt(data.reviewer_calibration.calibration.revise_alive_rate) }} =
            <b>{{ data.reviewer_calibration.calibration.lift ?? '-' }}</b>
          </p>
        </div>

        <!-- ── 每 run 明细 ── -->
        <div class="summary-panel">
          <div class="summary-panel-head"><h3>Run 明细（新 → 旧）</h3></div>
          <table class="summary-table metrics-run-table">
            <thead>
              <tr><th>run</th><th>时长min</th><th>LLM次</th><th>输入K</th><th>输出K</th><th>缓存率</th>
                  <th>思维链K</th><th>评估</th><th>提交</th><th>入库</th><th>晋升</th><th>错误率</th><th>min/产出</th></tr>
            </thead>
            <tbody>
              <tr v-for="r in data.runs" :key="r.run_id">
                <td class="metrics-run-id" :title="r.run_id">{{ r.run_id.slice(0, 8) }}</td>
                <td>{{ fmt(r.wall_minutes) }}</td>
                <td>{{ r.llm_calls }}</td>
                <td>{{ fmt(r.input_k_tokens) }}</td>
                <td>{{ fmt(r.output_k_tokens) }}</td>
                <td>{{ r.cache_hit_rate ? (r.cache_hit_rate * 100).toFixed(0) + '%' : '-' }}</td>
                <td>{{ fmt(r.thinking_k_chars) }}</td>
                <td>{{ r.n_eval + r.n_eval_val }}</td>
                <td>{{ r.n_submit }}</td>
                <td>{{ r.stored_candidate }}</td>
                <td>{{ r.stored_production }}</td>
                <td>{{ r.tool_error_rate == null ? '-' : (r.tool_error_rate * 100).toFixed(0) + '%' }}</td>
                <td>{{ r.minutes_per_delivered ?? '-' }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </template>
    </div>
  </main>
</template>

<script>
import { api } from '../../utils/api.js'

export default {
  name: 'MetricsPanel',
  data() {
    return { data: null, loading: false, error: '', lastN: 20 }
  },
  computed: {
    errorRows() { return Object.entries(this.data?.summary?.error_breakdown || {}) },
    advisoryRows() { return Object.entries(this.data?.summary?.advisory_breakdown || {}) },
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
      } catch (e) {
        this.error = String(e.message || e)
      } finally {
        this.loading = false
      }
    },
  },
}
</script>
