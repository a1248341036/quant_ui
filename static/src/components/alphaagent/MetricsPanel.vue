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
        <button
          class="metrics-poll-toggle"
          :class="{ active: autoPoll }"
          @click="toggleAutoPoll"
          :title="autoPoll ? '已开启 8s 轻量自动轮询更新，点击暂停' : '点击开启 8s 自动轮询更新'"
        >
          <span class="metrics-poll-dot" :class="{ pulse: autoPoll }"></span>
          {{ autoPoll ? '自动更新中' : '自动更新已暂停' }}
        </button>
        <select v-model.number="lastN" class="metrics-last-select" @change="refresh(false)">
          <option :value="10">最近 10 个 run</option>
          <option :value="20">最近 20 个 run</option>
          <option :value="50">最近 50 个 run</option>
          <option :value="0">全部 run</option>
        </select>
        <button class="summary-refresh-btn" :disabled="loading || refreshing" @click="refresh(false)">刷新</button>
      </div>
    </header>

    <div class="agent-thread">
      <div v-if="error" class="normal-mode-empty">统计不可用：{{ error }}</div>
      <div v-else-if="loading && !data" class="normal-mode-empty">加载统计中…</div>
      <template v-else-if="data">
        <!-- ── 汇总卡片 ── -->
        <div class="summary-panel">
          <div class="summary-panel-head"><h3>汇总（{{ data.summary.n_runs ?? 0 }} 个 run 的成本 · 因子评估/提交/入库为研究记忆库全库口径）</h3></div>
          <div class="metrics-cards">
            <div class="metrics-card"><b>{{ fmt(data.summary.total_wall_minutes ?? 0) }}<i>min</i></b><span>总时长</span></div>
            <div class="metrics-card"><b>{{ fmt(data.summary.total_input_k_tokens ?? 0) }}<i>K</i></b><span>输入 tokens</span></div>
            <div class="metrics-card"><b>{{ fmt(data.summary.total_output_k_tokens ?? 0) }}<i>K</i></b><span>输出 tokens</span></div>
            <div class="metrics-card"><b>{{ data.summary.total_cache_hit_rate == null ? '-' : (data.summary.total_cache_hit_rate * 100).toFixed(0) }}<i>%</i></b><span>缓存命中率</span></div>
            <div class="metrics-card"><b>{{ fmt(data.summary.total_thinking_k_chars ?? 0) }}<i>K</i></b><span>思维链字符</span></div>
            <div class="metrics-card"><b>{{ fmt(data.research_funnel?.total ?? 0) }}</b><span>因子评估</span></div>
            <div class="metrics-card"><b>{{ fmt(data.research_funnel?.submitted ?? 0) }}</b><span>发起提交</span></div>
            <div class="metrics-card"><b>{{ data.research_funnel?.candidate_stored ?? 0 }}</b><span>入候选池</span></div>
            <div class="metrics-card"><b>{{ data.research_funnel?.production_stored ?? 0 }}</b><span>晋升正式库</span></div>
            <div class="metrics-card"><b>{{ data.summary.mean_minutes_per_delivered == null ? '-' : fmt(data.summary.mean_minutes_per_delivered) }}</b><span>min/产出</span></div>
          </div>
          <div v-if="data.summary.v4_aggregates" class="metrics-cards" style="margin-top: 10px;">
            <div class="metrics-card"><b>{{ data.summary.v4_aggregates?.median_effective_novelty_rate != null ? (data.summary.v4_aggregates.median_effective_novelty_rate * 100).toFixed(1) + '%' : '-' }}</b><span>有效新颖率(中位)</span></div>
            <div class="metrics-card"><b>{{ data.summary.v4_aggregates?.median_facet_coverage ?? '-' }}</b><span>面覆盖数(中位)</span></div>
            <div class="metrics-card"><b>{{ data.summary.v4_aggregates?.median_family_coverage ?? '-' }}</b><span>信号族覆盖(中位)</span></div>
            <div class="metrics-card"><b>{{ data.summary.v4_aggregates?.mean_prediction_coverage != null ? (data.summary.v4_aggregates.mean_prediction_coverage * 100).toFixed(1) + '%' : '-' }}</b><span>预测对账率(均值)</span></div>
            <div class="metrics-card"><b>{{ data.summary.v4_aggregates?.mean_temporal_stability != null ? (data.summary.v4_aggregates.mean_temporal_stability * 100).toFixed(1) + '%' : '-' }}</b><span>月度稳定性</span></div>
            <div class="metrics-card"><b>{{ data.summary.v4_aggregates?.mean_unsubmitted_passing_rate != null ? (data.summary.v4_aggregates.mean_unsubmitted_passing_rate * 100).toFixed(1) + '%' : '-' }}</b><span>未交付过线率</span></div>
          </div>
        </div>

        <!-- ── ABCD 评测指标卡组（随每个 Run 动态变化 · 迷你走势线） ── -->
        <div class="summary-panel">
          <div class="summary-panel-head">
            <h3>评测维度矩阵 A–F（卡片随 Run 动态变化 · 点击指标直达下方折线图）</h3>
            <span class="summary-facet-hint">每行展示该指标在最新 Run 的数值、相对上一 Run 的 Δ 增量，以及随历史各个 Run 演进的迷你走势线 (Sparkline)</span>
          </div>
          <div class="abcd-cards-grid">
            <div v-for="grp in metricGroups" :key="grp.id" class="abcd-group-card">
              <div class="abcd-group-head">
                <h4><span class="abcd-group-badge">{{ grp.id }}</span> {{ grp.name }}</h4>
              </div>
              <div class="abcd-group-body">
                <div
                  v-for="m in grp.metrics"
                  :key="m.key"
                  class="abcd-metric-row"
                  :class="{ active: trendMetric === m.key }"
                  @click="selectTrendMetric(m.key)"
                  :title="'点击在下方折线图查看 ' + m.name + ' 的逐 Run 趋势'"
                >
                  <span class="abcd-metric-name">{{ m.name }}</span>
                  <span class="abcd-metric-val">{{ formatMetricVal(m, getSpark(m.key).lastVal) }}</span>
                  <span
                    class="abcd-metric-delta"
                    :class="deltaClass(m, getSpark(m.key).delta)"
                  >
                    {{ formatDelta(m, getSpark(m.key).delta) }}
                  </span>
                  <div class="abcd-metric-spark">
                    <svg v-if="getSpark(m.key).points" viewBox="0 0 76 18" style="width:100%;height:100%;overflow:visible;">
                      <polyline
                        :points="getSpark(m.key).points"
                        fill="none"
                        :stroke="trendMetric === m.key ? '#4fc3a1' : '#4f8cff'"
                        stroke-width="1.8"
                        stroke-linecap="round"
                        stroke-linejoin="round"
                      />
                      <circle
                        v-if="getSpark(m.key).lastPoint"
                        :cx="getSpark(m.key).lastPoint.x"
                        :cy="getSpark(m.key).lastPoint.y"
                        r="2.5"
                        :fill="deltaColor(m, getSpark(m.key).delta)"
                      />
                    </svg>
                    <span v-else style="color:var(--muted);font-size:10px;">—</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- ── 漏斗转化（echarts 漏斗图） ── -->
        <div class="summary-panel">
          <div class="summary-panel-head">
            <h3>漏斗转化</h3>
            <span class="summary-facet-hint" title="口径 = 研究记忆库全量（Web/CLI/整夜全渠道），跨 run 稳定、不随上方 run 窗口变化。评估尝试 = 记忆库因子结构数；有效尝试剔除 eval_error（面板缺列/超时等没算出来的）；海选过线 = promising/validated/入库；发起提交 = 出现过 submit 阶段的去重因子；入候选池/晋升取因子库 registry 条目数（与因子库页一致）。每层标注相对上一层的转化率。">ⓘ</span>
          </div>
          <div id="metrics-funnel-chart" class="metrics-chart" :style="{ height: funnelHeight + 'px' }"></div>
        </div>

        <!-- ── 数据面 / 算子 成功率（研究记忆库聚合） ── -->
        <div class="summary-panel">
          <div class="summary-panel-head">
            <h3>数据面 / 算子 成功率</h3>
            <div class="metrics-split-controls">
              <select v-model="facetMetric" class="metrics-last-select">
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

        <!-- ── 逐 Run 指标变化趋势（时间轴） ── -->
        <div class="summary-panel">
          <div class="summary-panel-head">
            <h3>指标逐 Run 趋势（横坐标为开始时间）</h3>
            <div class="metrics-split-controls">
              <select v-model="trendMetric" class="metrics-last-select">
                <option value="effective_novelty_rate">有效新颖率 (A5)</option>
                <option value="facet_coverage">数据面覆盖数 (A1)</option>
                <option value="family_coverage">信号族覆盖数 (A3)</option>
                <option value="structure_variety">结构指纹多样率 (A6)</option>
                <option value="prediction_coverage">预测对账覆盖率 (E1)</option>
                <option value="advisory_follow_rate">死路提示遵循率 (E6)</option>
                <option value="eval_latency_p50_s">单次评估耗时 P50 (D3)</option>
                <option value="stage_one_yield_pct">海选过线率 (头条1)</option>
                <option value="unsubmitted_passing_rate">未交付过线率 (F1)</option>
              </select>
              <span class="summary-facet-hint" title="横坐标为 Run 开始时间（MM-DD HH:MM），点悬停展示详细时长与增量 Δ。">ⓘ</span>
            </div>
          </div>
          <div id="metrics-trend-chart" class="metrics-chart" :style="{ height: trendChartHeight + 'px' }"></div>
        </div>

        <!-- ── 每 run 明细（图表固定紧凑高度 + 表格内部滚动，避免按 run 数线性撑高页面） ── -->
        <div class="summary-panel">
          <div class="summary-panel-head">
            <h3>Run 明细（按时间倒序）</h3>
            <div class="metrics-split-controls">
              <button class="summary-refresh-btn" style="padding:2px 8px;font-size:11px;" @click="showFullMetrics = !showFullMetrics">
                {{ showFullMetrics ? '收起扩展列' : '展开扩展列' }}
              </button>
              <span v-if="(data.runs || []).length > runsChartCap" class="summary-facet-hint">
                图表仅显示最近 {{ runsChartCap }} 个 run，完整明细见下表
              </span>
            </div>
          </div>
          <div id="metrics-runs-chart" class="metrics-chart" :style="{ height: runsChartHeight + 'px' }"></div>
          <div class="summary-table-wrap metrics-run-wrap" style="margin-top:14px">
          <table class="summary-table metrics-run-table">
            <thead>
              <tr>
                <th>开始时间</th>
                <th>Run</th>
                <th>时长</th>
                <th>说明/Note</th>
                <th>评估</th>
                <th>新颖率</th>
                <th>面覆盖</th>
                <th>预测率</th>
                <th>提交</th>
                <th>入库</th>
                <th v-if="showFullMetrics">族覆盖</th>
                <th v-if="showFullMetrics">算子熵</th>
                <th v-if="showFullMetrics">消融覆盖</th>
                <th v-if="showFullMetrics">死路遵循</th>
                <th v-if="showFullMetrics">耗时P50</th>
                <th>错误率</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="r in formattedRuns" :key="r.run_id">
                <td style="white-space: nowrap; font-size: 11px;">{{ r.time_str }}</td>
                <td class="metrics-run-id" :title="r.run_id">{{ r.run_id.slice(0, 8) }}</td>
                <td>{{ fmt(r.wall_minutes ?? 0) }}m</td>
                <td :title="r.bench_note || ''" style="max-width: 130px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{{ r.bench_note || r.bench_commit || '-' }}</td>
                <td>{{ (r.n_eval ?? 0) + (r.n_eval_val ?? 0) }}</td>
                <td>
                  {{ r.nov_str }}
                  <span v-if="r.nov_delta != null" :class="r.nov_delta >= 0 ? 'metrics-delta-up' : 'metrics-delta-down'">{{ r.nov_delta >= 0 ? '+' : '' }}{{ r.nov_delta }}</span>
                </td>
                <td>{{ r.facet_cov }}</td>
                <td>{{ r.pred_cov }}</td>
                <td>{{ r.n_submit ?? 0 }}</td>
                <td>{{ (r.stored_candidate ?? 0) + (r.stored_production ?? 0) }}</td>
                <td v-if="showFullMetrics">{{ r.fam_cov }}</td>
                <td v-if="showFullMetrics">{{ r.op_ent }}</td>
                <td v-if="showFullMetrics">{{ r.abl_cov }}</td>
                <td v-if="showFullMetrics">{{ r.adv_follow }}</td>
                <td v-if="showFullMetrics">{{ r.lat_p50 }}</td>
                <td>{{ r.tool_error_rate == null ? '-' : (r.tool_error_rate * 100).toFixed(0) + '%' }}</td>
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
    return {
      data: null,
      loading: false,
      refreshing: false,
      error: '',
      lastN: 20,
      facetMetric: 'rate',
      trendMetric: 'effective_novelty_rate',
      showFullMetrics: false,
      autoPoll: true,
      pollTimer: null,
      metricGroups: [
        {
          id: 'A',
          name: '探索质量 (Exploration)',
          metrics: [
            { key: 'effective_novelty_rate', name: '有效新颖率', isPct: true, higherGood: true },
            { key: 'facet_coverage', name: '数据面覆盖数', isPct: false, higherGood: true },
            { key: 'family_coverage', name: '信号族覆盖数', isPct: false, higherGood: true },
            { key: 'operator_entropy', name: '算子分布熵', isPct: false, higherGood: true },
            { key: 'structure_variety', name: '结构多样率', isPct: true, higherGood: true },
          ],
        },
        {
          id: 'B',
          name: '收敛轨迹 (Dynamics)',
          metrics: [
            { key: 'saturation_turn', name: '饱和轮次', isPct: false, higherGood: false },
            { key: 'late_gain_share', name: '后段增益占比', isPct: true, higherGood: true },
            { key: 'early_mean_abs_ic', name: '早期均值 IC', isPct: false, higherGood: true },
            { key: 'late_mean_abs_ic', name: '后期均值 IC', isPct: false, higherGood: true },
          ],
        },
        {
          id: 'C',
          name: '产出质量 (Quality)',
          metrics: [
            { key: 'temporal_stability', name: '月度符号稳定性', isPct: true, higherGood: true },
            { key: 'claim_implementation_consistency', name: '机制算子一致率', isPct: true, higherGood: true },
            { key: 'max_corr_median', name: '库内相关中位', isPct: false, higherGood: false },
          ],
        },
        {
          id: 'D',
          name: '成本与耗时 (Cost & Latency)',
          metrics: [
            { key: 'cost_per_candidate_k_tokens', name: '每候选 Token(K)', isPct: false, higherGood: false },
            { key: 'cost_of_first_pass_k_tokens', name: '首过线 Token(K)', isPct: false, higherGood: false },
            { key: 'eval_latency_p50_s', name: '单次耗时 P50(s)', isPct: false, higherGood: false },
            { key: 'eval_throughput', name: '评估吞吐(次/分)', isPct: false, higherGood: true },
          ],
        },
        {
          id: 'E',
          name: '过程控制 (Process Control)',
          metrics: [
            { key: 'prediction_coverage', name: '预测对账覆盖率', isPct: true, higherGood: true },
            { key: 'ablation_coverage', name: '门控消融覆盖率', isPct: true, higherGood: true },
            { key: 'advisory_follow_rate', name: '死路提示遵循率', isPct: true, higherGood: true },
            { key: 'min_gap', name: '近线最小差距', isPct: true, higherGood: false },
          ],
        },
        {
          id: 'F',
          name: '诚实性与漏斗 (Integrity & Funnel)',
          metrics: [
            { key: 'stage_one_yield_pct', name: '海选过线率', isPct: true, higherGood: true },
            { key: 'gate_survival_pct', name: '实盘门存活率', isPct: true, higherGood: true },
            { key: 'unsubmitted_passing_rate', name: '未交付过线率', isPct: true, higherGood: false },
            { key: 'reviewer_calibration_lift', name: 'Reviewer 校准 Lift', isPct: false, higherGood: true },
          ],
        },
      ],
    }
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
    trendChartHeight() { return 200 },
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
    formattedRuns() {
      const runs = this.data?.runs || []
      const chrono = [...runs].reverse()
      let prevNov = null
      const withDelta = chrono.map(r => {
        const v4 = r.v4 || {}
        const hl = v4.headline || {}
        const exp = v4.exploration || {}
        const proc = v4.process || {}
        const cost = v4.cost || {}
        const timeMeta = v4.time_meta || {}

        const rawTs = r.created_at || timeMeta.created_at || ''
        const time_str = rawTs ? rawTs.slice(5, 16).replace('T', ' ') : '-'

        const nov = hl.effective_novelty_rate != null ? Number(hl.effective_novelty_rate) : (exp.effective_novelty_rate != null ? Number(exp.effective_novelty_rate) : null)
        let nov_delta = null
        if (nov != null && prevNov != null) {
          nov_delta = Number((nov - prevNov).toFixed(3))
        }
        if (nov != null) prevNov = nov

        const facet_cov = exp.facet_coverage != null ? exp.facet_coverage : '-'
        const pred_cov = proc.prediction_coverage != null ? (proc.prediction_coverage * 100).toFixed(0) + '%' : '-'
        const fam_cov = exp.family_coverage != null ? exp.family_coverage : '-'
        const op_ent = exp.operator_entropy != null ? Number(exp.operator_entropy).toFixed(2) : '-'
        const abl_cov = proc.ablation_coverage != null ? (proc.ablation_coverage * 100).toFixed(0) + '%' : '-'
        const adv_follow = proc.advisory_follow_rate != null ? (proc.advisory_follow_rate * 100).toFixed(0) + '%' : '-'
        const lat_p50 = cost.eval_latency_p50_s != null ? Number(cost.eval_latency_p50_s).toFixed(1) + 's' : '-'

        return {
          ...r,
          time_str,
          bench_note: r.bench_note || timeMeta.bench_note || '',
          bench_commit: r.bench_commit || timeMeta.bench_commit || '',
          config_hash: r.config_hash || timeMeta.config_hash || '',
          nov,
          nov_str: nov != null ? (nov * 100).toFixed(1) + '%' : '-',
          nov_delta,
          facet_cov,
          pred_cov,
          fam_cov,
          op_ent,
          abl_cov,
          adv_follow,
          lat_p50,
        }
      })
      return withDelta.reverse()
    },
  },
  mounted() {
    this.refresh(false)
    this.startPolling()
    document.addEventListener('visibilitychange', this.onVisibilityChange)
  },
  beforeUnmount() {
    this.stopPolling()
    document.removeEventListener('visibilitychange', this.onVisibilityChange)
  },
  watch: {
    // 口径切换后重画（等 v-model 落值再渲染，避免用旧口径画图）
    facetMetric() { this.$nextTick(() => this.renderFacetOperatorCharts()) },
    trendMetric() { this.$nextTick(() => this.renderTrendChart()) },
  },
  methods: {
    getMetricVal(r, key) {
      if (!r) return null
      const v4 = r.v4 || {}
      for (const grp of ['headline', 'exploration', 'dynamics', 'quality', 'cost', 'process', 'integrity']) {
        if (v4[grp] && v4[grp][key] != null) return Number(v4[grp][key])
      }
      if (v4.process?.near_miss_progress && v4.process.near_miss_progress[key] != null) {
        return Number(v4.process.near_miss_progress[key])
      }
      if (v4.quality?.library_novelty && v4.quality.library_novelty[key] != null) {
        return Number(v4.quality.library_novelty[key])
      }
      if (v4.dynamics?.long_horizon_decay) {
        if (key === 'early_mean_abs_ic') return Number(v4.dynamics.long_horizon_decay.early?.mean_abs_ic)
        if (key === 'late_mean_abs_ic') return Number(v4.dynamics.long_horizon_decay.late?.mean_abs_ic)
      }
      if (r[key] != null) return Number(r[key])
      if (r.funnel && r.funnel[key] != null) return Number(r.funnel[key])
      if (r.summary && r.summary[key] != null) return Number(r.summary[key])
      return null
    },
    getSpark(key) {
      const runs = (this.data?.runs || []).slice().reverse()
      if (!runs.length) return { points: '', lastVal: null, delta: null, lastPoint: null }
      const pointsData = []
      runs.forEach(r => {
        const v = this.getMetricVal(r, key)
        if (v != null && !isNaN(v)) pointsData.push(v)
      })
      if (!pointsData.length) return { points: '', lastVal: null, delta: null, lastPoint: null }

      const lastVal = pointsData[pointsData.length - 1]
      let delta = null
      if (pointsData.length >= 2) {
        delta = Number((lastVal - pointsData[pointsData.length - 2]).toFixed(4))
      }

      const w = 76
      const h = 18
      const min = Math.min(...pointsData)
      const max = Math.max(...pointsData)
      const range = max - min

      const pts = pointsData.map((val, idx) => {
        const x = pointsData.length === 1 ? w / 2 : (idx / (pointsData.length - 1)) * w
        const y = range <= 1e-6 ? h / 2 : h - 2 - ((val - min) / range) * (h - 4)
        return `${x.toFixed(1)},${y.toFixed(1)}`
      })

      const lastCoord = pts[pts.length - 1].split(',')
      return {
        points: pts.join(' '),
        lastVal,
        delta,
        lastPoint: { x: Number(lastCoord[0]), y: Number(lastCoord[1]) },
      }
    },
    selectTrendMetric(key) {
      this.trendMetric = key
      this.$nextTick(() => {
        this.renderTrendChart()
        const el = document.getElementById('metrics-trend-chart')
        if (el) el.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
      })
    },
    formatMetricVal(spec, val) {
      if (val == null || isNaN(val)) return '-'
      if (spec.isPct) {
        return Math.abs(val) <= 1.0 ? (val * 100).toFixed(1) + '%' : val.toFixed(1) + '%'
      }
      return Math.abs(val) >= 100 ? val.toFixed(0) : Math.abs(val) >= 10 ? val.toFixed(1) : val.toFixed(3)
    },
    formatDelta(spec, delta) {
      if (delta == null || isNaN(delta)) return '—'
      if (Math.abs(delta) < 1e-4) return '0.0'
      const sign = delta > 0 ? '+' : ''
      if (spec.isPct) {
        return Math.abs(delta) <= 1.0 ? `${sign}${(delta * 100).toFixed(1)}%` : `${sign}${delta.toFixed(1)}%`
      }
      return `${sign}${delta.toFixed(3)}`
    },
    deltaClass(spec, delta) {
      if (delta == null || Math.abs(delta) < 1e-4) return ''
      const isGood = spec.higherGood ? delta > 0 : delta < 0
      return isGood ? 'metrics-delta-up' : 'metrics-delta-down'
    },
    deltaColor(spec, delta) {
      if (delta == null || Math.abs(delta) < 1e-4) return '#8ab4d8'
      const isGood = spec.higherGood ? delta > 0 : delta < 0
      return isGood ? '#4fc3a1' : '#ef6b73'
    },
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
      this.renderTrendChart()
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
      // 漏斗 = 研究记忆库 + 因子库 registry（全库口径，跨 run 稳定）；
      // run 日志只承载成本类指标，不再作为漏斗事实源。
      const f = this.data.research_funnel || {}
      const total = f.total ?? 0
      const stages = [
        { name: '评估尝试', value: f.total ?? 0 },
        { name: '有效尝试', value: f.valid ?? 0 },
        { name: '海选过线', value: f.positive ?? 0 },
        { name: '发起提交', value: f.submitted ?? 0 },
        { name: '入候选池', value: f.candidate_stored ?? 0 },
        { name: '晋升正式库', value: f.production_stored ?? 0 },
      ]
      const c = chart('metrics-funnel-chart')
      if (!c) return
      c.setOption({
        tooltip: {
          trigger: 'item',
          formatter: p => `${p.name}：<b>${p.value}</b>（占评估尝试 ${(total ? (p.value / total * 100).toFixed(1) : 0)}%）`,
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
    renderTrendChart() {
      if (!window.echarts || !this.data) return
      const runs = this.data.runs || []
      if (!runs.length) return
      const chrono = [...runs].reverse().slice(-this.runsChartCap)
      const c = chart('metrics-trend-chart')
      if (!c) return

      const metric = this.trendMetric
      const getVal = r => {
        const v4 = r.v4 || {}
        for (const grp of ['headline', 'exploration', 'dynamics', 'quality', 'cost', 'process', 'integrity']) {
          if (v4[grp] && v4[grp][metric] != null) return Number(v4[grp][metric])
        }
        return null
      }

      const times = chrono.map(r => {
        const raw = r.created_at || r.v4?.time_meta?.created_at || ''
        return raw ? raw.slice(5, 16).replace('T', ' ') : r.run_id.slice(0, 6)
      })
      const vals = chrono.map(r => getVal(r))

      c.setOption({
        tooltip: {
          trigger: 'axis',
          formatter: params => {
            const idx = params[0]?.dataIndex
            if (idx == null) return ''
            const r = chrono[idx]
            const val = vals[idx]
            const prev = idx > 0 ? vals[idx - 1] : null
            const delta = (val != null && prev != null) ? (val - prev).toFixed(4) : '-'
            const dSign = (val != null && prev != null && val >= prev) ? '+' : ''
            const note = r.bench_note || r.bench_commit || ''
            return `<b>${r.run_id}</b><br/>`
              + `开始时间: ${times[idx]} (时长 ${Number(r.wall_minutes || 0).toFixed(0)}m)<br/>`
              + `数值: <b>${val != null ? val : 'N/A'}</b> (Δ上一run: ${dSign}${delta})`
              + (note ? `<br/>说明: ${note}` : '')
          },
        },
        grid: { left: 50, right: 20, top: 16, bottom: 28 },
        xAxis: {
          type: 'category',
          data: times,
          axisLabel: { ...AXIS_LABEL, rotate: 30, hideOverlap: true },
        },
        yAxis: {
          type: 'value',
          axisLabel: { ...AXIS_LABEL },
          splitLine: { lineStyle: { color: '#1c2536' } },
        },
        series: [{
          name: metric,
          type: 'line',
          smooth: true,
          data: vals,
          symbolSize: 7,
          itemStyle: { color: '#4fc3a1' },
          lineStyle: { color: '#4f8cff', width: 2 },
          areaStyle: { color: 'rgba(79, 140, 255, 0.15)' },
        }],
      }, true)
    },
    advisoryLabel(kind) {
      return ({
        duplicate_known_dead_end: '重复死路结构提醒',
        edit_veto: '意向编辑否决',
        duplicate_prior_result: '正向结构重复',
      })[kind] || kind
    },
    startPolling() {
      this.stopPolling()
      if (!this.autoPoll) return
      // 8 秒一次轻量自动轮询（静默刷新，不打扰用户交互）
      this.pollTimer = setInterval(() => {
        this.refresh(true)
      }, 8000)
    },
    stopPolling() {
      if (this.pollTimer) {
        clearInterval(this.pollTimer)
        this.pollTimer = null
      }
    },
    onVisibilityChange() {
      if (document.hidden) {
        this.stopPolling()
      } else if (this.autoPoll) {
        this.refresh(true)
        this.startPolling()
      }
    },
    toggleAutoPoll() {
      this.autoPoll = !this.autoPoll
      if (this.autoPoll) {
        this.refresh(true)
        this.startPolling()
      } else {
        this.stopPolling()
      }
    },
    async refresh(silent = false) {
      if (this.refreshing) return
      this.refreshing = true
      if (!silent) {
        this.loading = true
        this.error = ''
      }
      try {
        this.data = await api(`/api/alphaagent/metrics/overview?last=${this.lastN}`)
        this.$nextTick(() => this.renderCharts())
      } catch (e) {
        if (!silent) this.error = String(e.message || e)
      } finally {
        this.refreshing = false
        if (!silent) this.loading = false
      }
    },
  },
}
</script>
