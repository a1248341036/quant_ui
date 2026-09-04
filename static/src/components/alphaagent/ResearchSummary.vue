<template>
  <main class="agent-main research-summary-page">
    <header class="agent-header">
      <div class="agent-title">
        <span class="agent-orb">✦</span>
        <div>
          <h1>研究总结</h1>
          <p class="agent-subtitle">所有试过的因子状态与指标</p>
        </div>
      </div>
      <div class="header-actions">
        <button class="summary-refresh-btn" :disabled="loading" @click="refresh">刷新</button>
      </div>
    </header>

    <div class="agent-thread">
      <!-- 盲测结果（单因子的真实 OOS）：人看的样本外检验，刻意不进 agent 记忆 -->
      <div class="summary-panel">
        <div class="summary-panel-head">
          <h3>盲测结果（2025+ 真实样本外）</h3>
          <span class="summary-facet-hint" title="盲测段对挖掘循环锁死，结果不回流 agent 记忆（防多重检验）。保留比 = 盲测 IC / 参考段 IC。">
            ⓘ
          </span>
        </div>
        <div v-if="agent.blindTestLoading" class="normal-mode-empty">加载盲测报告中…</div>
        <div v-else-if="agent.blindTestError" class="normal-mode-empty">
          盲测面板不可用：{{ agent.blindTestError }}（重启后端后刷新本页可修复）
        </div>
        <template v-else>
          <div v-for="bt in blindReports" :key="bt.run_ts" class="blind-test-block">
            <div class="blind-test-head">
              {{ bt.run_ts }} · 窗口 {{ bt.report?.test_window?.start }} ~ {{ bt.report?.test_window?.end }}
              （{{ bt.report?.test_window?.trading_days }} 个交易日）
            </div>
            <table class="summary-table blind-test-table">
              <thead>
                <tr>
                  <th>因子</th><th>库</th><th>参考 IC</th><th>盲测 IC</th><th>保留比</th><th>盲测 ICIR</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="f in bt.factors" :key="bt.run_ts + f.factor_id">
                  <td class="summary-name">{{ f.name }}</td>
                  <td>{{ (f.library || '').replace('candidate_', '').replace('production_', '') }}</td>
                  <td :class="icClass(f.ref_ic)">{{ formatMetricValue(f.ref_ic ?? '—') }}</td>
                  <td :class="icClass(f.test_ic)">{{ formatMetricValue(f.test_ic ?? '—') }}</td>
                  <td :class="{ neg: (f.ic_retention_vs_ref ?? 1) < 0.3 }">
                    {{ f.ic_retention_vs_ref != null ? (f.ic_retention_vs_ref * 100).toFixed(0) + '%' : '—' }}
                  </td>
                  <td>{{ formatMetricValue(f.test_icir ?? '—') }}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div v-if="blindPendingCount" class="blind-test-head">
            另有 {{ blindPendingCount }} 次盲测运行未产出报告（进行中或已失败，完成后刷新可见）
          </div>
          <div v-if="!blindReports.length" class="normal-mode-empty">
            还没有已完成的盲测报告（跑 scripts/blind_test_factors.py 生成）
          </div>
        </template>
      </div>

      <div class="summary-panel">
        <div class="summary-panel-head">
          <h3>研究记忆因子一览</h3>
        </div>
        <div class="summary-filter-bar" v-if="agent.summaryTotal || loading">
          <button
            class="verdict-filter-btn"
            :class="{ active: !agent.summaryVerdictFilter }"
            @click="agent.setSummaryVerdictFilter('')"
          >全部 <b>{{ agent.summaryAllCount }}</b></button>
          <button
            v-for="verdict in verdictOrder"
            :key="verdict"
            class="verdict-filter-btn"
            :class="['memv-' + verdict, { active: agent.summaryVerdictFilter === verdict }]"
            @click="agent.setSummaryVerdictFilter(verdict)"
          >
            <span class="summary-verdict-dot" :class="'memv-' + verdict"></span>
            {{ memoryVerdictLabel(verdict) }}
            <b>{{ summaryVerdictCounts[verdict] || 0 }}</b>
          </button>
        </div>
        <div v-if="!agent.summaryEntries.length && !loading" class="normal-mode-empty">暂无研究记忆</div>
        <div v-else-if="!agent.summaryEntries.length && loading" class="normal-mode-empty">加载中…</div>
        <div v-else-if="summaryFiltered.length" class="summary-table-wrap">
          <table class="summary-table">
          <thead>
            <tr>
              <th @click="agent.setSummarySort('factor_name')" :class="{ sortable: true, 'sort-active': agent.summarySortKey === 'factor_name' }">
                因子名称 <span class="sort-ind" v-if="agent.summarySortKey === 'factor_name'">{{ agent.summarySortOrder === 1 ? '▲' : '▼' }}</span>
              </th>
              <th @click="agent.setSummarySort('verdict')" :class="{ sortable: true, 'sort-active': agent.summarySortKey === 'verdict' }">
                状态 <span class="sort-ind" v-if="agent.summarySortKey === 'verdict'">{{ agent.summarySortOrder === 1 ? '▲' : '▼' }}</span>
              </th>
              <th @click="agent.setSummarySort('stage')" :class="{ sortable: true, 'sort-active': agent.summarySortKey === 'stage' }">
                阶段 <span class="sort-ind" v-if="agent.summarySortKey === 'stage'">{{ agent.summarySortOrder === 1 ? '▲' : '▼' }}</span>
              </th>
              <th @click="agent.setSummarySort('ic')" :class="{ sortable: true, 'sort-active': agent.summarySortKey === 'ic' }">
                IC <span class="sort-ind" v-if="agent.summarySortKey === 'ic'">{{ agent.summarySortOrder === 1 ? '▲' : '▼' }}</span>
              </th>
              <th @click="agent.setSummarySort('icir')" :class="{ sortable: true, 'sort-active': agent.summarySortKey === 'icir' }">
                ICIR <span class="sort-ind" v-if="agent.summarySortKey === 'icir'">{{ agent.summarySortOrder === 1 ? '▲' : '▼' }}</span>
              </th>
              <th @click="agent.setSummarySort('coverage')" :class="{ sortable: true, 'sort-active': agent.summarySortKey === 'coverage' }">
                覆盖率 <span class="sort-ind" v-if="agent.summarySortKey === 'coverage'">{{ agent.summarySortOrder === 1 ? '▲' : '▼' }}</span>
              </th>
              <th title="截面 lag-1 自相关（换手代理）：低于 0.18 的因子排名日度变化过快，交易成本吃掉 alpha，海选自动拦截">
                换手 <span class="summary-facet-hint">ⓘ</span>
              </th>
              <th @click="agent.setSummarySort('annualized_excess_return')" :class="{ sortable: true, 'sort-active': agent.summarySortKey === 'annualized_excess_return' }">
                年化超额 <span class="sort-ind" v-if="agent.summarySortKey === 'annualized_excess_return'">{{ agent.summarySortOrder === 1 ? '▲' : '▼' }}</span>
              </th>
              <th @click="agent.setSummarySort('sharpe')" :class="{ sortable: true, 'sort-active': agent.summarySortKey === 'sharpe' }">
                夏普 <span class="sort-ind" v-if="agent.summarySortKey === 'sharpe'">{{ agent.summarySortOrder === 1 ? '▲' : '▼' }}</span>
              </th>
              <th @click="agent.setSummarySort('excess_sharpe')" :class="{ sortable: true, 'sort-active': agent.summarySortKey === 'excess_sharpe' }">
                超额夏普 <span class="sort-ind" v-if="agent.summarySortKey === 'excess_sharpe'">{{ agent.summarySortOrder === 1 ? '▲' : '▼' }}</span>
              </th>
              <th @click="agent.setSummarySort('annualized_return')" :class="{ sortable: true, 'sort-active': agent.summarySortKey === 'annualized_return' }">
                年化收益 <span class="sort-ind" v-if="agent.summarySortKey === 'annualized_return'">{{ agent.summarySortOrder === 1 ? '▲' : '▼' }}</span>
              </th>
              <th @click="agent.setSummarySort('max_drawdown')" :class="{ sortable: true, 'sort-active': agent.summarySortKey === 'max_drawdown' }">
                最大回撤 <span class="sort-ind" v-if="agent.summarySortKey === 'max_drawdown'">{{ agent.summarySortOrder === 1 ? '▲' : '▼' }}</span>
              </th>
              <th @click="agent.setSummarySort('annual_turnover')" :class="{ sortable: true, 'sort-active': agent.summarySortKey === 'annual_turnover' }" title="组合层年换手（仅走过提交回测的因子有值）">
                组合年换手 <span class="sort-ind" v-if="agent.summarySortKey === 'annual_turnover'">{{ agent.summarySortOrder === 1 ? '▲' : '▼' }}</span>
              </th>
              <th @click="agent.setSummarySort('daily_overlap')" :class="{ sortable: true, 'sort-active': agent.summarySortKey === 'daily_overlap' }">
                日重叠 <span class="sort-ind" v-if="agent.summarySortKey === 'daily_overlap'">{{ agent.summarySortOrder === 1 ? '▲' : '▼' }}</span>
              </th>
              <th @click="agent.setSummarySort('monotonicity')" :class="{ sortable: true, 'sort-active': agent.summarySortKey === 'monotonicity' }">
                单调性 <span class="sort-ind" v-if="agent.summarySortKey === 'monotonicity'">{{ agent.summarySortOrder === 1 ? '▲' : '▼' }}</span>
              </th>
              <th @click="agent.setSummarySort('attempts')" :class="{ sortable: true, 'sort-active': agent.summarySortKey === 'attempts' }">
                评估次数 <span class="sort-ind" v-if="agent.summarySortKey === 'attempts'">{{ agent.summarySortOrder === 1 ? '▲' : '▼' }}</span>
              </th>
              <th>
                调仓 <span class="summary-facet-hint" title="门禁调仓频率（周/日/月）。入库的新因子为真实记录；老因子按评估标签推导（hover 见口径）。仅入库过的因子有值。">ⓘ</span>
              </th>
              <th>
                数据面 <span class="summary-facet-hint" title="触及的数据面（×连接 = 跨数据源组融合因子）">ⓘ</span>
              </th>
              <th>拒绝原因</th>
              <th @click="agent.setSummarySort('updated_at')" :class="{ sortable: true, 'sort-active': agent.summarySortKey === 'updated_at' }">
                更新时间 <span class="sort-ind" v-if="agent.summarySortKey === 'updated_at'">{{ agent.summarySortOrder === 1 ? '▲' : '▼' }}</span>
              </th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="entry in summaryFiltered" :key="entry.id" @click="agent.memoryDetail = entry" class="summary-row">
              <td class="summary-name" :title="entry.factor_name">{{ entry.factor_name || 'unnamed' }}</td>
              <td><span class="summary-verdict-tag" :class="'memv-' + entry.verdict">{{ memoryVerdictLabel(entry.verdict) }}</span></td>
              <td class="summary-stage" :title="'指标来自该评估阶段（单因子无 OOS：盲测段对挖掘循环锁死，防多重检验污染）'">{{ stageLabel(entry.stage) }}</td>
              <td :class="icClass(entry.metrics?.ic)">{{ formatMetricValue(entry.metrics?.ic ?? '—') }}</td>
              <td>{{ formatMetricValue(entry.metrics?.icir ?? '—') }}</td>
              <td>{{ formatMetricValue(entry.metrics?.coverage ?? entry.metrics?.factor_coverage ?? '—') }}</td>
              <td :class="{ neg: (entry.metrics?.cs_pearson_autocorr ?? 1) < 0.18 }">{{ formatMetricValue(entry.metrics?.cs_pearson_autocorr ?? '—') }}</td>
              <td :class="{ neg: (entry.metrics?.annualized_excess_return ?? entry.metrics?.long_group_annual_excess_return ?? 0) < 0 }">{{ formatMetricValue(entry.metrics?.annualized_excess_return ?? entry.metrics?.long_group_annual_excess_return ?? '—') }}</td>
              <td :class="{ neg: (entry.metrics?.sharpe ?? 0) < 0 }">{{ formatMetricValue(entry.metrics?.sharpe ?? '—') }}</td>
              <td :class="{ neg: (entry.metrics?.excess_sharpe ?? 0) < 0 }">{{ formatMetricValue(entry.metrics?.excess_sharpe ?? '—') }}</td>
              <td :class="{ neg: (entry.metrics?.annualized_return ?? 0) < 0 }">{{ formatMetricValue(entry.metrics?.annualized_return ?? '—') }}</td>
              <td :class="{ neg: (entry.metrics?.max_drawdown ?? 0) > 0 }">{{ formatMetricValue(entry.metrics?.max_drawdown ?? '—') }}</td>
              <td>{{ formatMetricValue(entry.metrics?.annual_turnover ?? '—') }}</td>
              <td>{{ formatMetricValue(entry.metrics?.daily_overlap ?? '—') }}</td>
              <td>{{ formatMetricValue(entry.metrics?.monotonicity ?? '—') }}</td>
              <td class="summary-attempts">{{ entry.attempts || 1 }}</td>
              <td class="summary-freq">
                <span v-if="entry.rebalance_freq" class="lib-freq-tag"
                      :class="'freq-' + entry.rebalance_freq"
                      :title="'门禁调仓频率' + (entry.freq_source === 'derived' ? '（按评估标签推导）' : '')">
                  {{ freqShort(entry.rebalance_freq) }}</span>
                <span v-else class="summary-freq-missing" title="未入库或无频率记录">—</span>
              </td>
              <td class="summary-facets" :title="(entry.facets || []).join(' + ')">
                <span v-if="entry.is_fusion" class="summary-fusion-tag">融合</span>
                <span class="summary-facet-text">{{ facetText(entry) }}</span>
              </td>
              <td class="summary-reason" :title="entry.conclusion">{{ entry.conclusion || entry.error || '—' }}</td>
              <td class="summary-time">{{ formatTime(entry.updated_at) }}</td>
            </tr>
          </tbody>
        </table>
        </div>
        <div v-if="!summaryFiltered.length && agent.summaryEntries.length" class="normal-mode-empty">没有匹配的因子</div>
        <div v-if="agent.summaryTotalPages > 1" class="summary-pagination">
          <button
            class="page-btn"
            :disabled="agent.summaryPage <= 1 || agent.summaryLoading"
            @click="agent.setSummaryPage(agent.summaryPage - 1)"
          >‹</button>
          <template v-for="(p, i) in pageList" :key="i">
            <span v-if="p === '…'" class="page-ellipsis">…</span>
            <button
              v-else
              class="page-btn"
              :class="{ active: p === agent.summaryPage }"
              :disabled="agent.summaryLoading"
              @click="agent.setSummaryPage(p)"
            >{{ p }}</button>
          </template>
          <button
            class="page-btn"
            :disabled="agent.summaryPage >= agent.summaryTotalPages || agent.summaryLoading"
            @click="agent.setSummaryPage(agent.summaryPage + 1)"
          >›</button>
          <span class="page-info">共 {{ agent.summaryTotal }} 条 · 第 {{ agent.summaryPage }} / {{ agent.summaryTotalPages }} 页</span>
        </div>
      </div>
    </div>

    <!-- 长期记忆详情弹窗 -->
    <memory-detail-modal />
  </main>
</template>

<script>
import '../../styles/alphaagent.css'
import { agentStore, summaryView } from '../../store/alphaagent.js'
import MemoryDetailModal from './MemoryDetailModal.vue'
import {
  formatTime, memoryVerdictLabel,
  formatMetricValue, icClass, freqShort,
} from '../../utils/alphaagent.js'

export default {
  name: 'ResearchSummary',
  components: {
    'memory-detail-modal': MemoryDetailModal,
  },
  data() {
    return {
      agent: agentStore,
      loading: false,
      verdictOrder: [
        'production_approved',
        'validated',
        'candidate_approved',
        'promising',
        'revise_required',
        'rejected',
        'weak',
      ],
    }
  },
  computed: {
    summaryVerdictCounts() {
      return summaryView.counts.value
    },
    summaryFiltered() {
      return summaryView.filtered.value
    },
    /** 最新一份盲测报告的因子明细（ref IC vs 盲测 IC 对比） */
    blindReports() {
      const reports = (this.agent.blindTest && this.agent.blindTest.reports) || []
      const latest = reports.find(r => r.report && Array.isArray(r.report.factors)) || null
      if (!latest) return []
      return [{
        run_ts: latest.run_ts,
        report: latest.report,
        factors: latest.report.factors || [],
      }]
    },
    /** 未产出报告的盲测运行数（进行中/失败） */
    blindPendingCount() {
      const reports = (this.agent.blindTest && this.agent.blindTest.reports) || []
      return reports.filter(r => !r.report).length
    },
    totalPages() {
      return this.agent.summaryTotalPages
    },
    /** 页码列表：首尾常驻 + 当前页 ±1，间隙用 … 占位 */
    pageList() {
      const total = this.totalPages
      const cur = this.agent.summaryPage
      if (total <= 7) return Array.from({ length: total }, (_, i) => i + 1)
      const pages = [1]
      const lo = Math.max(2, cur - 1)
      const hi = Math.min(total - 1, cur + 1)
      if (lo > 2) pages.push('…')
      for (let p = lo; p <= hi; p++) pages.push(p)
      if (hi < total - 1) pages.push('…')
      pages.push(total)
      return pages
    },
  },
  mounted() {
    this.refresh()
    this.agent.loadBlindTest()
  },
  methods: {
    formatTime,
    memoryVerdictLabel,
    formatMetricValue,
    icClass,
    freqShort,
    facetText(entry) {
      const facets = entry.facets || []
      if (!facets.length) return '—'
      // 与记忆 family 口径一致：跨数据源组融合用 × 连接，单面/同组用 +
      return entry.is_fusion ? facets.join('×') : facets.join('+')
    },
    stageLabel(stage) {
      // 指标来自挖掘循环内的评估段；盲测段（真正的 OOS）刻意不回流记忆，防止多重检验污染
      return { train: '训练段', val: '验证段', test: '盲测段' }[stage] || stage || '—'
    },
    async refresh() {
      this.loading = true
      try {
        await agentStore.loadSummaryPage()
      } finally {
        this.loading = false
      }
    },
  },
}
</script>
