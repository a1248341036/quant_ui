<template>
  <main class="agent-main research-summary-page">
    <header class="agent-header">
      <div class="agent-title">
        <span class="agent-orb">✦</span>
        <div>
          <h1>研究记忆库</h1>
          <p class="agent-subtitle">三层记忆全景：经验层 / SSPM 编辑统计层 / 原始证据层</p>
        </div>
      </div>
      <div class="header-actions">
        <button class="summary-refresh-btn" :disabled="loading" @click="refresh">刷新</button>
      </div>
    </header>

    <div class="agent-thread">
      <div class="summary-panel">
        <!-- ═══ 统计概览 ═══ -->
        <div class="summary-stats" v-if="stats">
          <span class="summary-stat">记忆条目 <b>{{ stats.entries }}</b></span>
          <span class="summary-stat">评估观测 <b>{{ stats.observations }}</b></span>
          <span class="summary-stat">SSPM 单元 <b>{{ stats.cells }}</b></span>
          <span class="summary-stat">经验结论 <b>{{ stats.experience }}</b></span>
          <span class="summary-stat" v-for="(count, verdict) in stats.verdict_counts" :key="verdict">
            <span class="summary-verdict-dot" :class="'memv-' + verdict"></span>{{ memoryVerdictLabel(verdict) }} <b>{{ count }}</b>
          </span>
          <span class="summary-stat" v-if="stats.train_to_validated_rate != null">验证通过率 <b>{{ pct(stats.train_to_validated_rate) }}</b></span>
          <span class="summary-stat" v-if="stats.production_rate != null">入库率 <b>{{ pct(stats.production_rate) }}</b></span>
        </div>

        <!-- ═══ 经验层 ═══ -->
        <div class="rmb-section">
          <div class="summary-panel-head">
            <h3>经验层</h3>
            <span class="rmb-section-sub">跨因子蒸馏：成功模式 / 禁忌方向 / 洞察</span>
          </div>
          <div v-if="!experience.length" class="normal-mode-empty">经验蒸馏尚未产出（挖掘过程会自动沉淀）</div>
          <div v-else class="rmb-exp-groups">
            <div class="rmb-exp-group" v-for="group in experienceGroups" :key="group.kind">
              <h4>{{ group.label }}</h4>
              <div v-if="!group.items.length" class="normal-mode-empty">暂无</div>
              <div v-for="item in group.items" :key="item.id" class="rmb-exp-card">
                <div class="rmb-exp-content">{{ item.content }}</div>
                <div class="rmb-exp-meta">
                  <span v-if="item.template">模板：{{ item.template }}</span>
                  <span v-if="item.occurrence_count > 1">出现 {{ item.occurrence_count }} 次</span>
                  <span v-if="item.example_factors && item.example_factors.length">实例：{{ item.example_factors.join('、') }}</span>
                  <span v-if="item.correlated && item.correlated.length">关联：{{ item.correlated.join('、') }}</span>
                  <span v-if="item.typical_correlation != null">典型相关 {{ item.typical_correlation }}</span>
                  <span class="rmb-exp-time">{{ formatTime(item.updated_at) }}</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- ═══ SSPM 编辑统计层 ═══ -->
        <div class="rmb-section">
          <div class="summary-panel-head">
            <h3>编辑统计层（SSPM）</h3>
            <span class="rmb-section-sub">信号族 × 编辑类型 × 父本桶 · 门控口径与 Agent 运行时一致</span>
          </div>
          <div class="rmb-gate-legend">
            <span class="rmb-legend-label">门控图例</span>
            <span v-for="g in gateLegend" :key="g.key" class="rmb-gate-badge" :class="'rmb-gate-' + g.key" :title="g.desc">{{ g.label }}</span>
          </div>
          <div class="summary-table-wrap rmb-cells-wrap" v-if="cells.length">
            <table class="summary-table rmb-cells-table">
              <thead>
                <tr>
                  <th :class="thClass('family')" @click="setSort('family')">信号族 <span class="sort-ind" v-if="sortKey==='family'">{{ arrow }}</span></th>
                  <th :class="thClass('motif')" @click="setSort('motif')">编辑类型 <span class="sort-ind" v-if="sortKey==='motif'">{{ arrow }}</span></th>
                  <th :class="thClass('parent_bucket')" @click="setSort('parent_bucket')">父本桶 <span class="sort-ind" v-if="sortKey==='parent_bucket'">{{ arrow }}</span></th>
                  <th>显式 成/败</th>
                  <th>隐式 成/败</th>
                  <th :class="thClass('weighted_n')" @click="setSort('weighted_n')">加权尝试 <span class="sort-ind" v-if="sortKey==='weighted_n'">{{ arrow }}</span></th>
                  <th :class="thClass('fail_rate')" @click="setSort('fail_rate')">失败率 <span class="sort-ind" v-if="sortKey==='fail_rate'">{{ arrow }}</span></th>
                  <th :class="thClass('confidence')" @click="setSort('confidence')">置信度 <span class="sort-ind" v-if="sortKey==='confidence'">{{ arrow }}</span></th>
                  <th :class="thClass('weighted_fail')" @click="setSort('weighted_fail')">加权失败 <span class="sort-ind" v-if="sortKey==='weighted_fail'">{{ arrow }}</span></th>
                  <th :class="thClass('residual_count')" @click="setSort('residual_count')">残差n <span class="sort-ind" v-if="sortKey==='residual_count'">{{ arrow }}</span></th>
                  <th>门控</th>
                  <th :class="thClass('updated_at')" @click="setSort('updated_at')">更新时间 <span class="sort-ind" v-if="sortKey==='updated_at'">{{ arrow }}</span></th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="cell in sortedCells" :key="cell.family + '/' + cell.motif + '/' + cell.parent_bucket">
                  <td class="rmb-cell-key">{{ cell.family }}</td>
                  <td class="rmb-cell-key">{{ cell.motif }}</td>
                  <td>{{ cell.parent_bucket }}</td>
                  <td class="rmb-num">{{ fmtW(cell.explicit_s) }} / {{ fmtW(cell.explicit_f) }}</td>
                  <td class="rmb-num">{{ fmtW(cell.implicit_s) }} / {{ fmtW(cell.implicit_f) }}</td>
                  <td class="rmb-num">{{ fmtW(cell.weighted_n) }}</td>
                  <td class="rmb-num" :class="{ 'rmb-neg': cell.fail_rate >= 0.6 }">{{ pct(cell.fail_rate) }}</td>
                  <td class="rmb-num">{{ cell.confidence }}</td>
                  <td class="rmb-num" :class="{ 'rmb-neg': cell.weighted_fail > 0 }">{{ fmtW(cell.weighted_fail) }}</td>
                  <td class="rmb-num">{{ cell.residual_count }}</td>
                  <td><span class="rmb-gate-badge" :class="'rmb-gate-' + cell.gate" :title="gateDesc(cell.gate)">{{ gateLabel(cell.gate) }}</span></td>
                  <td class="summary-time">{{ formatTime(cell.updated_at) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div v-else class="normal-mode-empty">暂无 SSPM 单元</div>
        </div>

        <!-- ═══ 原始证据层 ═══ -->
        <div class="rmb-section">
          <div class="summary-panel-head">
            <h3>原始证据层</h3>
            <span class="rmb-section-sub">点击行查看详情 · 共 {{ agent.researchMemoryTotal }} 条（已载 {{ agent.researchMemory.length }}）</span>
          </div>
          <div class="rmb-filters">
            <input class="rmb-search" v-model="search" placeholder="搜索因子名 / 表达式 / 结论 / 失败码" />
            <select class="rmb-select" v-model="verdictFilter">
              <option value="">全部状态</option>
              <option v-for="v in verdictOptions" :key="v" :value="v">{{ memoryVerdictLabel(v) }}</option>
            </select>
            <select class="rmb-select" v-model="familyFilter">
              <option value="">全部信号族</option>
              <option v-for="f in familyOptions" :key="f" :value="f">{{ f }}</option>
            </select>
          </div>
          <div class="summary-table-wrap" v-if="filteredEntries.length">
            <table class="summary-table rmb-entries-table">
              <thead>
                <tr>
                  <th>因子名称</th>
                  <th>状态</th>
                  <th>信号族</th>
                  <th>IC</th>
                  <th>ICIR</th>
                  <th>次数</th>
                  <th>结论 / 失败原因</th>
                  <th>更新时间</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="entry in filteredEntries" :key="entry.id" class="summary-row" @click="agent.memoryDetail = entry">
                  <td class="summary-name" :title="entry.factor_name">{{ entry.factor_name || 'unnamed' }}</td>
                  <td><span class="summary-verdict-tag" :class="'memv-' + entry.verdict">{{ memoryVerdictLabel(entry.verdict) }}</span></td>
                  <td>{{ entry.family || '—' }}</td>
                  <td :class="icClass(entry.metrics?.ic)">{{ formatMetricValue(entry.metrics?.ic ?? '—') }}</td>
                  <td>{{ formatMetricValue(entry.metrics?.icir ?? '—') }}</td>
                  <td>{{ entry.attempts || 1 }}</td>
                  <td class="summary-reason" :title="entry.conclusion || entry.error || ''">{{ entry.conclusion || entry.error || '—' }}</td>
                  <td class="summary-time">{{ formatTime(entry.updated_at) }}</td>
                  <td><button class="rmb-del" @click.stop="removeEntry(entry)">删除</button></td>
                </tr>
              </tbody>
            </table>
          </div>
          <div v-else class="normal-mode-empty">{{ agent.researchMemory.length ? '没有匹配的条目' : '暂无研究记忆' }}</div>
          <div class="rmb-load-more" v-if="agent.researchMemoryHasMore">
            <button class="summary-refresh-btn" :disabled="agent.researchMemoryLoading" @click="agent.loadMoreResearchMemory()">
              {{ agent.researchMemoryLoading ? '加载中…' : '加载更多' }}
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- 长期记忆详情弹窗 -->
    <memory-detail-modal />
  </main>
</template>

<script>
import '../../styles/alphaagent.css'
import { agentStore } from '../../store/alphaagent.js'
import MemoryDetailModal from './MemoryDetailModal.vue'
import {
  formatTime, memoryVerdictLabel,
  formatMetricValue, icClass,
} from '../../utils/alphaagent.js'

// 门控口径 = retrieval._edit_prior_block（推荐/否决四档）+ advisory APV 双门（聚合否决）
// 描述里的阈值占位符 {hard}/{recommend}/{veto} 由后端返回的 thresholds 动态填充
const GATE_META = {
  hard_recommend: { label: '硬推荐', desc: '有显式成功且置信 > {hard}，注入"优先采用"档' },
  soft_recommend: { label: '软推荐', desc: '有显式成功且置信 > {recommend}，注入"优先尝试"档' },
  hard_veto: { label: '硬否决', desc: '有失败记录且置信 > {hard}，注入"不要采用"档' },
  soft_veto: { label: '软否决', desc: '有失败记录且置信 > {veto}，注入"谨慎/换向"档' },
  apv_hard_veto: { label: 'APV硬否决', desc: '评估前双门否决（置信>τc 且失败后验>τv），advisory 硬提醒换方向' },
  not_injected: { label: '不注入', desc: '置信不足或无显式成败，该单元不会进入提示词' },
}
const ASC_FIRST_KEYS = new Set(['family', 'motif', 'parent_bucket'])

export default {
  name: 'ResearchMemoryBank',
  components: {
    'memory-detail-modal': MemoryDetailModal,
  },
  data() {
    return {
      agent: agentStore,
      loading: false,
      search: '',
      verdictFilter: '',
      familyFilter: '',
      sortKey: 'weighted_fail',
      sortOrder: -1,
    }
  },
  computed: {
    stats() {
      return this.agent.researchMemoryStats
    },
    cells() {
      return this.agent.memoryLayers?.cells || []
    },
    experience() {
      return this.agent.memoryLayers?.experience || []
    },
    thresholds() {
      const t = this.agent.memoryLayers?.thresholds || {}
      return {
        hard_conf: t.hard_conf ?? 0.7,
        recommend_conf: t.recommend_conf ?? 0.4,
        veto_conf: t.veto_conf ?? 0.3,
      }
    },
    experienceGroups() {
      const defs = [
        { kind: 'success_pattern', label: '成功模式' },
        { kind: 'forbidden', label: '禁忌方向' },
        { kind: 'insight', label: '洞察' },
      ]
      return defs.map(d => ({ ...d, items: this.experience.filter(e => e.kind === d.kind) }))
    },
    gateLegend() {
      return ['hard_recommend', 'soft_recommend', 'hard_veto', 'soft_veto', 'apv_hard_veto', 'not_injected']
        .map(k => ({ key: k, label: GATE_META[k].label, desc: this.gateDesc(k) }))
    },
    verdictOptions() {
      return Object.keys(this.stats?.verdict_counts || {}).sort()
    },
    familyOptions() {
      const set = new Set()
      for (const e of this.agent.researchMemory) if (e.family) set.add(e.family)
      return [...set].sort()
    },
    filteredEntries() {
      const q = this.search.trim().toLowerCase()
      return this.agent.researchMemory.filter((e) => {
        if (this.verdictFilter && e.verdict !== this.verdictFilter) return false
        if (this.familyFilter && (e.family || '') !== this.familyFilter) return false
        if (!q) return true
        const hay = [e.factor_name, e.expression, e.conclusion, e.error, e.failure_code, e.family]
          .join('\n').toLowerCase()
        return hay.includes(q)
      })
    },
    sortedCells() {
      const key = this.sortKey
      const dir = this.sortOrder
      return [...this.cells].sort((a, b) => {
        const va = a[key]
        const vb = b[key]
        if (typeof va === 'string' || typeof vb === 'string') {
          return String(va ?? '').localeCompare(String(vb ?? '')) * dir
        }
        return ((va ?? 0) - (vb ?? 0)) * dir
      })
    },
    arrow() {
      return this.sortOrder === 1 ? '▲' : '▼'
    },
  },
  mounted() {
    this.refresh()
  },
  methods: {
    formatTime,
    memoryVerdictLabel,
    formatMetricValue,
    icClass,
    pct(v) {
      return (v * 100).toFixed(1) + '%'
    },
    fmtW(v) {
      const n = Number(v || 0)
      return Number.isInteger(n) ? String(n) : n.toFixed(1)
    },
    gateLabel(key) {
      return GATE_META[key]?.label || key
    },
    gateDesc(key) {
      const th = this.thresholds
      return (GATE_META[key]?.desc || '')
        .replaceAll('{hard}', th.hard_conf)
        .replaceAll('{recommend}', th.recommend_conf)
        .replaceAll('{veto}', th.veto_conf)
    },
    thClass(key) {
      return { sortable: true, 'sort-active': this.sortKey === key }
    },
    setSort(key) {
      if (this.sortKey === key) {
        this.sortOrder = -this.sortOrder
      } else {
        this.sortKey = key
        this.sortOrder = ASC_FIRST_KEYS.has(key) ? 1 : -1
      }
    },
    async removeEntry(entry) {
      const name = entry.factor_name || entry.id
      if (!window.confirm(`删除研究记忆「${name}」？Agent 将不再参考该条目，SSPM 单元与观察记录同步清理。`)) return
      await agentStore.deleteMemoryEntry(entry)
    },
    async refresh() {
      this.loading = true
      try {
        await Promise.all([
          agentStore.loadResearchMemory(true),
          agentStore.loadMemoryLayers(),
        ])
      } finally {
        this.loading = false
      }
    },
  },
}
</script>
