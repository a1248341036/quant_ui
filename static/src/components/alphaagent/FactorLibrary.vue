<template>
  <div class="lib-panel">
    <div class="lib-head">
      <div class="lib-tabs">
        <button :class="{active: lib.library==='production'}" @click="switchLibrary('production')">正式因子库</button>
        <button :class="{active: lib.library==='candidate'}" @click="switchLibrary('candidate')">候选因子库</button>
      </div>
      <div class="lib-cat-tabs" v-if="false" hidden>
        <!-- 统一大库（2026-09-03）：模式页签退役——两模式共享同一物理库，类别过滤由 facet 筛选取代 -->
        <button v-for="m in agent.researchModes" :key="m.value"
                :class="{active: lib.category===m.value}"
                @click="switchCategory(m.value)">{{ m.label }}</button>
      </div>
      <div class="lib-cat-tabs" title="按数据面筛选（后端 facets 标签）">
        <button :class="{active: !lib.facet}" @click="switchFacet('')">全部</button>
        <button v-for="f in agent.focusFacetOptions" :key="f.key"
                :class="{active: lib.facet===f.key}"
                @click="switchFacet(f.key)">{{ f.label }}</button>
        <button :class="{active: lib.facet==='融合'}" @click="switchFacet('融合')">融合</button>
      </div>
      <span class="lib-count" v-if="lib.data">{{ lib.data.n_factors || 0 }} 个因子</span>
      <button class="lib-export-json" @click="exportAllJSON" :disabled="!lib.data?.factors?.length"
              :title="'导出当前' + (lib.library==='production'?'正式':'候选') + '因子库全部 ' + (lib.data?.n_factors || 0) + ' 个因子的 registry 和结果到 JSON'">
        导出{{ lib.library==='production' ? '正式库' : '候选库' }}JSON
      </button>
      <button class="lib-refresh" @click="loadFactors">刷新</button>
    </div>
    <div v-if="lib.error" class="lib-error">{{ lib.error }}</div>
    <div v-if="lib.loading" class="lib-loading">加载中…</div>
    <div v-if="lib.data && !lib.data.factors?.length && !lib.loading" class="lib-empty">
      {{ lib.data.error === 'library_not_initialized' ? '因子库尚未初始化，请在挖掘流程中启用因子提交。' : '暂无因子。' }}
    </div>
    <div class="lib-toolbar">
      <span class="lib-toolbar-label">数据面</span>
      <button class="lib-facet-btn" :class="{active: !lib.facetFilter}"
              @click="lib.facetFilter = ''" title="显示全部数据面">全部</button>
      <button v-for="opt in agent.focusFacetOptions" :key="opt.key"
              class="lib-facet-btn" :class="{active: lib.facetFilter === opt.key}"
              :title="opt.hint" @click="lib.facetFilter = lib.facetFilter === opt.key ? '' : opt.key">
        {{ opt.label }}
      </button>
      <button class="lib-facet-btn fusion" :class="{active: lib.facetFilter === '融合'}"
              title="跨数据面融合因子（触及 ≥2 个数据面）"
              @click="lib.facetFilter = lib.facetFilter === '融合' ? '' : '融合'">融合</button>
      <span class="lib-toolbar-label" style="margin-left:12px">调仓</span>
      <button class="lib-facet-btn" :class="{active: !lib.freqFilter}"
              @click="lib.freqFilter = ''" title="显示全部调仓频率">全部</button>
      <button class="lib-facet-btn" :class="{active: lib.freqFilter === 'weekly'}"
              title="每周五信号、次日起仓（名义每 5 个交易日；短周期档默认）"
              @click="lib.freqFilter = lib.freqFilter === 'weekly' ? '' : 'weekly'">5天</button>
      <button class="lib-facet-btn" :class="{active: lib.freqFilter === 'daily'}"
              title="每日调仓（间隔 1 个交易日）"
              @click="lib.freqFilter = lib.freqFilter === 'daily' ? '' : 'daily'">1天</button>
      <button class="lib-facet-btn" :class="{active: lib.freqFilter === 'monthly'}"
              title="每月末调仓一次（自然月，约 20 个交易日；慢信号档默认）"
              @click="lib.freqFilter = lib.freqFilter === 'monthly' ? '' : 'monthly'">≈20天</button>
      <span class="lib-toolbar-label" style="margin-left:auto">按加入时间导出</span>
      <input type="date" v-model="lib.exportStart">
      <span class="lib-range-sep">~</span>
      <input type="date" v-model="lib.exportEnd">
      <button class="lib-export-all" @click="exportLibraryByTime"
              :title="'将导出 ' + (libExportRows?.length || 0) + ' 个因子'">
        导出 CSV（{{ libExportRows?.length || 0 }}）
      </button>
    </div>
    <table v-if="lib.data?.factors?.length" class="lib-table">
      <thead>
        <tr>
          <th v-for="c in lib.columns" :key="c.key"
              :class="{ sortable: c.sortable, active: lib.sortKey === c.key }"
              @click="c.sortable && toggleLibSort(c.key)"
              :title="c.sortable ? '点击排序' : ''">
            {{ c.label }}<span v-if="lib.sortKey === c.key">{{ lib.sortDir > 0 ? ' ▲' : ' ▼' }}</span>
          </th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="f in libFactorsSorted" :key="f.factor_id">
          <td class="lib-fid" @click="showFactorDetail(f)" :title="f.factor_id">{{ f.name }}</td>
          <td class="lib-facets" :title="(f.facets || []).join(' + ')">
            <span v-if="f.is_fusion" class="lib-facet-tag fusion" :title="'跨面融合：' + (f.facets || []).join(' + ')">融</span>
            <span class="lib-facet-text">{{ f.facets_label || '—' }}</span>
          </td>
          <td :class="icClass(f.train_ic)">{{ formatMetricValue(f.train_ic ?? '—') }}</td>
          <td :class="icClass(f.val_ic)">{{ formatMetricValue(f.val_ic ?? '—') }}</td>
          <td :class="icClass(f.metrics?.ic)"><strong>{{ formatMetricValue(f.metrics?.ic) }}</strong></td>
          <td>{{ formatMetricValue(f.metrics?.icir) }}</td>
          <td :class="{neg: (f.annualized_return ?? 0) < 0}">{{ formatMetricValue(f.annualized_return ?? '—') }}</td>
          <td>{{ formatMetricValue(f.sharpe ?? '—') }}</td>
          <td class="lib-time" :title="f.created_at">{{ fmtTime(f.created_at) }}</td>
          <td class="lib-freq">
            <span v-if="f.rebalance_freq" class="lib-freq-tag"
                  :class="'freq-' + f.rebalance_freq"
                  :title="freqCadence(f.rebalance_freq) + freqSourceHint(f.freq_source)">
              {{ freqShort(f.rebalance_freq) }}</span>
            <span v-else class="lib-freq-missing" title="该条目没有频率记录，也无法从评估标签推导">—</span>
          </td>
          <td class="lib-label" :title="f.label_col">{{ labelShort(f.label_col) }}</td>
          <td><span class="lib-status" :class="'status-' + f.status">{{ f.status }}</span></td>
          <td class="lib-review" :title="f.review_reasons">{{ f.review_reasons || '—' }}</td>
          <td class="lib-actions">
            <button class="lib-export" @click.stop="exportOne(f)" title="复制该因子的完整 registry JSON">{{ lib.exportCopied === f.factor_id ? '✓' : '导出' }}</button>
            <button class="lib-backtest" @click.stop="$emit('backtest', f)">回测</button>
            <button class="lib-del" @click="confirmDelete(f)">删除</button>
          </td>
        </tr>
      </tbody>
    </table>

    <!-- 因子详情弹窗 -->
    <Teleport to="body">
      <div v-if="factorDetail" class="factor-modal-overlay" @click="factorDetail = null">
        <div class="factor-modal" @click.stop>
          <div class="factor-modal-head">
            <strong>{{ factorDetail.name }}</strong>
            <code>{{ factorDetail.factor_id }}</code>
            <button class="factor-modal-close" @click="factorDetail = null">×</button>
          </div>
          <div class="factor-modal-body">
            <div class="factor-modal-section">
              <label>表达式</label>
              <pre class="factor-modal-expr">{{ factorDetail.expr }}</pre>
            </div>
            <div class="factor-modal-section" v-if="factorDetail.registry_entry">
              <label>Registry 记录</label>
              <pre class="factor-modal-registry">{{ JSON.stringify(factorDetail.registry_entry, null, 2) }}</pre>
            </div>
            <div class="factor-modal-meta">
              <span>状态: {{ factorDetail.status }}</span>
              <span>有限值: {{ factorDetail.finite_count }}</span>
              <span>创建: {{ formatTime(factorDetail.created_at) }}</span>
            </div>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script>
import { api } from '../../utils/api.js'
import { agentStore } from '../../store/alphaagent.js'
import { formatMetricValue, formatTime, fmtTime, labelShort, icClass, freqShort, freqCadence, freqSourceHint } from '../../utils/alphaagent.js'

export default {
  name: 'FactorLibrary',
  emits: ['backtest'],
  data() {
    return {
      agent: agentStore,
      lib: {
        library: 'production',
        category: 'technical',
        facet: '',
        facetFilter: '',
        freqFilter: '',
        data: null,
        loading: false,
        error: '',
        exportCopied: '',
        sortKey: 'created_at',
        sortDir: -1,
        exportStart: '',
        exportEnd: '',
        columns: [
          { key: 'name', label: '名称', sortable: true },
          { key: 'facets', label: '数据面', sortable: true },
          { key: 'train_ic', label: 'Train IC', sortable: true },
          { key: 'val_ic', label: 'Val IC', sortable: true },
          { key: 'ic', label: '全区间 IC', sortable: true },
          { key: 'icir', label: 'ICIR', sortable: true },
          { key: 'annualized_return', label: '多头年化', sortable: true },
          { key: 'sharpe', label: '夏普', sortable: true },
          { key: 'created_at', label: '加入时间', sortable: true },
          { key: 'rebalance_freq', label: '调仓', sortable: true },
          { key: 'label_col', label: 'Label', sortable: true },
          { key: 'status', label: '状态', sortable: true },
          { key: 'review', label: 'Reviewer 意见', sortable: false },
        ],
      },
      factorDetail: null,
    }
  },
  computed: {
    libFactorsFaceted() {
      const facet = this.lib.facetFilter
      const freq = this.lib.freqFilter
      const fs = this.lib.data?.factors || []
      return fs.filter(f => {
        if (freq) {
          if (f.rebalance_freq !== freq) return false
        }
        if (!facet) return true
        const facets = f.facets || []
        if (facet === '融合') {
          return typeof f.is_fusion === 'boolean' ? f.is_fusion : facets.length >= 2
        }
        return facets.includes(facet)
      })
    },
    libFactorsSorted() {
      const fs = this.libFactorsFaceted
      const key = this.lib.sortKey
      if (!key) return fs
      const dir = this.lib.sortDir
      const val = (f) => {
        switch (key) {
          case 'name': return String(f.name || '').toLowerCase()
          case 'label_col': return String(f.label_col || '')
          case 'status': return String(f.status || '')
          case 'rebalance_freq': return String(f.rebalance_freq || '')
          case 'facets': return String(f.facets_label || '')
          case 'created_at': { const t = Date.parse(f.created_at || ''); return Number.isFinite(t) ? t : -Infinity }
          default: {
            const raw = key === 'ic' ? f.metrics?.ic : key === 'icir' ? f.metrics?.icir : f[key]
            const v = parseFloat(raw)
            return Number.isFinite(v) ? v : -Infinity
          }
        }
      }
      return [...fs].sort((a, b) => {
        const x = val(a), y = val(b)
        return (x < y ? -1 : x > y ? 1 : 0) * dir
      })
    },
    libExportRows() {
      const s = this.lib.exportStart ? new Date(this.lib.exportStart + 'T00:00:00').getTime() : null
      const e = this.lib.exportEnd ? new Date(this.lib.exportEnd + 'T23:59:59').getTime() : null
      if (!s && !e) return this.libFactorsSorted
      return this.libFactorsFaceted.filter(f => {
        const t = Date.parse(f.created_at || '')
        if (!Number.isFinite(t)) return false
        if (s && t < s) return false
        if (e && t > e) return false
        return true
      })
    },
  },
  watch: {
    // 研究模式列表异步到达后修正类别（对齐原 loadResearchModes 中的 lib.category 兜底）
    'agent.researchModes': {
      immediate: true,
      handler(modes) {
        if (!modes?.length) return
        if (!modes.some(m => m.value === this.lib.category)) this.lib.category = modes[0].value
      },
    },
  },
  mounted() {
    this.loadFactors()
  },
  methods: {
    formatMetricValue,
    formatTime,
    fmtTime,
    labelShort,
    freqShort,
    freqCadence,
    freqSourceHint,
    icClass,
    async loadFactors() {
      this.lib.loading = true
      this.lib.error = ''
      this.lib.data = null
      try {
        const facetQ = this.lib.facet ? '&facet=' + encodeURIComponent(this.lib.facet) : ''
        this.lib.data = await api('/api/alphaagent/factors?library=' + this.lib.library + '&category=' + this.lib.category + facetQ + '&t=' + Date.now())
      } catch (e) {
        this.lib.error = e.message
      } finally {
        this.lib.loading = false
      }
    },
    switchFacet(f) {
      this.lib.facet = f
      this.loadFactors()
    },
    switchLibrary(lib) {
      this.lib.library = lib
      this.lib.facetFilter = ''
      this.loadFactors()
    },
    switchCategory(cat) {
      this.lib.category = cat
      this.lib.facetFilter = ''
      this.loadFactors()
    },
    toggleLibSort(key) {
      if (this.lib.sortKey === key) {
        this.lib.sortDir = -this.lib.sortDir
      } else {
        this.lib.sortKey = key
        this.lib.sortDir = key === 'created_at' ? -1 : 1
      }
    },
    exportLibraryByTime() {
      const rows = this.libExportRows || []
      if (!rows.length) {
        alert('所选时间范围内没有因子')
        return
      }
      const cols = ['加入时间', 'factor_id', '名称', '数据面', '融合', '调仓频率', '研究档位', '准入状态', '审查判定',
                    'Train IC', 'Val IC', '全区间 IC', 'ICIR', 'RankIC', 'Coverage',
                    '多头年化', '超额年化', '夏普', 'val保留比', 'val多头超额',
                    'Label', 'Expr']
      const esc = v => {
        const s = v === null || v === undefined ? '' : String(v)
        return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s
      }
      const num = v => (v === null || v === undefined || Number.isNaN(Number(v))) ? '' : Number(v)
      const lines = [cols.join(',')]
      for (const f of rows) {
        lines.push([
          this.fmtTime(f.created_at), f.factor_id, f.name,
          (f.facets || []).join('+'), f.is_fusion ? '是' : '',
          f.rebalance_freq ? (freqShort(f.rebalance_freq) + '/' + f.rebalance_freq) : '', f.research_mode || '',
          f.promotion_status || f.status, f.review_verdict || '',
          num(f.train_ic), num(f.val_ic), num(f.metrics?.ic), num(f.metrics?.icir),
          num(f.metrics?.rank_ic), num(f.metrics?.factor_coverage),
          num(f.annualized_return), num(f.annualized_excess_return), num(f.sharpe),
          num(f.val_ic_retention), num(f.val_long_excess),
          f.label_col || '', f.expr || '',
        ].map(esc).join(','))
      }
      const blob = new Blob(['\ufeff' + lines.join('\r\n')],
                            { type: 'text/csv;charset=utf-8' })
      const a = document.createElement('a')
      a.href = URL.createObjectURL(blob)
      a.download = 'factors_' + this.lib.library + '_' +
                   (this.lib.exportStart || 'all') + '_' + (this.lib.exportEnd || 'all') + '.csv'
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(a.href)
    },
    exportAllJSON() {
      const data = this.lib.data
      if (!data || !data.factors?.length) return
      const payload = {
        exported_at: new Date().toISOString(),
        library: data.library,
        root: data.root,
        n_factors: data.n_factors,
        factors: data.factors,
        registry: data.registry,
      }
      const text = JSON.stringify(payload, null, 2)
      const blob = new Blob([text], { type: 'application/json;charset=utf-8' })
      const a = document.createElement('a')
      a.href = URL.createObjectURL(blob)
      a.download = 'factors_' + this.lib.library + '_' +
                   new Date().toISOString().slice(0, 10) + '.json'
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(a.href)
    },
    async exportOne(factor) {
      const reg = this.lib.data?.registry
      let entry = null
      if (reg && typeof reg === 'object') {
        entry = reg[factor.factor_id] || null
      }
      if (!entry) {
        // fallback: 从 factors 列表找 extra
        const f = (this.lib.data?.factors || []).find(x => x.factor_id === factor.factor_id)
        entry = f?.extra || f || factor
      }
      const text = JSON.stringify(entry, null, 2)
      try {
        await navigator.clipboard.writeText(text)
      } catch (_) {
        const ta = document.createElement('textarea')
        ta.value = text
        document.body.appendChild(ta)
        ta.select()
        document.execCommand('copy')
        document.body.removeChild(ta)
      }
      this.lib.exportCopied = factor.factor_id
      setTimeout(() => { this.lib.exportCopied = '' }, 2000)
    },
    async showFactorDetail(f) {
      try {
        this.factorDetail = await api('/api/alphaagent/factors/' + encodeURIComponent(f.factor_id) + '?library=' + this.lib.library + '&category=' + this.lib.category + '&t=' + Date.now())
      } catch (e) {
        this.lib.error = e.message
      }
    },
    async confirmDelete(f) {
      if (!confirm('确认删除因子 ' + f.name + ' (' + String(f.factor_id).slice(0, 8) + ')？')) return
      try {
        await api('/api/alphaagent/factors/' + encodeURIComponent(f.factor_id) + '?library=' + this.lib.library + '&category=' + this.lib.category, { method: 'DELETE' })
        this.loadFactors()
      } catch (e) {
        this.lib.error = e.message
      }
    },
  },
}
</script>
