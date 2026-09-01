<template>
  <section>
    <!-- 运行切换条 -->
    <div class="card" style="padding:10px 14px;margin-bottom:10px">
      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
        <button class="ghost" @click="store.goto('code')">← 代码页</button>
        <strong>回测详情</strong>
        <span v-if="cur" class="muted">
          {{ (cur.req && cur.req.start) || cur.start }} ~ {{ (cur.req && cur.req.end) || cur.end }}
          · 资金 {{ fmt((cur.req && cur.req.capital) || cur.capital, 0) }}
        </span>
        <span class="threshold-spacer"></span>
        <button class="ghost" v-if="active" @click="stopRun">■ 停止</button>
        <button class="ghost" @click="refreshRuns">刷新</button>
      </div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:8px" v-if="runs.length">
        <button v-for="r in runs" :key="r.run_id" class="ghost"
                :style="r.run_id === runId ? 'border-color:var(--accent);color:var(--accent-strong)' : ''"
                @click="switchRun(r.run_id)"
                :title="runTitle(r)">
          <span :class="'run-dot run-' + r.phase"></span>
          {{ runLabel(r) }}
        </button>
      </div>
    </div>

    <div v-if="!cur" class="card"><div class="empty">
      暂无回测运行——去「代码」页粘贴策略并点「▶ 运行策略」，或在「历史」页打开历史回测。
    </div></div>

    <template v-else>
      <!-- 运行中: 进度 + 渐进曲线 -->
      <div v-if="active" class="card" style="padding:10px 14px;margin-bottom:10px">
        <div style="font-size:13px;display:flex;justify-content:space-between;gap:12px">
          <span><strong>{{phaseText}}</strong>
            <span class="muted" v-if="cur.total > 0"> {{cur.done}}/{{cur.total}}</span>
            <span class="muted" v-if="etaText"> · 剩余约 {{etaText}}</span>
          </span>
          <span class="muted">已用 {{fmtElapsed(cur.elapsed)}}</span>
        </div>
        <div class="jq-pbar"><div class="jq-pfill" :style="{width: pctText}"></div></div>
        <div id="jqEquity" class="chart" v-if="(cur.nav || []).length > 1"></div>
      </div>

      <div v-if="cur.phase === 'error'" class="card" style="border-left:4px solid #c62828;padding:8px 12px;margin-bottom:8px">
        <strong style="color:#c62828">运行失败: {{cur.error}}</strong>
        <pre v-if="cur.traceback" style="font-size:12px;max-height:300px;overflow:auto">{{cur.traceback}}</pre>
      </div>
      <div v-else-if="cur.phase === 'cancelled'" class="card" style="border-left:4px solid #b8860b;padding:8px 12px;margin-bottom:8px">
        <strong>已手动停止</strong>
      </div>

      <template v-if="cur.result">
        <div class="card">
          <h3>回测结果 <span class="muted" style="font-size:12px">{{cur.result.start}} ~ {{cur.result.end}} · 资金 {{fmt(cur.result.capital,0)}} · {{cur.result.codes_count}} 只候选域</span></h3>
          <div class="cards">
            <div class="metric" v-for="(v,k) in cur.result.metrics" :key="k"><div class="label">{{k}}</div><div class="value" :class="sign(v)">{{metricText(k,v)}}</div></div>
          </div>
          <div id="jqEquity" class="chart"></div>
        </div>
        <div class="card">
          <h3>期末持仓</h3>
          <div class="table-wrap"><table>
            <tr><th>代码</th><th>名称</th><th>权重</th><th>现价</th><th>市值</th></tr>
            <tr v-for="(h,i) in cur.result.holdings" :key="i">
              <td>{{h.code}}</td><td>{{h.name}}</td><td>{{pct(h.weight)}}</td><td>{{fmt(h.price,2)}}</td><td>{{fmt(h.market_value,0)}}</td>
            </tr>
          </table></div>
        </div>
        <div class="card">
          <h3>运行日志</h3>
          <pre style="max-height:300px;overflow:auto;font-size:12px">{{(cur.result.logs || []).join('\n')}}</pre>
        </div>
      </template>
    </template>
  </section>
</template>

<script>
import { api } from '../utils/api.js'
import { fmt, pct, sign, metricText } from '../utils/format.js'
import { renderLine } from '../utils/charts.js'
import { store } from '../store/index.js'

const ACTIVE = ['queued', 'context', 'minutes', 'engine']
const PHASE_TEXT = { queued: '排队中', context: '构建数据上下文', minutes: '预取分钟线',
                     engine: '逐日回测', done: '完成', error: '失败', cancelled: '已停止' }

export default {
  name: 'JqRun',
  data() {
    return {
      store,
      runId: store.jqRunId || '',
      runs: [],
      cur: null,
      pollTimer: null,
    }
  },
  computed: {
    active() { return this.cur && ACTIVE.includes(this.cur.phase) },
    phaseText() { return PHASE_TEXT[this.cur && this.cur.phase] || '' },
    pctText() {
      const s = this.cur
      if (!s || !s.total) return '0%'
      return Math.min(100, Math.round(s.done / s.total * 100)) + '%'
    },
    etaText() {
      const s = this.cur
      if (!s || !s.total || !s.done || !s.elapsed) return ''
      const eta = s.elapsed * (s.total - s.done) / s.done
      if (eta >= 90) return Math.round(eta / 60) + ' 分钟'
      return Math.round(eta) + ' 秒'
    },
  },
  mounted() {
    this.refreshRuns()
    this.openRun(this.runId)
  },
  beforeUnmount() { this.stopPoll() },
  methods: {
    fmt, pct, sign, metricText,
    runLabel(r) {
      const st = { done: '', error: '✗', cancelled: '⏹', queued: '…' }[r.phase] ?? '●'
      return `${st} ${(r.start || '').slice(5)}~${(r.end || '').slice(5)}`
    },
    runTitle(r) {
      const pctS = (v) => v == null ? '—' : (v * 100).toFixed(2) + '%'
      const st = { done: '完成', error: '失败', cancelled: '已停止' }[r.phase] || r.phase
      return `策略 ${pctS(r['策略收益'])} · 基准 ${pctS(r['基准收益'])} · ${st}`
    },
    stopPoll() {
      if (this.pollTimer) { clearInterval(this.pollTimer); this.pollTimer = null }
    },
    async refreshRuns() {
      try {
        const r = await api('/api/code/jq/runs?limit=30')
        this.runs = r.items || []
      } catch (e) { /* ignore */ }
    },
    switchRun(runId) {
      store.jqRunId = runId
      this.openRun(runId)
    },
    async openRun(runId) {
      this.stopPoll()
      this.runId = runId
      this.cur = null
      if (!runId) return
      await this.pollOnce()
      if (this.active && this.runId === runId) {
        this.pollTimer = setInterval(() => this.pollOnce(), 1000)
      }
    },
    async pollOnce() {
      let s
      try {
        s = await api('/api/code/jq/runs/' + this.runId)
      } catch (e) { return }
      if (this.runId !== s.run_id) return   // 已切换到别的运行
      this.cur = s
      if (this.active && (s.nav || []).length > 1) this.renderNav(s.nav)
      if (!this.active) {
        this.stopPoll()
        this.refreshRuns()
        if (s.result) this.renderFinal(s.result)
      }
    },
    renderNav(nav) {
      this.$nextTick(() => renderLine('jqEquity', [
        { name: '策略', dates: nav.map(x => x.date), values: nav.map(x => x.value) },
      ]))
    },
    renderFinal(r) {
      this.$nextTick(() => {
        const series = [{ name: '策略', dates: r.nav.map(x => x.date), values: r.nav.map(x => x.value) }]
        if (Array.isArray(r.benchmark) && r.benchmark.length) {
          series.push({ name: '基准 ' + (r.bench_code || ''), dash: true,
                        dates: r.benchmark.map(x => x.date), values: r.benchmark.map(x => x.value) })
        }
        renderLine('jqEquity', series)
      })
    },
    fmtElapsed(sec) {
      if (sec == null) return '0 秒'
      const s = Math.round(sec)
      if (s >= 60) return Math.floor(s / 60) + ' 分 ' + (s % 60) + ' 秒'
      return s + ' 秒'
    },
    async stopRun() {
      if (!this.runId) return
      try { await api('/api/code/jq/runs/' + this.runId + '/stop', { method: 'POST' }) } catch (e) { /* ignore */ }
    },
  },
}
</script>
