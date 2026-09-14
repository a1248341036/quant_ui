<template>
  <div class="ovm-widget" :class="{ dragging }" v-if="visible" :style="widgetPosStyle">
    <!-- 收起态：只显示一个圆形按钮 -->
    <button v-if="!expanded" class="ovm-fab" @click="expanded = true" title="整夜挖掘监控">
      <span class="ovm-fab-icon">⛏</span>
      <i v-if="status === 'running'" class="ovm-fab-dot"></i>
    </button>

    <!-- 展开态：悬浮窗面板 -->
    <div v-else class="ovm-panel">
      <div class="ovm-header" @mousedown="onDragStart" @touchstart.passive="onDragStart" title="按住拖动">
        <span class="ovm-drag-handle">⠿</span>
        <span class="ovm-title">整夜挖掘监控</span>
        <span :class="['ovm-status', 'ovm-status-' + status]">{{ statusLabel }}</span>
        <button class="ovm-close" @mousedown.stop @click="expanded = false" title="收起">−</button>
      </div>

      <div class="ovm-body">
        <!-- 启动表单（idle/stopped/failed 时显示） -->
        <div v-if="status === 'idle' || status === 'stopped' || status === 'failed'" class="ovm-form">
          <div class="ovm-field ovm-field-wide">
            <span>停止日期与时间</span>
            <div class="ovm-datetime">
              <input type="date" v-model="form.deadline_date" :min="todayStr" />
              <input type="time" v-model="form.deadline_time" step="300" />
            </div>
            <div class="ovm-quick-btns">
              <button v-for="q in quickOptions" :key="q.label" type="button"
                class="ovm-quick-btn" :class="{ active: quickActive(q) }"
                :title="q.title" @click="applyQuick(q)">{{ q.label }}</button>
            </div>
            <span class="ovm-facet-hint ovm-deadline-hint">{{ deadlineHint }}</span>
          </div>
          <label class="ovm-field">
            <span>最大 run 数（0=不限）</span>
            <input type="number" v-model.number="form.max_runs" min="0" max="50" />
          </label>
          <label class="ovm-field ovm-field-wide">
            <span>数据面聚焦</span>
            <span class="ovm-facet-hint">{{ facetHint }}</span>
            <div class="facet-chips ovm-facet-chips">
              <button
                v-for="f in agent.focusFacetOptions"
                :key="f.key"
                class="facet-chip"
                :class="{ active: form.focus_facets.includes(f.key) }"
                :title="f.hint"
                @click="toggleFacet(f.key)"
              >{{ f.label }}</button>
            </div>
          </label>
          <div class="ovm-toggles">
            <label class="ovm-toggle">
              <input type="checkbox" v-model="form.restart_backend" />
              <span>启动时重启后端</span>
            </label>
            <label class="ovm-toggle">
              <input type="checkbox" v-model="form.keep_awake" />
              <span>阻止系统睡眠</span>
            </label>
          </div>
          <button class="ovm-start-btn" @click="startMonitor" :disabled="starting">
            {{ starting ? '启动中...' : '启动监控' }}
          </button>
          <p v-if="startError" class="ovm-error">{{ startError }}</p>
        </div>

        <!-- 运行态信息 -->
        <div v-if="status === 'running'" class="ovm-running-info">
          <div class="ovm-info-row">
            <span class="ovm-info-label">停止时间</span>
            <span class="ovm-info-val">{{ params.deadline || '07:00' }}</span>
          </div>
          <div class="ovm-info-row" v-if="params.max_runs">
            <span class="ovm-info-label">最大 run 数</span>
            <span class="ovm-info-val">{{ params.max_runs }}</span>
          </div>
          <div class="ovm-info-row" v-if="params.focus_facets">
            <span class="ovm-info-label">数据面</span>
            <span class="ovm-info-val">{{ focusFacetsText }}</span>
          </div>
          <div class="ovm-info-row">
            <span class="ovm-info-label">启动时间</span>
            <span class="ovm-info-val">{{ params.started_at || '-' }}</span>
          </div>

          <!-- 当前运行中的挖掘 run：可逐条停止 -->
          <div v-if="activeRuns.length" class="ovm-runs-block">
            <div class="ovm-runs-title">当前挖掘 run（{{ activeRuns.length }}）</div>
            <div v-for="r in activeRuns" :key="r.run_id" class="ovm-run-item">
              <div class="ovm-run-meta">
                <span class="ovm-run-id" :title="r.run_id">{{ shortRunId(r.run_id) }}</span>
                <span :class="['ovm-run-status', 'ovm-run-status-' + r.status]">{{ runStatusLabel(r.status) }}</span>
                <span class="ovm-run-events" v-if="r.event_count">{{ r.event_count }} 事件</span>
              </div>
              <button
                class="ovm-run-stop"
                :disabled="isStoppingRun(r.run_id) || r.status === 'stopping'"
                @click="stopRun(r.run_id)"
              >{{ r.status === 'stopping' ? '停止中…' : '停止' }}</button>
            </div>
          </div>
          <div v-else class="ovm-no-runs">当前无运行中的 run</div>

          <button class="ovm-stop-btn" @click="stopMonitor" :disabled="stopping">
            {{ stopping ? '停止中...' : '停止监控' }}
          </button>
        </div>

        <!-- 日志尾 -->
        <div v-if="logTail.length" class="ovm-log-section">
          <div class="ovm-log-header">
            <span>实时日志</span>
            <span class="ovm-log-count">{{ logTail.length }} 行</span>
          </div>
          <pre class="ovm-log-tail">{{ logTailText }}</pre>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { api } from '../../utils/api.js'
import { agentStore } from '../../store/alphaagent.js'

const POS_KEY = 'ovm-widget-pos'

export default {
  name: 'OvernightMonitorWidget',
  data() {
    return {
      agent: agentStore,
      visible: true,
      expanded: false,
      status: 'idle',
      params: {},
      logTail: [],
      activeRuns: [],
      stoppingRuns: {},
      starting: false,
      stopping: false,
      startError: '',
      pollTimer: null,
      // 拖动位置（像素，相对视口右下角锚点）
      pos: { right: 20, bottom: 20 },
      dragging: false,
      form: {
        deadline_date: '',   // YYYY-MM-DD；空 = 用户未选
        deadline_time: '07:00',
        max_runs: 0,
        focus_facets: [],
        restart_backend: true,
        keep_awake: true,
      },
    }
  },
  computed: {
    todayStr() {
      // 本地时区今天的 YYYY-MM-DD（date input min 用）
      const d = new Date()
      const p = n => String(n).padStart(2, '0')
      return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`
    },
    // 快捷选项：相对当前时刻生成（每次展开面板时刷新）
    quickOptions() {
      const mk = (dt) => {
        const p = n => String(n).padStart(2, '0')
        return `${dt.getFullYear()}-${p(dt.getMonth() + 1)}-${p(dt.getDate())}` +
               'T' + `${p(dt.getHours())}:${p(dt.getMinutes())}`
      }
      const tomorrow7 = new Date(Date.now() + 24 * 3600e3)
      tomorrow7.setHours(7, 0, 0, 0)
      return [
        { label: '今夜 24:00', title: '今天午夜前不再新开 run', at: mk(new Date(Date.now() + 24 * 3600e3)) },
        { label: '明早 7:00', title: '明早 7 点前不再新开 run', at: mk(tomorrow7) },
        { label: '明早 8:30', title: '明早 8 点半前不再新开 run', at: mk(new Date(tomorrow7.getTime() + 90 * 60e3)) },
        { label: '+24h', title: '从现在起 24 小时', at: mk(new Date(Date.now() + 24 * 3600e3)) },
        { label: '+48h', title: '从现在起 48 小时', at: mk(new Date(Date.now() + 48 * 3600e3)) },
      ]
    },
    deadlineValue() {
      // 组装发送给后端的 deadline：未选日期时只发 HH:MM（后端/脚本自动顺延到明天）
      if (!this.form.deadline_date) return this.form.deadline_time || '07:00'
      return `${this.form.deadline_date}T${this.form.deadline_time || '07:00'}`
    },
    deadlineHint() {
      if (!this.form.deadline_date) {
        return `未选日期 = 明天 ${this.form.deadline_time || '07:00'} 自动停止`
      }
      return `到 ${this.form.deadline_date.replace(/-/g, '/')} ${this.form.deadline_time} 为止不再新开 run`
    },
    statusLabel() {
      const map = {
        idle: '空闲',
        running: '运行中',
        stopped: '已停止',
        failed: '已退出',
        completed: '已完成',
      }
      return map[this.status] || this.status
    },
    logTailText() {
      return (this.logTail || []).join('\n')
    },
    // 面标签：key → label（运行态展示用）
    focusFacetsText() {
      const list = this.params.focus_facets
      if (!list) return ''
      const arr = Array.isArray(list) ? list : String(list).split(',')
      const labels = arr.map(k => this.facetLabel(k.trim()))
      return labels.join(' + ')
    },
    facetHint() {
      const n = this.form.focus_facets.length
      if (!n) return '未选 = 不限（Agent 自主探索）'
      if (n >= 2) return `已聚焦 ${n} 面，可跨面融合`
      return '单面聚焦：表达式应触及该面列族'
    },
    widgetPosStyle() {
      return {
        right: this.pos.right + 'px',
        bottom: this.pos.bottom + 'px',
      }
    },
  },
  mounted() {
    this.loadPos()
    this.fetchStatus()
    this.startPolling()
  },
  beforeUnmount() {
    this.stopPolling()
    this.unbindDrag()
  },
  methods: {
    facetLabel(key) {
      const opt = this.agent.focusFacetOptions.find(o => o.key === key)
      return opt ? opt.label : key
    },
    toggleFacet(key) {
      const list = this.form.focus_facets
      this.form.focus_facets = list.includes(key)
        ? list.filter(k => k !== key)
        : [...list, key]
    },
    // ── 拖动 ──
    loadPos() {
      try {
        const saved = JSON.parse(localStorage.getItem(POS_KEY) || 'null')
        if (saved && typeof saved.right === 'number' && typeof saved.bottom === 'number') {
          this.pos = saved
        }
      } catch { /* 忽略损坏数据 */ }
    },
    savePos() {
      try { localStorage.setItem(POS_KEY, JSON.stringify(this.pos)) } catch { /* 忽略 */ }
    },
    onDragStart(e) {
      // 关闭按钮已通过 @mousedown.stop 阻断；此处记录起始位置
      this.dragging = true
      const rect = this.$el.getBoundingClientRect()
      this._drag = {
        startX: e.clientX,
        startY: e.clientY,
        startRight: this.pos.right,
        startBottom: this.pos.bottom,
        elRight: window.innerWidth - rect.right,
        elBottom: window.innerHeight - rect.bottom,
      }
      document.addEventListener('mousemove', this.onDragMove)
      document.addEventListener('mouseup', this.onDragEnd)
      e.preventDefault()
    },
    onDragMove(e) {
      if (!this.dragging) return
      const dx = this._drag.startX - e.clientX
      const dy = this._drag.startY - e.clientY
      // 以"元素右/下边到视口边缘的距离"为锚，向左/上拖动时 right/bottom 增大
      this.pos.right = this._drag.startRight + dx
      this.pos.bottom = this._drag.startBottom + dy
      this.clampPos()
    },
    onDragEnd() {
      if (!this.dragging) return
      this.dragging = false
      document.removeEventListener('mousemove', this.onDragMove)
      document.removeEventListener('mouseup', this.onDragEnd)
      this.savePos()
    },
    unbindDrag() {
      document.removeEventListener('mousemove', this.onDragMove)
      document.removeEventListener('mouseup', this.onDragEnd)
    },
    // 位置收边：不许拖出视口（留出最小可视宽度）
    clampPos() {
      const el = this.$el
      const w = el ? el.getBoundingClientRect().width : 48
      const h = el ? el.getBoundingClientRect().height : 48
      const maxRight = window.innerWidth - 48
      const maxBottom = window.innerHeight - 48
      this.pos.right = Math.max(0, Math.min(this.pos.right, maxRight))
      this.pos.bottom = Math.max(0, Math.min(this.pos.bottom, maxBottom))
    },
    // ── 快捷选项 ──
    quickActive(q) {
      return this.deadlineValue === q.at
    },
    applyQuick(q) {
      // 快捷选项直接填入 date/time 输入框（用户可再微调）
      const [d, t] = q.at.split('T')
      this.form.deadline_date = d
      this.form.deadline_time = t
    },
    // ── 轮询 / API ──
    startPolling() {
      this.stopPolling()
      // 展开时 3 秒轮询，收起时 10 秒
      this.pollTimer = setInterval(() => {
        this.fetchStatus()
      }, this.expanded ? 3000 : 10000)
    },
    stopPolling() {
      if (this.pollTimer) {
        clearInterval(this.pollTimer)
        this.pollTimer = null
      }
    },
    async fetchStatus() {
      try {
        const data = await api('/api/alphaagent/overnight-monitor/status')
        this.status = data.status || 'idle'
        this.params = data.params || (data.started_at ? { started_at: data.started_at } : {})
        this.logTail = data.log_tail || []
        this.activeRuns = Array.isArray(data.active_runs) ? data.active_runs : []
        // 如果发现 monitor 在运行，自动展开面板（首次检测到）
        if (this.status === 'running' && !this.expanded && !this._autoExpanded) {
          this.expanded = true
          this._autoExpanded = true
        }
      } catch {
        // 静默
      }
    },
    async startMonitor() {
      this.startError = ''
      // 前置校验：选定日期时间必须在未来（提前拦住，避免后端 422 才发现）
      const at = this.form.deadline_date
        ? new Date(`${this.form.deadline_date}T${this.form.deadline_time || '07:00'}`)
        : null
      if (at && (isNaN(at.getTime()) || at.getTime() <= Date.now())) {
        this.startError = '停止时间必须晚于当前时刻，请重新选择'
        return
      }
      this.starting = true
      try {
        await api('/api/alphaagent/overnight-monitor/start', {
          method: 'POST',
          body: {
            ...this.form,
            deadline: this.deadlineValue,
          },
        })
        this.starting = false
        this.fetchStatus()
      } catch (e) {
        this.startError = e.message || '启动失败'
        this.starting = false
      }
    },
    async stopMonitor() {
      this.stopping = true
      try {
        await api('/api/alphaagent/overnight-monitor/stop', {
          method: 'POST',
        })
        this.stopping = false
        this.fetchStatus()
      } catch {
        this.stopping = false
      }
    },
    // ── 停止当前挖掘 run ──
    shortRunId(id) {
      return String(id || '').slice(0, 8)
    },
    runStatusLabel(s) {
      const map = { starting: '启动中', running: '运行中', stopping: '停止中' }
      return map[s] || s
    },
    isStoppingRun(runId) {
      return !!this.stoppingRuns[runId]
    },
    async stopRun(runId) {
      if (!runId || this.stoppingRuns[runId]) return
      this.stoppingRuns = { ...this.stoppingRuns, [runId]: true }
      try {
        await api('/api/alphaagent/runs/' + encodeURIComponent(runId) + '/stop', {
          method: 'POST',
        })
      } catch {
        // 停止请求失败：短暂提示后清除状态，让轮询恢复展示
      }
      this.stoppingRuns = { ...this.stoppingRuns, [runId]: false }
      this.fetchStatus()
    },
  },
  watch: {
    expanded() {
      // 展开状态变化时调整轮询频率
      this.startPolling()
    },
  },
}
</script>
