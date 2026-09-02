/**
 * AlphaAgent 研究会话 store — 跨子组件共享的响应式单例
 *
 * 从原 static/src/views/AlphaAgent.vue 单体组件中抽出：
 * - 任务（run）列表 / 当前会话 / SSE 事件流 / usage 统计
 * - 研究规范（ResearchSpec）与门槛弹窗状态
 * - 长期研究记忆 + 研究总结
 *
 * 与 store/index.js 同风格：reactive() 单例，各组件 import 后直接使用。
 * 模板里通过 computed 代理 `agent` 访问，方法名与原组件保持一致。
 */
import { reactive, computed, nextTick } from 'vue'
import { api, getWindowDefaults } from '../utils/api.js'
import {
  addUsage, usageFromEvents, emptyUsage,
  buildTimeline, computeCurrentActivity, computeLiveActivity,
  runTitle,
} from '../utils/alphaagent.js'

// EventSource 连接句柄不放进 reactive（避免被代理，也无需渲染）
let stream = null

export const agentStore = reactive({
  // ── 会话状态 ──
  runs: [],
  current: null,
  events: [],
  memory: [],
  usage: emptyUsage(),
  running: false,
  error: '',
  pendingMessages: 0,
  stickToBottom: true,
  showArchived: false,
  menuRunId: '',
  menuPosition: { top: 0, left: 0 },
  renameRunId: '',
  renameTitle: '',
  renamePosition: { top: 0, left: 0 },
  memoryDetail: null,
  composerCollapsed: false,
  agentMode: 'research',

  // ── 研究规范 / 门槛 ──
  showResearchSpec: false,
  showThresholdModal: false,
  thresholdDraft: null,
  researchSpecText: '',
  defaultResearchSpecText: '',
  specError: '',
  specSavedAt: null,
  researchSpecSaving: false,
  specLoading: false,
  specDefaultsByMode: {},
  specOverridesByMode: {},

  // ── 启动表单 ──
  form: {
    train_start: '2020-01-01',
    train_end: '2022-12-31',
    val_start: '2023-01-01',
    val_end: '2024-12-31',
    population_max: 0,
    user_message: '',
    max_turns: 5,
    max_tool_calls_per_round: 20,
    max_tool_workers: 12,
    max_parallel_eval: 12,
    allow_submit: false,
    research_mode: 'technical',
    label_col: 'label_1d_open_to_open',
    focus_facets: [],
    // 调仓频率（交付门禁）：'' = 自动（随档位默认 + LLM 按评估证据自选）
    rebalance_freq: '',
  },

  // 数据面多选选项（与 alphaagent/factor/mining/memory/expressions.py FACET_DEFS 对齐）
  focusFacetOptions: [
    { key: '价量面', label: '价量', hint: '日线 K 线/动量/反转/波动' },
    { key: '量能面', label: '量能', hint: '成交量/成交额/换手' },
    { key: '筹码面', label: '筹码', hint: 'CHIP_* 筹码分布算子' },
    { key: '拥挤面', label: '拥挤', hint: 'CROWD_* 拥挤度算子' },
    { key: '基本面', label: '基本面', hint: 'funda_* 财务列族（30+ 列）' },
    { key: '股东面', label: '股东', hint: 'holder_* 股东户数/持股集中度' },
    { key: '事件面', label: '事件', hint: '业绩预告/龙虎榜/大宗交易' },
    { key: '资金面', label: '资金流', hint: 'fund_flow 资金流入流出' },
  ],

  // ── 研究模式 / 记忆 / 总结 ──
  researchModes: [],
  researchModeOptions: [],
  researchMemory: [],
  researchMemoryTotal: 0,
  researchMemoryPage: 0,
  researchMemoryPageSize: 50,
  researchMemoryLoading: false,
  researchMemoryStats: null,
  memoryLayers: null,
  memoryLayersLoading: false,
  summarySortKey: 'updated_at',
  summarySortOrder: -1,
  summaryVerdictFilter: '',  // '' = 全部
  // 研究总结分页（服务端排序/verdict 过滤）。独立于 researchMemory 共享列表：
  // 后者的"加载更多"累计语义被 AgentThread 记忆面板 / 研究记忆库证据层使用。
  summaryEntries: [],
  summaryPage: 1,
  summaryPageSize: 50,
  summaryTotal: 0,
  summaryLoading: false,

  // 统一时间窗口默认值（后端 window_config 唯一真源，供因子实验室等复用）
  windowDefaults: null,

  // 滚动请求计数：子组件 watch 该值后执行滚动（store 不持有 DOM）
  scrollTick: 0,
  scrollForce: false,

  // ── computed 代理 ──
  agentBusy: computed(() => {
    const status = agentStore.current?.status
    return agentStore.running || ['starting', 'running', 'stopping'].includes(status)
  }),
  canStopAgent: computed(() => ['starting', 'running'].includes(agentStore.current?.status)),
  summaryTotalPages: computed(() =>
    Math.max(1, Math.ceil(agentStore.summaryTotal / agentStore.summaryPageSize))),
  summaryAllCount: computed(() =>
    agentStore.researchMemoryStats?.entries ?? agentStore.summaryTotal),
  menuRun: computed(() => agentStore.runs.find(run => run.run_id === agentStore.menuRunId) || null),
  renameRun: computed(() => agentStore.runs.find(run => run.run_id === agentStore.renameRunId) || null),
  timeline: computed(() => buildTimeline(agentStore.events)),
  currentActivity: computed(() => computeCurrentActivity(
    agentStore.current?.status,
    agentStore.pendingMessages,
    agentStore.events,
  )),
  liveActivity: computed(() => computeLiveActivity(agentStore.events, agentStore.currentActivity)),
  researchSpecCustom: computed(() => {
    const mode = agentStore.form.research_mode
    const overrides = agentStore.specOverridesByMode[mode] || {}
    return Object.keys(overrides).length > 0
  }),
  // 方案 B：档位由数据面组合推断（与后端 core.research_modes.infer_research_mode 同口径）
  inferredMode: computed(() => {
    const slow = ['基本面', '股东面']
    return slow.some(f => (agentStore.form.focus_facets || []).includes(f)) ? 'fundamental' : 'technical'
  }),
  inferredModeLabel: computed(() => {
    return agentStore.inferredMode === 'fundamental' ? '慢信号档（label_20d · 月调仓）' : '短周期档（label_1d · 周调仓）'
  }),
  researchSpecDirty: computed(() => {
    const mode = agentStore.form.research_mode
    const effective = agentStore.specDefaultsByMode[mode]
    if (!effective) return false
    return agentStore.researchSpecText !== JSON.stringify(effective, null, 2)
  }),

  // ── 生命周期 ──
  async init() {
    this.loadAgentRuns()
    this.loadResearchMemory(true)
    this.loadResearchModes()
    await this.loadWindowsConfig()
    this.loadDefaultResearchSpec().catch(() => {})
  },
  dispose() {
    if (stream) { stream.close(); stream = null }
  },
  requestScroll(force = false) {
    this.scrollForce = force
    nextTick(() => { this.scrollTick++ })
  },

  // ── 任务列表 / 会话操作 ──
  async loadAgentRuns() {
    try {
      this.runs = await api('/api/alphaagent/runs?archived_only=' + this.showArchived + '&t=' + Date.now())
    } catch (e) {
      this.error = '读取任务失败: ' + e.message
    }
  },
  async toggleArchived() {
    this.showArchived = !this.showArchived
    this.menuRunId = ''
    await this.loadAgentRuns()
  },
  async pinRun(run) {
    this.menuRunId = ''
    try {
      const updated = await api('/api/alphaagent/runs/' + encodeURIComponent(run.run_id) + '/pin', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pinned: !run.pinned }),
      })
      if (this.current?.run_id === updated.run_id) this.current = { ...this.current, pinned: updated.pinned }
      await this.loadAgentRuns()
    } catch (e) {
      this.error = (run.pinned ? '取消置顶失败: ' : '置顶失败: ') + e.message
    }
  },
  async archiveRun(run) {
    this.menuRunId = ''
    try {
      await api('/api/alphaagent/runs/' + encodeURIComponent(run.run_id) + '/archive', { method: 'POST' })
      await this.loadAgentRuns()
    } catch (e) {
      this.error = '归档失败: ' + e.message
    }
  },
  async deleteRun(run) {
    this.menuRunId = ''
    if (!window.confirm('确定删除该归档任务？日志轨迹将一并删除且不可恢复。')) return
    try {
      await api('/api/alphaagent/runs/' + encodeURIComponent(run.run_id), { method: 'DELETE' })
      this.dropCurrentIfDeleted([run.run_id])
      await this.loadAgentRuns()
    } catch (e) {
      this.error = e.message.includes('run_still_running') ? '任务仍在运行，请先停止再删除' : '删除失败: ' + e.message
    }
  },
  async deleteAllArchived() {
    const count = this.runs.length
    if (!count) return
    if (!window.confirm('确定一键删除全部 ' + count + ' 个已归档任务？日志轨迹将一并删除且不可恢复。')) return
    try {
      const result = await api('/api/alphaagent/runs/archived', { method: 'DELETE' })
      this.dropCurrentIfDeleted(result.deleted || [])
      await this.loadAgentRuns()
      if (result.skipped && result.skipped.length) {
        this.error = '已删除 ' + result.count + ' 个，' + result.skipped.length + ' 个因仍在运行被跳过'
      }
    } catch (e) {
      this.error = '一键删除失败: ' + e.message
    }
  },
  dropCurrentIfDeleted(deletedIds) {
    if (!this.current || !deletedIds.includes(this.current.run_id)) return
    if (stream) { stream.close(); stream = null }
    this.current = null
    this.events = []
  },
  async commitRename(run) {
    if (this.renameRunId !== run.run_id) return
    const title = this.renameTitle.trim()
    this.cancelRename()
    if (!title || title === (run.title || runTitle(run))) return
    try {
      const updated = await api('/api/alphaagent/runs/' + encodeURIComponent(run.run_id) + '/rename', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title }),
      })
      const index = this.runs.findIndex(item => item.run_id === updated.run_id)
      if (index >= 0) this.runs.splice(index, 1, { ...this.runs[index], ...updated })
      if (this.current?.run_id === updated.run_id) this.current = { ...this.current, title: updated.title }
      await this.loadAgentRuns()
    } catch (e) {
      this.error = '重命名失败: ' + e.message
    }
  },
  cancelRename() {
    this.renameRunId = ''
    this.renameTitle = ''
  },
  async branchRun(run) {
    this.menuRunId = ''
    const content = window.prompt('给新分支的研究指令', '基于以上研究轨迹，换一个角度继续挖掘并完成训练集和验证集检验。')
    if (content == null || !content.trim()) return
    try {
      if (stream) stream.close()
      const parent = await api('/api/alphaagent/runs/' + encodeURIComponent(run.run_id) + '?t=' + Date.now())
      const parentEvents = await this.conversationEvents(parent)
      const result = await api('/api/alphaagent/runs/' + encodeURIComponent(run.run_id) + '/branch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: content.trim() }),
      })
      this.current = result
      this.current.status = result.status || 'starting'
      this.events = [
        ...parentEvents,
        { event: 'branch_started', ts: new Date().toISOString(), parent_run_id: run.run_id, content: content.trim() },
        ...(result.events || []),
      ]
      this.usage = usageFromEvents(this.events)
      this.pendingMessages = 0
      this.running = true
      await this.loadAgentRuns()
      this.connectAgentEvents(result.run_id)
      this.requestScroll(true)
    } catch (e) {
      this.error = '新建分支失败: ' + e.message
    }
  },
  newRun() {
    if (stream) stream.close()
    stream = null
    this.current = null
    this.events = []
    this.usage = emptyUsage()
    this.pendingMessages = 0
    this.error = ''
    this.running = false
    this.form.user_message = ''
  },
  async selectAgentRun(run) {
    if (stream) stream.close()
    try {
      const detail = await api('/api/alphaagent/runs/' + encodeURIComponent(run.run_id) + '?t=' + Date.now())
      this.current = detail
      if (detail.research_spec) {
        this.researchSpecText = JSON.stringify(detail.research_spec, null, 2)
        this.specError = ''
      }
      this.events = await this.conversationEvents(detail)
      this.usage = usageFromEvents(this.events)
      this.pendingMessages = 0
      this.running = ['starting', 'running'].includes(detail.status)
      if (this.running) this.connectAgentEvents(run.run_id)
      this.requestScroll(true)
    } catch (e) {
      this.error = '读取任务失败: ' + e.message
    }
  },
  async conversationEvents(detail) {
    const events = detail.events || []
    if (!detail.parent_run_id) return events
    try {
      const parent = await api('/api/alphaagent/runs/' + encodeURIComponent(detail.parent_run_id) + '?t=' + Date.now())
      return [
        ...(await this.conversationEvents(parent)),
        { event: 'continuation_started', ts: detail.created_at, parent_run_id: parent.run_id, content: detail.user_message || '' },
        ...events,
      ]
    } catch (e) {
      return events
    }
  },
  async stopAgent() {
    if (!this.canStopAgent || !this.current) return
    try {
      await api('/api/alphaagent/runs/' + encodeURIComponent(this.current.run_id) + '/stop', { method: 'POST' })
      this.current.status = 'stopping'
      this.running = false
    } catch (e) {
      this.error = '停止失败: ' + e.message
    }
  },

  // ── 启动 / 追加指令 / 续跑 ──
  useSuggestion(text) {
    this.form.user_message = text
  },
  toggleFocusFacet(key) {
    const list = this.form.focus_facets || []
    const next = list.includes(key) ? list.filter(k => k !== key) : [...list, key]
    this.form.focus_facets = next
    this.syncInferredMode()
  },
  // 方案 B：勾面变化时自动同步评估档位（label/门槛/门禁频率跟随），复用
  // switchResearchMode 的 spec 加载链路；label_col 由 spec.recommended_label_col 覆盖。
  async syncInferredMode() {
    const mode = this.inferredMode
    if (mode && mode !== this.form.research_mode) {
      await this.switchResearchMode(mode)
    }
  },
  async startAgent() {
    if (this.agentBusy || !this.form.user_message.trim()) return
    this.error = ''
    let researchSpec
    try {
      researchSpec = this.parseResearchSpec()
    } catch (e) {
      return
    }
    this.running = true
    try {
      if (stream) stream.close()
      const result = await api('/api/alphaagent/runs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...this.form,
          // 方案 B：research_mode 取推断档位（勾面时 syncInferredMode 已同步 spec 文本）
          research_mode: this.inferredMode,
          no_fundamentals: !['基本面', '股东面'].some(f => (this.form.focus_facets || []).includes(f)) && !(this.form.focus_facets || []).length,
          rebalance_freq: this.form.rebalance_freq || null,
          research_spec: researchSpec,
          allow_submit: Boolean(researchSpec.delivery_policy?.allow_submit),
        }),
      })
      this.current = result
      this.current.status = result.status || 'starting'
      this.events = result.events || []
      this.usage = emptyUsage()
      this.pendingMessages = 0
      this.form.user_message = ''
      await this.loadAgentRuns()
      this.connectAgentEvents(result.run_id)
      this.requestScroll(true)
    } catch (e) {
      this.error = '启动失败: ' + e.message
      this.running = false
    }
  },
  async startDefaultResearch() {
    const mode = this.researchModes.find(m => m.value === this.form.research_mode)
    if (mode && mode.default_user_message) {
      this.form.user_message = mode.default_user_message
    } else {
      this.form.user_message = '请自主挖掘A股因子，先训练集评估，再验证集检验；只有通过验证和去重门槛的因子才提交。'
    }
    await this.startAgent()
  },
  async sendMessage() {
    if (!this.form.user_message.trim()) return
    if (!this.current) {
      await this.startAgent()
      return
    }
    if (this.current.status === 'stopping') return
    const content = this.form.user_message.trim()
    if (!this.agentBusy) {
      await this.resumeConversation(content)
      return
    }
    try {
      const result = await api('/api/alphaagent/runs/' + encodeURIComponent(this.current.run_id) + '/messages', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content }),
      })
      if (!result.ok) throw new Error('追加失败')
      this.events.push({ event: 'continuation_queued', ts: new Date().toISOString(), content })
      this.pendingMessages += 1
      this.form.user_message = ''
      this.requestScroll(true)
    } catch (e) {
      this.error = '追加指令失败: ' + e.message
    }
  },
  async resumeConversation(content) {
    const previousEvents = this.events.slice()
    const previousRun = this.current
    try {
      const result = await api('/api/alphaagent/runs/' + encodeURIComponent(previousRun.run_id) + '/continue', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content }),
      })
      if (result.status === 'queued') {
        this.events.push({ event: 'continuation_queued', ts: new Date().toISOString(), content })
        this.form.user_message = ''
        return
      }
      this.current = result
      this.current.status = result.status || 'starting'
      this.events = [
        ...previousEvents,
        { event: 'continuation_started', ts: new Date().toISOString(), parent_run_id: previousRun.run_id, content },
        ...(result.events || []),
      ]
      this.running = true
      this.form.user_message = ''
      await this.loadAgentRuns()
      this.connectAgentEvents(result.run_id)
      this.requestScroll(true)
    } catch (e) {
      this.error = '恢复历史会话失败: ' + e.message
    }
  },
  connectAgentEvents(runId) {
    if (stream) stream.close()
    const es = new EventSource('/api/alphaagent/runs/' + encodeURIComponent(runId) + '/events?t=' + Date.now())
    stream = es
    es.onmessage = (message) => {
      try {
        const event = JSON.parse(message.data)
        if (event.event === 'heartbeat') return
        if (event.event === 'stream_end') {
          this.running = false
          if (this.current) this.current.status = event.status || 'completed'
          es.close()
          stream = null
          this.loadAgentRuns()
          this.requestScroll(true)
          return
        }
        if (event.event === 'stream_start') {
          if (this.current) this.current.status = 'running'
        } else if (event.event === 'usage_total') {
          this.usage = { ...this.usage, ...event }
        } else {
          this.events.push(event)
          if (event.event === 'research_memory_updated') {
            this.loadResearchMemory(true)
            this.loadSummaryPage()
          }
          if (event.event === 'continuation_accepted') {
            this.pendingMessages = Math.max(0, this.pendingMessages - Number(event.count || 1))
          }
          if (event.event === 'usage') this.usage = addUsage(this.usage, event)
          if (this.current) {
            this.current.status = 'running'
            this.current.event_count = this.events.length
          }
        }
        this.requestScroll()
      } catch (e) {
        this.error = '事件解析失败: ' + e.message
      }
    }
    es.onerror = () => {
      // EventSource 会自动重连；任务运行时不要主动关闭。
      if (this.current && ['completed', 'failed', 'stopping'].includes(this.current.status)) {
        es.close()
        stream = null
        this.running = false
      }
    }
  },

  // ── 研究记忆 / 模式 ──
  async loadResearchMemory(reset = true) {
    if (this.researchMemoryLoading) return
    this.researchMemoryLoading = true
    try {
      if (reset) this.researchMemoryPage = 0
      const offset = this.researchMemoryPage * this.researchMemoryPageSize
      const payload = await api(`/api/alphaagent/research-memory?limit=${this.researchMemoryPageSize}&offset=${offset}&t=${Date.now()}`)
      const entries = payload.entries || []
      this.researchMemoryTotal = payload.total || entries.length
      this.researchMemoryStats = payload.statistics || null
      if (reset) {
        this.researchMemory = entries
      } else {
        this.researchMemory = [...this.researchMemory, ...entries]
      }
    } catch (e) {
      this.error = '读取长期研究记忆失败: ' + e.message
    } finally {
      this.researchMemoryLoading = false
    }
  },
  get researchMemoryHasMore() {
    return this.researchMemory.length < this.researchMemoryTotal
  },
  async loadMoreResearchMemory() {
    if (!this.researchMemoryHasMore || this.researchMemoryLoading) return
    this.researchMemoryPage += 1
    await this.loadResearchMemory(false)
  },
  // 已加载范围的整体刷新（轮询用）：按已加载条数一次取回并替换，不重置浏览位置
  async refreshResearchMemory() {
    if (this.researchMemoryLoading) return
    this.researchMemoryLoading = true
    try {
      const limit = Math.min(500, Math.max(this.researchMemoryPageSize, this.researchMemory.length))
      const payload = await api(`/api/alphaagent/research-memory?limit=${limit}&offset=0&t=${Date.now()}`)
      const entries = payload.entries || []
      this.researchMemoryTotal = payload.total || entries.length
      this.researchMemoryStats = payload.statistics || null
      this.researchMemory = entries
      this.researchMemoryPage = Math.max(0, Math.ceil(entries.length / this.researchMemoryPageSize) - 1)
    } catch (e) {
      this.error = '刷新长期研究记忆失败: ' + e.message
    } finally {
      this.researchMemoryLoading = false
    }
  },
  // ── 研究总结分页（服务端排序 + verdict 过滤）──
  async loadSummaryPage(page) {
    if (this.summaryLoading) return
    if (Number.isFinite(page) && page >= 1) this.summaryPage = page
    this.summaryLoading = true
    try {
      const offset = (this.summaryPage - 1) * this.summaryPageSize
      const params = new URLSearchParams({
        limit: String(this.summaryPageSize),
        offset: String(offset),
        sort: this.summarySortKey,
        dir: String(this.summarySortOrder),
        t: String(Date.now()),
      })
      if (this.summaryVerdictFilter) params.set('verdict', this.summaryVerdictFilter)
      const payload = await api('/api/alphaagent/research-memory?' + params.toString())
      this.summaryEntries = payload.entries || []
      this.summaryTotal = payload.total || this.summaryEntries.length
      this.researchMemoryStats = payload.statistics || null
      // 删除/过滤后页码越界：回退到最后一页重取一次
      const totalPages = Math.max(1, Math.ceil(this.summaryTotal / this.summaryPageSize))
      if (!this.summaryEntries.length && this.summaryPage > totalPages) {
        this.summaryLoading = false
        await this.loadSummaryPage(totalPages)
        return
      }
    } catch (e) {
      this.error = '读取研究总结失败: ' + e.message
    } finally {
      this.summaryLoading = false
    }
  },
  async setSummaryPage(page) {
    if (page < 1 || page > this.summaryTotalPages || page === this.summaryPage) return
    await this.loadSummaryPage(page)
  },
  async deleteMemoryEntry(entry) {
    try {
      await api('/api/alphaagent/research-memory/' + encodeURIComponent(entry.id), { method: 'DELETE' })
      this.researchMemory = this.researchMemory.filter(item => item.id !== entry.id)
      if (this.summaryEntries.some(item => item.id === entry.id)) {
        this.summaryTotal = Math.max(0, this.summaryTotal - 1)
        await this.loadSummaryPage()
      }
    } catch (e) {
      this.error = '删除记忆失败: ' + e.message
    }
  },
  async loadMemoryLayers() {
    if (this.memoryLayersLoading) return
    this.memoryLayersLoading = true
    try {
      const payload = await api('/api/alphaagent/research-memory/layers?t=' + Date.now())
      this.memoryLayers = { cells: payload.cells || [], experience: payload.experience || [] }
    } catch (e) {
      this.error = '读取研究记忆分层失败: ' + e.message
    } finally {
      this.memoryLayersLoading = false
    }
  },
  switchAgentMode(mode) {
    this.agentMode = mode
    if (mode === 'normal') this.loadResearchMemory(true)
  },
  setSummarySort(key) {
    const NUMERIC_SORT_KEYS = new Set([
      'ic', 'icir', 'coverage', 'annualized_excess_return', 'sharpe', 'excess_sharpe',
      'annualized_return', 'max_drawdown', 'annual_turnover', 'daily_overlap',
      'monotonicity', 'attempts',
    ])
    if (this.summarySortKey === key) {
      // 同列再点：升序 ↔ 降序
      this.summarySortOrder = this.summarySortOrder === 1 ? -1 : 1
    } else {
      this.summarySortKey = key
      // 数值列默认降序（最好在前），文本列默认升序
      this.summarySortOrder = (key === 'updated_at' || NUMERIC_SORT_KEYS.has(key)) ? -1 : 1
    }
    this.loadSummaryPage(1)
  },
  setSummaryVerdictFilter(verdict) {
    // 同一 verdict 再点 → 取消筛选
    this.summaryVerdictFilter = this.summaryVerdictFilter === verdict ? '' : verdict
    this.loadSummaryPage(1)
  },
  async loadResearchModes() {
    try {
      const r = await api('/api/alphaagent/research-modes?t=' + Date.now())
      this.researchModes = (r.modes || []).map(m => ({
        value: m.value,
        label: m.label,
        hint: m.hint || '',
        recommended_label_col: m.recommended_label_col,
        default_user_message: m.default_user_message,
        needs_fundamentals: m.needs_fundamentals,
      }))
      this.researchModeOptions = this.researchModes.map(m => ({ value: m.value, label: m.label }))
      // data 初始 research_mode 可能不在列表里 → 回退第一个
      if (!this.researchModes.some(m => m.value === this.form.research_mode) && this.researchModes.length) {
        this.form.research_mode = this.researchModes[0].value
      }
    } catch (e) {
      // 后端不可用时静默，页面仍可操作（按钮为空）
      this.researchModes = []
    }
  },
  async loadWindowsConfig() {
    // 统一时间窗口默认值来自后端配置中心（window_config.py 唯一真源），
    // 前端不再散落硬编码日期。仅当用户尚未手动改过对应字段时才覆盖，
    // 避免刷新页面把用户已填的值重置。
    try {
      const w = await getWindowDefaults()
      const set = (obj, key, val) => { if (val) obj[key] = val }
      if (w && w.test_start) {
        set(this.form, 'train_start', w.train_start)
        set(this.form, 'train_end', w.train_end)
        set(this.form, 'val_start', w.val_start)
        set(this.form, 'val_end', w.val_end)
        this.windowDefaults = w
      }
    } catch (e) {
      // 后端不可用时静默，保留前端兜底日期
    }
  },

  // ── 研究规范 / 门槛 ──
  async loadDefaultResearchSpec(mode) {
    const m = mode || this.form.research_mode || 'technical'
    try {
      const payload = await api('/api/alphaagent/research-specs/' + encodeURIComponent(m) + '?t=' + Date.now())
      const effective = payload?.effective
      const overrides = payload?.overrides || {}
      this.specDefaultsByMode[m] = effective
      this.specOverridesByMode[m] = overrides
      const text = JSON.stringify(effective, null, 2)
      this.defaultResearchSpecText = text
      if (!this.researchSpecText || this.form.research_mode === m) {
        this.researchSpecText = text
        this.specError = ''
      }
    } catch (e) {
      this.error = '读取默认 ResearchSpec 失败: ' + e.message
      throw e
    }
  },
  async switchResearchMode(mode) {
    if (this.agentBusy || this.form.research_mode === mode) return
    if (!this.researchModes.some(item => item.value === mode)) return
    const previousMode = this.form.research_mode
    const previousText = this.researchSpecText
    const previousDefault = this.defaultResearchSpecText
    this.form.research_mode = mode
    let payload = null
    try {
      payload = await api('/api/alphaagent/research-specs/' + encodeURIComponent(mode) + '?t=' + Date.now())
    } catch (e) {
      this.form.research_mode = previousMode
      this.error = '切换研究模式失败: ' + e.message
      return
    }
    const spec = payload.effective
    this.specDefaultsByMode[mode] = spec
    this.specOverridesByMode[mode] = payload.overrides || {}
    // 未手动编辑过规范时跟随模式整体切换；已编辑则保留自定义（后端会与该模式默认值深合并）。
    const untouched = !previousText || previousText === previousDefault
    const text = JSON.stringify(spec, null, 2)
    this.defaultResearchSpecText = text
    if (untouched) {
      this.researchSpecText = text
      this.specError = ''
    }
    if (spec?.recommended_label_col) this.form.label_col = spec.recommended_label_col
    // 门槛弹窗打开时同步切换 draft
    if (this.showThresholdModal) this.syncThresholdDraft()
  },
  parseResearchSpec() {
    try {
      const spec = JSON.parse(this.researchSpecText || this.defaultResearchSpecText || '{}')
      if (!spec || typeof spec !== 'object' || Array.isArray(spec)) throw new Error('必须是 JSON 对象')
      this.specError = ''
      return spec
    } catch (e) {
      this.specError = 'ResearchSpec JSON 无效: ' + e.message
      throw e
    }
  },
  async saveResearchSpec() {
    const mode = this.form.research_mode
    let spec
    try {
      spec = this.parseResearchSpec()
    } catch (e) {
      return
    }
    this.researchSpecSaving = true
    try {
      const payload = await api('/api/alphaagent/research-specs/' + encodeURIComponent(mode), {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ spec }),
      })
      // 服务端返回 normalize 后的生效 spec，编辑器与后续运行都以它为准
      this.specDefaultsByMode[mode] = payload.effective
      this.specOverridesByMode[mode] = payload.overrides || {}
      const text = JSON.stringify(payload.effective, null, 2)
      this.defaultResearchSpecText = text
      this.researchSpecText = text
      this.specError = ''
      this.specSavedAt = new Date()
    } catch (e) {
      this.specError = '保存门槛失败: ' + e.message
    } finally {
      this.researchSpecSaving = false
    }
  },
  async resetResearchSpec() {
    const mode = this.form.research_mode
    try {
      const payload = await api('/api/alphaagent/research-specs/' + encodeURIComponent(mode), { method: 'DELETE' })
      this.specDefaultsByMode[mode] = payload.effective
      this.specOverridesByMode[mode] = {}
      const text = JSON.stringify(payload.effective, null, 2)
      this.defaultResearchSpecText = text
      this.researchSpecText = text
      this.specError = ''
      this.specSavedAt = null
    } catch (e) {
      this.specError = '恢复默认失败: ' + e.message
    }
    if (this.showThresholdModal) this.syncThresholdDraft()
  },
  syncThresholdDraft() {
    const spec = this.specDefaultsByMode[this.form.research_mode]
    const draft = JSON.parse(JSON.stringify(spec || {}))
    // 未保存的 JSON 编辑（高级区）优先于默认 effective
    if (this.researchSpecText) {
      try {
        const edited = JSON.parse(this.researchSpecText)
        if (edited && typeof edited === 'object') {
          draft.evaluation_policy = Object.assign((spec?.evaluation_policy || {}), edited.evaluation_policy || {})
          draft.delivery_policy = Object.assign((spec?.delivery_policy || {}), edited.delivery_policy || {})
        }
      } catch (e) { /* 忽略无效 JSON，保留默认 */ }
    }
    // 防御：即使 spec 加载失败也保证嵌套结构存在，避免模板在 undefined 上抛错导致弹窗空白
    draft.evaluation_policy = draft.evaluation_policy || {}
    draft.delivery_policy = draft.delivery_policy || {}
    draft.delivery_policy.candidate = draft.delivery_policy.candidate || {}
    draft.delivery_policy.production = draft.delivery_policy.production || {}
    draft.delivery_policy.production.engine_gate = draft.delivery_policy.production.engine_gate || {}
    draft.delivery_policy.screener = draft.delivery_policy.screener || {}
    draft.delivery_policy.screener.enabled = draft.delivery_policy.screener.enabled ?? false
    draft.delivery_policy.screener.lookback = draft.delivery_policy.screener.lookback ?? 10
    draft.delivery_policy.screener.min_ic = draft.delivery_policy.screener.min_ic ?? 0.02
    draft.delivery_policy.screener.max_corr = draft.delivery_policy.screener.max_corr ?? 0.7
    draft.delivery_policy.screener.use_family_boost = draft.delivery_policy.screener.use_family_boost ?? true
    draft.delivery_policy.screener.adx_threshold = draft.delivery_policy.screener.adx_threshold ?? 25
    draft.delivery_policy.screener.ma_period = draft.delivery_policy.screener.ma_period ?? 60
    draft.delivery_policy.screener.min_cross_section = draft.delivery_policy.screener.min_cross_section ?? 30
    draft.search_policy = draft.search_policy || {}
    draft.search_policy.max_candidates_per_round = draft.search_policy.max_candidates_per_round ?? 8
    this.thresholdDraft = draft
  },
  async openThresholdModal() {
    // 若该模式 spec 未加载（如 mount 时后端未就绪），先补拉再开弹窗；
    // 失败时弹窗内显示错误而非空白表单。
    if (!this.specDefaultsByMode[this.form.research_mode]) {
      this.specLoading = true
      try {
        await this.loadDefaultResearchSpec(this.form.research_mode)
        this.syncThresholdDraft()
      } catch (e) {
        this.specError = '读取门槛数据失败: ' + (e?.message || e)
        this.syncThresholdDraft()
      } finally {
        this.specLoading = false
      }
    } else {
      this.syncThresholdDraft()
    }
    this.showThresholdModal = true
  },
  validateThresholdDraft() {
    const d = this.thresholdDraft
    if (!d || !d.evaluation_policy || !d.delivery_policy?.candidate || !d.delivery_policy?.production?.engine_gate) {
      throw new Error('门槛数据不完整')
    }
    const nums = [
      ['Train |IC|', d.evaluation_policy.min_train_abs_ic, 0, 1],
      ['Train |ICIR|', d.evaluation_policy.min_train_icir, -5, 20],
      ['Coverage', d.evaluation_policy.min_train_coverage, 0, 1],
      ['Val |IC|', d.evaluation_policy.min_val_abs_ic, 0, 1],
      ['Val 保留比', d.evaluation_policy.min_val_ic_retention_ratio, 0, 2],
      ['换手自相关', d.evaluation_policy.min_cs_autocorr, 0, 1],
      ['候选 |IC|', d.delivery_policy.candidate.min_abs_ic, 0, 1],
      ['候选 ICIR', d.delivery_policy.candidate.min_icir, -5, 20],
      ['候选 Coverage', d.delivery_policy.candidate.min_coverage, 0, 1],
      ['候选最大相关', d.delivery_policy.candidate.max_abs_corr, 0, 1],
      ['候选自相关', d.delivery_policy.candidate.min_cs_autocorr, 0, 1],
      ['候选 Val 保留比', d.delivery_policy.candidate.min_val_ic_retention, 0, 1],
      ['正式 Train |IC|', d.delivery_policy.production.min_train_abs_ic, 0, 1],
      ['正式 Train ICIR', d.delivery_policy.production.min_train_icir, -5, 20],
      ['正式 Val |IC|', d.delivery_policy.production.min_val_abs_ic, 0, 1],
      ['正式 Val 保留比', d.delivery_policy.production.min_val_ic_retention, 0, 2],
      ['正式多头超额', d.delivery_policy.production.min_val_long_excess, -1, 1],
      ['正式截尾衰减', d.delivery_policy.production.max_winsorized_abs_ic_decay, 0, 1],
      ['正式最大相关', d.delivery_policy.production.max_abs_corr, 0, 1],
      ['超额年化', d.delivery_policy.production.engine_gate.min_excess_annual, -1, 5],
      ['超额夏普', d.delivery_policy.production.engine_gate.min_excess_sharpe, 0, 10],
      ['最大回撤', d.delivery_policy.production.engine_gate.max_drawdown, 0, 10],
      ['选股百分比', d.delivery_policy.production.engine_gate.selection_pct, 0.0001, 0.1],
      ['选股数量', d.delivery_policy.production.engine_gate.top_n, 1, 500],
      ['门禁资金', d.delivery_policy.production.engine_gate.capital, 10000, 1e9],
      ['滑点', d.delivery_policy.production.engine_gate.slippage_bps, 0, 1000],
      ['参与率', d.delivery_policy.production.engine_gate.max_participation, 0.001, 1],
      ['持仓重叠', d.delivery_policy.production.engine_gate.min_daily_overlap, 0, 10],
      ['仓位利用率', d.delivery_policy.production.engine_gate.min_invested_ratio, 0, 1],
      ['日均成交额', d.delivery_policy.production.engine_gate.min_am20_yuan, 0, 1e12],
      ['Screener 回看天数', d.delivery_policy.screener?.lookback, 3, 60],
      ['Screener |IC| 下限', d.delivery_policy.screener?.min_ic, 0, 1],
      ['Screener 最大相关性', d.delivery_policy.screener?.max_corr, 0, 1],
      ['Screener ADX 阈值', d.delivery_policy.screener?.adx_threshold, 0, 100],
      ['Screener 均线周期', d.delivery_policy.screener?.ma_period, 10, 500],
      ['Screener 最小截面数', d.delivery_policy.screener?.min_cross_section, 1, 1000],
      ['每轮候选数', d.search_policy?.max_candidates_per_round, 1, 24],
    ]
    for (const [label, value, lo, hi] of nums) {
      if (value == null || Number.isNaN(Number(value))) throw new Error(label + ' 必须是数字')
      const v = Number(value)
      if (v < lo || v > hi) throw new Error(label + ' 超出范围 [' + lo + ', ' + hi + ']')
    }
    const freq = String(d.delivery_policy.production.engine_gate.freq || '').toLowerCase()
    if (!['weekly', 'monthly', 'daily'].includes(freq)) throw new Error('调仓频率必须是 weekly/monthly/daily')
    const selMode = String(d.delivery_policy.production.engine_gate.selection_mode || '').toLowerCase()
    if (!['top_pct', 'top_n'].includes(selMode)) throw new Error('选股模式必须是 top_pct/top_n')
    if (d.delivery_policy.production.engine_gate.enabled != null && typeof d.delivery_policy.production.engine_gate.enabled !== 'boolean') {
      throw new Error('启用门禁必须是布尔值')
    }
    return d
  },
  async saveThresholds() {
    let draft
    try {
      draft = this.validateThresholdDraft()
    } catch (e) {
      this.specError = String(e.message || e)
      return
    }
    const mode = this.form.research_mode
    // 高级 JSON 区已编辑过的键，以 JSON 为准（覆盖表单），未编辑的以表单为准
    let advanced = null
    if (this.researchSpecText) {
      try {
        advanced = JSON.parse(this.researchSpecText)
      } catch (e) { /* 高级 JSON 无效则忽略，只用表单 */ }
    }
    if (advanced && typeof advanced === 'object') {
      draft = JSON.parse(JSON.stringify(draft))
      for (const key of Object.keys(advanced)) {
        if (key !== 'research_mode') draft[key] = advanced[key]
      }
    }
    draft.research_mode = mode
    this.researchSpecSaving = true
    try {
      const payload = await api('/api/alphaagent/research-specs/' + encodeURIComponent(mode), {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ spec: draft }),
      })
      this.specDefaultsByMode[mode] = payload.effective
      this.specOverridesByMode[mode] = payload.overrides || {}
      const text = JSON.stringify(payload.effective, null, 2)
      this.defaultResearchSpecText = text
      this.researchSpecText = text
      this.specError = ''
      this.specSavedAt = new Date()
      this.syncThresholdDraft()
    } catch (e) {
      this.specError = '保存门槛失败: ' + e.message
    } finally {
      this.researchSpecSaving = false
    }
  },
})

// 研究总结表格视图：排序/verdict 过滤已在服务端完成（分页口径），
// 这里只暴露当前页条目与全量 verdict 计数（来自后端 statistics）。
export const summaryView = {
  counts: computed(() => {
    // 优先用后端 statistics 的 verdict_counts（全量统计），分页时也准确
    const stats = agentStore.researchMemoryStats
    if (stats && stats.verdict_counts) {
      return { ...stats.verdict_counts }
    }
    // 兜底：前端本地统计已加载条目
    const counts = {}
    for (const entry of agentStore.summaryEntries) {
      const v = entry.verdict || 'unknown'
      counts[v] = (counts[v] || 0) + 1
    }
    return counts
  }),
  filtered: computed(() => agentStore.summaryEntries),
}
