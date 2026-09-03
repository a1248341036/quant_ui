<template>
  <section class="agent-page">
    <div class="agent-subtabs">
      <button :class="{active: subtab==='research'}" @click="subtab='research'">研究</button>
      <button :class="{active: subtab==='lab'}" @click="subtab='lab'">因子实验室</button>
      <button :class="{active: subtab==='library'}" @click="subtab='library'">因子库</button>
      <button :class="{active: subtab==='ml'}" @click="subtab='ml'">ML 组合</button>
      <button :class="{active: subtab==='summary'}" @click="subtab='summary'">研究总结</button>
      <button :class="{active: subtab==='memory'}" @click="subtab='memory'">研究记忆库</button>
    </div>

    <!-- ═══ 因子实验室 ═══ -->
    <factor-lab v-if="subtab==='lab'" @open-backtest="openLabBacktest" />

    <!-- ═══ ML 组合 ═══ -->
    <ml-panel v-else-if="subtab==='ml'" />

    <!-- ═══ 研究总结 ═══ -->
    <research-summary v-else-if="subtab==='summary'" />

    <!-- ═══ 研究记忆库 ═══ -->
    <research-memory-bank v-else-if="subtab==='memory'" />

    <!-- ═══ 因子库管理 ═══ -->
    <factor-library v-else-if="subtab==='library'" @backtest="openLibraryBacktest" />

    <!-- ═══ 研究面板 ═══ -->
    <div v-show="subtab==='research'" class="agent-shell">
      <agent-sidebar />
      <agent-thread />
    </div>

    <!-- 门槛配置弹窗 -->
    <threshold-modal />

    <!-- 长期记忆详情弹窗 -->
    <memory-detail-modal />

    <!-- 因子回测弹窗（因子实验室 / 因子库共用） -->
    <factor-backtest-dialog
      v-if="btOpen"
      :expr="btTarget.expr"
      :factor-name="btTarget.factorName"
      :defaults="btDefaults"
      @close="btOpen = false"
    />
  </section>
</template>

<script>
import '../styles/alphaagent.css'
import { agentStore } from '../store/alphaagent.js'
import { store } from '../store/index.js'
import AgentSidebar from '../components/alphaagent/AgentSidebar.vue'
import AgentThread from '../components/alphaagent/AgentThread.vue'
import ThresholdModal from '../components/alphaagent/ThresholdModal.vue'
import MemoryDetailModal from '../components/alphaagent/MemoryDetailModal.vue'
import FactorLab from '../components/alphaagent/FactorLab.vue'
import FactorLibrary from '../components/alphaagent/FactorLibrary.vue'
import FactorBacktestDialog from '../components/alphaagent/FactorBacktestDialog.vue'
import MlPanel from '../components/alphaagent/MlPanel.vue'
import ResearchSummary from '../components/alphaagent/ResearchSummary.vue'
import ResearchMemoryBank from '../components/alphaagent/ResearchMemoryBank.vue'

export default {
  name: 'AlphaAgent',
  components: {
    'agent-sidebar': AgentSidebar,
    'agent-thread': AgentThread,
    'threshold-modal': ThresholdModal,
    'memory-detail-modal': MemoryDetailModal,
    'factor-lab': FactorLab,
    'factor-library': FactorLibrary,
    'factor-backtest-dialog': FactorBacktestDialog,
    'ml-panel': MlPanel,
    'research-summary': ResearchSummary,
    'research-memory-bank': ResearchMemoryBank,
  },
  data() {
    return {
      subtab: 'research',
      btOpen: false,
      btTarget: { expr: '', factorName: 'expr' },
      btDefaults: null,
      pollTimer: null,
      pollBusy: false,
    }
  },
  mounted() {
    agentStore.init()
    store.loadLabHistory()
    // 从"历史"tab 打开一条因子评估 → 切到因子实验室（FactorLab 挂载后消费载荷）
    if (store.labLoadPayload) this.subtab = 'lab'
    // 页面切换会卸载本组件并关闭 SSE；store 单例保留了当前会话，
    // 回到本页时若 run 仍在运行则重连事件流，避免时间线冻结
    if (agentStore.current && ['starting', 'running'].includes(agentStore.current.status)) {
      agentStore.running = true
      agentStore.connectAgentEvents(agentStore.current.run_id)
    }
    this.startPolling()
    document.addEventListener('visibilitychange', this.onVisibilityChange)
  },
  beforeUnmount() {
    this.stopPolling()
    document.removeEventListener('visibilitychange', this.onVisibilityChange)
    agentStore.dispose()
  },
  methods: {
    // ── 页面内动态刷新：侧栏状态点 + 当前子标签数据，免 F5 ──
    startPolling() {
      this.stopPolling()
      this.pollTimer = setInterval(() => { this.pollOnce() }, 10000)
    },
    stopPolling() {
      if (this.pollTimer) {
        clearInterval(this.pollTimer)
        this.pollTimer = null
      }
    },
    onVisibilityChange() {
      if (document.visibilityState === 'visible') this.pollOnce()
    },
    async pollOnce() {
      if (this.pollBusy || document.visibilityState !== 'visible') return
      this.pollBusy = true
      try {
        const s = agentStore
        const jobs = [s.loadAgentRuns()]
        // 当前子标签的数据跟着刷（研究 tab 的时间线由 SSE 实时推送，不靠轮询）
        if (this.subtab === 'summary') jobs.push(s.loadSummaryPage())
        else if (this.subtab === 'memory') jobs.push(s.refreshResearchMemory())
        await Promise.all(jobs)
        // 当前会话的状态与最新列表对齐（SSE 断开时状态点也能跟上）
        if (s.current && s.current.status !== 'stopping') {
          const fresh = (s.runs || []).find(r => r.run_id === s.current.run_id)
          if (fresh && fresh.status && fresh.status !== s.current.status) {
            s.current.status = fresh.status
          }
        }
      } catch (e) {
        // 轮询失败静默，不打扰用户
      } finally {
        this.pollBusy = false
      }
    },
    // 因子实验室"回测"按钮：默认窗口跟随实验室的验证区间
    openLabBacktest(payload) {
      this.btTarget = { expr: payload.expr, factorName: payload.factorName }
      this.btDefaults = { start: payload.valStart, end: payload.valEnd }
      this.btOpen = true
    },
    // 因子库行内"回测"：载入该因子表达式，窗口走全局默认
    openLibraryBacktest(factor) {
      this.btTarget = { expr: factor.expr || '', factorName: factor.name || factor.factor_id }
      this.btDefaults = null
      this.btOpen = true
    },
  },
}
</script>
