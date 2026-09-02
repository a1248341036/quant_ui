<template>
  <main class="agent-main">
    <header class="agent-header">
      <div class="agent-title">
        <span class="agent-orb">✦</span>
        <div>
          <h1>{{ agent.current ? runTitle(agent.current) : 'AlphaAgent 研究助手' }}</h1>
          <p v-if="agent.current" class="agent-subtitle">
            {{ statusLabel(agent.current.status) }} · {{ agent.events.length }} 条事件
          </p>
          <p v-else class="agent-subtitle">让 Agent 自主提出、评估和筛选因子</p>
        </div>
      </div>
      <div class="header-actions">
        <div class="mode-toggle">
          <button :class="{active: agent.agentMode==='research'}" @click="agent.switchAgentMode('research')">研究</button>
          <button :class="{active: agent.agentMode==='normal'}" @click="agent.switchAgentMode('normal')">普通</button>
        </div>
        <span v-if="agent.agentBusy" class="activity-line"><i></i>{{ agent.currentActivity }}</span>
        <span v-if="agent.usage.calls" class="usage-chip" title="本次 Agent 模型调用累计 usage">
          ↑ {{ formatTokens(agent.usage.input_tokens) }} · ↓ {{ formatTokens(agent.usage.output_tokens) }}
          <em v-if="agent.usage.cache_input_tokens">缓存 {{ formatTokens(agent.usage.cache_input_tokens) }}</em>
          <em v-else>缓存未返回</em>
        </span>
        <span v-if="agent.current" class="live-status" :class="'live-' + agent.current.status">
          <i></i>{{ statusLabel(agent.current.status) }}
        </span>
        <button v-if="agent.canStopAgent" class="stop-btn" @click="agent.stopAgent">停止</button>
      </div>
    </header>

    <div ref="thread" class="agent-thread" @scroll="onThreadScroll">
      <div v-if="agent.agentMode==='normal' && !agent.events.length" class="normal-mode-panel">
        <h3>研究记忆（{{ agent.researchMemoryTotal }} 个）</h3>
        <p class="normal-mode-desc">查看和清理历史评估记忆。删除后 Agent 不再参考对应条目。</p>
        <div v-if="!agent.researchMemory.length && !agent.researchMemoryLoading" class="normal-mode-empty">暂无研究记忆</div>
        <div v-else-if="!agent.researchMemory.length" class="normal-mode-empty">加载中…</div>
        <div v-for="entry in agent.researchMemory" :key="entry.id" class="memory-entry-row">
          <div class="memory-entry-main" @click="agent.memoryDetail = entry">
            <strong>{{ entry.factor_name || 'unnamed' }}</strong>
            <span class="memory-verdict-tag" :class="'memv-' + entry.verdict">{{ memoryVerdictLabel(entry.verdict) }}</span>
            <small>{{ formatTime(entry.updated_at) }}</small>
          </div>
          <button class="memory-del-btn" title="删除此条目" @click.stop="agent.deleteMemoryEntry(entry)">×</button>
        </div>
        <div v-if="agent.researchMemoryHasMore" class="normal-mode-load-more">
          <button class="load-more-btn" :disabled="agent.researchMemoryLoading" @click="agent.loadMoreResearchMemory()">
            {{ agent.researchMemoryLoading ? '加载中…' : '加载更多（剩余 ' + (agent.researchMemoryTotal - agent.researchMemory.length) + ' 条）' }}
          </button>
        </div>
      </div>

      <div v-else-if="!agent.events.length" class="welcome">
        <div class="welcome-orb">✦</div>
        <h2>开始一次因子研究</h2>
        <p>描述你想研究的方向，或者让 Agent 自己探索。它会实时展示思考摘要、工具调用和评估结果。</p>
        <div class="suggestions">
          <button @click="agent.useSuggestion('从价量、动量和反转方向自主挖掘候选因子，并在训练集和验证集上检验。')">价量与动量</button>
          <button @click="agent.useSuggestion('重点研究成交量异常、流动性和波动率收缩，要求控制极端值和覆盖率。')">成交量与波动率</button>
          <button @click="agent.useSuggestion('检查已有因子库中的研究空白，提出不重复的 A 股日频因子。')">寻找研究空白</button>
        </div>
      </div>

      <div v-for="(message, index) in timeline" :key="message.key + '-' + index" class="message-row" :class="'message-' + message.kind">
        <button v-if="message.text" class="msg-copy-btn" title="复制内容" @click="copyMessage(message.text)">⧉</button>
        <div v-if="message.kind === 'user'" class="user-bubble">{{ message.text }}</div>

        <div v-else-if="message.kind === 'assistant'" class="assistant-message">
          <div class="avatar agent-avatar">✦</div>
          <div class="message-body">
            <div class="message-author">AlphaAgent</div>
            <div class="message-text">{{ message.text }}</div>
          </div>
        </div>

        <details v-else-if="message.kind === 'thinking'" class="thinking-card" open>
          <summary><span class="thinking-icon">◌</span> {{ message.label }}</summary>
          <pre>{{ message.text }}</pre>
        </details>

        <details v-else-if="message.kind === 'tool_call'" class="tool-card">
          <summary>
            <span class="tool-icon">⚙</span>
            <span>{{ toolLabel(message.name) }}<b v-if="message.factorName"> · {{ message.factorName }}</b></span>
            <span class="tool-state">调用中 / 已提交</span>
          </summary>
          <div v-if="message.expression" class="tool-expression">{{ message.expression }}</div>
          <pre>{{ message.text }}</pre>
        </details>

        <div v-else-if="message.kind === 'tool_result'" class="result-card">
          <div class="result-head">
            <span class="result-icon" :class="{ bad: !message.ok }">{{ message.ok ? '✓' : '×' }}</span>
            <strong>{{ toolLabel(message.name) }}</strong>
            <strong v-if="message.factorName" class="result-factor">{{ message.factorName }}</strong>
            <span class="result-time" v-if="message.elapsed">{{ message.elapsed }}s</span>
          </div>
          <div v-if="message.expression" class="result-expression">{{ message.expression }}</div>
          <div class="result-metrics" v-if="message.metrics.length">
            <span v-for="metric in message.metrics" :key="metric">{{ metric }}</span>
          </div>
          <!-- 完整引擎回测结果 -->
          <details v-if="message.backtest" class="result-backtest">
            <summary>完整引擎回测 · {{ message.backtest.freq }} · TopN {{ message.backtest.top_n }} · 方向 {{ message.backtest.direction === 1 ? '多' : '空' }}</summary>
            <div class="backtest-metrics">
              <span :class="{neg: message.backtest.annual < 0}">年化 {{ pct(message.backtest.annual) }}</span>
              <span :class="{neg: message.backtest.excess < 0}">超额 {{ pct(message.backtest.excess) }}</span>
              <span>夏普 {{ fmt(message.backtest.sharpe) }}</span>
              <span :class="{neg: message.backtest.drawdown < -0.2}">回撤 {{ pct(message.backtest.drawdown) }}</span>
              <span>重合率 {{ fmt(message.backtest.overlap) }}</span>
              <span>胜率 {{ pct(message.backtest.winRate) }}</span>
            </div>
            <div v-if="message.backtest.byFreq" class="backtest-freq-grid">
              <div v-for="(f, freq) in message.backtest.byFreq" :key="freq" class="backtest-freq-item">
                <strong>{{ freq }}</strong>
                <span :class="{neg: f.annualized_return < 0}">年化 {{ pct(f.annualized_return) }}</span>
                <span :class="{neg: f.annualized_excess_return < 0}">超额 {{ pct(f.annualized_excess_return) }}</span>
                <span>夏普 {{ fmt(f.sharpe) }}</span>
                <span>重合 {{ fmt(f.daily_overlap) }}</span>
              </div>
            </div>
            <div v-if="message.backtest.gateReasons?.length" class="backtest-gate">
              <span class="backtest-gate-label">门禁未通过：</span>
              <span v-for="r in message.backtest.gateReasons" :key="r" class="backtest-gate-reason">{{ r }}</span>
            </div>
          </details>
          <div v-if="message.text" class="result-note">{{ message.text }}</div>
        </div>

        <details v-else-if="message.kind === 'reviewer_thinking'" class="thinking-card reviewer-card" open>
          <summary><span class="thinking-icon">◌</span> FactorReviewer 审查中</summary>
          <pre>{{ message.text }}</pre>
        </details>

        <div v-else-if="message.kind === 'review'" class="review-card" :class="'review-' + message.verdict">
          <div class="review-head">
            <strong>FactorReviewer</strong>
            <span>{{ reviewLabel(message.verdict) }}</span>
            <em>{{ message.novelty }} 新颖性</em>
          </div>
          <div class="review-canonical">{{ message.canonical }}</div>
          <ul v-if="message.reasons.length">
            <li v-for="reason in message.reasons" :key="reason">{{ reason }}</li>
          </ul>
          <div v-if="message.changes.length" class="review-changes">下一步：{{ message.changes.join('；') }}</div>
        </div>

        <div v-else class="system-message">
          <span>{{ message.text }}</span>
        </div>
      </div>

      <div v-if="agent.agentBusy" class="typing-row">
        <div class="avatar agent-avatar">✦</div>
        <div class="typing"><i></i><i></i><i></i></div>
        <span>{{ agent.liveActivity }}</span>
      </div>
    </div>

    <div class="composer-wrap" :class="{ collapsed: agent.composerCollapsed }">
      <template v-if="!agent.composerCollapsed">
        <div v-if="agent.error" class="composer-error">{{ agent.error }}</div>
        <div v-if="agent.agentBusy" class="activity-bar"><i></i><span>{{ agent.currentActivity }}</span></div>
        <div class="composer">
          <textarea
            v-model="agent.form.user_message"
            :disabled="agent.current?.status === 'stopping'"
            rows="3"
            :placeholder="composerPlaceholder"
            @keydown.ctrl.enter.prevent="agent.sendMessage"
            @keydown.meta.enter.prevent="agent.sendMessage"
          ></textarea>
          <div class="composer-bottom">
            <div v-if="agent.agentMode==='research'" class="composer-options">
              <div class="mode-switch" role="tablist" aria-label="评估档位">
                <span class="mode-auto-badge" title="档位由数据面自动推断：勾选基本面/股东面 → 慢信号档（label_20d+松门槛+月调仓）；否则短周期档（label_1d+严门槛+周调仓）">
                  {{ agent.inferredModeLabel }}
                </span>
                <button class="threshold-btn" :class="{ active: agent.showThresholdModal }" :disabled="agent.agentBusy" title="编辑当前档位的挖掘/入库/回测门槛（保存后全链路生效）" @click="agent.openThresholdModal">{{ agent.researchSpecCustom ? '门槛·已改' : '门槛' }}</button>
              </div>
              <span class="composer-label-hint" :title="'本次评估使用的 label 列，随数据面组合自动切换'">{{ agent.form.label_col }}</span>
              <label class="composer-date" title="交付门禁的调仓频率：影响 engine_gate 回测与实盘可交易性判定；默认随档位（短周期=weekly，慢信号=monthly）">调仓 <select v-model="agent.form.rebalance_freq" :disabled="agent.agentBusy" class="composer-date-input">
                <option value="">自动</option>
                <option value="daily">日</option>
                <option value="weekly">周</option>
                <option value="monthly">月</option>
              </select></label>
              <label class="composer-date">训练 <input v-model="agent.form.train_start" type="date" :disabled="agent.agentBusy" class="composer-date-input"> → <input v-model="agent.form.train_end" type="date" :disabled="agent.agentBusy" class="composer-date-input"></label>
              <label class="composer-date">验证 <input v-model="agent.form.val_start" type="date" :disabled="agent.agentBusy" class="composer-date-input"> → <input v-model="agent.form.val_end" type="date" :disabled="agent.agentBusy" class="composer-date-input"></label>
              <label class="composer-date" title="每轮并行评估数（train/val 同时算多少个因子）；上限受机器 CPU/内存约束。每轮候选总数在「门槛」弹窗的搜索策略里配置">并发 <select v-model.number="agent.form.max_parallel_eval" :disabled="agent.agentBusy" class="composer-date-input">
                <option :value="6">×6</option>
                <option :value="12">×12</option>
                <option :value="16">×16</option>
                <option :value="24">×24</option>
              </select></label>
              <label class="composer-date" title="种群批量筛选：每轮一次参数网格扫描（propose_population）；关闭则仅单点迭代">种群 <select v-model.number="agent.form.population_max" :disabled="agent.agentBusy" class="composer-date-input">
                <option :value="0">关</option>
                <option :value="12">×12</option>
                <option :value="24">×24</option>
                <option :value="36">×36</option>
              </select></label>
            </div>
            <div v-if="agent.agentMode==='research'" class="facet-row" title="数据面聚焦：多选后 Agent 优先探索所选面的因子与跨面融合（选中非价量面会自动载入对应列族）">
              <span class="facet-label">数据面</span>
              <button
                v-for="f in agent.focusFacetOptions"
                :key="f.key"
                class="facet-chip"
                :class="{ active: (agent.form.focus_facets || []).includes(f.key) }"
                :disabled="agent.agentBusy"
                :title="f.hint"
                @click="agent.toggleFocusFacet(f.key)"
              >{{ f.label }}</button>
              <span v-if="!(agent.form.focus_facets || []).length" class="facet-hint">未选 = 不限（Agent 自主探索）</span>
              <span v-else-if="(agent.form.focus_facets || []).length >= 2" class="facet-hint fusion">融合 {{ (agent.form.focus_facets || []).length }} 面</span>
            </div>
            <div class="composer-actions">
              <button v-if="agent.agentMode==='research'" class="spec-toggle" :class="{ active: agent.showResearchSpec }" :disabled="agent.agentBusy" @click="agent.showResearchSpec = !agent.showResearchSpec">研究规范</button>
              <button v-if="agent.agentMode==='research' && !agent.current" class="spec-toggle quick-start-btn" :disabled="agent.agentBusy" @click="agent.startDefaultResearch" title="使用默认研究规范和提示词立即启动">▶ 默认研究</button>
              <button class="composer-collapse-btn" title="收起对话框" @click="agent.composerCollapsed = true">▾</button>
              <button class="send-btn" :disabled="agent.current?.status === 'stopping' || !agent.form.user_message.trim()" @click="agent.sendMessage" :title="composerActionTitle">
                <span>{{ agent.agentBusy || agent.current ? '↵' : '↑' }}</span>
              </button>
            </div>
          </div>
        </div>
        <div v-if="agent.showResearchSpec" class="research-spec-editor">
          <div class="research-spec-head">
            <strong>ResearchSpec · {{ agent.inferredModeLabel }}</strong>
            <div>
              <span v-if="agent.researchSpecCustom" class="research-spec-custom" title="该模式门槛文件已自定义保存，运行/晋升全链路生效">已自定义</span>
              <span v-if="agent.researchSpecDirty" class="research-spec-dirty" title="当前 JSON 与已保存门槛不一致">未保存</span>
              <span v-if="agent.specError" class="research-spec-error">{{ agent.specError }}</span>
              <span v-if="agent.specSavedAt" class="research-spec-saved" title="已写入该模式门槛文件">已保存 {{ formatTime(agent.specSavedAt) }}</span>
              <button title="保存当前 JSON 为该模式的门槛文件（全链路生效）" :disabled="agent.agentBusy || agent.researchSpecSaving" @click="agent.saveResearchSpec">
                {{ agent.researchSpecSaving ? '保存中…' : '保存门槛' }}
              </button>
              <button title="删除该模式门槛文件并恢复注册表默认" :disabled="agent.agentBusy" @click="agent.resetResearchSpec">恢复默认</button>
              <button class="research-spec-close" title="关闭研究规范" aria-label="关闭研究规范" @click="agent.showResearchSpec = false">×</button>
            </div>
          </div>
          <textarea v-model="agent.researchSpecText" :disabled="agent.agentBusy" spellcheck="false" aria-label="ResearchSpec JSON"></textarea>
        </div>
        <p class="composer-hint">AgentScope 实时事件 · 训练集评估 · 验证集检验 · FactorZoo 提交</p>
      </template>
      <button v-else class="composer-expand-bar" title="展开对话框" @click="agent.composerCollapsed = false">
        <span>输入消息…</span>
        <span class="composer-expand-arrow">▴</span>
      </button>
    </div>
  </main>
</template>

<script>
import { agentStore } from '../../store/alphaagent.js'
import { fmt, pct } from '../../utils/format.js'
import {
  runTitle, statusLabel, formatTokens, formatTime,
  toolLabel, reviewLabel, memoryVerdictLabel,
} from '../../utils/alphaagent.js'

export default {
  name: 'AgentThread',
  data() {
    return {
      agent: agentStore,
    }
  },
  computed: {
    timeline() {
      return agentStore.timeline
    },
    composerPlaceholder() {
      if (agentStore.agentBusy) return '向当前 Agent 追加研究指令…（Ctrl/⌘ + Enter）'
      if (agentStore.current) return '从当前历史会话继续研究…（Ctrl/⌘ + Enter）'
      return '告诉 AlphaAgent 你想研究什么…（Ctrl/⌘ + Enter 发送）'
    },
    composerActionTitle() {
      if (agentStore.agentBusy) return '追加到当前研究会话'
      if (agentStore.current) return '从当前历史会话继续'
      return '启动研究'
    },
  },
  watch: {
    // store 发出滚动请求后执行实际滚动（store 不持有 DOM）
    'agent.scrollTick'() {
      this.$nextTick(() => this.scrollThread(agentStore.scrollForce))
    },
  },
  methods: {
    fmt,
    pct,
    runTitle,
    statusLabel,
    formatTokens,
    formatTime,
    toolLabel,
    reviewLabel,
    memoryVerdictLabel,
    onThreadScroll() {
      const el = this.$refs.thread
      if (!el) return
      agentStore.stickToBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 100
    },
    scrollThread(force = false) {
      const el = this.$refs.thread
      if (!el || (!force && !agentStore.stickToBottom)) return
      el.scrollTop = el.scrollHeight
    },
    async copyMessage(text) {
      try {
        await navigator.clipboard.writeText(String(text || ''))
      } catch (e) {
        const ta = document.createElement('textarea')
        ta.value = String(text || '')
        ta.style.cssText = 'position:fixed;left:-9999px'
        document.body.appendChild(ta)
        ta.select()
        document.execCommand('copy')
        document.body.removeChild(ta)
      }
    },
  },
}
</script>
