<template>
  <section class="agent-page">
    <div class="agent-subtabs">
      <button :class="{active: subtab==='research'}" @click="subtab='research'">研究</button>
      <button :class="{active: subtab==='lab'}" @click="subtab='lab'">因子实验室</button>
      <button :class="{active: subtab==='library'}" @click="subtab='library'; loadFactors()">因子库</button>
    </div>

    <!-- ═══ 因子实验室 ═══ -->
    <div v-if="subtab==='lab'" class="lab-panel">
      <div class="lab-left">
        <h3>因子表达式</h3>
        <textarea v-model="lab.expr" class="lab-editor" rows="14" spellcheck="false" placeholder="输入 DSL 因子表达式…"></textarea>
        <div class="lab-options">
          <label>因子名称 <input v-model="lab.factorName" type="text" placeholder="expr"></label>
          <div class="lab-date-row">
            <label>训练 <input v-model="lab.trainStart" type="date"> → <input v-model="lab.trainEnd" type="date"></label>
            <label>验证 <input v-model="lab.valStart" type="date"> → <input v-model="lab.valEnd" type="date"></label>
          </div>
          <label class="lab-funda">
            <input type="checkbox" v-model="lab.includeFundamentals"> 包含基本面列
          </label>
        </div>
        <button class="lab-run-btn" :disabled="lab.busy || !lab.expr.trim()" @click="runEval">
          {{ lab.busy ? '评估中…' : '评估因子' }}
        </button>
        <p v-if="lab.error" class="lab-error">{{ lab.error }}</p>
      </div>
      <div class="lab-right">
        <div v-if="!lab.results && !lab.busy" class="lab-empty">
          输入 DSL 表达式后点击评估，将同时在 train_screen / validation / size_neutral_validation 三个 profile 上评估。
        </div>
        <div v-if="lab.busy" class="lab-loading">正在加载 panel 并评估（首次约 30-60 秒）…</div>
        <div v-if="lab.results" class="lab-results">
          <div v-for="(result, profileId) in lab.results" :key="profileId" class="lab-result-card" :class="{ok: result.ok, bad: !result.ok}">
            <div class="lab-result-head">
              <strong>{{ profileId }}</strong>
              <span v-if="result.ok" class="lab-pass" :class="{pass: result.passed, fail: !result.passed}">
                {{ result.passed ? '通过' : '未通过' }}
              </span>
              <span v-if="result.date_range" class="lab-daterange">{{ result.date_range.start }} → {{ result.date_range.end }}</span>
            </div>
            <div v-if="!result.ok" class="lab-result-error">{{ result.error }}</div>
            <div v-if="result.ok && result.metrics" class="lab-metrics">
              <template v-for="(m, mid) in result.metrics" :key="mid">
                <div v-if="typeof m === 'number' || m == null" class="lab-metric">
                  <span class="lab-metric-label">{{ metricLabel(mid) }}</span>
                  <span class="lab-metric-value">{{ formatMetricValue(m) }}</span>
                </div>
              </template>
            </div>
            <details v-if="result.ok && result.rule_results?.length" class="lab-rules">
              <summary>规则详情 ({{ result.rule_results.filter(r => r.passed).length }}/{{ result.rule_results.length }})</summary>
              <div v-for="(rule, i) in result.rule_results" :key="i" class="lab-rule" :class="{pass: rule.passed, fail: !rule.passed}">
                <span>{{ rule.metric }} {{ rule.op }} {{ rule.expected }}</span>
                <span>{{ rule.actual != null ? formatMetricValue(rule.actual) : '—' }}</span>
                <span>{{ rule.passed ? '✓' : '✗' }}</span>
              </div>
            </details>
          </div>
        </div>
      </div>
    </div>

    <!-- ═══ 因子库管理 ═══ -->
    <div v-if="subtab==='library'" class="lib-panel">
      <div class="lib-head">
        <div class="lib-tabs">
          <button :class="{active: lib.library==='production'}" @click="switchLibrary('production')">正式因子库</button>
          <button :class="{active: lib.library==='candidate'}" @click="switchLibrary('candidate')">候选因子库</button>
        </div>
        <span class="lib-count" v-if="lib.data">{{ lib.data.n_factors || 0 }} 个因子</span>
        <button class="lib-refresh" @click="loadFactors">刷新</button>
      </div>
      <div v-if="lib.error" class="lib-error">{{ lib.error }}</div>
      <div v-if="lib.loading" class="lib-loading">加载中…</div>
      <div v-if="lib.data && !lib.data.factors?.length && !lib.loading" class="lib-empty">
        {{ lib.data.error === 'library_not_initialized' ? '因子库尚未初始化，请在挖掘流程中启用因子提交。' : '暂无因子。' }}
      </div>
      <table v-if="lib.data?.factors?.length" class="lib-table">
        <thead>
          <tr>
            <th>ID</th><th>名称</th><th>IC</th><th>ICIR</th><th>RankIC</th><th>覆盖率</th><th>状态</th><th>创建时间</th><th></th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="f in lib.data.factors" :key="f.factor_id">
            <td class="lib-fid" @click="showFactorDetail(f)">{{ f.factor_id }}</td>
            <td @click="showFactorDetail(f)">{{ f.name }}</td>
            <td :class="icClass(f.metrics?.ic)">{{ formatMetricValue(f.metrics?.ic) }}</td>
            <td :class="icClass(f.metrics?.icir)">{{ formatMetricValue(f.metrics?.icir) }}</td>
            <td>{{ formatMetricValue(f.metrics?.rank_ic) }}</td>
            <td>{{ formatMetricValue(f.metrics?.factor_coverage) }}</td>
            <td><span class="lib-status" :class="'status-' + f.status">{{ f.status }}</span></td>
            <td class="lib-time">{{ formatTime(f.created_at) }}</td>
            <td><button class="lib-del" @click="confirmDelete(f)">删除</button></td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- ═══ 研究面板（原有） ═══ -->
    <div v-show="subtab==='research'" class="agent-shell">
      <aside class="agent-sidebar">
        <div class="sidebar-head">
          <div>
            <div class="eyebrow">RESEARCH AGENT</div>
            <h2>AlphaAgent</h2>
          </div>
          <button class="icon-btn" title="新建会话" @click="newRun">+</button>
        </div>

        <button class="new-run" @click="newRun">
          <span>＋</span> 新建研究任务
        </button>

        <div class="session-label">
          <span>{{ agent.showArchived ? '已归档任务' : '最近任务' }}</span>
          <button class="archived-toggle" :title="agent.showArchived ? '返回最近任务' : '查看已归档任务'" @click="toggleArchived">
            {{ agent.showArchived ? '返回' : '归档' }}
          </button>
        </div>
        <div class="session-list">
          <div
            v-for="run in agent.runs"
            :key="run.run_id"
            class="session-item"
            :class="{ active: agent.current?.run_id === run.run_id }"
          >
            <div class="session-select" role="button" tabindex="0" @click="selectAgentRun(run)" @keydown.enter.prevent="selectAgentRun(run)" @keydown.space.prevent="selectAgentRun(run)">
              <span class="status-dot" :class="'status-' + run.status"></span>
              <span class="session-copy" :data-run-title="run.run_id">
                <strong><span v-if="run.pinned" class="session-pin" title="已置顶">PIN</span>{{ runTitle(run) }}</strong>
                <small>{{ formatTime(run.created_at) }}</small>
              </span>
              <span class="session-count">{{ run.event_count || 0 }}</span>
            </div>
            <button class="session-menu" title="会话操作" aria-label="会话操作" @click.stop="toggleMenu(run.run_id, $event)">...</button>
          </div>
          <div v-if="!agent.runs.length" class="sidebar-empty">{{ agent.showArchived ? '还没有归档任务' : '还没有研究任务' }}</div>
        </div>

        <section class="memory-panel" aria-label="长期研究记忆">
          <div class="memory-head"><span>长期研究记忆</span><b>{{ agent.memory.length }}</b></div>
          <div v-if="agent.memory.length" class="memory-items">
            <div v-for="entry in agent.memory" :key="entry.id" class="memory-item" :class="'memory-' + entry.verdict" role="button" tabindex="0" @click="showMemoryDetail(entry)" @keydown.enter.prevent="showMemoryDetail(entry)">
              <strong>{{ memoryVerdictLabel(entry.verdict) }} · {{ entry.factor_name }}</strong>
              <small>{{ entry.conclusion }}</small>
            </div>
          </div>
          <small v-else>评估结果会在这里沉淀为跨会话研究记忆</small>
        </section>

        <div class="sidebar-footer">
          <span class="status-dot status-running"></span>
          <span>Codex 当前模型配置</span>
        </div>
      </aside>

      <Teleport to="body">
        <div
          v-if="menuRun"
          class="session-menu-popover"
          :style="{ top: agent.menuPosition.top + 'px', left: agent.menuPosition.left + 'px' }"
          @click.stop
        >
          <button @click="pinRun(menuRun)">{{ menuRun.pinned ? '取消置顶' : '置顶' }}</button>
          <button @click="branchRun(menuRun)">新建分支</button>
          <button @click="beginRename(menuRun)">重命名</button>
          <button class="archive-action" @click="archiveRun(menuRun)">归档</button>
        </div>
      </Teleport>

      <Teleport to="body">
        <form
          v-if="renameRun"
          class="session-rename-popover"
          :style="{ top: agent.renamePosition.top + 'px', left: agent.renamePosition.left + 'px' }"
          @click.stop
          @submit.prevent="commitRename(renameRun)"
        >
          <input
            ref="renameInput"
            v-model="agent.renameTitle"
            aria-label="会话名称"
            @keydown.esc.stop.prevent="cancelRename"
          >
          <button type="submit" title="保存会话名称" aria-label="保存会话名称">✓</button>
        </form>
      </Teleport>

      <Teleport to="body">
        <div v-if="agent.memoryDetail" class="memory-modal-overlay" @click="agent.memoryDetail = null">
          <div class="memory-modal" @click.stop>
            <div class="memory-modal-head">
              <strong>{{ agent.memoryDetail.factor_name }}</strong>
              <span class="memory-modal-verdict" :class="'memory-' + agent.memoryDetail.verdict">{{ memoryVerdictLabel(agent.memoryDetail.verdict) }}</span>
              <button class="memory-modal-close" @click="agent.memoryDetail = null">×</button>
            </div>
            <div class="memory-modal-body">
              <div class="memory-modal-section">
                <label>因子表达式</label>
                <pre class="memory-modal-expr">{{ agent.memoryDetail.expression }}</pre>
              </div>
              <div class="memory-modal-section" v-if="agent.memoryDetail.conclusion">
                <label>结论</label>
                <p class="memory-modal-text">{{ agent.memoryDetail.conclusion }}</p>
              </div>
              <div class="memory-modal-section" v-if="agent.memoryDetail.metrics && Object.keys(agent.memoryDetail.metrics).length">
                <label>评估指标</label>
                <div class="memory-modal-metrics">
                  <span v-for="(value, key) in agent.memoryDetail.metrics" :key="key">{{ metricLabel(key) }}: {{ formatMetricValue(value) }}</span>
                </div>
              </div>
              <div class="memory-modal-section" v-if="agent.memoryDetail.error">
                <label>错误/跳过原因</label>
                <pre class="memory-modal-error">{{ agent.memoryDetail.error }}</pre>
              </div>
              <div class="memory-modal-section" v-if="agent.memoryDetail.observations && agent.memoryDetail.observations.length">
                <label>评估历史 ({{ agent.memoryDetail.attempts || agent.memoryDetail.observations.length }} 次)</label>
                <div class="memory-modal-observations">
                  <div v-for="(obs, i) in agent.memoryDetail.observations" :key="i" class="memory-modal-observation">
                    <span>{{ obs.stage || '-' }}</span>
                    <span class="memory-modal-obs-verdict" :class="'memory-' + obs.verdict">{{ memoryVerdictLabel(obs.verdict) }}</span>
                    <span class="memory-modal-obs-time">{{ formatTime(obs.at) }}</span>
                  </div>
                </div>
              </div>
              <div class="memory-modal-meta">
                <span v-if="agent.memoryDetail.created_at">创建: {{ formatTime(agent.memoryDetail.created_at) }}</span>
                <span v-if="agent.memoryDetail.updated_at">更新: {{ formatTime(agent.memoryDetail.updated_at) }}</span>
                <span v-if="agent.memoryDetail.profile_id">Profile: {{ agent.memoryDetail.profile_id }}</span>
                <span v-if="agent.memoryDetail.candidate_id">候选ID: {{ agent.memoryDetail.candidate_id }}</span>
                <span v-if="agent.memoryDetail.failure_code">失败码: {{ agent.memoryDetail.failure_code }}</span>
              </div>
            </div>
          </div>
        </div>
      </Teleport>

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
            <span v-if="agentBusy" class="activity-line"><i></i>{{ currentActivity }}</span>
            <span v-if="agent.usage.calls" class="usage-chip" title="本次 Agent 模型调用累计 usage">
              ↑ {{ formatTokens(agent.usage.input_tokens) }} · ↓ {{ formatTokens(agent.usage.output_tokens) }}
              <em v-if="agent.usage.cache_input_tokens">缓存 {{ formatTokens(agent.usage.cache_input_tokens) }}</em>
              <em v-else>缓存未返回</em>
            </span>
            <span v-if="agent.current" class="live-status" :class="'live-' + agent.current.status">
              <i></i>{{ statusLabel(agent.current.status) }}
            </span>
            <button v-if="canStopAgent" class="stop-btn" @click="stopAgent">停止</button>
          </div>
        </header>

        <div ref="thread" class="agent-thread" @scroll="onThreadScroll">
          <div v-if="!agent.events.length" class="welcome">
            <div class="welcome-orb">✦</div>
            <h2>开始一次因子研究</h2>
            <p>描述你想研究的方向，或者让 Agent 自己探索。它会实时展示思考摘要、工具调用和评估结果。</p>
            <div class="suggestions">
              <button @click="useSuggestion('从价量、动量和反转方向自主挖掘候选因子，并在训练集和验证集上检验。')">价量与动量</button>
              <button @click="useSuggestion('重点研究成交量异常、流动性和波动率收缩，要求控制极端值和覆盖率。')">成交量与波动率</button>
              <button @click="useSuggestion('检查已有因子库中的研究空白，提出不重复的 A 股日频因子。')">寻找研究空白</button>
            </div>
          </div>

          <div v-for="(message, index) in timeline" :key="message.key + '-' + index" class="message-row" :class="'message-' + message.kind">
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

          <div v-if="agentBusy && !timeline.some(x => x.kind === 'thinking' || x.kind === 'tool_call')" class="typing-row">
            <div class="avatar agent-avatar">✦</div>
            <div class="typing"><i></i><i></i><i></i></div>
            <span>AlphaAgent 正在准备下一步…</span>
          </div>
        </div>

        <div class="composer-wrap">
          <div v-if="agent.error" class="composer-error">{{ agent.error }}</div>
          <div v-if="agentBusy" class="activity-bar"><i></i><span>{{ currentActivity }}</span></div>
          <div class="composer">
            <textarea
              v-model="agent.form.user_message"
              :disabled="agent.current?.status === 'stopping'"
              rows="3"
              :placeholder="composerPlaceholder"
              @keydown.ctrl.enter.prevent="sendMessage"
              @keydown.meta.enter.prevent="sendMessage"
            ></textarea>
            <div class="composer-bottom">
              <div class="composer-options">
                <label>训练 {{ agent.form.train_start }} → {{ agent.form.train_end }}</label>
                <label>验证 {{ agent.form.val_start }} → {{ agent.form.val_end }}</label>
                <button class="spec-toggle" :class="{ active: agent.showResearchSpec }" :disabled="agentBusy" @click="agent.showResearchSpec = !agent.showResearchSpec">研究规范</button>
              </div>
              <button class="send-btn" :disabled="agent.current?.status === 'stopping' || !agent.form.user_message.trim()" @click="sendMessage" :title="composerActionTitle">
                <span>{{ agentBusy || agent.current ? '↵' : '↑' }}</span>
              </button>
            </div>
          </div>
          <div v-if="agent.showResearchSpec" class="research-spec-editor">
            <div class="research-spec-head">
              <strong>ResearchSpec</strong>
              <div>
                <span v-if="agent.specError" class="research-spec-error">{{ agent.specError }}</span>
                <button title="恢复默认研究规范" @click="resetResearchSpec">恢复默认</button>
                <button class="research-spec-close" title="关闭研究规范" aria-label="关闭研究规范" @click="agent.showResearchSpec = false">×</button>
              </div>
            </div>
            <textarea v-model="agent.researchSpecText" :disabled="agentBusy" spellcheck="false" aria-label="ResearchSpec JSON"></textarea>
          </div>
        <p class="composer-hint">AgentScope 实时事件 · 训练集评估 · 验证集检验 · FactorZoo 提交</p>
        </div>
      </main>
    </div>

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
  </section>
</template>

<script>
import { api } from '../utils/api.js'

export default {
  name: 'AlphaAgent',
  data() {
    return {
      agent: {
        runs: [],
        current: null,
        events: [],
        memory: [],
        usage: { calls: 0, input_tokens: 0, output_tokens: 0, cache_input_tokens: 0, cache_creation_input_tokens: 0 },
        stream: null,
        running: false,
        error: '',
        pendingMessages: 0,
        stickToBottom: true,
        menuRunId: '',
        menuPosition: { top: 0, left: 0 },
        renameRunId: '',
        renameTitle: '',
        renamePosition: { top: 0, left: 0 },
        showArchived: false,
        memoryDetail: null,
        showResearchSpec: false,
        researchSpecText: '',
        defaultResearchSpecText: '',
        specError: '',
        form: {
          train_start: '2018-01-01',
          train_end: '2022-12-31',
          val_start: '2023-01-01',
          val_end: '2025-12-31',
          user_message: '',
          max_turns: 5,
          max_tool_calls_per_round: 8,
          max_parallel_eval: 2,
          allow_submit: false,
        },
      },
      subtab: 'research',
      lab: {
        expr: '',
        factorName: 'expr',
        trainStart: '2018-01-01',
        trainEnd: '2022-12-31',
        valStart: '2023-01-01',
        valEnd: '2025-12-31',
        includeFundamentals: false,
        busy: false,
        error: '',
        results: null,
      },
      lib: {
        library: 'production',
        data: null,
        loading: false,
        error: '',
      },
      factorDetail: null,
    }
  },
  computed: {
    menuRun() {
      return this.agent.runs.find(run => run.run_id === this.agent.menuRunId) || null
    },
    renameRun() {
      return this.agent.runs.find(run => run.run_id === this.agent.renameRunId) || null
    },
    agentBusy() {
      const status = this.agent.current?.status
      return this.agent.running || ['starting', 'running', 'stopping'].includes(status)
    },
    currentActivity() {
      const status = this.agent.current?.status
      if (status === 'stopping') return '正在停止 Agent…'
      if (this.agent.pendingMessages) return '已排队 ' + this.agent.pendingMessages + ' 条追加指令，等待当前步骤结束…'
      if (!this.agent.events.length) return '正在加载数据并初始化 Agent…'
      const last = [...this.agent.events].reverse().find(e => !['heartbeat', 'stream_start', 'usage', 'reviewer_usage', 'usage_total'].includes(e.event)) || {}
      if (last.event === 'session_start') return '研究会话已建立 · 模型 ' + (last.model || '当前模型')
      if (last.event === 'agent_thinking') return this.dynamicThinking(last.content)
      if (last.event === 'assistant_tool_call') {
        const args = this.parseArgs(last.arguments_raw)
        const name = last.name || '研究工具'
        const factor = args.factor_name ? ' · ' + args.factor_name : ''
        return '调用 ' + name + factor
      }
      if (last.event === 'tool_results') {
        const names = (last.results || []).map(x => x.name).filter(Boolean)
        const unique = [...new Set(names)]
        return '已完成 ' + (unique.length ? unique.join('、') : '工具调用') + ' · 正在比较结果'
      }
      if (last.event === 'assistant_message') return this.dynamicThinking(last.content, 'Agent 输出')
      if (last.event === 'reviewer_start') return 'FactorReviewer 正在独立审查候选的新颖性与稳健性'
      if (last.event === 'reviewer_thinking') return this.dynamicThinking(last.content, 'FactorReviewer 审查')
      if (last.event === 'reviewer_message') return this.dynamicThinking(last.content, 'FactorReviewer 输出')
      if (last.event === 'nudge') return 'Agent 正在根据当前结果继续推进研究'
      return last.event ? '事件：' + last.event : '正在研究…'
    },
    canStopAgent() {
      return ['starting', 'running'].includes(this.agent.current?.status)
    },
    composerPlaceholder() {
      if (this.agentBusy) return '向当前 Agent 追加研究指令…（Ctrl/⌘ + Enter）'
      if (this.agent.current) return '从当前历史会话继续研究…（Ctrl/⌘ + Enter）'
      return '告诉 AlphaAgent 你想研究什么…（Ctrl/⌘ + Enter 发送）'
    },
    composerActionTitle() {
      if (this.agentBusy) return '追加到当前研究会话'
      if (this.agent.current) return '从当前历史会话继续'
      return '启动研究'
    },
    timeline() {
      const out = []
      for (const [index, event] of this.agent.events.entries()) {
        const key = event.ts || String(index)
        if (event.event === 'heartbeat' || event.event === 'stream_start' || event.event === 'usage' || event.event === 'reviewer_usage' || event.event === 'usage_total') continue
        if (event.event === 'user_message') {
          out.push({ key, kind: 'user', text: event.content || '' })
        } else if (event.event === 'agent_thinking') {
          out.push({ key, kind: 'thinking', label: '思考摘要', text: event.content || '' })
        } else if (event.event === 'assistant_tool_call') {
          const args = this.parseArgs(event.arguments_raw)
          out.push({ key, kind: 'tool_call', name: event.name || 'tool', factorName: args.factor_name || '', expression: args.multi_line_expr || '', text: event.arguments_raw || '' })
        } else if (event.event === 'assistant_message') {
          out.push({ key, kind: 'assistant', text: event.content || '' })
        } else if (event.event === 'assistant') {
          if (event.reasoning) out.push({ key: key + '-r', kind: 'thinking', label: '思考摘要', text: event.reasoning })
          if (event.content) out.push({ key: key + '-a', kind: 'assistant', text: event.content })
          for (const call of event.tool_calls || []) {
            const args = this.parseArgs(call.function?.arguments || '')
            out.push({ key: key + '-t-' + (call.id || call.function?.name || ''), kind: 'tool_call', name: call.function?.name || call.name || 'tool', factorName: args.factor_name || '', expression: args.multi_line_expr || '', text: call.function?.arguments || '' })
          }
        } else if (event.event === 'tool_results') {
          for (const result of event.results || []) out.push(this.toolResultMessage(result, key))
        } else if (event.event === 'run_summary') {
          const tools = event.tool_calls || {}
          out.push({ key, kind: 'system', text: '运行总结 · 工具调用 ' + (tools.count || 0) + ' 次 · 成功 ' + (tools.ok || 0) + ' 次' })
        } else if (event.event === 'session_start') {
          out.push({ key, kind: 'system', text: '会话开始 · ' + (event.model || '-') })
        } else if (event.event === 'session_end') {
          out.push({ key, kind: 'system', text: '会话结束 · ' + (event.reason || '-') })
        } else if (event.event === 'nudge') {
          out.push({ key, kind: 'system', text: 'Agent 继续推进研究' })
        } else if (event.event === 'continuation_queued') {
          out.push({ key, kind: 'system', text: '已追加指令，等待当前模型调用或工具批次结束后继续' })
          out.push({ key: key + '-user', kind: 'user', text: event.content || '' })
        } else if (event.event === 'continuation_accepted') {
          out.push({ key, kind: 'system', text: 'Agent 已接收追加指令，开始下一轮研究' })
        } else if (event.event === 'continuation_started') {
          out.push({ key, kind: 'system', text: '已从这段研究历史恢复上下文，继续新的 Agent 回合' })
          out.push({ key: key + '-user', kind: 'user', text: event.content || '' })
        } else if (event.event === 'branch_started') {
          out.push({ key, kind: 'system', text: '已从此处新建分支，分支会继承此前研究上下文' })
          out.push({ key: key + '-user', kind: 'user', text: event.content || '' })
        } else if (event.event === 'reviewer_start') {
          out.push({ key, kind: 'system', text: 'FactorReviewer 开始独立审查候选因子' })
        } else if (event.event === 'reviewer_thinking') {
          out.push({ key, kind: 'reviewer_thinking', text: event.content || '' })
        } else if (event.event === 'reviewer_message') {
          out.push({ key, kind: 'assistant', text: 'FactorReviewer：' + (event.content || '') })
        } else if (event.event === 'factor_review') {
          out.push({
            key,
            kind: 'review',
            verdict: event.verdict || 'reject',
            novelty: event.novelty || 'low',
            canonical: event.canonical_form || '未分类',
            reasons: Array.isArray(event.reasons) ? event.reasons : [],
            changes: Array.isArray(event.required_changes) ? event.required_changes : [],
          })
        }
      }
      return out
    },
  },
  methods: {
    async loadAgentRuns() {
      try {
        this.agent.runs = await api('/api/alphaagent/runs?archived_only=' + this.agent.showArchived + '&t=' + Date.now())
      } catch (e) {
        this.agent.error = '读取任务失败: ' + e.message
      }
    },
    async loadResearchMemory() {
      try {
        const payload = await api('/api/alphaagent/research-memory?limit=50&t=' + Date.now())
        this.agent.memory = payload.entries || []
      } catch (e) {
        this.agent.error = '读取长期研究记忆失败: ' + e.message
      }
    },
    async loadDefaultResearchSpec() {
      try {
        const spec = await api('/api/alphaagent/research-spec/default?t=' + Date.now())
        const text = JSON.stringify(spec, null, 2)
        this.agent.defaultResearchSpecText = text
        if (!this.agent.researchSpecText) this.agent.researchSpecText = text
      } catch (e) {
        this.agent.error = '读取默认 ResearchSpec 失败: ' + e.message
      }
    },
    resetResearchSpec() {
      this.agent.researchSpecText = this.agent.defaultResearchSpecText
      this.agent.specError = ''
    },
    parseResearchSpec() {
      try {
        const spec = JSON.parse(this.agent.researchSpecText || this.agent.defaultResearchSpecText || '{}')
        if (!spec || typeof spec !== 'object' || Array.isArray(spec)) throw new Error('必须是 JSON 对象')
        this.agent.specError = ''
        return spec
      } catch (e) {
        this.agent.specError = 'ResearchSpec JSON 无效: ' + e.message
        throw e
      }
    },
    toggleMenu(runId, event) {
      if (this.agent.menuRunId === runId) {
        this.agent.menuRunId = ''
        return
      }
      const rect = event.currentTarget.getBoundingClientRect()
      const width = 128
      const height = 142
      this.agent.menuPosition = {
        top: Math.max(8, Math.min(rect.top, window.innerHeight - height - 8)),
        left: Math.max(8, Math.min(rect.right - width, window.innerWidth - width - 8)),
      }
      this.agent.menuRunId = runId
    },
    async toggleArchived() {
      this.agent.showArchived = !this.agent.showArchived
      this.agent.menuRunId = ''
      await this.loadAgentRuns()
    },
    beginRename(run) {
      this.agent.menuRunId = ''
      this.agent.renameRunId = run.run_id
      this.agent.renameTitle = run.title || this.runTitle(run)
      const titleEl = document.querySelector('[data-run-title="' + run.run_id + '"]')
      const rect = titleEl?.getBoundingClientRect()
      this.agent.renamePosition = {
        top: Math.max(8, Math.min(rect?.top || 8, window.innerHeight - 42)),
        left: Math.max(8, Math.min(rect?.left || 8, window.innerWidth - 268)),
      }
      this.$nextTick(() => {
        const input = this.$refs.renameInput
        input?.focus()
        input?.select()
      })
    },
    cancelRename() {
      this.agent.renameRunId = ''
      this.agent.renameTitle = ''
    },
    async commitRename(run) {
      if (this.agent.renameRunId !== run.run_id) return
      const title = this.agent.renameTitle.trim()
      this.cancelRename()
      if (!title || title === (run.title || this.runTitle(run))) return
      try {
        const updated = await api('/api/alphaagent/runs/' + encodeURIComponent(run.run_id) + '/rename', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title }),
        })
        const index = this.agent.runs.findIndex(item => item.run_id === updated.run_id)
        if (index >= 0) this.agent.runs.splice(index, 1, { ...this.agent.runs[index], ...updated })
        if (this.agent.current?.run_id === updated.run_id) this.agent.current = { ...this.agent.current, title: updated.title }
        await this.loadAgentRuns()
      } catch (e) {
        this.agent.error = '重命名失败: ' + e.message
      }
    },
    async pinRun(run) {
      this.agent.menuRunId = ''
      try {
        const updated = await api('/api/alphaagent/runs/' + encodeURIComponent(run.run_id) + '/pin', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ pinned: !run.pinned }),
        })
        if (this.agent.current?.run_id === updated.run_id) this.agent.current = { ...this.agent.current, pinned: updated.pinned }
        await this.loadAgentRuns()
      } catch (e) {
        this.agent.error = (run.pinned ? '取消置顶失败: ' : '置顶失败: ') + e.message
      }
    },
    async archiveRun(run) {
      this.agent.menuRunId = ''
      try {
        await api('/api/alphaagent/runs/' + encodeURIComponent(run.run_id) + '/archive', { method: 'POST' })
        await this.loadAgentRuns()
      } catch (e) {
        this.agent.error = '归档失败: ' + e.message
      }
    },
    async branchRun(run) {
      this.agent.menuRunId = ''
      const content = window.prompt('给新分支的研究指令', '基于以上研究轨迹，换一个角度继续挖掘并完成训练集和验证集检验。')
      if (content == null || !content.trim()) return
      try {
        if (this.agent.stream) this.agent.stream.close()
        const parent = await api('/api/alphaagent/runs/' + encodeURIComponent(run.run_id) + '?t=' + Date.now())
        const parentEvents = await this.conversationEvents(parent)
        const result = await api('/api/alphaagent/runs/' + encodeURIComponent(run.run_id) + '/branch', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ content: content.trim() }),
        })
        this.agent.current = result
        this.agent.current.status = result.status || 'starting'
        this.agent.events = [
          ...parentEvents,
          { event: 'branch_started', ts: new Date().toISOString(), parent_run_id: run.run_id, content: content.trim() },
          ...(result.events || []),
        ]
        this.agent.usage = this.usageFromEvents(this.agent.events)
        this.agent.pendingMessages = 0
        this.agent.running = true
        await this.loadAgentRuns()
        this.connectAgentEvents(result.run_id)
        this.$nextTick(() => this.scrollThread(true))
      } catch (e) {
        this.agent.error = '新建分支失败: ' + e.message
      }
    },
    newRun() {
      if (this.agent.stream) this.agent.stream.close()
      this.agent.stream = null
      this.agent.current = null
      this.agent.events = []
      this.agent.usage = { calls: 0, input_tokens: 0, output_tokens: 0, cache_input_tokens: 0, cache_creation_input_tokens: 0 }
      this.agent.pendingMessages = 0
      this.agent.error = ''
      this.agent.running = false
      this.agent.form.user_message = ''
    },
    useSuggestion(text) {
      this.agent.form.user_message = text
      this.$nextTick(() => this.$refs.composer?.focus())
    },
    parseArgs(raw) {
      if (!raw) return {}
      try { return typeof raw === 'string' ? JSON.parse(raw) : raw } catch (e) { return {} }
    },
    dynamicThinking(text, prefix = '思考') {
      const value = String(text || '').replace(/```[\s\S]*?```/g, '').replace(/\s+/g, ' ').trim()
      if (!value) return prefix + '…'
      const clean = value.replace(/^#+\s*/, '').replace(/^[-*]\s*/, '')
      return prefix + ' · ' + (clean.length > 140 ? clean.slice(0, 140) + '…' : clean)
    },
    async startAgent() {
      if (this.agentBusy || !this.agent.form.user_message.trim()) return
      this.agent.error = ''
      let researchSpec
      try {
        researchSpec = this.parseResearchSpec()
      } catch (e) {
        return
      }
      this.agent.running = true
      try {
        if (this.agent.stream) this.agent.stream.close()
        const result = await api('/api/alphaagent/runs', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            ...this.agent.form,
            research_spec: researchSpec,
            allow_submit: Boolean(researchSpec.delivery_policy?.allow_submit),
          }),
        })
        this.agent.current = result
        this.agent.current.status = result.status || 'starting'
        this.agent.events = result.events || []
        this.agent.usage = { calls: 0, input_tokens: 0, output_tokens: 0, cache_input_tokens: 0, cache_creation_input_tokens: 0 }
        this.agent.pendingMessages = 0
        this.agent.form.user_message = ''
        await this.loadAgentRuns()
        this.connectAgentEvents(result.run_id)
        this.$nextTick(() => this.scrollThread(true))
      } catch (e) {
        this.agent.error = '启动失败: ' + e.message
        this.agent.running = false
      }
    },
    async sendMessage() {
      if (!this.agent.form.user_message.trim()) return
      if (!this.agent.current) {
        await this.startAgent()
        return
      }
      if (this.agent.current.status === 'stopping') return
      const content = this.agent.form.user_message.trim()
      if (!this.agentBusy) {
        await this.resumeConversation(content)
        return
      }
      try {
        const result = await api('/api/alphaagent/runs/' + encodeURIComponent(this.agent.current.run_id) + '/messages', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ content }),
        })
        if (!result.ok) throw new Error('追加失败')
        this.agent.events.push({ event: 'continuation_queued', ts: new Date().toISOString(), content })
        this.agent.pendingMessages += 1
        this.agent.form.user_message = ''
        this.$nextTick(() => this.scrollThread(true))
      } catch (e) {
        this.agent.error = '追加指令失败: ' + e.message
      }
    },
    async resumeConversation(content) {
      const previousEvents = this.agent.events.slice()
      const previousRun = this.agent.current
      try {
        const result = await api('/api/alphaagent/runs/' + encodeURIComponent(previousRun.run_id) + '/continue', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ content }),
        })
        if (result.status === 'queued') {
          this.agent.events.push({ event: 'continuation_queued', ts: new Date().toISOString(), content })
          this.agent.form.user_message = ''
          return
        }
        this.agent.current = result
        this.agent.current.status = result.status || 'starting'
        this.agent.events = [
          ...previousEvents,
          { event: 'continuation_started', ts: new Date().toISOString(), parent_run_id: previousRun.run_id, content },
          ...(result.events || []),
        ]
        this.agent.running = true
        this.agent.form.user_message = ''
        await this.loadAgentRuns()
        this.connectAgentEvents(result.run_id)
        this.$nextTick(() => this.scrollThread(true))
      } catch (e) {
        this.agent.error = '恢复历史会话失败: ' + e.message
      }
    },
    connectAgentEvents(runId) {
      if (this.agent.stream) this.agent.stream.close()
      const es = new EventSource('/api/alphaagent/runs/' + encodeURIComponent(runId) + '/events?t=' + Date.now())
      this.agent.stream = es
      es.onmessage = (message) => {
        try {
          const event = JSON.parse(message.data)
      if (event.event === 'heartbeat') return
          if (event.event === 'stream_end') {
            this.agent.running = false
            if (this.agent.current) this.agent.current.status = event.status || 'completed'
            es.close()
            this.agent.stream = null
            this.loadAgentRuns()
            this.$nextTick(() => this.scrollThread(true))
            return
          }
          if (event.event === 'stream_start') {
            if (this.agent.current) this.agent.current.status = 'running'
          } else if (event.event === 'usage_total') {
            this.agent.usage = { ...this.agent.usage, ...event }
          } else {
            this.agent.events.push(event)
            if (event.event === 'research_memory_updated') this.loadResearchMemory()
            if (event.event === 'continuation_accepted') {
              this.agent.pendingMessages = Math.max(0, this.agent.pendingMessages - Number(event.count || 1))
            }
            if (event.event === 'usage') this.agent.usage = this.addUsage(this.agent.usage, event)
            if (this.agent.current) {
              this.agent.current.status = 'running'
              this.agent.current.event_count = this.agent.events.length
            }
          }
          this.$nextTick(() => this.scrollThread())
        } catch (e) {
          this.agent.error = '事件解析失败: ' + e.message
        }
      }
      es.onerror = () => {
        // EventSource 会自动重连；任务运行时不要主动关闭。
        if (this.agent.current && ['completed', 'failed', 'stopping'].includes(this.agent.current.status)) {
          es.close()
          this.agent.stream = null
          this.agent.running = false
        }
      }
    },
    async selectAgentRun(run) {
      if (this.agent.stream) this.agent.stream.close()
      try {
        const detail = await api('/api/alphaagent/runs/' + encodeURIComponent(run.run_id) + '?t=' + Date.now())
        this.agent.current = detail
        if (detail.research_spec) {
          this.agent.researchSpecText = JSON.stringify(detail.research_spec, null, 2)
          this.agent.specError = ''
        }
        this.agent.events = await this.conversationEvents(detail)
        this.agent.usage = this.usageFromEvents(this.agent.events)
        this.agent.pendingMessages = 0
        this.agent.running = ['starting', 'running'].includes(detail.status)
        if (this.agent.running) this.connectAgentEvents(run.run_id)
        this.$nextTick(() => this.scrollThread(true))
      } catch (e) {
        this.agent.error = '读取任务失败: ' + e.message
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
      if (!this.canStopAgent || !this.agent.current) return
      try {
        await api('/api/alphaagent/runs/' + encodeURIComponent(this.agent.current.run_id) + '/stop', { method: 'POST' })
        this.agent.current.status = 'stopping'
        this.agent.running = false
      } catch (e) {
        this.agent.error = '停止失败: ' + e.message
      }
    },
    onThreadScroll() {
      const el = this.$refs.thread
      if (!el) return
      this.agent.stickToBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 100
    },
    scrollThread(force = false) {
      const el = this.$refs.thread
      if (!el || (!force && !this.agent.stickToBottom)) return
      el.scrollTop = el.scrollHeight
    },
    toolResultMessage(row, key) {
      const result = row.result || {}
      const summary = result.summary || result.metrics?.cross_sectional_core || {}
      const args = this.parseArgs(row.arguments_raw || '')
      const profile = result.profile?.profile_id || args.profile_id || ''
      const metrics = [
        profile ? 'Profile ' + profile : '',
        summary.ic != null ? 'IC ' + summary.ic : '',
        summary.rank_ic != null ? 'RankIC ' + summary.rank_ic : '',
        summary.icir != null ? 'ICIR ' + summary.icir : '',
        summary.factor_coverage != null ? 'Coverage ' + summary.factor_coverage : '',
        result.stored != null ? 'stored ' + result.stored : '',
        result.candidate_stored ? '候选池已保存' : '',
        result.metrics?.long_group_annual_excess_return != null ? '多头年化超额 ' + result.metrics.long_group_annual_excess_return : '',
        result.metrics?.winsorized_abs_ic_decay != null ? '截尾IC衰减 ' + result.metrics.winsorized_abs_ic_decay : '',
        result.candidate_state ? '状态 ' + result.candidate_state : '',
        result.rule_results?.length ? '规则 ' + result.rule_results.filter(x => x.passed).length + '/' + result.rule_results.length : '',
      ].filter(Boolean)
      return {
        key: key + '-' + (row.tool_call_id || row.name || Math.random()),
        kind: 'tool_result',
        name: row.name || 'tool',
        factorName: args.factor_name || '',
        expression: args.multi_line_expr || '',
        ok: result.ok !== false || result.candidate_stored === true,
        elapsed: row.elapsed_seconds != null ? Number(row.elapsed_seconds).toFixed(1) : '',
        metrics,
        text: result.error || result.skipped_reason || '',
      }
    },
    toolLabel(name) {
      return ({
        evaluate_factor: 'Profile 评估',
        eval_on_train_set: '训练集评估',
        eval_on_val_set: '验证集评估',
        submit_factor: '提交因子',
      }[name] || name || '工具')
    },
    reviewLabel(verdict) {
      return ({ approve: '允许提交', revise: '需要修订', reject: '拒绝提交' }[verdict] || '拒绝提交')
    },
    memoryVerdictLabel(verdict) {
      return ({
        production_approved: '正式入库',
        candidate_approved: '候选保留',
        validated: '验证有效',
        promising: '训练有潜力',
        rejected: '明确否定',
        revise_required: '需修订',
        weak: '证据不足',
      }[verdict] || '待评估')
    },
    showMemoryDetail(entry) {
      this.agent.memoryDetail = entry
    },
    metricLabel(key) {
      return ({
        ic: 'IC',
        icir: 'ICIR',
        rank_ic: 'RankIC',
        factor_coverage: '覆盖率',
        coverage: '覆盖率',
        long_group_annual_excess_return: '多头年化超额',
        winsorized_abs_ic_decay: '截尾IC衰减',
      }[key] || key)
    },
    formatMetricValue(value) {
      const n = Number(value)
      return isNaN(n) ? String(value) : Math.abs(n) < 1 ? n.toFixed(4) : n.toFixed(2)
    },
    addUsage(total, event) {
      return {
        calls: (total.calls || 0) + 1,
        input_tokens: (total.input_tokens || 0) + Number(event.input_tokens || 0),
        output_tokens: (total.output_tokens || 0) + Number(event.output_tokens || 0),
        cache_input_tokens: (total.cache_input_tokens || 0) + Number(event.cache_input_tokens || 0),
        cache_creation_input_tokens: (total.cache_creation_input_tokens || 0) + Number(event.cache_creation_input_tokens || 0),
      }
    },
    usageFromEvents(events) {
      let total = { calls: 0, input_tokens: 0, output_tokens: 0, cache_input_tokens: 0, cache_creation_input_tokens: 0 }
      for (const event of events || []) {
        if (event.event === 'usage_total') total = { ...total, ...event }
        else if (event.event === 'usage') total = this.addUsage(total, event)
      }
      return total
    },
    formatTokens(value) {
      const n = Number(value || 0)
      return n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1) + 'k' : String(n)
    },
    runTitle(run) {
      if (run.title) return run.title.slice(0, 64)
      if (run.user_message) return run.user_message.slice(0, 32)
      return '因子研究任务 ' + String(run.run_id || '').slice(0, 8)
    },
    formatTime(value) {
      return value ? String(value).slice(0, 16).replace('T', ' ') : ''
    },
    statusLabel(status) {
      return ({ starting: '准备中', running: '运行中', stopping: '停止中', completed: '已完成', failed: '失败', interrupted: '已中断' }[status] || status || '未开始')
    },
    async runEval() {
      if (!this.lab.expr.trim() || this.lab.busy) return
      this.lab.busy = true
      this.lab.error = ''
      this.lab.results = null
      try {
        const data = await api('/api/alphaagent/eval-factor', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            multi_line_expr: this.lab.expr,
            factor_name: this.lab.factorName || 'expr',
            train_start: this.lab.trainStart,
            train_end: this.lab.trainEnd,
            val_start: this.lab.valStart,
            val_end: this.lab.valEnd,
            include_fundamentals: this.lab.includeFundamentals,
            all_profiles: true,
          }),
        })
        this.lab.results = data.results
      } catch (e) {
        this.lab.error = e.message
      } finally {
        this.lab.busy = false
      }
    },
    async loadFactors() {
      this.lib.loading = true
      this.lib.error = ''
      try {
        this.lib.data = await api('/api/alphaagent/factors?library=' + this.lib.library + '&t=' + Date.now())
      } catch (e) {
        this.lib.error = e.message
      } finally {
        this.lib.loading = false
      }
    },
    switchLibrary(lib) {
      this.lib.library = lib
      this.loadFactors()
    },
    async showFactorDetail(f) {
      try {
        this.factorDetail = await api('/api/alphaagent/factors/' + encodeURIComponent(f.factor_id) + '?library=' + this.lib.library + '&t=' + Date.now())
      } catch (e) {
        this.lib.error = e.message
      }
    },
    async confirmDelete(f) {
      if (!confirm('确认删除因子 ' + f.name + ' (' + String(f.factor_id).slice(0, 8) + ')？')) return
      try {
        await api('/api/alphaagent/factors/' + encodeURIComponent(f.factor_id) + '?library=' + this.lib.library, { method: 'DELETE' })
        this.loadFactors()
      } catch (e) {
        this.lib.error = e.message
      }
    },
    icClass(v) {
      if (v == null) return ''
      return v >= 0 ? 'ic-pos' : 'ic-neg'
    },
  },
  mounted() {
    this._closeSessionMenu = () => {
      this.agent.menuRunId = ''
      if (this.agent.renameRunId) this.cancelRename()
    }
    document.addEventListener('click', this._closeSessionMenu)
    this.loadAgentRuns()
    this.loadResearchMemory()
    this.loadDefaultResearchSpec()
  },
  beforeUnmount() {
    document.removeEventListener('click', this._closeSessionMenu)
    if (this.agent.stream) this.agent.stream.close()
  },
}
</script>

<style scoped>
.agent-page { height: calc(100vh - 122px); min-height: 620px; }
.agent-shell { display: grid; grid-template-columns: 260px minmax(0, 1fr); height: 100%; overflow: hidden; border: 1px solid var(--line); border-radius: 14px; background: var(--card); box-shadow: var(--shadow); }
.agent-sidebar { display: flex; flex-direction: column; border-right: 1px solid var(--line); background: rgb(8 14 27 / .32); min-width: 0; }
.sidebar-head { display:flex; align-items:center; justify-content:space-between; padding: 20px 16px 14px; }
.eyebrow { color: var(--muted); font-size: 10px; letter-spacing: .14em; }
.sidebar-head h2 { margin-top: 4px; font-size: 18px; }
.icon-btn { width: 30px; height: 30px; padding: 0; border-radius: 8px; font-size: 20px; }
.new-run { margin: 0 12px 18px; text-align: left; color: var(--text); background: rgb(79 140 255 / .14); border-color: rgb(79 140 255 / .32); }
.new-run span { color: var(--accent); font-size: 18px; margin-right: 5px; }
.session-label { display:flex; align-items:center; justify-content:space-between; padding: 0 16px 8px; color: var(--muted); font-size: 11px; }
.archived-toggle { padding:0; border:0; background:transparent; color:var(--muted); font-size:10px; }
.archived-toggle:hover { color:var(--text); }
.session-list { overflow: auto; padding: 0 8px; }
.session-item { position:relative; display:flex; width:100%; align-items:center; margin-bottom:3px; border:1px solid transparent; border-radius:7px; }
.session-item:hover, .session-item.active { background: rgb(79 140 255 / .11); border-color: rgb(79 140 255 / .24); }
.session-select { display:flex; min-width:0; flex:1; align-items:center; gap:8px; padding:10px 4px 10px 8px; text-align:left; border:0; background:transparent; }
.session-menu { width:25px; height:25px; margin-right:4px; padding:0; border:0; border-radius:5px; background:transparent; color:var(--muted); font:16px/1 var(--font-mono); letter-spacing:0; }
.session-menu:hover, .session-menu:focus-visible { background:rgb(255 255 255 / .09); color:var(--text); }
.session-menu-popover { position:fixed; z-index:9999; min-width:128px; padding:4px; border:1px solid var(--line-strong); border-radius:7px; background:var(--bg-soft); box-shadow:0 10px 25px rgb(0 0 0 / .28); }
.session-menu-popover button { display:block; width:100%; padding:7px 8px; border:0; border-radius:4px; background:transparent; color:var(--text); text-align:left; font-size:11px; }
.session-menu-popover button:hover { background:rgb(79 140 255 / .14); }
.session-menu-popover .archive-action { color:#ef9ca1; }
.session-rename-popover { position:fixed; z-index:10000; display:flex; width:260px; height:30px; padding:2px; border:1px solid var(--accent); border-radius:6px; background:var(--bg-soft); box-shadow:0 10px 25px rgb(0 0 0 / .32); }
.session-rename-popover input { flex:1; min-width:0; border:0; outline:0; padding:0 7px; background:transparent; color:var(--text); font:600 12px/1 var(--font-sans); letter-spacing:0; }
.session-rename-popover button { width:27px; padding:0; border:0; border-radius:4px; background:rgb(79 140 255 / .18); color:var(--accent); font:16px/1 var(--font-mono); }
.session-rename-popover button:hover { background:rgb(79 140 255 / .3); }
.status-dot { width: 7px; height: 7px; flex: none; border-radius: 50%; background: var(--muted); }
.status-running { background: #43d17a; box-shadow: 0 0 10px rgb(67 209 122 / .65); }
.status-starting { background: #f5bd4f; }
.status-stopping { background: #f5bd4f; }
.status-completed { background: #6680a8; }
.status-failed { background: #ef6b73; }
.status-interrupted { background: #f5bd4f; }
.session-copy { min-width:0; flex:1; }
.session-copy strong, .session-copy small { display:block; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.session-copy strong { font-size:12px; font-weight:550; }
.session-pin { display:inline-block; margin-right:5px; color:#f5bd4f; font:9px var(--font-mono); vertical-align:1px; }
.session-copy small { margin-top:3px; color:var(--muted); font-size:10px; }
.session-count { color:var(--muted); font-size:10px; }
.sidebar-empty { padding:18px 8px; color:var(--muted); font-size:12px; text-align:center; }
.memory-panel { margin:10px 12px 0; padding:9px; border-top:1px solid var(--line); color:var(--muted); font-size:10px; }
.memory-head { display:flex; align-items:center; justify-content:space-between; margin-bottom:7px; color:var(--text); font-size:11px; }
.memory-head b { color:var(--accent); font:10px var(--font-mono); }
.memory-items { display:grid; gap:6px; max-height:220px; overflow-y:auto; }
.memory-item { padding:5px 6px 5px 7px; border-left:2px solid var(--muted); border-radius:0 4px 4px 0; cursor:pointer; transition:background .12s; }
.memory-item:hover, .memory-item:focus-visible { background:rgb(255 255 255 / .05); outline:none; }
.memory-item strong, .memory-item small { display:block; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.memory-item strong { color:var(--text); font-size:10px; font-weight:550; }
.memory-item small { margin-top:2px; font-size:9px; }
.memory-production_approved, .memory-validated { border-color:#43d17a; }
.memory-candidate_approved, .memory-promising { border-color:#f5bd4f; }
.memory-rejected, .memory-weak { border-color:#ef6b73; }
.memory-revise_required { border-color:#f5bd4f; }
.sidebar-footer { margin-top:auto; display:flex; align-items:center; gap:8px; padding:14px 16px; border-top:1px solid var(--line); color:var(--muted); font-size:11px; }
.agent-main { display:flex; flex-direction:column; min-width:0; min-height:0; }
.agent-header { display:flex; align-items:center; justify-content:space-between; gap:16px; padding:16px 24px; border-bottom:1px solid var(--line); }
.agent-title { display:flex; align-items:center; gap:11px; min-width:0; }
.agent-orb, .welcome-orb { display:grid; place-items:center; flex:none; border-radius:50%; color:#d7c7ff; background:linear-gradient(145deg,#7b61ff,#3e83ff); box-shadow:0 0 22px rgb(100 100 255 / .32); }
.agent-orb { width:31px; height:31px; }
.agent-title h1 { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:15px; }
.agent-subtitle { margin-top:3px; color:var(--muted); font-size:11px; }
.header-actions { display:flex; align-items:center; gap:10px; }
.usage-chip { display:flex; align-items:center; gap:5px; padding:5px 8px; border:1px solid var(--line); border-radius:7px; color:var(--muted); font:10px var(--font-mono); white-space:nowrap; }
.usage-chip em { color:#91d4ac; font-style:normal; }
.activity-line { display:flex; align-items:center; gap:6px; color:#b9a3e8; font-size:11px; white-space:nowrap; }
.activity-line i, .activity-bar i { display:inline-block; width:6px; height:6px; border-radius:50%; background:#c792ea; box-shadow:0 0 9px rgb(199 146 234 / .85); animation:agentPulse 1.1s infinite; }
.live-status { display:flex; align-items:center; gap:6px; color:var(--muted); font-size:12px; }
.live-status i { width:7px; height:7px; border-radius:50%; background:var(--muted); }
.live-running i { background:#43d17a; box-shadow:0 0 8px rgb(67 209 122 / .75); }
.live-starting i, .live-stopping i { background:#f5bd4f; }
.stop-btn { min-width:54px; color:#ff9b9f; border-color:rgb(239 107 115 / .35); background:transparent; }
.agent-thread { flex:1; min-height:0; overflow:auto; padding:28px max(24px, calc((100% - 860px) / 2)); scroll-behavior:smooth; }
.welcome { max-width:650px; margin:10vh auto 0; text-align:center; }
.welcome-orb { width:58px; height:58px; margin:0 auto 18px; font-size:25px; }
.welcome h2 { font-size:21px; }
.welcome p { margin:10px auto 20px; max-width:520px; color:var(--muted); font-size:13px; line-height:1.7; }
.suggestions { display:flex; justify-content:center; gap:8px; flex-wrap:wrap; }
.suggestions button { color:var(--muted); background:transparent; font-size:12px; }
.message-row { margin:0 auto 20px; max-width:820px; }
.user-bubble { max-width:80%; margin-left:auto; padding:11px 14px; border-radius:13px 13px 3px 13px; background:rgb(79 140 255 / .18); border:1px solid rgb(79 140 255 / .26); white-space:pre-wrap; line-height:1.6; font-size:13px; }
.assistant-message { display:flex; gap:10px; }
.avatar { display:grid; place-items:center; flex:none; width:27px; height:27px; border-radius:50%; }
.agent-avatar { color:#d7c7ff; background:linear-gradient(145deg,#7b61ff,#3e83ff); }
.message-body { min-width:0; flex:1; }
.message-author { color:var(--accent-strong); font-size:11px; font-weight:650; margin:4px 0 5px; }
.message-text { white-space:pre-wrap; color:var(--text); font-size:13px; line-height:1.7; overflow-wrap:break-word; }
.thinking-card, .tool-card, .result-card { margin-left:37px; border:1px solid var(--line); border-radius:10px; background:rgb(255 255 255 / .018); }
.thinking-card summary, .tool-card summary { cursor:pointer; padding:9px 11px; color:#b9a3e8; font-size:12px; list-style:none; }
.thinking-card summary::-webkit-details-marker, .tool-card summary::-webkit-details-marker { display:none; }
.thinking-icon { color:#c792ea; }
.tool-icon { color:#7dd3fc; margin-right:5px; }
.tool-state { float:right; color:var(--muted); font-size:10px; }
.tool-expression, .result-expression { margin:0 11px 8px; padding:7px 8px; border-left:2px solid rgb(125 211 252 / .65); border-radius:3px; color:#c7d8ee; background:rgb(125 211 252 / .055); white-space:pre-wrap; overflow-wrap:break-word; word-break:break-all; font:11px/1.5 var(--font-mono); }
.thinking-card pre, .tool-card pre { margin:0; padding:0 11px 11px; color:#aebbd2; white-space:pre-wrap; overflow-wrap:break-word; word-break:break-all; font:12px/1.55 var(--font-mono); }
.result-card { padding:10px 12px; border-color:rgb(67 209 122 / .2); }
.result-head { display:flex; align-items:center; gap:7px; font-size:12px; }
.result-icon { color:#43d17a; font-weight:700; }
.result-icon.bad { color:#ef6b73; }
.result-time { margin-left:auto; color:var(--muted); font-size:10px; }
.result-factor { color:#d7e3ff; font:11px var(--font-mono); }
.result-expression { border-left-color:rgb(67 209 122 / .7); background:rgb(67 209 122 / .055); }
.result-metrics { display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; }
.result-metrics span { padding:3px 6px; border-radius:5px; color:#a9e9bd; background:rgb(67 209 122 / .1); font:11px var(--font-mono); overflow-wrap:break-word; word-break:break-all; }
.result-note { margin-top:7px; color:#e89a9f; font-size:11px; overflow-wrap:break-word; word-break:break-all; }
.review-card { margin-left:37px; padding:11px 12px; border:1px solid var(--line); border-radius:8px; background:rgb(255 255 255 / .018); font-size:12px; }
.review-head { display:flex; align-items:center; gap:8px; }
.review-head strong { color:#d7c7ff; }.review-head span { font-weight:650; }.review-head em { margin-left:auto; color:var(--muted); font:10px var(--font-mono); }
.review-approve { border-color:rgb(67 209 122 / .38); }.review-approve .review-head span { color:#72d69a; }
.review-revise { border-color:rgb(245 189 79 / .42); }.review-revise .review-head span { color:#f5bd4f; }
.review-reject { border-color:rgb(239 107 115 / .42); }.review-reject .review-head span { color:#ef8b92; }
.review-canonical { margin-top:7px; color:#c7d8ee; font:11px var(--font-mono); }
.review-card ul { margin:8px 0 0; padding-left:17px; color:var(--muted); line-height:1.6; }.review-changes { margin-top:8px; color:#d8c7a4; line-height:1.55; }
.system-message { color:var(--muted); text-align:center; font-size:11px; }
.typing-row { display:flex; align-items:center; gap:8px; max-width:820px; margin:0 auto 14px; color:var(--muted); font-size:11px; }
.typing { display:flex; gap:3px; }
.typing i { width:5px; height:5px; border-radius:50%; background:var(--accent); animation:agentPulse 1.2s infinite; }
.typing i:nth-child(2) { animation-delay:.15s; }.typing i:nth-child(3) { animation-delay:.3s; }
@keyframes agentPulse { 0%,80%,100%{opacity:.25;transform:translateY(0)}40%{opacity:1;transform:translateY(-3px)} }
.composer-wrap { max-width:860px; width:calc(100% - 48px); margin:0 auto 18px; }
.composer { border:1px solid var(--line-strong); border-radius:13px; background:var(--bg-soft); box-shadow:0 8px 28px rgb(0 0 0 / .15); }
.composer textarea { display:block; width:100%; resize:none; border:0; outline:0; background:transparent; color:var(--text); padding:13px 14px 7px; line-height:1.55; font-size:13px; }
.composer-bottom { display:flex; align-items:center; justify-content:space-between; gap:10px; padding:6px 8px 8px 14px; }
.composer-options { display:flex; align-items:center; gap:10px; flex-wrap:wrap; color:var(--muted); font-size:10px; }
.submit-toggle { display:flex; flex-direction:row; align-items:center; gap:4px; font-size:10px; }
.spec-toggle { padding:2px 6px; border-color:transparent; background:transparent; color:var(--muted); font-size:10px; }
.spec-toggle:hover, .spec-toggle.active { border-color:var(--line); color:var(--text); background:rgb(79 140 255 / .1); }
.research-spec-editor { margin-top:8px; border:1px solid var(--line); border-radius:8px; background:var(--bg-soft); overflow:hidden; }
.research-spec-head { display:flex; align-items:center; justify-content:space-between; gap:12px; padding:7px 9px; border-bottom:1px solid var(--line); color:var(--muted); font-size:11px; }
.research-spec-head strong { color:var(--text); font:11px var(--font-mono); }
.research-spec-head div { display:flex; align-items:center; gap:8px; }
.research-spec-head button { padding:0; border:0; background:transparent; color:var(--accent); font-size:10px; }
.research-spec-head .research-spec-close { width:18px; height:18px; border-radius:4px; color:var(--muted); font-size:16px; line-height:16px; }
.research-spec-head .research-spec-close:hover { background:rgb(255 255 255 / .09); color:var(--text); }
.research-spec-error { color:#ef8b92; font-size:10px; }
.research-spec-editor textarea { display:block; width:100%; min-height:250px; resize:vertical; border:0; outline:0; background:transparent; color:#c7d8ee; padding:10px; font:11px/1.55 var(--font-mono); }
.send-btn { display:grid; place-items:center; width:31px; height:31px; padding:0; border:0; border-radius:9px; background:var(--accent); color:#fff; font-size:18px; }
.send-btn:disabled { background:var(--line-strong); }
.composer-hint { margin-top:7px; color:var(--muted); text-align:center; font-size:10px; }
.composer-error { margin-bottom:6px; color:#f58b93; font-size:11px; }
.activity-bar { display:flex; align-items:center; gap:7px; margin:0 0 7px 3px; color:#b9a3e8; font-size:11px; }
.memory-modal-overlay { position:fixed; inset:0; z-index:10001; display:grid; place-items:center; background:rgb(0 0 0 / .55); backdrop-filter:blur(2px); }
.memory-modal { width:min(560px, 92vw); max-height:80vh; overflow:auto; border:1px solid var(--line-strong); border-radius:12px; background:var(--bg-soft); box-shadow:0 20px 60px rgb(0 0 0 / .4); }
.memory-modal-head { display:flex; align-items:center; gap:10px; padding:14px 16px; border-bottom:1px solid var(--line); }
.memory-modal-head strong { flex:1; min-width:0; color:var(--text); font-size:14px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.memory-modal-verdict { flex:none; padding:2px 8px; border-radius:5px; color:var(--text); font-size:10px; white-space:nowrap; }
.memory-modal-close { flex:none; width:24px; height:24px; padding:0; border:0; border-radius:5px; background:transparent; color:var(--muted); font-size:18px; line-height:20px; }
.memory-modal-close:hover { background:rgb(255 255 255 / .09); color:var(--text); }
.memory-modal-body { padding:14px 16px; }
.memory-modal-section { margin-bottom:14px; }
.memory-modal-section:last-child { margin-bottom:0; }
.memory-modal-section label { display:block; margin-bottom:5px; color:var(--muted); font-size:10px; letter-spacing:.05em; text-transform:uppercase; }
.memory-modal-expr { margin:0; padding:10px; border:1px solid var(--line); border-radius:7px; background:rgb(255 255 255 / .025); color:#c7d8ee; white-space:pre-wrap; overflow-wrap:break-word; word-break:break-all; font:12px/1.5 var(--font-mono); }
.memory-modal-text { margin:0; color:var(--text); font-size:12px; line-height:1.6; overflow-wrap:break-word; }
.memory-modal-metrics { display:flex; flex-wrap:wrap; gap:6px; }
.memory-modal-metrics span { padding:4px 8px; border-radius:5px; color:#a9e9bd; background:rgb(67 209 122 / .1); font:11px var(--font-mono); }
.memory-modal-error { margin:0; padding:10px; border:1px solid rgb(239 107 115 / .25); border-radius:7px; background:rgb(239 107 115 / .06); color:#e89a9f; white-space:pre-wrap; overflow-wrap:break-word; font:11px/1.5 var(--font-mono); }
.memory-modal-observations { display:flex; flex-direction:column; gap:5px; }
.memory-modal-observation { display:flex; align-items:center; gap:8px; padding:5px 8px; border:1px solid var(--line); border-radius:6px; font-size:11px; }
.memory-modal-observation span:first-child { color:var(--text); font-weight:550; }
.memory-modal-obs-verdict { padding:1px 6px; border-radius:4px; font-size:10px; }
.memory-modal-obs-time { margin-left:auto; color:var(--muted); font:10px var(--font-mono); }
.memory-modal-meta { display:flex; flex-wrap:wrap; gap:8px 14px; padding-top:10px; border-top:1px solid var(--line); color:var(--muted); font:10px var(--font-mono); }

/* ═══ 子 Tab ═══ */
.agent-subtabs { display:flex; gap:4px; margin-bottom:14px; }
.agent-subtabs button { padding:7px 14px; border:1px solid var(--line); border-radius:8px; background:transparent; color:var(--muted); font-size:12px; cursor:pointer; transition:all .12s; }
.agent-subtabs button:hover { color:var(--text); border-color:var(--line-strong); }
.agent-subtabs button.active { color:var(--text); border-color:var(--accent); background:rgb(79 140 255 / .14); }

/* ═══ 因子实验室 ═══ */
.lab-panel { display:grid; grid-template-columns:360px minmax(0,1fr); gap:16px; height:calc(100% - 40px); }
.lab-left { display:flex; flex-direction:column; padding:16px; border:1px solid var(--line); border-radius:12px; background:var(--card); }
.lab-left h3 { margin:0 0 10px; font-size:13px; color:var(--text); }
.lab-editor { width:100%; resize:vertical; border:1px solid var(--line-strong); border-radius:8px; background:var(--bg-soft); color:#c7d8ee; padding:10px; font:12px/1.55 var(--font-mono); outline:none; }
.lab-editor:focus { border-color:var(--accent); }
.lab-options { display:flex; flex-direction:column; gap:8px; margin-top:10px; }
.lab-options label { display:flex; align-items:center; gap:6px; color:var(--muted); font-size:11px; }
.lab-options input[type="text"] { flex:1; padding:4px 7px; border:1px solid var(--line); border-radius:5px; background:var(--bg-soft); color:var(--text); font-size:11px; }
.lab-options input[type="date"] { padding:3px 6px; border:1px solid var(--line); border-radius:5px; background:var(--bg-soft); color:var(--text); font-size:11px; }
.lab-date-row { display:flex; gap:8px; }
.lab-date-row label { flex:1; }
.lab-funda { cursor:pointer; }
.lab-run-btn { margin-top:10px; padding:9px; border:0; border-radius:8px; background:var(--accent); color:#fff; font-size:13px; cursor:pointer; transition:opacity .12s; }
.lab-run-btn:disabled { opacity:.5; cursor:not-allowed; }
.lab-error { margin-top:8px; color:#f58b93; font-size:11px; }
.lab-right { overflow:auto; padding:16px; border:1px solid var(--line); border-radius:12px; background:var(--card); }
.lab-empty { display:grid; place-items:center; height:100%; color:var(--muted); font-size:12px; text-align:center; line-height:1.7; }
.lab-loading { display:grid; place-items:center; height:100%; color:#b9a3e8; font-size:12px; }
.lab-results { display:flex; flex-direction:column; gap:12px; }
.lab-result-card { padding:12px 14px; border:1px solid var(--line); border-radius:10px; background:rgb(255 255 255 / .018); }
.lab-result-card.ok { border-color:rgb(67 209 122 / .22); }
.lab-result-card.bad { border-color:rgb(239 107 115 / .22); }
.lab-result-head { display:flex; align-items:center; gap:8px; margin-bottom:8px; font-size:12px; }
.lab-result-head strong { color:var(--text); font-size:13px; }
.lab-pass { padding:2px 8px; border-radius:5px; font-size:10px; font-weight:600; }
.lab-pass.pass { color:#72d69a; background:rgb(67 209 122 / .12); }
.lab-pass.fail { color:#ef8b92; background:rgb(239 107 115 / .12); }
.lab-daterange { margin-left:auto; color:var(--muted); font:10px var(--font-mono); }
.lab-result-error { color:#e89a9f; font-size:11px; }
.lab-metrics { display:grid; grid-template-columns:repeat(auto-fill,minmax(120px,1fr)); gap:6px; }
.lab-metric { display:flex; align-items:center; justify-content:space-between; padding:5px 8px; border-radius:6px; background:rgb(255 255 255 / .035); }
.lab-metric-label { color:var(--muted); font-size:10px; }
.lab-metric-value { color:#a9e9bd; font:11px var(--font-mono); }
.lab-rules { margin-top:8px; }
.lab-rules summary { cursor:pointer; color:var(--muted); font-size:11px; }
.lab-rule { display:flex; align-items:center; gap:8px; padding:4px 0; font:11px var(--font-mono); color:var(--muted); }
.lab-rule.pass span:first-child { color:#a9e9bd; }
.lab-rule.fail span:first-child { color:#e89a9f; }

/* ═══ 因子库管理 ═══ */
.lib-panel { height:calc(100% - 40px); overflow:auto; padding:16px; border:1px solid var(--line); border-radius:12px; background:var(--card); }
.lib-head { display:flex; align-items:center; gap:10px; margin-bottom:14px; }
.lib-tabs { display:flex; gap:4px; }
.lib-tabs button { padding:5px 12px; border:1px solid var(--line); border-radius:7px; background:transparent; color:var(--muted); font-size:11px; cursor:pointer; }
.lib-tabs button:hover { color:var(--text); }
.lib-tabs button.active { color:var(--text); border-color:var(--accent); background:rgb(79 140 255 / .14); }
.lib-count { color:var(--muted); font-size:11px; }
.lib-refresh { margin-left:auto; padding:4px 10px; border:1px solid var(--line); border-radius:6px; background:transparent; color:var(--muted); font-size:11px; cursor:pointer; }
.lib-refresh:hover { color:var(--text); border-color:var(--line-strong); }
.lib-error { color:#f58b93; font-size:12px; }
.lib-loading { color:var(--muted); font-size:12px; padding:20px 0; text-align:center; }
.lib-empty { color:var(--muted); font-size:12px; padding:20px 0; text-align:center; }
.lib-table { width:100%; border-collapse:collapse; }
.lib-table th { text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); color:var(--muted); font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:.05em; }
.lib-table td { padding:8px 10px; border-bottom:1px solid var(--line); font-size:12px; }
.lib-table tr:hover td { background:rgb(79 140 255 / .06); }
.lib-fid { color:var(--accent); font:11px var(--font-mono); cursor:pointer; }
.lib-table td:nth-child(2) { cursor:pointer; }
.lib-time { color:var(--muted); font:10px var(--font-mono); }
.lib-del { padding:3px 8px; border:1px solid rgb(239 107 115 / .3); border-radius:5px; background:transparent; color:#ef8b92; font-size:10px; cursor:pointer; }
.lib-del:hover { background:rgb(239 107 115 / .1); }
.lib-status { padding:2px 7px; border-radius:4px; font-size:10px; }
.status-active { color:#72d69a; background:rgb(67 209 122 / .12); }
.status-archived { color:var(--muted); background:rgb(255 255 255 / .05); }
.ic-pos { color:#72d69a; }
.ic-neg { color:#ef8b92; }

/* ═══ 因子详情弹窗 ═══ */
.factor-modal-overlay { position:fixed; inset:0; z-index:10001; display:grid; place-items:center; background:rgb(0 0 0 / .55); backdrop-filter:blur(2px); }
.factor-modal { width:min(600px, 92vw); max-height:80vh; overflow:auto; border:1px solid var(--line-strong); border-radius:12px; background:var(--bg-soft); box-shadow:0 20px 60px rgb(0 0 0 / .4); }
.factor-modal-head { display:flex; align-items:center; gap:10px; padding:14px 16px; border-bottom:1px solid var(--line); }
.factor-modal-head strong { flex:1; min-width:0; color:var(--text); font-size:14px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.factor-modal-head code { color:var(--muted); font:11px var(--font-mono); }
.factor-modal-close { flex:none; width:24px; height:24px; padding:0; border:0; border-radius:5px; background:transparent; color:var(--muted); font-size:18px; line-height:20px; cursor:pointer; }
.factor-modal-close:hover { background:rgb(255 255 255 / .09); color:var(--text); }
.factor-modal-body { padding:14px 16px; }
.factor-modal-section { margin-bottom:14px; }
.factor-modal-section:last-child { margin-bottom:0; }
.factor-modal-section label { display:block; margin-bottom:5px; color:var(--muted); font-size:10px; letter-spacing:.05em; text-transform:uppercase; }
.factor-modal-expr { margin:0; padding:10px; border:1px solid var(--line); border-radius:7px; background:rgb(255 255 255 / .025); color:#c7d8ee; white-space:pre-wrap; overflow-wrap:break-word; word-break:break-all; font:12px/1.5 var(--font-mono); }
.factor-modal-registry { margin:0; padding:10px; border:1px solid var(--line); border-radius:7px; background:rgb(255 255 255 / .025); color:#aebbd2; white-space:pre-wrap; overflow-wrap:break-word; font:11px/1.5 var(--font-mono); max-height:300px; overflow:auto; }
.factor-modal-meta { display:flex; flex-wrap:wrap; gap:8px 14px; padding-top:10px; border-top:1px solid var(--line); color:var(--muted); font:10px var(--font-mono); }

@media (max-width: 820px) {
  .agent-page { height:auto; min-height:calc(100vh - 110px); }
  .agent-shell { grid-template-columns:1fr; min-height:calc(100vh - 130px); }
  .agent-sidebar { max-height:180px; border-right:0; border-bottom:1px solid var(--line); }
  .sidebar-footer { display:none; }
  .session-list { display:flex; overflow-x:auto; }
  .session-item { min-width:180px; }
  .agent-thread { padding:20px 14px; }
  .composer-wrap { width:calc(100% - 28px); }
}
</style>
