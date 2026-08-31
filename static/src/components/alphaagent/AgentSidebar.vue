<template>
  <aside class="agent-sidebar">
    <div class="sidebar-head">
      <div>
        <div class="eyebrow">RESEARCH AGENT</div>
        <h2>AlphaAgent</h2>
      </div>
      <button class="icon-btn" title="新建会话" @click="agent.newRun">+</button>
    </div>

    <button class="new-run" @click="agent.newRun">
      <span>＋</span> 新建研究任务
    </button>

    <div class="session-label">
      <span>{{ agent.showArchived ? '已归档任务' : '最近任务' }}</span>
      <span class="label-actions">
        <button v-if="agent.showArchived && agent.runs.length" class="archived-toggle danger" title="删除全部已归档任务" @click="agent.deleteAllArchived">一键删除</button>
        <button class="archived-toggle" :title="agent.showArchived ? '返回最近任务' : '查看已归档任务'" @click="agent.toggleArchived">
          {{ agent.showArchived ? '返回' : '归档' }}
        </button>
      </span>
    </div>
    <div class="session-list">
      <div
        v-for="run in agent.runs"
        :key="run.run_id"
        class="session-item"
        :class="{ active: agent.current?.run_id === run.run_id }"
      >
        <div class="session-select" role="button" tabindex="0" @click="agent.selectAgentRun(run)" @keydown.enter.prevent="agent.selectAgentRun(run)" @keydown.space.prevent="agent.selectAgentRun(run)">
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

    <section class="memory-panel" :class="{ collapsed: agent.memoryCollapsed }" aria-label="长期研究记忆">
      <div class="memory-head" @click="agent.memoryCollapsed = !agent.memoryCollapsed" role="button" tabindex="0" @keydown.enter.prevent="agent.memoryCollapsed = !agent.memoryCollapsed">
        <span class="memory-head-label"><span class="memory-caret">{{ agent.memoryCollapsed ? '▸' : '▾' }}</span>长期研究记忆</span>
        <b>{{ agent.memory.length }}</b>
      </div>
      <template v-if="!agent.memoryCollapsed">
        <div v-if="agent.memory.length" class="memory-items">
          <div v-for="entry in agent.memory" :key="entry.id" class="memory-item" :class="'memory-' + entry.verdict" role="button" tabindex="0" @click="agent.memoryDetail = entry" @keydown.enter.prevent="agent.memoryDetail = entry">
            <strong>{{ memoryVerdictLabel(entry.verdict) }} · {{ entry.factor_name }}</strong>
            <small>{{ entry.conclusion }}</small>
          </div>
        </div>
        <small v-else>评估结果会在这里沉淀为跨会话研究记忆</small>
      </template>
    </section>

    <div class="sidebar-footer">
      <span class="status-dot status-running"></span>
      <span>Codex 当前模型配置</span>
    </div>

    <Teleport to="body">
      <div
        v-if="agent.menuRun"
        class="session-menu-popover"
        :style="{ top: agent.menuPosition.top + 'px', left: agent.menuPosition.left + 'px' }"
        @click.stop
      >
        <button @click="agent.pinRun(agent.menuRun)">{{ agent.menuRun.pinned ? '取消置顶' : '置顶' }}</button>
        <button @click="agent.branchRun(agent.menuRun)">新建分支</button>
        <button @click="beginRename(agent.menuRun)">重命名</button>
        <button class="archive-action" @click="agent.archiveRun(agent.menuRun)">归档</button>
        <button v-if="agent.menuRun.archived" class="delete-action" @click="agent.deleteRun(agent.menuRun)">删除</button>
      </div>
    </Teleport>

    <Teleport to="body">
      <form
        v-if="agent.renameRun"
        class="session-rename-popover"
        :style="{ top: agent.renamePosition.top + 'px', left: agent.renamePosition.left + 'px' }"
        @click.stop
        @submit.prevent="agent.commitRename(agent.renameRun)"
      >
        <input
          ref="renameInput"
          v-model="agent.renameTitle"
          aria-label="会话名称"
          @keydown.esc.stop.prevent="agent.cancelRename"
        >
        <button type="submit" title="保存会话名称" aria-label="保存会话名称">✓</button>
      </form>
    </Teleport>
  </aside>
</template>

<script>
import { agentStore } from '../../store/alphaagent.js'
import { runTitle, formatTime, memoryVerdictLabel } from '../../utils/alphaagent.js'

export default {
  name: 'AgentSidebar',
  data() {
    return {
      agent: agentStore,
    }
  },
  mounted() {
    this._closeSessionMenu = () => {
      agentStore.menuRunId = ''
      if (agentStore.renameRunId) agentStore.cancelRename()
    }
    document.addEventListener('click', this._closeSessionMenu)
  },
  beforeUnmount() {
    document.removeEventListener('click', this._closeSessionMenu)
  },
  methods: {
    runTitle,
    formatTime,
    memoryVerdictLabel,
    toggleMenu(runId, event) {
      if (agentStore.menuRunId === runId) {
        agentStore.menuRunId = ''
        return
      }
      const rect = event.currentTarget.getBoundingClientRect()
      const width = 128
      const height = 178
      agentStore.menuPosition = {
        top: Math.max(8, Math.min(rect.top, window.innerHeight - height - 8)),
        left: Math.max(8, Math.min(rect.right - width, window.innerWidth - width - 8)),
      }
      agentStore.menuRunId = runId
    },
    beginRename(run) {
      agentStore.menuRunId = ''
      agentStore.renameRunId = run.run_id
      agentStore.renameTitle = run.title || runTitle(run)
      const titleEl = document.querySelector('[data-run-title="' + run.run_id + '"]')
      const rect = titleEl?.getBoundingClientRect()
      agentStore.renamePosition = {
        top: Math.max(8, Math.min(rect?.top || 8, window.innerHeight - 42)),
        left: Math.max(8, Math.min(rect?.left || 8, window.innerWidth - 268)),
      }
      this.$nextTick(() => {
        const input = this.$refs.renameInput
        input?.focus()
        input?.select()
      })
    },
  },
}
</script>
