<template>
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
</template>

<script>
import { agentStore } from '../../store/alphaagent.js'
import { memoryVerdictLabel, metricLabel, formatMetricValue, formatTime } from '../../utils/alphaagent.js'

export default {
  name: 'MemoryDetailModal',
  data() {
    return {
      agent: agentStore,
    }
  },
  methods: {
    memoryVerdictLabel,
    metricLabel,
    formatMetricValue,
    formatTime,
  },
}
</script>
