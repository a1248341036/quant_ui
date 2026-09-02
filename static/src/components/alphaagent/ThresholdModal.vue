<template>
  <Teleport to="body">
    <div v-if="agent.showThresholdModal" class="threshold-modal-overlay" @click="agent.showThresholdModal = false">
      <div class="threshold-modal" @click.stop>
        <div class="threshold-modal-head">
          <div>
            <strong>门槛配置</strong>
            <span class="threshold-modal-mode">{{ agent.inferredModeLabel }}</span>
            <span v-if="agent.researchSpecCustom" class="research-spec-custom">已自定义</span>
            <span v-if="agent.researchSpecDirty" class="research-spec-dirty">未保存</span>
            <span v-if="agent.specError" class="research-spec-error">{{ agent.specError }}</span>
            <span v-if="agent.specSavedAt" class="research-spec-saved">已保存 {{ formatTime(agent.specSavedAt) }}</span>
          </div>
          <button class="threshold-modal-close" @click="agent.showThresholdModal = false">×</button>
        </div>
        <div class="threshold-modal-body">
          <p class="threshold-hint">保存后写入该模式门槛文件（增量覆盖），挖掘 / 晋升 / CLI 全链路生效。恢复默认会删除自定义覆盖。</p>
          <p v-if="agent.specLoading" class="threshold-loading">正在加载门槛数据…</p>

          <section class="threshold-group">
            <h4>① 训练集评估门槛（train_screen）</h4>
            <div class="threshold-grid">
              <label>Train |IC| ≥
                <input type="number" step="0.001" class="threshold-input" v-model.number="thresholdDraft.evaluation_policy.min_train_abs_ic">
              </label>
              <label>Train |ICIR| ≥
                <input type="number" step="0.01" class="threshold-input" v-model.number="thresholdDraft.evaluation_policy.min_train_icir">
              </label>
              <label>Coverage ≥
                <input type="number" step="0.01" class="threshold-input" v-model.number="thresholdDraft.evaluation_policy.min_train_coverage">
              </label>
            </div>
          </section>

          <section class="threshold-group">
            <h4>② 验证集门槛（validation）</h4>
            <div class="threshold-grid">
              <label>Val |IC| ≥
                <input type="number" step="0.001" class="threshold-input" v-model.number="thresholdDraft.evaluation_policy.min_val_abs_ic">
              </label>
              <label>Val/Train 保留比 ≥
                <input type="number" step="0.01" class="threshold-input" v-model.number="thresholdDraft.evaluation_policy.min_val_ic_retention_ratio">
              </label>
              <label>截面自相关 ≥（换手约束）
                <input type="number" step="0.01" class="threshold-input" v-model.number="thresholdDraft.evaluation_policy.min_cs_autocorr">
              </label>
            </div>
          </section>

          <section class="threshold-group">
            <h4>③ 候选池门槛（stage_one 海选）</h4>
            <div class="threshold-grid">
              <label>|IC| ≥
                <input type="number" step="0.001" class="threshold-input" v-model.number="thresholdDraft.delivery_policy.candidate.min_abs_ic">
              </label>
              <label>|ICIR| >
                <input type="number" step="0.01" class="threshold-input" v-model.number="thresholdDraft.delivery_policy.candidate.min_icir">
              </label>
              <label>Coverage >
                <input type="number" step="0.01" class="threshold-input" v-model.number="thresholdDraft.delivery_policy.candidate.min_coverage">
              </label>
              <label>最大截面相关 &lt;
                <input type="number" step="0.01" class="threshold-input" v-model.number="thresholdDraft.delivery_policy.candidate.max_abs_corr">
              </label>
              <label>截面自相关 ≥
                <input type="number" step="0.01" class="threshold-input" v-model.number="thresholdDraft.delivery_policy.candidate.min_cs_autocorr">
              </label>
              <label>Val 保留比 ≥
                <input type="number" step="0.01" class="threshold-input" v-model.number="thresholdDraft.delivery_policy.candidate.min_val_ic_retention">
              </label>
            </div>
          </section>

          <section class="threshold-group">
            <h4>④ 正式库门槛（stage_two 精筛）</h4>
            <div class="threshold-grid">
              <label>Train |IC| ≥
                <input type="number" step="0.001" class="threshold-input" v-model.number="thresholdDraft.delivery_policy.production.min_train_abs_ic">
              </label>
              <label>Train |ICIR| ≥
                <input type="number" step="0.01" class="threshold-input" v-model.number="thresholdDraft.delivery_policy.production.min_train_icir">
              </label>
              <label>Val |IC| ≥
                <input type="number" step="0.001" class="threshold-input" v-model.number="thresholdDraft.delivery_policy.production.min_val_abs_ic">
              </label>
              <label>Val 保留比 ≥
                <input type="number" step="0.01" class="threshold-input" v-model.number="thresholdDraft.delivery_policy.production.min_val_ic_retention">
              </label>
              <label>Val 多头端年化超额 ≥
                <input type="number" step="0.01" class="threshold-input" v-model.number="thresholdDraft.delivery_policy.production.min_val_long_excess">
              </label>
              <label>截尾 IC 衰减 ≤
                <input type="number" step="0.01" class="threshold-input" v-model.number="thresholdDraft.delivery_policy.production.max_winsorized_abs_ic_decay">
              </label>
              <label>最大截面相关 &lt;
                <input type="number" step="0.01" class="threshold-input" v-model.number="thresholdDraft.delivery_policy.production.max_abs_corr">
              </label>
            </div>
          </section>

          <section class="threshold-group">
            <h4>⑤ engine_gate 完整回测门禁</h4>
            <div class="threshold-grid">
              <label>启用门禁
                <select class="threshold-input" v-model="thresholdDraft.delivery_policy.production.engine_gate.enabled">
                  <option :value="true">启用（推荐）</option>
                  <option :value="false">停用</option>
                </select>
              </label>
              <label>调仓频率
                <select class="threshold-input" v-model="thresholdDraft.delivery_policy.production.engine_gate.freq">
                  <option value="weekly">weekly（周）</option>
                  <option value="monthly">monthly（月）</option>
                  <option value="daily">daily（日）</option>
                </select>
              </label>
              <label>选股模式
                <select class="threshold-input" v-model="thresholdDraft.delivery_policy.production.engine_gate.selection_mode">
                  <option value="top_pct">动态百分比 top_pct</option>
                  <option value="top_n">固定数量 top_n</option>
                </select>
              </label>
              <label v-if="thresholdDraft.delivery_policy.production.engine_gate.selection_mode === 'top_pct'">选股数（%）
                <input type="number" step="0.001" min="0.0001" class="threshold-input" v-model.number="thresholdDraft.delivery_policy.production.engine_gate.selection_pct">
              </label>
              <label v-else>选股数（只）
                <input type="number" step="1" min="1" class="threshold-input" v-model.number="thresholdDraft.delivery_policy.production.engine_gate.top_n">
              </label>
              <label>门禁资金（元）
                <input type="number" step="10000" min="10000" class="threshold-input" v-model.number="thresholdDraft.delivery_policy.production.engine_gate.capital">
              </label>
              <label>滑点（bps）
                <input type="number" step="0.5" min="0" class="threshold-input" v-model.number="thresholdDraft.delivery_policy.production.engine_gate.slippage_bps">
              </label>
              <label>参与率 ≤
                <input type="number" step="0.01" min="0.001" class="threshold-input" v-model.number="thresholdDraft.delivery_policy.production.engine_gate.max_participation">
              </label>
              <label>净超额年化 ≥
                <input type="number" step="0.01" class="threshold-input" v-model.number="thresholdDraft.delivery_policy.production.engine_gate.min_excess_annual">
              </label>
              <label>超额夏普 ≥
                <input type="number" step="0.05" class="threshold-input" v-model.number="thresholdDraft.delivery_policy.production.engine_gate.min_excess_sharpe">
              </label>
              <label>最大回撤 ≤
                <input type="number" step="0.01" class="threshold-input" v-model.number="thresholdDraft.delivery_policy.production.engine_gate.max_drawdown">
              </label>
              <label>持仓日重叠 ≥
                <input type="number" step="0.01" class="threshold-input" v-model.number="thresholdDraft.delivery_policy.production.engine_gate.min_daily_overlap">
              </label>
              <label>仓位利用率 ≥
                <input type="number" step="0.01" class="threshold-input" v-model.number="thresholdDraft.delivery_policy.production.engine_gate.min_invested_ratio">
              </label>
              <label>日均成交额下限（元）
                <input type="number" step="10000" class="threshold-input" v-model.number="thresholdDraft.delivery_policy.production.engine_gate.min_am20_yuan">
              </label>
            </div>
          </section>

          <section class="threshold-group">
            <h4>⑥ Screener · regime 感知筛选（agent 工具）</h4>
            <div class="threshold-grid">
              <label>启用 Screener
                <select class="threshold-input" v-model="thresholdDraft.delivery_policy.screener.enabled">
                  <option :value="false">停用（默认）</option>
                  <option :value="true">启用</option>
                </select>
              </label>
              <label>Rank IC 回看天数
                <input type="number" step="1" min="3" max="60" class="threshold-input" v-model.number="thresholdDraft.delivery_policy.screener.lookback">
              </label>
              <label>|IC| 下限
                <input type="number" step="0.001" min="0" max="1" class="threshold-input" v-model.number="thresholdDraft.delivery_policy.screener.min_ic">
              </label>
              <label>最大相关性
                <input type="number" step="0.01" min="0" max="1" class="threshold-input" v-model.number="thresholdDraft.delivery_policy.screener.max_corr">
              </label>
              <label>因子族偏好
                <select class="threshold-input" v-model="thresholdDraft.delivery_policy.screener.use_family_boost">
                  <option :value="true">启用</option>
                  <option :value="false">停用</option>
                </select>
              </label>
              <label>ADX 趋势阈值
                <input type="number" step="0.5" min="0" max="100" class="threshold-input" v-model.number="thresholdDraft.delivery_policy.screener.adx_threshold">
              </label>
              <label>均线周期（日）
                <input type="number" step="1" min="10" max="500" class="threshold-input" v-model.number="thresholdDraft.delivery_policy.screener.ma_period">
              </label>
              <label>最小截面数
                <input type="number" step="1" min="1" max="1000" class="threshold-input" v-model.number="thresholdDraft.delivery_policy.screener.min_cross_section">
              </label>
            </div>
          </section>

          <section class="threshold-group">
            <h4>⑥ 搜索策略</h4>
            <div class="threshold-grid">
              <label title="每轮 LLM 并行提出的候选因子数上限（同时渲染进提示词）；上限 24">
                每轮候选数 ≤
                <input type="number" step="1" min="1" max="24" class="threshold-input" v-model.number="thresholdDraft.search_policy.max_candidates_per_round">
              </label>
            </div>
            <p class="threshold-search-hint">每轮实际并行评估还受 composer 的"并发"（评估并发度）与 max_tool_calls_per_round 约束；批量越大，单轮耗时与 token 消耗越高。</p>
          </section>

          <details class="threshold-advanced">
            <summary>高级（完整 JSON，含搜索/审查/交互/记忆策略）</summary>
            <textarea v-model="agent.researchSpecText" spellcheck="false" class="threshold-json" aria-label="ResearchSpec 完整 JSON"></textarea>
          </details>
        </div>
        <div class="threshold-modal-foot">
          <button class="threshold-btn-reset" :disabled="agent.researchSpecSaving" @click="agent.resetResearchSpec">恢复默认</button>
          <span class="threshold-spacer"></span>
          <button class="threshold-btn-cancel" @click="agent.showThresholdModal = false">取消</button>
          <button class="threshold-btn-save" :disabled="agent.researchSpecSaving" @click="agent.saveThresholds">
            {{ agent.researchSpecSaving ? '保存中…' : '保存门槛' }}
          </button>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<script>
import { agentStore } from '../../store/alphaagent.js'
import { formatTime } from '../../utils/alphaagent.js'

export default {
  name: 'ThresholdModal',
  data() {
    return {
      agent: agentStore,
    }
  },
  computed: {
    thresholdDraft() {
      return agentStore.thresholdDraft
    },
  },
  methods: {
    formatTime,
  },
}
</script>
