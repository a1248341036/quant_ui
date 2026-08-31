<template>
  <div class="ml-panel">
    <div class="ml-config">
      <h3>ML 组合训练（walk-forward 时间隔离）</h3>
      <div class="ml-form">
        <label>模式
          <select v-model="ml.form.modes" multiple class="ml-input" size="2">
            <option value="technical">日线技术</option>
            <option value="fundamental">基本面</option>
          </select>
        </label>
        <label>模型
          <select v-model="ml.form.model" class="ml-input">
            <option value="both">Ridge + LGBM</option>
            <option value="ridge">Ridge</option>
            <option value="lgbm">LightGBM</option>
          </select>
        </label>
        <label>持有天数 <input type="number" v-model.number="ml.form.label_days" class="ml-input ml-num"></label>
        <label>训练月数 <input type="number" v-model.number="ml.form.train_months" class="ml-input ml-num"></label>
        <label>折长(月) <input type="number" v-model.number="ml.form.step_months" class="ml-input ml-num"></label>
        <label>去重阈值 <input type="number" step="0.05" v-model.number="ml.form.max_corr" class="ml-input ml-num"></label>
        <label>隔离模式
          <select v-model="ml.form.isolation" class="ml-input">
            <option value="holdout">留出测试（推荐）</option>
            <option value="strict">严格隔离</option>
          </select>
        </label>
        <label class="ml-check"><input type="checkbox" v-model="ml.form.no_candidate"> 只用正式库</label>
        <label class="ml-check"><input type="checkbox" v-model="ml.form.no_gate"> 跳过 engine_gate</label>
        <button class="send-btn" :disabled="ml.starting || ml.anyRunning" @click="startMl">{{ ml.anyRunning ? '训练中…' : '开始训练' }}</button>
      </div>
      <div v-if="ml.error" class="ml-error">{{ ml.error }}</div>
    </div>
    <div class="ml-list">
      <h4>训练历史</h4>
      <table class="lib-table">
        <thead><tr><th>训练</th><th>状态</th><th>OOS IC</th><th>OOS ICIR</th><th>折数</th><th>特征</th><th>gate</th><th>时间隔离</th><th></th></tr></thead>
        <tbody>
          <tr v-for="t in ml.list" :key="t.train_id" :class="{active: ml.selected===t.train_id}" @click="viewMl(t.train_id)">
            <td>{{ t.train_id }}</td>
            <td>{{ t.status }}</td>
            <td>{{ fmtNum(t.oos_ic_mean) }}</td>
            <td>{{ fmtNum(t.oos_ic_ir) }}</td>
            <td>{{ t.n_folds ?? '—' }}</td>
            <td>{{ t.n_features ?? '—' }}</td>
            <td>{{ t.gate_passed === true ? '通过' : (t.gate_passed === false ? '未过' : '—') }}</td>
            <td>{{ t.time_isolation || '—' }}</td>
            <td><button v-if="t.status==='running'" class="ml-stop" @click.stop="stopMl(t.train_id)">停止</button></td>
          </tr>
        </tbody>
      </table>
    </div>
    <div v-if="ml.detail" class="ml-detail">
      <h4>训练详情 · {{ ml.selected }} · {{ ml.detail.status }}</h4>
      <pre v-if="ml.detail.status==='running'" class="ml-log">{{ (ml.detail.progress_tail || []).slice(-12).join('\n') || '（等待输出…）' }}</pre>
      <template v-if="ml.detail.report">
        <div class="ml-summary">
          <span>时间隔离: <b>{{ ml.detail.report.time_isolation }}</b></span>
          <span>OOS IC(混合): <b>{{ fmtNum((ml.detail.report.oos_ic_blended||{}).ic_mean) }}</b></span>
          <span>ICIR: <b>{{ fmtNum((ml.detail.report.oos_ic_blended||{}).ic_ir) }}</b></span>
          <span>gate: <b>{{ (ml.detail.report.gate||{}).passed ? '通过' : '未过' }}</b>
            <span v-if="!(ml.detail.report.gate||{}).passed">（{{ ((ml.detail.report.gate||{}).fail_reasons||[]).join('、') }}）</span></span>
        </div>
        <h5>折级 OOS 表现</h5>
        <div v-for="(rows, model) in ml.detail.report.fold_metrics" :key="model" class="ml-fold">
          <strong>{{ model }}</strong>
          <table class="lib-table">
            <thead><tr><th>OOS 起</th><th>OOS 止</th><th>IC 均值</th><th>ICIR</th><th>天数</th><th>多空日差</th></tr></thead>
            <tbody>
              <tr v-for="(row, i) in rows" :key="i">
                <td>{{ row.oos_start }}</td><td>{{ row.oos_end }}</td>
                <td>{{ fmtNum(row.ic_mean) }}</td><td>{{ fmtNum(row.ic_ir) }}</td>
                <td>{{ row.n_days }}</td><td>{{ fmtNum(row.long_short_daily_spread) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <h5>特征（{{ (ml.detail.report.feature_names||[]).length }}）与剔除（{{ (ml.detail.report.dropped||[]).length }}）</h5>
        <div class="ml-features">{{ (ml.detail.report.feature_names||[]).join(' · ') }}</div>
        <div v-for="d in ml.detail.report.dropped" :key="d.name" class="ml-drop">− {{ d.name }}（{{ d.library }}）：{{ d.reason }}</div>
        <h5>衰减对照（挖掘期 IC vs OOS IC）</h5>
        <table class="lib-table">
          <thead><tr><th>因子</th><th>挖掘 IC</th><th>OOS IC</th><th>衰减比</th></tr></thead>
          <tbody>
            <tr v-for="row in ml.detail.report.decay_table" :key="row.name">
              <td>{{ row.name }}</td><td>{{ fmtNum(row.ic_mining) }}</td>
              <td>{{ fmtNum(row.ic_oos) }}</td><td>{{ fmtNum(row.decay_ratio) }}</td>
            </tr>
          </tbody>
        </table>
      </template>
    </div>
  </div>
</template>

<script>
import { api } from '../../utils/api.js'

export default {
  name: 'MlPanel',
  data() {
    return {
      ml: {
        form: { modes: ['technical'], model: 'both', label_days: 5, train_months: 18, step_months: 6, max_corr: 0.6, isolation: 'holdout', no_candidate: false, no_gate: false },
        list: [],
        selected: null,
        detail: null,
        starting: false,
        error: '',
      },
    }
  },
  mounted() {
    this.loadMl()
  },
  beforeUnmount() {
    // 组件随子 tab 切换卸载时停止轮询；再次进入时 mounted 会重启
    if (this._mlTimer) {
      clearInterval(this._mlTimer)
      this._mlTimer = null
    }
  },
  methods: {
    fmtNum(v) {
      if (v === null || v === undefined || Number.isNaN(Number(v))) return '—'
      return Number(v).toFixed(4)
    },
    async loadMl() {
      try {
        this.ml.list = await api('/api/alphaagent/stacking/trainings')
        if (this.ml.list.some(t => t.status === 'running')) this.scheduleMlPoll()
      } catch (e) {
        this.ml.error = e.message
      }
    },
    async startMl() {
      this.ml.starting = true
      this.ml.error = ''
      try {
        const res = await api('/api/alphaagent/stacking/train', { method: 'POST', body: JSON.stringify(this.ml.form) })
        this.ml.selected = res.train_id
        await this.loadMl()
        await this.viewMl(res.train_id)
      } catch (e) {
        this.ml.error = e.message
      } finally {
        this.ml.starting = false
      }
    },
    async viewMl(trainId, silent) {
      try {
        const d = await api('/api/alphaagent/stacking/trainings/' + encodeURIComponent(trainId))
        if (!silent || this.ml.selected === trainId) {
          this.ml.selected = trainId
          this.ml.detail = d
        }
        if (d.status === 'running') this.scheduleMlPoll()
      } catch (e) {
        if (!silent) this.ml.error = e.message
      }
    },
    scheduleMlPoll() {
      if (this._mlTimer) return
      this._mlTimer = setInterval(async () => {
        await this.loadMl()
        if (this.ml.selected) await this.viewMl(this.ml.selected, true)
        if (!(this.ml.list || []).some(t => t.status === 'running')) {
          clearInterval(this._mlTimer)
          this._mlTimer = null
        }
      }, 5000)
    },
    async stopMl(trainId) {
      try {
        await api('/api/alphaagent/stacking/trainings/' + encodeURIComponent(trainId) + '/stop', { method: 'POST' })
        await this.loadMl()
      } catch (e) {
        this.ml.error = e.message
      }
    },
  },
}
</script>
