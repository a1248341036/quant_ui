<template>
  <div class="ml-panel">
    <!-- ── 状态条：当前训练状态 + 主行动 ── -->
    <div class="mlv-status">
      <span class="mlv-dot" :class="{ running: anyRunning, ok: !anyRunning && latest }"></span>
      <div class="mlv-status-text">
        <b>{{ anyRunning ? '训练中 · ' + runningTrain.train_id : latest ? '空闲 · 最新 ' + latest.train_id : '尚未训练' }}</b>
        <span v-if="anyRunning && runningTail" class="mlv-sub">{{ runningTail }}</span>
        <span v-else-if="latest" class="mlv-sub">{{ paramSummary(latest) }}</span>
        <span v-else class="mlv-sub">配置参数后开始第一次 walk-forward 组合训练</span>
      </div>
      <div class="mlv-actions">
        <button class="mlv-ghost" @click="showConfig = !showConfig">{{ showConfig ? '收起配置 ▴' : '训练配置 ▾' }}</button>
        <button v-if="anyRunning" class="mlv-stop" @click="stopMl(runningTrain.train_id)">停止</button>
        <button class="mlv-primary" :disabled="ml.starting || anyRunning" @click="startMl">{{ anyRunning ? '训练中…' : (latest ? '再次训练' : '开始训练') }}</button>
      </div>
    </div>

    <!-- ── 训练配置（折叠） ── -->
    <div v-show="showConfig" class="ml-config">
      <div class="ml-form">
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
        <label class="ml-check" title="按置换贡献降序取 Top-k 逐级重训（成本 ≈ 2n 次拟合，训练时间明显变长）。输出'子集规模 vs OOS IC'曲线，定位边际收益归零的最优因子数。"><input type="checkbox" v-model="ml.form.subset_curve"> 累积子集曲线</label>
        <span class="ml-gate-mode" title="因子池已是统一大库（不分技术/基本面）；engine_gate 档位自动跟随持有天数：≤7 天→技术档（周调仓·严门槛），>7 天→基本面档（月调仓·松门槛）。">gate 档位：{{ gateModeLabel }}（自动）</span>
      </div>
      <div v-if="ml.error" class="ml-error">{{ ml.error }}</div>
    </div>

    <!-- ── 空状态引导 ── -->
    <div v-if="!ml.list.length && !ml.loadingList" class="mlv-empty">
      <p>还没有训练记录。ML 组合会用因子库中的全部因子做 walk-forward 时间隔离训练，</p>
      <p>产出混合 OOS 分数并通过 engine_gate 可交易性裁决——这是检验"因子库整体是否有真 alpha"的最终考场。</p>
      <button class="mlv-primary" @click="showConfig = true; $nextTick(() => startMl())">开始第一次训练</button>
    </div>

    <template v-if="ml.list.length">
      <!-- ── 最新一次成功训练的核心指标 ── -->
      <div v-if="latest" class="metrics-cards">
        <div class="metrics-card" title="混合模型 OOS IC 均值（时间隔离，挖掘期外）">
          <b :class="latest.oos_ic_mean >= 0 ? 'ic-pos' : 'ic-neg'">{{ fmtNum(latest.oos_ic_mean) }}</b><span>最新 OOS IC</span>
        </div>
        <div class="metrics-card" title="混合模型 OOS ICIR">
          <b>{{ fmtNum(latest.oos_ic_ir) }}</b><span>OOS ICIR</span>
        </div>
        <div class="metrics-card" title="engine_gate 可交易性裁决（OOS 段周调仓）">
          <b :class="latest.gate_passed ? 'ic-pos' : 'ic-neg'">{{ latest.gate_passed == null ? '—' : (latest.gate_passed ? '通过' : '未过') }}</b><span>engine_gate</span>
        </div>
        <div class="metrics-card" title="gate 超额年化">
          <b>{{ gateOf(latest, 'excess_annual', true) }}</b><span>gate 超额年化</span>
        </div>
        <div class="metrics-card" title="进入模型的特征数 / walk-forward 折数">
          <b>{{ latest.n_features ?? '—' }}<i>/</i>{{ latest.n_folds ?? '—' }}</b><span>特征 / 折数</span>
        </div>
      </div>

      <!-- ── 跨训练趋势 ── -->
      <div v-if="trendRows.length >= 2" class="mlv-block">
        <div class="mlv-block-head">
          <h4>训练趋势</h4>
          <span class="summary-facet-hint" title="每次训练的 OOS IC / ICIR。看组合 alpha 随因子库扩大是在增强还是衰减。">ⓘ</span>
        </div>
        <div id="ml-trend-chart" class="metrics-chart" style="height: 210px"></div>
      </div>

      <!-- ── 历史训练（点击行展开详情；数值列点击排序） ── -->
      <div class="mlv-block">
        <div class="mlv-block-head">
          <h4>训练历史</h4>
          <span class="mlv-sub">点击行展开权重 / 折级 / 衰减详情 · 点击表头按该列排序</span>
        </div>
        <table class="lib-table">
          <thead>
            <tr>
              <th>训练</th>
              <th v-for="col in historyColumns" :key="col.key" :class="{ sortable: true, active: historySortKey === col.key }"
                  @click="toggleHistorySort(col.key)" :title="col.title || ''">
                {{ col.label }}<span v-if="historySortKey === col.key">{{ historySortDir > 0 ? ' ▲' : ' ▼' }}</span>
              </th>
              <th>gate</th>
              <th title="衰减保留比：decay_table 各因子 OOS IC / 挖掘 IC 的均值（跨训练可比）">衰减保留</th>
              <th title="全部折中最差的 OOS IC（稳定性下界）">最差折</th>
              <th title="分模型折均 OOS IC">R/L IC</th>
              <th title="被相关性去重剔除的因子数">剔除</th>
            </tr>
          </thead>
          <tbody>
            <template v-for="t in historySorted" :key="t.train_id">
              <tr :class="{active: ml.selected===t.train_id}" @click="toggleDetail(t.train_id)">
                <td class="lib-fid">{{ fmtTrainId(t.train_id) }}</td>
                <td :class="icClass(t.oos_ic_mean)"><strong>{{ fmtNum(t.oos_ic_mean) }}</strong></td>
                <td>{{ fmtNum(t.oos_ic_ir) }}</td>
                <td :class="{ neg: (t.gate_excess_annual ?? 0) < 0 }">{{ pct(t.gate_excess_annual) }}</td>
                <td>{{ num2(t.gate_excess_sharpe) }}</td>
                <td :class="{ neg: (t.gate_max_drawdown ?? 0) > 0.4 }">{{ pct(t.gate_max_drawdown) }}</td>
                <td>{{ pct(t.gate_daily_overlap) }}</td>
                <td>{{ pct(t.gate_daily_turnover) }}</td>
                <td>{{ t.n_folds ?? '—' }}</td>
                <td>{{ t.n_features ?? '—' }}<span v-if="t.n_dropped" class="mlv-sub">−{{ t.n_dropped }}</span></td>
                <td>{{ (t.model || '—').toUpperCase() }}</td>
                <td>{{ t.label_days ? t.label_days + 'd' : '—' }}</td>
                <td><span class="lib-status" :class="t.gate_passed === true ? 'status-completed' : (t.gate_passed === false ? 'mlv-bad' : '')">{{ gateText(t) }}</span></td>
                <td><span class="mlv-sub">{{ isolationShort(t) }}</span></td>
                <td>
                  <span v-if="t.model_ic && t.model_ic.ridge != null" class="mlv-sub">{{ fmtNum(t.model_ic.ridge) }}</span>
                  <span v-else class="mlv-sub">—</span>
                  <span class="mlv-sub"> / </span>
                  <span v-if="t.model_ic && t.model_ic.lgbm != null" class="mlv-sub">{{ fmtNum(t.model_ic.lgbm) }}</span>
                  <span v-else class="mlv-sub">—</span>
                </td>
              </tr>
              <tr v-if="ml.selected === t.train_id && ml.detail && ml.detail.train_id === t.train_id" class="mlv-expand-row">
                <td colspan="15">
                  <div class="mlv-expand">
                    <pre v-if="ml.detail.status==='running'" class="ml-log">{{ (ml.detail.progress_tail || []).slice(-8).join('\n') || '（等待输出…）' }}</pre>
                    <template v-if="ml.detail.report">
                      <div class="mlv-expand-head">
                        <span class="mlv-sub">{{ ml.detail.report.time_isolation }}</span>
                        <span class="mlv-sub">训练窗口终点（mining_end）: {{ ml.detail.report.mining_end || '—' }}</span>
                        <span v-if="(ml.detail.report.gate||{}).passed === false" class="mlv-bad">
                          未过原因：{{ ((ml.detail.report.gate||{}).fail_reasons||[]).join('、') }}
                        </span>
                      </div>
                      <div v-if="gateCards.length" class="metrics-cards">
                        <div v-for="c in gateCards" :key="c.label" class="metrics-card" :title="c.title"><b>{{ c.value }}</b><span>{{ c.label }}</span></div>
                      </div>
                      <template v-if="weightKinds.length">
                        <div class="mlv-block-head">
                          <h5>特征权重 Top15（跨折归一平均）</h5>
                          <span class="mlv-sub">组合 OOS IC {{ fmtNum(blendedIc) }} vs 最强单因子 {{ fmtNum(bestSingleIc) }}
                            <b :class="diversificationGain >= 0 ? 'ic-pos' : 'ic-neg'">{{ diversificationGain == null ? '' : (diversificationGain >= 0 ? '+' : '') + fmtNum(diversificationGain) }}</b>
                            （多样性增益 = 组合 − 最强单因子）</span>
                        </div>
                        <div class="mlv-grid2">
                          <div v-for="kind in weightKinds" :key="'w-' + kind">
                            <strong>{{ kind.toUpperCase() }}</strong>
                            <div :id="'ml-weights-' + kind" class="metrics-chart" :style="{ height: Math.max(160, weightRows(kind).length * 24 + 40) + 'px' }"></div>
                          </div>
                        </div>
                      </template>
                      <template v-if="contribKinds.length">
                        <div class="mlv-block-head">
                          <h5>置换贡献（打乱该因子后组合损失多少）</h5>
                          <span class="mlv-metric-toggle">
                            <button v-for="m in contribMetrics" :key="m.key" type="button"
                                    :class="{ active: contribMetric === m.key }"
                                    @click="setContribMetric(m.key)">{{ m.label }}</button>
                          </span>
                          <span class="summary-facet-hint" :title="contribHint">ⓘ</span>
                        </div>
                        <div class="mlv-grid2">
                          <div v-for="kind in contribKinds" :key="'c-' + kind">
                            <strong>{{ kind.toUpperCase() }}</strong>
                            <div :id="'ml-contrib-' + kind" class="metrics-chart"
                                 :style="{ height: Math.max(160, Math.min(15, contribRows(kind).length) * 24 + 40) + 'px' }"></div>
                          </div>
                        </div>
                      </template>
                      <template v-if="(ml.detail.report.subset_curve || []).length">
                        <div class="mlv-block-head">
                          <h5>累积子集曲线（子集规模 vs OOS IC）</h5>
                          <span v-if="bestSubset" class="mlv-sub">
                            IC 峰值 <b>k={{ bestSubset.k }}</b>：blended OOS IC <b>{{ fmtNum(bestSubset.blended) }}</b>
                          </span>
                          <span v-if="bestSubsetRisk" class="mlv-sub">
                            Sharpe 峰值 <b>k={{ bestSubsetRisk.k }}</b>：Sharpe <b>{{ fmtNum(bestSubsetRisk.oos_sharpe) }}</b> · 回撤 {{ pct(bestSubsetRisk.oos_max_drawdown) }}
                          </span>
                        </div>
                        <div id="ml-subset-curve" class="metrics-chart" style="height: 240px"></div>
                        <template v-if="hasSubsetRisk">
                          <div class="mlv-block-head">
                            <h5>子集风险曲线（Top-k 重训组合的 OOS Sharpe / 最大回撤）</h5>
                            <span class="summary-facet-hint" title="每级重训的组合在 OOS 段按预测取前 20% 等权多头、按持有期去重叠后的年化 Sharpe 与最大回撤（轻量口径，不含换手成本，供相对比较）。IC 峰值与 Sharpe 峰值不一致时，以 Sharpe/回撤侧为准——IC 高不等于可交易。">ⓘ</span>
                          </div>
                          <div id="ml-subset-risk" class="metrics-chart" style="height: 220px"></div>
                        </template>
                      </template>
                      <div class="mlv-block-head"><h5>折级 OOS IC</h5></div>
                      <div class="mlv-grid2">
                        <div v-for="(rows, model) in ml.detail.report.fold_metrics" :key="'f-' + model">
                          <strong>{{ model.toUpperCase() }}</strong>
                          <div :id="'ml-folds-' + model" class="metrics-chart" style="height: 170px"></div>
                        </div>
                      </div>
                      <div class="mlv-block-head"><h5>衰减对照（挖掘期 IC vs OOS IC，按 |OOS IC| Top20）</h5></div>
                      <div v-if="decayRows.length" id="ml-decay-chart" class="metrics-chart" :style="{ height: Math.max(160, decayRows.length * 26 + 50) + 'px' }"></div>
                      <details class="mlv-details">
                        <summary>特征清单（{{ (ml.detail.report.feature_names||[]).length }}）/ 剔除（{{ (ml.detail.report.dropped||[]).length }}）/ 衰减明细表</summary>
                        <div class="ml-features">{{ (ml.detail.report.feature_names||[]).join(' · ') }}</div>
                        <div v-for="d in ml.detail.report.dropped" :key="d.name" class="ml-drop">− {{ d.name }}（{{ d.library }}）：{{ d.reason }}</div>
                        <table class="lib-table">
                          <thead><tr><th>因子</th><th>挖掘 IC</th><th>OOS IC</th><th>衰减比</th></tr></thead>
                          <tbody>
                            <tr v-for="row in ml.detail.report.decay_table" :key="row.name">
                              <td>{{ row.name }}</td><td>{{ fmtNum(row.ic_mining) }}</td>
                              <td>{{ fmtNum(row.ic_oos) }}</td><td>{{ fmtNum(row.decay_ratio) }}</td>
                            </tr>
                          </tbody>
                        </table>
                      </details>
                    </template>
                  </div>
                </td>
              </tr>
            </template>
          </tbody>
        </table>
      </div>
    </template>
  </div>
</template>

<script>
import { api } from '../../utils/api.js'
import { chart } from '../../utils/charts.js'

const AXIS_LABEL = { color: '#8494b5', fontSize: 10 }

export default {
  name: 'MlPanel',
  data() {
    return {
      ml: {
        form: { model: 'both', label_days: 5, train_months: 18, step_months: 6, max_corr: 0.6, isolation: 'holdout', no_candidate: false, no_gate: false, subset_curve: false },
        list: [],
        selected: null,
        detail: null,
        starting: false,
        error: '',
      },
      loadingList: false,
      showConfig: false,
      exporting: false,
      contribMetric: 'ic_drop',
      historySortKey: '',
      historySortDir: -1,
    }
  },
  computed: {
    historyColumns: () => [
      { key: 'oos_ic_mean', label: 'OOS IC' },
      { key: 'oos_ic_ir', label: 'ICIR' },
      { key: 'gate_excess_annual', label: '超额年化' },
      { key: 'gate_excess_sharpe', label: '超额夏普' },
      { key: 'gate_max_drawdown', label: '回撤' },
      { key: 'gate_daily_overlap', label: '日重叠' },
      { key: 'gate_daily_turnover', label: '日换手' },
      { key: 'n_folds', label: '折数' },
      { key: 'n_features', label: '特征' },
      { key: 'worst_fold_ic', label: '最差折 IC' },
      { key: 'decay_retention', label: '衰减保留' },
    ],
    anyRunning() {
      return (this.ml.list || []).some(t => t.status === 'running')
    },
    runningTrain() {
      return (this.ml.list || []).find(t => t.status === 'running') || null
    },
    latest() {
      return (this.ml.list || []).find(t => t.status === 'completed' && Number.isFinite(Number(t.oos_ic_mean))) || null
    },
    runningTail() {
      const tail = this.ml.detail?.progress_tail || []
      return tail.length ? tail[tail.length - 1] : ''
    },
    trendRows() {
      return (this.ml.list || [])
        .filter(t => t.status === 'completed' && Number.isFinite(Number(t.oos_ic_mean)))
        .slice()
        .sort((a, b) => (a.train_id < b.train_id ? -1 : 1))
    },
    weightKinds() {
      return Object.keys(this.ml.detail?.report?.feature_weights || {})
    },
    contribKinds() {
      return Object.keys(this.ml.detail?.report?.feature_contribution || {})
    },
    bestSubset() {
      const curve = this.ml.detail?.report?.subset_curve || []
      const valid = curve.filter(r => r.blended != null)
      if (!valid.length) return null
      return valid.reduce((a, b) => (b.blended > a.blended ? b : a))
    },
    bestSubsetRisk() {
      const curve = this.ml.detail?.report?.subset_curve || []
      const valid = curve.filter(r => r.oos_sharpe != null)
      if (!valid.length) return null
      return valid.reduce((a, b) => (b.oos_sharpe > a.oos_sharpe ? b : a))
    },
    hasSubsetRisk() {
      return (this.ml.detail?.report?.subset_curve || []).some(r => r.oos_sharpe != null)
    },
    contribMetrics: () => [
      { key: 'ic_drop', label: 'IC 口径' },
      { key: 'sharpe_drop', label: 'Sharpe 口径' },
      { key: 'dd_impact', label: '回撤口径' },
    ],
    /** gate 档位自动跟随持有天数：≤7 天→技术档（周调仓严门槛），>7 天→基本面档（月调仓松门槛）。
     *  因子池 09-03 统一大库后 modes 已无筛选作用，仅 modes[0] 决定 engine_gate 政策。 */
    gateMode() {
      return Number(this.ml.form.label_days) > 7 ? 'fundamental' : 'technical'
    },
    gateModeLabel() {
      return this.gateMode === 'fundamental' ? '基本面档 · 月调仓松门槛' : '技术档 · 周调仓严门槛'
    },
    contribHint() {
      return ({
        ic_drop: '逐折在 OOS 段把单因子行内打乱后重预测，组合 OOS IC 下降量跨折平均。正值 = 真贡献（掉得越多越重要）；负值（红）= 打乱反而更好，该因子在拖后腿，是剔除候选。',
        sharpe_drop: '同一份打乱预测上重算组合 OOS Sharpe 的下降量。正 = 该因子在贡献收益稳定性；负（红）= 打乱后 Sharpe 反而更高，该因子在拖累风险调整收益。轻量口径：OOS 段前 20% 等权多头、按持有期去重叠年化，不含成本。',
        dd_impact: '打乱该因子后组合最大回撤的变化（幅度差）。正 = 打乱后回撤更深 ⇒ 该因子在压回撤；负（红）= 打乱后回撤反而收窄 ⇒ 该因子在放大回撤，是回撤剔除候选。',
      })[this.contribMetric]
    },
    blendedIc() {
      return this.ml.detail?.report?.oos_ic_blended?.ic_mean ?? null
    },
    bestSingleIc() {
      const rows = this.ml.detail?.report?.decay_table || []
      const ics = rows.map(r => r.ic_oos).filter(v => Number.isFinite(Number(v)))
      return ics.length ? Math.max(...ics.map(Number)) : null
    },
    diversificationGain() {
      if (this.blendedIc == null || this.bestSingleIc == null) return null
      return this.blendedIc - this.bestSingleIc
    },
    gateCards() {
      const g = this.ml.detail?.report?.gate || {}
      const gm = g.metrics || {}
      const dg = g.diagnostics || {}
      const pct = v => (v == null ? '—' : (Number(v) * 100).toFixed(1) + '%')
      return [
        { label: '超额年化', value: pct(gm.excess_annual), title: 'engine_gate OOS 段净超额年化' },
        { label: '超额夏普', value: gm.excess_sharpe == null ? '—' : Number(gm.excess_sharpe).toFixed(2), title: 'engine_gate 超额夏普' },
        { label: '最大回撤', value: pct(gm.max_drawdown ?? dg.max_drawdown), title: 'OOS 段最大回撤' },
        { label: '日换手', value: pct(dg.avg_daily_turnover), title: '组合日均单边换手' },
      ]
    },
    decayRows() {
      const rows = this.ml.detail?.report?.decay_table || []
      return rows.slice().sort((a, b) => Math.abs(b.ic_oos || 0) - Math.abs(a.ic_oos || 0)).slice(0, 20)
    },
    /** 历史表排序：默认时间倒序（train_id 字典序），点击数值列切换 */
    historySorted() {
      const key = this.historySortKey
      const rows = (this.ml.list || []).slice()
      if (!key) return rows.sort((a, b) => (a.train_id < b.train_id ? 1 : -1))
      const dir = this.historySortDir
      return rows.sort((a, b) => {
        const va = a[key] == null ? -Infinity : Number(a[key])
        const vb = b[key] == null ? -Infinity : Number(b[key])
        return (va - vb) * dir
      })
    },
  },
  mounted() {
    this.loadMl()
  },
  beforeUnmount() {
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
    num2(v) {
      if (v === null || v === undefined || Number.isNaN(Number(v))) return '—'
      return Number(v).toFixed(2)
    },
    pct(v) {
      if (v === null || v === undefined || Number.isNaN(Number(v))) return '—'
      return (Number(v) * 100).toFixed(1) + '%'
    },
    toggleHistorySort(key) {
      if (this.historySortKey === key) {
        this.historySortDir = -this.historySortDir
      } else {
        this.historySortKey = key
        this.historySortDir = -1  // 指标默认降序（最好在前）
      }
    },
    icClass(v) {
      const n = Number(v)
      if (!Number.isFinite(n)) return ''
      return n >= 0 ? 'ic-pos' : 'ic-neg'
    },
    fmtTrainId(id) {
      const s = String(id || '')
      return s.length >= 8 ? `${s.slice(0, 4)}-${s.slice(4, 6)}-${s.slice(6, 8)} ${s.slice(9, 11) || ''}:${s.slice(11, 13) || ''}`.trim() : s
    },
    gateText(t) {
      if (t.gate_passed === true) return '通过'
      if (t.gate_passed === false) return '未过'
      return '—'
    },
    isolationShort(t) {
      const s = String(t.time_isolation || '')
      if (s.startsWith('holdout')) return '留出'
      if (s.startsWith('strict')) return '严格'
      return s.slice(0, 4) || '—'
    },
    paramSummary(t) {
      const p = t.params || {}
      const parts = []
      parts.push((p.modes && p.modes.join('/')) || 'technical')
      parts.push((t.model || p.model || 'both').toUpperCase())
      parts.push(`持有${t.label_days ?? p.label_days ?? 5}d`)
      parts.push(`${t.n_folds ?? '—'}折`)
      parts.push(`gate=${(p.modes && p.modes[0]) === 'fundamental' ? '基本面档' : '技术档'}`)
      return parts.join(' · ')
    },
    gateOf(t, key, asPct) {
      const gm = (this.ml.detail?.report?.gate?.metrics) || {}
      void t
      const v = gm[key]
      if (v == null) return '—'
      return asPct ? (Number(v) * 100).toFixed(1) + '%' : Number(v).toFixed(2)
    },
    weightRows(kind) {
      return (this.ml.detail?.report?.feature_weights?.[kind] || []).slice(0, 15)
    },
    contribRows(kind) {
      return (this.ml.detail?.report?.feature_contribution?.[kind] || []).slice(0, 15)
    },
    renderTrend() {
      const rows = this.trendRows
      if (rows.length < 2) return
      const names = rows.map(t => {
        const s = String(t.train_id)
        return `${s.slice(4, 6)}-${s.slice(6, 8)}`
      })
      const c = chart('ml-trend-chart')
      if (!c) return
      c.setOption({
        tooltip: { trigger: 'axis' },
        legend: { textStyle: { color: '#8494b5', fontSize: 10 }, top: 0 },
        grid: { left: 44, right: 14, top: 26, bottom: 24 },
        xAxis: { type: 'category', data: names, axisLabel: { ...AXIS_LABEL, rotate: 30 } },
        yAxis: { type: 'value', axisLabel: { ...AXIS_LABEL } },
        series: [
          { name: 'OOS IC', type: 'line', data: rows.map(t => t.oos_ic_mean), showSymbol: true, lineStyle: { width: 2 }, itemStyle: { color: '#4f8cff' } },
          { name: 'OOS ICIR', type: 'line', data: rows.map(t => t.oos_ic_ir), showSymbol: true, lineStyle: { width: 2, type: 'dashed' }, itemStyle: { color: '#4fc3a1' } },
        ],
      }, true)
    },
    renderWeights(kind) {
      const rows = this.weightRows(kind)
      if (!rows.length) return
      const sorted = [...rows].reverse()
      const c = chart('ml-weights-' + kind)
      if (!c) return
      c.setOption({
        tooltip: { trigger: 'item', formatter: p => `${p.name}：<b>${(p.value * 100).toFixed(1)}%</b>` },
        grid: { left: 190, right: 60, top: 6, bottom: 6 },
        xAxis: { type: 'value', axisLabel: { ...AXIS_LABEL, formatter: v => (v * 100).toFixed(0) + '%' }, splitLine: { lineStyle: { color: '#1c2536' } } },
        yAxis: { type: 'category', data: sorted.map(r => r.name), axisLabel: { ...AXIS_LABEL, width: 180, overflow: 'truncate' } },
        series: [{
          type: 'bar', data: sorted.map(r => r.weight), barMaxWidth: 14,
          label: { show: true, position: 'right', color: '#c6d2e8', fontSize: 10, formatter: p => (p.value * 100).toFixed(1) + '%' },
          itemStyle: { borderRadius: [0, 3, 3, 0], color: '#4f8cff' },
        }],
      }, true)
    },
    renderFolds(model) {
      const rows = (this.ml.detail?.report?.fold_metrics?.[model] || []).filter(r => !r.skipped)
      if (!rows.length) return
      const c = chart('ml-folds-' + model)
      if (!c) return
      const positive = rows.every(r => (r.ic_mean ?? 0) >= 0)
      c.setOption({
        tooltip: { trigger: 'item', formatter: p => `${p.name}：<b>${Number(p.value).toFixed(4)}</b>` },
        grid: { left: 44, right: 14, top: 20, bottom: 26 },
        xAxis: { type: 'category', data: rows.map(r => String(r.oos_start).slice(2)), axisLabel: { ...AXIS_LABEL, rotate: 30 } },
        yAxis: { type: 'value', axisLabel: { ...AXIS_LABEL } },
        series: [{
          type: 'bar', data: rows.map(r => r.ic_mean), barMaxWidth: 26,
          label: { show: true, position: 'top', color: '#c6d2e8', fontSize: 10, formatter: p => Number(p.value).toFixed(3) },
          itemStyle: { borderRadius: [3, 3, 0, 0], color: positive ? '#4fc3a1' : '#e8c491' },
        }],
      }, true)
    },
    renderContribution(kind) {
      const field = this.contribMetric
      const rows = (this.ml.detail?.report?.feature_contribution?.[kind] || [])
        .filter(r => r[field] != null)
        .sort((a, b) => b[field] - a[field])
        .slice(0, 15)
      const sorted = [...rows].reverse()
      const c = chart('ml-contrib-' + kind)
      if (!c) return
      if (!rows.length) {
        // 旧训练未采集该口径
        c.setOption({
          title: { text: '该训练未采集此口径（需重新训练）', left: 'center', top: 'middle', textStyle: { color: '#8494b5', fontSize: 12, fontWeight: 'normal' } },
          xAxis: { show: false }, yAxis: { show: false }, series: [],
        }, true)
        return
      }
      const relTxt = row => (field === 'ic_drop' && row.ic_drop_rel != null
        ? `（占组合 IC ${(row.ic_drop_rel * 100).toFixed(0)}%）` : '')
      c.setOption({
        tooltip: { trigger: 'item', formatter: p => {
          const row = sorted[p.dataIndex]
          return `${p.name}：<b>${Number(p.value).toFixed(4)}</b> ${relTxt(row || {})}`
        } },
        grid: { left: 190, right: 70, top: 6, bottom: 6 },
        xAxis: { type: 'value', axisLabel: { ...AXIS_LABEL }, splitLine: { lineStyle: { color: '#1c2536' } } },
        yAxis: { type: 'category', data: sorted.map(r => r.name), axisLabel: { ...AXIS_LABEL, width: 180, overflow: 'truncate' } },
        series: [{
          type: 'bar', data: sorted.map(r => r[field]), barMaxWidth: 14,
          label: { show: true, position: 'right', color: '#c6d2e8', fontSize: 10, formatter: p => Number(p.value).toFixed(4) },
          itemStyle: { borderRadius: [0, 3, 3, 0], color: p => (p.value >= 0 ? '#4fc3a1' : '#ef6b73') },
        }],
      }, true)
    },
    setContribMetric(key) {
      this.contribMetric = key
      for (const kind of this.contribKinds) this.renderContribution(kind)
    },
    renderSubsetCurve() {
      const curve = this.ml.detail?.report?.subset_curve || []
      if (!curve.length) return
      const c = chart('ml-subset-curve')
      if (!c) return
      const series = []
      for (const kind of this.contribKinds) {
        const data = curve.map(r => r[kind])
        if (data.some(v => v != null)) series.push({ name: kind.toUpperCase() + ' IC', type: 'line', data, showSymbol: true, lineStyle: { width: 1.5, type: 'dashed' } })
      }
      series.push({ name: 'Blended IC', type: 'line', data: curve.map(r => r.blended), showSymbol: true, lineStyle: { width: 2.5 }, itemStyle: { color: '#4f8cff' },
        markPoint: this.bestSubset ? { data: [{ coord: [this.bestSubset.k - 1, this.bestSubset.blended], value: '最优 k=' + this.bestSubset.k }] } : undefined,
      })
      c.setOption({
        tooltip: { trigger: 'axis' },
        legend: { textStyle: { color: '#8494b5', fontSize: 10 }, top: 0 },
        grid: { left: 44, right: 20, top: 28, bottom: 24 },
        xAxis: { type: 'category', data: curve.map(r => 'k=' + r.k), axisLabel: { ...AXIS_LABEL, rotate: 45 } },
        yAxis: { type: 'value', axisLabel: { ...AXIS_LABEL }, scale: true },
        series,
      }, true)
    },
    renderSubsetRisk() {
      const curve = (this.ml.detail?.report?.subset_curve || []).filter(r => r.oos_sharpe != null)
      if (!curve.length) return
      const c = chart('ml-subset-risk')
      if (!c) return
      c.setOption({
        tooltip: { trigger: 'axis' },
        legend: { textStyle: { color: '#8494b5', fontSize: 10 }, top: 0 },
        grid: { left: 44, right: 48, top: 28, bottom: 24 },
        xAxis: { type: 'category', data: curve.map(r => 'k=' + r.k), axisLabel: { ...AXIS_LABEL, rotate: 45 } },
        yAxis: [
          { type: 'value', name: 'Sharpe', nameTextStyle: AXIS_LABEL, axisLabel: { ...AXIS_LABEL }, scale: true, splitLine: { lineStyle: { color: '#1c2536' } } },
          { type: 'value', name: '回撤', nameTextStyle: AXIS_LABEL, axisLabel: { ...AXIS_LABEL, formatter: v => (v * 100).toFixed(0) + '%' }, scale: true, splitLine: { show: false } },
        ],
        series: [
          {
            name: 'OOS Sharpe', type: 'line', data: curve.map(r => r.oos_sharpe), yAxisIndex: 0,
            showSymbol: true, lineStyle: { width: 2 }, itemStyle: { color: '#4f8cff' },
            markPoint: this.bestSubsetRisk ? { data: [{ coord: [curve.indexOf(this.bestSubsetRisk), this.bestSubsetRisk.oos_sharpe], value: '峰值 k=' + this.bestSubsetRisk.k }] } : undefined,
          },
          { name: '最大回撤', type: 'line', data: curve.map(r => r.oos_max_drawdown), yAxisIndex: 1, showSymbol: true, lineStyle: { width: 2, type: 'dashed' }, itemStyle: { color: '#ef6b73' } },
        ],
      }, true)
    },
    renderDecay() {
      const rows = this.decayRows
      if (!rows.length) return
      const sorted = [...rows].reverse()
      const c = chart('ml-decay-chart')
      if (!c) return
      c.setOption({
        tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
        legend: { textStyle: { color: '#8494b5', fontSize: 10 }, top: 0 },
        grid: { left: 190, right: 30, top: 24, bottom: 6 },
        xAxis: { type: 'value', axisLabel: { ...AXIS_LABEL } },
        yAxis: { type: 'category', data: sorted.map(r => r.name), axisLabel: { ...AXIS_LABEL, width: 180, overflow: 'truncate' } },
        series: [
          { name: '挖掘 IC', type: 'bar', data: sorted.map(r => r.ic_mining), barMaxWidth: 8, itemStyle: { color: '#5aa2e8' } },
          { name: 'OOS IC', type: 'bar', data: sorted.map(r => r.ic_oos), barMaxWidth: 8, itemStyle: { color: '#4fc3a1' } },
        ],
      }, true)
    },
    renderAll() {
      if (!window.echarts) return
      this.$nextTick(() => {
        this.renderTrend()
        for (const kind of this.weightKinds) this.renderWeights(kind)
        for (const kind of this.contribKinds) this.renderContribution(kind)
        this.renderSubsetCurve()
        this.renderSubsetRisk()
        for (const model of Object.keys(this.ml.detail?.report?.fold_metrics || {})) this.renderFolds(model)
        this.renderDecay()
      })
    },
    async loadMl() {
      this.loadingList = true
      try {
        this.ml.list = await api('/api/alphaagent/stacking/trainings')
        if (this.ml.list.some(t => t.status === 'running')) {
          this.scheduleMlPoll()
          if (!this.ml.selected) await this.viewMl(this.runningTrain?.train_id, true)
        }
        this.renderAll()
      } catch (e) {
        this.ml.error = e.message
      } finally {
        this.loadingList = false
      }
    },
    async startMl() {
      this.ml.starting = true
      this.ml.error = ''
      try {
        const res = await api('/api/alphaagent/stacking/train', { method: 'POST', body: { ...this.ml.form, modes: [this.gateMode] } })
        this.showConfig = false
        this.ml.selected = res.train_id
        await this.loadMl()
        await this.viewMl(res.train_id)
      } catch (e) {
        this.ml.error = e.message
      } finally {
        this.ml.starting = false
      }
    },
    async toggleDetail(trainId) {
      if (this.ml.selected === trainId && this.ml.detail?.train_id === trainId) {
        this.ml.selected = null
        this.ml.detail = null
        return
      }
      await this.viewMl(trainId)
    },
    async viewMl(trainId, silent) {
      try {
        const d = await api('/api/alphaagent/stacking/trainings/' + encodeURIComponent(trainId))
        if (!silent || this.ml.selected === trainId) {
          this.ml.selected = trainId
          this.ml.detail = d
          this.renderAll()
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

<style scoped>
.mlv-status { display: flex; align-items: center; gap: 10px; padding: 10px 14px; border: 1px solid var(--line); border-radius: 10px; background: var(--bg-soft); margin-bottom: 12px; }
.mlv-dot { width: 9px; height: 9px; border-radius: 50%; background: #5b6b8c; flex: none; }
.mlv-dot.ok { background: #4fc3a1; }
.mlv-dot.running { background: #e8c491; animation: mlv-pulse 1.2s ease-in-out infinite; }
@keyframes mlv-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
.mlv-status-text { display: flex; flex-direction: column; gap: 2px; min-width: 0; flex: 1; }
.mlv-status-text b { font-size: 13px; color: var(--text); }
.mlv-sub { color: var(--muted); font-size: 11px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.mlv-actions { display: flex; gap: 8px; align-items: center; flex: none; }
.mlv-ghost { padding: 6px 12px; border: 1px solid var(--line); border-radius: 8px; background: transparent; color: var(--muted); font-size: 12px; cursor: pointer; }
.mlv-ghost:hover { color: var(--text); border-color: var(--accent, #5b9dff); }
.mlv-primary { padding: 7px 16px; border: 0; border-radius: 8px; background: var(--accent, #4f8cff); color: #fff; font-size: 12px; cursor: pointer; white-space: nowrap; }
.mlv-primary:disabled { background: var(--line-strong, #3a465e); cursor: default; }
.mlv-stop { padding: 6px 12px; border: 1px solid #ef6b7355; border-radius: 8px; background: transparent; color: #ef6b73; font-size: 12px; cursor: pointer; }
.mlv-empty { padding: 32px 20px; text-align: center; color: var(--muted); border: 1px dashed var(--line); border-radius: 10px; margin: 10px 0; }
.mlv-empty p { margin: 4px 0; font-size: 12px; }
.mlv-empty .mlv-primary { margin-top: 12px; }
.mlv-block { margin-top: 14px; }
.mlv-block-head { display: flex; align-items: center; gap: 8px; margin: 6px 0 8px; }
.mlv-block-head h4, .mlv-block-head h5 { margin: 0; }
.mlv-expand-row > td { background: rgb(79 140 255 / 0.04); }
.mlv-expand { padding: 8px 6px; display: flex; flex-direction: column; gap: 12px; }
.mlv-expand-head { display: flex; flex-wrap: wrap; gap: 14px; align-items: center; }
.mlv-grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
@media (max-width: 1100px) { .mlv-grid2 { grid-template-columns: 1fr; } }
.mlv-details summary { cursor: pointer; color: var(--muted); font-size: 12px; margin: 6px 0; }
.mlv-details summary:hover { color: var(--text); }
.mlv-bad { color: #ef6b73; }
.mlv-metric-toggle { display: inline-flex; gap: 0; border: 1px solid var(--line); border-radius: 7px; overflow: hidden; }
.mlv-metric-toggle button { border: 0; background: transparent; color: var(--muted); font-size: 11px; padding: 3px 10px; cursor: pointer; }
.mlv-metric-toggle button + button { border-left: 1px solid var(--line); }
.mlv-metric-toggle button.active { background: rgb(79 140 255 / 0.18); color: var(--text); }
.ml-gate-mode { align-self: center; color: var(--muted); font-size: 11px; border: 1px dashed var(--line); border-radius: 7px; padding: 4px 10px; cursor: help; }
</style>
