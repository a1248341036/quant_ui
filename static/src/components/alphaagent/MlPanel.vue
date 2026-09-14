<template>
  <div class="ml-agent-shell agent-shell">
    <!-- ════════════════ 左侧边栏：Runs 列表 ════════════════ -->
    <aside class="agent-sidebar ml-sidebar">
      <div class="sidebar-head">
        <div>
          <span class="eyebrow">STACKING</span>
          <h2>ML 组合任务</h2>
        </div>
        <button class="new-run" type="button" @click="openNewModal">
          <span>＋</span> 新建
        </button>
      </div>

      <div class="session-label">
        <span>任务列表（{{ ml.list.length }}）</span>
        <button class="archived-toggle" type="button" title="刷新列表" @click="loadMl">刷新</button>
      </div>

      <!-- ── 运行中任务常驻专属置顶卡片 ── -->
      <div
        v-if="runningTrain"
        class="ml-running-pinned-card"
        :class="{ active: ml.selected === runningTrain.train_id }"
        @click="selectTrain(runningTrain.train_id)"
      >
        <div class="ml-running-pinned-head">
          <span class="status-dot status-running"></span>
          <strong>当前任务正在运行中</strong>
          <span class="ml-pulse-tag">RUNNING</span>
        </div>
        <div class="ml-running-pinned-body">
          <b>{{ fmtTrainId(runningTrain.train_id) }}</b>
          <small>{{ runningTail || '计算中，点击随时返回实时视窗…' }}</small>
        </div>
      </div>

      <div class="session-list">
        <div
          v-for="t in ml.list"
          :key="t.train_id"
          class="session-item"
          :class="{ active: ml.selected === t.train_id, 'is-running': t.status === 'running' }"
          @click="selectTrain(t.train_id)"
        >
          <div class="session-select">
            <span class="status-dot" :class="runStatusClass(t)"></span>
            <div class="session-copy">
              <strong>{{ fmtTrainId(t.train_id) }} <i v-if="t.status === 'running'" class="ml-tag-running">运行中</i></strong>
              <small>{{ schemeBadge(t) }} · {{ formatOos(t) }} · {{ t.n_folds ?? '—' }}折</small>
            </div>
          </div>
        </div>
        <div v-if="!ml.list.length && !loadingList" class="sidebar-empty">
          暂无组合任务记录
        </div>
      </div>

      <div class="sidebar-footer">
        <button class="ml-drawer-btn" type="button" @click="showCfDrawer = true">
          📁 组合因子库 ({{ compositeList.length }})
        </button>
      </div>
    </aside>

    <!-- ════════════════ 右侧主视窗：Agent 工作台 ════════════════ -->
    <main class="agent-main ml-main">
      <!-- 若当前在看历史任务，但后台有正在运行的任务，显示醒目的悬浮横幅随时一键跳回 -->
      <div
        v-if="runningTrain && activeTrain && activeTrain.train_id !== runningTrain.train_id"
        class="ml-viewing-history-alert"
        @click="selectTrain(runningTrain.train_id)"
      >
        <span class="status-dot status-running"></span>
        <span>后台任务 <b>{{ fmtTrainId(runningTrain.train_id) }}</b> 正在运行中（{{ runningTail || '计算中' }}）</span>
        <button class="ml-jump-back-btn" type="button">点击返回正在运行的任务 ➜</button>
      </div>

      <!-- 顶部 Header -->
      <header class="agent-header ml-header">
        <div class="agent-title">
          <div class="agent-orb">ML</div>
          <div>
            <h1>{{ activeTitle }}</h1>
            <div class="agent-subtitle">
              {{ activeSubtitle }}
            </div>
          </div>
        </div>

        <div class="header-actions">
          <span class="ml-safe-badge" title="盲测物理隔离：当前处于研发验证态，数据严格物理截断至 2024-12-31，2025+ 盲测段安全锁定未加载，LLM 绝无法窥探盲测数据。">
            🛡️ 盲测隔离（≤2024-12-31）
          </span>

          <div class="mode-toggle">
            <button :class="{ active: viewMode === 'timeline' }" @click="viewMode = 'timeline'">⚡ 流式时间线</button>
            <button :class="{ active: viewMode === 'analytics' }" @click="viewMode = 'analytics'">📊 深度归因看板</button>
          </div>

          <button
            v-if="activeTrain && activeTrain.status === 'completed' && activeReport"
            class="mlv-ghost"
            type="button"
            :disabled="!!savingCompositeId"
            @click="saveCompositeFactor(activeTrain)"
          >
            {{ savingCompositeId === activeTrain.train_id ? '保存中…' : '另存为组合因子' }}
          </button>

          <button
            v-if="activeTrain && activeTrain.status === 'running'"
            class="stop-btn"
            type="button"
            @click="stopMl(activeTrain.train_id)"
          >
            停止
          </button>
        </div>
      </header>

      <!-- ══ 主体：流式时间线视图 ══ -->
      <div v-if="viewMode === 'timeline'" class="agent-thread ml-thread" ref="threadContainer">
        <!-- 空状态引导 -->
        <div v-if="!activeTrain" class="welcome">
          <div class="welcome-orb">ML</div>
          <h2>开启多因子智能组合</h2>
          <p>基于统一因子大库与 Walk-Forward 时间隔离，通过机器学习（Ridge/LGBM）或规则加权（1/N、ICIR、HRP）进行样本外滚动评估与实盘门禁裁决。</p>
          <div class="suggestions">
            <button class="mlv-primary" @click="openNewModal">开始新建组合运行</button>
          </div>
        </div>

        <!-- 结构化事件时间线卡片 -->
        <template v-else>
          <div v-for="(ev, idx) in currentEvents" :key="idx" class="message-row">
            <!-- 1. session_start: 参数与任务概览 -->
            <div v-if="ev.event === 'session_start'" class="ml-card ml-start-card">
              <div class="ml-card-head">
                <span class="ml-card-tag">任务启动</span>
                <strong>{{ ev.scheme_label || ev.scheme?.toUpperCase() }}</strong>
                <span class="ml-card-time">{{ ev.ts }}</span>
              </div>
              <div class="ml-card-body">
                <div class="ml-meta-grid">
                  <span><i>模型：</i>{{ (ev.model || 'both').toUpperCase() }}</span>
                  <span><i>持有天数：</i>{{ ev.label_days }}d</span>
                  <span><i>隔离模式：</i>{{ ev.isolation }}</span>
                  <span><i>去重阈值：</i>{{ ev.max_corr }}</span>
                  <span><i>模式档位：</i>{{ ev.eval_mode || 'tuning' }}</span>
                </div>
              </div>
            </div>

            <!-- 2. agent_thinking: 思考过程气泡（展开/折叠） -->
            <details v-else-if="ev.event === 'agent_thinking'" open class="thinking-card ml-think-bubble">
              <summary>
                <span class="thinking-icon">💭</span>
                <b>{{ ev.title || '思考过程' }}</b>
                <span class="tool-state">{{ ev.ts }}</span>
              </summary>
              <pre>{{ ev.content }}</pre>
            </details>

            <!-- 3. ml_stage: 阶段状态卡片 -->
            <div v-else-if="ev.event === 'ml_stage'" class="ml-stage-card" :class="ev.stage">
              <div class="ml-stage-icon">
                <span v-if="ev.stage === 'panel_loading'" class="spinner"></span>
                <span v-else>✓</span>
              </div>
              <div class="ml-stage-content">
                <strong>{{ ev.title || '运行阶段' }}</strong>
                <p>{{ ev.message }}</p>
              </div>
              <span class="ml-card-time">{{ ev.ts }}</span>
            </div>

            <!-- 3b. tool_call: 工具调用卡片 -->
            <div v-else-if="ev.event === 'tool_call'" class="ml-stage-card" style="border-left: 3px solid #7dd3fc;">
              <div class="ml-stage-icon" style="background: rgb(125 211 252 / 0.15); color: #7dd3fc;">⚙️</div>
              <div class="ml-stage-content">
                <strong>{{ ev.title || ev.name }}</strong>
                <p>{{ ev.message }}</p>
              </div>
              <span class="ml-card-time">{{ ev.ts }}</span>
            </div>

            <!-- 3c. tool_result: 工具结果卡片 -->
            <div v-else-if="ev.event === 'tool_result'" class="ml-card ml-screen-card" style="border-left: 3px solid #4fc3a1;">
              <div class="ml-card-head">
                <span class="ml-card-tag prod">TOOL 结果</span>
                <strong>{{ ev.title }}</strong>
                <span class="ml-card-time">{{ ev.ts }}</span>
              </div>
              <div class="ml-card-body">
                <p class="ml-reco-rationale">{{ ev.summary || ev.error }}</p>
                <div v-if="ev.recommended && ev.recommended.length" class="ml-factor-chips">
                  <span
                    v-for="name in ev.recommended"
                    :key="name"
                    class="ml-chip"
                    @click.stop.prevent="toggleFactorCard(poolFactor(name), $event)"
                  >
                    {{ name }}
                  </span>
                </div>
                <div v-if="ev.recommended && ev.recommended.length" class="mlv-sub" style="margin-top:6px; color:#4fc3a1">
                  ✓ 已将上述 {{ ev.recommended.length }} 个因子填入当前训练勾选（窗口 {{ ev.window }}）
                </div>
              </div>
            </div>

            <!-- 4. ml_pool_screened: 因子语义研判卡片 -->
            <div v-else-if="ev.event === 'ml_pool_screened'" class="ml-card ml-screen-card">
              <div class="ml-card-head">
                <span class="ml-card-tag prod">因子语义研判</span>
                <strong>推荐 {{ ev.n_recommended }} / {{ ev.n_total }} 个候选因子</strong>
                <span v-if="ev.reused" class="mlv-sub">· 复用锁定推荐</span>
                <span class="ml-card-time">{{ ev.ts }}</span>
              </div>
              <div class="ml-card-body">
                <p v-if="ev.rationale" class="ml-reco-rationale"><b>研判依据：</b>{{ ev.rationale }}</p>
                <div class="ml-factor-chips">
                  <span
                    v-for="name in (ev.recommended || [])"
                    :key="name"
                    class="ml-chip"
                    @click.stop.prevent="toggleFactorCard(poolFactor(name), $event)"
                  >
                    {{ name }}
                  </span>
                </div>
              </div>
            </div>

            <!-- 5. ml_features_filtered: 特征物化与去重卡片 -->
            <div v-else-if="ev.event === 'ml_features_filtered'" class="ml-card ml-filter-card">
              <div class="ml-card-head">
                <span class="ml-card-tag">特征过滤</span>
                <strong>入选 {{ ev.n_features }} 个有效特征（剔除 {{ ev.n_dropped }} 个）</strong>
                <span class="ml-card-time">{{ ev.ts }}</span>
              </div>
              <div class="ml-card-body">
                <details v-if="ev.dropped && ev.dropped.length" class="ml-drop-details">
                  <summary>查看剔除明细 ({{ ev.dropped.length }})</summary>
                  <div v-for="d in ev.dropped" :key="d.name" class="ml-drop-item">
                    − {{ d.name }}: {{ d.reason }}
                  </div>
                </details>
              </div>
            </div>

            <!-- 6. ml_fold_progress: Walk-Forward 拟合进度卡片 -->
            <div v-else-if="ev.event === 'ml_fold_progress'" class="ml-card ml-fold-card">
              <div class="ml-card-head">
                <span class="ml-card-tag">模型拟合</span>
                <strong>{{ ev.title || ev.kind?.toUpperCase() }}</strong>
                <span class="ml-card-time">{{ ev.ts }}</span>
              </div>
              <div class="ml-card-body">
                <div v-if="ev.fold_reports && ev.fold_reports.length" class="ml-folds-grid">
                  <div v-for="f in ev.fold_reports" :key="f.fold" class="ml-fold-item">
                    <b>折 {{ f.fold }} ({{ String(f.oos_start).slice(2) }}~{{ String(f.oos_end).slice(2) }})</b>
                    <span :class="icClass(f.ic_mean)">IC: {{ fmtNum(f.ic_mean) }}</span>
                    <span>IR: {{ fmtNum(f.ic_ir) }}</span>
                  </div>
                </div>
                <div v-if="ev.feature_weights && ev.feature_weights.length" class="ml-top-weights">
                  <small>特征权重 Top5：</small>
                  <span v-for="w in ev.feature_weights.slice(0, 5)" :key="w.name">
                    {{ w.name }} ({{ (w.weight * 100).toFixed(1) }}%)
                  </span>
                </div>
              </div>
            </div>

            <!-- 7. ml_gate_evaluated: engine_gate 门禁回测 -->
            <div v-else-if="ev.event === 'ml_gate_evaluated'" class="ml-card ml-gate-card">
              <div class="ml-card-head">
                <span class="ml-card-tag" :class="ev.passed ? 'prod' : 'bad'">
                  {{ ev.passed ? 'GATE 通过' : 'GATE 未过' }}
                </span>
                <strong>实盘可交易性裁决（周调仓）</strong>
                <span class="ml-card-time">{{ ev.ts }}</span>
              </div>
              <div class="ml-card-body">
                <div class="metrics-cards">
                  <div class="metrics-card">
                    <b>{{ pct(ev.metrics?.excess_annual) }}</b><span>超额年化</span>
                  </div>
                  <div class="metrics-card">
                    <b>{{ num2(ev.metrics?.excess_sharpe) }}</b><span>超额夏普</span>
                  </div>
                  <div class="metrics-card">
                    <b>{{ pct(ev.metrics?.max_drawdown) }}</b><span>最大回撤</span>
                  </div>
                  <div class="metrics-card">
                    <b>{{ pct(ev.metrics?.daily_overlap) }}</b><span>日重叠</span>
                  </div>
                </div>
                <div v-if="ev.fail_reasons && ev.fail_reasons.length" class="mlv-bad" style="margin-top:8px">
                  未过原因：{{ ev.fail_reasons.join('、') }}
                </div>
              </div>
            </div>

            <!-- 8. ml_summary_generated: 组合说明书 -->
            <div v-else-if="ev.event === 'ml_summary_generated'" class="ml-card ml-summary-card">
              <div class="ml-card-head">
                <span class="ml-card-tag prod">组合研判说明书</span>
                <strong>LLM 深度解读报告</strong>
                <span class="ml-card-time">{{ ev.ts }}</span>
              </div>
              <div class="ml-card-body">
                <div class="ml-llm-summary">
                  <div class="ml-llm-sec">
                    <b>总体总结</b>
                    <p>{{ ev.summary?.summary }}</p>
                  </div>
                  <div v-if="ev.summary?.strengths?.length" class="ml-llm-sec">
                    <b>核心优势</b>
                    <ul><li v-for="(s, i) in ev.summary.strengths" :key="'s'+i">{{ s }}</li></ul>
                  </div>
                  <div v-if="ev.summary?.risks?.length" class="ml-llm-sec">
                    <b>潜在风险</b>
                    <ul class="ml-llm-risk"><li v-for="(r, i) in ev.summary.risks" :key="'r'+i">{{ r }}</li></ul>
                  </div>
                  <div v-if="ev.summary?.suggestions?.length" class="ml-llm-sec">
                    <b>改进建议</b>
                    <ul><li v-for="(g, i) in ev.summary.suggestions" :key="'g'+i">{{ g }}</li></ul>
                  </div>
                </div>
              </div>
            </div>

            <!-- 9. session_end: 收敛完成 -->
            <div v-else-if="ev.event === 'session_end'" class="ml-card ml-end-card">
              <div class="ml-card-head">
                <span class="ml-card-tag prod">训练完成</span>
                <strong>组合 OOS IC: {{ fmtNum(ev.oos_ic) }} · ICIR: {{ fmtNum(ev.oos_ir) }}</strong>
                <span class="ml-card-time">{{ ev.ts }}</span>
              </div>
            </div>
          </div>

          <!-- 运行中实时输出日志尾部 -->
          <div v-if="activeTrain.status === 'running'" class="message-row">
            <div class="typing-row">
              <div class="typing"><i></i><i></i><i></i></div>
              <span>组合拟合计算中…</span>
            </div>
            <pre class="ml-log">{{ runningTail || '等待子进程实时输出…' }}</pre>
          </div>
        </template>
      </div>

      <!-- ══ 主体：深度归因看板视图 ══ -->
      <div v-else-if="viewMode === 'analytics'" class="agent-thread ml-thread ml-analytics-panel">
        <template v-if="activeReport">
          <!-- 核心指标卡片 -->
          <div class="metrics-cards">
            <div class="metrics-card">
              <b :class="icClass(activeReport.oos_ic_blended?.ic_mean)">{{ fmtNum(activeReport.oos_ic_blended?.ic_mean) }}</b>
              <span>OOS IC 均值</span>
            </div>
            <div class="metrics-card">
              <b>{{ fmtNum(activeReport.oos_ic_blended?.ic_ir) }}</b>
              <span>OOS ICIR</span>
            </div>
            <div class="metrics-card">
              <b :class="activeReport.gate?.passed ? 'ic-pos' : 'ic-neg'">{{ activeReport.gate?.passed ? '通过' : '未过' }}</b>
              <span>engine_gate</span>
            </div>
            <div class="metrics-card">
              <b>{{ (activeReport.feature_names || []).length }}/{{ activeReport.folds }}</b>
              <span>特征 / 折数</span>
            </div>
          </div>

          <!-- 特征权重 Top15 -->
          <div v-if="weightKinds.length" class="mlv-block">
            <div class="mlv-block-head">
              <h5>特征权重 Top15（跨折归一平均）</h5>
              <span class="mlv-sub">
                组合 OOS IC {{ fmtNum(blendedIc) }} vs 最强单因子 {{ fmtNum(bestSingleIc) }}
                <b :class="diversificationGain >= 0 ? 'ic-pos' : 'ic-neg'">
                  {{ diversificationGain == null ? '' : (diversificationGain >= 0 ? '+' : '') + fmtNum(diversificationGain) }}
                </b>
                （多样性增益）
              </span>
            </div>
            <div class="mlv-grid2">
              <div v-for="kind in weightKinds" :key="'w-' + kind">
                <strong>{{ kind.toUpperCase() }}</strong>
                <div :id="'ml-weights-' + kind" class="metrics-chart" :style="{ height: Math.max(160, weightRows(kind).length * 24 + 40) + 'px' }"></div>
              </div>
            </div>
          </div>

          <!-- 置换重要性贡献 -->
          <div v-if="contribKinds.length" class="mlv-block">
            <div class="mlv-block-head">
              <h5>置换贡献（打乱该因子后组合损失多少）</h5>
              <span class="mlv-metric-toggle">
                <button v-for="m in contribMetrics" :key="m.key" type="button" :class="{ active: contribMetric === m.key }" @click="setContribMetric(m.key)">{{ m.label }}</button>
              </span>
              <span class="summary-facet-hint" :title="contribHint">ⓘ</span>
            </div>
            <div class="mlv-grid2">
              <div v-for="kind in contribKinds" :key="'c-' + kind">
                <strong>{{ kind.toUpperCase() }}</strong>
                <div :id="'ml-contrib-' + kind" class="metrics-chart" :style="{ height: Math.max(160, Math.min(15, contribRows(kind).length) * 24 + 40) + 'px' }"></div>
              </div>
            </div>
          </div>

          <!-- 累积子集曲线 -->
          <div v-if="(activeReport.subset_curve || []).length" class="mlv-block">
            <div class="mlv-block-head">
              <h5>累积子集曲线（子集规模 vs OOS IC）</h5>
              <span v-if="bestSubset" class="mlv-sub">IC 峰值 k={{ bestSubset.k }}：blended {{ fmtNum(bestSubset.blended) }}</span>
            </div>
            <div id="ml-subset-curve" class="metrics-chart" style="height: 240px"></div>
          </div>

          <!-- 不挑全上对照 -->
          <div v-if="schemeRows.length" class="mlv-block">
            <div class="mlv-block-head">
              <h5>不挑全上对照（简单投票 vs 学习加权）</h5>
            </div>
            <table class="lib-table">
              <thead>
                <tr><th>方案</th><th>OOS IC</th><th>ICIR</th><th>OOS Sharpe</th><th>最大回撤</th><th>样本日</th></tr>
              </thead>
              <tbody>
                <tr v-for="row in schemeRows" :key="row.scheme">
                  <td><strong>{{ row.label }}</strong></td>
                  <td :class="icClass(row.ic_mean)"><strong>{{ fmtNum(row.ic_mean) }}</strong></td>
                  <td>{{ fmtNum(row.ic_ir) }}</td>
                  <td>{{ row.oos_sharpe == null ? '—' : num2(row.oos_sharpe) }}</td>
                  <td>{{ row.oos_max_drawdown == null ? '—' : pct(row.oos_max_drawdown) }}</td>
                  <td>{{ row.n_days }}</td>
                </tr>
              </tbody>
            </table>
          </div>

          <!-- 衰减对照表 -->
          <div v-if="decayRows.length" class="mlv-block">
            <div class="mlv-block-head"><h5>衰减对照（挖掘期 IC vs OOS IC，Top20）</h5></div>
            <div id="ml-decay-chart" class="metrics-chart" :style="{ height: Math.max(160, decayRows.length * 26 + 50) + 'px' }"></div>
          </div>
        </template>
        <div v-else class="normal-mode-empty">
          当前训练暂无深度归因报表
        </div>
      </div>
    </main>

    <!-- ════════════════ 弹窗：新建组合运行 ════════════════ -->
    <div v-if="showNewModal" class="ml-modal-mask" @click.self="showNewModal = false">
      <div class="ml-modal">
        <div class="ml-modal-head">
          <h3>新建 ML 组合训练</h3>
          <button class="ml-modal-close" @click="showNewModal = false">×</button>
        </div>
        <div class="ml-modal-body">
          <div class="ml-form">
            <label>组合方法
              <select v-model="ml.form.scheme" class="ml-input">
                <option value="ml">ML 学习加权（Ridge + LGBM）</option>
                <option value="equal">等权 1/N</option>
                <option value="icir">ICIR 加权</option>
                <option value="hrp">HRP 加权</option>
              </select>
            </label>
            <label v-if="ml.form.scheme === 'ml'">模型
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
            <label v-if="ml.form.scheme === 'ml'" class="ml-check"><input type="checkbox" v-model="ml.form.subset_curve"> 累积子集曲线</label>
            <label v-if="ml.form.scheme === 'ml'" class="ml-check"><input type="checkbox" v-model="ml.form.llm_assist"> LLM 辅助（推荐 + 说明书）</label>
            <label v-if="ml.form.scheme === 'ml'" class="ml-check"><input type="checkbox" v-model="ml.form.multi_path"> 多路径对照</label>
          </div>

          <div class="ml-factor-picker">
            <div class="mlv-block-head" style="margin:0 0 6px">
              <h5>训练因子自选（未选 = 全部）</h5>
              <span v-if="ml.form.include_factors.length" class="mlv-sub">已选 {{ ml.form.include_factors.length }}</span>
              <button class="mlv-ghost" type="button" @click="setAllFactors(true)">全选</button>
              <button class="mlv-ghost" type="button" @click="setAllFactors(false)">清空</button>
              <button class="mlv-ghost" type="button" @click="recommendFactors" :disabled="ml.recommending">
                {{ ml.recommending ? '推荐中…' : '帮我推荐 Top-8' }}
              </button>
            </div>
            <div class="ml-factor-list">
              <div v-for="f in ml.factorPool" :key="f.name" class="ml-factor-row">
                <input type="checkbox" :value="f.name" v-model="ml.form.include_factors">
                <span class="ml-factor-name" @click.stop.prevent="toggleFactorCard(f, $event)">{{ f.name }}</span>
                <i class="ml-factor-lib">{{ f.library }}</i>
              </div>
            </div>
          </div>
          <div v-if="ml.error" class="ml-error">{{ ml.error }}</div>
        </div>
        <div class="ml-modal-foot">
          <button class="mlv-ghost" @click="showNewModal = false">取消</button>
          <button class="mlv-primary" :disabled="ml.starting" @click="startMl">开始运行</button>
        </div>
      </div>
    </div>

    <!-- ════════════════ 抽屉：组合因子库 ════════════════ -->
    <div v-if="showCfDrawer" class="ml-drawer-mask" @click.self="showCfDrawer = false">
      <div class="ml-drawer">
        <div class="ml-drawer-head">
          <h3>组合因子中台库 ({{ compositeList.length }})</h3>
          <button class="ml-modal-close" @click="showCfDrawer = false">×</button>
        </div>
        <div class="ml-drawer-body">
          <table class="lib-table">
            <thead>
              <tr><th>名称</th><th>方案</th><th>OOS IC</th><th>gate</th></tr>
            </thead>
            <tbody>
              <tr v-for="c in compositeList" :key="c.id" :class="{ active: compositeSelected === c.id }" @click="viewComposite(c.id)">
                <td><strong>{{ c.name }}</strong></td>
                <td>{{ c.scheme_label }}</td>
                <td :class="icClass(c.oos_ic)">{{ fmtNum(c.oos_ic) }}</td>
                <td>{{ c.gate_passed ? '通过' : '未过' }}</td>
              </tr>
            </tbody>
          </table>
          <div v-if="compositeDetail" class="ml-cf-detail">
            <h4>{{ compositeDetail.name }} 复现详情</h4>
            <pre class="ml-log">{{ (compositeDetail.provenance||{}).repro_command }}</pre>
            <div class="mlv-sub">分数文件：{{ (compositeDetail.provenance||{}).score_path }}</div>
          </div>
        </div>
      </div>
    </div>

    <!-- ════════════════ 全局 Teleport 机制卡片 ════════════════ -->
    <teleport to="body">
      <div v-if="factorCard" class="ml-factor-card" :style="{ left: factorCardPos.x + 'px', top: factorCardPos.y + 'px' }" @click.stop>
        <div class="mlfc-head">
          <b>{{ factorCard.name }}</b>
          <span class="mlfc-tag" :class="factorCard.library === '正式' ? 'tag-prod' : 'tag-cand'">{{ factorCard.library }}</span>
          <button class="mlfc-close" type="button" @click="factorCard = null">×</button>
        </div>
        <div class="mlfc-metrics">
          <span><i>train IC</i>{{ fmtNum(factorCard.data.train_ic) }}</span>
          <span><i>val IC</i>{{ fmtNum(factorCard.data.val_ic) }}</span>
          <span><i>保留比</i>{{ factorCard.data.val_ic_retention != null ? (factorCard.data.val_ic_retention * 100).toFixed(0) + '%' : '—' }}</span>
        </div>
        <div v-if="hoverSegments.length" class="mlfc-body">
          <div v-for="(seg, i) in hoverSegments" :key="i" class="mlfc-seg">
            <b v-if="seg.title" class="mlfc-seg-title">【{{ seg.title }}】</b>
            <span class="mlfc-seg-text">{{ seg.body }}</span>
          </div>
        </div>
      </div>
    </teleport>
  </div>
</template>

<script>
import { api } from '../../utils/api.js'
import { chart } from '../../utils/charts.js'

const AXIS_LABEL = { color: '#8494b5', fontSize: 10 }
const SEG_LABELS = [
  '经济直觉', '机制', '结构', '统计', '消融佐证', '单腿消融', '消融', '预测对账', '对账',
  '设计目的', '变异类型', 'Reviewer备注', 'IC方向', 'train指标', '盲测', '验证', '风险', '注意',
].sort((a, b) => b.length - a.length).join('|')

function parseCommentSegments(text) {
  if (!text) return []
  const trimmed = String(text).trim()
  if (/【[^】]+】/.test(trimmed)) {
    const parts = trimmed.split(/(【[^】]*】)/).filter(s => s && s.trim())
    const segs = []
    let cur = null
    for (const p of parts) {
      const m = p.match(/^【([^】]*)】\s*([\s\S]*)$/)
      if (m) {
        cur = { title: m[1], body: m[2].trim() }
        segs.push(cur)
      } else if (cur) {
        cur.body += (cur.body ? '\n' : '') + p.trim()
      } else {
        segs.push({ title: '', body: p.trim() })
      }
    }
    return segs
  }
  const re = new RegExp('(?:^|[。；;！!])\\s*(' + SEG_LABELS + ')[：:=]', 'g')
  const marks = []
  let m
  while ((m = re.exec(trimmed)) !== null) {
    marks.push({
      title: m[1],
      start: m.index + m[0].indexOf(m[1]),
      bodyStart: m.index + m[0].length,
    })
    re.lastIndex = m.index + m[0].length
  }
  if (!marks.length) return [{ title: '', body: trimmed }]
  const segs = []
  if (marks[0].start > 0) segs.push({ title: '', body: trimmed.slice(0, marks[0].start).trim() })
  for (let i = 0; i < marks.length; i++) {
    const end = i + 1 < marks.length ? marks[i + 1].start : trimmed.length
    segs.push({ title: marks[i].title, body: trimmed.slice(marks[i].bodyStart, end).trim() })
  }
  return segs.filter(s => s.body || s.title)
}

export default {
  name: 'MlPanel',
  data() {
    return {
      viewMode: 'timeline', // 'timeline' | 'analytics'
      showNewModal: false,
      showCfDrawer: false,
      ml: {
        form: { eval_mode: 'tuning', scheme: 'ml', model: 'both', label_days: 5, train_months: 18, step_months: 6, max_corr: 0.6, isolation: 'holdout', no_candidate: false, no_gate: false, subset_curve: false, multi_path: false, llm_assist: false, include_factors: [] },
        factorPool: [],
        factorPoolLoading: false,
        list: [],
        selected: null,
        detail: null,
        starting: false,
        error: '',
        recommending: false,
        recommend: null,
      },
      currentEvents: [],
      loadingList: false,
      contribMetric: 'ic_drop',
      factorCard: null,
      factorCardPos: { x: 0, y: 0 },
      compositeList: [],
      compositeSelected: null,
      compositeDetail: null,
      savingCompositeId: '',
    }
  },
  computed: {
    runningTrain() {
      return (this.ml.list || []).find(t => t.status === 'running') || null
    },
    activeTrain() {
      return (this.ml.list || []).find(t => t.train_id === this.ml.selected) || (this.ml.list[0] || null)
    },
    activeReport() {
      return this.ml.detail?.report || null
    },
    activeTitle() {
      if (!this.activeTrain) return 'ML 组合智能体'
      return this.fmtTrainId(this.activeTrain.train_id) + ' · ' + (this.activeTrain.status === 'running' ? '运行中' : '已完成')
    },
    activeSubtitle() {
      if (!this.activeTrain) return '统一因子大库 · Walk-Forward 滚动组合与实盘门禁裁决'
      return this.paramSummary(this.activeTrain)
    },
    runningTail() {
      const tail = this.ml.detail?.progress_tail || []
      return tail.length ? tail[tail.length - 1] : ''
    },
    weightKinds() {
      return Object.keys(this.activeReport?.feature_weights || {})
    },
    contribKinds() {
      return Object.keys(this.activeReport?.feature_contribution || {})
    },
    bestSubset() {
      const curve = this.activeReport?.subset_curve || []
      const valid = curve.filter(r => r.blended != null)
      if (!valid.length) return null
      return valid.reduce((a, b) => (b.blended > a.blended ? b : a))
    },
    schemeRows() {
      return this.activeReport?.scheme_compare?.schemes || []
    },
    decayRows() {
      const rows = this.activeReport?.decay_table || []
      return rows.slice().sort((a, b) => Math.abs(b.ic_oos || 0) - Math.abs(a.ic_oos || 0)).slice(0, 20)
    },
    blendedIc() {
      return this.activeReport?.oos_ic_blended?.ic_mean ?? null
    },
    bestSingleIc() {
      const rows = this.activeReport?.decay_table || []
      const ics = rows.map(r => r.ic_oos).filter(v => Number.isFinite(Number(v)))
      return ics.length ? Math.max(...ics.map(Number)) : null
    },
    diversificationGain() {
      if (this.blendedIc == null || this.bestSingleIc == null) return null
      return this.blendedIc - this.bestSingleIc
    },
    contribMetrics: () => [
      { key: 'ic_drop', label: 'IC 口径' },
      { key: 'sharpe_drop', label: 'Sharpe 口径' },
      { key: 'dd_impact', label: '回撤口径' },
    ],
    hoverComment() {
      const d = this.factorCard?.data || {}
      return d.comment_full || d.comment || ''
    },
    hoverSegments() {
      return parseCommentSegments(this.hoverComment)
    },
    gateMode() {
      return Number(this.ml.form.label_days) > 7 ? 'fundamental' : 'technical'
    },
  },
  watch: {
    viewMode(val) {
      if (val === 'analytics') {
        this.renderAll()
      }
    },
    'ml.selected'(newId) {
      if (newId) {
        this.connectEvents(newId)
      }
    },
  },
  mounted() {
    this.loadMl()
    this.loadFactorPool()
    this.loadCompositeFactors()
    this._docClick = (e) => {
      if (!this.factorCard) return
      const t = e.target
      if (t.closest && (t.closest('.ml-factor-card') || t.closest('.ml-factor-name') || t.closest('.ml-chip'))) return
      this.factorCard = null
    }
    document.addEventListener('click', this._docClick)
  },
  beforeUnmount() {
    document.removeEventListener('click', this._docClick)
    if (this._eventSource) {
      this._eventSource.close()
      this._eventSource = null
    }
    if (this._mlTimer) {
      clearInterval(this._mlTimer)
      this._mlTimer = null
    }
  },
  methods: {
    openNewModal() {
      this.showNewModal = true
      this.loadFactorPool(true)
    },
    runStatusClass(t) {
      if (t.status === 'running') return 'status-running'
      if (t.status === 'completed') return 'status-completed'
      return 'status-failed'
    },
    schemeBadge(t) {
      const s = t.scheme || 'ml'
      return ({ ml: 'ML', equal: '1/N', icir: 'ICIR', hrp: 'HRP' })[s] || s.toUpperCase()
    },
    formatOos(t) {
      if (t.oos_ic_mean == null) return '—'
      return 'IC ' + Number(t.oos_ic_mean).toFixed(3)
    },
    fmtTrainId(id) {
      const s = String(id || '')
      return s.length >= 8 ? `${s.slice(0, 4)}-${s.slice(4, 6)}-${s.slice(6, 8)} ${s.slice(9, 11) || ''}:${s.slice(11, 13) || ''}`.trim() : s
    },
    paramSummary(t) {
      const p = t.params || {}
      const parts = []
      parts.push(({ ml: 'ML学习加权', equal: '等权1/N', icir: 'ICIR', hrp: 'HRP' })[t.scheme || 'ml'] || 'ML')
      parts.push(`持有${t.label_days ?? 5}d`)
      parts.push(`${t.n_folds ?? '—'}折`)
      return parts.join(' · ')
    },
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
    icClass(v) {
      const n = Number(v)
      if (!Number.isFinite(n)) return ''
      return n >= 0 ? 'ic-pos' : 'ic-neg'
    },
    selectTrain(trainId) {
      this.ml.selected = trainId
      this.viewMl(trainId)
    },
    async viewMl(trainId, silent) {
      try {
        const d = await api('/api/alphaagent/stacking/trainings/' + encodeURIComponent(trainId))
        if (!silent || this.ml.selected === trainId) {
          this.ml.selected = trainId
          this.ml.detail = d
          if (Array.isArray(d.events) && d.events.length) {
            this.currentEvents = d.events
          }
          if (this.viewMode === 'analytics') this.renderAll()
        }
      } catch (e) {
        if (!silent) this.ml.error = e.message
      }
    },
    connectEvents(trainId) {
      if (this._eventSource) {
        this._eventSource.close()
        this._eventSource = null
      }
      const es = new EventSource('/api/alphaagent/stacking/trainings/' + encodeURIComponent(trainId) + '/events')
      this._eventSource = es
      es.onmessage = (e) => {
        try {
          const row = JSON.parse(e.data)
          this.appendEvent(row)
          if (row.event === 'session_end') {
            es.close()
            this._eventSource = null
            this.loadMl()
          }
        } catch (_) {}
      }
      es.onerror = () => {
        es.close()
        this._eventSource = null
      }
    },
    appendEvent(ev) {
      if (!this.currentEvents.some(x => x.event === ev.event && x.ts === ev.ts && JSON.stringify(x) === JSON.stringify(ev))) {
        this.currentEvents.push(ev)
      }
      this.$nextTick(() => {
        const container = this.$refs.threadContainer
        if (container) container.scrollTop = container.scrollHeight
      })
    },
    async loadMl() {
      this.loadingList = true
      try {
        this.ml.list = await api('/api/alphaagent/stacking/trainings')
        const running = this.ml.list.find(t => t.status === 'running')
        if (running && (!this.ml.selected || !this.ml.list.some(x => x.train_id === this.ml.selected))) {
          this.selectTrain(running.train_id)
        } else if (!this.ml.selected && this.ml.list.length) {
          this.selectTrain(this.ml.list[0].train_id)
        }
        if (this.ml.list.some(t => t.status === 'running')) {
          this.scheduleMlPoll()
        }
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
        this.showNewModal = false
        this.ml.selected = res.train_id
        this.currentEvents = []
        await this.loadMl()
        this.selectTrain(res.train_id)
      } catch (e) {
        this.ml.error = e.message
      } finally {
        this.ml.starting = false
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
      }, 4000)
    },
    async stopMl(trainId) {
      try {
        await api('/api/alphaagent/stacking/trainings/' + encodeURIComponent(trainId) + '/stop', { method: 'POST' })
        await this.loadMl()
      } catch (e) {
        this.ml.error = e.message
      }
    },
    async loadFactorPool(force = false) {
      if (!force && (this.ml.factorPool.length || this.ml.factorPoolLoading)) return
      this.ml.factorPoolLoading = true
      try {
        const t = Date.now()
        const [prod, cand] = await Promise.all([
          api('/api/alphaagent/factors?library=production&category=technical&t=' + t),
          api('/api/alphaagent/factors?library=candidate&category=technical&t=' + t),
        ])
        const seen = new Set()
        const pool = []
        for (const batch of [prod, cand]) {
          for (const f of (batch?.factors || [])) {
            const name = f.name || f.factor_id
            if (!name || seen.has(name)) continue
            seen.add(name)
            pool.push({ name, library: batch.library === 'production' ? '正式' : '候选', data: f })
          }
        }
        pool.sort((a, b) => a.name.localeCompare(b.name))
        this.ml.factorPool = pool
      } catch (_) {}
      finally {
        this.ml.factorPoolLoading = false
      }
    },
    setAllFactors(on) {
      this.ml.form.include_factors = on ? this.ml.factorPool.map(f => f.name) : []
    },
    poolFactor(name) {
      return (this.ml.factorPool || []).find(f => f.name === name) || null
    },
    toggleFactorCard(f, evt) {
      if (!f) return
      if (this.factorCard && this.factorCard.name === f.name) {
        this.factorCard = null
        return
      }
      this.factorCard = f
      const rect = evt.currentTarget.getBoundingClientRect()
      const W = 380
      let x = rect.right + 12
      if (x + W > window.innerWidth - 8) x = Math.max(8, rect.left - W - 12)
      const y = Math.min(rect.top - 6, window.innerHeight - 140)
      this.factorCardPos = { x: Math.max(8, x), y: Math.max(8, y) }
    },
    async recommendFactors() {
      if (this.ml.recommending) return
      this.ml.recommending = true
      this.viewMode = 'timeline'
      this.appendEvent({
        event: 'tool_call',
        ts: new Date().toLocaleTimeString(),
        name: 'recommend_mrmr_factors',
        title: '调用 Tool: recommend_mrmr_factors(k=8)',
        message: '正在使用 mRMR 贪心算法在统一因子库中计算 IC 与两两相关度，挑选互补因子...',
      })
      try {
        const res = await api('/api/alphaagent/stacking/recommend', {
          method: 'POST',
          body: {
            label_days: this.ml.form.label_days,
            max_corr: this.ml.form.max_corr,
            no_candidate: this.ml.form.no_candidate,
            include_factors: this.ml.form.include_factors.length ? [...this.ml.form.include_factors] : null,
            k: 8,
          },
        })
        const names = (res.ranking || []).map(r => r.name).filter(Boolean)
        if (names.length) this.ml.form.include_factors = names
        this.appendEvent({
          event: 'tool_result',
          ts: new Date().toLocaleTimeString(),
          name: 'recommend_mrmr_factors',
          title: 'Tool: recommend_mrmr_factors 执行就绪',
          summary: res.summary || `mRMR 算法已精选出 ${names.length} 个互补因子`,
          recommended: names,
          window: res.window || 'mining 窗口',
          ranking: res.ranking || [],
        })
      } catch (e) {
        this.ml.error = '推荐失败：' + (e.message || e)
        this.appendEvent({
          event: 'tool_result',
          ts: new Date().toLocaleTimeString(),
          name: 'recommend_mrmr_factors',
          title: 'Tool: recommend_mrmr_factors 执行异常',
          error: String(e.message || e),
        })
      } finally {
        this.ml.recommending = false
      }
    },
    async loadCompositeFactors(force = false) {
      if (!force && this._cfLoaded) return
      try {
        this.compositeList = await api('/api/alphaagent/composite-factors')
        this._cfLoaded = true
      } catch (_) {}
    },
    async viewComposite(id) {
      try {
        this.compositeDetail = await api('/api/alphaagent/composite-factors/' + encodeURIComponent(id))
        this.compositeSelected = id
      } catch (e) {
        this.ml.error = e.message
      }
    },
    async saveCompositeFactor(t) {
      if (!t?.out_dir || this.savingCompositeId) return
      this.savingCompositeId = t.train_id
      try {
        const res = await api('/api/alphaagent/composite-factors', {
          method: 'POST',
          body: { out_dir: t.out_dir },
        })
        await this.loadCompositeFactors(true)
        this.showCfDrawer = true
        this.viewComposite(res.id)
      } catch (e) {
        this.ml.error = '保存组合因子失败：' + (e.message || e)
      } finally {
        this.savingCompositeId = ''
      }
    },
    setContribMetric(key) {
      this.contribMetric = key
      for (const kind of this.contribKinds) this.renderContribution(kind)
    },
    weightRows(kind) {
      return (this.activeReport?.feature_weights?.[kind] || []).slice(0, 15)
    },
    contribRows(kind) {
      return (this.activeReport?.feature_contribution?.[kind] || []).slice(0, 15)
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
    renderContribution(kind) {
      const field = this.contribMetric
      const rows = (this.activeReport?.feature_contribution?.[kind] || [])
        .filter(r => r[field] != null)
        .sort((a, b) => b[field] - a[field])
        .slice(0, 15)
      const sorted = [...rows].reverse()
      const c = chart('ml-contrib-' + kind)
      if (!c) return
      c.setOption({
        tooltip: { trigger: 'item', formatter: p => `${p.name}：<b>${Number(p.value).toFixed(4)}</b>` },
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
    renderSubsetCurve() {
      const curve = this.activeReport?.subset_curve || []
      if (!curve.length) return
      const c = chart('ml-subset-curve')
      if (!c) return
      c.setOption({
        tooltip: { trigger: 'axis' },
        grid: { left: 44, right: 20, top: 28, bottom: 24 },
        xAxis: { type: 'category', data: curve.map(r => 'k=' + r.k), axisLabel: { ...AXIS_LABEL } },
        yAxis: { type: 'value', axisLabel: { ...AXIS_LABEL }, scale: true },
        series: [{
          name: 'Blended IC', type: 'line', data: curve.map(r => r.blended), showSymbol: true, lineStyle: { width: 2.5 }, itemStyle: { color: '#4f8cff' },
        }],
      }, true)
    },
    renderDecay() {
      const rows = this.decayRows
      if (!rows.length) return
      const sorted = [...rows].reverse()
      const c = chart('ml-decay-chart')
      if (!c) return
      c.setOption({
        tooltip: { trigger: 'axis' },
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
        for (const kind of this.weightKinds) this.renderWeights(kind)
        for (const kind of this.contribKinds) this.renderContribution(kind)
        this.renderSubsetCurve()
        this.renderDecay()
      })
    },
  },
}
</script>

<style scoped>
.ml-agent-shell { height: 100%; min-height: 600px; }
.ml-sidebar { min-width: 260px; max-width: 280px; }
.ml-main { display: flex; flex-direction: column; height: 100%; min-width: 0; }
.ml-header { padding: 12px 20px; }
.ml-thread { padding: 20px 24px; flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 14px; }

.ml-safe-badge {
  color: #4fc3a1; font-size: 11px;
  background: rgb(79 195 161 / 0.12);
  border: 1px solid rgb(79 195 161 / 0.35);
  border-radius: 6px; padding: 4px 9px; font-weight: 500;
}

/* 置顶运行中卡片 */
.ml-running-pinned-card {
  margin: 0 8px 10px;
  padding: 10px 12px;
  border-radius: 8px;
  background: rgb(67 209 122 / 0.12);
  border: 1px solid rgb(67 209 122 / 0.4);
  cursor: pointer;
  transition: all .15s;
}
.ml-running-pinned-card:hover, .ml-running-pinned-card.active {
  background: rgb(67 209 122 / 0.2);
  border-color: #43d17a;
  box-shadow: 0 0 12px rgb(67 209 122 / 0.25);
}
.ml-running-pinned-head {
  display: flex; align-items: center; gap: 6px; font-size: 11px;
}
.ml-running-pinned-head strong { color: #43d17a; }
.ml-pulse-tag {
  margin-left: auto; font-size: 9px; padding: 1px 5px; border-radius: 4px;
  background: #43d17a; color: #000; font-weight: 700;
  animation: agentPulse 1.2s infinite;
}
.ml-running-pinned-body {
  margin-top: 4px; display: flex; flex-direction: column; gap: 2px;
}
.ml-running-pinned-body b { font-size: 12px; color: var(--text); }
.ml-running-pinned-body small {
  color: #a8d5b8; font-size: 10px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}

.session-item.is-running {
  border-color: rgb(67 209 122 / 0.3);
  background: rgb(67 209 122 / 0.06);
}
.ml-tag-running {
  font-size: 9px; font-style: normal; padding: 1px 4px; border-radius: 3px;
  background: rgb(67 209 122 / 0.25); color: #43d17a; margin-left: 4px;
}

/* 正在查看历史任务时的跳回横幅 */
.ml-viewing-history-alert {
  display: flex; align-items: center; gap: 10px;
  padding: 8px 16px;
  background: rgb(245 189 79 / 0.14);
  border-bottom: 1px solid rgb(245 189 79 / 0.35);
  color: #f5bd4f;
  font-size: 12px;
  cursor: pointer;
}
.ml-viewing-history-alert:hover {
  background: rgb(245 189 79 / 0.22);
}
.ml-jump-back-btn {
  margin-left: auto; padding: 3px 10px; border-radius: 5px;
  background: #f5bd4f; color: #000; font-weight: 600; border: 0;
  font-size: 11px; cursor: pointer;
}

/* 结构化时间线卡片 */
.ml-card {
  border: 1px solid var(--line);
  border-radius: 10px;
  background: var(--bg-soft, #131a29);
  padding: 12px 14px;
}
.ml-card-head {
  display: flex; align-items: center; gap: 8px; margin-bottom: 8px;
}
.ml-card-tag {
  font-size: 10px; padding: 2px 6px; border-radius: 4px;
  background: rgb(79 140 255 / 0.15); color: #7fb0ff;
}
.ml-card-tag.prod { background: rgb(79 195 161 / 0.15); color: #4fc3a1; }
.ml-card-tag.bad { background: rgb(239 107 115 / 0.15); color: #ef6b73; }
.ml-card-time { margin-left: auto; color: var(--muted); font-size: 10px; font-family: var(--font-mono); }

.ml-meta-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 6px;
  font-size: 11px; color: var(--text);
}
.ml-meta-grid i { color: var(--muted); font-style: normal; margin-right: 4px; }

.ml-stage-card {
  display: flex; align-items: center; gap: 12px; padding: 10px 14px;
  border: 1px solid var(--line); border-radius: 8px; background: rgba(255,255,255,0.02);
}
.ml-stage-icon {
  width: 24px; height: 24px; border-radius: 50%;
  display: grid; place-items: center; font-size: 12px;
  background: rgb(79 140 255 / 0.15); color: #7fb0ff;
}
.ml-stage-content strong { display: block; font-size: 12px; }
.ml-stage-content p { margin: 2px 0 0; font-size: 11px; color: var(--muted); }

.ml-think-bubble {
  margin-left: 0 !important;
  background: rgb(199 146 234 / 0.04) !important;
  border-color: rgb(199 146 234 / 0.25) !important;
}

.ml-factor-chips {
  display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px;
}
.ml-chip {
  font-size: 11px; font-family: var(--font-mono);
  padding: 3px 8px; border-radius: 5px;
  background: rgb(79 140 255 / 0.12); color: #a3c5ff;
  cursor: pointer; border: 1px solid transparent;
}
.ml-chip:hover { border-color: var(--accent); }

.ml-folds-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 8px; margin-bottom: 8px;
}
.ml-fold-item {
  padding: 6px 8px; border-radius: 6px; background: rgba(0,0,0,0.2);
  display: flex; flex-direction: column; gap: 2px; font-size: 11px;
}
.ml-top-weights { font-size: 11px; color: var(--muted); }
.ml-top-weights span { margin-right: 8px; color: #c6d2e8; }

.ml-drop-details summary { cursor: pointer; color: var(--muted); font-size: 11px; }
.ml-drop-item { font-size: 11px; color: var(--muted); margin-top: 3px; }

.ml-llm-summary { display: flex; flex-direction: column; gap: 8px; }
.ml-llm-sec { border-left: 2px solid rgb(79 140 255 / 0.4); padding-left: 8px; }
.ml-llm-sec b { color: #7fb0ff; font-size: 11px; display: block; margin-bottom: 2px; }
.ml-llm-sec p { margin: 0; font-size: 12px; line-height: 1.6; color: var(--text); }
.ml-llm-sec ul { margin: 0; padding-left: 16px; }
.ml-llm-sec li { font-size: 12px; line-height: 1.6; color: var(--text); }
.ml-llm-sec ul.ml-llm-risk li { color: #e8c491; }

.ml-analytics-panel { padding: 18px 24px; }
.ml-drawer-btn {
  width: 100%; padding: 8px; border: 1px solid var(--line); border-radius: 6px;
  background: transparent; color: var(--muted); font-size: 11px; cursor: pointer;
}
.ml-drawer-btn:hover { color: var(--text); border-color: var(--accent); }

/* 弹窗与抽屉 */
.ml-modal-mask, .ml-drawer-mask {
  position: fixed; inset: 0; z-index: 9999;
  background: rgba(0, 0, 0, 0.6); backdrop-filter: blur(2px);
  display: flex; justify-content: center; align-items: center;
}
.ml-modal {
  width: min(90vw, 760px); max-height: 85vh; border-radius: 12px;
  background: var(--bg-soft, #161f30); border: 1px solid var(--line);
  display: flex; flex-direction: column; overflow: hidden;
}
.ml-modal-head, .ml-drawer-head {
  display: flex; justify-content: space-between; align-items: center;
  padding: 14px 18px; border-bottom: 1px solid var(--line);
}
.ml-modal-close {
  background: transparent; border: 0; color: var(--muted); font-size: 18px; cursor: pointer;
}
.ml-modal-body, .ml-drawer-body { padding: 16px 18px; overflow-y: auto; flex: 1; }
.ml-modal-foot {
  display: flex; justify-content: flex-end; gap: 10px;
  padding: 12px 18px; border-top: 1px solid var(--line);
}
.ml-drawer {
  position: absolute; right: 0; top: 0; bottom: 0; width: min(85vw, 560px);
  background: var(--bg-soft, #161f30); border-left: 1px solid var(--line);
  display: flex; flex-direction: column;
}

/* 因子选择器列表 */
.ml-factor-list {
  max-height: 180px; overflow-y: auto; border: 1px solid var(--line);
  border-radius: 6px; padding: 6px; display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 4px;
}
.ml-factor-row { display: flex; align-items: center; gap: 6px; font-size: 11px; }
.ml-factor-name { cursor: pointer; text-overflow: ellipsis; overflow: hidden; white-space: nowrap; }
.ml-factor-name:hover { color: #7fb0ff; text-decoration: underline; }
.ml-factor-lib { font-size: 9px; color: var(--muted); font-style: normal; margin-left: auto; }
</style>
