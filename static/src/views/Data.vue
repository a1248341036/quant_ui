<template>
  <section>
    <div class="card">
      <div class="cne-head">
        <div>
          <h3>数据湖看板（CNE）</h3>
          <p class="muted">CNE 数据湖看板 · 覆盖热力、数据集、跑批、质量总览</p>
        </div>
        <a class="ghost" :href="cneUrl" target="_blank" rel="noopener">新窗口打开 ↗</a>
      </div>
      <iframe :src="cneUrl" class="cne-frame" title="CNE 数据湖看板" loading="lazy" referrerpolicy="no-referrer"></iframe>
      <p class="muted">加载失败？请确认数据湖服务已启动（<code>cne serve --config configs/cnequity.quant_dataset.toml</code>，默认端口 8787）。</p>
    </div>
    <template v-if="false">
    <template v-if="false">
    <div class="card">
      <h3>数据源状态</h3>
      <div class="table-wrap"><table><tr><th>数据</th><th>位置</th><th>状态</th><th>大小MB</th><th>说明</th></tr>
      <tr v-for="(v,k) in statusRows" :key="k">
        <td>{{v.name}}</td>
        <td class="mono">{{v.loc}}</td>
        <td :class="v.exists?'ok':'err'">{{v.exists?'存在':'缺失'}}</td>
        <td>{{v.size??'-'}}</td>
        <td>{{v.desc}}<span class="muted"><br>{{v.source}} · {{v.update}}</span></td>
      </tr></table></div>
      <p class="muted" v-if="meta.last_update">上次刷新：{{meta.last_update}}</p>
      <p class="muted">主交易面板截至：{{meta.end || store.panelInfo.last_date || '-'}} · AlphaAgent 面板截至：{{store.panelInfo.alphaagent_last_date || '-'}} · 代码 {{meta.n_codes}} · 行数 {{meta.n_rows}}</p>
    </div>
    <div class="card">
      <h3>数据更新说明</h3>
      <ul class="muted">
        <li>「开始更新」会刷新：股票池、指数、行业分类、股票日线、ETF、场外基金净值，并同步 PostgreSQL <code>stock_daily</code>、重建基金衍生面板。</li>
        <li>「增量」只追加最新行情；「全量重建」重新抓取全部日线，耗时更长。</li>
        <li>舆情数据不跟随一键更新，由 <code>~/quant/sentiment-mvp/run_pipeline.py daily</code> 独立更新。</li>
        <li>状态只表示文件/表是否存在；数据新旧以「上次刷新 / 数据截至」为准。</li>
        <li>更新带内存保护：启动前可用内存低于约 700MB 会拒绝启动；运行中低于约 500MB 会自动中止。</li>
      </ul>
    </div>
    <div class="card">
      <h3>更新数据</h3>
      <div class="form-grid">
        <label class="field"><span>模式</span><select v-model="updateMode"><option value="incremental">增量（推荐）</option><option value="full">全量重建</option></select></label>
      </div>
      <div class="form-actions"><button class="primary" @click="startUpdate" :disabled="updating"><span v-if="updating" class="spinner"></span>{{updating ? '更新中…' : '开始更新'}}</button></div>
      <div class="progress" v-if="updating"><div class="bar" :style="{width: (updateProgress*100)+'%'}"></div></div>
      <p class="muted" v-if="updating">{{updateText}}</p>
      <p class="ok" v-if="updateDone">更新完成：{{updateResult?.n_codes}} 只 · {{updateResult?.n_rows}} 行</p>
      <p class="err" v-if="updateError">{{updateError}}</p>
      <p class="muted">本次更新含：行情源 + PostgreSQL 日线同步 + 基金衍生面板重建</p>
    </div>
    <div class="card">
      <div class="config-update-head">
        <div>
          <h3>配置化更新</h3>
          <p class="muted">按任务配置独立执行；数据面板会使用当前上游日线重建。</p>
        </div>
        <button class="ghost" @click="loadConfiguredTasks" :disabled="loadingConfigs">刷新配置</button>
      </div>
      <div v-if="configError" class="err">{{ configError }}</div>
      <div v-for="(tasks, group) in configTaskGroups" :key="group" class="config-task-group">
        <h4>{{ group }}</h4>
        <div v-for="task in tasks" :key="task.id" class="config-task">
          <div class="config-task-main">
            <strong>{{ task.name }}</strong>
            <small class="mono">{{ task.script }}</small>
            <small v-if="task.state?.status === 'success'" class="ok">已完成 {{ formatTaskTime(task.state.finished_at) }}</small>
            <small v-else-if="task.state?.status === 'failed'" class="err">失败：{{ task.state.error }}</small>
            <small v-else-if="task.state?.running" class="muted">正在运行 {{ formatTaskTime(task.state.started_at) }}</small>
          </div>
          <div v-if="Object.keys(task.params || {}).length" class="config-params">
            <label v-for="(type, name) in task.params" :key="name">
              <span>{{ name }}<b v-if="(task.required || []).includes(name)">*</b></span>
              <input v-if="type !== 'bool'" v-model="taskParams[task.id][name]" :type="type === 'int' || type === 'float' ? 'number' : 'text'">
              <input v-else v-model="taskParams[task.id][name]" type="checkbox">
            </label>
          </div>
          <button class="primary config-update-button" :disabled="task.state?.running || configTaskRunning" @click="runConfiguredTask(task)">
            <span v-if="task.state?.running" class="spinner"></span>{{ task.state?.running ? '更新中…' : '更新' }}
          </button>
        </div>
      </div>
    </div>
    </template>
    </template>
  </section>
</template>

<script>
import { store } from '../store/index.js'
import { api } from '../utils/api.js'

export default {
  name: 'DataView',
  data() {
    return {
      store,
      statusRows: [],
      meta: {},
      updateMode: 'incremental',
      updating: false,
      updateProgress: 0,
      updateText: '',
      updateDone: false,
      updateResult: null,
      updateError: '',
      configuredTasks: [],
      taskParams: {},
      configError: '',
      loadingConfigs: false,
      configTimer: null,
    }
  },
  mounted() {
    // 下架期间不再请求旧数据源 API；恢复时取消注释下方三行
    // this.loadStatus()
    // this.loadDataInfo()
    // this.loadConfiguredTasks()
    // this.configTimer = setInterval(this.loadConfiguredTasks, 3000)
  },
  beforeUnmount() {
    if (this.configTimer) clearInterval(this.configTimer)
  },
  computed: {
    cneUrl() {
      return 'http://127.0.0.1:8787/'
    },
    configTaskGroups() {
      return this.configuredTasks.reduce((groups, task) => {
        const group = task.group || '其他'
        ;(groups[group] || (groups[group] = [])).push(task)
        return groups
      }, {})
    },
    configTaskRunning() {
      return this.configuredTasks.some(task => task.state?.running)
    },
  },
  methods: {
    async loadStatus() {
      try {
        const st = await api('/api/data/status');
        this.meta = st.meta || {};
        const rows = [];
        for (const [name, src] of Object.entries(st)) {
          if (name === 'meta') continue;
          for (const [loc, s] of Object.entries(src)) {
            let desc = s.desc || '';
            if (s.info) {
              if (s.info.last_date) desc += ' · 截至 ' + s.info.last_date;
              if (s.info.n_rows != null) desc += ' · 约 ' + Number(s.info.n_rows).toLocaleString() + ' 行';
            }
            rows.push({
              name: s.name || name, loc: s.path || loc, exists: s.exists,
              size: s.size_mb, desc, source: s.source || '', update: s.update || ''
            });
          }
        }
        this.statusRows = rows;
        // 同步到 store.meta（供其他页面引用）
        store.meta = this.meta;
      } catch (e) {}
    },
    async loadDataInfo() {
      try { store.panelInfo = await api('/api/data/panel-info'); } catch (e) {}
    },
    async startUpdate() {
      this.updating = true; this.updateDone = false; this.updateError = '';
      try {
        await api('/api/data/update', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ mode: this.updateMode }) });
        const poll = setInterval(async () => {
          const s = await api('/api/data/update/status');
          this.updateProgress = s.progress; this.updateText = s.text;
          if (!s.running) {
            clearInterval(poll); this.updating = false;
            if (s.error) this.updateError = s.error;
            else { this.updateDone = true; this.updateResult = s.result; this.loadStatus(); this.loadDataInfo(); }
          }
        }, 2000);
      } catch (e) { this.updating = false; this.updateError = e.message; }
    },
    async loadConfiguredTasks() {
      if (this.loadingConfigs) return
      this.loadingConfigs = true
      try {
        const payload = await api('/api/data/update-configs?t=' + Date.now())
        this.configuredTasks = payload.items || []
        for (const task of this.configuredTasks) {
          if (!this.taskParams[task.id]) {
            const values = {}
            for (const [name, type] of Object.entries(task.params || {})) values[name] = type === 'bool' ? false : ''
            this.taskParams[task.id] = values
          }
        }
        this.configError = ''
      } catch (e) { this.configError = '读取更新配置失败: ' + e.message }
      finally { this.loadingConfigs = false }
    },
    async runConfiguredTask(task) {
      this.configError = ''
      try {
        await api('/api/data/update-configs/' + encodeURIComponent(task.id), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ params: this.taskParams[task.id] || {} }),
        })
        await this.loadConfiguredTasks()
      } catch (e) { this.configError = task.name + '启动失败: ' + e.message }
    },
    formatTaskTime(value) {
      return value ? String(value).replace('T', ' ').slice(0, 19) : ''
    },
  },
}
</script>

<style scoped>
.cne-head { display:flex; align-items:flex-start; justify-content:space-between; gap:16px; }
.cne-head h3 { margin-bottom:4px; }
.cne-frame { width:100%; height:1200px; border:1px solid var(--line); border-radius:8px; background:var(--bg,#fff); margin-top:12px; }
.config-update-head { display:flex; align-items:flex-start; justify-content:space-between; gap:16px; }
.config-update-head h3 { margin-bottom:4px; }
.config-task-group { margin-top:20px; }
.config-task-group h4 { margin:0 0 8px; color:var(--muted); font-size:12px; }
.config-task { display:grid; grid-template-columns:minmax(170px, 1fr) minmax(180px, 1.4fr) auto; gap:12px; align-items:center; padding:11px 0; border-top:1px solid var(--line); }
.config-task-main { min-width:0; display:flex; flex-direction:column; gap:3px; }
.config-task-main strong { font-size:13px; }
.config-task-main small { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.config-params { display:flex; flex-wrap:wrap; gap:7px; }
.config-params label { display:flex; align-items:center; gap:5px; color:var(--muted); font-size:11px; }
.config-params input[type="text"], .config-params input[type="number"] { width:88px; height:27px; padding:3px 6px; }
.config-params input[type="checkbox"] { width:15px; height:15px; }
.config-params b { color:var(--err); margin-left:2px; }
.config-update-button { min-width:72px; }
@media (max-width: 760px) { .config-task { grid-template-columns:1fr auto; } .config-params { grid-column:1 / -1; } }
</style>
