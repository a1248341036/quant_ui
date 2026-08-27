<template>
  <div class="card">
    <h3>研究代码（qweave / Python）</h3>
    <p class="muted">这里运行 qweave 因子研究：代码只需要返回 qweave 表达式列表，不再绑定旧的 STRATEGIES / EVENT_STRATEGIES 交易引擎。</p>
    <div class="form-grid" style="margin-top:12px">
      <label class="field wide"><span>研究代码模板</span>
        <select v-model="code.tplName" @change="loadTemplate">
          <option value="" disabled>选择预制研究代码…</option>
          <option v-for="s in code.tplOptions" :key="s.value" :value="s.value">{{s.label}}</option>
        </select>
      </label>
      <label class="field"><span>保存名称</span><input v-model="code.savedName" placeholder="如：我的双均线 v1"></label>
      <label class="field"><span>已保存</span>
        <select v-model="code.savedName" @change="loadSaved">
          <option value="" disabled>选择保存项…</option>
          <option v-for="s in code.savedList" :key="s.name" :value="s.name">{{s.name}}</option>
        </select>
      </label>
    </div>
    <div class="tpl-info" v-if="code.tplInfo">
      <div class="tpl-info-row">
        <span class="chip">{{code.tplInfo.group}}</span>
        <strong>{{code.tplInfo.name}}</strong>
        <span class="muted">{{code.tplInfo.desc}}</span>
      </div>
      <div class="tpl-info-row">
        <span class="muted">方向：{{code.tplInfo.ascending ? '买低分' : '买高分'}}</span>
        <span class="muted">因子：{{code.tplInfo.factor}}</span>
        <span class="muted">改文件头「参数」区即可，不用动函数体</span>
      </div>
    </div>
    <div class="tpl-note">qweave 输入为长表行情（date/code/open/high/low/close/volume/amount/turnover/vwap），自动生成未来收益标签并计算 IC、分组收益、换手和覆盖率。研究代码只允许使用当前及之前的数据，避免未来函数。</div>
    <details class="helpers">
      <summary>📖 常用函数说明（点开查看）</summary>
      <div class="helpers-body" v-html="helpersHtml"></div>
    </details>
    <div class="form-actions" style="margin-top:6px">
      <span class="ok left" v-if="codeSaveMsg">{{codeSaveMsg}}</span>
      <button class="ghost" @click="parseCode(false)" :disabled="codeRunning">解析因子</button>
      <button class="ghost" @click="saveCode" :disabled="codeRunning">保存</button>
      <button class="ghost" @click="loadCodeDefault">恢复默认</button>
    </div>
    <div style="margin-top:14px">
      <code-editor v-model="code.code"></code-editor>
    </div>
  </div>

  <div class="card">
    <h3>运行参数</h3>
    <div class="form-section">
      <div class="form-section-title">研究标的</div>
      <div class="form-grid">
        <label class="field"><span>股票池</span><select v-model="code.universe"><option>科技TMT</option><option>沪深300+中证500+中证1000</option><option>ETF</option><option>场外基金</option></select></label>
        <label class="field"><span>Alpha 集</span><select v-model="code.alpha_set"><option value="alpha158">Alpha158</option><option value="alpha101">Alpha101</option><option value="alpha191">Alpha191</option><option value="custom">自定义代码</option></select></label>
        <label class="field"><span>因子数量上限</span><input type="number" v-model.number="code.alpha_limit" min="1" placeholder="自定义代码忽略"></label>
        <label class="field wide"><span>回测信号因子</span><select v-model="code.score_factor"><option value="">自动使用评估排名第一的因子</option><option v-for="f in code.strategies" :key="f" :value="f">{{f}}</option></select></label>
        <label class="field"><span>回测 TopN</span><input type="number" v-model.number="code.top_n" min="1"></label>
        <label class="field"><span>选股数量模式</span><select v-model="code.selection_mode"><option value="top_n">固定 TopN</option><option value="top_pct">按有效股票比例</option></select></label>
        <label class="field" v-if="code.selection_mode === 'top_pct'"><span>动态比例</span><input type="number" v-model.number="code.selection_pct" min="0.01" max="1" step="0.01"><small class="muted">例如 0.10 = 每次选有效股票的前 10%</small></label>
        <label class="field"><span>最少持仓数</span><input type="number" v-model.number="code.min_positions" min="1"></label>
        <label class="field"><span>最多持仓数（可选）</span><input type="number" v-model.number="code.max_positions" min="1" placeholder="不限制"></label>
        <label class="field"><span>买入费率%</span><input type="number" v-model.number="code.buy_cost_pct" step="0.01" min="0"></label>
        <label class="field"><span>卖出费率%</span><input type="number" v-model.number="code.sell_cost_pct" step="0.01" min="0"></label>
        <label class="field"><span>滑点 bps</span><input type="number" v-model.number="code.slippage_bps" step="1" min="0"></label>
        <label class="field"><span>单笔成交参与率</span><input type="number" v-model.number="code.max_participation" step="0.01" min="0" max="1" placeholder="0=不限"></label>
        <label class="field"><span>成交额过滤分位</span><input type="number" v-model.number="code.amount_q" step="0.05" min="0" max="1"></label>
        <label class="field"><span>回测频率</span><select v-model="code.freq"><option value="daily">每日</option><option value="weekly">每周</option><option value="monthly">每月</option></select></label>
        <label class="field"><span>回测资金</span><input type="number" v-model.number="code.capital" step="10000"></label>
      </div>
    </div>
    <div class="form-section">
      <div class="form-section-title">时间与标签</div>
      <div class="form-grid">
        <label class="field"><span>开始</span><input type="date" v-model="code.start"></label>
        <label class="field"><span>结束</span><input type="date" v-model="code.end"></label>
        <label class="field wide"><span>标签 horizon（交易日，逗号分隔）</span><input v-model="code.horizonsText" placeholder="1,5,10,20"></label>
      </div>
    </div>
    <div class="form-section">
      <div class="form-section-title">评估参数</div>
      <div class="form-grid">
        <label class="field-check"><input type="checkbox" v-model="code.exclude"> 剔除科创/创业</label>
        <label class="field"><span>分位数</span><input type="number" v-model.number="code.quantiles" min="2" max="20"></label>
        <label class="field"><span>最小截面数</span><input type="number" v-model.number="code.min_cs_count" min="1"></label>
        <label class="field"><span>成本 bps</span><input type="number" v-model.number="code.cost_bps" step="1" min="0"></label>
      </div>
    </div>
    <div class="form-actions">
      <p class="muted left">当前因子：<span v-if="code.strategies.length">{{code.strategies.join('、')}}</span><span v-else>（未解析，点「解析因子」）</span></p>
      <p class="err left" v-if="codeError">{{codeError}}</p>
      <button class="primary" @click="runCode" :disabled="codeRunning"><span v-if="codeRunning" class="spinner"></span>{{codeRunning ? '研究中…' : '运行 qweave'}}</button>
      <button class="primary" @click="runQweaveBacktest" :disabled="codeRunning"><span v-if="codeRunning" class="spinner"></span>{{codeRunning ? '回测中…' : '研究并回测 TopN'}}</button>
    </div>
  </div>

  <div v-if="codeResult">
    <div class="cards"><div class="metric"><div class="label">因子数</div><div class="value">{{codeResult.factor_count}}</div></div><div class="metric"><div class="label">样本行数</div><div class="value">{{codeResult.rows}}</div></div><div class="metric"><div class="label">研究区间</div><div class="value">{{code.start}} ~ {{code.end}}</div></div><div class="metric"><div class="label">输出目录</div><div class="value" style="font-size:12px">{{codeResult.run_dir}}</div></div></div>
    <div class="card"><h3>Top 因子评估</h3><div class="table-wrap"><table><tr><th>因子</th><th>H</th><th>IC 均值</th><th>Rank IC</th><th>ICIR</th><th>多空年化</th><th>多空 IR</th><th>换手</th></tr><tr v-for="(s,i) in codeResult.summary" :key="i"><td>{{s.factor}}</td><td>{{s.horizon}}</td><td :class="sign(s.ic_mean)">{{fmt(s.ic_mean,4)}}</td><td>{{fmt(s.rank_ic_mean,4)}}</td><td>{{fmt(s.ic_ir,3)}}</td><td>{{pct(s.ls_net_ann)}}</td><td>{{fmt(s.ls_ir,3)}}</td><td>{{fmt(s.ls_turnover,3)}}</td></tr></table></div></div>
    <div class="card"><h3>最新交易日股票因子排名</h3><p class="muted">上面的 Top 因子是“因子级”统计，不会对应单个股票。这里展示 Top 因子在最新交易日的具体股票代码、因子值和截面排名。</p><div class="table-wrap"><table><tr><th>日期</th><th>方向</th><th>排名</th><th>代码</th><th>因子</th><th>因子值</th><th>截面分位</th></tr><tr v-for="(r,i) in topFactorRows()" :key="i"><td>{{r.date}}</td><td>{{r.side === 'top' ? '高值' : '低值'}}</td><td>{{r.rank}}</td><td>{{stock(r.code)}}</td><td>{{r.factor}}</td><td>{{fmt(r.value,6)}}</td><td>{{pct(r.percentile)}}</td></tr></table></div><div v-if="!topFactorRows().length" class="empty">暂无股票级因子明细</div></div>
    <div class="card" v-if="codeResult.backtest"><h3>qweave 信号 → 日线交易回测</h3><p class="muted">因子：{{codeResult.backtest.factor}} · 选股：{{code.selection_mode === 'top_pct' ? '有效股票前 ' + (code.selection_pct * 100).toFixed(1) + '%' : '固定 TopN=' + code.top_n}} · {{code.freq}} · qweave 输出的股票级分数已交给交易执行层，模拟 T+1、整手、费用和交易限制。</p><div class="cards"><div class="metric" v-for="(v,k) in codeResult.backtest.metrics" :key="k"><div class="label">{{k}}</div><div class="value" :class="sign(v)">{{metricText(k,v)}}</div></div></div><div id="qweaveBacktestEquity" class="chart"></div></div>
    <div class="grid" style="grid-template-columns:1fr 1fr"><div class="card"><h3>Rank IC（Top 因子）</h3><div id="codeIc" class="chart"></div></div><div class="card"><h3>分组收益（Top 因子）</h3><div id="codeQuantile" class="chart"></div></div></div>
  </div>
  <div v-else class="card"><div class="empty">编辑代码后点击「跑代码」查看回测结果</div></div>
</template>

<script>
import { store } from '../store/index.js'
import { api } from '../utils/api.js'
import { fmt, pct, sign, stock, metricText, today } from '../utils/format.js'
import { renderLine, renderBar } from '../utils/charts.js'

export default {
  name: 'CodeView',
  data() {
    return {
      store,
      code: {
        code: '', tplName: '', tplOptions: [], presetList: [],
        tplInfo: null,
        strategy: '', strategies: [],
        universe: '沪深300+中证500+中证1000', alpha_set: 'alpha158', alpha_limit: 30,
        horizonsText: '1,5,10,20', quantiles: 10, min_cs_count: 30, cost_bps: 8,
        start: '2022-01-01', end: today(), exclude: true, score_factor: '', top_n: 10, selection_mode: 'top_n', selection_pct: 0.10, min_positions: 1, max_positions: null, capital: 100000, freq: 'weekly', amount_q: 0.2, buy_cost_pct: 0.025, sell_cost_pct: 0.125, slippage_bps: 3, max_participation: 0.10,
        savedName: '', savedList: [],
      },
      codeResult: null, codeRunning: false, codeError: '', codeSaveMsg: '',
      helpersHtml: '',
    };
  },
  mounted() {
    this.loadCodeDefault();
    this.loadHelpers();
  },
  methods: {
    today,
    fmt,
    pct,
    sign,
    metricText,
    stock(code) { return stock(code, undefined, store.names); },
    async loadCodeDefault() {
      try {
        const d = await api('/api/code/qweave/default');
        this.code.code = d.code || '';
        this.code.tplInfo = { name: 'qweave Alpha 研究模板', group: 'qweave', desc: '因子表达式 → 未来收益标签 → IC / 分组 / 换手评估', factor: 'alpha', ascending: false };
        this.code.end = today();
        this.codeError = ''; this.codeSaveMsg = '';
        await this.refreshSaved();
        await this.refreshQweaveTemplates();
        if (!this.code.tplName && this.code.tplOptions.length) this.code.tplName = this.code.tplOptions[0].value;
        await this.parseCode(true);
      } catch (e) { this.codeError = '加载默认代码失败: ' + e.message; }
    },
    buildTplOptions() {
      const presets = (this.code.presetList || []).map(s => ({ value: 'preset:' + s.name, label: '预制 · ' + s.label }));
      const labs = (this.code.savedList || []).filter(s => s.engine === 'qweave').map(s => ({ value: 'lab:' + s.name, label: '已保存 · ' + s.name }));
      this.code.tplOptions = [...presets, ...labs];
    },
    async refreshQweaveTemplates() {
      try {
        const r = await api('/api/code/qweave/templates');
        this.code.presetList = r.items || [];
        this.buildTplOptions();
      } catch (e) { this.codeError = '加载预制代码失败: ' + e.message; }
    },
    async refreshSaved() {
      try { const r = await api('/api/code/saved'); this.code.savedList = r.items || []; this.buildTplOptions(); } catch (e) {}
    },
    async loadTemplate() {
      const v = this.code.tplName;
      if (!v) return;
      this.codeError = '';
      try {
        const isLab = v.startsWith('lab:');
        const isPreset = v.startsWith('preset:');
        const name = isPreset ? v.slice(7) : v.slice(4);
        const r = isLab ? await api('/api/code/saved/' + encodeURIComponent(name))
                        : isPreset ? await api('/api/code/qweave/template?name=' + encodeURIComponent(name))
                                   : await api('/api/code/qweave/default');
        if (r.error) { this.codeError = r.error; return; }
        this.code.code = r.code || '';
        this.code.tplInfo = { name: r.label || r.name || name, group: isPreset ? '预制 qweave' : 'qweave', desc: isPreset ? '可直接运行，也可以继续修改代码' : '已保存的 qweave 研究代码', factor: 'alpha', ascending: false };
        await this.parseCode(true);
        this.codeSaveMsg = '已加载：' + r.name;
        setTimeout(() => { if (this.codeSaveMsg) this.codeSaveMsg = ''; }, 3000);
      } catch (e) { this.codeError = '加载失败: ' + e.message; }
    },
    async parseCode(silent) {
      if (!this.code.code) { if (!silent) alert('代码为空'); return; }
      this.codeRunning = true; this.codeError = '';
      try {
        const r = await api('/api/code/qweave/parse', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ code: this.code.code }) });
        if (r.error || r.ok === false) { this.codeError = r.error || '解析失败'; return; }
        this.code.strategies = r.factors || [];
      } catch (e) { this.codeError = '解析失败: ' + e.message; }
      finally { this.codeRunning = false; }
    },
    codeParams() {
      const horizons = (this.code.horizonsText || '').split(',').map(x => Number(x.trim())).filter(x => Number.isFinite(x) && x > 0);
      return {
        code: this.code.code, universe: this.code.universe, alpha_set: this.code.alpha_set,
        alpha_limit: this.code.alpha_limit, horizons, start: this.code.start, end: this.code.end,
        exclude_kechuang: this.code.exclude, quantiles: this.code.quantiles,
        min_cs_count: this.code.min_cs_count, cost_bps: this.code.cost_bps,
        score_factor: this.code.score_factor, top_n: this.code.top_n,
        selection_mode: this.code.selection_mode, selection_pct: this.code.selection_pct,
        min_positions: this.code.min_positions, max_positions: this.code.max_positions,
        amount_q: this.code.amount_q,
        capital: this.code.capital, freq: this.code.freq,
        buy_cost: (this.code.buy_cost_pct || 0) / 100,
        sell_cost: (this.code.sell_cost_pct || 0) / 100,
        slippage_bps: this.code.slippage_bps || 0,
        max_participation: this.code.max_participation || 0,
      };
    },
    async runCode() {
      if (!this.code.strategies.length) { this.codeError = '请先点「解析因子」'; return; }
      await this.executeQweave(false);
    },
    async runQweaveBacktest() {
      if (!this.code.strategies.length) { this.codeError = '请先点「解析因子」'; return; }
      await this.executeQweave(true);
    },
    async executeQweave(withBacktest) {
      this.codeRunning = true; this.codeError = ''; this.codeSaveMsg = '';
      try {
        const r = await api('/api/code/qweave/run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...this.codeParams(), run_backtest: withBacktest }) });
        if (r.error || r.ok === false) { this.codeError = r.error || '运行失败'; return; }
        if (r.strategies && r.strategies.length) this.code.strategies = r.strategies;
        this.codeResult = r;
        this.$nextTick(() => this.renderCode());
      } catch (e) { this.codeError = '运行失败: ' + e.message; }
      finally { this.codeRunning = false; }
    },
    renderCode() {
      const rows = this.codeResult.summary || [];
      const factor = rows[0] && rows[0].factor;
      const ic = (this.codeResult.ic || []).filter(x => x.factor === factor).slice(-500);
      renderLine('codeIc', [{ name: 'Rank IC', dates: ic.map(x => x.date), values: ic.map(x => x.rank_ic) }]);
      const qr = (this.codeResult.quantile_returns || []).filter(x => x.factor === factor).slice(-this.code.quantiles * 2);
      renderBar('codeQuantile', qr.map(x => 'Q' + x.bin), qr.map(x => x.mean_ret_5));
      if (this.codeResult.backtest) {
        const bt = this.codeResult.backtest;
        renderLine('qweaveBacktestEquity', [{ name: 'qweave score 回测', dates: bt.nav.map(x => x.date), values: bt.nav.map(x => x.value) }]);
      }
    },
    topFactorRows() {
      const factor = this.codeResult && this.codeResult.summary && this.codeResult.summary[0] && this.codeResult.summary[0].factor;
      return (this.codeResult && this.codeResult.latest_factor_rows || []).filter(x => x.factor === factor);
    },
    async saveCode() {
      if (!this.code.savedName.trim()) { this.codeError = '请填保存名称'; return; }
      this.codeError = '';
      try {
        const r = await api('/api/code/save', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: this.code.savedName, code: this.code.code, engine: 'qweave' }) });
        if (r.error) { this.codeError = r.error; return; }
        this.codeSaveMsg = '已保存：' + r.name;
        await this.refreshSaved();
        setTimeout(() => { if (this.codeSaveMsg) this.codeSaveMsg = ''; }, 3000);
      } catch (e) { this.codeError = '保存失败: ' + e.message; }
    },
    async loadSaved() {
      if (!this.code.savedName) return;
      this.codeError = '';
      try {
        const r = await api('/api/code/saved/' + encodeURIComponent(this.code.savedName));
        if (r.error) { this.codeError = r.error; return; }
        this.code.code = r.code || ((r.registry && r.factors) ? r.registry + '\n\n\n' + r.factors : '');
        await this.parseCode(true);
        this.codeSaveMsg = '已载入：' + r.name;
        setTimeout(() => { if (this.codeSaveMsg) this.codeSaveMsg = ''; }, 3000);
      } catch (e) { this.codeError = '载入失败: ' + e.message; }
    },
    async loadHelpers() {
      if (this.helpersHtml) return;
      try {
        const r = await fetch('strategy_helpers.md', { credentials: 'include' });
        if (!r.ok) return;
        this.helpersHtml = this.renderMarkdown(await r.text());
      } catch (e) { /* 说明文档加载失败时静默，不影响页面 */ }
    },
    renderMarkdown(src) {
      const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
      const inline = s => esc(s)
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
        .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
      const lines = src.replace(/\r\n/g, '\n').split('\n');
      let html = '', inCode = false, codeBuf = [], listBuf = [], tableBuf = [], paraBuf = [];
      const flushList = () => {
        if (!listBuf.length) return;
        html += '<ul>' + listBuf.map(x => '<li>' + inline(x) + '</li>').join('') + '</ul>';
        listBuf = [];
      };
      const flushTable = () => {
        if (!tableBuf.length) return;
        const rows = tableBuf.map(r => r.map(c => '<td>' + inline(c.trim()) + '</td>').join(''));
        html += '<table><tr>' + rows[0].replace(/<td>/g, '<th>').replace(/<\/td>/g, '</th>') + '</tr>' +
          rows.slice(1).map(r => '<tr>' + r + '</tr>').join('') + '</table>';
        tableBuf = [];
      };
      const flushPara = () => {
        if (!paraBuf.length) return;
        html += '<p>' + paraBuf.map(inline).join('<br>') + '</p>';
        paraBuf = [];
      };
      const flushAll = () => { flushList(); flushTable(); flushPara(); };
      for (const raw of lines) {
        const line = raw.trim();
        if (line.startsWith('```')) {
          flushAll();
          if (inCode) {
            html += '<pre><code>' + codeBuf.join('\n') + '</code></pre>';
            codeBuf = []; inCode = false;
          } else { inCode = true; }
          continue;
        }
        if (inCode) { codeBuf.push(esc(raw)); continue; }
        if (!line) { flushAll(); continue; }
        if (/^#{1,4}\s+/.test(line)) {
          flushAll();
          const m = line.match(/^(#{1,4})\s+(.*)$/);
          html += '<h' + m[1].length + '>' + inline(m[2]) + '</h' + m[1].length + '>';
          continue;
        }
        if (/^[-*]\s+/.test(line)) { flushTable(); flushPara(); listBuf.push(line.replace(/^[-*]\s+/, '')); continue; }
        if (/^\|[\s:|-]+\|$/.test(line)) { continue; }
        if (/^\|.*\|$/.test(line)) { flushList(); flushPara(); tableBuf.push(line.split('|').filter((_, i, a) => i > 0 && i < a.length - 1)); continue; }
        flushList(); flushTable(); paraBuf.push(raw);
      }
      if (inCode) html += '<pre><code>' + codeBuf.join('\n') + '</code></pre>';
      flushAll();
      return html;
    },
  },
}
</script>
