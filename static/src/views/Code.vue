<template>
  <div class="card">
    <div class="form-actions" style="margin-bottom:8px">
      <h3 style="margin:0">策略代码 <span class="muted" style="font-size:12px">聚宽风格 · 全A主板域 · T日收盘信号 → T+1开盘成交</span></h3>
      <span class="ok left" v-if="codeSaveMsg">{{codeSaveMsg}}</span>
      <span class="err left" v-if="codeError">{{codeError}}</span>
      <button class="ghost" @click="saveCode" :disabled="codeRunning">保存</button>
      <button class="ghost" @click="loadSavedDialog = !loadSavedDialog">载入</button>
      <button class="ghost" @click="runPreflight" :disabled="codeRunning || preflightRunning">
        <span v-if="preflightRunning" class="spinner"></span>{{preflightRunning ? '预检中…' : '🔍 API 预检'}}
      </button>
      <button class="primary" @click="runJq" :disabled="codeRunning">
        <span v-if="codeRunning" class="spinner"></span>{{codeRunning ? '回测中…' : '▶ 运行策略'}}
      </button>
    </div>

    <div v-if="preflightResult" class="card" :style="preflightResult.ok ? 'border-left:4px solid #2e7d32;padding:8px 12px;margin-bottom:8px' : 'border-left:4px solid #c62828;padding:8px 12px;margin-bottom:8px'">
      <strong v-if="preflightResult.ok">✅ {{preflightResult.message}}</strong>
      <template v-else>
        <strong style="color:#c62828">⚠ {{preflightResult.message}}</strong>
        <div style="font-size:12px;color:#888;margin-top:4px">缺失 API 回测会在运行时报错；可对照「可用 API 说明」改写，或提交需求接入。</div>
      </template>
    </div>

    <div class="form-grid" style="margin-bottom:10px;align-items:end">
      <label class="field"><span>初始资金</span><input type="number" v-model.number="code.capital" step="10000" min="10000"></label>
      <label class="field"><span>开始</span><input type="date" v-model="code.start"></label>
      <label class="field"><span>结束</span><input type="date" v-model="code.end"></label>
      <label class="field"><span>保存名称</span><input v-model="code.savedName" placeholder="如：小盘三正 v1"></label>
      <div class="field" style="max-width:120px">
        <span>&nbsp;</span>
        <button class="ghost" style="width:100%" @click="openSettings">⚙ 设置</button>
      </div>
    </div>

    <div v-if="loadSavedDialog" class="form-grid" style="margin-bottom:10px">
      <label class="field wide"><span>已保存策略</span>
        <select v-model="code.savedName" @change="loadSaved">
          <option value="" disabled>选择保存项…</option>
          <option v-for="s in code.savedList" :key="s.name" :value="s.name">{{s.name}} <span v-if="s.saved_at">({{(s.saved_at||'').slice(0,10)}})</span></option>
        </select>
      </label>
    </div>

    <code-editor v-model="code.code"></code-editor>

    <details class="helpers" style="margin-top:8px">
      <summary>📖 可用 API 说明（点开查看）</summary>
      <div class="helpers-body">
        <p><strong>任务注册</strong>（initialize 内调用；每日任务按注册的 time 先后执行，同时刻按注册顺序；reference_security 等参数接受但忽略）：</p>
        <ul>
          <li><code>run_daily(func, time='9:30')</code> — 每日执行（支持 'before_open'/'open'/'after_close'/'HH:MM'）</li>
          <li><code>run_weekly(func, weekday=1, time='10:00')</code> — 每周第 N 个交易日（1=周内首个交易日，负数=倒数）</li>
          <li><code>run_monthly(func, monthday=1, time='9:30')</code> — 每月第 N 个交易日（缺省首个，负数=倒数）</li>
        </ul>
        <p><strong>数据 API</strong>（点时口径，无未来函数）：</p>
        <ul>
          <li><code>get_snapshot(date=None)</code> → DataFrame（index=code）：<code>close/close_raw/open_raw/market_cap(亿)/turnover/st/paused/listed_ok/hl(收盘涨停)/high_limit/low_limit/fin_三正</code></li>
          <li><code>get_fundamentals(query(valuation.code, income.net_profit).filter(...).order_by(valuation.market_cap.asc()).limit(20), date=)</code> — 支持 ==/!=/</<=/>/>=/in_()/between()/&amp;/|</li>
          <li><code>get_price(security, end_date=, count=, fields=['close','open','high','low','pre_close','high_limit','low_limit','volume','money'], panel=False)</code> — 真实价；单标的返回 DataFrame(time×fields)，多标的返回 long 表；指数代码(399101.XSHE/000300.XSHG 等)走本地指数日线</li>
          <li><code>get_factor('TS_MEAN($close, 20)')</code> — AlphaAgent DSL 因子表达式 → 信号日截面 Series(index=code)；<code>$close/$open/$high/$low/$amount/$volume/$turnover_rate/$mv</code> 引用列，TS_*/CS_* 算子，多行表达式末行为输出（与因子实验室同语法）</li>
          <li><code>history(count, unit='1d', field='close', security_list=[...])</code> → {code: list}（'1m' 按日线近似）</li>
          <li><code>get_current_data()[code].paused / .is_st / .name / .high_limit / .low_limit</code></li>
          <li><code>get_index_stocks('000300')</code> — 统一返回全量池（域内全部股票）；<code>get_security_info(code).start_date / .display_name</code></li>
          <li><code>attribute_history(security, count, fields)</code> → DataFrame；<code>get_all_securities()</code> → 全域代码表</li>
        </ul>
        <p><strong>下单</strong>（信号 T 日收盘数据下单 → T+1 开盘撮合，涨停买不进/跌停卖不出，整手）：</p>
        <ul>
          <li><code>order_target_value(code, value)</code> / <code>order_value(code, value)</code>（正买负卖）</li>
          <li><code>order_target_percent(code, pct)</code> / <code>order_target(code, shares)</code> / <code>order_shares(code, delta)</code></li>
        </ul>
        <p><strong>上下文</strong>：<code>context.previous_date</code>（信号日）、<code>context.current_dt</code>（执行日）、
          <code>context.portfolio.cash / .total_value / .positions[code].avg_cost / .price / .value / .total_amount</code>、
          <code>g.xxx</code>（全局变量）、<code>log.info(...)</code>、<code>OrderStatus</code>。
          下单返回订单回执（<code>order.filled / .status</code>，T+1 开盘撮合前 filled=0）。
          <code>set_order_cost / set_slippage</code> 会采集为本次回测费率（优先于「⚙ 设置」里的默认值）。</p>
        <p><strong>近似说明</strong>：<code>finance.run_query</code> 返回空表（审计过滤恒通过）；分钟级请求按日线近似
          （'1m' 现价以当日开盘价代理、尾盘判断以收盘价代理）；空仓月货基 ETF(511880/511990) 为合成行情（年化约2%）。
          费率/滑点/参与率等在「⚙ 设置」里调（仅本页生效）。</p>
      </div>
    </details>
  </div>

  <div v-if="jqResult">
    <div class="card">
      <h3>回测结果 <span class="muted" style="font-size:12px">{{jqResult.start}} ~ {{jqResult.end}} · 资金 {{fmt(jqResult.capital,0)}} · {{jqResult.codes_count}} 只候选域</span></h3>
      <div class="cards">
        <div class="metric" v-for="(v,k) in jqResult.metrics" :key="k"><div class="label">{{k}}</div><div class="value" :class="sign(v)">{{metricText(k,v)}}</div></div>
      </div>
      <div id="jqEquity" class="chart"></div>
    </div>
    <div class="card">
      <h3>期末持仓</h3>
      <div class="table-wrap"><table>
        <tr><th>代码</th><th>名称</th><th>权重</th><th>现价</th><th>市值</th></tr>
        <tr v-for="(h,i) in jqResult.holdings" :key="i">
          <td>{{h.code}}</td><td>{{h.name}}</td><td>{{pct(h.weight)}}</td><td>{{fmt(h.price,2)}}</td><td>{{fmt(h.market_value,0)}}</td>
        </tr>
      </table></div>
    </div>
    <div class="card">
      <h3>运行日志</h3>
      <pre style="max-height:300px;overflow:auto;font-size:12px">{{jqResult.logs.join('\n')}}</pre>
    </div>
  </div>
  <div v-else class="card"><div class="empty">写好策略后点「▶ 运行策略」查看回测结果</div></div>

  <!-- 设置弹窗(仅本页生效的参数副本) -->
  <Teleport to="body">
    <div v-if="showSettings" class="threshold-modal-overlay" @click="showSettings = false">
      <div class="threshold-modal" @click.stop>
        <div class="threshold-modal-head">
          <div><strong>设置 · 交易参数</strong>
            <span class="threshold-modal-mode">仅代码面板生效</span>
            <span v-if="cfgCustomized" class="research-spec-custom">已自定义</span>
          </div>
          <button class="threshold-modal-close" @click="showSettings = false">×</button>
        </div>
        <div class="threshold-modal-body">
          <p class="threshold-hint">这里是全局统一配置中心的副本，改动只影响本页的聚宽策略回测，不影响回测对比/模拟盘/门禁。</p>
          <section class="threshold-group">
            <h4>交易成本</h4>
            <div class="threshold-grid">
              <label>买入费率
                <input type="number" step="0.0001" min="0" class="threshold-input" v-model.number="cfgDraft.buy_cost">
              </label>
              <label>卖出费率(含税)
                <input type="number" step="0.0001" min="0" class="threshold-input" v-model.number="cfgDraft.sell_cost">
              </label>
              <label>滑点 (bps)
                <input type="number" step="0.5" min="0" class="threshold-input" v-model.number="cfgDraft.slippage_bps">
              </label>
            </div>
          </section>
          <section class="threshold-group">
            <h4>撮合与流动性</h4>
            <div class="threshold-grid">
              <label>单笔参与率 ≤
                <input type="number" step="0.01" min="0" max="1" class="threshold-input" v-model.number="cfgDraft.max_participation">
              </label>
              <label>整手股数
                <input type="number" step="1" min="1" class="threshold-input" v-model.number="cfgDraft.lot_size">
              </label>
              <label>成交额分位过滤
                <input type="number" step="0.05" min="0" max="1" class="threshold-input" v-model.number="cfgDraft.amount_q">
              </label>
              <label>涨跌停限制
                <select class="threshold-input" v-model="cfgDraft.limit_flags">
                  <option :value="true">启用（推荐）</option>
                  <option :value="false">停用</option>
                </select>
              </label>
              <label>预热天数
                <input type="number" step="10" min="0" class="threshold-input" v-model.number="cfgDraft.warmup_days">
              </label>
              <label>日均成交额下限(元)
                <input type="number" step="100000" min="0" class="threshold-input" v-model.number="cfgDraft.min_am20_yuan">
              </label>
            </div>
          </section>
        </div>
        <div class="threshold-modal-foot">
          <button class="threshold-btn-reset" :disabled="cfgSaving" @click="resetConfig">恢复默认</button>
          <span class="threshold-spacer"></span>
          <button class="threshold-btn-cancel" @click="showSettings = false">取消</button>
          <button class="threshold-btn-save" :disabled="cfgSaving" @click="saveConfig">
            {{ cfgSaving ? '保存中…' : '保存' }}
          </button>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<script>
import { api } from '../utils/api.js'
import { fmt, pct, sign, metricText, today } from '../utils/format.js'
import { renderLine } from '../utils/charts.js'

export default {
  name: 'CodeView',
  data() {
    return {
      code: {
        code: '', savedName: '', savedList: [],
        capital: 100000, start: '2025-01-01', end: today(),
      },
      codeResult: null, codeRunning: false, codeError: '', codeSaveMsg: '',
      jqResult: null,
      preflightRunning: false, preflightResult: null,
      showSettings: false, cfgDraft: {}, cfgCustomized: false, cfgSaving: false,
      loadSavedDialog: false,
    };
  },
  mounted() {
    this.refreshSaved();
    this.loadConfig();
    this.loadDefaultCode();
  },
  methods: {
    fmt, pct, sign, metricText,
    async loadDefaultCode() {
      // 默认代码: 小盘三正示例(若已保存过则用保存项)
      try {
        const r = await api('/api/code/saved');
        const hit = (r.items || []).find(x => x.name === '小盘三正示例');
        if (hit) { this.code.savedName = hit.name; await this.loadSaved(); return; }
      } catch (e) { /* ignore */ }
      this.code.code = this.demoCode();
    },
    demoCode() {
      return `# 聚宽风格策略示例 —— 小盘三正
# 改选股逻辑直接改 get_stock_list / select 部分, 参数在 initialize 里
from jqdata import *
import numpy as np
import pandas as pd
from datetime import timedelta

def initialize(context):
    g.stock_num = 7            # 基准持仓数
    g.min_mv, g.max_mv = 3, 1000   # 市值区间(亿)
    g.highest = 60             # 收盘价上限(元)
    g.stoploss = 0.07
    g.pass_months = [4]
    g.hold_list = []
    g.yesterday_HL_list = []
    g.target_list = []
    run_daily(prepare_stock_list, time='9:05')
    run_daily(sell_stocks, time='10:00')
    run_daily(trade_afternoon, time='14:00')
    run_weekly(weekly_adjustment, 2, time='10:00')

def prepare_stock_list(context):
    g.hold_list = [p.security for p in context.portfolio.positions.values()]
    if g.hold_list:
        df = get_price(g.hold_list, end_date=context.previous_date,
                       fields=['close', 'high_limit'], count=1, panel=False)
        g.yesterday_HL_list = list(df[df['close'] == df['high_limit']]['code'])
    else:
        g.yesterday_HL_list = []

def get_stock_list(context):
    stocks = get_index_stocks('000300')   # 全量池
    cur = get_current_data()
    initial = [s for s in stocks
               if not cur[s].paused and not cur[s].is_st and '退' not in cur[s].name
               and not s.startswith(('30', '68', '8', '4'))]
    q = query(valuation.code, valuation.market_cap,
              income.np_parent_company_owners, income.net_profit,
              income.operating_revenue).filter(
        valuation.code.in_(initial),
        valuation.market_cap.between(g.min_mv, g.max_mv),
        income.np_parent_company_owners > 0,
        income.net_profit > 0,
        income.operating_revenue > 1e8,
    ).order_by(valuation.market_cap.asc()).limit(21)
    df = get_fundamentals(q, date=context.previous_date)
    last = history(1, unit='1d', field='close', security_list=list(df['code']))
    return [s for s in df['code'] if s in g.hold_list
            or (s in last and last[s][-1] <= g.highest)]

def weekly_adjustment(context):
    if context.current_dt.month in g.pass_months:
        for s in list(context.portfolio.positions):
            order_target_value(s, 0)
        return
    num = adjust_stock_num(context)
    g.target_list = get_stock_list(context)[:num]
    for s in g.hold_list:
        if s not in g.target_list and s not in g.yesterday_HL_list \\
                and s in context.portfolio.positions:
            order_target_value(s, 0)
    buy_list = [s for s in g.target_list if s not in g.hold_list]
    if not buy_list:
        return
    pv = context.portfolio.total_value
    exposure = sum(p.value for p in context.portfolio.positions.values()) / pv
    avail = 0.70 - exposure
    per = min(0.12, max(avail, 0.0) / len(buy_list))
    for s in buy_list:
        order_target_value(s, per * pv)

def adjust_stock_num(context):
    idx = get_price('000300', end_date=context.previous_date, count=30)
    ma = idx['close'].rolling(10).mean()
    if pd.isna(ma.iloc[-1]):
        return g.stock_num
    diff = idx['close'].iloc[-1] - ma.iloc[-1]
    frac = 1.0 / (1.0 + np.exp(-diff / (idx['close'].iloc[-1] * 0.025)))
    return max(4, min(g.stock_num, int(round(g.stock_num - 3 * frac))))

def sell_stocks(context):
    pos = context.portfolio.positions
    for s in list(pos.keys()):
        if pos[s].price >= pos[s].avg_cost * 2:
            order_target_value(s, 0)
        elif pos[s].price < pos[s].avg_cost * (1 - g.stoploss):
            order_target_value(s, 0)
    df = get_price(get_index_stocks('000300'), end_date=context.previous_date,
                   fields=['close', 'open'], count=1, panel=False)
    if not df.empty and (1 - df['close'] / df['open']).mean() >= 0.05:
        for s in list(pos.keys()):
            order_target_value(s, 0)

def trade_afternoon(context):
    # 涨停开板近似: 昨收涨停 且 今开 <+9.5% (未封死) -> 开盘卖
    for s in g.yesterday_HL_list:
        if s in context.portfolio.positions:
            cur = get_current_data()[s]
            op = get_price(s, end_date=context.current_dt, fields=['open'], count=1)
            if not op.empty and op['open'].iloc[-1] < cur.last_price * 1.095:
                order_target_value(s, 0)
`;
    },
    async refreshSaved() {
      try {
        const r = await api('/api/code/saved');
        this.code.savedList = r.items || [];
      } catch (e) { /* ignore */ }
    },
    async saveCode() {
      if (!this.code.savedName.trim()) { this.codeError = '请填保存名称'; return; }
      this.codeError = '';
      try {
        const r = await api('/api/code/save', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: this.code.savedName, code: this.code.code, engine: 'jq' }) });
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
        if (!r.code) { this.codeError = '该保存项没有代码内容'; return; }
        this.code.code = r.code;
        this.loadSavedDialog = false;
        this.codeSaveMsg = '已载入：' + r.name;
        setTimeout(() => { if (this.codeSaveMsg) this.codeSaveMsg = ''; }, 3000);
      } catch (e) { this.codeError = '载入失败: ' + e.message; }
    },
    // ---- 设置中心(代码面板副本) ----
    async loadConfig() {
      try {
        const r = await api('/api/code/config');
        this.cfgDraft = { ...r.config };
        this.cfgCustomized = r.customized;
      } catch (e) { /* 配置中心不可用时保持空 */ }
    },
    openSettings() {
      this.cfgDraft = { ...this.cfgDraft };
      this.showSettings = true;
    },
    async saveConfig() {
      this.cfgSaving = true;
      try {
        const r = await api('/api/code/config', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ config: this.cfgDraft }) });
        if (r.ok) { this.cfgDraft = { ...r.config }; this.cfgCustomized = r.customized; this.showSettings = false; }
      } catch (e) { this.codeError = '保存设置失败: ' + e.message; }
      finally { this.cfgSaving = false; }
    },
    async resetConfig() {
      this.cfgSaving = true;
      try {
        const r = await api('/api/code/config/reset', { method: 'POST' });
        if (r.ok) { this.cfgDraft = { ...r.config }; this.cfgCustomized = false; }
      } catch (e) { this.codeError = '恢复默认失败: ' + e.message; }
      finally { this.cfgSaving = false; }
    },
    // ---- 预检 ----
    async runPreflight() {
      if (!this.code.code.trim()) { this.codeError = '请先粘贴策略代码'; return; }
      this.preflightRunning = true; this.preflightResult = null; this.codeError = '';
      try {
        const r = await api('/api/code/jq/preflight', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ code: this.code.code }) });
        this.preflightResult = r;
      } catch (e) { this.preflightResult = { ok: false, missing: [], message: '预检请求失败: ' + e.message }; }
      finally { this.preflightRunning = false; }
    },
    // ---- 运行 ----
    async runJq() {
      if (!this.code.code.trim()) { this.codeError = '请先粘贴策略代码'; return; }
      this.codeRunning = true; this.codeError = ''; this.codeSaveMsg = ''; this.preflightResult = null;
      try {
        const r = await api('/api/code/jq/run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ code: this.code.code, start: this.code.start, end: this.code.end, capital: this.code.capital }) });
        if (r.ok === false) { this.codeError = r.error + (r.traceback ? '\n' + r.traceback : ''); return; }
        this.jqResult = r;
        this.$nextTick(() => {
          const series = [{ name: '策略', dates: r.nav.map(x => x.date), values: r.nav.map(x => x.value) }];
          if (Array.isArray(r.benchmark) && r.benchmark.length) {
            series.push({ name: '基准 ' + (r.bench_code || ''), dash: true, dates: r.benchmark.map(x => x.date), values: r.benchmark.map(x => x.value) });
          }
          renderLine('jqEquity', series);
        });
      } catch (e) { this.codeError = '运行失败: ' + e.message; }
      finally { this.codeRunning = false; }
    },
  },
}
</script>
