<template>
  <div class="hold-tabs" style="margin-bottom:14px">
    <button :class="{active: btSub==='bt'}" @click="btSub='bt'">基础回测</button>
    <button :class="{active: btSub==='sweep'}" @click="btSub='sweep'">参数稳健性</button>
  </div>

  <!-- 基础回测 -->
  <template v-if="btSub==='bt'">
    <div class="card">
      <h3>回测参数</h3>
      <div class="form-section">
        <div class="form-section-title">标的与策略</div>
        <div class="form-grid">
          <label class="field"><span>股票池</span><select v-model="bt.universe"><option>科技TMT</option><option>沪深300+中证500+中证1000</option><option>ETF</option><option>场外基金</option></select></label>
          <label class="field wide"><span>策略</span><strategy-select v-model="bt.strategy" :strategies="store.strategies" placeholder="选择策略"></strategy-select></label>
          <label class="field"><span>TopN</span><select v-model.number="bt.top_n"><option v-for="n in [1,2,3,5,8,10]" :value="n">{{n}}</option></select></label>
          <label class="field"><span>初始资金</span><input type="number" v-model.number="bt.capital" step="1000"></label>
          <label class="field"><span>基准</span>
            <select v-model="bt.bench">
              <option>等权股票池</option>
              <option v-for="i in store.indices" :value="i.name">{{i.name}}</option>
            </select>
          </label>
        </div>
      </div>
      <div class="form-section">
        <div class="form-section-title">时间与频率</div>
        <div class="form-grid">
          <label class="field"><span>频率</span><select v-model="bt.freq"><option value="monthly">月频</option><option value="weekly">周频</option></select></label>
          <label class="field"><span>开始</span><input type="date" v-model="bt.start"></label>
          <label class="field"><span>结束</span><input type="date" v-model="bt.end"></label>
          <label class="field"><span>预热天数</span><select v-model.number="bt.warmup_days"><option :value="0">关闭</option><option :value="120">120 天</option><option :value="400">400 天</option><option :value="9999">全量</option></select></label>
        </div>
      </div>
      <div class="form-section">
        <div class="form-section-title">过滤条件</div>
        <div class="form-grid">
          <label class="field-check"><input type="checkbox" v-model="bt.exclude"> 剔除科创/创业</label>
          <label class="field-check"><input type="checkbox" v-model="bt.affordable"> 一手过滤</label>
          <label class="field"><span>成交额分位</span><input type="number" v-model.number="bt.amount_q" step="0.05" min="0" max="1"></label>
        </div>
      </div>
      <div class="form-section">
        <div class="form-section-title">组合构建</div>
        <div class="form-check-row">
          <label class="field-check"><input type="checkbox" v-model="bt.long_short"> 多空对冲（多头 TopN + 空头最弱 N 只，模拟融券）</label>
          <label class="field-check"><input type="checkbox" v-model="bt.neutral"> 行业中性化选股</label>
          <label class="field-check"><input type="checkbox" v-model="bt.risk_neutral"> 风险中性化（风格+行业）</label>
          <label class="field-check"><input type="checkbox" v-model="bt.analyze"> 因子质量分析（IC/分组/多空价差）</label>
          <label class="field-check"><input type="checkbox" v-model="bt.use_financial"> 财务因子（ROE/PB/EP 等）</label>
        </div>
        <div class="form-grid" style="margin-top:10px">
          <label class="field"><span>空头只数</span><input type="number" v-model.number="bt.short_n" min="1" max="20" :disabled="!bt.long_short"></label>
          <label class="field"><span>融券年化费率%</span><input type="number" v-model.number="bt.short_rate" step="0.5" min="0" :disabled="!bt.long_short"></label>
          <label class="field"><span>每行业上限</span><input type="number" v-model.number="bt.industry_cap" min="0" max="5" placeholder="0=不限"></label>
        </div>
      </div>
      <div class="form-actions">
        <p class="err left" v-if="btError">{{btError}}</p>
        <button class="primary" @click="runBacktest" :disabled="btRunning"><span v-if="btRunning" class="spinner"></span>{{btRunning ? '回测中…' : '跑回测'}}</button>
      </div>
    </div>
    <div v-if="btResult">
      <div class="row" style="justify-content:flex-end; gap:8px">
        <button class="ghost" @click="genReport" :disabled="reportLoading"><span v-if="reportLoading" class="spinner"></span>{{reportLoading ? '生成中…' : 'QuantStats 绩效报告'}}</button>
      </div>
      <div class="cards">
        <div class="metric" v-for="(v,k) in btResult.metrics" :key="k"><div class="label">{{k}}</div><div class="value" :class="sign(v)">{{metricText(k,v)}}</div></div>
      </div>
      <div class="cards" v-if="store.benchInfo.metrics">
        <div class="metric"><div class="label">基准 {{store.benchInfo.name}} 总收益</div><div class="value" :class="sign(store.benchInfo.metrics.total)">{{pct(store.benchInfo.metrics.total)}}</div></div>
        <div class="metric"><div class="label">基准年化</div><div class="value" :class="sign(store.benchInfo.metrics.annual)">{{pct(store.benchInfo.metrics.annual)}}</div></div>
        <div class="metric"><div class="label">基准夏普</div><div class="value">{{fmt(store.benchInfo.metrics.sharpe, 2)}}</div></div>
        <div class="metric"><div class="label">基准最大回撤</div><div class="value" :class="sign(store.benchInfo.metrics.mdd)">{{pct(store.benchInfo.metrics.mdd)}}</div></div>
      </div>
      <div class="card"><h3>资金曲线（策略 vs 基准）</h3><div id="btEquity" class="chart"></div></div>
      <div class="card"><h3>月度收益热力图</h3><div id="btMonthly" class="chart heatmap"></div><p class="muted">按回测净值计算；首月和末月按实际区间统计，年度列为月度收益复合值。</p></div>
      <div class="card"><h3>回撤</h3><div id="btDraw" class="chart small"></div></div>
      <div class="card" v-if="btResult.factor_quality">
        <h3>因子质量分析（未来 20 日收益 · Spearman IC / 5 分组）</h3>
        <div class="cards">
          <div class="metric"><div class="label">IC 均值</div><div class="value" :class="sign(btResult.factor_quality.ic?.mean_ic)">{{fmt(btResult.factor_quality.ic?.mean_ic, 4)}}</div></div>
          <div class="metric"><div class="label">ICIR</div><div class="value">{{fmt(btResult.factor_quality.ic?.icir, 3)}}</div></div>
          <div class="metric"><div class="label">t 值</div><div class="value">{{fmt(btResult.factor_quality.ic?.t_stat, 2)}}</div></div>
          <div class="metric"><div class="label">IC>0 占比</div><div class="value">{{pct(btResult.factor_quality.ic?.win_rate)}}</div></div>
          <div class="metric"><div class="label">多空价差（年化）</div><div class="value" :class="sign(btResult.factor_quality.direction_adjusted?.spread)">{{pct(btResult.factor_quality.direction_adjusted?.spread)}}</div></div>
        </div>
        <div class="grid" style="grid-template-columns:1.6fr 1fr">
          <div><h4 class="muted">逐日 IC</h4><div id="btIc" class="chart small"></div></div>
          <div>
            <h4 class="muted">5 分组平均前瞻收益</h4>
            <div class="table-wrap"><table><tr><th>分组</th><th>平均前瞻收益</th></tr>
            <tr v-for="g in btResult.factor_quality.group?.groups || []" :key="g.group">
              <td>第 {{g.group}} 组</td><td :class="sign(g.mean_fwd_ret)">{{pct(g.mean_fwd_ret)}}</td></tr></table></div>
          </div>
        </div>
      </div>
      <div class="card" v-if="btResult.risk_attribution">
        <h3>风险归因（期末持仓 · 风险贡献）</h3>
        <p class="muted">风格/行业因子 + specific 残差的组合风险占比，识别收益来源与集中度。</p>
        <div class="table-wrap"><table>
          <tr><th>因子</th><th>风险贡献%</th></tr>
          <tr v-for="(item,i) in sortedRiskAttr()" :key="i">
            <td>{{item[0]}}</td><td>{{pct(item[1])}}</td>
          </tr>
        </table></div>
      </div>
      <div class="card" v-if="btResult.brinson && btResult.brinson.summary && btResult.brinson.summary.length">
        <h3>行业归因（Brinson · 月度 vs 股票池等权基准）</h3>
        <div class="table-wrap"><table>
          <tr><th>行业</th><th>配置效应%</th><th>选择效应%</th><th>交互%</th><th>合计%</th><th>组合均权</th><th>基准均权</th></tr>
          <tr v-for="(s,i) in btResult.brinson.summary" :key="i">
            <td>{{s.industry}}</td><td>{{pct(s.allocation)}}</td><td>{{pct(s.selection)}}</td><td>{{pct(s.interaction)}}</td><td>{{pct(s.total)}}</td><td>{{fmt(s.avg_combo_weight)}}</td><td>{{fmt(s.avg_bench_weight)}}</td>
          </tr>
        </table></div>
      </div>
      <div class="grid" style="grid-template-columns:1fr 1fr">
        <div class="card"><h3>持仓明细</h3>
          <div style="overflow-x:auto"><table><tr><th>代码</th><th>名称</th><th>方向</th><th>权重%</th><th>价格</th><th>市值</th></tr>
          <tr v-for="h in btResult.holdings" :key="h.code"><td>{{stock(h.code, h.name)}}</td><td>{{h.name||'-'}}</td><td>{{h.direction||'多'}}</td><td>{{fmt(h.weight_pct)}}</td><td>{{fmt(h.price)}}</td><td>{{fmt(h.market_value)}}</td></tr></table></div>
        </div>
        <div class="card"><h3>调仓记录</h3>
          <div style="overflow-x:auto"><table><tr><th>日期</th><th>信号日</th><th>持仓数</th><th>换手</th></tr>
          <tr v-for="(t,i) in btResult.trades" :key="i"><td>{{t.date.slice(0,10)}}</td><td>{{t.signal_date.slice(0,10)}}</td><td>{{t.num_hold}}</td><td>{{fmt(t.turnover)}}</td></tr></table></div>
        </div>
      </div>
    </div>
    <div v-else class="card"><div class="empty">设置参数后点击「跑回测」查看净值、回撤、持仓与调仓结果</div></div>
  </template>

  <!-- QuantStats 弹窗 -->
  <div v-if="reportHtml" class="modal" @click.self="reportHtml=''">
    <div class="modal-box">
      <div class="row"><h3>QuantStats 绩效报告</h3><button class="ghost" @click="reportHtml=''">关闭</button></div>
      <iframe :srcdoc="reportHtml" style="width:100%; height:calc(100vh - 140px); border:0; background:#fff; border-radius:10px"></iframe>
    </div>
  </div>

  <!-- 参数稳健性 -->
  <template v-if="btSub==='sweep'">
    <div class="card">
      <h3>参数稳健性 / Walk-forward</h3>
      <p class="muted">把时间轴切成多个连续窗口独立回测，看策略跨窗口的指标分布。判断依据：均值夏普、胜率、最差窗口——单段收益高但窗口间波动大的组合通常是过拟合信号。</p>
      <div class="form-grid">
        <label class="field"><span>模式</span><select v-model="sweep.mode">
          <option value="event">双均线金叉参数扫描</option>
          <option value="factor">因子策略 walk-forward</option>
          <option value="rolling">滚动训练-测试（因子）</option>
          <option value="rolling_event">滚动训练-测试（双均线）</option>
        </select></label>
        <label class="field"><span>起点</span><input type="date" v-model="sweep.start"></label>
        <label class="field"><span>终点</span><input type="date" v-model="sweep.end"></label>
        <label class="field"><span>窗口数</span><select v-model.number="sweep.folds">
          <option v-for="n in [2,3,4,6]" :value="n">{{n}}</option>
        </select></label>
        <template v-if="sweep.mode==='event' || sweep.mode==='rolling_event'">
          <label class="field"><span>短期均线</span><input v-model="sweep.shortList" placeholder="3,5,8,10,13"></label>
          <label class="field"><span>长期均线</span><input v-model="sweep.longList" placeholder="10,20,30,60"></label>
          <label class="field"><span>持仓数</span><select v-model.number="sweep.top_n">
            <option v-for="n in [1,2,3,5]" :value="n">{{n}}</option>
          </select></label>
        </template>
        <template v-else>
          <label class="field wide"><span>策略</span><strategy-select v-model="sweep.strategy" :strategies="store.strategies" placeholder="选择策略"></strategy-select></label>
        </template>
      </div>
      <div class="form-actions">
        <button class="primary" @click="runSweep" :disabled="sweepRunning">
          <span v-if="sweepRunning" class="spinner"></span>{{sweepRunning ? '扫描中（约 1-2 分钟）…' : '运行扫描'}}
        </button>
      </div>
      <p class="err left" v-if="sweepError">{{sweepError}}</p>
    </div>

    <template v-if="sweepResult">
      <div class="card" v-if="sweepResult.mode!=='rolling' && sweepResult.mode!=='rolling_event' && sweepResult.summary && sweepResult.summary.length">
        <h3>参数组合汇总 <span class="muted">（点击表头排序）</span></h3>
        <div class="table-wrap"><table>
          <tr>
            <th class="sortable" @click="sortSweep('short')">短期{{swArrow('short')}}</th>
            <th class="sortable" @click="sortSweep('long')">长期{{swArrow('long')}}</th>
            <th class="sortable" @click="sortSweep('mean_sharpe')">均值夏普{{swArrow('mean_sharpe')}}</th>
            <th class="sortable" @click="sortSweep('median_sharpe')">中位夏普{{swArrow('median_sharpe')}}</th>
            <th class="sortable" @click="sortSweep('std_sharpe')">夏普std{{swArrow('std_sharpe')}}</th>
            <th class="sortable" @click="sortSweep('mean_total')">均值收益{{swArrow('mean_total')}}</th>
            <th class="sortable" @click="sortSweep('worst_total')">最差窗口{{swArrow('worst_total')}}</th>
            <th class="sortable" @click="sortSweep('win_rate')">胜率{{swArrow('win_rate')}}</th>
            <th class="sortable" @click="sortSweep('mean_mdd')">均值回撤{{swArrow('mean_mdd')}}</th>
          </tr>
          <tr v-for="(r,i) in sortedSweep()" :key="i">
            <td>{{r.short}}</td><td>{{r.long}}</td>
            <td>{{fmt(r.mean_sharpe)}}</td><td>{{fmt(r.median_sharpe)}}</td>
            <td>{{fmt(r.std_sharpe)}}</td><td>{{pct(r.mean_total)}}</td>
            <td>{{pct(r.worst_total)}}</td><td>{{Math.round((r.win_rate||0)*100)}}%</td>
            <td>{{pct(r.mean_mdd)}}</td>
          </tr>
        </table></div>
      </div>

      <div class="card" v-if="(sweepResult.mode==='rolling' || sweepResult.mode==='rolling_event') && sweepResult.summary && sweepResult.summary.length">
        <h3>滚动训练-测试汇总</h3>
        <div class="table-wrap"><table>
          <tr><th>窗口数</th><th>已训练窗口</th><th>均值夏普</th><th>中位夏普</th><th>胜率</th><th>均值收益</th><th>最差窗口</th><th>最好窗口</th></tr>
          <tr v-for="(r,i) in sweepResult.summary" :key="i">
            <td>{{r.n_windows}}</td><td>{{r.trained_windows}}</td>
            <td>{{fmt(r.mean_sharpe)}}</td><td>{{fmt(r.median_sharpe)}}</td>
            <td>{{Math.round((r.win_rate||0)*100)}}%</td>
            <td>{{pct(r.mean_total)}}</td><td>{{pct(r.worst_total)}}</td><td>{{pct(r.best_total)}}</td>
          </tr>
        </table></div>
      </div>

      <div class="card" v-if="sweepResult.param_history && sweepResult.param_history.length">
        <h3>滚动训练-测试：逐窗口参数选择</h3>
        <p class="muted">每个测试窗口用之前全部历史选参（按训练夏普），再跑当前窗口做样本外验证。</p>
        <div class="table-wrap"><table>
          <tr><th>窗口</th><th>训练起点</th><th>训练终点</th><th>测试起点</th><th>测试终点</th><th>选中参数</th><th>训练夏普</th></tr>
          <tr v-for="(p,i) in sweepResult.param_history" :key="i">
            <td>{{p.fold}}</td><td>{{p.train_start}}</td><td>{{p.train_end}}</td>
            <td>{{p.test_start}}</td><td>{{p.test_end}}</td>
            <td>{{p.chosen_top_n ? 'TopN='+p.chosen_top_n : 'MA '+p.chosen_short+'/'+p.chosen_long}}</td>
            <td>{{p.train_sharpe == null ? '-' : fmt(p.train_sharpe)}}</td>
          </tr>
        </table></div>
      </div>

      <div class="card" v-if="sweepResult.heatmap && sweepResult.heatmap.length">
        <h3>均值夏普热力图</h3>
        <div class="table-wrap"><table>
          <tr><th>short \ long</th><th v-for="c in sweepResult.heatmap_cols" :key="c">{{c}}</th></tr>
          <tr v-for="row in sweepResult.heatmap" :key="row.short">
            <td>{{row.short}}</td>
            <td v-for="c in sweepResult.heatmap_cols" :key="c">{{fmt(row[c], 2)}}</td>
          </tr>
        </table></div>
      </div>

      <div class="card" v-if="sweepResult.windows && sweepResult.windows.length">
        <h3>逐窗口明细</h3>
        <div class="table-wrap"><table>
          <tr><th v-for="c in windowCols()" :key="c">{{c}}</th></tr>
          <tr v-for="(w,i) in sweepResult.windows" :key="i">
            <td v-for="c in windowCols()" :key="c">{{fmtCell(c, w[c])}}</td>
          </tr>
        </table></div>
      </div>
    </template>
    <div v-if="!sweepResult" class="card"><div class="empty">设置参数后运行扫描</div></div>
  </template>
</template>

<script>
import { store } from '../store/index.js'
import { api } from '../utils/api.js'
import { fmt, pct, sign, stock, metricText, sortCompare } from '../utils/format.js'
import { today } from '../utils/format.js'
import { renderLine, renderMonthlyHeatmap } from '../utils/charts.js'

export default {
  name: 'Backtest',
  data() {
    return {
      store,
      bt: { universe: '科技TMT', strategy: '低换手冷门', top_n: 3, capital: 5000, freq: 'monthly', start: '2026-02-02', end: today(), exclude: true, affordable: true, amount_q: 0.2, warmup_days: 400, long_short: false, neutral: false, risk_neutral: false, use_financial: false, short_n: 3, short_rate: 8.6, industry_cap: 0, analyze: false, bench: '沪深300' },
      btSub: 'bt',
      btResult: null,
      btRunning: false,
      btError: '',
      reportHtml: '',
      reportLoading: false,
      sweep: { mode: 'event', start: '2023-01-03', end: '', folds: 4,
               shortList: '3,5,8,10,13', longList: '10,20,30,60', top_n: 3,
               strategy: '双均线多头 5/20' },
      sweepResult: null,
      sweepRunning: false,
      sweepError: '',
      sweepSort: { key: 'mean_sharpe', dir: 'desc' },
    };
  },
  watch: {
    'bt.strategy'(name) {
      this.applyBtStrategyDefaults();
    },
    'bt.bench'() { if (this.btResult) this.renderBacktest(); },
    'bt.universe'(universe) {
      store.reloadStrategiesForUniverse(universe);
    },
    btSub(v) {
      this.$nextTick(() => {
        if (v === 'bt' && this.btResult) this.renderBacktest();
      });
    },
  },
  methods: {
    today,
    fmt,
    pct,
    sign,
    metricText,
    stock(code, name) { return stock(code, name, store.names); },
    applyBtStrategyDefaults() {
      const s = store.strategies.find(x => x.name === this.bt.strategy) || {};
      this.bt.long_short = !!s.long_short;
      this.bt.short_n = s.short_n || 3;
      this.bt.short_rate = Math.round(((s.short_cost_rate || 0) * 100) * 10) / 10;
    },
    async runBacktest() {
      this.btRunning = true; this.btError = '';
      try {
        const payload = {
          ...this.bt,
          exclude_kechuang: this.bt.exclude,
          industry_neutral: this.bt.neutral,
          short_cost_rate: (this.bt.short_rate || 0) / 100,
          industry_cap: this.bt.industry_cap > 0 ? this.bt.industry_cap : null,
          analyze: !!this.bt.analyze,
        };
        const ctrl = new AbortController();
        const timer = setTimeout(() => ctrl.abort(), 120000);
        let r;
        try {
          r = await api('/api/backtest', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload), signal: ctrl.signal });
        } finally { clearTimeout(timer); }
        if (r.error) { this.btError = r.error; return; }
        this.btResult = r;
        this.$nextTick(async () => this.renderBacktest());
      } catch (e) {
        this.btError = (e && e.name === 'AbortError')
          ? '回测超时（120 秒）。后台补数或内存不足时可能变慢，稍后重试。'
          : '回测失败: ' + (e && e.message ? e.message : e);
      } finally { this.btRunning = false; }
    },
    async renderBacktest() {
      const nav = this.btResult.nav;
      const series = [
        { name: '策略资金', dates: nav.map(x => x.date), values: nav.map(x => +(x.value * this.bt.capital).toFixed(2)) },
      ];
      const bs = await store.benchSeries(this.btResult, this.bt, this.bt.start, this.bt.end);
      if (bs.points.length) {
        series.push({ name: bs.name, dates: bs.points.map(x => x.date), values: bs.points.map(x => +(x.value * this.bt.capital).toFixed(2)), dash: true });
        store.benchInfo = { name: bs.name, metrics: store.calcSeriesMetrics(bs.points) };
      } else {
        store.benchInfo = { name: '', metrics: null };
      }
      renderLine('btEquity', series);
      renderMonthlyHeatmap('btMonthly', nav);
      const dd = this.btResult.drawdown;
      renderLine('btDraw', [{ name: '回撤', dates: dd.map(x => x.date), values: dd.map(x => +(x.value * 100).toFixed(2)), fill: true }]);
      const ic = this.btResult.factor_quality;
      if (ic && ic.ic_series) {
        renderLine('btIc', [{ name: 'IC', dates: ic.ic_series.map(x => x.date), values: ic.ic_series.map(x => +(x.ic || 0).toFixed(4)) }]);
      }
    },
    async genReport() {
      this.reportLoading = true;
      try {
        const payload = {
          universe: this.bt.universe, strategy: this.bt.strategy,
          top_n: this.bt.top_n, capital: this.bt.capital, freq: this.bt.freq,
          start: this.bt.start, end: this.bt.end,
          exclude_kechuang: this.bt.exclude, affordable: this.bt.affordable,
          amount_q: this.bt.amount_q, warmup_days: this.bt.warmup_days,
          long_short: this.bt.long_short, short_n: this.bt.short_n,
          short_cost_rate: (this.bt.short_rate || 0) / 100,
          industry_neutral: this.bt.neutral,
          industry_cap: this.bt.industry_cap > 0 ? this.bt.industry_cap : null,
          use_financial: !!this.bt.use_financial,
          risk_neutral: !!this.bt.risk_neutral,
        };
        const r = await api('/api/backtest/quantstats', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
        if (r.error) { alert(r.error); return; }
        this.reportHtml = r.html;
      } catch (e) { alert('生成报告失败: ' + e.message); }
      finally { this.reportLoading = false; }
    },
    async runSweep() {
      this.sweepRunning = true; this.sweepError = ''; this.sweepResult = null;
      try {
        const body = {
          mode: this.sweep.mode, start: this.sweep.start, end: this.sweep.end,
          folds: this.sweep.folds, top_n: this.sweep.top_n,
        };
        if (this.sweep.mode === 'event' || this.sweep.mode === 'rolling_event') {
          body.short_list = this.sweep.shortList.split(',').map(x => parseInt(x.trim(), 10)).filter(Number.isFinite);
          body.long_list = this.sweep.longList.split(',').map(x => parseInt(x.trim(), 10)).filter(Number.isFinite);
        } else {
          body.strategy = this.sweep.strategy;
        }
        const r = await api('/api/sweep', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
        if (r.error) { this.sweepError = r.error; return; }
        this.sweepResult = r;
      } catch (e) {
        this.sweepError = e.message || String(e);
      } finally {
        this.sweepRunning = false;
      }
    },
    windowCols() {
      if (!this.sweepResult || !this.sweepResult.windows || !this.sweepResult.windows.length) return [];
      const want = ['short','long','fold','start','end','n_days','chosen_top_n','chosen_freq','chosen_short','chosen_long','trained','total','annual','sharpe','mdd','calmar','win_rate','end_nav'];
      return want.filter(k => k in this.sweepResult.windows[0]);
    },
    sortedSweep() {
      const rows = (this.sweepResult?.summary || []).slice();
      const key = this.sweepSort.key;
      const dir = this.sweepSort.dir === 'asc' ? 1 : -1;
      return rows.sort((a, b) => sortCompare(a[key], b[key], dir));
    },
    sortSweep(key) {
      if (this.sweepSort.key === key) {
        this.sweepSort.dir = this.sweepSort.dir === 'asc' ? 'desc' : 'asc';
      } else {
        this.sweepSort.key = key;
        this.sweepSort.dir = (key === 'short' || key === 'long') ? 'asc' : 'desc';
      }
    },
    swArrow(key) {
      return this.sweepSort.key === key ? (this.sweepSort.dir === 'asc' ? '↑' : '↓') : '';
    },
    sortedRiskAttr() {
      const o = this.btResult && this.btResult.risk_attribution || {};
      return Object.entries(o).sort((a,b) => Math.abs(b[1]) - Math.abs(a[1]));
    },
    fmtCell(c, v) {
      if (v == null) return '-';
      if (['total','annual','mdd'].includes(c)) return this.pct(v);
      if (c === 'win_rate') return (v * 100).toFixed(0) + '%';
      if (['sharpe','calmar','end_nav','n_days'].includes(c)) return this.fmt(v, 2);
      return v;
    },
  },
}
</script>
