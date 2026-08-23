<template>
  <section>
    <div class="card">
      <h3>日级模拟盘</h3>
      <p class="muted">信号日（T-1）收盘生成目标持仓 → T 日开盘成交 → 收盘估值。因子账户由同一回测引擎（cash_mode=True）重放，执行口径一致：现金/整手/费用/涨跌停/拒单；事件策略仍由事件引擎重放。每日数据更新后由 systemd 盘后自动执行，也可手动触发。</p>
      <div class="form-section">
        <div class="form-section-title">创建账户</div>
        <div class="form-grid">
          <label class="field"><span>名称</span><input v-model="paper.form.name" placeholder="模拟盘A"></label>
          <label class="field"><span>类型</span><select v-model="paper.form.strategy_type">
            <option value="factor">因子策略</option>
            <option value="event">事件策略</option>
          </select></label>
          <template v-if="paper.form.strategy_type === 'factor'">
            <label class="field"><span>策略</span><strategy-select v-model="paper.form.strategy" :strategies="store.strategies" placeholder="选择策略"></strategy-select></label>
          </template>
          <template v-else>
            <label class="field"><span>代码模块</span><select v-model="paper.form.module" @change="paper.form.event_strategy=''">
              <option value="" disabled>选择已保存代码</option>
              <option v-for="m in paper.eventModules" :value="m.module">{{m.name}}</option>
            </select></label>
            <label class="field"><span>事件策略</span><select v-model="paper.form.event_strategy">
              <option value="" disabled>选择事件策略</option>
              <option v-for="s in paperEventStrategies()" :value="s">{{s}}</option>
            </select></label>
          </template>
          <label class="field"><span>股票池</span><select v-model="paper.form.universe"><option>科技TMT</option><option>沪深300+中证500+中证1000</option><option>ETF</option><option>场外基金</option></select></label>
          <label class="field"><span>频率</span><select v-model="paper.form.freq"><option value="daily">每日</option><option value="weekly">每周</option><option value="monthly">每月</option></select></label>
          <label class="field"><span>资金</span><input type="number" v-model.number="paper.form.capital" step="10000"></label>
          <label class="field"><span>TopN</span><select v-model.number="paper.form.top_n"><option v-for="n in [1,2,3,5,8,10]" :value="n">{{n}}</option></select></label>
          <label class="field"><span>单票权重上限</span><input type="number" v-model.number="paper.form.max_weight" step="0.05" min="0.05" max="1"></label>
          <label class="field"><span>流动性分位</span><input type="number" v-model.number="paper.form.amount_q" step="0.05" min="0" max="1"></label>
        </div>
        <div class="form-actions">
          <button class="primary" @click="createPaperAccount" :disabled="paper.creating">
            <span v-if="paper.creating" class="spinner"></span>{{paper.creating ? '创建中…' : '创建账户'}}
          </button>
        </div>
      </div>
      <p class="err left" v-if="paper.error">{{paper.error}}</p>
    </div>

    <template v-if="paper.accounts.length">
      <div class="card">
        <h3>账户</h3>
        <div class="row">
          <label class="field" style="min-width:300px"><span>账户</span>
            <select v-model.number="paper.pick" @change="loadPaperDetail">
              <option v-for="a in paper.accounts" :key="a.id" :value="a.id">#{{a.id}} · {{a.name}} · {{a.strategy_name}} · {{a.status}}</option>
            </select>
          </label>
          <button class="primary" @click="runPaper" :disabled="paper.running">
            <span v-if="paper.running" class="spinner"></span>{{paper.running ? '执行中…' : '手动执行一次'}}
          </button>
          <button class="ghost" @click="runPaperDry">Dry-run 预览</button>
          <button class="ghost" @click="togglePaperStatus">{{paperDetail?.account?.status === 'active' ? '暂停' : '启用'}}</button>
          <button class="ghost" @click="resetPaper">重置</button>
          <button class="ghost danger" @click="deletePaper">删除</button>
        </div>
        <p class="err left" v-if="paper.runError">{{paper.runError}}</p>
        <div class="cards" v-if="paper.lastRun">
          <div class="metric"><div class="label">执行日</div><div class="value">{{paper.lastRun.run_date}}</div></div>
          <div class="metric"><div class="label">状态</div><div class="value">{{paper.lastRun.accounts?.[0]?.processed}}</div></div>
          <div class="metric"><div class="label">调仓</div><div class="value">{{paper.lastRun.accounts?.[0]?.rebalanced ? '是' : '否'}}</div></div>
          <div class="metric"><div class="label">订单/成交/拒单</div><div class="value">{{paper.lastRun.accounts?.[0]?.orders}} / {{paper.lastRun.accounts?.[0]?.filled}} / {{paper.lastRun.accounts?.[0]?.rejected}}</div></div>
        </div>
      </div>

      <div class="cards" v-if="paperDetail?.latest">
        <div class="metric"><div class="label">最新权益</div><div class="value">{{fmt(paperDetail.latest.equity)}}</div></div>
        <div class="metric"><div class="label">现金</div><div class="value">{{fmt(paperDetail.latest.cash)}}</div></div>
        <div class="metric"><div class="label">持仓市值</div><div class="value">{{fmt(paperDetail.latest.market_value)}}</div></div>
        <div class="metric"><div class="label">累计盈亏</div><div class="value" :class="sign(paperDetail.latest.pnl)">{{fmt(paperDetail.latest.pnl)}}</div></div>
        <div class="metric"><div class="label">收益率</div><div class="value" :class="sign(paperDetail.latest.pnl_pct)">{{pct(paperDetail.latest.pnl_pct)}}</div></div>
      </div>

      <div class="card" v-if="paper.equity.length">
        <h3>权益曲线</h3><div id="paperEquity" class="chart"></div>
      </div>

      <div class="grid" style="grid-template-columns:1fr 1fr">
        <div class="card"><h3>持仓</h3>
          <div style="overflow-x:auto"><table><tr><th>代码</th><th>名称</th><th>股数</th><th>成本价</th><th>现价</th><th>市值</th><th>浮动盈亏</th></tr>
          <tr v-for="p in paperDetail?.positions || []" :key="p.code">
            <td>{{p.code}}</td><td>{{p.name}}</td><td>{{fmt(p.shares, 0)}}</td><td>{{fmt(p.avg_cost)}}</td>
            <td>{{fmt(p.price)}}</td><td>{{fmt(p.market_value)}}</td><td :class="sign(p.pnl)">{{fmt(p.pnl)}}</td>
          </tr></table></div>
        </div>
        <div class="card"><h3>订单</h3>
          <div style="overflow-x:auto"><table>
            <tr>
              <th class="sortable" @click="setOrderSort('code')">代码<span class="sort-arrow">{{orderArrow('code')}}</span></th>
              <th class="sortable" @click="setOrderSort('name')">名称<span class="sort-arrow">{{orderArrow('name')}}</span></th>
              <th class="sortable" @click="setOrderSort('side')">方向<span class="sort-arrow">{{orderArrow('side')}}</span></th>
              <th class="sortable" @click="setOrderSort('exec_date')">执行日<span class="sort-arrow">{{orderArrow('exec_date')}}</span></th>
              <th class="sortable" @click="setOrderSort('status')">状态<span class="sort-arrow">{{orderArrow('status')}}</span></th>
              <th class="sortable" @click="setOrderSort('shares')">股数<span class="sort-arrow">{{orderArrow('shares')}}</span></th>
              <th class="sortable" @click="setOrderSort('fill_price')">成交价<span class="sort-arrow">{{orderArrow('fill_price')}}</span></th>
              <th class="sortable" @click="setOrderSort('amount')">成交金额<span class="sort-arrow">{{orderArrow('amount')}}</span></th>
              <th>原因</th>
            </tr>
          <tr v-for="o in sortedPaperOrders()" :key="o.id">
            <td>{{o.code}}</td><td>{{o.name || '-'}}</td><td>{{o.side}}</td><td>{{(o.exec_date||'').slice(0,10)}}</td>
            <td>{{o.status}}</td><td>{{fmt(o.shares, 0)}}</td><td>{{fmt(o.fill_price)}}</td><td>{{fmt(o.amount)}}</td><td>{{o.reject_reason || '-'}}</td>
          </tr></table></div>
        </div>
      </div>

      <div class="grid" style="grid-template-columns:1fr 1fr">
        <div class="card"><h3>成交</h3>
          <div style="overflow-x:auto"><table>
            <tr>
              <th class="sortable" @click="setTradeSort('code')">代码<span class="sort-arrow">{{tradeArrow('code')}}</span></th>
              <th class="sortable" @click="setTradeSort('name')">名称<span class="sort-arrow">{{tradeArrow('name')}}</span></th>
              <th class="sortable" @click="setTradeSort('side')">方向<span class="sort-arrow">{{tradeArrow('side')}}</span></th>
              <th class="sortable" @click="setTradeSort('exec_date')">成交日<span class="sort-arrow">{{tradeArrow('exec_date')}}</span></th>
              <th class="sortable" @click="setTradeSort('shares')">股数<span class="sort-arrow">{{tradeArrow('shares')}}</span></th>
              <th class="sortable" @click="setTradeSort('price')">价格<span class="sort-arrow">{{tradeArrow('price')}}</span></th>
              <th class="sortable" @click="setTradeSort('amount')">总交易额<span class="sort-arrow">{{tradeArrow('amount')}}</span></th>
              <th class="sortable" @click="setTradeSort('fee')">费用<span class="sort-arrow">{{tradeArrow('fee')}}</span></th>
            </tr>
          <tr v-for="t in sortedPaperTrades()" :key="t.id">
            <td>{{t.code}}</td><td>{{t.name || '-'}}</td><td>{{t.side}}</td><td>{{(t.exec_date||'').slice(0,10)}}</td>
            <td>{{fmt(t.shares, 0)}}</td><td>{{fmt(t.price)}}</td><td>{{fmt(t.amount)}}</td><td>{{fmt(t.fee)}}</td>
          </tr></table></div>
        </div>
        <div class="card"><h3>事件日志</h3>
          <div style="overflow-x:auto"><table><tr><th>日期</th><th>级别</th><th>消息</th></tr>
          <tr v-for="e in paper.events" :key="e.id">
            <td>{{(e.date||'').slice(0,10)}}</td><td>{{e.level}}</td><td>{{e.msg}}</td>
          </tr></table></div>
        </div>
      </div>
    </template>
    <div v-else class="card"><div class="empty">还没有模拟盘账户，先创建一个</div></div>
  </section>
</template>

<script>
import { store } from '../store/index.js'
import { api } from '../utils/api.js'
import { fmt, pct, sign, toNum, sortCompare } from '../utils/format.js'
import { renderLine } from '../utils/charts.js'

export default {
  name: 'Paper',
  data() {
    return {
      store,
      fmt,
      pct,
      sign,
      paper: {
        accounts: [], pick: null, detail: null,
        orders: [], trades: [], events: [], equity: [],
        form: {
          name: '模拟盘A', strategy_type: 'factor', strategy: '动量 20 日',
          module: '', event_strategy: '', universe: '科技TMT',
          freq: 'monthly', capital: 100000, top_n: 3,
          max_weight: 0.5, amount_q: 0.2,
        },
        eventModules: [],
        creating: false, running: false, error: '', runError: '', lastRun: null,
        orderSort: { key: 'exec_date', dir: 'desc' },
        tradeSort: { key: 'exec_date', dir: 'desc' },
      },
    }
  },
  computed: {
    paperDetail() {
      return this.paper.detail;
    },
  },
  methods: {
    async loadPaperAccounts() {
      try {
        await this.loadPaperEventModules();
        this.paper.accounts = await api('/api/paper/accounts');
        if (this.paper.accounts.length) {
          if (!this.paper.pick || !this.paper.accounts.some(a => a.id === this.paper.pick)) {
            this.paper.pick = this.paper.accounts[0].id;
          }
          await this.loadPaperDetail();
        } else {
          this.paper.detail = null;
        }
      } catch (e) {
        this.paper.error = '加载账户失败: ' + e.message;
      }
    },
    async loadPaperDetail() {
      if (!this.paper.pick) return;
      const id = this.paper.pick;
      try { this.paper.detail = await api('/api/paper/accounts/' + id + '/summary'); } catch (e) {}
      try { this.paper.orders = await api('/api/paper/accounts/' + id + '/orders'); } catch (e) {}
      try { this.paper.trades = await api('/api/paper/accounts/' + id + '/trades'); } catch (e) {}
      try {
        const eq = await api('/api/paper/accounts/' + id + '/equity');
        this.paper.equity = eq.items || [];
      } catch (e) {}
      try { this.paper.events = await api('/api/paper/accounts/' + id + '/events'); } catch (e) {}
      this.$nextTick(() => this.renderPaperEquity());
    },
    renderPaperEquity() {
      const rows = this.paper.equity || [];
      if (!rows.length) return;
      renderLine('paperEquity', [{
        name: '模拟盘权益',
        dates: rows.map(x => x.date),
        values: rows.map(x => +(toNum(x.equity)).toFixed(2)),
      }]);
    },
    sortedPaperOrders() {
      const rows = (this.paper.orders || []).slice();
      const k = this.paper.orderSort.key;
      const dir = this.paper.orderSort.dir === 'asc' ? 1 : -1;
      return rows.sort((a, b) => sortCompare(a[k], b[k], dir));
    },
    setOrderSort(key) {
      if (this.paper.orderSort.key === key) {
        this.paper.orderSort.dir = this.paper.orderSort.dir === 'asc' ? 'desc' : 'asc';
      } else {
        this.paper.orderSort.key = key;
        this.paper.orderSort.dir = 'desc';
      }
    },
    orderArrow(key) {
      return this.paper.orderSort.key === key ? (this.paper.orderSort.dir === 'asc' ? '↑' : '↓') : '';
    },
    sortedPaperTrades() {
      const rows = (this.paper.trades || []).slice();
      const k = this.paper.tradeSort.key;
      const dir = this.paper.tradeSort.dir === 'asc' ? 1 : -1;
      return rows.sort((a, b) => sortCompare(a[k], b[k], dir));
    },
    setTradeSort(key) {
      if (this.paper.tradeSort.key === key) {
        this.paper.tradeSort.dir = this.paper.tradeSort.dir === 'asc' ? 'desc' : 'asc';
      } else {
        this.paper.tradeSort.key = key;
        this.paper.tradeSort.dir = 'desc';
      }
    },
    tradeArrow(key) {
      return this.paper.tradeSort.key === key ? (this.paper.tradeSort.dir === 'asc' ? '↑' : '↓') : '';
    },
    async loadPaperEventModules() {
      try {
        const r = await api('/api/paper/event-strategies');
        this.paper.eventModules = r.items || [];
      } catch (e) {}
    },
    paperEventStrategies() {
      const m = this.paper.eventModules.find(x => x.module === this.paper.form.module);
      return m ? m.strategies : [];
    },
    async createPaperAccount() {
      this.paper.error = '';
      this.paper.creating = true;
      try {
        const f = this.paper.form;
        const body = {
          name: f.name, strategy_type: f.strategy_type,
          module: f.module || null, event_strategy: f.event_strategy || null,
          strategy_name: f.strategy_type === 'event' ? f.event_strategy : f.strategy,
          universe: f.universe, capital: f.capital, top_n: f.top_n, freq: f.freq,
          risk_config: { max_weight: f.max_weight, amount_q: f.amount_q },
        };
        const r = await api('/api/paper/accounts', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (r.error) { this.paper.error = r.error; return; }
        await this.loadPaperAccounts();
      } catch (e) {
        this.paper.error = '创建失败: ' + e.message;
      } finally {
        this.paper.creating = false;
      }
    },
    async runPaper() {
      if (!this.paper.pick) return;
      this.paper.runError = '';
      this.paper.running = true;
      try {
        this.paper.lastRun = await api('/api/paper/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ account_id: this.paper.pick }),
        });
        if (this.paper.lastRun.error) {
          this.paper.runError = this.paper.lastRun.error;
          return;
        }
        await this.loadPaperAccounts();
      } catch (e) {
        this.paper.runError = '执行失败: ' + e.message;
      } finally {
        this.paper.running = false;
      }
    },
    async runPaperDry() {
      if (!this.paper.pick) return;
      this.paper.runError = '';
      try {
        const r = await api('/api/paper/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ account_id: this.paper.pick, dry_run: true }),
        });
        this.paper.lastRun = r;
        if (r.error) this.paper.runError = r.error;
      } catch (e) {
        this.paper.runError = 'dry-run 失败: ' + e.message;
      }
    },
    async togglePaperStatus() {
      const a = this.paper.accounts.find(x => x.id === this.paper.pick);
      if (!a) return;
      const status = a.status === 'active' ? 'paused' : 'active';
      try {
        await api('/api/paper/accounts/' + this.paper.pick, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ status }),
        });
        await this.loadPaperAccounts();
      } catch (e) {
        this.paper.runError = '状态切换失败: ' + e.message;
      }
    },
    async resetPaper() {
      if (!this.paper.pick) return;
      try {
        await api('/api/paper/accounts/' + this.paper.pick + '/reset', { method: 'POST' });
        await this.loadPaperAccounts();
      } catch (e) {
        this.paper.runError = '重置失败: ' + e.message;
      }
    },
    async deletePaper() {
      if (!this.paper.pick) return;
      try {
        await api('/api/paper/accounts/' + this.paper.pick, { method: 'DELETE' });
        this.paper.pick = null;
        await this.loadPaperAccounts();
      } catch (e) {
        this.paper.runError = '删除失败: ' + e.message;
      }
    },
  },
  mounted() {
    this.loadPaperAccounts();
  },
}
</script>
