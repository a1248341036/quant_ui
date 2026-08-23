<template>
    <section v-if="tab==='account'">
      <div class="grid" style="grid-template-columns:1fr 1fr">
        <div class="card">
          <h3>出入金</h3>
          <div class="form-grid">
            <label class="field"><span>日期</span><input type="date" v-model="dep.date"></label>
            <label class="field"><span>金额</span><input type="number" v-model.number="dep.amount"></label>
          </div>
          <div class="form-actions"><button class="primary" @click="addDeposit">入金</button></div>
          <div class="table-wrap" style="margin-top:10px"><table><tr><th>日期</th><th>金额</th><th>备注</th></tr>
          <tr v-for="(d,i) in deposits" :key="i"><td>{{d.date.slice(0,10)}}</td><td>{{fmt(d.amount)}}</td><td>{{d.note||'-'}}</td></tr></table></div>
          <div v-if="!deposits.length" class="empty">暂无出入金记录</div>
        </div>
        <div class="card">
          <h3>录入交易</h3>
          <div class="form-grid">
            <label class="field"><span>日期</span><input type="date" v-model="tx.date"></label>
            <label class="field"><span>代码</span><input v-model="tx.code" placeholder="601728"></label>
            <label class="field"><span>名称</span><input v-model="tx.name"></label>
            <label class="field"><span>方向</span><select v-model="tx.action"><option value="buy">买入</option><option value="sell">卖出</option></select></label>
            <label class="field"><span>股数</span><input type="number" v-model.number="tx.shares"></label>
            <label class="field"><span>价格</span><input type="number" v-model.number="tx.price"></label>
          </div>
          <div class="form-actions"><button class="primary" @click="addTx">提交</button></div>
          <div class="table-wrap" style="margin-top:10px"><table><tr><th>日期</th><th>代码</th><th>名称</th><th>方向</th><th>股数</th><th>价格</th></tr>
          <tr v-for="(t,i) in transactions" :key="i"><td>{{t.date.slice(0,10)}}</td><td>{{stock(t.code, t.name, names)}}</td><td>{{t.name}}</td><td>{{t.action}}</td><td>{{t.shares}}</td><td>{{fmt(t.price)}}</td></tr></table></div>
          <div v-if="!transactions.length" class="empty">暂无交易记录</div>
        </div>
      </div>
      <div class="card"><h3>当前持仓</h3>
        <div class="table-wrap"><table><tr><th>代码</th><th>名称</th><th>股数</th><th>成本</th><th>现价</th><th>市值</th><th>盈亏</th><th>盈亏%</th></tr>
        <tr v-for="p in positions" :key="p.code"><td>{{stock(p.code, p.name, names)}}</td><td>{{p.name||'-'}}</td><td>{{p.shares}}</td><td>{{fmt(p.avg_cost)}}</td><td>{{fmt(p.price)}}</td><td>{{fmt(p.market_value)}}</td><td :class="sign(p.pnl)">{{fmt(p.pnl)}}</td><td :class="sign(p.pnl_pct)">{{pct(p.pnl_pct)}}</td></tr></table></div>
        <div v-if="!positions.length" class="empty">当前空仓</div>
      </div>
    </section>
</template>

<script>
import { store } from '../store/index.js'
import { fmt, pct, sign, stock, today } from '../utils/format.js'
import { api } from '../utils/api.js'
import { renderLine } from '../utils/charts.js'

export default {
  name: 'AccountView',
  data() {
    return {
      tab: 'account',
      deposits: [],
      transactions: [],
      positions: [],
      dep: { date: today(), amount: 5000 },
      tx: { date: today(), code: '', name: '', action: 'buy', shares: 100, price: 0, fee: 0 },
      accSummary: null,
      accItems: [],
      accEmpty: true,
    };
  },
  computed: {
    names() {
      return store.names || {};
    },
  },
  methods: {
    fmt,
    pct,
    sign,
    stock,
    today,
    async addDeposit() {
      await api('/api/ledger/deposits', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(this.dep) });
      await this.loadAccount(); this.$root.switchTab('dashboard');
    },
    async addTx() {
      if (!this.tx.code) { alert('请填代码'); return; }
      await api('/api/ledger/transactions', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(this.tx) });
      await this.loadAccount(); this.$root.switchTab('dashboard');
    },
    async loadAccount() {
      try {
        const eq = await api('/api/ledger/equity');
        this.accSummary = eq.summary; this.accItems = eq.items || []; this.accEmpty = eq.items.length === 0;
        if (this.accEmpty) { this.accSummary = null; }
        else { this.$nextTick(() => renderLine('dashEquity', this.accountSeries())); }
      } catch (e) { this.accEmpty = true; }
      try { this.deposits = await api('/api/ledger/deposits'); } catch (e) {}
      try { this.transactions = await api('/api/ledger/transactions'); } catch (e) {}
      try { this.positions = await api('/api/ledger/positions'); } catch (e) {}
    },
    accountSeries() {
      const items = this.accItems;
      return [
        { name: '总资产', dates: items.map(x => x.date.slice(0, 10)), values: items.map(x => +x.equity.toFixed(2)) },
        { name: '持仓市值', dates: items.map(x => x.date.slice(0, 10)), values: items.map(x => +x.market_value.toFixed(2)) }
      ];
    },
  },
  mounted() {
    this.loadAccount();
  },
};
</script>
