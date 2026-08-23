<template>
  <a class="skip-link" href="#main">跳到主内容</a>
  <div class="topbar">
    <header>
      <div class="brand">
        <span class="brand-mark"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 17l5-6 4 4 7-9"/><path d="M14 6h6v6"/></svg></span>
        <h1>量化回测工作台</h1>
      </div>
      <div class="header-right">
        <span class="status-chip"><span class="status-dot"></span>数据至 {{store.panelInfo.last_date || '—'}} · {{store.panelInfo.n_codes || 0}} 只</span>
      </div>
    </header>

    <nav aria-label="主导航">
      <button :class="{active: tab==='dashboard'}" @click="switchTab('dashboard')">
        <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></svg>看板
      </button>
      <button :class="{active: tab==='pool'}" @click="switchTab('pool')">
        <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 6h16M4 12h16M4 18h10"/><circle cx="18" cy="18" r="2.5"/></svg>策略池
      </button>
      <button :class="{active: tab==='stock'}" @click="switchTab('stock')">
        <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="m7 15 3-4 3 2 5-7"/></svg>个股
      </button>
      <button :class="{active: tab==='backtest'}" @click="switchTab('backtest')">
        <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19 9 13l4 4 7-9"/><path d="M15 5h5v5"/></svg>回测
      </button>
      <button :class="{active: tab==='composite'}" @click="switchTab('composite')">
        <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2 3 7v10l9 5 9-5V7z"/><path d="M12 22V12"/><path d="M3 7l9 5 9-5"/></svg>组合
      </button>
      <button :class="{active: tab==='code'}" @click="switchTab('code')">
        <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="m8 7-5 5 5 5"/><path d="m16 7 5 5-5 5"/></svg>代码
      </button>
      <button :class="{active: tab==='alphaagent'}" @click="switchTab('alphaagent')">
        <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="8"/><path d="M8 12h8M12 8v8"/><path d="M19 5 21 3"/></svg>AlphaAgent
      </button>
      <button :class="{active: tab==='paper'}" @click="switchTab('paper')">
        <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M8 9h8M8 13h5"/></svg>模拟盘
      </button>
      <button :class="{active: tab==='history'}" @click="switchTab('history')">
        <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 3"/></svg>历史
      </button>

      <button :class="{active: tab==='account'}" @click="switchTab('account')">
        <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="6" width="18" height="14" rx="2"/><path d="M16 12h5v3h-5a2 2 0 0 1 0-4z"/></svg>账户
      </button>
      <button :class="{active: tab==='data'}" @click="switchTab('data')">
        <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/></svg>数据
      </button>
      <button :class="{active: tab==='sentiment'}" @click="switchTab('sentiment')">
        <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a8 8 0 0 1-8 8H4l2-3a8 8 0 1 1 15-5z"/></svg>舆情
      </button>
    </nav>
  </div>

  <main id="main">
    <Dashboard v-if="tab==='dashboard'" />
    <Pool v-else-if="tab==='pool'" />
    <Stock v-else-if="tab==='stock'" />
    <Backtest v-else-if="tab==='backtest'" />
    <Composite v-else-if="tab==='composite'" />
    <CodeView v-else-if="tab==='code'" />
    <AlphaAgent v-else-if="tab==='alphaagent'" />
    <Paper v-else-if="tab==='paper'" />
    <History v-else-if="tab==='history'" />

    <Account v-else-if="tab==='account'" />
    <DataView v-else-if="tab==='data'" />
    <Sentiment v-else-if="tab==='sentiment'" />
  </main>
</template>

<script>
import { store } from './store/index.js'
import { api } from './utils/api.js'
import { today } from './utils/format.js'

import Dashboard from './views/Dashboard.vue'
import Pool from './views/Pool.vue'
import Stock from './views/Stock.vue'
import Backtest from './views/Backtest.vue'
import Composite from './views/Composite.vue'
import CodeView from './views/Code.vue'
import AlphaAgent from './views/AlphaAgent.vue'
import Paper from './views/Paper.vue'
import History from './views/History.vue'
import Account from './views/Account.vue'
import DataView from './views/Data.vue'
import Sentiment from './views/Sentiment.vue'

export default {
  name: 'App',
  components: { Dashboard, Pool, Stock, Backtest, Composite, CodeView, AlphaAgent, Paper, History, Account, DataView, Sentiment },
  data() {
    return {
      store,
      tab: 'dashboard',
    }
  },
  mounted() {
    this.init()
  },
  methods: {
    today,
    async init() {
      store.loadIndices()
      try { store.panelInfo = await api('/api/data/panel-info'); } catch (e) {}
      try { store.strategies = await api('/api/strategies?universe=科技TMT'); } catch (e) {}
      store.loadPoolData()
      try { store.names = await api('/api/names'); } catch (e) {}
    },
    switchTab(t) {
      this.tab = t
    },
  },
}
</script>
