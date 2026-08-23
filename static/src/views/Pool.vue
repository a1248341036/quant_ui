<template>
    <section v-if="tab==='pool'">
      <div class="row" style="align-items:center; justify-content:space-between; margin-bottom:4px">
        <h3>策略池</h3>
        <span class="muted">全量池 = 代码注册策略 + 回测归档策略；配置池统一管理所有页面（回测 / 模拟盘 / 信号 / 对比 / 参数稳健性）的策略下拉</span>
      </div>
      <div class="grid" style="grid-template-columns:1.55fr 1fr">
        <div class="card">
          <h3>配置池 <span class="muted">（{{pool.pooled.length}}）</span></h3>
          <div v-if="!pool.pooled.length" class="empty">配置池为空，从下方全量池选择策略加入。</div>
          <div v-else>
            <div class="table-wrap">
              <table>
                <tr><th>策略</th><th>分组</th><th>因子</th><th>来源</th><th>回测</th><th>夏普</th><th>总收益</th><th>年化</th><th>最大回撤</th></tr>
                <tr v-for="s in pool.pooled" :key="s.name">
                  <td>{{s.name}}</td><td>{{s.group}}</td><td>{{s.factor}}</td><td>{{s.source}}</td>
                  <td>{{s.n_runs || 0}}</td><td>{{fmt(s.sharpe,2)}}</td><td>{{pct(s.total_return)}}</td><td>{{pct(s.annual)}}</td><td>{{pct(s.mdd)}}</td>
                </tr>
              </table>
            </div>
            <div class="row" style="gap:8px; margin-top:10px">
              <select v-model="pool.rmSel" class="pool-rm-sel" multiple style="min-width:240px; height:auto">
                <option v-for="s in pool.pooled" :value="s.name">{{s.name}}</option>
              </select>
              <button class="ghost" @click="removePool" :disabled="!pool.rmSel.length">移除所选（仅页面配置移除，全量池保留）</button>
            </div>
            <p class="muted" style="margin-top:6px">移除后，回测 / 模拟盘 / 信号等页面的策略下拉同步消失；策略仍在下方全量池，可随时重新加入。</p>
          </div>
        </div>
        <div class="card">
          <h3>回收站 <span class="muted">（{{pool.trash.length}}）</span></h3>
          <div v-if="!pool.trash.length" class="empty">回收站为空（从全量池删除的策略在这里，彻底删除才真正移除）</div>
          <div v-else>
            <div class="table-wrap">
              <table>
                <tr><th>策略</th><th>来源</th><th>操作</th></tr>
                <tr v-for="s in pool.trash" :key="s.name">
                  <td>{{s.name}}</td><td>{{s.source}}</td>
                  <td style="white-space:nowrap">
                    <button class="ghost sm" @click="restorePool(s.name)">恢复</button>
                    <button class="danger sm" @click="purgePool(s.name)">彻底删除</button>
                  </td>
                </tr>
              </table>
            </div>
            <button class="danger" style="margin-top:10px" @click="emptyTrashPool">清空回收站</button>
          </div>
        </div>
      </div>
      <div class="card">
        <h3>全量池 <span class="muted">（{{pool.full.length}}）</span></h3>
        <div class="row" style="gap:8px; margin-bottom:8px">
          <input v-model="pool.q" placeholder="搜索策略名称 / 说明…" style="flex:1">
          <select v-model="pool.grp" style="width:180px">
            <option>全部</option>
            <option v-for="g in poolGroups" :value="g">{{g}}</option>
          </select>
        </div>
        <div class="table-wrap" style="max-height:420px; overflow:auto">
          <table>
            <tr><th>策略</th><th>分组</th><th>因子</th><th>来源</th><th>股票池</th><th>回测</th><th>夏普</th><th>总收益</th><th>年化</th><th>最大回撤</th><th>说明</th></tr>
            <tr v-for="s in fullFiltered()" :key="s.name">
              <td>{{s.name}} <span v-if="pooledNames.includes(s.name)" class="chip">已配置</span></td>
              <td>{{s.group}}</td><td>{{s.factor}}</td><td>{{s.source}}</td>
              <td>{{s.universe || '-'}}</td><td>{{s.n_runs || 0}}</td>
              <td>{{fmt(s.sharpe,2)}}</td><td>{{pct(s.total_return)}}</td><td>{{pct(s.annual)}}</td><td>{{pct(s.mdd)}}</td>
              <td class="muted">{{s.desc || ''}}</td>
            </tr>
          </table>
        </div>
        <div class="row" style="gap:8px; margin-top:10px">
          <select v-model="pool.addSel" class="pool-add-sel" multiple style="min-width:280px; height:auto">
            <option v-for="s in fullUnpooled" :value="s.name">{{s.name}}</option>
          </select>
          <button class="primary" @click="addPool" :disabled="!pool.addSel.length">加入配置池</button>
          <span class="muted">加入后，回测 / 模拟盘 / 信号 / 对比 / 参数稳健性的策略下拉同步生效</span>
        </div>
        <div class="row" style="gap:8px; margin-top:10px">
          <select v-model="pool.delSel" class="pool-del-sel" multiple style="min-width:280px; height:auto">
            <option v-for="s in fullFiltered()" :value="s.name">{{s.name}}</option>
          </select>
          <button class="danger" @click="deleteFromFull" :disabled="!pool.delSel.length">删除进回收站</button>
          <span class="muted">全量池删除会同时从配置池移除；回收站可恢复</span>
        </div>
      </div>
    </section>
</template>

<script>
import { store } from '../store/index.js'
import { fmt, pct, sign } from '../utils/format.js'
import { api } from '../utils/api.js'

export default {
  name: 'PoolView',
  data() {
    return {
      tab: 'pool',
    };
  },
  computed: {
    pool() {
      return store.pool;
    },
    pooledNames() {
      return this.pool.pooled.map(s => s.name);
    },
    poolGroups() {
      const set = new Set();
      (this.pool.full || []).forEach(s => { if (s.group) set.add(s.group); });
      return [...set].sort();
    },
    fullUnpooled() {
      const pooled = new Set(this.pooledNames);
      return this.fullFiltered().filter(s => !pooled.has(s.name));
    },
  },
  methods: {
    fmt,
    pct,
    sign,
    async loadPoolData() {
      await store.loadPoolData();
    },
    fullFiltered() {
      let arr = this.pool.full || [];
      const q = (this.pool.q || '').trim().toLowerCase();
      if (q) {
        arr = arr.filter(s =>
          (s.name || '').toLowerCase().includes(q) ||
          (s.desc || '').toLowerCase().includes(q) ||
          (s.factor || '').toLowerCase().includes(q));
      }
      if (this.pool.grp !== '全部') arr = arr.filter(s => s.group === this.pool.grp);
      return arr;
    },
    ensureSelectedInPool() {
      store.ensureSelectedInPool();
    },
    async reloadStrategies() {
      await store.reloadStrategies();
    },
    async addPool() {
      if (!this.pool.addSel.length) return;
      const names = this.pool.addSel.slice();
      let ok = 0, err = '';
      for (const n of names) {
        try {
          const r = await api('/api/strategy-pool/add', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: n }) });
          if (r && r.ok) ok++; else err = (r && r.error) || '';
        } catch (e) { err = e.message || String(e); }
      }
      await this.loadPoolData();
      await this.reloadStrategies();
      if (ok) alert('已加入 ' + ok + ' 个策略到配置池');
      if (err) alert('部分失败：' + err);
    },
    async removePool() {
      if (!this.pool.rmSel.length) return;
      const names = this.pool.rmSel.slice();
      let ok = 0, err = '';
      for (const n of names) {
        try {
          const r = await api('/api/strategy-pool/remove', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: n }) });
          if (r && r.ok) ok++; else err = (r && r.error) || '';
        } catch (e) { err = e.message || String(e); }
      }
      await this.loadPoolData();
      await this.reloadStrategies();
      if (ok) alert('已从配置池移除 ' + ok + ' 个（全量池保留）');
      if (err) alert('移除失败：' + err);
    },
    async deleteFromFull() {
      if (!this.pool.delSel.length) return;
      const names = this.pool.delSel.slice();
      let ok = 0, err = '';
      for (const n of names) {
        try {
          const r = await api('/api/strategy-pool/full-delete', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: n }) });
          if (r && r.ok) ok++; else err = (r && r.error) || '';
        } catch (e) { err = e.message || String(e); }
      }
      await this.loadPoolData();
      await this.reloadStrategies();
      if (ok) alert('已删除 ' + ok + ' 个到回收站');
      if (err) alert('删除失败：' + err);
    },
    async restorePool(name) {
      try {
        await api('/api/strategy-pool/restore', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name }) });
      } catch (e) {}
      await this.loadPoolData();
      await this.reloadStrategies();
    },
    async purgePool(name) {
      try { await api('/api/strategy-pool/trash/purge', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name }) }); } catch (e) {}
      await this.loadPoolData();
      await this.reloadStrategies();
    },
    async emptyTrashPool() {
      try { await api('/api/strategy-pool/trash/empty', { method: 'POST' }); } catch (e) {}
      await this.loadPoolData();
      await this.reloadStrategies();
    },
  },
  mounted() {
    this.loadPoolData();
  },
};
</script>
