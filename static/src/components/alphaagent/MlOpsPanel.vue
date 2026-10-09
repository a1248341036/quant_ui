<template>
  <div class="ml-ops">
    <div class="card">
      <div class="card-head">
        <h3>ML 组合运营</h3>
        <button class="ghost" type="button" :disabled="loading" @click="load">
          <span v-if="loading" class="spinner"></span>{{ loading ? '加载中…' : '刷新' }}
        </button>
      </div>
      <p v-if="error" class="err left">{{ error }}</p>

      <template v-if="ops">
        <!-- ── 周节奏四卡 ── -->
        <div class="cards">
          <div class="metric">
            <div class="label">分数截止日</div>
            <div class="value">{{ ops.score?.date_max || '—' }}</div>
          </div>
          <div class="metric">
            <div class="label">覆盖股票 / 天数</div>
            <div class="value">{{ ops.score?.n_codes ?? '—' }} / {{ ops.score?.days ?? '—' }}</div>
          </div>
          <div class="metric">
            <div class="label">下次分数更新（周五收盘后）</div>
            <div class="value">{{ ops.weekly_rhythm?.next_score_update || '—' }}</div>
          </div>
          <div class="metric">
            <div class="label">下次调仓（周一开盘）</div>
            <div class="value">{{ ops.weekly_rhythm?.next_rebalance || '—' }}</div>
          </div>
        </div>

        <!-- ── 生效分数 ── -->
        <div class="card sub">
          <h4>当前生效分数</h4>
          <table v-if="ops.score?.exists">
            <tbody>
              <tr><th>来源训练</th><td>{{ ops.score.source_train_id || '—' }}<span class="muted">（{{ ops.score.source_rows?.toLocaleString() }} 行全池分数）</span></td></tr>
              <tr><th>日期覆盖</th><td>{{ ops.score.date_min }} ~ {{ ops.score.date_max }}（{{ ops.score.days }} 个交易日）</td></tr>
              <tr><th>股票数</th><td>{{ ops.score.n_codes }} 只 / 最新日 {{ ops.score.latest_day_codes }} 只</td></tr>
              <tr>
                <th>低价过滤（≤10 元）</th>
                <td>
                  <span v-if="ops.score.filter_low_price === true" class="badge ok">已启用</span>
                  <span v-else-if="ops.score.filter_low_price === false" class="badge bad">未启用</span>
                  <span v-else class="badge">未知</span>
                </td>
              </tr>
            </tbody>
          </table>
          <p v-else class="muted">pred_demo.parquet 不存在——先完成一次训练并推送分数</p>
        </div>

        <!-- ── 最新名单 ── -->
        <div class="card sub">
          <h4>最新主板名单 <span class="muted">（{{ ops.watchlist?.score_date }} · 池 {{ ops.watchlist?.pool_size }} 只 · top10 一手合计 {{ fmt(ops.watchlist?.total_lot_amount) }} 元）</span></h4>
          <table v-if="ops.watchlist?.items?.length">
            <thead><tr><th>#</th><th>代码</th><th>分数</th><th>收盘</th><th>一手金额</th><th>≤700元/手</th></tr></thead>
            <tbody>
              <tr v-for="it in ops.watchlist.items" :key="it.code">
                <td>{{ it.rank }}</td>
                <td><b>{{ it.code }}</b></td>
                <td>{{ it.score.toFixed(4) }}</td>
                <td>{{ it.close ?? '—' }}</td>
                <td>{{ it.lot_amount ?? '—' }}</td>
                <td><span :class="it.affordable_700 ? 'badge ok' : 'badge bad'">{{ it.affordable_700 ? '✓' : '✗' }}</span></td>
              </tr>
            </tbody>
          </table>
          <p v-else class="muted">暂无可出名单</p>
          <p class="muted" style="margin-top:6px">执行：周一开盘按序每只一手；无创业板/科创板权限，本表仅含沪深主板。</p>
        </div>

        <!-- ── 模拟盘对照 ── -->
        <div class="card sub">
          <h4>模拟盘对照</h4>
          <table v-if="ops.paper_accounts?.length && !ops.paper_accounts[0].error">
            <thead><tr><th>账户</th><th>池/频率/TopN</th><th>状态</th><th>权益</th><th>累计盈亏</th><th>持仓</th><th>起始日</th></tr></thead>
            <tbody>
              <tr v-for="a in ops.paper_accounts" :key="a.id">
                <td>#{{ a.id }} {{ a.name }}</td>
                <td>{{ a.universe }} · {{ a.freq }} · {{ a.top_n }}</td>
                <td>{{ a.status }}</td>
                <td>{{ fmt(a.equity) }}</td>
                <td :class="a.pnl >= 0 ? 'pos' : 'neg'">{{ fmt(a.pnl) }}（{{ a.pnl_pct }}%）</td>
                <td>{{ a.n_positions ?? '—' }}</td>
                <td>{{ a.start_date || '—' }}</td>
              </tr>
            </tbody>
          </table>
          <p v-else class="muted">无 pred 因子驱动的模拟盘账户</p>
        </div>

        <!-- ── 训练历史 ── -->
        <div class="card sub">
          <h4>训练历史 <span class="muted">（详见 AlphaAgent → ML 组合面板）</span></h4>
          <table v-if="ops.trainings?.length">
            <thead><tr><th>训练</th><th>方案</th><th>特征/折</th><th>OOS IC</th><th>ICIR</th><th>衰减保留</th><th>Gate</th><th>状态</th></tr></thead>
            <tbody>
              <tr v-for="t in ops.trainings" :key="t.train_id" :class="{ current: t.train_id === ops.score?.source_train_id }">
                <td>{{ t.train_id }}</td>
                <td>{{ t.scheme_label || t.scheme || '—' }}</td>
                <td>{{ t.n_features ?? '—' }} / {{ t.n_folds ?? '—' }}</td>
                <td>{{ num(t.oos_ic_mean) }}</td>
                <td>{{ num(t.oos_ic_ir) }}</td>
                <td>{{ t.decay_retention != null ? (t.decay_retention * 100).toFixed(0) + '%' : '—' }}</td>
                <td><span :class="t.gate_passed ? 'badge ok' : 'badge'">{{ t.gate_passed ? '通过' : '未过' }}</span></td>
                <td>{{ t.status }}</td>
              </tr>
            </tbody>
          </table>
          <p v-else class="muted">暂无训练记录</p>
        </div>
      </template>
    </div>
  </div>
</template>

<script>
import { api } from '../../utils/api.js'

export default {
  name: 'MlOpsPanel',
  data() {
    return { ops: null, loading: false, error: '' }
  },
  mounted() { this.load() },
  methods: {
    async load() {
      this.loading = true
      this.error = ''
      try {
        this.ops = await api('/api/alphaagent/stacking/ops')
      } catch (e) {
        this.error = '加载失败: ' + e.message
      } finally {
        this.loading = false
      }
    },
    fmt(v) {
      return v == null ? '—' : Number(v).toLocaleString(undefined, { maximumFractionDigits: 2 })
    },
    num(v) {
      return v == null ? '—' : Number(v).toFixed(4)
    },
  },
}
</script>

<style scoped>
.ml-ops .card-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.ml-ops .card.sub { margin-top: 14px; }
.ml-ops h4 { margin: 0 0 8px; font-size: 14px; }
.ml-ops table { width: 100%; border-collapse: collapse; font-size: 13px; }
.ml-ops th, .ml-ops td { text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--border, #e5e7eb); }
.ml-ops thead th { color: var(--muted, #6b7280); font-weight: 600; font-size: 12px; }
.ml-ops tr.current td { background: var(--accent-weak, rgba(59,130,246,.08)); }
.ml-ops .badge { display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 12px; background: var(--muted-weak, #f3f4f6); }
.ml-ops .badge.ok { color: #047857; background: rgba(16,185,129,.12); }
.ml-ops .badge.bad { color: #b91c1c; background: rgba(239,68,68,.12); }
.ml-ops .pos { color: #047857; }
.ml-ops .neg { color: #b91c1c; }
.ml-ops .left { text-align: left; }
</style>
