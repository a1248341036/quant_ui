<template>
  <section>
    <div class="cards" v-if="sentStats.ok">
      <div class="metric"><div class="label">新闻总数</div><div class="value">{{sentStats.n_articles}}</div></div>
      <div class="metric"><div class="label">覆盖股票</div><div class="value">{{sentStats.n_codes}}</div></div>
      <div class="metric"><div class="label">数据区间</div><div class="value range">{{sentStats.first_date || '—'}} ~ {{sentStats.last_date || '—'}}</div></div>
      <div class="metric"><div class="label">情绪分均值</div><div class="value">{{fmt(sentStats.mean_score, 3)}}</div></div>
    </div>
    <div v-else class="card"><div class="empty">{{sentError || '暂无舆情数据：确认 data/sentiment-mvp/articles.db 或 CNE curated sentiment_articles 可用'}}</div></div>

    <div class="card src-card" v-if="sentSources.length">
      <h3>数据源</h3>
      <div class="src-bar">
        <span class="src-chip" v-for="s in sentSources" :key="s.name">
          <b>{{srcLabel(s)}}</b>
          <span class="src-n">{{s.rows}} 条</span>
          <span class="src-range" v-if="s.date_range">{{s.date_range.start}} ~ {{s.date_range.end}}</span>
          <span class="src-err" v-if="s.error" :title="s.error">不可用</span>
        </span>
        <span class="src-total">合并去重后 {{sentStats.n_articles}} 条（按 article_id）</span>
      </div>
    </div>

    <div class="grid" style="grid-template-columns:1fr 1fr">
      <div class="card"><h3>情绪标签分布</h3><div id="sentLabel" class="chart small"></div></div>
      <div class="card"><h3>每日新闻条数（近 30 天）</h3><div id="sentDaily" class="chart small"></div></div>
    </div>

    <div class="card">
      <h3>舆情分桶回测 IC / 分组 <span class="sub" v-if="sentIcWindow">回测区间 {{sentIcWindow}}</span></h3>
      <p class="muted">由 scripts/sentiment_backtest.py 输出，衡量舆情情绪因子对未来收益的预测力。</p>
      <div class="table-wrap"><table>
        <tr><th>策略</th><th>类型</th><th>IC 均值</th><th>ICIR</th><th>t 值</th><th>IC&gt;0 占比</th><th>多空价差</th><th>年化价差</th><th>方向调整 IC</th></tr>
        <tr v-for="(r,i) in sentIc" :key="i">
          <td>{{r['策略']}}</td><td>{{r['类型']}}</td>
          <td :class="sign(r['IC均值'])">{{fmt(r['IC均值'], 4)}}</td><td>{{fmt(r['ICIR'], 3)}}</td><td>{{fmt(r['t值'], 2)}}</td>
          <td>{{pct(r['IC>0占比'])}}</td><td>{{fmt(r['多空价差%'], 2)}}%</td><td>{{fmt(r['多空价差年化%'], 2)}}%</td><td :class="sign(r['方向调整IC'])">{{fmt(r['方向调整IC'], 4)}}</td>
        </tr>
      </table></div>
      <div v-if="!sentIc.length" class="empty">暂无 IC 结果，先运行 scripts/sentiment_backtest.py</div>
    </div>

    <div class="grid" style="grid-template-columns:1fr 1fr">
      <div class="card">
        <h3>情绪最强新闻 <span class="sub">近 7 天</span></h3>
        <div class="news-list">
          <a v-for="(n,i) in sentNewsHigh" :key="'h'+i" :href="n.url" target="_blank" class="news-item">
            <div class="news-title">{{n.title}}</div>
            <div class="news-meta">{{n.code}} · {{n.publish_time}} · {{n.media}} · {{n.source}} · score {{fmt(n.score, 3)}}</div>
          </a>
          <div v-if="!sentNewsHigh.length" class="empty">暂无</div>
        </div>
      </div>
      <div class="card">
        <h3>情绪最弱新闻 <span class="sub">近 7 天</span></h3>
        <div class="news-list">
          <a v-for="(n,i) in sentNewsLow" :key="'l'+i" :href="n.url" target="_blank" class="news-item">
            <div class="news-title">{{n.title}}</div>
            <div class="news-meta">{{n.code}} · {{n.publish_time}} · {{n.media}} · {{n.source}} · score {{fmt(n.score, 3)}}</div>
          </a>
          <div v-if="!sentNewsLow.length" class="empty">暂无</div>
        </div>
      </div>
    </div>
  </section>
</template>

<script>
import { api } from '../utils/api.js'
import { fmt, pct, sign } from '../utils/format.js'
import { renderBar, renderLine } from '../utils/charts.js'

const SOURCE_LABELS = {
  legacy_articles_db: '本地历史库',
  cne_curated: 'CNE curated',
}

export default {
  name: 'Sentiment',
  data() {
    return {
      sentStats: { ok: false },
      sentError: '',
      sentSources: [],
      sentIc: [],
      sentIcWindow: '',
      sentNewsHigh: [],
      sentNewsLow: [],
    }
  },
  mounted() {
    this.loadSentiment()
  },
  methods: {
    fmt,
    pct,
    sign,
    srcLabel(s) {
      return SOURCE_LABELS[s.name] || s.name
    },
    async loadSentiment() {
      try {
        const s = await api('/api/sentiment/stats');
        this.sentStats = s; this.sentError = s.error || '';
        this.sentSources = s.sources || [];
        this.$nextTick(() => this.renderSentiment());
      } catch (e) { this.sentError = e.message; }
      try { const r = await api('/api/sentiment/ic'); this.sentIc = r.items || []; this.sentIcWindow = r.window || ''; } catch (e) {}
      try { const h = await api('/api/sentiment/news?top=8&sort=high&days=7'); this.sentNewsHigh = h.items || []; } catch (e) {}
      try { const l = await api('/api/sentiment/news?top=8&sort=low&days=7'); this.sentNewsLow = l.items || []; } catch (e) {}
    },
    renderSentiment() {
      const ld = this.sentStats.label_dist || {};
      const keys = Object.keys(ld);
      if (keys.length) renderBar('sentLabel', keys, keys.map(k => ld[k]));
      const daily = this.sentStats.daily || [];
      if (daily.length) renderLine('sentDaily', [{ name: '条数', dates: daily.map(d => d.date), values: daily.map(d => d.n) }], 220);
    },
  },
}
</script>

<style scoped>
.metric .value.range { font-size: 13px; line-height: 1.6; }
.src-card { margin-top: 4px; }
.src-bar { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.src-chip {
  display: inline-flex; align-items: baseline; gap: 6px;
  border: 1px solid var(--line); border-radius: 8px; padding: 4px 10px;
  font-size: 11px; color: var(--muted); background: rgb(255 255 255 / .02);
}
.src-chip b { color: var(--text, #d8dee9); font-weight: 600; }
.src-n { font-family: var(--font-mono); }
.src-range { font-family: var(--font-mono); color: var(--muted); }
.src-err { color: #ef6b73; }
.src-total { font-size: 11px; color: var(--muted); }
.sub { font-size: 10px; font-weight: 400; color: var(--muted); margin-left: 4px; }
</style>
