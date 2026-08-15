from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

from core.data import data_status, load_index, load_panel, load_tech, load_universe
from core.engine import latest_signals, run_backtest
from core.fetcher import update_data
from strategies.registry import get_strategy, list_strategies


st.set_page_config(page_title="A股量化回测工作台", page_icon="📈", layout="wide")


@st.cache_data(show_spinner=False)
def load_data(version: str = ""):
    return load_panel(), load_universe(), load_tech(), load_index()


def data_version() -> str:
    from core.store import META_FILE
    return str(META_FILE.stat().st_mtime) if META_FILE.exists() else "legacy"


# ---------------- 舆情情绪（sentiment-mvp 数据） ----------------
SENT_ROOT = Path("/home/ubuntu/quant/sentiment-mvp")
sys.path.insert(0, str(SENT_ROOT))

from lib import store as sent_store  # noqa: E402

SENT_DB = SENT_ROOT / "data/articles.db"
SENT_STUDY_CSV = SENT_ROOT / "outputs/event_study_daily.csv"
SENT_UNIVERSE_CSV = SENT_ROOT / "data/universe.csv"


@st.cache_data(ttl=300, show_spinner=False)
def load_news_stats():
    return sent_store.stats(SENT_DB) if SENT_DB.exists() else {}


@st.cache_data(ttl=300, show_spinner=False)
def load_news_articles():
    return sent_store.load_articles(SENT_DB) if SENT_DB.exists() else pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def load_news_study():
    if not SENT_STUDY_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(SENT_STUDY_CSV)


@st.cache_data(ttl=3600, show_spinner=False)
def load_news_universe():
    if not SENT_UNIVERSE_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(SENT_UNIVERSE_CSV, dtype={"code": str})
    df["code"] = df["code"].astype(str).str.zfill(6)
    return df


def get_industry_map(tech) -> dict[str, str]:
    return {str(c).zfill(6): str(ind)
            for c, ind in zip(tech["code"], tech["industry"])}


def get_name_map(uni, tech) -> dict[str, str]:
    m = {}
    for df in (uni, tech):
        if "code" in df and "name" in df:
            for code, name in zip(df["code"], df["name"]):
                code = str(code).zfill(6)
                if name and not pd.isna(name):
                    m.setdefault(code, str(name))
    return m


def build_codes(universe: str, exclude_kechuang: bool, panel, uni, tech) -> list[str]:
    if universe == "科技行业":
        codes = set(tech["code"])
    else:
        codes = set(uni["code"])
    codes &= set(panel["code"].unique())
    if exclude_kechuang:
        codes = {c for c in codes if not c.startswith(("300", "301", "688", "689"))}
    return sorted(codes)


def format_pct(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{x * 100:.2f}%"


def equity_compare_chart(navs: dict[str, pd.Series], capital: float,
                         bench: pd.Series | None = None) -> go.Figure:
    fig = go.Figure()
    if bench is not None:
        fig.add_trace(go.Scatter(x=bench.index, y=bench.values * capital,
                                 name="基准(等权)", mode="lines",
                                 line=dict(width=1.5, dash="dash", color="#8b98b5")))
    for name, nav in navs.items():
        fig.add_trace(go.Scatter(x=nav.index, y=nav.values * capital, name=name,
                                 mode="lines", line=dict(width=2)))
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10),
                      title="策略资金对比", legend=dict(orientation="h", y=1.05),
                      yaxis_title="资金")
    return fig


def equity_chart(nav: pd.Series, bench: pd.Series, capital: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nav.index, y=nav.values * capital,
                             name="策略资金", line=dict(color="#e8503a", width=2)))
    fig.add_trace(go.Scatter(x=bench.index, y=bench.values * capital,
                             name="基准(等权股票池)", line=dict(color="#888", width=1.5, dash="dash")))
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10),
                      title="资金曲线", legend=dict(orientation="h", y=1.05),
                      yaxis_title="资金")
    return fig


def drawdown_chart(dd: pd.Series) -> go.Figure:
    fig = go.Figure(go.Scatter(x=dd.index, y=dd.values * 100, name="回撤",
                               fill="tozeroy", line=dict(color="#e8503a")))
    fig.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10),
                      title="回撤(%)", yaxis_title="%")
    return fig


def render_metrics(cols, metrics: dict):
    order = ["总收益", "年化收益", "夏普", "最大回撤", "卡玛", "胜率"]
    for col, key in zip(cols, order):
        col.metric(key, format_pct(metrics.get(key)))


def main():
    try:
        panel, uni, tech, index = load_data(data_version())
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    last_date = panel["date"].max()
    first_date = panel["date"].min()

    with st.sidebar:
        st.title("📈 量化回测工作台")
        st.caption(f"数据区间 {first_date.date()} ~ {last_date.date()}")
        st.caption(f"股票池 {panel['code'].nunique()} 只")
        st.markdown("---")
        st.markdown("**Demo 说明**：回测使用本地 2020-2026 CSI800 面板数据，"
                    "资金曲线为策略模拟跟踪，真实账户记账将在下阶段加入。")

    tab_dash, tab_bt, tab_sig, tab_news, tab_data = st.tabs(
        ["📊 资金看板", "🛠 回测工作台", "🎯 今日信号", "📰 舆情情绪", "ℹ️ 数据状态"]
    )

    # ---------------- 资金看板 ----------------
    with tab_dash:
        st.subheader("资金看板（模拟跟踪）")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            dash_capital = st.number_input("初始资金", value=5000.0, min_value=1000.0,
                                           step=1000.0, key="dash_capital")
        with col2:
            dash_strategies = st.multiselect("策略（可多选对比）", list_strategies(),
                                             default=["低换手冷门", "反转 20 日",
                                                      "低波动", "动量 20 日"],
                                             key="dash_strategies")
        with col3:
            dash_topn = st.select_slider("持仓数 TopN", options=[1, 2, 3, 5, 8],
                                         value=3, key="dash_topn")
        with col4:
            dash_months = st.slider("回看月份", 1, 24, 6, key="dash_months")

        if dash_strategies:
            dash_end = last_date
            dash_start = last_date - pd.DateOffset(months=dash_months)
            codes = build_codes("科技行业", True, panel, uni, tech)
            navs: dict[str, pd.Series] = {}
            rows = []
            bench_line: pd.Series | None = None
            for sname in dash_strategies:
                strat = get_strategy(sname)
                with st.spinner(f"跑 {sname}..."):
                    res = run_backtest(
                        panel=panel, codes=codes, factor=strat["factor"],
                        ascending=strat["ascending"],
                        start=dash_start.strftime("%Y-%m-%d"), end=dash_end.strftime("%Y-%m-%d"),
                        capital=float(dash_capital), top_n=int(dash_topn), freq="monthly",
                        affordable=True,
                        industry_map=get_industry_map(tech) if strat.get("industry_cap") else None,
                        industry_cap=strat.get("industry_cap"),
                    )
                navs[sname] = res["nav"]
                if bench_line is None:
                    bench_line = res["bench"]
                m = res["metrics"]
                rows.append({
                    "策略": sname, "总收益%": round(m["总收益"] * 100, 2),
                    "年化%": round(m["年化收益"] * 100, 2),
                    "夏普": round(m["夏普"], 3),
                    "最大回撤%": round(m["最大回撤"] * 100, 2),
                    "信号日": str(res["last_signal_date"].date()),
                })

            st.plotly_chart(equity_compare_chart(navs, float(dash_capital), bench=bench_line),
                            use_container_width=True)
            st.markdown("#### 策略指标对比")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("请至少选择一个策略")

    # ---------------- 回测工作台 ----------------
    with tab_bt:
        st.subheader("回测工作台")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            universe = st.selectbox("股票池", ["科技行业", "沪深300+500"], key="bt_universe")
        with c2:
            strategy = st.selectbox("策略", list_strategies(), key="bt_strategy")
        with c3:
            top_n = st.select_slider("TopN", options=[1, 2, 3, 5, 8, 10], value=5,
                                     key="bt_topn")
        with c4:
            freq_ui = st.selectbox("调仓频率", ["月频", "周频"], key="bt_freq")

        c5, c6, c7 = st.columns(3)
        with c5:
            capital = st.number_input("初始资金", value=50000.0, min_value=1000.0,
                                      step=1000.0, key="bt_capital")
        with c6:
            exclude_kc = st.checkbox("剔除科创/创业板(300/301/688/689)", value=True,
                                     key="bt_exclude")
        with c7:
            affordable = st.checkbox("只买得起一手(小资金过滤)", value=True,
                                     key="bt_affordable")

        d1, d2 = st.columns(2)
        with d1:
            start = st.date_input("开始日期", value=(last_date - pd.DateOffset(months=6)).date(),
                                  min_value=first_date.date(), max_value=last_date.date(),
                                  key="bt_start")
        with d2:
            end = st.date_input("结束日期", value=last_date.date(),
                                min_value=first_date.date(), max_value=last_date.date(),
                                key="bt_end")

        if st.button("🚀 跑回测", type="primary"):
            if start >= end:
                st.error("开始日期必须早于结束日期")
            else:
                strat = get_strategy(strategy)
                codes = build_codes(universe, exclude_kc, panel, uni, tech)
                freq = "weekly" if freq_ui == "周频" else "monthly"
                with st.spinner("计算中..."):
                    res = run_backtest(
                        panel=panel, codes=codes, factor=strat["factor"],
                        ascending=strat["ascending"], start=str(start), end=str(end),
                        capital=float(capital), top_n=int(top_n), freq=freq,
                        affordable=affordable,
                        industry_map=get_industry_map(tech) if strat.get("industry_cap") else None,
                        industry_cap=strat.get("industry_cap"),
                    )
                st.session_state["bt_result"] = res
                st.session_state["bt_desc"] = (
                    f"{universe} · {strategy} · Top{top_n} · {freq} · "
                    f"{start} ~ {end} · 资金 {capital:,.0f}"
                )

        if "bt_result" in st.session_state:
            res = st.session_state["bt_result"]
            st.markdown(f"**{st.session_state['bt_desc']}**")
            st.markdown("##### 策略指标")
            render_metrics(st.columns(6), res["metrics"])
            st.markdown("##### 基准指标（等权股票池）")
            render_metrics(st.columns(6), res["bench_metrics"])
            st.plotly_chart(equity_chart(res["nav"], res["bench"], float(capital)),
                            use_container_width=True)
            st.plotly_chart(drawdown_chart(res["drawdown"]), use_container_width=True)

            tab_h, tab_t = st.tabs(["持仓明细", "调仓记录"])
            with tab_h:
                if res["holdings"].empty:
                    st.info("当前空仓")
                else:
                    nm = get_name_map(uni, tech)
                    h = res["holdings"].copy()
                    h["名称"] = [nm.get(str(c), "") for c in h["code"]]
                    st.dataframe(
                        h[["code", "名称", "weight_pct", "price", "market_value"]]
                        .rename(columns={"code": "代码", "weight_pct": "权重%",
                                         "price": "价格", "market_value": "市值"}),
                        use_container_width=True, hide_index=True)
            with tab_t:
                if res["trades"].empty:
                    st.info("无调仓记录")
                else:
                    st.dataframe(res["trades"], use_container_width=True, hide_index=True)

    # ---------------- 今日信号 ----------------
    with tab_sig:
        st.subheader("今日信号")
        s1, s2 = st.columns(2)
        with s1:
            sig_universe = st.selectbox("股票池", ["科技行业", "沪深300+500"], key="sig_universe")
        with s2:
            sig_strategy = st.selectbox("策略", list_strategies(), index=0, key="sig_strategy")
        sig_n = st.slider("显示条数", 5, 30, 15, key="sig_n")
        strat = get_strategy(sig_strategy)
        codes = build_codes(sig_universe, True, panel, uni, tech)
        sig, sig_date = latest_signals(panel, codes, strat["factor"],
                                       strat["ascending"], top_n=int(sig_n))
        st.caption(f"信号日：{sig_date.date()} · 因子：{strat['factor']} · "
                   f"排序：{'升序(买低)' if strat['ascending'] else '降序(买高)'}")
        nm = get_name_map(uni, tech)
        sig2 = pd.DataFrame({
            "code": sig["code"],
            "名称": [nm.get(str(c), "") for c in sig["code"]],
            "score": sig["score"],
            "close": sig["close"],
            "turnover": sig["turnover"],
        })
        st.dataframe(sig2.rename(columns={"code": "代码", "score": "因子得分",
                                          "close": "收盘价", "turnover": "换手率"}),
                     use_container_width=True, hide_index=True)

    # ---------------- 舆情情绪 ----------------
    with tab_news:
        st.subheader("📰 舆情情绪看板")
        st.caption("数据：东方财富个股新闻 + 财联社电报 | 打分：中文金融词典 | "
                   "研究：事件研究（沪深300超额）")
        stats = load_news_stats()
        if not stats or stats.get("total", 0) == 0:
            st.warning("暂无舆情数据。先运行：cd ~/quant/sentiment-mvp && "
                       "python run_pipeline.py daily")
        else:
            st.markdown("## 数据概览")
            n1, n2, n3, n4 = st.columns(4)
            n1.metric("新闻总数", f"{stats['total']:,}")
            n2.metric("数据起始", (stats.get("min_time") or "-")[:10])
            n3.metric("数据截止", (stats.get("max_time") or "-")[:10])
            src = stats.get("by_source", {})
            n4.metric("来源", f"EM {src.get('em', 0)} / CLS {src.get('cls', 0)}")

            labels = stats.get("by_label", {})
            l1, l2, l3 = st.columns(3)
            l1.metric("正面", labels.get("positive", 0))
            l2.metric("中性", labels.get("neutral", 0))
            l3.metric("负面", labels.get("negative", 0))

            arts = load_news_articles()
            if len(arts):
                arts = arts.copy()
                arts["publish_time"] = arts["publish_time"].astype(str)

                st.markdown("## 今日快照（财联社）")
                today = pd.Timestamp.now().strftime("%Y-%m-%d")
                today_df = arts[arts["publish_time"].str.startswith(today)].copy()
                if len(today_df):
                    uni_n = load_news_universe()
                    nm = dict(zip(uni_n["code"], uni_n["name"])) if len(uni_n) else {}
                    today_df["name"] = today_df["code"].map(nm)
                    snap = (today_df.groupby(["code", "name"])
                            .agg(n=("score", "size"), mean_score=("score", "mean"),
                                 positive=("label", lambda s: int((s == "positive").sum())),
                                 negative=("label", lambda s: int((s == "negative").sum())))
                            .sort_values("mean_score", ascending=False).reset_index())
                    st.dataframe(snap, use_container_width=True, hide_index=True)
                else:
                    st.info("今日暂无匹配标的的财联社电报")

                st.markdown("## 情绪分布")
                d1, d2 = st.columns(2)
                with d1:
                    fig = px.histogram(arts, x="score", nbins=40, title="情绪分直方图",
                                       color_discrete_sequence=["#5470c6"])
                    fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                with d2:
                    lab = arts.groupby(["source", "label"]).size().reset_index(name="n")
                    fig2 = px.bar(lab, x="label", y="n", color="source", barmode="group",
                                  title="标签 × 来源",
                                  color_discrete_sequence=["#5470c6", "#ee6666"])
                    fig2.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig2, use_container_width=True)

                st.markdown("## 事件研究（沪深300超额，次日开盘买入）")
                study = load_news_study()
                if len(study):
                    col = [c for c in study.columns if c.startswith("excess_open_")]
                    if col:
                        rows = []
                        for c in col:
                            h = c.split("_")[-1]
                            for lab in ("positive", "neutral", "negative"):
                                sub = study[study["label"] == lab][c].dropna()
                                if len(sub):
                                    rows.append({"持有期": f"{h}日", "情绪桶": lab,
                                                 "事件数": len(sub), "平均超额": sub.mean(),
                                                 "胜率": (sub > 0).mean()})
                        summ = pd.DataFrame(rows)
                        t1, t2 = st.columns([1, 1.6])
                        with t1:
                            pivot = (summ.pivot(index="情绪桶", columns="持有期",
                                                values="平均超额")
                                     .reindex(["positive", "neutral", "negative"]))
                            pivot.index = ["正面", "中性", "负面"]
                            st.dataframe(pivot.map(lambda x: f"{x:.4f}"),
                                         use_container_width=True)
                        with t2:
                            fig3 = px.bar(summ, x="持有期", y="平均超额", color="情绪桶",
                                          barmode="group",
                                          color_discrete_map={"positive": "#e8503a",
                                                              "neutral": "#aaa",
                                                              "negative": "#3b7dd8"})
                            fig3.add_hline(y=0, line_color="#666", line_dash="dash")
                            fig3.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10),
                                               legend_title="情绪桶")
                            st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.info("暂无事件研究结果，运行 daily 后生成")

                arts["date"] = pd.to_datetime(arts["publish_time"], errors="coerce").dt.date
                trend = arts.dropna(subset=["date"]).groupby("date").size().reset_index(name="n")
                if len(trend):
                    fig4 = px.bar(trend, x="date", y="n", title="每日新闻条数",
                                  color_discrete_sequence=["#91cc75"])
                    fig4.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig4, use_container_width=True)

                st.markdown("## 情绪最强 / 最弱新闻")
                uni_n = load_news_universe()
                nm = dict(zip(uni_n["code"], uni_n["name"])) if len(uni_n) else {}
                arts["name"] = arts["code"].map(nm)
                top = arts.nlargest(10, "score")[["date", "code", "name", "title",
                                                  "label", "score", "source"]]
                bot = arts.nsmallest(10, "score")[["date", "code", "name", "title",
                                                   "label", "score", "source"]]
                e1, e2 = st.columns(2)
                with e1:
                    st.markdown("**最正面**")
                    st.dataframe(top, use_container_width=True, hide_index=True)
                with e2:
                    st.markdown("**最负面**")
                    st.dataframe(bot, use_container_width=True, hide_index=True)
            else:
                st.info("暂无文章数据")

    # ---------------- 数据状态 ----------------
    with tab_data:
        st.subheader("数据状态")
        status = data_status()
        rows = []
        for key, src in status.items():
            if key == "meta":
                continue
            for label, s in src.items():
                rows.append({"数据": key, "位置": label,
                             "存在": "✅" if s["exists"] else "❌",
                             "大小MB": s["size_mb"]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        meta = status.get("meta", {})
        if meta:
            st.markdown(f"- 上次更新：{meta.get('last_update', '-')} · "
                        f"代码数 {meta.get('n_codes', '-')} · 行数 {meta.get('n_rows', '-'):,}")
        st.markdown(f"- 当前加载：面板行数 {len(panel):,}，个股 {panel['code'].nunique()}，"
                    f"日期 {first_date.date()} ~ {last_date.date()}")

        st.markdown("---")
        st.markdown("### 数据更新（腾讯行情 + 中证指数官网）")
        u1, u2 = st.columns(2)
        with u1:
            mode = st.selectbox("更新模式", ["增量（推荐，只抓新增区间）", "全量重建"],
                                key="data_mode")
        with u2:
            update_end = st.date_input("更新到", value=pd.Timestamp.today().date(),
                                       key="data_end")
        st.caption("首次全量约需几分钟；之后每天增量约 1-2 分钟。东方财富行业接口在部分"
                   "服务器不可达，行业分类失败时自动沿用本地缓存。")
        if st.button("🚀 开始更新", type="primary", key="data_update"):
            bar = st.progress(0.0, text="准备中...")
            try:
                result = update_data(
                    mode="incremental" if mode.startswith("增量") else "full",
                    end=update_end.strftime("%Y-%m-%d"),
                    progress=lambda p, t, label: bar.progress(min(float(p) / max(float(t), 1.0), 1.0),
                                                              text=label),
                )
                bar.progress(1.0, text="完成")
                load_data.clear()
                st.success(f"更新完成：{result['n_codes']} 只，{result['n_rows']:,} 行。刷新页面生效。")
            except Exception as exc:
                st.error(f"更新失败：{exc}")


if __name__ == "__main__":
    main()
