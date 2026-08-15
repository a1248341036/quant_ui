from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

SENT_ROOT = Path("/home/ubuntu/sentiment-mvp")
sys.path.insert(0, str(SENT_ROOT))

from lib import store as stk  # noqa: E402

st.set_page_config(page_title="舆情情绪看板", page_icon="📰", layout="wide")

# 与主应用一致的登录门
if not st.session_state.get("authed"):
    st.title("🔐 量化回测工作台")
    st.caption("仅限授权用户访问")
    pw = st.text_input("密码", type="password")
    if st.button("登录"):
        if pw == os.environ.get("QUANT_UI_PASSWORD", "REDACTED_PASSWORD"):
            st.session_state.authed = True
            st.rerun()
        else:
            st.error("密码错误")
    st.stop()

cfg = yaml.safe_load((SENT_ROOT / "config.yaml").read_text(encoding="utf-8"))
DB = SENT_ROOT / cfg["paths"]["articles_db"]
STUDY_CSV = SENT_ROOT / "outputs/event_study_daily.csv"
UNIVERSE_CSV = SENT_ROOT / cfg["universe"]["output"]


@st.cache_data(ttl=300, show_spinner=False)
def load_stats():
    return stk.stats(DB)


@st.cache_data(ttl=300, show_spinner=False)
def load_articles():
    return stk.load_articles(DB)


@st.cache_data(ttl=300, show_spinner=False)
def load_study():
    if not STUDY_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(STUDY_CSV)


@st.cache_data(ttl=3600, show_spinner=False)
def load_universe():
    if not UNIVERSE_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(UNIVERSE_CSV, dtype={"code": str})
    df["code"] = df["code"].astype(str).str.zfill(6)
    return df


st.title("📰 舆情情绪看板")
st.caption("数据：东方财富个股新闻 + 财联社电报 | 打分：中文金融词典 | 研究：事件研究（沪深300超额）")

stats = load_stats()
if not stats or stats["total"] == 0:
    st.warning("暂无舆情数据。先运行：cd ~/sentiment-mvp && python run_pipeline.py daily")
    st.stop()

# ---------- 概览 ----------
st.markdown("## 数据概览")
c1, c2, c3, c4 = st.columns(4)
c1.metric("新闻总数", f"{stats['total']:,}")
c2.metric("数据起始", (stats["min_time"] or "-")[:10])
c3.metric("数据截止", (stats["max_time"] or "-")[:10])
src = stats.get("by_source", {})
c4.metric("来源", "EM " + str(src.get("em", 0)) + " / CLS " + str(src.get("cls", 0)))

labels = stats.get("by_label", {})
lc1, lc2, lc3 = st.columns(3)
lc1.metric("正面", labels.get("positive", 0))
lc2.metric("中性", labels.get("neutral", 0))
lc3.metric("负面", labels.get("negative", 0))

# ---------- 今日快照 ----------
st.markdown("## 今日快照（财联社）")
arts = load_articles()
if len(arts):
    arts = arts.copy()
    arts["publish_time"] = arts["publish_time"].astype(str)
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    today_df = arts[arts["publish_time"].str.startswith(today)].copy()
    if len(today_df):
        uni = load_universe()
        name_map = dict(zip(uni["code"], uni["name"])) if len(uni) else {}
        today_df["name"] = today_df["code"].map(name_map)
        snap = (
            today_df.groupby(["code", "name"])
            .agg(n=("score", "size"), mean_score=("score", "mean"),
                 positive=("label", lambda s: int((s == "positive").sum())),
                 negative=("label", lambda s: int((s == "negative").sum())))
            .sort_values("mean_score", ascending=False)
            .reset_index()
        )
        st.dataframe(snap, use_container_width=True, hide_index=True)
    else:
        st.info("今日暂无匹配标的的财联社电报")
else:
    st.info("暂无文章数据")

# ---------- 情绪分布 ----------
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
                  title="标签 × 来源", color_discrete_sequence=["#5470c6", "#ee6666"])
    fig2.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig2, use_container_width=True)

# ---------- 事件研究 ----------
st.markdown("## 事件研究（沪深300超额，次日开盘买入）")
study = load_study()
if len(study):
    col = [c for c in study.columns if c.startswith("excess_open_")]
    if col:
        rows = []
        for c in col:
            h = c.split("_")[-1]
            for lab in ("positive", "neutral", "negative"):
                sub = study[study["label"] == lab][c].dropna()
                if len(sub):
                    rows.append({"持有期": f"{h}日", "情绪桶": lab, "事件数": len(sub),
                                 "平均超额": sub.mean(), "胜率": (sub > 0).mean()})
        summ = pd.DataFrame(rows)
        t1, t2 = st.columns([1, 1.6])
        with t1:
            pivot = summ.pivot(index="情绪桶", columns="持有期", values="平均超额").reindex(
                ["positive", "neutral", "negative"])
            pivot.index = ["正面", "中性", "负面"]
            st.dataframe(pivot.map(lambda x: f"{x:.4f}"), use_container_width=True)
        with t2:
            fig3 = px.bar(summ, x="持有期", y="平均超额", color="情绪桶", barmode="group",
                          color_discrete_map={"positive": "#e8503a", "neutral": "#aaa", "negative": "#3b7dd8"})
            fig3.add_hline(y=0, line_color="#666", line_dash="dash")
            fig3.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), legend_title="情绪桶")
            st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("暂无事件研究结果，运行 daily 后生成")

# ---------- 每日舆情量 ----------
st.markdown("## 每日舆情量")
arts["date"] = pd.to_datetime(arts["publish_time"], errors="coerce").dt.date
trend = arts.dropna(subset=["date"]).groupby("date").size().reset_index(name="n")
if len(trend):
    fig4 = px.bar(trend, x="date", y="n", title="每日新闻条数",
                  color_discrete_sequence=["#91cc75"])
    fig4.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig4, use_container_width=True)

# ---------- 极端事件 ----------
st.markdown("## 情绪最强 / 最弱新闻")
uni = load_universe()
name_map = dict(zip(uni["code"], uni["name"])) if len(uni) else {}
arts["name"] = arts["code"].map(name_map)
top = arts.nlargest(10, "score")[["date", "code", "name", "title", "label", "score", "source"]]
bot = arts.nsmallest(10, "score")[["date", "code", "name", "title", "label", "score", "source"]]
e1, e2 = st.columns(2)
with e1:
    st.markdown("**最正面**")
    st.dataframe(top, use_container_width=True, hide_index=True)
with e2:
    st.markdown("**最负面**")
    st.dataframe(bot, use_container_width=True, hide_index=True)

st.caption("看板数据缓存 5 分钟；每日 21:30 由 cron 自动执行 daily 增量入库。")
