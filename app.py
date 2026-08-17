from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

from core.data import (data_status, load_etf, load_fund, load_index,  # noqa: E402
                       load_panel, load_tech, load_universe)
from core.composites import (FACTOR_OPTIONS, delete_composite, load_composites,
                             save_composite)
from core.engine import latest_signals, run_backtest
from core.updater import refresh_all
from core.metrics import compute_metrics
from core.store import normalize_universe
from core import strategy_pool as sp
from strategies.registry import STRATEGIES, get_strategy, list_strategies
from backend.routers import code as lab_api


st.set_page_config(page_title="A股量化回测工作台", page_icon="📈", layout="wide")


ACCENT = "#4f8cff"
POS_COLOR = "#ff5d4d"
NEG_COLOR = "#2bb98a"
MUTED = "#8494b5"


def inject_theme() -> None:
    """全局暗色主题微调：metric 卡片、tabs、按钮、表格、间距。"""
    st.markdown("""
<style>
    /* 整体 */
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { right: 1rem; }
    [data-testid="stFooter"] { display: none; }
    .block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1440px; }

    /* 侧边栏 */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgb(79 140 255 / .07), transparent 260px), #0c1322;
        border-right: 1px solid #1f2c45;
    }
    [data-testid="stSidebar"] .block-container { padding-top: 0 !important; padding-bottom: 0 !important; }
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding: .8rem .9rem 1rem !important; }
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] > div:first-child { display: none; }
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] > div:last-child { padding-top: 12px !important; padding-bottom: 12px !important; }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 4px !important; }
    [data-testid="stSidebar"] .stMarkdown p { color: #a8b6d3; }
    [data-testid="stSidebar"] hr { display: none; }

    /* 品牌区 */
    .side-brand { display: flex; align-items: center; gap: 10px; padding: 2px 2px 16px; }
    .side-brand-mark {
        width: 32px; height: 32px; flex: none; display: grid; place-items: center;
        border-radius: 10px;
        background: linear-gradient(145deg, rgb(79 140 255 / .3), rgb(79 140 255 / .07));
        border: 1px solid rgb(79 140 255 / .35);
        box-shadow: 0 6px 18px rgb(79 140 255 / .2);
    }
    .side-brand-mark svg { width: 17px; height: 17px; color: #6aa5ff; }
    .side-brand-title { font-size: 15px; font-weight: 700; color: #e6ecf7; line-height: 1.3; }
    .side-brand-sub { font-size: 12px; color: #a8b8d8; line-height: 1.4; }

    /* 侧边栏导航按钮组 */
    [data-testid="stSidebar"] .stButton { margin-bottom: 3px; }
    [data-testid="stSidebar"] .stButton > button {
        justify-content: flex-start; text-align: left;
        background: transparent; border: 1px solid transparent;
        color: #c3cfe6; font-weight: 500; font-size: 13.5px;
        padding: 7px 12px; border-radius: 10px; min-height: 34px;
        box-shadow: none;
        transition: background-color .15s, color .15s, border-color .15s;
    }
    [data-testid="stSidebar"] .stButton > button:hover:not(:disabled) {
        background: rgb(255 255 255 / .05); color: #e6ecf7; border-color: #2b3d61;
    }
    [data-testid="stSidebar"] .stButton > button:disabled {
        opacity: 1;
        background: linear-gradient(90deg, rgb(79 140 255 / .22), rgb(79 140 255 / .05));
        border: 1px solid rgb(79 140 255 / .28);
        border-left: 4px solid #6aa5ff;
        color: #fff; font-weight: 600;
    }

    /* 信息区 */
    .side-meta {
        margin-top: 10px;
        padding: 10px 12px;
        background: #0d1424; border: 1px solid #1f2c45; border-radius: 10px;
        font-size: 12px;
    }
    .side-meta-row { display: flex; justify-content: space-between; gap: 10px; padding: 2px 0; }
    .side-meta-row span:first-child { color: #8fa0c2; }
    .side-meta-row span:last-child { color: #c6d2ea; font-variant-numeric: tabular-nums; }
    .side-note { font-size: 12px; color: #b9c6e2; line-height: 1.8; padding: 8px 2px 0; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #1f2c45; }
    .stTabs [data-baseweb="tab"] {
        background: transparent; color: #8494b5;
        border-radius: 8px; padding: 6px 14px; font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: rgb(79 140 255 / .14); color: #e6ecf7 !important;
        box-shadow: 0 0 0 1px rgb(79 140 255 / .3) inset;
    }

    /* Metric 卡片 */
    [data-testid="stMetric"] {
        background: linear-gradient(180deg, rgb(255 255 255 / .02), transparent), #111a2e;
        border: 1px solid #1f2c45; border-radius: 12px; padding: 14px 16px;
        box-shadow: 0 1px 0 rgb(255 255 255 / .03) inset, 0 12px 30px rgb(2 6 14 / .3);
    }
    [data-testid="stMetricLabel"] { color: #8494b5; font-size: .85rem; }
    [data-testid="stMetricValue"] { color: #e6ecf7; font-variant-numeric: tabular-nums; }
    [data-testid="stMetricDelta"] [data-testid="stMetricDeltaPositive"] { color: #2bb98a; }
    [data-testid="stMetricDelta"] [data-testid="stMetricDeltaNegative"] { color: #ff5d4d; }

    /* 按钮 */
    .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
        border-radius: 9px; border: 1px solid #2b3d61;
        font-weight: 500; transition: border-color .15s, box-shadow .15s, transform .1s;
    }
    .stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {
        border-color: #4f8cff; box-shadow: 0 0 0 3px rgb(79 140 255 / .15);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(145deg, #4f8cff, #3b6fe0);
        border: none; box-shadow: 0 6px 18px rgb(79 140 255 / .28);
    }
    .stButton > button[kind="primary"]:hover { filter: brightness(1.08); }

    /* 输入控件 */
    [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
    [data-baseweb="select"] > div, [data-testid="stTextArea"] textarea,
    [data-testid="stDateInput"] input {
        border-radius: 9px; background: #0d1424; border-color: #1f2c45;
    }
    [data-baseweb="select"] > div:focus-within,
    [data-testid="stTextInput"] input:focus, [data-testid="stNumberInput"] input:focus {
        border-color: #4f8cff; box-shadow: 0 0 0 3px rgb(79 140 255 / .15);
    }

    /* 数据表格 */
    [data-testid="stDataFrame"] {
        border: 1px solid #1f2c45; border-radius: 12px; overflow: hidden;
        box-shadow: 0 1px 0 rgb(255 255 255 / .03) inset;
    }

    /* 卡片容器 */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(180deg, rgb(255 255 255 / .018), transparent 90px), #111a2e;
        border: 1px solid #1f2c45; border-radius: 14px; padding: .25rem .25rem;
        box-shadow: 0 1px 0 rgb(255 255 255 / .03) inset, 0 14px 36px rgb(2 6 14 / .3);
    }

    /* 标题层级 */
    h1, h2, h3 { letter-spacing: .01em; }
    .stMarkdown h3 { font-size: 1.05rem; font-weight: 650; }

    code { color: #6aa5ff !important; background: #16233d !important; }
</style>
""", unsafe_allow_html=True)


def style_chart(fig, height: int = 360, title: str | None = None):
    """统一 plotly 图表：透明底、暗色网格、水平图例。"""
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, size=12),
        legend=dict(orientation="h", y=1.1, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#16233d", bordercolor="#2b3d61",
                        font=dict(color="#e6ecf7")),
    )
    if title:
        fig.update_layout(title=dict(text=title, x=0, font=dict(size=14, color="#e6ecf7")))
    fig.update_xaxes(gridcolor="#1f2c45", zerolinecolor="#2b3d61")
    fig.update_yaxes(gridcolor="#1f2c45", zerolinecolor="#2b3d61")
    return fig


def show_table(df, height: int | None = None, **kwargs):
    """统一表格：隐藏索引、等宽数字列。"""
    return st.dataframe(df, use_container_width=True, hide_index=True,
                        height=height, **kwargs)


NAV_ITEMS = [
    ("dashboard", "资金看板"),
    ("pool", "策略池"),
    ("backtest", "回测工作台"),
    ("composite", "组合策略"),
    ("code", "代码"),
    ("sweep", "参数稳健性"),
    ("paper", "模拟盘"),
    ("history", "历史回测"),
    ("signals", "今日信号"),
    ("news", "舆情情绪"),
    ("data", "数据状态"),
]


def _history_title(kind: str, p: dict, s: dict) -> str:
    """历史列表里的人类可读标题。"""
    if kind == "compare":
        names = "、".join(str(x) for x in (p.get("strategies") or [])[:4])
        return f"对比 {names}"
    if kind in ("sweep", "sweep_cli"):
        mode = p.get("mode", "")
        if mode == "factor":
            return f"扫描[factor] {p.get('strategy', '')}"
        sl = p.get("short_list") or []
        ll = p.get("long_list") or []
        return f"扫描[event] short×long {len(sl)}×{len(ll)}"
    if s.get("composite"):
        return f"组合[{s.get('composite_name') or '自定义'}] {p.get('universe', '')}"
    return f"{p.get('strategy', '')} · {p.get('universe', '')}"


def _render_history_detail(rec: dict) -> None:
    """历史回测详情：参数/指标/净值/交易。"""
    st.caption(
        f"run_id={rec.get('run_id')} · created_at={rec.get('created_at')} "
        f"· kind={rec.get('kind')} · data_version={rec.get('data_version')}")
    tab_p, tab_m, tab_n, tab_t = st.tabs(["参数", "指标", "净值", "交易"])
    with tab_p:
        params = rec.get("params") or {}
        if params:
            show_table(pd.DataFrame(
                [{"key": k, "value": "" if v is None else str(v)}
                 for k, v in params.items()]))
    with tab_m:
        metrics = rec.get("metrics") or {}
        if metrics:
            mdf = pd.DataFrame([{"指标": k, "策略": v} for k, v in metrics.items()])
            bm = rec.get("bench_metrics") or {}
            if bm:
                mdf["基准"] = [bm.get(k) for k in metrics.keys()]
            show_table(mdf)
        else:
            st.info("无单策略指标（对比/扫描记录）")
    with tab_n:
        nav = rec.get("nav")
        if not nav or not isinstance(nav, list):
            st.info("无净值")
        elif "points" in nav[0]:
            fig = go.Figure()
            for item in nav:
                pts = item.get("points") or []
                fig.add_trace(go.Scatter(
                    x=[p["date"] for p in pts],
                    y=[p["value"] for p in pts],
                    name=str(item.get("name", "")), mode="lines"))
            fig.update_layout(margin=dict(l=10, r=10, t=30, b=10),
                              height=380)
            st.plotly_chart(fig, use_container_width=True)
        elif "value" in nav[0]:
            fig = go.Figure(go.Scatter(
                x=[p["date"] for p in nav], y=[p["value"] for p in nav],
                name="净值", mode="lines"))
            fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=380)
            st.plotly_chart(fig, use_container_width=True)
        else:
            show_table(pd.DataFrame(nav))
    with tab_t:
        trades = rec.get("trades")
        if not trades or not isinstance(trades, list):
            st.info("无交易明细")
        elif "records" in trades[0]:
            for item in trades:
                st.markdown(f"**{item.get('name', '')}**")
                show_table(pd.DataFrame(item.get("records") or []))
        else:
            show_table(pd.DataFrame(trades))


def render_sidebar_nav(current: str) -> None:
    """侧边栏导航：全部用 st.button，当前页渲染为 disabled 高亮态，避免切页时布局跳动。"""
    for key, label in NAV_ITEMS:
        if key == current:
            st.button(label, key=f"nav_{key}", disabled=True, use_container_width=True)
        else:
            if st.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state["page"] = key
                st.rerun()


def strategy_groups() -> list[tuple[str, list[str]]]:
    """按策略注册表里的 group 分组，保持注册顺序。"""
    groups: dict[str, list[str]] = {}
    for name in list_strategies():
        groups.setdefault(STRATEGIES[name].get("group", "其他"), []).append(name)
    return list(groups.items())


def grouped_strategy_order() -> list[str]:
    """按分组排好序的策略名列表（selectbox 用）。"""
    return [name for _, names in strategy_groups() for name in names]


def strategy_display(name: str) -> str:
    """下拉显示为「分组 | 名称」，值仍是策略名。"""
    try:
        group = STRATEGIES[name].get("group", "其他")
    except KeyError:
        group = "配置池"
    return f"{group} | {name}"


def strategy_scope_ui(key: str, label: str = "策略来源") -> str:
    """策略来源选择：配置池优先，空时默认全部策略。"""
    default_idx = 0 if sp.pool_names() else 1
    return st.selectbox(label, ["配置池", "全部策略"], index=default_idx, key=key)


def strategy_options(scope: str) -> list[str]:
    """回测/模拟盘/信号下拉选项，随配置池同步。"""
    return sp.pool_strategy_options(scope)


def resolve_strategy(name: str) -> dict:
    """策略定义：注册表优先，其次配置池（支持归档策略名）。"""
    return sp.resolve_strategy(name)


def _fmt_metric_series(series: pd.Series) -> pd.Series:
    return series.map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")


def render_pool_page() -> None:
    """策略池：全量池 + 配置池（含回收站）。"""
    st.markdown("### 策略池")
    st.caption("全量池 = 代码注册策略 + 回测归档里跑过的全部策略；"
               "配置池 = 你精选的策略，回测 / 模拟盘 / 今日信号的下拉框会以配置池优先。"
               "配置池移除的策略先进回收站，回收站里删除才是彻底删除。")

    left, right = st.columns([2, 1])
    with left:
        st.markdown("#### 配置池")
        rows = sp.pool_rows()
        if rows.empty:
            st.info("配置池为空，从下方全量池选择策略加入。")
        else:
            show = rows.copy()
            show["ascending"] = show["ascending"].map(
                {True: "买低", False: "买高"}).fillna("-")
            show["sharpe"] = _fmt_metric_series(show["sharpe"])
            show["annual"] = _fmt_metric_series(show["annual"])
            show["mdd"] = _fmt_metric_series(show["mdd"])
            show = show.rename(columns={
                "name": "策略", "group": "分组", "factor": "因子",
                "source": "来源", "n_runs": "回测次数",
                "sharpe": "夏普", "annual": "年化", "mdd": "最大回撤",
                "ascending": "方向",
            })
            st.dataframe(show[["策略", "分组", "因子", "方向", "来源",
                               "回测次数", "夏普", "年化", "最大回撤"]],
                         use_container_width=True, hide_index=True,
                         height=min(320, 38 * len(show) + 38))
            rm_sel = st.multiselect(
                "移除策略（进回收站）", rows["name"].tolist(),
                key="pool_rm_sel")
            if st.button("移除所选", key="pool_rm_btn", disabled=not rm_sel):
                for n in rm_sel:
                    sp.remove_from_pool(n)
                st.success(f"已移除 {len(rm_sel)} 个到回收站")
                st.rerun()
    with right:
        st.markdown("#### 回收站")
        tr = sp.trash_rows()
        if tr.empty or "name" not in tr.columns:
            st.caption("回收站为空")
        else:
            for _, r in tr.iterrows():
                n = r["name"]
                c1, c2 = st.columns([3, 1])
                c1.caption(n)
                with c2:
                    if st.button("恢复", key=f"tr_restore_{n}"):
                        sp.restore_from_trash(n)
                        st.rerun()
            tr_purge = st.multiselect(
                "彻底删除（不可恢复）", tr["name"].tolist(), key="pool_tr_purge_sel")
            if st.button("彻底删除所选", key="pool_tr_purge_btn",
                         disabled=not tr_purge):
                for n in tr_purge:
                    sp.purge_from_trash(n)
                st.success(f"已彻底删除 {len(tr_purge)} 个")
                st.rerun()
            if st.button("清空回收站", key="pool_tr_empty"):
                st.warning(f"已清空 {sp.empty_trash()} 个")
                st.rerun()

    st.divider()
    st.markdown("#### 全量池")
    fp = sp.full_pool()
    if fp.empty:
        st.info("暂无策略")
        return
    cq, cg = st.columns([2, 1])
    with cq:
        q = st.text_input("搜索策略名称/说明", key="pool_q")
    with cg:
        grps = ["全部"] + sorted(fp["group"].dropna().unique().tolist())
        grp = st.selectbox("分组", grps, key="pool_grp")
    filtered = fp
    if q:
        mask = filtered["name"].str.contains(q, case=False, na=False) | \
            filtered["desc"].fillna("").str.contains(q, case=False, na=False)
        filtered = filtered[mask]
    if grp != "全部":
        filtered = filtered[filtered["group"] == grp]

    pooled = set(sp.pool_names())
    show = filtered.copy()
    show["状态"] = show["name"].map(lambda n: "已配置" if n in pooled else "未配置")
    show["方向"] = show["ascending"].map({True: "买低", False: "买高"}).fillna("-")
    show["夏普"] = _fmt_metric_series(show["sharpe"])
    show["年化"] = _fmt_metric_series(show["annual"])
    show["最大回撤"] = _fmt_metric_series(show["mdd"])
    show = show.rename(columns={
        "name": "策略", "group": "分组", "factor": "因子", "desc": "说明",
        "source": "来源", "n_runs": "回测次数", "universe": "股票池",
    })
    st.dataframe(show[["策略", "分组", "因子", "方向", "来源", "股票池",
                       "回测次数", "夏普", "年化", "最大回撤", "说明"]],
                 use_container_width=True, hide_index=True)
    cand = filtered[~filtered["name"].isin(pooled)]["name"].tolist()
    add_sel = st.multiselect("从全量池选择要加入配置池的策略", cand,
                             key="pool_add_sel")
    if st.button(f"加入配置池（{len(add_sel)}）", type="primary",
                 key="pool_add_btn", disabled=not add_sel):
        ok = sum(1 for n in add_sel if sp.add_from_full(n))
        st.success(f"已加入 {ok} 个策略到配置池（回收站同名自动恢复）")
        st.rerun()


def _collect_composite() -> tuple[dict[str, float], dict[str, bool]]:
    """从 session_state 收集勾选因子的权重与方向（买低=True）。"""
    weights: dict[str, float] = {}
    directions: dict[str, bool] = {}
    for f in FACTOR_OPTIONS:
        name = f["name"]
        if st.session_state.get(f"comp_en_{name}"):
            weights[name] = float(st.session_state.get(f"comp_w_{name}", 1.0))
            directions[name] = st.session_state.get(f"comp_dir_{name}", "买高") == "买低"
    return weights, directions


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

SENT_DB = SENT_ROOT / "data" / "articles.db"
SENT_STUDY_CSV = SENT_ROOT / "outputs" / "event_study_daily.csv"
SENT_UNIVERSE_CSV = SENT_ROOT / "data" / "universe.csv"


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
    # ETF/场外基金代码与股票池不重叠（不在股票 panel 中），后写覆盖保证
    # ETF/基金回测与信号页显示正确的产品名。
    for loader in (load_etf, load_fund):
        try:
            df = loader()
        except Exception:
            continue
        for code, name in zip(df["code"], df["name"]):
            code = str(code).zfill(6)
            if name and not pd.isna(name):
                m[code] = str(name)
    return m


def build_codes(universe: str, exclude_kechuang: bool, panel, uni, tech) -> list[str]:
    if normalize_universe(universe) == "科技TMT":
        codes = set(tech["code"])
    elif normalize_universe(universe) == "ETF":
        from core.data import load_etf, load_etf_panel
        etf_panel = load_etf_panel()
        if len(etf_panel) == 0:
            return []
        return sorted(set(load_etf()["code"]) & set(etf_panel["code"].unique()))
    elif normalize_universe(universe) == "场外科技基金":
        from core.data import load_fund, load_fund_nav
        fund_nav = load_fund_nav()
        if len(fund_nav) == 0:
            return []
        return sorted(set(load_fund()["code"]) & set(fund_nav["code"].unique()))
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


def fmt_num(x, digits: int = 2) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{x:.{digits}f}"


def index_series_map(index_df: pd.DataFrame, start, end) -> dict[str, pd.Series]:
    """把长表指数数据切成 {名称: 归一化收盘价序列}，起点为窗口内第一个交易日。"""
    out: dict[str, pd.Series] = {}
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if index_df is None or index_df.empty:
        return out
    for name, g in index_df.groupby("name"):
        s = g.set_index("date")["close"].sort_index()
        s = s[(s.index >= start_ts) & (s.index <= end_ts)]
        if len(s) > 1:
            out[str(name)] = s / s.iloc[0]
    return out


def _bench_trace(name: str, s: pd.Series, capital: float):
    return go.Scatter(x=s.index, y=s.values * capital, name=name, mode="lines",
                      line=dict(width=1.5, dash="dash", color=MUTED))


def equity_compare_chart(navs: dict[str, pd.Series], capital: float,
                         benches: dict[str, pd.Series] | None = None) -> go.Figure:
    fig = go.Figure()
    if benches:
        for name, s in benches.items():
            fig.add_trace(_bench_trace(name, s, capital))
    for name, nav in navs.items():
        fig.add_trace(go.Scatter(x=nav.index, y=nav.values * capital, name=name,
                                 mode="lines", line=dict(width=2)))
    style_chart(fig, height=400, title="策略资金对比")
    fig.update_yaxes(title="资金")
    return fig


def equity_chart(nav: pd.Series, capital: float,
                 benches: dict[str, pd.Series] | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nav.index, y=nav.values * capital,
                             name="策略资金", line=dict(color=POS_COLOR, width=2)))
    if benches:
        for name, s in benches.items():
            fig.add_trace(_bench_trace(name, s, capital))
    style_chart(fig, height=400, title="资金曲线")
    fig.update_yaxes(title="资金")
    return fig


def drawdown_chart(dd: pd.Series) -> go.Figure:
    fig = go.Figure(go.Scatter(x=dd.index, y=dd.values * 100, name="回撤",
                               fill="tozeroy", line=dict(color=POS_COLOR)))
    style_chart(fig, height=220, title="回撤 (%)")
    fig.update_yaxes(title="%")
    return fig


def render_metrics(cols, metrics: dict):
    order = ["总收益", "年化收益", "夏普", "最大回撤", "卡玛", "胜率"]
    for col, key in zip(cols, order):
        col.metric(key, format_pct(metrics.get(key)))


def benchmark_rows(benches: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, s in benches.items():
        m = compute_metrics(s)
        rows.append({
            "基准": name,
            "总收益%": round(m["总收益"] * 100, 2),
            "年化%": round(m["年化收益"] * 100, 2),
            "夏普": round(m["夏普"], 3),
            "最大回撤%": round(m["最大回撤"] * 100, 2),
        })
    return pd.DataFrame(rows)


def points_to_series(points: list[dict]) -> pd.Series:
    if not points:
        return pd.Series(dtype=float)
    return pd.Series({p["date"]: p["value"] for p in points})


def equal_weight_benchmark(panel: pd.DataFrame, codes: list[str],
                           dates: pd.Series) -> pd.Series | None:
    """与回测引擎等价的等权基准净值（open→open 收益，口径见 core.engine）。

    返回 None 表示数据不足/计算失败，调用方降级为不展示基准。
    """
    try:
        if not codes or panel.empty or dates.empty:
            return None
        sub = panel[panel["code"].isin(set(codes))]
        sub = sub[(sub["date"] >= dates.min()) & (sub["date"] <= dates.max())]
        if sub.empty:
            return None
        open_piv = sub.pivot_table(index="date", columns="code", values="open",
                                   observed=False)
        close_piv = sub.pivot_table(index="date", columns="code", values="close",
                                    observed=False)
        ret = open_piv.pct_change()
        elig = open_piv.notna() & close_piv.notna()
        daily = ret.where(elig).mean(axis=1, skipna=True).fillna(0.0)
        bench = (1.0 + daily).cumprod()
        bench = bench.reindex(pd.DatetimeIndex(dates)).ffill()
        bench = bench.dropna()
        if len(bench) < 2 or bench.iloc[0] == 0:
            return None
        return bench / bench.iloc[0]
    except Exception:
        return None


def _lab_tpl_options() -> list[dict]:
    sys_opts = [{"kind": "sys", "name": n, "label": "系统 · " + n}
                for n in list_strategies()]
    lab_opts = [{"kind": "lab", "name": s["name"], "label": "已保存 · " + s["name"]}
                for s in lab_api.list_saved()["items"]]
    return sys_opts + lab_opts


def _lab_on_tpl_change() -> None:
    """下拉切换策略时自动加载对应代码模板到 lab_code。"""
    _label = st.session_state.get("lab_tpl_pick", "")
    _opt = next((o for o in st.session_state.get("lab_tpl_options", [])
                 if o["label"] == _label), None)
    if not _opt:
        return
    _r = (lab_api.get_saved(_opt["name"]) if _opt["kind"] == "lab"
          else lab_api.get_template(_opt["name"]))
    if _r.get("code"):
        st.session_state["lab_code"] = _r["code"]
        st.session_state["lab_last_loaded"] = _r.get("name", "")


def main():
    inject_theme()
    try:
        panel, uni, tech, index = load_data(data_version())
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    last_date = panel["date"].max()
    first_date = panel["date"].min()

    page = st.session_state.get("page", "dashboard")
    if page not in {k for k, _ in NAV_ITEMS}:
        page = "dashboard"

    with st.sidebar:
        st.markdown(
            "<div class='side-brand'>"
            "<div class='side-brand-mark'><svg viewBox='0 0 24 24' fill='none' stroke='currentColor' "
            "stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>"
            "<path d='M3 17l5-6 4 4 7-9'/><path d='M14 6h6v6'/></svg></div>"
            "<div><div class='side-brand-title'>量化回测工作台</div>"
            "<div class='side-brand-sub'>本地面板 · 策略研究 · 信号跟踪</div></div></div>",
            unsafe_allow_html=True,
        )
        render_sidebar_nav(page)
        st.markdown(
            "<div class='side-meta'>"
            f"<div class='side-meta-row'><span>数据区间</span>"
            f"<span>{first_date.date()} ~ {last_date.date()}</span></div>"
            f"<div class='side-meta-row'><span>股票池</span><span>{panel['code'].nunique()} 只</span></div>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='side-note'>回测基于本地沪深300+中证500+中证1000 面板数据；"
            "资金曲线为策略模拟跟踪，非真实账户。</div>",
            unsafe_allow_html=True,
        )

    # ---------------- 资金看板 ----------------
    if page == "dashboard":
        st.markdown("### 资金看板（模拟跟踪）")
        st.caption("等权基准：当前股票池全部股票每个交易日等权收益（剔除停牌/无成交），"
                   "起点归 1；A股指数用收盘价涨跌幅归一，与策略资金同起点对比。")
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                dash_capital = st.number_input("初始资金", value=5000.0, min_value=1000.0,
                                               step=1000.0, key="dash_capital")
            with c2:
                dash_topn = st.select_slider("持仓数 TopN", options=[1, 2, 3, 5, 8],
                                             value=3, key="dash_topn")
            with c3:
                dash_start = st.date_input("回看起点", value=(last_date - pd.DateOffset(months=6)).date(),
                                           min_value=first_date.date(), max_value=last_date.date(),
                                           key="dash_start_date")
                dash_end = st.date_input("回看终点", value=last_date.date(),
                                         min_value=first_date.date(), max_value=last_date.date(),
                                         key="dash_end_date")
            dash_idx_names = sorted(index["name"].unique()) if len(index) else []
            dash_bench = st.multiselect(
                "基准（可多选）", ["等权股票池"] + dash_idx_names,
                default=["等权股票池", "沪深300", "中证500"],
                key="dash_bench",
            )

        default_sel = {"低换手冷门", "反转 20 日", "低波动", "动量 20 日"}
        for name in list_strategies():
            key = f"dash_sel_{name}"
            if key not in st.session_state:
                st.session_state[key] = name in default_sel
        dash_strategies = [n for n in list_strategies()
                           if st.session_state.get(f"dash_sel_{n}")]

        with st.popover(f"策略（已选 {len(dash_strategies)}/{len(list_strategies())}）"):
            st.caption("点选或搜索，勾选即时生效")
            dash_search = st.text_input("搜索策略", placeholder="搜索名称/说明…",
                                        label_visibility="collapsed", key="dash_search")
            p1, p2, p3 = st.columns(3)
            with p1:
                if st.button("默认 4 个", key="dash_sel_default", use_container_width=True):
                    for name in list_strategies():
                        st.session_state[f"dash_sel_{name}"] = name in default_sel
                    st.rerun()
            with p2:
                if st.button("全选", key="dash_sel_all", use_container_width=True):
                    for name in list_strategies():
                        st.session_state[f"dash_sel_{name}"] = True
                    st.rerun()
            with p3:
                if st.button("清空", key="dash_sel_none", use_container_width=True):
                    for name in list_strategies():
                        st.session_state[f"dash_sel_{name}"] = False
                    st.rerun()
            q = dash_search.strip().lower()
            for g, names in strategy_groups():
                show = [n for n in names
                        if not q or q in n.lower() or q in STRATEGIES[n].get("desc", "").lower()]
                if not show:
                    continue
                gcol, bcol = st.columns([3, 1])
                with gcol:
                    st.markdown(f"**{g}**")
                with bcol:
                    if st.button("本组全选/取消", key=f"dash_grp_{g}",
                                 use_container_width=True):
                        group_all = all(st.session_state.get(f"dash_sel_{n}") for n in names)
                        for n in names:
                            st.session_state[f"dash_sel_{n}"] = not group_all
                        st.rerun()
                cols = st.columns(2)
                for i, name in enumerate(show):
                    with cols[i % 2]:
                        st.checkbox(name, key=f"dash_sel_{name}",
                                    help=STRATEGIES[name].get("desc", ""))
        if dash_strategies:
            st.caption("已选策略：" + "、".join(dash_strategies))
        else:
            st.caption("未选择策略")

        if dash_strategies:
            s, e = min(dash_start, dash_end), max(dash_start, dash_end)
            dash_start, dash_end = pd.Timestamp(s), pd.Timestamp(e)
            codes = build_codes("科技TMT", True, panel, uni, tech)
            navs: dict[str, pd.Series] = {}
            holdings: dict[str, pd.DataFrame] = {}
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
                        long_short=strat.get("long_short", False),
                        short_n=strat.get("short_n"),
                        short_cost_rate=strat.get("short_cost_rate", 0.0),
                        industry_neutral=strat.get("industry_neutral", False),
                    )
                navs[sname] = res["nav"]
                holdings[sname] = res["holdings"]
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

            benches: dict[str, pd.Series] = {}
            if "等权股票池" in dash_bench and bench_line is not None:
                benches["等权股票池"] = bench_line
            idx_map = index_series_map(index, dash_start, dash_end)
            for b in dash_bench:
                if b != "等权股票池" and b in idx_map:
                    benches[b] = idx_map[b]

            st.plotly_chart(equity_compare_chart(navs, float(dash_capital), benches=benches),
                            use_container_width=True)
            st.markdown("#### 策略指标对比")
            show_table(pd.DataFrame(rows))
            if benches:
                st.markdown("#### 基准对比")
                show_table(benchmark_rows(benches))

            st.markdown("#### 当前持仓（最近一次调仓）")
            h_tabs = st.tabs([f"{sname} · Top{int(dash_topn)}" for sname in dash_strategies])
            nm = get_name_map(uni, tech)
            for h_tab, sname in zip(h_tabs, dash_strategies):
                with h_tab:
                    h = holdings[sname]
                    if h.empty:
                        st.info("当前空仓")
                        continue
                    h = h.copy()
                    h["名称"] = [nm.get(str(c), "") for c in h["code"]]
                    show_table(
                        h[["code", "名称", "weight_pct", "price", "market_value"]]
                        .rename(columns={"code": "代码", "weight_pct": "权重%",
                                         "price": "价格", "market_value": "市值"}))
        else:
            st.info("请至少选择一个策略")

    # ---------------- 策略池 ----------------
    elif page == "pool":
        render_pool_page()

    # ---------------- 回测工作台 ----------------
    elif page == "backtest":
        st.markdown("### 回测工作台")
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                universe = st.selectbox("股票池", ["科技TMT", "沪深300+中证500+中证1000", "ETF", "场外科技基金"], key="bt_universe")
            with c2:
                bt_scope = strategy_scope_ui("bt_scope")
                strategy = st.selectbox("策略", strategy_options(bt_scope), key="bt_strategy",
                                        format_func=strategy_display)
            with c3:
                top_n = st.select_slider("TopN", options=[1, 2, 3, 5, 8, 10], value=5,
                                         key="bt_topn")
            with c4:
                freq_ui = st.selectbox("调仓频率", ["月频", "周频", "半年频(3/9月)"], key="bt_freq")

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

            strat = resolve_strategy(strategy)
            c8, c9, c10, c11 = st.columns(4)
            with c8:
                bt_long_short = st.checkbox(
                    "多空对冲（多头TopN+空头N）",
                    value=bool(strat.get("long_short", False)), key="bt_long_short",
                    help="多头买因子最强 TopN，空头卖最弱 N 只，净敞口 0（模拟融券）")
            with c9:
                bt_neutral = st.checkbox("行业中性化选股", value=False,
                                         key="bt_neutral",
                                         help="选股前把因子得分按行业内截面去均值")
            with c10:
                bt_short_n = st.number_input("空头只数", value=int(strat.get("short_n", 3) or 3),
                                             min_value=1, max_value=20, step=1,
                                             key="bt_short_n", disabled=not bt_long_short)
            with c11:
                bt_short_rate = st.number_input(
                    "融券年化费率%",
                    value=round(float(strat.get("short_cost_rate", 0.086) or 0.086) * 100, 1),
                    step=0.5,
                                                key="bt_short_rate",
                                                disabled=not bt_long_short)

            c12, c13, c14 = st.columns(3)
            with c12:
                bt_financial = st.checkbox(
                    "使用财务因子(ROE/PB/EP等)", value=False, key="bt_financial",
                    help="从 PG 财务宽表读取基本面因子（当前为样例数据，全市场需"
                         "先运行 scripts/sync_postgres.py --fina）")
            with c13:
                bt_risk_neutral = st.checkbox(
                    "风险中性化(风格+行业)", value=False, key="bt_risk_neutral",
                    help="选股前把因子得分对流动性/动量/波动/换手/行业暴露回归取残差，"
                         "并返回期末持仓的风险归因")
            with c14:
                st.caption("风险中性化会额外输出风格/行业风险贡献，用于识别收益来源。")

            d1, d2 = st.columns(2)
            with d1:
                start = st.date_input("开始日期", value=(last_date - pd.DateOffset(months=6)).date(),
                                      min_value=first_date.date(), max_value=last_date.date(),
                                      key="bt_start")
            with d2:
                end = st.date_input("结束日期", value=last_date.date(),
                                    min_value=first_date.date(), max_value=last_date.date(),
                                    key="bt_end")
            bt_idx_names = sorted(index["name"].unique()) if len(index) else []
            bt_bench = st.multiselect(
                "基准（可多选）", ["等权股票池"] + bt_idx_names,
                default=["等权股票池", "沪深300", "中证500"],
                key="bt_bench",
            )

            if st.button("跑回测", type="primary"):
                if start >= end:
                    st.error("开始日期必须早于结束日期")
                else:
                    codes = build_codes(universe, exclude_kc, panel, uni, tech)
                    freq = {"月频": "monthly", "周频": "weekly",
                            "半年频(3/9月)": "semiannual"}[freq_ui]
                    with st.spinner("计算中..."):
                        ind_map = get_industry_map(tech) if (
                            strat.get("industry_cap") or
                            st.session_state.get("bt_neutral", False) or
                            st.session_state.get("bt_risk_neutral", False)
                        ) else None
                        res = run_backtest(
                            panel=panel, codes=codes, factor=strat["factor"],
                            ascending=strat["ascending"], start=str(start), end=str(end),
                            capital=float(capital), top_n=int(top_n), freq=freq,
                            affordable=affordable,
                            industry_map=ind_map,
                            industry_cap=strat.get("industry_cap"),
                            long_short=bool(st.session_state.get("bt_long_short", False)),
                            short_n=int(st.session_state.get("bt_short_n", 3)),
                            short_cost_rate=float(st.session_state.get("bt_short_rate", 0.0)) / 100.0,
                            industry_neutral=bool(st.session_state.get("bt_neutral", False)),
                            use_financial=bool(st.session_state.get("bt_financial", False)),
                            risk_neutral=bool(st.session_state.get("bt_risk_neutral", False)),
                        )
                    st.session_state["bt_result"] = res
                    try:
                        from core.attribution import brinson_attribution
                        b_detail, b_summary = brinson_attribution(
                            panel, codes, res["weight_history"],
                            res["dates"], ind_map or get_industry_map(tech))
                        st.session_state["bt_brinson"] = (b_detail, b_summary)
                    except Exception:
                        st.session_state["bt_brinson"] = None
                    st.session_state["bt_desc"] = (
                        f"{universe} · {strategy} · Top{top_n} · {freq} · "
                        f"{start} ~ {end} · 资金 {capital:,.0f}"
                        + (" · 多空对冲" if st.session_state.get("bt_long_short") else "")
                        + (" · 行业中性" if st.session_state.get("bt_neutral") else "")
                        + (" · 财务因子" if st.session_state.get("bt_financial") else "")
                        + (" · 风险中性" if st.session_state.get("bt_risk_neutral") else "")
                    )

        if "bt_result" in st.session_state:
            res = st.session_state["bt_result"]
            st.markdown(f"**{st.session_state['bt_desc']}**")
            st.markdown("##### 策略指标")
            render_metrics(st.columns(6), res["metrics"])
            st.markdown("##### 基准指标（等权股票池）")
            render_metrics(st.columns(6), res["bench_metrics"])
            bt_benches: dict[str, pd.Series] = {}
            if "等权股票池" in bt_bench:
                bt_benches["等权股票池"] = res["bench"]
            bt_idx_map = index_series_map(index, start, end)
            for b in bt_bench:
                if b != "等权股票池" and b in bt_idx_map:
                    bt_benches[b] = bt_idx_map[b]
            st.plotly_chart(equity_chart(res["nav"], float(capital), benches=bt_benches),
                            use_container_width=True)
            if bt_benches:
                st.markdown("##### 基准对比")
                show_table(benchmark_rows(bt_benches))
            st.plotly_chart(drawdown_chart(res["drawdown"]), use_container_width=True)

            tab_h, tab_t, tab_r, tab_b = st.tabs(
                ["持仓明细", "调仓记录", "风险归因", "行业归因(Brinson)"])
            with tab_h:
                if res["holdings"].empty:
                    st.info("当前空仓")
                else:
                    nm = get_name_map(uni, tech)
                    h = res["holdings"].copy()
                    h["名称"] = [nm.get(str(c), "") for c in h["code"]]
                    cols = ["code", "名称"]
                    if "direction" in h.columns:
                        cols.append("direction")
                    cols += ["weight_pct", "price", "market_value"]
                    rename = {"code": "代码", "weight_pct": "权重%",
                              "price": "价格", "market_value": "市值",
                              "direction": "方向"}
                    show_table(h[cols].rename(columns=rename))
            with tab_t:
                if res["trades"].empty:
                    st.info("无调仓记录")
                else:
                    show_table(res["trades"])
            with tab_r:
                ra = res.get("risk_attribution")
                if not ra:
                    st.info("未开启风险中性化，无风险归因。勾选「风险中性化(风格+行业)」后重跑回测。")
                else:
                    ra_df = pd.DataFrame([{"因子": k, "风险贡献%": v * 100}
                                          for k, v in sorted(
                                              ra.items(), key=lambda kv: -abs(kv[1]))])
                    show_table(ra_df)
                    st.caption("风险贡献 = 因子项方差占比 + specific 残差占比；"
                               "行业因子来自 tech 缓存（轻量 Barra 近似）。")
            with tab_b:
                br = st.session_state.get("bt_brinson")
                if not br or len(br[1]) == 0:
                    st.info("暂无可用的行业归因结果（可能月初一直空仓或股票池无行业映射）。")
                else:
                    _, b_summary = br
                    bs = b_summary.copy()
                    bs["配置(%)"] = (bs["allocation"] * 100).round(2)
                    bs["选择(%)"] = (bs["selection"] * 100).round(2)
                    bs["交互(%)"] = (bs["interaction"] * 100).round(2)
                    bs["合计(%)"] = (bs["total_pct"]).round(2)
                    show_table(bs[["industry", "配置(%)", "选择(%)", "交互(%)",
                                   "合计(%)", "avg_combo_weight", "avg_bench_weight"]]
                               .rename(columns={"industry": "行业",
                                                "avg_combo_weight": "组合均权",
                                                "avg_bench_weight": "基准均权"}))
                    with st.expander("逐月逐行业明细"):
                        b_detail, _ = br
                        show_table(b_detail)

    # ---------------- 组合策略 ----------------
    elif page == "composite":
        st.markdown("### 多因子自由组合")
        st.caption("勾选因子、设定方向与权重后可直接回测；组合可保存/加载/删除。"
                   "每个因子先做横截面百分位排名（0~1），按方向翻转后乘以权重求和，"
                   "再按综合得分降序选股（买高）。")

        if "comp_result" not in st.session_state:
            st.session_state["comp_result"] = None
        if "comp_desc" not in st.session_state:
            st.session_state["comp_desc"] = ""
        if "comp_sig" not in st.session_state:
            st.session_state["comp_sig"] = None
        if "comp_sig_date" not in st.session_state:
            st.session_state["comp_sig_date"] = None
        comp_saved = list(load_composites().values())

        with st.container(border=True):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                comp_universe = st.selectbox("股票池", ["科技TMT", "沪深300+中证500+中证1000", "ETF", "场外科技基金"],
                                             key="comp_universe")
            with c2:
                comp_top_n = st.select_slider("TopN", options=[1, 2, 3, 5, 8, 10],
                                              value=5, key="comp_topn")
            with c3:
                comp_capital = st.number_input("初始资金", value=50000.0, min_value=1000.0,
                                               step=1000.0, key="comp_capital")
            with c4:
                comp_freq_ui = st.selectbox("调仓频率", ["月频", "周频", "半年频(3/9月)"], key="comp_freq")
            d1, d2 = st.columns(2)
            with d1:
                comp_start = st.date_input("开始日期",
                                           value=(last_date - pd.DateOffset(months=6)).date(),
                                           min_value=first_date.date(), max_value=last_date.date(),
                                           key="comp_start")
            with d2:
                comp_end = st.date_input("结束日期", value=last_date.date(),
                                         min_value=first_date.date(), max_value=last_date.date(),
                                         key="comp_end")

        fcol, mcol = st.columns([2.3, 1])
        with fcol:
            with st.container(border=True):
                st.markdown("**因子配置**")
                for f in FACTOR_OPTIONS:
                    name = f["name"]
                    r = st.columns([2.5, 1.1, 1])
                    with r[0]:
                        enabled = st.checkbox(f["label"], key=f"comp_en_{name}",
                                              help=f.get("desc", ""))
                    with r[1]:
                        st.selectbox("方向", ["买高", "买低"], key=f"comp_dir_{name}",
                                     label_visibility="collapsed", disabled=not enabled)
                    with r[2]:
                        st.number_input("权重", value=1.0, step=0.1, key=f"comp_w_{name}",
                                        label_visibility="collapsed", disabled=not enabled)
        with mcol:
            with st.container(border=True):
                st.markdown("**组合管理**")
                comp_names = [c["name"] for c in comp_saved]
                if comp_names:
                    comp_pick = st.selectbox("已保存组合", comp_names, key="comp_pick")
                    if st.button("加载", key="comp_load", use_container_width=True):
                        item = next(c for c in comp_saved if c["name"] == comp_pick)
                        for name, w in item["weights"].items():
                            st.session_state[f"comp_en_{name}"] = True
                            st.session_state[f"comp_w_{name}"] = w
                        for name, d in item.get("directions", {}).items():
                            st.session_state[f"comp_dir_{name}"] = "买低" if d else "买高"
                        st.rerun()
                    if st.button("删除", key="comp_delete", use_container_width=True):
                        delete_composite(comp_pick)
                        st.rerun()
                else:
                    st.caption("暂无已保存组合")
                st.text_input("组合名称", key="comp_name", placeholder="如：动量+低波")
                if st.button("保存组合", key="comp_save", use_container_width=True):
                    weights, directions = _collect_composite()
                    cname = st.session_state.get("comp_name", "").strip()
                    if not weights:
                        st.error("请至少勾选一个因子")
                    else:
                        try:
                            item = save_composite(cname, weights, directions)
                            st.success(f"已保存组合：{item['name']}")
                        except ValueError as exc:
                            st.error(str(exc))

        b1, b2 = st.columns([1, 1])
        with b1:
            if st.button("跑组合回测", type="primary"):
                weights, directions = _collect_composite()
                if not weights:
                    st.error("请至少勾选一个因子")
                elif comp_start >= comp_end:
                    st.error("开始日期必须早于结束日期")
                else:
                    codes = build_codes(comp_universe, True, panel, uni, tech)
                    freq = {"月频": "monthly", "周频": "weekly",
                            "半年频(3/9月)": "semiannual"}[comp_freq_ui]
                    with st.spinner("计算中..."):
                        res = run_backtest(
                            panel=panel, codes=codes, factor="composite", ascending=False,
                            start=str(comp_start), end=str(comp_end),
                            capital=float(comp_capital), top_n=int(comp_top_n), freq=freq,
                            affordable=True,
                            factor_weights=weights,
                            factor_directions=directions,
                        )
                    st.session_state["comp_result"] = res
                    st.session_state["comp_desc"] = (
                        f"{comp_universe} · 组合 {st.session_state.get('comp_name', '').strip() or '(未命名)'} · "
                        f"Top{comp_top_n} · {freq} · {comp_start} ~ {comp_end} · 资金 {comp_capital:,.0f}"
                    )
        with b2:
            if st.button("看今日信号"):
                weights, directions = _collect_composite()
                if not weights:
                    st.error("请至少勾选一个因子")
                else:
                    codes = build_codes(comp_universe, True, panel, uni, tech)
                    sig, sig_date = latest_signals(
                        panel, codes, "composite", False, top_n=10,
                        factor_weights=weights, factor_directions=directions,
                    )
                    st.session_state["comp_sig"] = sig
                    st.session_state["comp_sig_date"] = sig_date

        comp_res = st.session_state["comp_result"]
        if comp_res is not None:
            st.markdown(f"**{st.session_state['comp_desc']}**")
            st.markdown("##### 策略指标")
            render_metrics(st.columns(6), comp_res["metrics"])
            st.markdown("##### 基准指标（等权股票池）")
            render_metrics(st.columns(6), comp_res["bench_metrics"])
            comp_benches: dict[str, pd.Series] = {"等权股票池": comp_res["bench"]}
            comp_idx_map = index_series_map(index, comp_start, comp_end)
            for b in ("沪深300", "中证500"):
                if b in comp_idx_map:
                    comp_benches[b] = comp_idx_map[b]
            st.plotly_chart(equity_chart(comp_res["nav"], float(comp_capital),
                                         benches=comp_benches), use_container_width=True)
            st.plotly_chart(drawdown_chart(comp_res["drawdown"]), use_container_width=True)
            ctab_h, ctab_t = st.tabs(["持仓明细", "调仓记录"])
            with ctab_h:
                if comp_res["holdings"].empty:
                    st.info("当前空仓")
                else:
                    nm = get_name_map(uni, tech)
                    h = comp_res["holdings"].copy()
                    h["名称"] = [nm.get(str(c), "") for c in h["code"]]
                    show_table(h[["code", "名称", "weight_pct", "price", "market_value"]]
                               .rename(columns={"code": "代码", "weight_pct": "权重%",
                                                "price": "价格", "market_value": "市值"}))
            with ctab_t:
                if comp_res["trades"].empty:
                    st.info("无调仓记录")
                else:
                    show_table(comp_res["trades"])

        comp_sig = st.session_state["comp_sig"]
        if comp_sig is not None and len(comp_sig):
            st.markdown("##### 今日信号")
            sig_date = st.session_state["comp_sig_date"]
            st.caption(f"信号日：{sig_date.date()} · 数据截至：{last_date.date()}")
            nm = get_name_map(uni, tech)
            sdf = pd.DataFrame({
                "代码": comp_sig["code"],
                "名称": [nm.get(str(c), "") for c in comp_sig["code"]],
                "因子得分": comp_sig["score"],
                "收盘价": comp_sig["close"],
            })
            show_table(sdf)

    # ---------------- 代码实验室 ----------------
    elif page == "code":
        st.markdown("### 代码实验室")
        st.caption("从「已有策略」选一个加载代码模板，直接改下面的 Python 代码，"
                   "然后点「跑代码」回测；保存只写入服务器 labs/ 目录，不覆盖系统策略文件。")

        if "lab_code" not in st.session_state:
            st.session_state["lab_code"] = lab_api.get_default()["code"]
        if "lab_strategies" not in st.session_state:
            st.session_state["lab_strategies"] = []
        if "lab_result" not in st.session_state:
            st.session_state["lab_result"] = None
        if "lab_saved_list" not in st.session_state:
            st.session_state["lab_saved_list"] = lab_api.list_saved()["items"]
        if "lab_tpl_options" not in st.session_state:
            st.session_state["lab_tpl_options"] = _lab_tpl_options()

        lc0 = st.columns([2, 1, 1, 2])
        with lc0[0]:
            st.selectbox("已有策略",
                         [o["label"] for o in st.session_state["lab_tpl_options"]],
                         key="lab_tpl_pick", on_change=_lab_on_tpl_change)
            if "lab_last_loaded" in st.session_state:
                st.caption("已加载：" + st.session_state["lab_last_loaded"])
        with lc0[1]:
            if st.button("解析策略"):
                with st.spinner("解析中..."):
                    _r = lab_api.parse_code(lab_api.RunRequest(
                        code=st.session_state["lab_code"]))
                if _r.get("ok"):
                    st.session_state["lab_strategies"] = _r["strategies"]
                    st.success("解析成功：" + "、".join(_r["strategies"]))
                else:
                    st.error(_r.get("error", "解析失败"))
        with lc0[2]:
            if st.button("保存"):
                _name = st.session_state.get("lab_save_name", "").strip()
                if not _name:
                    st.error("请先填保存名称")
                else:
                    _r = lab_api.save_code(lab_api.SaveRequest(
                        name=_name, code=st.session_state["lab_code"]))
                    if _r.get("ok"):
                        st.session_state["lab_saved_list"] = lab_api.list_saved()["items"]
                        st.session_state["lab_tpl_options"] = _lab_tpl_options()
                        st.success("已保存：" + _r["name"])
                    else:
                        st.error(_r.get("error", "保存失败"))
        with lc0[3]:
            st.text_input("保存名称（保存时使用）", key="lab_save_name",
                          placeholder="如：我的双均线 v1")

        st.text_area("策略代码（Python）", key="lab_code", height=360,
                     placeholder="# 在这里修改策略代码")
        st.caption("提示：代码里 STRATEGIES 的策略名会自动解析；改了代码后先点「解析策略」。")

        st.markdown("---")
        with st.container(border=True):
            st.markdown("**运行参数**")
            rc1, rc2, rc3, rc4 = st.columns(4)
            with rc1:
                lab_universe = st.selectbox("股票池", ["科技TMT", "沪深300+中证500+中证1000", "ETF", "场外科技基金"],
                                            key="lab_universe")
            with rc2:
                _lab_strats = st.session_state["lab_strategies"] or ["低换手冷门"]
                if st.session_state.get("lab_strategy") not in _lab_strats:
                    st.session_state.pop("lab_strategy", None)
                lab_strategy = st.selectbox("策略", _lab_strats, key="lab_strategy")
            with rc3:
                lab_top_n = st.select_slider("TopN", options=[1, 2, 3, 5, 8, 10], value=3,
                                             key="lab_topn")
            with rc4:
                lab_freq = st.selectbox("调仓频率", ["月频", "周频", "半年频(3/9月)"], key="lab_freq")

            rc5, rc6, rc7 = st.columns(3)
            with rc5:
                lab_capital = st.number_input("初始资金", value=5000.0, min_value=1000.0,
                                              step=1000.0, key="lab_capital")
            with rc6:
                lab_exclude = st.checkbox("剔除科创/创业板", value=True, key="lab_exclude")
            with rc7:
                lab_affordable = st.checkbox("一手过滤", value=True, key="lab_affordable")

            rd1, rd2 = st.columns(2)
            with rd1:
                lab_start = st.date_input("开始日期", value=(last_date - pd.DateOffset(months=6)).date(),
                                          min_value=first_date.date(), max_value=last_date.date(),
                                          key="lab_start")
            with rd2:
                lab_end = st.date_input("结束日期", value=last_date.date(),
                                        min_value=first_date.date(), max_value=last_date.date(),
                                        key="lab_end")

            re1, re2 = st.columns(2)
            with re1:
                lab_amount_q = st.number_input("成交额分位", value=0.2, min_value=0.0,
                                               max_value=1.0, step=0.05, key="lab_amount_q")
            with re2:
                lab_warmup = st.selectbox("因子预热天数", [0, 120, 400, 9999], index=2,
                                          key="lab_warmup")

            rp1, rp2 = st.columns(2)
            with rp1:
                lab_slippage = st.number_input("滑点(bps，事件策略)", value=0.0,
                                               min_value=0.0, step=5.0,
                                               key="lab_slippage")
            with rp2:
                lab_participation = st.selectbox(
                    "流动性约束(单笔/20日均额，事件策略)",
                    [0.0, 0.05, 0.1, 0.2], index=0,
                    format_func=lambda x: "不限" if x == 0 else f"{x:.0%}",
                    key="lab_participation")

            if st.button("跑代码", type="primary"):
                if lab_start >= lab_end:
                    st.error("开始日期必须早于结束日期")
                else:
                    _req = lab_api.RunRequest(
                        code=st.session_state["lab_code"],
                        strategy=lab_strategy,
                        universe=lab_universe,
                        top_n=int(lab_top_n),
                        capital=float(lab_capital),
                        freq={"月频": "monthly", "周频": "weekly",
                              "半年频(3/9月)": "semiannual"}[lab_freq],
                        start=str(lab_start),
                        end=str(lab_end),
                        exclude_kechuang=lab_exclude,
                        affordable=lab_affordable,
                        amount_q=lab_amount_q,
                        warmup_days=int(lab_warmup) if lab_warmup else None,
                        slippage_bps=float(lab_slippage),
                        max_participation=float(lab_participation),
                    )
                    with st.spinner("运行中（子进程执行，请稍候）..."):
                        _r = lab_api.run_code(_req)
                    if _r.get("ok"):
                        st.session_state["lab_result"] = _r
                        if _r.get("strategies"):
                            st.session_state["lab_strategies"] = _r["strategies"]
                        st.success("运行完成")
                    else:
                        st.error(_r.get("error", "运行失败"))

        _lab_result = st.session_state.get("lab_result")
        if _lab_result:
            _nav = points_to_series(_lab_result["nav"])
            _bench = points_to_series(_lab_result["bench"])
            st.markdown("##### 策略指标")
            render_metrics(st.columns(6), _lab_result["metrics"])
            st.plotly_chart(equity_chart(_nav, float(lab_capital),
                                         {"等权基准": _bench}),
                            use_container_width=True)
            st.plotly_chart(drawdown_chart(points_to_series(_lab_result["drawdown"])),
                            use_container_width=True)
            tab_lh, tab_lt = st.tabs(["持仓明细", "调仓记录"])
            with tab_lh:
                if _lab_result["holdings"]:
                    _hdf = pd.DataFrame(_lab_result["holdings"])
                    if "name" in _hdf.columns:
                        _hdf["name"] = _hdf["name"].fillna("")
                    show_table(
                        _hdf.rename(columns={"code": "代码", "name": "名称",
                                             "weight_pct": "权重%", "price": "价格",
                                             "market_value": "市值"}))
                else:
                    st.info("当前空仓")
            with tab_lt:
                if _lab_result["trades"]:
                    show_table(pd.DataFrame(_lab_result["trades"]))
                else:
                    st.info("无调仓记录")

    # ---------------- 参数稳健性 / Walk-forward ----------------
    elif page == "sweep":
        st.markdown("### 参数稳健性 / Walk-forward")
        st.caption("把时间轴切成多个连续窗口独立回测，看策略跨窗口的指标分布。"
                   "判断依据：均值夏普、胜率、最差窗口——单段总收益高但窗口间波动大的组合，"
                   "通常是过拟合信号。")
        codes = build_codes("科技TMT", True, panel, uni, tech)
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                sweep_mode = st.selectbox(
                    "模式", ["双均线金叉参数扫描", "因子策略 walk-forward",
                             "滚动训练-测试(因子)", "滚动训练-测试(双均线)"],
                    key="sweep_mode")
            with c2:
                sweep_start = st.date_input("起点", value=pd.Timestamp("2023-01-03").date(),
                                            key="sweep_start")
            with c3:
                sweep_end = st.date_input("终点", value=last_date.date(), key="sweep_end")
            with c4:
                sweep_folds = st.select_slider("窗口数", options=[2, 3, 4, 6], value=4,
                                               key="sweep_folds")
            if sweep_mode == "双均线金叉参数扫描":
                c5, c6, c7 = st.columns(3)
                with c5:
                    sweep_short = st.text_input("短期均线列表", value="3,5,8,10,13",
                                                key="sweep_short")
                with c6:
                    sweep_long = st.text_input("长期均线列表", value="10,20,30,60",
                                               key="sweep_long")
                with c7:
                    sweep_topn = st.select_slider("持仓数", options=[1, 2, 3, 5], value=3,
                                                  key="sweep_topn")
                if st.button("运行参数扫描", type="primary", key="sweep_run"):
                    from core.walkforward import golden_cross_sweep
                    try:
                        shorts = [int(x.strip()) for x in sweep_short.split(",") if x.strip()]
                        longs = [int(x.strip()) for x in sweep_long.split(",") if x.strip()]
                    except ValueError:
                        st.error("参数列表必须是逗号分隔的整数，如 3,5,8")
                    else:
                        with st.spinner("扫描中（约 1-2 分钟）..."):
                            summary, heatmap, windows = golden_cross_sweep(
                                panel, codes, str(sweep_start), str(sweep_end), 50000,
                                short_list=shorts, long_list=longs,
                                n_folds=int(sweep_folds), top_n=int(sweep_topn))
                        st.session_state["sweep_summary"] = summary
                        st.session_state["sweep_heatmap"] = heatmap
                        st.session_state["sweep_windows"] = windows
                        st.session_state.pop("sweep_param_history", None)
            elif sweep_mode == "因子策略 walk-forward":
                sweep_scope = strategy_scope_ui("sweep_scope_wf")
                sweep_strategy = st.selectbox("策略", strategy_options(sweep_scope),
                                              key="sweep_strategy",
                                              format_func=strategy_display)
                if st.button("运行 walk-forward", type="primary", key="sweep_run_factor"):
                    from core.walkforward import walk_forward_factor
                    s = resolve_strategy(sweep_strategy)
                    with st.spinner("回测中（约 10-30 秒）..."):
                        wf = walk_forward_factor(
                            panel, codes, s["factor"], s["ascending"],
                            str(sweep_start), str(sweep_end), 50000,
                            top_n=3, n_folds=int(sweep_folds))
                    st.session_state["sweep_windows"] = wf
                    st.session_state.pop("sweep_summary", None)
                    st.session_state.pop("sweep_heatmap", None)
                    st.session_state.pop("sweep_param_history", None)
            elif sweep_mode == "滚动训练-测试(因子)":
                sweep_scope = strategy_scope_ui("sweep_scope_roll")
                sweep_strategy = st.selectbox("策略", strategy_options(sweep_scope),
                                              key="sweep_strategy",
                                              format_func=strategy_display)
                st.caption("每个测试窗口之前，用之前全部历史在 top_n ∈ {3,5} × 频率网格"
                           "选夏普最优参数，再跑当前窗口做样本外验证。")
                if st.button("运行滚动训练-测试", type="primary",
                             key="sweep_run_roll_factor"):
                    from core.walkforward import rolling_train_test_factor
                    s = resolve_strategy(sweep_strategy)
                    with st.spinner("滚动训练-测试中（约 30-90 秒）..."):
                        wf, summary, hist = rolling_train_test_factor(
                            panel, codes, s["factor"], s["ascending"],
                            str(sweep_start), str(sweep_end), 50000,
                            top_n_list=[3, 5], freq_list=["monthly"],
                            n_folds=int(sweep_folds))
                    st.session_state["sweep_windows"] = wf
                    st.session_state["sweep_summary"] = summary
                    st.session_state["sweep_param_history"] = hist
                    st.session_state.pop("sweep_heatmap", None)
            else:
                sweep_strategy = st.selectbox("策略", ["双均线多头 5/20"],
                                              key="sweep_strategy",
                                              format_func=strategy_display)
                c5, c6 = st.columns(2)
                with c5:
                    sweep_short = st.text_input("候选短期均线", value="3,5,8,10,13",
                                                key="sweep_short")
                with c6:
                    sweep_long = st.text_input("候选长期均线", value="10,20,30,60",
                                               key="sweep_long")
                st.caption("每个测试窗口之前，用之前全部历史做一次 short×long 参数扫描"
                           "（按均值夏普选优），再跑当前窗口样本外验证。")
                if st.button("运行滚动训练-测试", type="primary",
                             key="sweep_run_roll_event"):
                    from core.walkforward import rolling_train_test_event
                    try:
                        shorts = [int(x.strip()) for x in sweep_short.split(",") if x.strip()]
                        longs = [int(x.strip()) for x in sweep_long.split(",") if x.strip()]
                    except ValueError:
                        st.error("参数列表必须是逗号分隔的整数，如 3,5,8")
                    else:
                        with st.spinner("滚动训练-测试中（约 1-3 分钟）..."):
                            wf, summary, hist = rolling_train_test_event(
                                panel, codes, str(sweep_start), str(sweep_end), 50000,
                                short_list=shorts, long_list=longs,
                                n_folds=int(sweep_folds))
                        st.session_state["sweep_windows"] = wf
                        st.session_state["sweep_summary"] = summary
                        st.session_state["sweep_param_history"] = hist
                        st.session_state.pop("sweep_heatmap", None)

        summary = st.session_state.get("sweep_summary")
        heatmap = st.session_state.get("sweep_heatmap")
        windows = st.session_state.get("sweep_windows")
        param_history = st.session_state.get("sweep_param_history")

        if windows is not None and len(windows):
            st.markdown("---")
            st.markdown("#### 逐窗口明细")
            wcols = [c for c in ("short", "long", "fold", "start", "end", "n_days",
                                 "chosen_top_n", "chosen_freq",
                                 "chosen_short", "chosen_long", "trained",
                                 "total", "annual", "sharpe", "mdd", "calmar",
                                 "win_rate", "end_nav") if c in windows.columns]
            show_df = windows[wcols].copy()
            for c in ("total", "annual", "mdd"):
                if c in show_df.columns:
                    show_df[c] = (show_df[c] * 100).round(2)
                    show_df = show_df.rename(columns={c: f"{c}(%)"})
            show_table(show_df)

        if summary is not None and len(summary):
            st.markdown("#### 参数组合汇总（按均值夏普排序）")
            scols = ["short", "long", "mean_sharpe", "median_sharpe", "std_sharpe",
                     "mean_total", "worst_total", "win_rate", "mean_mdd"]
            if "chosen_top_n" in summary.columns or "chosen_short" in summary.columns:
                scols = ["mode", "n_windows", "trained_windows", "mean_sharpe",
                         "median_sharpe", "win_rate", "mean_total", "worst_total"]
                if "mean_mdd" in summary.columns:
                    scols.append("mean_mdd")
            sdf = summary[scols].copy()
            sdf["mean_total(%)"] = (sdf["mean_total"] * 100).round(2)
            sdf["worst_total(%)"] = (sdf["worst_total"] * 100).round(2)
            sdf["mean_mdd(%)"] = (sdf["mean_mdd"] * 100).round(2)
            sdf["win_rate(%)"] = (sdf["win_rate"] * 100).round(0)
            show_table(sdf.drop(columns=["mean_total", "worst_total", "mean_mdd",
                                         "win_rate"]))

        if param_history is not None and len(param_history):
            st.markdown("#### 滚动训练-测试：逐窗口参数选择")
            ph = param_history.copy()
            ph["train_sharpe"] = ph["train_sharpe"].round(3)
            show_table(ph.rename(columns={
                "fold": "窗口", "train_start": "训练起点", "train_end": "训练终点",
                "test_start": "测试起点", "test_end": "测试终点",
                "chosen_top_n": "选中TopN", "chosen_freq": "选中频率",
                "chosen_short": "选中短均线", "chosen_long": "选中长均线",
                "train_sharpe": "训练夏普"}))

        if heatmap is not None and len(heatmap):
            st.markdown("#### 均值夏普热力图")
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 5.5))
                im = ax.imshow(heatmap.values, cmap="RdYlGn", aspect="auto")
                ax.set_xticks(range(len(heatmap.columns)))
                ax.set_xticklabels([f"long={c}" for c in heatmap.columns])
                ax.set_yticks(range(len(heatmap.index)))
                ax.set_yticklabels([f"short={r}" for r in heatmap.index])
                for i in range(len(heatmap.index)):
                    for j in range(len(heatmap.columns)):
                        v = heatmap.iloc[i, j]
                        if np.isfinite(v):
                            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                    fontsize=9)
                ax.set_title("Mean Sharpe by MA period (walk-forward)")
                fig.colorbar(im, ax=ax)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            except Exception as exc:
                st.warning(f"热力图渲染失败: {exc}")
            st.caption("热力图数据（short x long）也可从 results/parameter_sweep/ 的 CSV 查看。")

    # ---------------- 日级模拟盘 ----------------
    elif page == "paper":
        from core.paper import (account_equity, account_events, account_orders,
                                account_summary, account_trades, create_account,
                                delete_account, list_accounts, reset_account,
                                run_paper_trade, set_account_status,
                                update_account_strategy)
        st.markdown("### 日级模拟盘")
        st.caption("信号日（T-1）收盘生成目标持仓 → T 日开盘成交 → 收盘估值。"
                   "因子账户由同一回测引擎（cash_mode=True）重放，执行口径一致："
                   "现金/整手/费用/涨跌停/拒单；事件策略仍由事件引擎重放。"
                   "每日数据增量更新后，由 systemd 定时任务盘后自动执行；也可手动触发。")

        accounts = list_accounts()
        acc_by_id = {a["id"]: a for a in accounts}

        with st.container(border=True):
            st.caption("以下参数仅用于创建新账户；已存在账户的策略切换请在下方「选择账户」后进行。")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                pa_name = st.text_input("账户名称", value="模拟盘", key="pa_name")
            with c2:
                pa_scope = strategy_scope_ui("pa_scope")
                pa_strategy = st.selectbox("策略", strategy_options(pa_scope), key="pa_strategy",
                                           format_func=strategy_display)
            with c3:
                pa_universe = st.selectbox("股票池", ["科技TMT", "沪深300+中证500+中证1000", "ETF", "场外科技基金"],
                                           key="pa_universe")
            with c4:
                pa_freq = st.selectbox("调仓频率", ["daily", "weekly", "monthly", "semiannual"],
                                       key="pa_freq")
            c5, c6, c7, c8 = st.columns(4)
            with c5:
                pa_capital = st.number_input("初始资金", value=100000.0, min_value=1000.0,
                                             step=10000.0, key="pa_capital")
            with c6:
                pa_topn = st.select_slider("TopN", options=[1, 2, 3, 5, 8, 10], value=3,
                                           key="pa_topn")
            with c7:
                pa_maxw = st.number_input("单票权重上限", value=0.5, min_value=0.05,
                                          max_value=1.0, step=0.05, key="pa_maxw")
            with c8:
                pa_amount_q = st.number_input("成交额分位(流动性)", value=0.2, min_value=0.0,
                                              max_value=1.0, step=0.05, key="pa_amount_q")
            if st.button("创建账户", type="primary"):
                try:
                    s = resolve_strategy(pa_strategy)
                    risk_cfg = {"max_weight": float(pa_maxw),
                                "amount_q": float(pa_amount_q)}
                    for k in ("adx_filter", "chandelier_mult", "chandelier_period",
                              "regime_adx", "regime_scale"):
                        if s.get(k) not in (None, ""):
                            risk_cfg[k] = (int(s[k]) if k == "chandelier_period"
                                           else float(s[k]))
                    acc = create_account(
                        pa_name.strip(), pa_strategy, s["factor"], s["ascending"],
                        universe=pa_universe, capital=float(pa_capital),
                        top_n=int(pa_topn), freq=pa_freq,
                        risk_config=risk_cfg)
                    st.success(f"已创建账户 #{acc['id']} · {acc['name']}")
                    accounts = list_accounts()
                    acc_by_id = {a["id"]: a for a in accounts}
                except ValueError as exc:
                    st.error(str(exc))

        if not accounts:
            st.info("还没有模拟盘账户，先创建第一个。")
            return

        sel_id = st.selectbox(
            "选择账户", [a["id"] for a in accounts],
            format_func=lambda i: f"#{i} · {acc_by_id[i]['name']} · {acc_by_id[i]['strategy_name']}",
        )
        sel = acc_by_id[sel_id]
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            if st.button("手动执行一次", type="primary"):
                with st.spinner("撮合中..."):
                    res = run_paper_trade(
                        panel, {
                            "科技TMT": build_codes("科技TMT", True, panel, uni, tech),
                            "沪深300+中证500+中证1000":
                                build_codes("沪深300+中证500+中证1000", True, panel, uni, tech),
                        }, account_id=sel_id)
                st.success("执行完成")
                st.write(res)
        with c2:
            if st.button("Dry-run 预览"):
                with st.spinner("生成目标持仓..."):
                    res = run_paper_trade(
                        panel, {
                            "科技TMT": build_codes("科技TMT", True, panel, uni, tech),
                            "沪深300+中证500+中证1000":
                                build_codes("沪深300+中证500+中证1000", True, panel, uni, tech),
                        }, account_id=sel_id, dry_run=True)
                st.write(res)
        with c3:
            new_status = "暂停" if sel["status"] == "active" else "启用"
            if st.button(new_status):
                set_account_status(sel_id, "paused" if sel["status"] == "active" else "active")
                st.rerun()
        with c4:
            if st.button("重置"):
                reset_account(sel_id)
                st.success("已重置，恢复初始资金")
                st.rerun()
        with c5:
            if st.button("删除"):
                delete_account(sel_id)
                st.success("已删除")
                st.rerun()

        with st.container(border=True):
            st.markdown("##### 模拟盘策略（切换即自动更新）")
            if sel.get("strategy_type") == "event":
                st.info("事件策略账户不支持在线切换，请新建账户。")
            else:
                sw_scope = strategy_scope_ui("pa_sw_scope")
                sw_opts = strategy_options(sw_scope)
                sw_cur = sel.get("strategy_name")
                if sw_cur not in sw_opts:
                    # 当前策略不在选项里时前置插入，避免 selectbox 自动跳到 index=0 误触发
                    sw_opts = [sw_cur] + sw_opts
                sw_strategy = st.selectbox(
                    "策略", sw_opts, index=sw_opts.index(sw_cur),
                    format_func=strategy_display, key=f"pa_sw_{sel_id}",
                    help="切换后自动重置该账户并按新策略重新回放")
                freq_opts = ["daily", "weekly", "monthly", "semiannual"]
                sw_freq = st.selectbox("调仓频率", freq_opts,
                                       index=freq_opts.index(sel.get("freq", "monthly")),
                                       key=f"pa_sw_freq_{sel_id}")
                sw_topn = st.select_slider("TopN", options=[1, 2, 3, 5, 8, 10],
                                           value=int(sel.get("top_n", 3)),
                                           key=f"pa_sw_topn_{sel_id}")
                st.caption("切换会自动清空历史并按新策略重新回放模拟盘。")
                applied_key = f"pa_sw_applied_{sel_id}"
                cur_combo = (sw_strategy, sw_freq, int(sw_topn))
                if applied_key not in st.session_state:
                    st.session_state[applied_key] = (
                        sw_cur, sel.get("freq", "monthly"),
                        int(sel.get("top_n", 3)))
                if cur_combo != st.session_state[applied_key]:
                    st.session_state[applied_key] = cur_combo
                    try:
                        s = resolve_strategy(sw_strategy)
                        risk_cfg = {**sel.get("risk_config", {})}
                        for k in ("adx_filter", "chandelier_mult", "chandelier_period",
                                  "regime_adx", "regime_scale"):
                            if s.get(k) not in (None, ""):
                                risk_cfg[k] = (int(s[k]) if k == "chandelier_period"
                                               else float(s[k]))
                        update_account_strategy(
                            sel_id, strategy_name=sw_strategy,
                            factor=s["factor"], ascending=s["ascending"],
                            freq=sw_freq, top_n=int(sw_topn),
                            risk_config=risk_cfg)
                        reset_account(sel_id)
                        codes_map = {
                            "科技TMT": build_codes("科技TMT", True, panel, uni, tech),
                            "沪深300+中证500+中证1000":
                                build_codes("沪深300+中证500+中证1000", True, panel, uni, tech),
                        }
                        if sel.get("universe") in codes_map:
                            with st.spinner("正在按新策略重放模拟盘..."):
                                run_paper_trade(panel, codes_map, account_id=sel_id)
                        st.success(f"已自动切换到「{sw_strategy}」并按新策略重放")
                        st.rerun()
                    except (KeyError, ValueError) as exc:
                        st.error(str(exc))

        summary = account_summary(sel_id, panel)
        if summary is None:
            st.info("账户不存在")
            return
        latest = summary.get("latest")
        if latest:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("最新权益", f"{latest['equity']:,.2f}")
            c2.metric("现金", f"{latest['cash']:,.2f}")
            c3.metric("市值", f"{latest['market_value']:,.2f}")
            c4.metric("累计盈亏", f"{latest['pnl']:+,.2f}")
            c5.metric("收益率", format_pct(latest["pnl_pct"]))
        eq_rows = account_equity(sel_id)
        if eq_rows:
            eq = pd.DataFrame(eq_rows)
            eq["date"] = pd.to_datetime(eq["date"])
            nav_s = pd.Series(eq["equity"].values, index=eq["date"], name="nav")
            m = compute_metrics(nav_s)
            bench_s = None
            try:
                codes = build_codes(sel["universe"], True, panel, uni, tech)
                bench_s = equal_weight_benchmark(panel, codes, eq["date"])
            except Exception:
                bench_s = None
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("年化收益", format_pct(m.get("年化收益")))
            mc2.metric("夏普", fmt_num(m.get("夏普")))
            mc3.metric("最大回撤", format_pct(m.get("最大回撤")))
            if bench_s is not None and len(bench_s) > 1:
                bm = compute_metrics(bench_s)
                mc4.metric("基准收益（等权股票池）", format_pct(bm.get("总收益")))
            else:
                mc4.metric("基准收益（等权股票池）", "-")
            benches = {"等权股票池": bench_s} if bench_s is not None else None
            st.plotly_chart(equity_chart(nav_s, 1.0, benches=benches),
                            use_container_width=True)
        else:
            st.info("暂无估值快照，先「手动执行一次」。")

        tab_p, tab_o, tab_t, tab_e = st.tabs(["持仓", "订单", "成交", "事件日志"])
        with tab_p:
            if summary["positions"]:
                pdf = pd.DataFrame(summary["positions"])
                show_table(pdf.rename(columns={
                    "code": "代码", "name": "名称", "shares": "股数",
                    "avg_cost": "成本价", "price": "现价",
                    "market_value": "市值", "pnl": "浮动盈亏",
                    "updated_date": "更新日"}))
            else:
                st.info("当前空仓")
        with tab_o:
            rows = account_orders(sel_id)
            if rows:
                odf = pd.DataFrame(rows)
                odf["signal_date"] = pd.to_datetime(odf["signal_date"]).dt.date
                odf["exec_date"] = pd.to_datetime(odf["exec_date"]).dt.date
                show_table(odf.rename(columns={
                    "code": "代码", "side": "方向", "target_pct": "目标权重",
                    "signal_date": "信号日", "exec_date": "执行日",
                    "status": "状态", "fill_price": "成交价", "fee": "费用",
                    "reject_reason": "拒单原因"}))
            else:
                st.info("暂无订单")
        with tab_t:
            rows = account_trades(sel_id)
            if rows:
                tdf = pd.DataFrame(rows)
                tdf["exec_date"] = pd.to_datetime(tdf["exec_date"]).dt.date
                show_table(tdf.rename(columns={
                    "code": "代码", "side": "方向", "exec_date": "成交日",
                    "shares": "股数", "price": "成交价", "fee": "费用"}))
            else:
                st.info("暂无成交")
        with tab_e:
            rows = account_events(sel_id)
            if rows:
                edf = pd.DataFrame(rows)
                edf["date"] = pd.to_datetime(edf["date"]).dt.date
                show_table(edf.rename(columns={
                    "date": "日期", "level": "级别", "msg": "消息"}))
            else:
                st.info("暂无事件")

    # ---------------- 历史回测（PG 归档） ----------------
    elif page == "history":
        st.markdown("### 历史回测（PG 归档）")
        st.caption("每次回测/对比/扫描自动落库，参数、指标、净值、交易可追溯。")
        from core import backtest_archive
        kind_map = {"": "全部", "backtest": "单策略回测", "compare": "多策略对比",
                    "sweep": "参数扫描", "sweep_cli": "参数扫描(CLI)"}
        c1, c2 = st.columns([1, 1])
        with c1:
            kind_label = st.selectbox("类型", list(kind_map.values()), key="hist_kind")
        with c2:
            limit = st.selectbox("条数", [50, 100, 200], index=1, key="hist_limit")
        kind = next((k for k, v in kind_map.items() if v == kind_label), None)
        df = backtest_archive.list_runs(kind=kind or None, limit=int(limit))
        if df.empty:
            st.info("暂无归档记录。在「回测工作台」/「组合策略」跑一次回测后，这里会出现可追溯的参数与净值。")
        else:
            rows = []
            for _, r in df.iterrows():
                p = r.get("params") or {}
                s = r.get("summary") or {}
                rows.append({
                    "ID": int(r["run_id"]),
                    "时间": r["created_at"].strftime("%Y-%m-%d %H:%M:%S"),
                    "类型": kind_map.get(r.get("kind", ""), r.get("kind", "")),
                    "标题": _history_title(r.get("kind", ""), p, s),
                    "区间": f"{p.get('start', '')}~{p.get('end', '')}",
                    "总收益": s.get("total_return"),
                    "年化": s.get("annual"),
                    "夏普": s.get("sharpe"),
                    "最大回撤": s.get("max_drawdown"),
                    "数据版本": r.get("data_version", ""),
                })
            show_table(pd.DataFrame(rows), height=460)
            st.markdown("---")
            rid = st.number_input("输入 run_id 查看详情", min_value=1, step=1,
                                  value=int(df.iloc[0]["run_id"]))
            if st.button("加载详情", type="primary"):
                rec = backtest_archive.get_run(int(rid))
                if rec is None:
                    st.error("记录不存在")
                else:
                    _render_history_detail(rec)

    # ---------------- 今日信号 ----------------
    elif page == "signals":
        st.markdown("### 今日信号")
        with st.container(border=True):
            s1, s2 = st.columns(2)
            with s1:
                sig_universe = st.selectbox("股票池", ["科技TMT", "沪深300+中证500+中证1000", "ETF", "场外科技基金"], key="sig_universe")
            with s2:
                sig_scope = strategy_scope_ui("sig_scope")
                sig_strategy = st.selectbox("策略", strategy_options(sig_scope), index=0,
                                            key="sig_strategy", format_func=strategy_display)
            sig_n = st.slider("显示条数", 5, 30, 15, key="sig_n")
        strat = resolve_strategy(sig_strategy)
        codes = build_codes(sig_universe, True, panel, uni, tech)
        sig, sig_date = latest_signals(panel, codes, strat["factor"],
                                       strat["ascending"], top_n=int(sig_n),
                                       long_short=strat.get("long_short", False),
                                       short_n=strat.get("short_n"))
        mode_txt = ("多空对冲" if strat.get("long_short") else
                    ("升序(买低)" if strat["ascending"] else "降序(买高)"))
        sig_meta = data_status().get("meta", {})
        st.caption(f"信号日：{sig_date.date()} · 数据截至：{last_date.date()} · "
                   f"上次刷新：{sig_meta.get('last_update', '-')} · "
                   f"因子：{strat['factor']} · {mode_txt}")
        nm = get_name_map(uni, tech)
        sig2 = pd.DataFrame({
            "code": sig["code"],
            "名称": [nm.get(str(c), "") for c in sig["code"]],
            **({"方向": sig["side"]} if "side" in sig.columns else {}),
            "score": sig["score"],
            "close": sig["close"],
            "turnover": sig["turnover"],
        })
        sig2 = sig2.rename(columns={"code": "代码", "score": "因子得分",
                                    "close": "收盘价"})
        sig2["换手率%"] = (sig2["turnover"] * 100).round(2)
        cols = ["代码", "名称"] + (["方向"] if "side" in sig.columns else [])
        show_table(sig2[cols + ["因子得分", "收盘价", "换手率%"]])

    # ---------------- 舆情情绪 ----------------
    elif page == "news":
        st.markdown("### 舆情情绪看板")
        st.caption("数据：东财个股新闻 + 财联社 + 东财全球/金十/股吧/热榜等扩展源 | "
                   "打分：中文金融词典 | 研究：事件研究（沪深300超额）")
        stats = load_news_stats()
        if not stats or stats.get("total", 0) == 0:
            st.warning("暂无舆情数据。先运行：cd ~/quant/sentiment-mvp && "
                       "python run_pipeline.py daily")
        else:
            st.markdown("#### 数据概览")
            n1, n2, n3, n4 = st.columns(4)
            n1.metric("新闻总数", f"{stats['total']:,}")
            n2.metric("数据起始", (stats.get("min_time") or "-")[:10])
            n3.metric("数据截止", (stats.get("max_time") or "-")[:10])
            src = stats.get("by_source", {})
            src_label = {  # 数据库值 -> 看板显示名
                "em": "东财新闻", "cls": "财联社", "em_global": "东财全球",
                "jin10": "金十", "guba": "东财股吧", "cninfo": "巨潮",
                "irm": "互动易", "hot": "热榜",
            }
            if src:
                src_text = " / ".join(f"{src_label.get(k, k)} {v}" for k, v in sorted(src.items()))
            else:
                src_text = "-"
            n4.metric("来源", src_text)

            labels = stats.get("by_label", {})
            l1, l2, l3 = st.columns(3)
            l1.metric("正面", labels.get("positive", 0))
            l2.metric("中性", labels.get("neutral", 0))
            l3.metric("负面", labels.get("negative", 0))

            arts = load_news_articles()
            if len(arts):
                arts = arts.copy()
                arts["publish_time"] = arts["publish_time"].astype(str)

                st.markdown("#### 今日快照（全来源）")
                today = pd.Timestamp.now().strftime("%Y-%m-%d")
                today_df = arts[arts["publish_time"].str.startswith(today)].copy()
                if len(today_df):
                    src_options = ["全部来源"] + sorted(today_df["source"].unique())
                    sel_src = st.selectbox("来源", src_options, key="today_src")
                    if sel_src != "全部来源":
                        today_df = today_df[today_df["source"] == sel_src]
                    if len(today_df) == 0:
                        st.info("所选来源今天暂无新闻")
                        today_df_empty = True
                    else:
                        today_df_empty = False
                else:
                    today_df_empty = True
                if not today_df_empty:
                    uni_n = load_news_universe()
                    nm = dict(zip(uni_n["code"], uni_n["name"])) if len(uni_n) else {}
                    today_df["name"] = today_df["code"].map(nm)
                    snap = (today_df.groupby(["code", "name"])
                            .agg(n=("score", "size"), mean_score=("score", "mean"),
                                 positive=("label", lambda s: int((s == "positive").sum())),
                                 negative=("label", lambda s: int((s == "negative").sum())),
                                 sources=("source", lambda s: "/".join(sorted(set(s)))))
                            .sort_values("mean_score", ascending=False).reset_index())
                    if "url" in today_df.columns:
                        latest = (today_df.dropna(subset=["url"])
                                  .sort_values("publish_time", ascending=False)
                                  .drop_duplicates("code", keep="first")
                                  .set_index("code"))
                        snap["最新标题"] = snap["code"].map(latest["title"]).fillna("")
                        snap["原文链接"] = snap["code"].map(latest["url"]).fillna("")
                        show_table(
                            snap,
                            column_config={
                                "最新标题": st.column_config.TextColumn("最新标题"),
                                "原文链接": st.column_config.LinkColumn("原文链接",
                                                                       display_text="🔗 原文"),
                            },
                        )
                        st.caption("「原文链接」为该股今天最新一条新闻；当天多条新闻只展示最新一条链接。")
                    else:
                        show_table(snap)
                else:
                    st.info("今日暂无匹配标的的新闻")

                st.markdown("#### 情绪分布")
                d1, d2 = st.columns(2)
                with d1:
                    fig = px.histogram(arts, x="score", nbins=40,
                                       color_discrete_sequence=[ACCENT])
                    style_chart(fig, height=300, title="情绪分直方图")
                    st.plotly_chart(fig, use_container_width=True)
                with d2:
                    lab = arts.groupby(["source", "label"]).size().reset_index(name="n")
                    fig2 = px.bar(lab, x="label", y="n", color="source", barmode="group",
                                  color_discrete_sequence=[ACCENT, NEG_COLOR, POS_COLOR, MUTED])
                    style_chart(fig2, height=300, title="标签 × 来源")
                    st.plotly_chart(fig2, use_container_width=True)

                st.markdown("#### 事件研究（沪深300超额，次日开盘买入）")
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
                            show_table(pivot.map(lambda x: f"{x:.4f}"))
                        with t2:
                            fig3 = px.bar(summ, x="持有期", y="平均超额", color="情绪桶",
                                          barmode="group",
                                          color_discrete_map={"positive": POS_COLOR,
                                                              "neutral": MUTED,
                                                              "negative": ACCENT})
                            fig3.add_hline(y=0, line_color="#666", line_dash="dash")
                            style_chart(fig3, height=320)
                            fig3.update_layout(legend_title="情绪桶")
                            st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.info("暂无事件研究结果，运行 daily 后生成")

                arts["date"] = pd.to_datetime(arts["publish_time"], errors="coerce").dt.date
                trend = arts.dropna(subset=["date"]).groupby("date").size().reset_index(name="n")
                if len(trend):
                    fig4 = px.bar(trend, x="date", y="n",
                                  color_discrete_sequence=[ACCENT])
                    style_chart(fig4, height=280, title="每日新闻条数")
                    st.plotly_chart(fig4, use_container_width=True)

                st.markdown("#### 情绪最强 / 最弱新闻")
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
                    show_table(top)
                with e2:
                    st.markdown("**最负面**")
                    show_table(bot)
            else:
                st.info("暂无文章数据")

    # ---------------- 数据状态 ----------------
    elif page == "data":
        st.markdown("### 数据状态")
        status = data_status()
        rows = []
        for key, src in status.items():
            if key == "meta":
                continue
            for label, s in src.items():
                desc = s.get("desc", "")
                info = s.get("info") or {}
                if info.get("last_date"):
                    desc += f" · 截至 {info['last_date']}"
                if info.get("n_rows") is not None:
                    desc += f" · 约 {info['n_rows']:,} 行"
                rows.append({
                    "数据": s.get("name", key),
                    "位置": s.get("path", label),
                    "状态": "存在" if s.get("exists") else "缺失",
                    "大小MB": s.get("size_mb"),
                    "来源/更新": f"{s.get('source','')} · {s.get('update','')}",
                    "说明": desc,
                })
        show_table(pd.DataFrame(rows))
        meta = status.get("meta", {})
        if meta:
            st.markdown(f"- 上次刷新：{meta.get('last_update', '-')}")
            st.markdown(f"- 数据截至：{meta.get('end', str(last_date.date()))} · "
                        f"代码数 {meta.get('n_codes', '-')} · 行数 {meta.get('n_rows', '-'):,}")
        st.markdown(f"- 当前加载：面板行数 {len(panel):,}，个股 {panel['code'].nunique()}，"
                    f"日期 {first_date.date()} ~ {last_date.date()}")

        st.markdown("#### 更新说明")
        st.markdown("""
- **开始更新 / `refresh_data.py`**：刷新股票池、指数、行业分类、股票日线、ETF、场外基金净值，并同步 PostgreSQL `stock_daily`、重建基金衍生面板。
- **舆情数据**由 `~/quant/sentiment-mvp/run_pipeline.py daily` 独立更新，不跟随一键更新。
- 状态只表示文件/表是否存在；数据新旧以「上次刷新 / 数据截至」为准。
""")

        st.markdown("#### 舆情数据状态")
        st.caption("舆情由 `~/quant/sentiment-mvp/run_pipeline.py daily` 更新，"
                   "不跟随上面的股票数据一键更新。")
        sent_stats = load_news_stats()
        if sent_stats:
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("文章总数", f"{sent_stats.get('total', 0):,}")
            s2.metric("数据起始", (sent_stats.get("min_time") or "-")[:10])
            s3.metric("数据截止", (sent_stats.get("max_time") or "-")[:10])
            src = sent_stats.get("by_source", {})
            s4.metric("来源数", f"{len(src)} 类")
            if src:
                st.markdown("来源：" + " · ".join(f"{k}={v:,}" for k, v in sorted(src.items())))
            lbl = sent_stats.get("by_label", {})
            if lbl:
                st.markdown("标签：" + " · ".join(f"{k}={v:,}" for k, v in sorted(lbl.items())))

        st.markdown("---")
        st.markdown("### 数据更新（腾讯行情 + 中证指数官网）")
        with st.expander("当前数据源", expanded=False):
            st.markdown("""
- **中证指数官网（OSS）**：股票池 = 沪深300 + 中证500 + 中证1000 成分股
- **腾讯行情**：个股日线（前复权）+ 指数日线（沪深300/中证500/中证1000/创业板指/科创50/上证指数）
- **东方财富（akshare）**：行业分类（电子/计算机/通信/传媒），接口不可达时自动回退本地缓存
""")
        with st.container(border=True):
            u1, u2 = st.columns(2)
            with u1:
                mode = st.selectbox("更新模式", ["增量（推荐，只抓新增区间）", "全量重建"],
                                    key="data_mode")
            with u2:
                update_end = st.date_input("更新到", value=pd.Timestamp.today().date(),
                                           key="data_end")
            st.caption("首次全量约需几分钟；之后每天增量约 1-2 分钟。东方财富行业接口在部分"
                       "服务器不可达，行业分类失败时自动沿用本地缓存。")
            if st.button("开始更新", type="primary", key="data_update"):
                bar = st.progress(0.0, text="准备中...")
                try:
                    result = refresh_all(
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
