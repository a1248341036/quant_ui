# -*- coding: utf-8 -*-
"""事件研究：新闻情绪分桶 → 前瞻收益（收盘/次日开盘两种口径，含超额 vs 沪深300）。

额外研究口径：
- 加权聚合：来源权重（财联社 > 东方财富）+ 新闻时效指数衰减
- 领先/滞后检验：情绪 vs 同日/滞后/领先收益的相关性，验证是否只是"价格镜像"
- 成交量过滤：事件日成交量相对过去 20 日中位数，过滤无人交易的假事件
- 行业拆分：按行业分别汇总多空价差
- 重叠事件窗口：Newey-West 修正（按股票聚簇 + 时间重叠），避免 t 值虚高
"""

import math
from datetime import time as dtime

import numpy as np
import pandas as pd

from .fetch_news import parse_time


def frame_from_df(df):
    """DataFrame(date/open/close[,volume]) -> indexed by date 的 frame。"""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.drop_duplicates("date").sort_values("date").set_index("date")
    cols = ["open", "close"] + (["volume"] if "volume" in df.columns else [])
    return df[cols].astype(float)


def index_frame_ffill(index_df):
    """index frame 按日重采样并前值填充，供查任意日期的 open/close。"""
    s = index_df.copy()
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
    return s.ffill()


def build_calendar(index_frame):
    return pd.DatetimeIndex(sorted(index_frame.index))


def event_trade_date(dt, calendar):
    """15:00 前发布的新闻视为当日收盘可知；之后/非交易日顺延到下一交易日。"""
    day = pd.Timestamp(dt.date())
    if dt.time() <= dtime(15, 0) and day in calendar:
        return day
    for d in calendar:
        if d > day:
            return d
    return None


def _decay_weight(dt, edate, half_life_hours):
    """新闻时效权重：距事件日 15:00（收盘可知点）越久衰减越快。

    w = exp(-age_hours / half_life_hours)，age<=0（当日 15:00 后发布、
    顺延到下一交易日）视为新消息，权重 1.0。
    """
    if half_life_hours is None or half_life_hours <= 0:
        return 1.0
    ref = pd.Timestamp(edate).normalize() + pd.Timedelta(hours=15)
    age_h = max(0.0, (ref - pd.Timestamp(dt)).total_seconds() / 3600.0)
    return float(math.exp(-age_h / half_life_hours))


def aggregate_stock_days(sent_df, pos_threshold=0.15, source_weights=None,
                         decay_half_life=None):
    """同股票同事件日聚合：来源权重 × 时效衰减后的加权平均分，按阈值分桶。

    返回带 mean_score（未加权平均，兼容旧口径）与 w_score（加权）两张表。
    """
    source_weights = source_weights or {}
    rows = []
    for (code, edate), g in sent_df.groupby(["code", "event_date"]):
        mean_score = float(g["score"].mean())
        if "dt" in g.columns:
            sw = g["source"].map(lambda s: float(source_weights.get(str(s), 1.0)))
            decay = g["dt"].map(lambda dt: _decay_weight(dt, edate, decay_half_life))
            w = (sw * decay).fillna(1.0)
            w_score = float((g["score"] * w).sum() / w.sum()) if w.sum() > 0 else mean_score
            eff_n = float(w.sum())
        else:
            w_score, eff_n = mean_score, float(len(g))
        label = "positive" if w_score > pos_threshold else ("negative" if w_score < -pos_threshold else "neutral")
        rows.append(
            {
                "code": code,
                "event_date": edate,
                "label": label,
                "n_articles": len(g),
                "mean_score": round(mean_score, 4),
                "w_score": round(w_score, 4),
                "eff_n": round(eff_n, 2),
                "pos_ratio": round(float((g["score"] > 0).mean()), 4),
                "source": (g["source"].mode().iloc[0] if "source" in g and len(g["source"].mode()) else "em"),
            }
        )
    return pd.DataFrame(rows)


def _forward_row(frame, idx, pos, h):
    """计算第 pos 个交易日起 h 日后的收盘/开盘买入收益与超额。"""
    j = pos + h
    n = len(frame)
    if j >= n:
        return {}
    base_close = float(frame["close"].iloc[pos])
    base_idx_close = float(idx["close"].get(frame.index[pos], np.nan))
    exit_close = float(frame["close"].iloc[j])
    exit_idx_close = float(idx["close"].get(frame.index[j], np.nan))
    row = {}
    if not (math.isnan(base_idx_close) or math.isnan(exit_idx_close)):
        row[f"ret_{h}"] = exit_close / base_close - 1.0
        row[f"excess_{h}"] = row[f"ret_{h}"] - (exit_idx_close / base_idx_close - 1.0)
    else:
        row[f"ret_{h}"] = np.nan
        row[f"excess_{h}"] = np.nan

    # 次日开盘买入口径：事件日收盘可知，次一交易日开盘入场
    if pos + 1 < n:
        entry_open = float(frame["open"].iloc[pos + 1])
        entry_idx_open = float(idx["open"].get(frame.index[pos + 1], np.nan))
        if entry_open > 0 and not math.isnan(entry_idx_open):
            row[f"ret_open_{h}"] = exit_close / entry_open - 1.0
            row[f"excess_open_{h}"] = row[f"ret_open_{h}"] - (exit_idx_close / entry_idx_open - 1.0)
        else:
            row[f"ret_open_{h}"] = np.nan
            row[f"excess_open_{h}"] = np.nan
    else:
        row[f"ret_open_{h}"] = np.nan
        row[f"excess_open_{h}"] = np.nan
    return row


def compute_events(day_df, prices, index_df, horizons):
    """prices: {code: frame(date/open/close[,volume])}；index_df: frame(date/open/close)。"""
    idx = index_frame_ffill(index_df)
    records = []
    for (code, edate), g in day_df.groupby(["code", "event_date"]):
        frame = prices.get(code)
        if frame is None or edate not in frame.index:
            continue
        pos = frame.index.get_loc(edate)
        row = {
            "code": code,
            "event_date": edate,
            "label": g["label"].iloc[0],
            "n_articles": int(g["n_articles"].iloc[0]),
            "mean_score": float(g["mean_score"].iloc[0]),
            "w_score": float(g["w_score"].iloc[0]) if "w_score" in g else float(g["mean_score"].iloc[0]),
            "source": g["source"].iloc[0],
        }
        for h in horizons:
            row.update(_forward_row(frame, idx, pos, h))
        records.append(row)
    return pd.DataFrame(records)


def compute_baseline(prices, index_df, event_days, horizons, lead_days=10):
    """每个股票样本窗口内非事件日的 h 日前瞻收益/超额。"""
    idx = index_frame_ffill(index_df)
    min_d = min((d for _, d in event_days), default=None)
    max_d = max((d for _, d in event_days), default=None)
    if min_d is None:
        return pd.DataFrame()
    cal = build_calendar(index_df)
    start_cal = min_d - pd.Timedelta(days=lead_days * 2)
    start = cal[max(0, cal.searchsorted(start_cal, side="left") - lead_days)]
    records = []
    for code, frame in prices.items():
        window = frame.index[(frame.index >= start) & (frame.index <= max_d)]
        events_this = {d for c, d in event_days if c == code}
        for pos in range(len(window)):
            edate = window[pos]
            if edate in events_this:
                continue
            row = {"code": code, "event_date": edate}
            for h in horizons:
                row.update(_forward_row(frame, idx, pos, h))
            records.append(row)
    return pd.DataFrame(records)


def _newey_west_se(values, dates, clusters, lags):
    """按股票聚簇 + 时间重叠的 Newey-West 标准误（仅含常数项的回归）。

    事件日间隔小于持有期时，前瞻收益互相重叠，普通 t 值会虚高；
    这里在每只股票内部按日期排序后加入滞后协方差项。
    """
    df = pd.DataFrame({"v": pd.to_numeric(values, errors="coerce"),
                       "d": dates, "c": clusters}).dropna()
    if len(df) < 2:
        return np.nan
    n = len(df)
    grand_mean = float(df["v"].mean())
    ss = 0.0
    m = int(df["c"].nunique())
    for _, g in df.groupby("c", sort=False):
        g = g.sort_values("d")
        ee = g["v"].values - grand_mean
        s0 = float(np.sum(ee ** 2))
        for l in range(1, lags + 1):
            if len(ee) <= l:
                continue
            sl = float(np.sum(ee[l:] * ee[:-l]))
            s0 += 2.0 * (1.0 - l / (lags + 1.0)) * sl
        ss += s0
    k = 1
    adj = (n - 1) / (n - k) * (m / (m - 1)) if m > 1 else (n - 1) / (n - k)
    var = adj * ss / (n * n)
    return float(math.sqrt(max(var, 0.0)))


def _summarize(s, name, dates=None, clusters=None, lags=0):
    s = pd.to_numeric(s, errors="coerce").dropna()
    n = int(len(s))
    if n == 0:
        return {"name": name, "n": 0, "mean": np.nan, "median": np.nan, "hit": np.nan, "t": np.nan, "se": np.nan}
    mean = float(s.mean())
    med = float(s.median())
    hit = float((s > 0).mean())
    if lags and dates is not None and clusters is not None and len(s) == len(dates):
        se = _newey_west_se(s, dates, clusters, lags)
    else:
        se = float(s.std(ddof=1) / math.sqrt(n)) if n > 1 else np.nan
    t = mean / se if se and not math.isnan(se) else 0.0
    return {"name": name, "n": n, "mean": mean, "median": med, "hit": hit, "t": t, "se": se}


def diff_t(a, b):
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    va = float(a.var(ddof=1)) / len(a)
    vb = float(b.var(ddof=1)) / len(b)
    se = math.sqrt(va + vb)
    if se == 0:
        return np.nan
    return (float(a.mean()) - float(b.mean())) / se


def _diff_t_nw(a_df, b_df, lags):
    """两组事件收益的均值差 t 值：每组用 Newey-West 标准误（保守合成）。"""
    if len(a_df) < 2 or len(b_df) < 2:
        return np.nan
    se_a = _newey_west_se(a_df["v"], a_df["d"], a_df["c"], lags)
    se_b = _newey_west_se(b_df["v"], b_df["d"], b_df["c"], lags)
    if se_a is None or se_b is None or math.isnan(se_a) or math.isnan(se_b):
        return np.nan
    se = math.sqrt(se_a ** 2 + se_b ** 2)
    if se == 0:
        return np.nan
    return (float(a_df["v"].mean()) - float(b_df["v"].mean())) / se


def _event_subset(events, col, label):
    sub = events[events["label"] == label][["code", "event_date", col]].dropna()
    sub = sub.rename(columns={col: "v", "event_date": "d", "code": "c"})
    return sub


def summarize_by_label(events, baseline, horizons, mode="close", autocorr=True):
    rows = []
    for h in horizons:
        col = f"excess_{h}" if mode == "close" else f"excess_open_{h}"
        lags = (h - 1) if autocorr else 0
        base = _summarize(baseline[col] if len(baseline) else pd.Series(dtype=float),
                          "baseline")
        base_df = (baseline[["code", "event_date", col]].dropna()
                   .rename(columns={col: "v", "event_date": "d", "code": "c"})
                   if len(baseline) else pd.DataFrame())
        for label in ("positive", "neutral", "negative"):
            sub = events[events["label"] == label][col] if len(events) else pd.Series(dtype=float)
            sub_df = _event_subset(events, col, label)
            st = _summarize(sub, label,
                            dates=sub_df["d"] if len(sub_df) else None,
                            clusters=sub_df["c"] if len(sub_df) else None,
                            lags=lags)
            st["horizon"] = h
            st["base_mean"] = base["mean"]
            st["base_n"] = base["n"]
            if len(base_df):
                st["diff_t"] = _diff_t_nw(sub_df, base_df, lags)
            else:
                st["diff_t"] = np.nan
            st["n_stocks"] = int(events[events["label"] == label]["code"].nunique()) if len(events) else 0
            rows.append(st)
    return pd.DataFrame(rows)


def _table_md(summary):
    show = summary[["horizon", "name", "n", "n_stocks", "mean", "median", "hit", "t", "base_mean", "diff_t"]].copy()
    show.columns = ["持有期", "情绪桶", "事件数", "涉及股票", "均值", "中位数", "胜率", "t值", "基线均值", "差值t值"]
    return show.to_markdown(index=False, floatfmt=".4f")


def _spread_lines(events, summary, horizons, mode="close", autocorr=True):
    col = "excess" if mode == "close" else "excess_open"
    lines = []
    for h in horizons:
        p = summary[(summary["horizon"] == h) & (summary["name"] == "positive")]
        n = summary[(summary["horizon"] == h) & (summary["name"] == "negative")]
        if len(p) and len(n) and p["n"].iloc[0] and n["n"].iloc[0]:
            spread = p["mean"].iloc[0] - n["mean"].iloc[0]
            lags = (h - 1) if autocorr else 0
            a_df = _event_subset(events, f"{col}_{h}", "positive")
            b_df = _event_subset(events, f"{col}_{h}", "negative")
            spread_t = _diff_t_nw(a_df, b_df, lags) if len(a_df) and len(b_df) else diff_t(
                events[events["label"] == "positive"][f"{col}_{h}"],
                events[events["label"] == "negative"][f"{col}_{h}"],
            )
            lines.append(f"- 持有 {h} 日：价差 {spread:.4f}（t={spread_t:.2f}）")
    return lines


def _corr_pvalue(r, n):
    if abs(r) >= 1.0 or n < 3:
        return np.nan
    try:
        from scipy import stats
        t = r * math.sqrt((n - 2) / (1 - r * r))
        return float(2.0 * stats.t.sf(abs(t), n - 2))
    except Exception:
        z = abs(r) * math.sqrt(n - 1)
        return float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))))


def compute_leadlag(day_df, prices, horizons):
    """逐事件输出情绪分 + 同日/滞后/领先收益；返回 (明细, 相关性汇总)。

    相关性使用按股票去均值（截面中性）后的 Pearson r：
    - ret_same：情绪与当日收益（若显著高，提示"价格镜像"风险）
    - ret_lag_h：情绪与过去 h 日收益（情绪是否在追涨杀跌）
    - ret_lead_h：情绪对未来 h 日收益（是否真有领先性）
    """
    score_col = "w_score" if "w_score" in day_df.columns else "mean_score"
    max_lag = max(horizons)
    recs = []
    for (code, edate), g in day_df.groupby(["code", "event_date"]):
        frame = prices.get(code)
        if frame is None or edate not in frame.index:
            continue
        pos = frame.index.get_loc(edate)
        row = {"code": code, "event_date": edate, "score": float(g[score_col].iloc[0])}
        close_pos = float(frame["close"].iloc[pos])
        row["ret_same"] = (float(frame["close"].iloc[pos + 1]) / close_pos - 1.0
                           if pos + 1 < len(frame) and close_pos else np.nan)
        for lag in range(1, max_lag + 1):
            j = pos - lag
            if j >= 0 and close_pos and float(frame["close"].iloc[j]):
                row[f"ret_lag_{lag}"] = close_pos / float(frame["close"].iloc[j]) - 1.0
            else:
                row[f"ret_lag_{lag}"] = np.nan
        for h in horizons:
            j = pos + h
            if j < len(frame) and close_pos:
                row[f"ret_lead_{h}"] = float(frame["close"].iloc[j]) / close_pos - 1.0
            else:
                row[f"ret_lead_{h}"] = np.nan
        recs.append(row)
    df = pd.DataFrame(recs)
    if df.empty:
        return df, pd.DataFrame()
    corr_rows = []
    cols = ["ret_same"] + [f"ret_lag_{h}" for h in horizons] + [f"ret_lead_{h}" for h in horizons]
    for col in cols:
        tmp = df[["code", "score", col]].dropna()
        if len(tmp) < 10:
            continue
        d = tmp.groupby("code")[["score", col]].transform(lambda x: x - x.mean())
        r = float(d["score"].corr(d[col]))
        corr_rows.append({"type": col, "n": len(tmp), "r": round(r, 4),
                          "p": round(_corr_pvalue(r, len(tmp)), 4)})
    return df, pd.DataFrame(corr_rows)


def apply_volume_filter(events, prices, window=20, min_vol_ratio=1.0):
    """事件日成交量相对过去 window 日中位数的倍数；低于阈值的事件剔除。"""
    rows = []
    for _, ev in events.iterrows():
        frame = prices.get(ev["code"])
        d = {**ev.to_dict()}
        if frame is None or "volume" not in frame.columns or ev["event_date"] not in frame.index:
            d["vol_ratio"] = np.nan
        else:
            pos = frame.index.get_loc(ev["event_date"])
            vol = float(frame["volume"].iloc[pos])
            base = frame["volume"].iloc[max(0, pos - window):pos]
            med = float(base.median()) if len(base) else np.nan
            d["vol_ratio"] = (vol / med if med and med > 0 else np.nan)
        rows.append(d)
    out = pd.DataFrame(rows)
    if len(out):
        out = out[out["vol_ratio"].isna() | (out["vol_ratio"] >= min_vol_ratio)].reset_index(drop=True)
    return out


def add_industry(events, industry_map):
    """events 加 industry 列；无法映射的归"其他"。"""
    events = events.copy()
    events["industry"] = events["code"].map(industry_map).fillna("其他")
    return events


def summarize_by_industry(events, horizons, mode="close", autocorr=True):
    """按行业分桶汇总：每个行业 × 持有期 × 情绪桶的事件数与超额收益均值。"""
    if not len(events) or "industry" not in events.columns:
        return pd.DataFrame()
    col = "excess" if mode == "close" else "excess_open"
    rows = []
    for ind, g in events.groupby("industry"):
        for h in horizons:
            lags = (h - 1) if autocorr else 0
            for label in ("positive", "neutral", "negative"):
                sub_df = _event_subset(g, f"{col}_{h}", label)
                s = _summarize(sub_df["v"] if len(sub_df) else pd.Series(dtype=float), label,
                               dates=sub_df["d"] if len(sub_df) else None,
                               clusters=sub_df["c"] if len(sub_df) else None,
                               lags=lags)
                rows.append({"industry": ind, "horizon": h, "name": label,
                             "n": s["n"], "mean": s["mean"], "hit": s["hit"], "t": s["t"]})
    return pd.DataFrame(rows)


def build_report(events, baseline, summary_close, summary_open, sent_df, cfg,
                 leadlag_corr=None, industry_close=None, industry_open=None,
                 vol_filter_note=""):
    lines = []
    lines.append("# A股舆情情绪事件研究报告（MVP）\n")
    lines.append(f"- 生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M}")
    lines.append(f"- 新闻条数：{len(sent_df)}")
    lines.append(f"- 股票-事件日数：{len(events)}")
    if len(events):
        lines.append(f"- 事件区间：{events['event_date'].min():%Y-%m-%d} ~ {events['event_date'].max():%Y-%m-%d}")
    if len(events) and "source" in events:
        src = events.groupby("source")["event_date"].count().to_dict()
        lines.append(f"- 数据源分布：{' / '.join(f'{k}:{v}' for k, v in src.items())}")
    lines.append(f"- 持有期：{'/'.join(str(h) for h in cfg['event_study']['horizons'])} 个交易日")
    lines.append("- 口径A（收盘）：事件日收盘买入 → h日后收盘卖出")
    lines.append("- 口径B（次日开盘）：事件日收盘可知 → 次一交易日开盘买入 → h日后收盘卖出（更贴近实盘）")
    lines.append("- 超额基准：沪深300 同窗口收益")
    if cfg.get("event_study", {}).get("autocorr", {}).get("enabled"):
        lines.append("- 自相关修正：t值采用按股票聚簇的 Newey-West 标准误（处理重叠事件窗口）")
    if vol_filter_note:
        lines.append(f"- 成交量过滤：{vol_filter_note}")
    lines.append("")

    lines.append("## 一、样本分布\n")
    if len(events):
        dist = events.groupby("label").agg(n=("code", "size"), stocks=("code", "nunique"),
                                           mean_score=("mean_score", "mean"),
                                           w_score=("w_score", "mean") if "w_score" in events else ("mean_score", "mean"))
        lines.append(dist.to_markdown())
        lines.append("")

    if leadlag_corr is not None and len(leadlag_corr):
        lines.append("## 二、领先/滞后检验（截面去均值后的相关性）\n")
        ll = leadlag_corr.copy()
        ll["type"] = ll["type"].map({
            "ret_same": "同日收益", **{f"ret_lag_{h}": f"滞后{h}日收益" for h in cfg["event_study"]["horizons"]},
            **{f"ret_lead_{h}": f"领先{h}日收益" for h in cfg["event_study"]["horizons"]},
        })
        lines.append(ll.to_markdown(index=False, floatfmt=".4f"))
        lines.append("")
        lines.append("说明：r 为按股票去均值后的 Pearson 相关；同日显著为正是'价格镜像'信号，"
                     "领先收益相关为正才说明情绪有预测力。\n")

    sec = 3
    lines.append(f"## {sec}、口径A：收盘买入前瞻超额收益（vs 沪深300）\n")
    sec += 1
    if len(summary_close):
        lines.append(_table_md(summary_close))
        lines.append("")
        lines.append("说明：t值为桶内超额收益 vs 0 的检验；差值t值为桶均值 vs 同窗口全部非事件日基线均值的差异检验。\n")

    lines.append(f"## {sec}、口径B：次日开盘买入前瞻超额收益（vs 沪深300）\n")
    sec += 1
    if len(summary_open):
        lines.append(_table_md(summary_open))
        lines.append("")

    lines.append(f"## {sec}、多空价差（positive - negative）\n")
    sec += 1
    lines.append("### 口径A：收盘买入\n")
    lines.extend(_spread_lines(events, summary_close, cfg["event_study"]["horizons"], "close",
                               bool(cfg.get("event_study", {}).get("autocorr", {}).get("enabled"))))
    lines.append("\n### 口径B：次日开盘买入\n")
    lines.extend(_spread_lines(events, summary_open, cfg["event_study"]["horizons"], "open",
                               bool(cfg.get("event_study", {}).get("autocorr", {}).get("enabled"))))

    if industry_close is not None and len(industry_close):
        lines.append(f"\n## {sec}、分行业多空价差（口径A：收盘买入，持有1日）\n")
        sec += 1
        ind1 = industry_close[industry_close["horizon"] == 1]
        pivot = ind1.pivot_table(index="industry", columns="name", values="mean", aggfunc="first")
        pivot["多空价差"] = pivot.get("positive", 0) - pivot.get("negative", 0)
        pivot["事件数"] = ind1.groupby("industry")["n"].sum()
        lines.append(pivot.round(4).to_markdown())
        lines.append("")

    lines.append(f"\n## {sec}、情绪最强/最弱事件示例\n")
    sec += 1
    if len(events):
        top = events.nlargest(5, "mean_score")
        bot = events.nsmallest(5, "mean_score")
        ex = pd.concat([top, bot])
        merged = ex.merge(
            sent_df.drop_duplicates("event_key")[["code", "event_date", "title", "publish_time", "source"]],
            on=["code", "event_date"], how="left"
        )
        for _, r in merged.iterrows():
            lines.append(f"- [{r['code']} {r['event_date']:%Y-%m-%d} {r.get('source', '')}] {r['label']} score={r['mean_score']:.2f} n={r['n_articles']} | {r['title']}")

    lines.append(f"\n## {sec}、方法与局限\n")
    lines.append("- 情绪标签：中文金融词典打分（确定性方法），单篇正负词净计数。")
    lines.append("- 时点假设：15:00 前新闻视为当日收盘可知，之后/非交易日顺延到下一交易日。")
    sw = cfg.get("event_study", {}).get("source_weights", {})
    hl = cfg.get("event_study", {}).get("decay_half_life_hours")
    lines.append(f"- 聚合加权：来源权重 {sw}；新闻时效半衰期 {hl} 小时。")
    if cfg.get("event_study", {}).get("autocorr", {}).get("enabled"):
        lines.append("- 重叠事件窗口：t值已做按股票聚簇的 Newey-West 修正。")
    else:
        lines.append("- 局限：事件窗口重叠未做自相关修正。")
    lines.append("- 局限：样本仅覆盖东方财富搜索可回溯的约最近1个月新闻；词典方法会漏掉反讽/隐含情绪。")
    lines.append("- 下一步：接入更多源（财联社今日快照已可采集）、用知识截止匹配的模型打标。\n")
    return "\n".join(lines)
