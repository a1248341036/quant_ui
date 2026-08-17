#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""舆情/新闻情绪策略回测（基于 sentiment-mvp 文章库 + 本地行情面板）。

调研要点落地方案：
  1. 舆情动量     SENTI/SENATT 思路：买入近期新闻情绪最正面的一批（ascending=False）
  2. 舆情反转     A股反转 + 情绪：买入近期情绪最负面的一批（ascending=True）
  3. 舆情关注度   注意力因子：买入近期新闻条数最多的一批（ascending=False）
  4. 舆情+反转增强 媒体增强效应(MRE)：买入「低动量 + 低情绪」的组合（ascending=True）

注意：本地舆情库仅覆盖 2026-02 起、top30 成交额股票，窗口短、股票少，
结果只能作为方法论验证，不能外推。
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.engine import run_backtest  # noqa: E402
from core.metrics import compute_metrics  # noqa: E402
from strategies.registry import get_strategy, list_strategies  # noqa: E402

SENT_DB = ROOT.parent / "sentiment-mvp" / "data" / "articles.db"
PANEL = ROOT / "data/panel.parquet"
INDEX = ROOT / "data/index.csv"


def load_sentiment() -> pd.DataFrame:
    conn = sqlite3.connect(str(SENT_DB))
    try:
        df = pd.read_sql_query(
            "SELECT code, publish_time, source, title, label, score FROM articles",
            conn,
        )
    finally:
        conn.close()
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["pt"] = pd.to_datetime(df["publish_time"], errors="coerce")
    # 无时间的热榜/快照行不参与情绪打分因子，仅参与条数统计
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
    return df


def build_sentiment_frames(close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """按引擎交易日历对齐，构造情绪因子矩阵。

    对每个 code：文章按自然日聚合日平均分/条数 → 重排到交易日历 →
    过去 20 个交易日的滚动均值（min_periods=1），信号日当天可用、
    次日开盘执行，无前视。
    """
    art = load_sentiment()
    art["d"] = art["pt"].dt.normalize()
    cal = close.index
    codes = close.columns
    daily = (art.assign(s=art["score"], c=1)
             .groupby(["code", "d"]).agg(s=("s", "mean"), c=("c", "sum"))
             .reset_index())
    sent = pd.DataFrame(index=cal, columns=codes, dtype=float)
    cnt = pd.DataFrame(index=cal, columns=codes, dtype=float)
    for code in codes:
        sub = daily[daily["code"] == code].set_index("d")
        s = sub["s"].reindex(cal)
        c = sub["c"].reindex(cal).fillna(0.0)
        sent[code] = s.rolling(20, min_periods=1).mean()
        cnt[code] = c.rolling(20, min_periods=1).sum()
    return {"sent20": sent, "cnt20": cnt}


def zscore_frame(df: pd.DataFrame) -> pd.DataFrame:
    z = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)
    return z.replace([np.inf, -np.inf], np.nan)


def sentiment_factor_builder(close, am20, turn20):
    """factor_builder 接口：close/am20/turn20 是引擎切好的子区间矩阵。"""
    frames = build_sentiment_frames(close)
    mom20 = close.pct_change(20, fill_method=None)
    mre = zscore_frame(frames["sent20"]) + zscore_frame(mom20)
    frames["mre"] = mre
    return frames


def main():
    panel = pd.read_parquet(PANEL)
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    uni = pd.read_csv(ROOT / "data/universe.csv", dtype={"code": str})
    uni["code"] = uni["code"].astype(str).str.zfill(6)
    index = pd.read_csv(INDEX)
    index["date"] = pd.to_datetime(index["date"])

    art = load_sentiment()
    sent_codes = sorted(set(art["code"]))
    print(f"舆情覆盖 {len(sent_codes)} 只：{art['pt'].min().date()} ~ {art['pt'].max().date()}")
    print(f"面板区间 {panel['date'].min().date()} ~ {panel['date'].max().date()}")

    start, end = "2026-03-01", str(panel["date"].max().date())
    capital, top_n = 50000.0, 3
    freq = "monthly"
    print(f"回测 {start} ~ {end} · {len(sent_codes)} 只 · Top{top_n} · {freq}")

    rows = []
    ic_rows = []
    sent_strategies = [
        ("舆情正面(动量)", "sent20", False, "买入近期新闻情绪最正面"),
        ("舆情负面(反转)", "sent20", True, "买入近期情绪最负面（博反转）"),
        ("舆情关注度", "cnt20", False, "买入近期新闻条数最多"),
        ("舆情+反转增强(MRE)", "mre", True, "买入低动量+低情绪组合"),
    ]
    for name, factor, asc, desc in sent_strategies:
        res = run_backtest(
            panel=panel, codes=sent_codes, factor=factor, ascending=asc,
            start=start, end=end, capital=capital, top_n=top_n, freq=freq,
            affordable=True, amount_q=0.2, warmup_days=45,
            factor_builder=sentiment_factor_builder,
            analyze=True,
        )
        m = res["metrics"]
        q = res.get("factor_quality")
        rows.append({"策略": name, "类型": "舆情", "说明": desc,
                     "总收益%": round(m["总收益"] * 100, 2),
                     "年化%": round(m["年化收益"] * 100, 2),
                     "夏普": round(m["夏普"], 2),
                     "最大回撤%": round(m["最大回撤"] * 100, 2),
                     "信号日": str(res["last_signal_date"].date()) if res["last_signal_date"] else "-"})
        if q:
            sign = -1.0 if asc else 1.0
            ic_rows.append({"策略": name, "类型": "舆情",
                            "IC均值": q["ic"]["mean_ic"], "ICIR": q["ic"]["icir"],
                            "t值": q["ic"]["t_stat"], "IC>0占比": q["ic"]["win_rate"],
                            "多空价差%": q["group"]["spread"] * 100 if q["group"]["spread"] is not None else None,
                            "多空价差年化%": q["group"]["spread_pa"] * 100 if q["group"]["spread_pa"] is not None else None,
                            "方向调整IC": q["ic"]["mean_ic"] * sign if q["ic"]["mean_ic"] is not None else None,
                            "方向调整价差%": q["group"]["spread"] * sign * 100 if q["group"]["spread"] is not None else None})

    # 同股票池下对照：现有价格/量因子策略
    for name in ("动量 20 日", "反转 20 日", "低波动", "低换手冷门"):
        s = get_strategy(name)
        res = run_backtest(
            panel=panel, codes=sent_codes, factor=s["factor"], ascending=s["ascending"],
            start=start, end=end, capital=capital, top_n=top_n, freq=freq,
            affordable=True, amount_q=0.2, warmup_days=45,
            analyze=True,
        )
        m = res["metrics"]
        q = res.get("factor_quality")
        rows.append({"策略": name, "类型": "对照", "说明": s["desc"],
                     "总收益%": round(m["总收益"] * 100, 2),
                     "年化%": round(m["年化收益"] * 100, 2),
                     "夏普": round(m["夏普"], 2),
                     "最大回撤%": round(m["最大回撤"] * 100, 2),
                     "信号日": str(res["last_signal_date"].date()) if res["last_signal_date"] else "-"})
        if q:
            sign = -1.0 if s["ascending"] else 1.0
            ic_rows.append({"策略": name, "类型": "对照",
                            "IC均值": q["ic"]["mean_ic"], "ICIR": q["ic"]["icir"],
                            "t值": q["ic"]["t_stat"], "IC>0占比": q["ic"]["win_rate"],
                            "多空价差%": q["group"]["spread"] * 100 if q["group"]["spread"] is not None else None,
                            "多空价差年化%": q["group"]["spread_pa"] * 100 if q["group"]["spread_pa"] is not None else None,
                            "方向调整IC": q["ic"]["mean_ic"] * sign if q["ic"]["mean_ic"] is not None else None,
                            "方向调整价差%": q["group"]["spread"] * sign * 100 if q["group"]["spread"] is not None else None})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    out = ROOT / "results" / "sentiment_backtest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"saved -> {out}")
    if ic_rows:
        ic_df = pd.DataFrame(ic_rows)
        ic_out = ROOT / "results" / "sentiment_ic_group.csv"
        ic_df.to_csv(ic_out, index=False, encoding="utf-8-sig")
        print(f"saved -> {ic_out}")


if __name__ == "__main__":
    main()
