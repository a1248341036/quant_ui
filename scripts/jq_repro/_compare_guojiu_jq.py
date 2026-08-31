# -*- coding: utf-8 -*-
"""国九条 对齐对比: 本引擎 vs 聚宽官方回测(用户导出的每日收益 CSV)。

聚宽 CSV: C:/Users/zhoubw/Downloads/result_1 (1).csv (gbk)
窗口 2025-01-02 ~ 2026-08-28, 本金 10万, JQ 结果 +11.98% / 基准 +32.08%。
输出: 双方指标对照 + 逐日累计差 + 分化最大日附近的成交对照。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

JQ_CSV = Path(r"C:\Users\zhoubw\Downloads\result_1 (1).csv")
START, END, CAPITAL = "2025-01-02", "2026-08-28", 100_000.0


def load_jq() -> pd.DataFrame:
    df = pd.read_csv(JQ_CSV, encoding="gbk")
    df["date"] = pd.to_datetime(df["时间"]).dt.strftime("%Y-%m-%d")
    return df.set_index("date")


def run_ours(code: str) -> dict:
    from core.event_engine.jq.entry import run_jq_backtest
    return run_jq_backtest(code, start=START, end=END, capital=CAPITAL)


def main() -> int:
    src = (ROOT / "scripts" / "jq_repro" / "_test_guojiu.py").read_text(
        encoding="utf-8")
    code = src.split("CODE = r'''", 1)[1].rsplit("'''", 1)[0]

    jq = load_jq()
    res = run_ours(code)

    ours = pd.Series({pd.Timestamp(p["date"]).strftime("%Y-%m-%d"): p["value"]
                      for p in res["nav"]}, name="ours_nav")
    comp = pd.DataFrame({
        "jq_strat": jq["策略收益"] / 100.0,
        "jq_bench": jq["基准收益"] / 100.0,
        "ours_nav": ours,
    }).dropna()
    comp["ours_strat"] = comp["ours_nav"] - 1.0
    comp["gap"] = (comp["ours_strat"] - comp["jq_strat"]) * 100

    print(f"对齐交易日: {len(comp)}  ({comp.index[0]} ~ {comp.index[-1]})")
    print(f"JQ  策略 {comp['jq_strat'].iloc[-1]:+.2%}   "
          f"我们 {comp['ours_strat'].iloc[-1]:+.2%}   "
          f"累计差 {comp['gap'].iloc[-1]:+.2f}pp")

    m = res["metrics"]
    print("我们的指标:", {k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in m.items()})

    print("\n-- 月末累计对照(策略% | 差pp) --")
    months = pd.Series([d[:7] for d in comp.index], index=comp.index)
    mends = comp.groupby(months).last()
    for d, row in mends.iterrows():
        print(f"  {d}  jq {row['jq_strat']*100:+7.2f}  "
              f"ours {row['ours_strat']*100:+7.2f}  gap {row['gap']:+6.2f}")

    daily_gap = comp["gap"].diff()
    top = daily_gap.abs().nlargest(8)
    print("\n-- 单日分化最大 8 日(我们-聚宽, 当日我们的成交) --")
    trades = res["trades"]
    tmap = {str(t["date"]): t for t in trades}
    for d in top.index:
        t = tmap.get(d)
        info = (f"买 {t['bought']} 卖 {t['sold']}" if t is not None else "(无成交)")
        print(f"  {d}  gap变化 {daily_gap[d]:+7.2f}pp  {info}")

    jdays = set(jq.index)
    tdays = set(tmap)
    only_ours = sorted(tdays - jdays)[:10]
    print(f"\n我们有成交而 JQ 当日无成交(前10): {only_ours}")
    print(f"我们成交笔数(事件) {len(trades)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
