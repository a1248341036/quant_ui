# -*- coding: utf-8 -*-
"""post47523 持仓级 diff: 聚宽成交明细 vs 本引擎逐笔成交。
定位首次分歧日期/标的, 并对照当日两边的完整成交。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

JQ_CSV = Path(r"C:\Users\zhoubw\Downloads\transaction\transaction.csv")
START, END, CAPITAL = "2025-01-02", "2026-06-01", 100_000.0


def load_jq_trades() -> pd.DataFrame:
    df = pd.read_csv(JQ_CSV, encoding="gbk")
    df["date"] = df["日期"].astype(str)
    df["code"] = df["标的"].str.extract(r"\((\d{6})\.")[0]
    df["side"] = df["交易类型"].map({"买": "buy", "卖": "sell"})
    df["shares"] = pd.to_numeric(
        df["成交数量"].str.replace("股", "", regex=False),
        errors="coerce").abs()
    df["price"] = pd.to_numeric(df["成交价"], errors="coerce")
    df["fee"] = pd.to_numeric(df["手续费"], errors="coerce")
    df["time"] = df["委托时间"].astype(str)
    df = df.dropna(subset=["code", "side", "shares", "price"])
    df = df[df["shares"] > 0]
    return df[["date", "time", "code", "side", "shares", "price", "fee"]]


def load_ours() -> pd.DataFrame:
    from core.event_engine.jq.entry import run_jq_backtest
    src = (ROOT / "scripts" / "jq_repro" / "_test_post47523.py").read_text(
        encoding="utf-8")
    code = src.split("CODE = r'''", 1)[1].rsplit("'''", 1)[0]
    res = run_jq_backtest(code, start=START, end=END, capital=CAPITAL)
    df = pd.DataFrame(res["trades_detail"])
    df["date"] = df["date"].astype(str)
    return df[["date", "code", "side", "shares", "price", "fee"]], res


def day_trades(df: pd.DataFrame, d: str) -> str:
    sub = df[df["date"] == d]
    parts = []
    for _, r in sub.iterrows():
        parts.append(f"{r['side']} {r['code']}({r['shares']:.0f}股@{r['price']:.2f})")
    return " ".join(parts) if parts else "(无)"


def main() -> int:
    jq = load_jq_trades()
    ours, res = load_ours()
    print(f"聚宽 {len(jq)} 笔, 我们 {len(ours)} 笔")

    dates = sorted(set(jq["date"]) | set(ours["date"]))
    first_div = None
    for d in dates:
        a = jq[jq["date"] == d]
        b = ours[ours["date"] == d]
        key = lambda x: (x["code"], x["side"])   # noqa: E731
        sa = {(r["code"], r["side"]): r["shares"] for _, r in a.iterrows()}
        sb = {(r["code"], r["side"]): r["shares"] for _, r in b.iterrows()}
        if sa != sb:
            first_div = d
            break

    print(f"\n首次分歧日期: {first_div}")
    if first_div:
        ctx_dates = dates[max(0, dates.index(first_div) - 2):
                          dates.index(first_div) + 4]
        for d in ctx_dates:
            print(f"  {d}")
            print(f"    聚宽: {day_trades(jq, d)}")
            print(f"    我们: {day_trades(ours, d)}")

    # 分歧日附近我们的委托日志(看选股/拒单线索)
    if first_div:
        print("\n-- 我们该日附近的委托/未成交/策略日志 --")
        for line in res["logs"]:
            for tag in ("[委托", "[未成交", "target_list", "['00"):
                if line.startswith("[info] ['") or tag in line:
                    if first_div[:7] in line or True:
                        pass
        # 简化: 打印首月前 40 条 info 日志
        shown = 0
        for line in res["logs"]:
            if shown >= 45:
                break
            if line.startswith("[info]") and "[日终" not in line:
                print("  ", line)
                shown += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
