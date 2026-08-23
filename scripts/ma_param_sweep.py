#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""双均线参数稳定性扫描：不同 fast/slow 组合在 ETF 池上的绩效矩阵。

判断参数平原：最优参数邻域绩效是否平滑；若相邻参数绩效断层=尖峰/过拟合。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from core import backtest_archive
from core.data import load_etf, load_etf_panel
from core.engine import run_backtest, build_factor_frames
from strategies.registry import STRATEGIES


def _points(ser):
    return [{"date": str(d), "value": float(v)} for d, v in ser.items()] if ser is not None and len(ser) else None


def fmt_pct(x):
    return f"{x*100:.1f}%" if pd.notna(x) else "NA"


def fmt_num(x, nd=2):
    return f"{x:.{nd}f}" if pd.notna(x) else "NA"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2026-08-14")
    ap.add_argument("--freq", default="monthly")
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--capital", type=float, default=6000.0)
    ap.add_argument("--min-history", type=int, default=60)
    ap.add_argument("--adx-filter", type=float, default=25)
    ap.add_argument("--fasts", default="5,10,20")
    ap.add_argument("--slows", default="20,30,60,90,120")
    ap.add_argument("--out", default="data/stock/ma_sweep.csv")
    args = ap.parse_args()

    etf = load_etf()
    panel = load_etf_panel()
    codes = sorted(set(etf["code"]) & set(panel["code"].unique()))
    if args.min_history > 0:
        before = pd.Timestamp(args.start) - pd.Timedelta(days=400)
        cnt = panel[panel["date"] < before].groupby("code", observed=True).size()
        keep = cnt[cnt >= args.min_history].index
        codes = [c for c in codes if c in set(keep)]
    print(f"ETF 池 {len(codes)} 只，ADX 过滤 {args.adx_filter}，"
          f"{args.start} ~ {args.end}，{args.freq}，Top{args.top_n}，本金 {args.capital:g}", flush=True)

    fasts = [int(x) for x in args.fasts.split(",") if x.strip()]
    slows = [int(x) for x in args.slows.split(",") if x.strip()]
    combos = [(f, s) for f in fasts for s in slows if s > f]

    rows = []
    for fast, slow in combos:
        fname = f"ma_cross{fast}_{slow}"

        def make_builder(fa=fast, sl=slow):
            def builder(close, am20, turn20, financial=None):
                d = build_factor_frames(close, am20, turn20, financial=financial)
                d[f"ma_cross{fa}_{sl}"] = (close.rolling(fa).mean()
                                           / close.rolling(sl).mean() - 1.0)
                return d
            return builder

        print(f"回测 MA({fast},{slow})...", flush=True)
        res = run_backtest(
            panel=panel, codes=codes, factor=fname,
            factor_builder=make_builder(),
            ascending=False, start=args.start, end=args.end,
            capital=args.capital, top_n=args.top_n, freq=args.freq,
            buy_cost=0.0008, sell_cost=0.0013,
            amount_q=0.2, affordable=True, lot_size=100,
            warmup_days=400, cash_mode=True,
            limit_flags=True, slippage_bps=0.0,
            max_participation=0.0, max_weight=None,
            analyze=True, adx_filter=(args.adx_filter or None),
        )
        m = res["metrics"]
        rows.append({
            "fast": fast, "slow": slow,
            "annual_return": m["年化收益"], "annual_vol": m["年化波动"],
            "sharpe": m["夏普"], "max_drawdown": m["最大回撤"],
            "calmar": m["卡玛"], "win_rate": m["胜率"],
            "bench_annual": res["bench_metrics"]["年化收益"],
        })
        print(f"  MA({fast},{slow}) 年化 {fmt_pct(m['年化收益'])} "
              f"夏普 {fmt_num(m['夏普'])} 回撤 {fmt_pct(m['最大回撤'])} "
              f"卡玛 {fmt_num(m['卡玛'])}", flush=True)
        try:
            backtest_archive.save_run(
                kind="sweep",
                params={"universe": "ETF", "factor": fname, "fast": fast,
                        "slow": slow, "ascending": False,
                        "top_n": args.top_n, "capital": args.capital,
                        "freq": args.freq, "start": args.start, "end": args.end,
                        "min_history": args.min_history,
                        "adx_filter": args.adx_filter},
                metrics=res["metrics"],
                bench_metrics=res["bench_metrics"],
                nav=_points(res["nav"]),
                drawdown=_points(res["drawdown"]),
                data_version=str(panel["date"].max().date()),
            )
        except Exception as exc:
            print(f"  MA({fast},{slow}) 归档失败: {exc}", file=sys.stderr)

    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    out = Path(__file__).resolve().parent.parent / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print("\n=== 按夏普排序 ===")
    show = df[["fast", "slow", "annual_return", "sharpe", "max_drawdown", "calmar", "win_rate"]].copy()
    for c in ("annual_return", "max_drawdown"):
        show[c] = show[c].map(fmt_pct)
    for c in ("sharpe", "calmar", "win_rate"):
        show[c] = show[c].map(fmt_num)
    print(show.to_string(index=False))

    print("\n=== 夏普热力图（fast 行 × slow 列） ===")
    mat = df.pivot_table(index="fast", columns="slow", values="sharpe")
    print(mat.round(2).to_string())
    print("\n=== 年化热力图 ===")
    mat2 = df.pivot_table(index="fast", columns="slow", values="annual_return") * 100
    print(mat2.round(1).to_string())
    print(f"\n已保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
