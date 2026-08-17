#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ETF 策略回测调研：在 ETF 面板上跑注册表里的代表性策略，输出指标对比。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from core import backtest_archive
from core.data import load_etf, load_etf_panel
from core.engine import run_backtest
from strategies.registry import STRATEGIES


def _points(ser):
    return [{"date": str(d), "value": float(v)} for d, v in ser.items()] if ser is not None and len(ser) else None


def fmt_pct(x):
    return f"{x*100:.2f}%" if pd.notna(x) else "NA"


def fmt_num(x, nd=2):
    return f"{x:.{nd}f}" if pd.notna(x) else "NA"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2026-08-14")
    ap.add_argument("--freq", default="monthly")
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--capital", type=float, default=100000.0)
    ap.add_argument("--min-codes", type=int, default=0,
                    help="只回测至少包含 N 个 ETF 的策略（0=不过滤）")
    ap.add_argument("--min-history", type=int, default=0,
                    help="过滤上市不足 N 个交易日的次新 ETF（0=不过滤）")
    ap.add_argument("--adx-filter", type=float, default=0,
                    help="ADX 趋势过滤阈值（信号日 ADX>=阈值才可买，0=不过滤）")
    ap.add_argument("--chandelier", type=float, default=0,
                    help="ATR Chandelier 出场乘数（0=关闭，如 3）")
    ap.add_argument("--chandelier-period", type=int, default=22)
    ap.add_argument("--regime-adx", type=float, default=0,
                    help="市场 ADX 低于阈值时降仓（0=关闭）")
    ap.add_argument("--regime-scale", type=float, default=0.5)
    ap.add_argument("--strategies", default="", help="逗号分隔策略名，默认全部")
    ap.add_argument("--out", default="data/etf_survey.csv", help="结果 CSV")
    args = ap.parse_args()

    etf = load_etf()
    panel = load_etf_panel()
    codes = sorted(set(etf["code"]) & set(panel["code"].unique()))
    if args.min_history > 0:
        before = pd.Timestamp(args.start) - pd.Timedelta(days=400)
        cnt = panel[panel["date"] < before].groupby("code", observed=True).size()
        keep = cnt[cnt >= args.min_history].index
        codes = [c for c in codes if c in set(keep)]
        print(f"过滤次新 ETF（上市>= {args.min_history} 个交易日）后: {len(codes)} 只", flush=True)
    if not codes:
        print("ETF 面板为空", file=sys.stderr)
        return 1
    print(f"ETF 池: {len(codes)} 只，面板 {len(panel)} 行 "
          f"({panel['date'].min().date()} ~ {panel['date'].max().date()})", flush=True)

    names = [s.strip() for s in args.strategies.split(",") if s.strip()]
    if names:
        unknown = [n for n in names if n not in STRATEGIES]
        if unknown:
            print(f"未知策略: {unknown}", file=sys.stderr)
            return 1
    else:
        names = list(STRATEGIES.keys())

    rows = []
    for name in names:
        strat = STRATEGIES[name]
        factor = strat["factor"]
        if factor == "composite":
            # 复合因子对 ETF 意义不大，跳过
            continue
        print(f"回测 {name} (factor={factor}, ascending={strat['ascending']})...", flush=True)
        try:
            res = run_backtest(
                panel=panel, codes=codes,
                factor=factor, ascending=strat["ascending"],
                start=args.start, end=args.end,
                capital=args.capital, top_n=args.top_n,
                freq=args.freq,
                buy_cost=0.0008, sell_cost=0.0013,
                amount_q=0.2, affordable=True, lot_size=100,
                warmup_days=400, cash_mode=True,
                limit_flags=True, slippage_bps=0.0,
                max_participation=0.0, max_weight=None,
                analyze=True,
                adx_filter=(args.adx_filter or None),
                chandelier_mult=args.chandelier,
                chandelier_period=args.chandelier_period,
                regime_adx=(args.regime_adx or None),
                regime_scale=args.regime_scale,
            )
        except Exception as exc:
            print(f"  {name} 回测失败: {exc}", file=sys.stderr)
            continue
        m = res["metrics"]
        bm = res["bench_metrics"]
        fq = res.get("factor_quality") or {}
        ic = None
        if fq and "ic_series" in fq and len(fq["ic_series"]):
            ic = float(fq["ic_series"].mean())
        rows.append({
            "strategy": name,
            "factor": factor,
            "ascending": strat["ascending"],
            "total_return": m["总收益"],
            "annual_return": m["年化收益"],
            "annual_vol": m["年化波动"],
            "sharpe": m["夏普"],
            "max_drawdown": m["最大回撤"],
            "calmar": m["卡玛"],
            "win_rate": m["胜率"],
            "bench_annual": bm["年化收益"],
            "bench_sharpe": bm["夏普"],
            "bench_maxdd": bm["最大回撤"],
            "mean_ic": ic,
            "n_signals": str(res["last_signal_date"]) if res.get("last_signal_date") is not None else None,
            "n_trades": len(res["trades"]) if res.get("trades") is not None else None,
        })
        print(f"  年化 {fmt_pct(m['年化收益'])}  夏普 {fmt_num(m['夏普'])}  "
              f"回撤 {fmt_pct(m['最大回撤'])}  IC {fmt_num(ic) if ic is not None else 'NA'}", flush=True)
        try:
            backtest_archive.save_run(
                kind="etf_survey",
                params={"universe": "ETF", "strategy": name, "factor": factor,
                        "ascending": strat["ascending"], "top_n": args.top_n,
                        "capital": args.capital, "freq": args.freq,
                        "start": args.start, "end": args.end,
                        "min_history": args.min_history,
                        "adx_filter": args.adx_filter,
                        "chandelier_mult": args.chandelier,
                        "chandelier_period": args.chandelier_period,
                        "regime_adx": args.regime_adx,
                        "regime_scale": args.regime_scale},
                metrics=res["metrics"],
                bench_metrics=res["bench_metrics"],
                nav=_points(res["nav"]),
                drawdown=_points(res["drawdown"]),
                data_version=str(panel["date"].max().date()),
            )
        except Exception as exc:
            print(f"  {name} 归档失败: {exc}", file=sys.stderr)

    if not rows:
        print("无结果", file=sys.stderr)
        return 1
    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent.parent / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print("\n=== 结果表 ===")
    show = df[["strategy", "total_return", "annual_return", "annual_vol",
               "sharpe", "max_drawdown", "calmar", "win_rate",
               "bench_annual", "bench_sharpe", "bench_maxdd", "mean_ic"]].copy()
    for c in show.columns:
        if c != "strategy":
            show[c] = show[c].map(lambda v: fmt_pct(v) if "return" in c or "drawdown" in c else fmt_num(v))
    print(show.to_string(index=False))
    print(f"\n已保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
