#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Brinson 归因：对事件策略回测结果做行业配置/选股分解。

用法：
  python scripts/attribution.py --start 2023-01-03 --end 2026-08-14
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.attribution import brinson_attribution  # noqa: E402
from core.data import load_panel, load_tech  # noqa: E402
from core.event_engine import GoldenCrossStrategy, run_event_backtest  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2023-01-03")
    ap.add_argument("--end", default="2026-08-14")
    ap.add_argument("--capital", type=float, default=50000.0)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--out", default="results/attribution")
    args = ap.parse_args()

    panel = load_panel()
    tech = load_tech()
    codes = sorted(set(tech["code"]) & set(panel["code"].unique()))
    codes = [c for c in codes if not c.startswith(("300", "301", "688", "689"))]
    ind_map = {str(c).zfill(6): str(i) for c, i in
               zip(tech["code"], tech["industry"])}

    print(f"回测: 金叉事件策略 {args.start} ~ {args.end}")
    res = run_event_backtest(panel, codes, GoldenCrossStrategy,
                             args.start, args.end, args.capital,
                             warmup_days=args.warmup)
    m = res["metrics"]
    print(f"组合: 总收益={m['总收益']*100:.2f}% 夏普={m['夏普']:.2f} "
          f"回撤={m['最大回撤']*100:.2f}%")

    detail, summary = brinson_attribution(
        panel, codes, res["weight_history"], res["dates"], ind_map)
    if summary.empty:
        print("无归因结果（组合可能长期空仓）")
        return 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out / "attribution_detail.csv", index=False)
    summary.to_csv(out / "attribution_summary.csv", index=False)

    pd.set_option("display.width", 160)
    print("\n===== 全期行业归因汇总（按总效应排序） =====")
    print(summary[["industry", "allocation", "selection", "interaction",
                   "total", "total_pct", "avg_combo_weight",
                   "avg_bench_weight"]].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n全期组合超额收益（行业效应合计）: "
          f"{summary['total'].sum()*100:.2f}%")
    print(f"\n输出: {out}/attribution_summary.csv, attribution_detail.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
