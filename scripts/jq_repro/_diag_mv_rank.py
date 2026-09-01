# -*- coding: utf-8 -*-
"""探针: 直接调 get_fundamentals, 看策略视角的市值排序 vs 原始 total_mv 排序。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    import core.event_engine.jq.entry as entry
    from core.event_engine.jq.entry import run_jq_backtest
    src = (ROOT / "scripts" / "jq_repro" / "_test_post47523.py").read_text(
        encoding="utf-8")
    code = src.split("CODE = r'''", 1)[1].rsplit("'''", 1)[0]
    # 缩小窗口只跑到 1 月中, 减少耗时
    run_jq_backtest(code, start="2025-01-02", end="2025-01-15",
                    capital=100_000.0)
    rt = entry._LAST_RT
    snap = rt.get_snapshot(pd.Timestamp("2025-01-02"))
    codes = ["002856", "002848", "002207", "002719", "002921",
             "002910", "002652", "002188"]
    print("-- get_snapshot 视角(策略实际看到的) --")
    for c in codes:
        if c in snap.index:
            print(f"  {c} mv={snap.loc[c, 'mv']:.1f}万 "
                  f"market_cap={snap.loc[c, 'market_cap']:.3f}亿")
        else:
            print(f"  {c} 不在截面!")
    print("\n-- 按 market_cap 升序前 10 --")
    sub = snap[snap["market_cap"].between(5, 50)]
    print(sub.sort_values("market_cap").head(10)[["market_cap"]].to_string())

    # 原始 total_mv 对照
    t = rt.ctx.tables
    import numpy as np
    i = list(t.dates).index(pd.Timestamp("2025-01-02"))
    rows = []
    for c in codes:
        k = rt.ctx._ci.get(c)
        rows.append((c, t.mv[i, k]))
    print("\n-- tables.mv 原始值(万) --")
    for c, mv in sorted(rows, key=lambda x: x[1]):
        print(f"  {c} {mv:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
