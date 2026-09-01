# -*- coding: utf-8 -*-
"""过滤链追踪: 002856/002207/002719 在我们引擎的选股链路里哪一步被剔?"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

CODES = ["002856", "002848", "002207", "002719", "002921",
         "002910", "002652", "002188"]


def main() -> int:
    import core.event_engine.jq.entry as entry
    from core.event_engine.jq.entry import run_jq_backtest
    src = (ROOT / "scripts" / "jq_repro" / "_test_post47523.py").read_text(
        encoding="utf-8")
    code = src.split("CODE = r'''", 1)[1].rsplit("'''", 1)[0]
    run_jq_backtest(code, start="2025-01-02", end="2025-01-15",
                    capital=100_000.0)
    rt = entry._LAST_RT
    snap = rt.get_snapshot(pd.Timestamp("2025-01-02"))

    status = pd.DataFrame(index=CODES)
    status["市值亿"] = [round(snap.loc[c, "market_cap"], 2) if c in snap.index
                      else None for c in CODES]
    status["在截面"] = [c in snap.index for c in CODES]
    status["st"] = [bool(snap.loc[c, "st"]) if c in snap.index else None
                    for c in CODES]
    status["paused"] = [bool(snap.loc[c, "paused"]) if c in snap.index
                        else None for c in CODES]
    status["name"] = [rt.ctx.name_map.get(c, "") for c in CODES]
    status["listed_ok"] = [bool(snap.loc[c, "listed_ok"])
                           if c in snap.index else None for c in CODES]
    status["上市日"] = [str(rt.ctx.list_date_map.get(c, ""))[:10]
                       for c in CODES]
    # 01-02 涨跌停状态(收盘==limit)
    t = rt.ctx.tables
    i = list(t.dates).index(pd.Timestamp("2025-01-02"))
    hl, ll = [], []
    for c in CODES:
        k = rt.ctx._ci.get(c)
        close = t.close_raw[i, k]
        hl.append(bool(close >= t.up_limit[i, k] - 1e-6))
        ll.append(bool(close <= t.down_limit[i, k] + 1e-6))
    status["01-02涨停"] = hl
    status["01-02跌停"] = ll
    print(status.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
