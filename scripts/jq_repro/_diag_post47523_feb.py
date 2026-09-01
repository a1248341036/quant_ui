# -*- coding: utf-8 -*-
"""post47523 分段成交诊断: 精确定位 2 月初每次成交的日期与通道。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    from core.event_engine.jq.entry import run_jq_backtest
    src = (ROOT / "scripts" / "jq_repro" / "_test_post47523.py").read_text(
        encoding="utf-8")
    code = src.split("CODE = r'''", 1)[1].rsplit("'''", 1)[0]

    for end in ("2025-01-31", "2025-02-14", "2025-02-28"):
        res = run_jq_backtest(code, start="2025-01-02", end=end,
                              capital=100_000.0)
        print(f"\n=== 至 {end} ===")
        for t in res["trades"]:
            print(f"  {t['date']}  买[{t['bought']}] 卖[{t['sold']}] "
                  f"持仓{t['num_hold']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
