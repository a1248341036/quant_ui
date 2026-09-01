# -*- coding: utf-8 -*-
"""post47523 一月诊断: 为什么聚宽 01-03 买入而我们空仓?"""
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

    res = run_jq_backtest(code, start="2025-01-02", end="2025-02-15",
                          capital=100_000.0)
    print("metrics:", {k: (round(v, 3) if isinstance(v, float) else v)
                       for k, v in res["metrics"].items()})
    print("trades:")
    for t in res["trades"]:
        print(f"  {t['date']}  买[{t['bought']}] 卖[{t['sold']}] "
              f"持仓{t['num_hold']}")
    print("\nlogs(前 60 条):")
    for line in res["logs"][:60]:
        print(" ", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
