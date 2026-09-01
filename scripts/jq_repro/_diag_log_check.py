# -*- coding: utf-8 -*-
"""检查 [委托]/[未成交] 日志行样本。"""
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
    res = run_jq_backtest(code, start="2025-01-02", end="2025-02-20",
                          capital=100_000.0)
    for line in res["logs"]:
        if ("[委托" in line or "[未成交" in line
                or "[成交" in line):
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
