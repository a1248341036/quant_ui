# -*- coding: utf-8 -*-
"""验证 冒烟模式 + API 预检失败路径。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    from core.event_engine.jq.entry import run_jq_backtest
    src = (ROOT / "scripts" / "jq_repro" / "_test_post48789.py").read_text(
        encoding="utf-8")
    code = src.split("CODE = r'''", 1)[1].rsplit("'''", 1)[0]

    t0 = time.time()
    res = run_jq_backtest(code, start="2025-01-02", end="2026-06-01",
                          capital=100_000.0, smoke=True)
    dt = time.time() - t0
    print(f"[smoke] 耗时 {dt:.0f}s  收益 {res['metrics']['策略收益']:.2%}")

    bad = code + "\n\ndef _extra(context):\n    get_mtss(context)\n"
    t1 = time.time()
    try:
        run_jq_backtest(bad, start="2025-01-02", end="2025-03-01",
                        smoke=True)
        print("[preflight] 未拦截(异常!)")
    except RuntimeError as e:
        print(f"[preflight] 拦截 {time.time()-t1:.1f}s: {str(e)[:100]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
