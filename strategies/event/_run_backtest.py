# -*- coding: utf-8 -*-
"""单文件策略快速回测 CLI.

用法:
  python strategies/event/_run_backtest.py jq_smallcap 2025-01-01 [end] [capital]
模块必须是 strategies/event/ 下定义了 EVENT_STRATEGIES 的单文件策略。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "strategies" / "event"))
sys.path.insert(0, str(ROOT))

from _runtime import run_backtest  # noqa: E402


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "jq_smallcap"
    start = sys.argv[2] if len(sys.argv) > 2 else "2025-01-01"
    end = sys.argv[3] if len(sys.argv) > 3 else pd.Timestamp.today().strftime("%Y-%m-%d")
    capital = float(sys.argv[4]) if len(sys.argv) > 4 else 100_000.0

    mod_path = ROOT / "strategies" / "event" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"strategy_{name}", str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    strategies = getattr(mod, "EVENT_STRATEGIES", {})
    if not strategies:
        print(f"{mod_path} 未定义 EVENT_STRATEGIES")
        return 1

    ctx = getattr(mod, "_CTX")
    out = {}
    for sname, cls in strategies.items():
        res = run_backtest(ctx, cls, start=start, end=end, capital=capital)
        nav = res["nav"]
        dd = nav / nav.cummax() - 1
        out[sname] = res
        print(f"\n== {sname} ({start} ~ {end}, {capital:,.0f}) ==")
        print(f"  总收益 {nav.iloc[-1]-1:+.1%}  终值 {nav.iloc[-1]:.4f}")
        print(f"  最大回撤 {dd.min():.2%}")
        print(f"  成交日 {len(res['trades'])}")
        print("  期末持仓:")
        print(res["holdings"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
