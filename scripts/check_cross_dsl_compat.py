#!/usr/bin/env python3
"""跨 DSL 兼容性验证：原版 AlphaAgent 已交付因子表达式在 quant_ui 引擎上复算。

用法:
  .venv/Scripts/python.exe scripts/check_cross_dsl_compat.py
"""

from __future__ import annotations

import sys
from pathlib import Path

QUANT_UI_ROOT = Path(__file__).resolve().parents[1]
EXPR_DIR = Path(r"D:\Quant\AlphaAgent\artifacts\factorzoo\stock_1d\expressions")

sys.path.insert(0, str(QUANT_UI_ROOT))

from alphaagent.data.adapters.cnequity import load_panel_from_cne  # noqa: E402
from alphaagent.dsl import eval_factor  # noqa: E402


def main() -> None:
    panel = load_panel_from_cne(start="2023-01-01", end="2023-06-30")
    print(f"panel: {panel.shape}")
    exprs = sorted(EXPR_DIR.glob("*.dsl"))
    print(f"checking {len(exprs)} expressions on quant_ui engine...\n")
    ok, fail = 0, []
    for f in exprs:
        expr = f.read_text(encoding="utf-8").strip()
        try:
            s = eval_factor(expr, panel)
            n = int(s.notna().sum())
            print(f"  OK   {f.stem:50s} finite={n:,}")
            ok += 1
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).splitlines()[0][:110]
            print(f"  FAIL {f.stem:50s} {type(exc).__name__}: {msg}")
            fail.append((f.stem, f"{type(exc).__name__}: {msg}"))
    print(f"\n{ok}/{len(exprs)} passed")
    if fail:
        print("failures:")
        for name, msg in fail:
            print(f"  {name}: {msg}")


if __name__ == "__main__":
    main()
