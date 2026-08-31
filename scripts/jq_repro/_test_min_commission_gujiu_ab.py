# -*- coding: utf-8 -*-
"""国九 A/B 对照: min_commission=5 vs 0 (当前数据下),
区分成交笔数漂移(61->50)来自最低佣金接线还是数据湖更新。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from core.event_engine.jq.entry import run_jq_backtest  # noqa: E402

SRC = (ROOT / "scripts" / "jq_repro" / "_test_guojiu.py").read_text(
    encoding="utf-8")
CODE = SRC.split("CODE = r'''", 1)[1].rsplit("'''", 1)[0]
assert "min_commission=5" in CODE, "国九代码未设 min_commission"


def run(mc: str) -> dict:
    if mc != "5":
        code = CODE.replace("min_commission=5", f"min_commission={mc}")
        assert f"min_commission={mc}" in code, "min_commission 替换失败"
    else:
        code = CODE
    out = run_jq_backtest(code, start="2024-01-01", end="2024-12-31",
                          capital=1_000_000.0)
    tr = out["metrics"].get("总收益", out["metrics"].get("total_return"))
    n = len(out["trades"])
    hold = [h["code"] for h in out["holdings"]][:6]
    print(f"min_commission={mc}: 总收益 {tr:.4%}  成交笔数(事件) {n}  "
          f"期末持仓 {hold}")
    return {"tr": float(tr), "n": n, "hold": hold}


def main() -> int:
    a = run("5")
    b = run("0")
    print(f"差异: 总收益 {a['tr'] - b['tr']:+.6%}  "
          f"事件笔数 {a['n']} vs {b['n']}  "
          f"持仓一致={a['hold'] == b['hold']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
