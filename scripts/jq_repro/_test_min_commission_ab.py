# -*- coding: utf-8 -*-
"""A/B 对照: 小盘三正策略 min_commission=0 vs 0.1 (当前数据下),
区分"结果漂移"来自最低佣金接线还是数据湖更新。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from core.event_engine.jq import run_jq_backtest  # noqa: E402

SRC = (ROOT / "scripts" / "jq_repro" / "_validate_jq_compat.py").read_text(
    encoding="utf-8")
CODE = SRC.split("CODE = '''", 1)[1].split("'''\n\n\ndef main", 1)[0]


def run(mc: str) -> float:
    code = CODE.replace("min_commission=0.1", f"min_commission={mc}")
    assert code != CODE or mc == "0.1", "min_commission 替换失败"
    out = run_jq_backtest(code, start="2025-09-01", capital=100_000.0)
    tr = out["metrics"]["总收益"]
    print(f"min_commission={mc}: 总收益 {tr:.4%}  "
          f"年化 {out['metrics']['年化收益']:.4%}")
    return float(tr)


def main() -> int:
    t0 = run("0.1")
    t1 = run("0")
    print(f"差值(mc=0.1 - mc=0): {t0 - t1:+.6%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
