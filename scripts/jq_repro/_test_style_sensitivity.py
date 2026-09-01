# -*- coding: utf-8 -*-
"""风格敏感性实验: 选股扰动(偏移1位)在小盘 vs 大盘变体上的结果放大。

同一 post/47523 策略框架, 2x2:
  {小盘 between(5,50)亿, 大盘 between(100,1000)亿} x {选股偏移0, 偏移1位}
"偏移1位" = 模拟跨引擎之间最小的一种数据/排序差异。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

START, END, CAPITAL = "2025-01-02", "2026-06-01", 100_000.0


def load_code() -> str:
    src = (ROOT / "scripts" / "jq_repro" / "_test_post47523.py").read_text(
        encoding="utf-8")
    return src.split("CODE = r'''", 1)[1].rsplit("'''", 1)[0]


def variant(code: str, cap_band: tuple[float, float],
            offset: int) -> str:
    out = code.replace(
        "valuation.market_cap.between(5,50)",
        f"valuation.market_cap.between({cap_band[0]},{cap_band[1]})")
    assert out != code or cap_band == (5, 50), "市值区间替换失败"
    if offset:
        anchor = "initial_list = list(df_fun.code)"
        assert anchor in out, "偏移锚点缺失"
        out = out.replace(anchor, f"initial_list = list(df_fun.code)[{offset}:]",
                          1)
    return out


def run(code: str) -> tuple[dict, pd.Series]:
    from core.event_engine.jq.entry import run_jq_backtest
    res = run_jq_backtest(code, start=START, end=END, capital=CAPITAL)
    nav = pd.Series({pd.Timestamp(p["date"]): p["value"] for p in res["nav"]})
    return res["metrics"], nav


def main() -> int:
    base = load_code()
    configs = {
        "小盘(5-50亿) 基准": ((5, 50), 0),
        "小盘(5-50亿) 偏移1位": ((5, 50), 1),
        "大盘(100-1000亿) 基准": ((100, 1000), 0),
        "大盘(100-1000亿) 偏移1位": ((100, 1000), 1),
    }
    results: dict[str, tuple[dict, pd.Series]] = {}
    for name, (band, off) in configs.items():
        m, nav = run(variant(base, band, off))
        results[name] = (m, nav)
        print(f"{name}: 总收益 {m['总收益']:+.2%}  年化 {m['年化收益']:+.2%}  "
              f"回撤 {m['最大回撤']:.2%}")

    def delta(a: str, b: str) -> None:
        ma, na = results[a]
        mb, nb = results[b]
        both = pd.concat([na, nb], axis=1).dropna()
        r = both.corr().iloc[0, 1]
        d = (ma["总收益"] - mb["总收益"]) * 100
        print(f"\n{a} vs {b}: 终值差 {d:+.2f}pp  日收益相关 r={r:.3f}")

    delta("小盘(5-50亿) 基准", "小盘(5-50亿) 偏移1位")
    delta("大盘(100-1000亿) 基准", "大盘(100-1000亿) 偏移1位")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
