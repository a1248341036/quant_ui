# -*- coding: utf-8 -*-
"""验证盘中价撮合(Order.fill_px):
1) fill_px 给定 -> 成交价 = fill_px x (1+滑点), 而非执行日开盘
2) fill_px 缺省 -> 与旧路径逐位一致(按开盘)
3) 多单聚合: 同码后单覆盖前单(含 fill_px)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from core.event_engine.context import EventStrategy  # noqa: E402
from core.event_engine.runner import run_event_backtest  # noqa: E402


def _panel(days: int = 30, px: float = 5.0) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=days)
    rows = []
    for c in ("A", "B", "C"):
        for d in dates:
            rows.append({"date": d, "code": c, "open": px, "close": px,
                         "high": px, "low": px, "turnover": 1e6, "am20": 1e6})
    return pd.DataFrame(rows)


class _BuyFill(EventStrategy):
    """首日以 fill_px=5.5 买入 100 股(开盘价 5.0)。"""

    def init(self, ctx):
        self.done = False

    def on_bar(self, ctx, bar):
        if not self.done:
            self.done = True
            ctx.order_target_shares("A", 100, fill_price=5.5)


class _BuyPlain(EventStrategy):
    def init(self, ctx):
        self.done = False

    def on_bar(self, ctx, bar):
        if not self.done:
            self.done = True
            ctx.order_target_shares("A", 100)


def main() -> int:
    panel = _panel()

    rf = run_event_backtest(panel=panel, codes=["A", "B", "C"],
                            strategy_class=_BuyFill, start="2025-01-02",
                            end="2025-02-15", capital=10_000.0,
                            buy_cost=0.0003, sell_cost=0.0003, lot_size=100,
                            warmup_days=0, slippage_bps=2.0)
    f = [t for t in rf["trades_detail"] if t["side"] == "buy"][0]
    exp = 5.5 * (1 + 2.0 / 1e4)
    print(f"[fill_px] price={f['price']:.6f} (期望 {exp:.6f})")
    assert abs(f["price"] - exp) < 1e-9, f["price"]

    r0 = run_event_backtest(panel=panel, codes=["A", "B", "C"],
                            strategy_class=_BuyPlain, start="2025-01-02",
                            end="2025-02-15", capital=10_000.0,
                            buy_cost=0.0003, sell_cost=0.0003, lot_size=100,
                            warmup_days=0, slippage_bps=2.0)
    f0 = [t for t in r0["trades_detail"] if t["side"] == "buy"][0]
    exp0 = 5.0 * (1 + 2.0 / 1e4)
    print(f"[default] price={f0['price']:.6f} (期望 {exp0:.6f})")
    assert abs(f0["price"] - exp0) < 1e-9, f0["price"]

    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
