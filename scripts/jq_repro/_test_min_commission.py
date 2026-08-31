# -*- coding: utf-8 -*-
"""验证 min_commission(每笔最低佣金) 接线:
1) 引擎级: 合成面板, 小额买单费用 = max(金额x费率, min_commission),
   大额单不触底; mc=0 路径与旧逐位一致
2) 兼容层: set_order_cost(OrderCost(..., min_commission=5)) 写入 cost_cfg
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from core.event_engine.context import EventStrategy  # noqa: E402
from core.event_engine.runner import run_event_backtest  # noqa: E402

PANEL = None


def _panel(days: int = 30, px: float = 5.0) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=days)
    rows = []
    for c in ("A", "B", "C"):
        for d in dates:
            rows.append({"date": d, "code": c, "open": px, "close": px,
                         "high": px, "low": px, "turnover": 1e6, "am20": 1e6})
    return pd.DataFrame(rows)


def _run(strategy_class, capital: float, min_commission: float) -> dict:
    return run_event_backtest(
        panel=PANEL, codes=["A", "B", "C"], strategy_class=strategy_class,
        start="2025-01-02", end="2025-02-15", capital=capital,
        buy_cost=0.0003, sell_cost=0.0003, lot_size=100,
        warmup_days=0, min_commission=min_commission)


class _BuyOnce(EventStrategy):
    def init(self, ctx):
        self.done = False

    def on_bar(self, ctx, bar):
        if not self.done:
            self.done = True
            ctx.order_target_shares("A", 100)   # 500 元小额单


class _BuyBig(EventStrategy):
    def init(self, ctx):
        self.done = False

    def on_bar(self, ctx, bar):
        if not self.done:
            self.done = True
            ctx.order_target_shares("A", 100 * 300)  # 15 万大单


def main() -> int:
    global PANEL
    PANEL = _panel()

    r0 = _run(_BuyOnce, 10_000.0, 0.0)
    r5 = _run(_BuyOnce, 10_000.0, 5.0)
    f0 = [t for t in r0["trades_detail"] if t["side"] == "buy"]
    f5 = [t for t in r5["trades_detail"] if t["side"] == "buy"]
    assert f0 and f5, "应有买入成交"
    fee0, fee5 = f0[0]["fee"], f5[0]["fee"]
    amount = f0[0]["amount"]
    exp0 = amount * 0.0003          # 500 x 0.0003 = 0.15
    print(f"[engine] 小单 amount={amount:.2f}  fee(mc=0)={fee0:.4f} "
          f"(期望 {exp0:.4f})  fee(mc=5)={fee5:.4f} (期望 5.0)")
    assert abs(fee0 - exp0) < 1e-9, f"mc=0 费用不符: {fee0}"
    assert abs(fee5 - 5.0) < 1e-9, f"mc=5 费用不符: {fee5}"

    rb = _run(_BuyBig, 200_000.0, 5.0)
    fb = [t for t in rb["trades_detail"] if t["side"] == "buy"][0]
    expb = fb["amount"] * 0.0003
    print(f"[engine] 大单 amount={fb['amount']:.0f}  fee={fb['fee']:.2f} "
          f"(期望 {expb:.2f}, 不触底)")
    assert abs(fb["fee"] - expb) < 1e-6, f"大单费用不符: {fb['fee']}"

    from core.event_engine.jq.api import settings as jq_settings
    from core.event_engine.jq.objects import _OrderCost

    ns: dict = {}
    rt = SimpleNamespace(log=SimpleNamespace(info=lambda *a, **k: None),
                         cost_cfg={})
    jq_settings.install(ns, rt)
    ns["set_order_cost"](_OrderCost(open_commission=0.0003,
                                    close_commission=0.0003,
                                    open_tax=0, close_tax=0.001,
                                    min_commission=5.0), type="stock")
    assert rt.cost_cfg.get("min_commission") == 5.0, rt.cost_cfg
    assert rt.cost_cfg.get("buy_cost") == 0.0003, rt.cost_cfg
    assert rt.cost_cfg.get("sell_cost") == 0.0013, rt.cost_cfg
    print("[compat] set_order_cost -> cost_cfg:",
          {k: rt.cost_cfg[k] for k in sorted(rt.cost_cfg)})

    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
