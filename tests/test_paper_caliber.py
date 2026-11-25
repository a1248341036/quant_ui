# -*- coding: utf-8 -*-
"""模拟盘可执行口径：价差/滑点/最低佣金 + score 加权。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.paper.rebalance import _compute_targets, _cost_bps, _fee


def test_cost_bps_uses_half_spread_plus_slippage():
    risk = {"spread_bps": 22.0, "slippage_bps": 5.0}
    assert _cost_bps(risk, "buy") == 16.0   # 11 + 5
    assert _cost_bps(risk, "sell") == 16.0
    assert _cost_bps({}, "buy") == 0.0
    assert _cost_bps({"spread_bps": -1, "slippage_bps": -1}, "buy") == 0.0


def test_fee_applies_min_commission():
    assert _fee(1000.0, 0.00025, {}) == 0.25
    assert _fee(1000.0, 0.00025, {"min_commission": 5.0}) == 5.0
    assert _fee(100_000.0, 0.00025, {"min_commission": 5.0}) == 25.0


def _panel_and_acc(weight_mode: str):
    dates = pd.bdate_range("2024-01-02", periods=40)
    codes = ["000001", "000002", "000003"]
    rows = []
    for d in dates:
        for i, c in enumerate(codes):
            px = 10.0 + i
            rows.append({"date": d, "code": c, "open": px, "high": px, "low": px,
                         "close": px, "turnover": 1e7, "amount": 1e8,
                         "am20": 1e8, "turn20": 1e7,
                         "score": float(i + 1)})
    panel = pd.DataFrame(rows)
    acc = {"id": 999, "factor": "score", "ascending": False, "top_n": 3,
           "risk_config": {"weight_mode": weight_mode}}
    return panel, acc, codes


def test_compute_targets_score_mode_is_rank_weighted(monkeypatch):
    panel, acc, codes = _panel_and_acc("score")
    fake = pd.DataFrame({"code": ["000001", "000002", "000003"],
                         "score": [1.0, 2.0, 3.0], "side": ["多", "多", "多"]})
    monkeypatch.setattr("core.engine.latest_signals",
                        lambda *a, **k: (fake, pd.Timestamp("2024-02-20")))
    targets, sig_date, err = _compute_targets(acc, panel, codes, pd.Timestamp("2024-02-21"))
    assert err is None and sig_date is not None
    assert targets["000003"] > targets["000002"] > targets["000001"]
    assert sum(targets.values()) == pytest.approx(1.0)


def test_compute_targets_equal_mode(monkeypatch):
    panel, acc, codes = _panel_and_acc("equal")
    fake = pd.DataFrame({"code": ["000001", "000002", "000003"],
                         "score": [1.0, 2.0, 3.0], "side": ["多", "多", "多"]})
    monkeypatch.setattr("core.engine.latest_signals",
                        lambda *a, **k: (fake, pd.Timestamp("2024-02-20")))
    targets, _, err = _compute_targets(acc, panel, codes, pd.Timestamp("2024-02-21"))
    assert err is None
    assert all(abs(v - 1 / 3) < 1e-12 for v in targets.values())
