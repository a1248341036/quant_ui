# -*- coding: utf-8 -*-
"""行业拥挤度过滤的单元测试。

实证背景（2015-2026，142 次调仓）：最拥挤 20% 行业未来 20 日相对全行业等权
超额 -0.41%（t=-2.31），是六个候选信号（动量20/60、拥挤换手、拥挤成交占比、
波动、PB）中唯一显著的一个。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.push_pred_scores import (
    apply_crowding_filter,
    build_crowding_blocklist,
)


def _daily(rows):
    return pd.DataFrame(rows, columns=["date", "code", "turnover_rate"])


def test_hot_industry_is_blocked():
    """12 个行业里，换手突然放大的那个应被判为最拥挤。"""
    dates = pd.bdate_range("2026-01-01", periods=8)
    rows = []
    for i, d in enumerate(dates):
        for k in range(12):
            code = f"00000{k}"
            turn = 1.0
            if k == 0:                       # 行业 I0：后半段换手放大 10 倍
                turn = 1.0 if i < 4 else 10.0
            elif k == 1:                     # I1：恒定高换手，但不放大
                turn = 20.0
            rows.append((d, code, turn))
    daily = _daily(rows)
    imap = {f"00000{k}": f"I{k}" for k in range(12)}

    block = build_crowding_blocklist(
        daily, imap, signal_dates=[dates[-1]], top_pct=0.2, fast=2, slow=4)
    assert dates[-1] in block
    hot = block[dates[-1]]
    assert "I0" in hot, "换手放大的行业必须被判为最拥挤"
    # 12 行业 × 20% = 2 个名额，不应把全部并列 1.0 的行业都打上
    assert len(hot) <= 3, f"黑名单应约 2 个行业，实际 {len(hot)}: {sorted(hot)}"


def test_crowding_filter_zeros_only_blocked_industry():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-02-02"] * 3),
        "code": ["000001", "000002", "000003"],
        "score": [1.0, 2.0, 3.0],
    })
    block = {pd.Timestamp("2026-02-02"): {"A"}}
    out, stats = apply_crowding_filter(df, block, {"000001": "A", "000002": "B",
                                                   "000003": "C"})
    assert stats["crowding_blocked"] == 1
    kept = out.dropna(subset=["score"])
    assert kept["code"].tolist() == ["000002", "000003"]


def test_crowding_filter_noop_when_blocklist_empty():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-02-02"]),
        "code": ["000001"], "score": [1.0],
    })
    out, stats = apply_crowding_filter(df, {}, {"000001": "A"})
    assert stats["crowding_blocked"] == 0
    assert out["score"].notna().all()


def test_insufficient_industries_skips_day():
    """行业数 < 10 时该信号日不过滤（避免小样本分位失真）。"""
    dates = pd.bdate_range("2026-01-01", periods=6)
    rows = [(d, f"00000{i}", float(i)) for d in dates for i in (1, 2, 3)]
    daily = _daily(rows)
    imap = {f"00000{i}": f"I{i}" for i in (1, 2, 3)}
    block = build_crowding_blocklist(
        daily, imap, signal_dates=[dates[-1]], top_pct=0.2, fast=2, slow=3)
    assert dates[-1] not in block


def test_signal_date_before_history_is_skipped():
    daily = _daily([
        (pd.Timestamp("2026-01-05"), "000001", 1.0),
        (pd.Timestamp("2026-01-06"), "000001", 1.0),
    ])
    block = build_crowding_blocklist(
        daily, {"000001": "A"}, signal_dates=[pd.Timestamp("2020-01-01")],
        top_pct=0.2, fast=2, slow=3)
    assert block == {}
