# -*- coding: utf-8 -*-
"""审计修复回归：F-001 population.py screen_population 未定义变量 icir。

2026-09-22 代码审计 F-001（严重）：死因直方图循环内 `icir` 从未赋值，
任何 ok 行进入即 NameError，propose_population 工具必崩。
修复：循环内补 `icir = r.get("icir")`。
"""
from __future__ import annotations

import pandas as pd

from alphaagent.factor.mining.agent import population as pop


def _fake_session() -> object:
    idx = pd.MultiIndex.from_product(
        [["2021-01-04", "2021-01-05"], ["S000", "S001"]],
        names=["datetime", "instrument"],
    )
    panel = pd.DataFrame({"close": [1.0, 1.1, 1.2, 1.3], "label_1d": [0.0, 0.0, 0.0, 0.0]}, index=idx)

    class _Ctx:
        label_col = "label_1d"

    class _Session:
        def __init__(self, panel):
            self.panel = panel
            self.ctx = _Ctx()

    return _Session(panel)


def test_screen_population_ok_row_no_nameerror(monkeypatch):
    """F-001：ok 行进入死因直方图不得因 icir 未定义抛 NameError。"""
    monkeypatch.setattr(
        pop, "screen_expr",
        lambda *a, **k: {
            "ok": True,
            "ic": 0.03,
            "icir": 0.31,
            "rank_ic": 0.02,
            "n_days": 2,
            "coverage": 0.9,
            "cs_pearson_autocorr": 0.5,
            "secs": 1.0,
        },
    )
    res = pop.screen_population(
        _fake_session(),
        [{"name": "mom", "template": "f = RANK($close, {w})\nf", "grid": {"w": [5]}}],
        autocorr_gate=0.18,
    )
    assert res.get("ok") is True, res
    # 死因直方图正常产出：ok 行不进入 dead（icir/coverage/autocorr 全达标）
    assert res.get("dead_ends_histogram") == {}, res.get("dead_ends_histogram")


def test_screen_population_dead_reasons_recorded(monkeypatch):
    """F-001 配套：不达标 ok 行按 icir/autocorr/coverage 归入死因直方图。"""
    monkeypatch.setattr(
        pop, "screen_expr",
        lambda *a, **k: {
            "ok": True,
            "ic": 0.02,
            "icir": 0.1,  # < 0.28 → icir 死因
            "rank_ic": 0.01,
            "n_days": 2,
            "coverage": 0.9,
            "cs_pearson_autocorr": 0.5,
            "secs": 1.0,
        },
    )
    res = pop.screen_population(
        _fake_session(),
        [{"name": "mom", "template": "f = RANK($close, {w})\nf", "grid": {"w": [5]}}],
        autocorr_gate=0.18,
    )
    assert res.get("ok") is True
    assert any("icir" in k for k in (res.get("dead_ends_histogram") or {})), res.get("dead_ends_histogram")
