# -*- coding: utf-8 -*-
"""预测链路 ST 剔除（_apply_st_mask）。

背景（2026-11-26）：ST 掩码此前只接入因子评估 / submit / 群体筛，**没有接入 ML 组合
预测链**，导致 pred_demo 的 Top10 全是 ST 股（2026-10 实测 9~10 只）。
且 st_row_mask 按 (datetime, instrument) 精确 join，instrument 带交易所后缀，
传 6 位 code 会一行都匹配不上而**静默失效**——本测试同时锁住这两点。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "predict_ml_composite.py"
_spec = importlib.util.spec_from_file_location("predict_ml_composite", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_apply_st_mask = _mod._apply_st_mask
_norm_code = _mod._norm_code


def _panel() -> pd.DataFrame:
    days = pd.bdate_range("2024-01-02", periods=3)
    idx = pd.MultiIndex.from_product(
        [days, ["000001.SZ", "000002.SZ", "600000.SH"]],
        names=["datetime", "instrument"],
    )
    return pd.DataFrame({"close": np.arange(len(idx), dtype=float)}, index=idx)


def _frame() -> pd.DataFrame:
    days = list(pd.bdate_range("2024-01-02", periods=3))
    rows = []
    for d in days:
        for c in ("000001", "000002", "600000"):
            rows.append({"date": d, "code": c, "score": 1.0})
    return pd.DataFrame(rows)


def test_norm_code_strips_suffix():
    assert _norm_code("000001.SZ") == "000001"
    assert _norm_code("600000.SH") == "600000"
    assert _norm_code(1) == "000001"


def test_st_rows_are_dropped(monkeypatch):
    mask = np.zeros(len(_panel()), dtype=bool)
    # 把 000002 的三天标记为 ST
    mask[1::3] = True
    monkeypatch.setattr(
        "alphaagent.factor.metrics.st_mask.st_row_mask", lambda *a, **k: mask
    )
    out, n = _apply_st_mask(_frame(), _panel(), True)
    assert n == 3
    assert "000002" not in set(out["code"])
    assert len(out) == 6


def test_disabled_keeps_everything(monkeypatch):
    monkeypatch.setattr(
        "alphaagent.factor.metrics.st_mask.st_row_mask",
        lambda *a, **k: np.ones(len(_panel()), dtype=bool),
    )
    out, n = _apply_st_mask(_frame(), _panel(), False)
    assert n == 0 and len(out) == 9


def test_mask_unavailable_is_fail_open(monkeypatch):
    monkeypatch.setattr(
        "alphaagent.factor.metrics.st_mask.st_row_mask", lambda *a, **k: None
    )
    out, n = _apply_st_mask(_frame(), _panel(), True)
    assert n == 0 and len(out) == 9


def test_six_digit_codes_would_not_match(monkeypatch):
    """回归锁：6 位 code 直接喂 st_row_mask 会全 False（静默失效），
    必须走 panel 索引；本测试确认我们的实现不依赖那个错误前提。"""
    mask = np.zeros(len(_panel()), dtype=bool)
    mask[:] = True  # 全 ST
    monkeypatch.setattr(
        "alphaagent.factor.metrics.st_mask.st_row_mask", lambda *a, **k: mask
    )
    out, n = _apply_st_mask(_frame(), _panel(), True)
    assert n == 9 and len(out) == 0  # 用 panel 索引映射成功，全部被剔除
