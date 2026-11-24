# -*- coding: utf-8 -*-
"""pred 多通道解析：通道名 → 文件路径，互不覆盖，默认仍走 demo。

回归背景（2026-11-23）：pred 只有一个写死的 pred_demo.parquet，任何一次
`train_ml_composite --write-pred` 都把它覆盖成"训练窗口内"的分数，
实测把 2024-07~2026-10 的生产分数冲成只剩 2024-07~2024-12，模拟盘随即失去信号。
"""
import os
from pathlib import Path

from core.store import PRED_CHANNEL_ENV, STOCK_DIR, pred_file


def test_default_channel_is_demo():
    assert pred_file() == STOCK_DIR / "pred_demo.parquet"
    assert pred_file("demo") == STOCK_DIR / "pred_demo.parquet"
    assert pred_file("") == STOCK_DIR / "pred_demo.parquet"


def test_named_channels_are_separate_files():
    a = pred_file("exp1")
    b = pred_file("night_rp39")
    assert a == STOCK_DIR / "pred_exp1.parquet"
    assert b == STOCK_DIR / "pred_night_rp39.parquet"
    assert a != b
    assert a != pred_file()


def test_env_var_selects_channel(monkeypatch):
    monkeypatch.setenv(PRED_CHANNEL_ENV, "night4")
    assert pred_file() == STOCK_DIR / "pred_night4.parquet"
    # 显式参数优先于环境变量
    assert pred_file("demo") == STOCK_DIR / "pred_demo.parquet"


def test_channel_name_is_sanitized(monkeypatch):
    monkeypatch.delenv(PRED_CHANNEL_ENV, raising=False)
    p = pred_file("a/b c..d")
    assert p.parent == STOCK_DIR
    assert "/" not in p.name and " " not in p.name
    assert p.name.startswith("pred_") and p.suffix == ".parquet"


def test_missing_channel_returns_none():
    from core.data.panel import load_pred_scores

    assert load_pred_scores(channel="__no_such_channel__") is None
