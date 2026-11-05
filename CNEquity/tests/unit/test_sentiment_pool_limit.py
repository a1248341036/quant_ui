# -*- coding: utf-8 -*-
"""舆情采集池上限：配置解析 + 截断逻辑（对应 [sentiment].articles_max_symbols）。

背景：``sentiment_articles`` 原先硬编码 ``_MAX_SYMBOLS_PER_RUN = 300``，只覆盖
成交额前 300 名；扩到配置项后默认 1800（吃满 universe.csv），0 表示不限制。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cnequity.config import load_config, validate_config
from cnequity.config.bootstrap import path_for_toml
from cnequity.steps.research import _MAX_SYMBOLS_FALLBACK, _apply_symbol_limit

_CODES = [f"{i:06d}" for i in range(2500)]


def _write_config(tmp_path, sentiment_block: str = "") -> object:
    cfg_path = tmp_path / "test.toml"
    cfg_path.write_text(
        f"""
[data]
root = "{path_for_toml(tmp_path / "data")}"

[orchestrator]
workers = 1
batch_size = 2

[tdx_protocol]
allow_mock = true

[[job.daily.waves]]
name = "reference"
parallel = true
steps = ["instruments", "trading_calendar"]

[sentiment]
{sentiment_block}
""",
        encoding="utf-8",
    )
    return load_config(cfg_path)


# --- 配置解析 -------------------------------------------------------------


def test_config_defaults_to_1800_when_key_absent(tmp_path):
    cfg = _write_config(tmp_path, "use_snownlp = false\n")
    assert cfg.sentiment_articles_max_symbols == 1800


def test_config_override_from_toml(tmp_path):
    cfg = _write_config(tmp_path, "articles_max_symbols = 500\n")
    assert cfg.sentiment_articles_max_symbols == 500


def test_config_zero_means_unlimited(tmp_path):
    cfg = _write_config(tmp_path, "articles_max_symbols = 0\n")
    assert cfg.sentiment_articles_max_symbols == 0


def test_validate_rejects_negative_limit(tmp_path):
    cfg = _write_config(tmp_path, "articles_max_symbols = -1\n")
    errors = validate_config(cfg)
    assert any("articles_max_symbols must be >= 0" in e for e in errors)


def test_validate_rejects_absurd_limit(tmp_path):
    cfg = _write_config(tmp_path, "articles_max_symbols = 6001\n")
    errors = validate_config(cfg)
    assert any("articles_max_symbols must be <= 6000" in e for e in errors)


# --- 截断逻辑 -------------------------------------------------------------


def _cfg(limit):
    return SimpleNamespace(sentiment_articles_max_symbols=limit)


def test_limit_truncates_to_configured_size():
    out = _apply_symbol_limit(_CODES, _cfg(300))
    assert len(out) == 300
    assert out == _CODES[:300]


def test_limit_zero_keeps_whole_universe():
    out = _apply_symbol_limit(_CODES, _cfg(0))
    assert out == _CODES


def test_limit_larger_than_universe_keeps_everything():
    out = _apply_symbol_limit(["000001", "600000"], _cfg(1800))
    assert out == ["000001", "600000"]


def test_limit_negative_falls_back_to_default():
    out = _apply_symbol_limit(_CODES, _cfg(-5))
    assert len(out) == _MAX_SYMBOLS_FALLBACK


def test_limit_missing_attribute_falls_back_to_default():
    out = _apply_symbol_limit(_CODES, SimpleNamespace())
    assert len(out) == _MAX_SYMBOLS_FALLBACK


def test_limit_non_numeric_falls_back_to_default():
    out = _apply_symbol_limit(_CODES, _cfg("not-a-number"))
    assert len(out) == _MAX_SYMBOLS_FALLBACK


def test_limit_does_not_mutate_input():
    codes = list(_CODES[:10])
    _apply_symbol_limit(codes, _cfg(3))
    assert len(codes) == 10
