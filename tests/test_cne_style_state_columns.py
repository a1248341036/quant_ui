# -*- coding: utf-8 -*-
"""聚焦 run 恒载风格状态列：style_state 必须经 cnequity 恒载机制进入面板（P1）。

背景（2026-10-14）：rotation_policy.report_state_metrics 挂载的状态条件 IC
（state_conditional_ic）因数据端从无插件产出 ``style_state`` 而整晚静默返回
``_missing_state``；即使插件落地，若 style_state 不进 ``_ALWAYS_WANTED_AUX_COLUMNS``，
聚焦 run 会把插件整个跳过（style_state 不属任何数据面，_column_facets 为空），
行业列 2026-09-27 的「不可用字段」会在评估端以「面板缺列 → 静默 missing」复现。
本文件锁定：聚焦加载/裁剪恒载 style_state + 状态计算纯函数（含无前视纪律）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaagent.data.adapters import cnequity
from alphaagent.data.adapters.plugins.style_state import (
    _STATE_CODES,
    _state_from_wide,
)
from alphaagent.data.adapters.registry import DataSourcePlugin


STYLE_COLUMNS = ("style_state",)


def _style_state_plugin() -> DataSourcePlugin:
    return DataSourcePlugin(
        name="style_state",
        dataset="style_state",
        join_keys=("trade_date", "symbol"),
        datetime_key="trade_date",
        instrument_key="symbol",
        column_map={"style_state": "style_state"},
        priority=70,
    )


def _make_wide(n: int = 400, k: int = 40) -> tuple[pd.DataFrame, ...]:
    """单调上行 size / value-growth 的合成宽表：暖机后应稳定落入已知枚举。"""
    idx = pd.date_range("2021-01-04", periods=n, freq="B")
    syms = [f"S{i:03d}" for i in range(k)]
    ci = pd.MultiIndex.from_product([idx, syms]).sort_values()
    dates, codes = ci.get_level_values(0), ci.get_level_values(1)
    close = pd.DataFrame(
        np.cumsum(np.abs(np.random.default_rng(1).standard_normal((n, k))) + 0.05, axis=0)
        + 10, index=idx, columns=syms,
    )
    amount = pd.DataFrame(
        np.random.default_rng(2).uniform(1e7, 1e9, (n, k)), index=idx, columns=syms
    )
    turnover = pd.DataFrame(
        np.random.default_rng(3).uniform(0.01, 0.10, (n, k)), index=idx, columns=syms
    )
    float_cap = pd.DataFrame(
        np.random.default_rng(4).uniform(1e9, 1e11, (n, k)), index=idx, columns=syms
    )
    pb = pd.DataFrame(
        np.random.default_rng(5).uniform(0.8, 5.0, (n, k)), index=idx, columns=syms
    )
    roe = pd.DataFrame(
        np.random.default_rng(6).uniform(0.02, 0.20, (n, k)), index=idx, columns=syms
    )
    return close, amount, turnover, float_cap, pb, roe


class TestAlwaysLoadColumns:
    def test_always_sets_contain_style_state(self) -> None:
        assert set(STYLE_COLUMNS) <= set(cnequity._ALWAYS_KEEP_COLUMNS)
        assert set(STYLE_COLUMNS) <= set(cnequity._ALWAYS_WANTED_AUX_COLUMNS)

    def test_focus_run_always_wants_style_state(self) -> None:
        """聚焦任意数据面时 style_state 也进入 wanted（不受面过滤影响）。"""
        wanted = cnequity._needed_aux_columns([_style_state_plugin()], {"价量面"})
        assert set(STYLE_COLUMNS) <= set(wanted)

    def test_facets_only_columns_still_filtered(self) -> None:
        """非中立列仍按面过滤：勾价量面不加载两融列，style_state 仍恒载。"""
        margin_plugin = DataSourcePlugin(
            name="margin",
            dataset="margin_detail",
            join_keys=("trade_date", "symbol"),
            datetime_key="trade_date",
            instrument_key="symbol",
            column_map={"mgn_balance": "mgn_balance"},
            priority=60,
        )

        plugins = [_style_state_plugin(), margin_plugin]
        wanted = cnequity._needed_aux_columns(plugins, {"价量面"})
        assert set(STYLE_COLUMNS) <= set(wanted)
        assert "mgn_balance" not in wanted


class TestPrunePanelColumns:
    def _panel(self) -> pd.DataFrame:
        idx = pd.MultiIndex.from_product(
            [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["000001.SZ", "000002.SZ"]],
            names=["datetime", "instrument"],
        )
        return pd.DataFrame(
            {"close": 1.0, "float_cap": 1.0, "style_state": 3.0, "funda_roe": 0.1},
            index=idx,
        )

    def test_prune_keeps_style_state(self) -> None:
        out = cnequity._prune_panel_columns(self._panel(), allowed={"价量面"})
        cols = set(out.columns)
        assert set(STYLE_COLUMNS) <= cols
        assert "funda_roe" not in cols
        assert "close" in cols and "float_cap" in cols


class TestStateFromWide:
    def test_output_domain_and_warmup(self) -> None:
        close, amount, turnover, float_cap, pb, roe = _make_wide()
        states = _state_from_wide(close, amount, turnover, float_cap, pb, roe)
        assert isinstance(states, pd.Series)
        assert (states.index == close.index).all()
        # 暖机段（前 120 日）应为 UNKNOWN → NaN
        assert states.iloc[:120].isna().all()
        valid = states.dropna()
        assert valid.index[0] > states.index[119]
        assert set(valid.unique()) <= set(_STATE_CODES.values())
        # 单调上行数据应产生成段稳定状态，不只 1-2 个散点
        assert len(valid) > 100

    def test_no_lookahead_last_day_perturbation(self) -> None:
        """T 日状态只用 (T-window, T-1] 数据 → 改最后一日输入不影响状态序列。"""
        close, amount, turnover, float_cap, pb, roe = _make_wide()
        base = _state_from_wide(close, amount, turnover, float_cap, pb, roe)
        alt_close = close.copy()
        alt_close.iloc[-1] = alt_close.iloc[-1] * 100 + 999
        alt_float = float_cap.copy()
        alt_float.iloc[-1] = alt_float.iloc[-1] * 1000
        alt = _state_from_wide(alt_close, amount, turnover, alt_float, pb, roe)
        pd.testing.assert_series_equal(base, alt)

    def test_empty_input_rejected(self) -> None:
        empty = pd.DataFrame()
        try:
            _state_from_wide(empty, empty, empty, empty, empty, empty)
        except ValueError:
            return
        raise AssertionError("空输入应抛 ValueError")
