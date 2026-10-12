# -*- coding: utf-8 -*-
"""聚焦 run 恒载行业列：CNE adapter 辅助插件过滤/裁剪不得丢掉行业列（P0）。

背景：industry 插件（alphaagent/data/adapters/plugins/industry.py）已供给
industry_sw_l1 / industry_sw_code / ind_asof_days，但 _needed_aux_columns 按
"列属面 ⊆ 允许范围"过滤辅助插件，行业列不属任何数据面（_column_facets 为空），
聚焦 run 永远加载不到 → $industry_sw_l1 报「不可用字段」（2026-09-27 整夜 run
每 run 2~6 次）。本文件锁定：聚焦时行业列必须进入 wanted 名单并在裁剪后保留。
"""

from __future__ import annotations

import pandas as pd

from alphaagent.data.adapters import cnequity
from alphaagent.data.adapters.registry import DataSourcePlugin


INDUSTRY_COLUMNS = ("industry_sw_l1", "industry_sw_code", "ind_asof_days")


def _industry_plugin() -> DataSourcePlugin:
    return DataSourcePlugin(
        name="industry",
        dataset="industry_members",
        join_keys=("trade_date", "symbol"),
        datetime_key="trade_date",
        instrument_key="symbol",
        column_map={
            "industry_sw_l1": "industry_sw_l1",
            "industry_sw_code": "industry_sw_code",
            "ind_asof_days": "ind_asof_days",
        },
        priority=60,
    )


def _margin_plugin() -> DataSourcePlugin:
    return DataSourcePlugin(
        name="margin",
        dataset="margin_detail",
        join_keys=("trade_date", "symbol"),
        datetime_key="trade_date",
        instrument_key="symbol",
        column_map={"mgn_balance": "mgn_balance"},
        priority=60,
    )


class TestNeededAuxColumns:
    def test_focus_run_always_wants_industry_columns(self) -> None:
        """聚焦任意数据面时行业列也进入 wanted（不受面过滤影响）。"""
        wanted = cnequity._needed_aux_columns([_industry_plugin()], {"价量面"})
        assert set(INDUSTRY_COLUMNS) <= set(wanted)

    def test_facet_only_columns_still_filtered(self) -> None:
        """非中立列仍按面过滤：没勾两融面就不加载两融列。"""
        plugins = [_industry_plugin(), _margin_plugin()]
        wanted = cnequity._needed_aux_columns(plugins, {"价量面"})
        assert set(INDUSTRY_COLUMNS) <= set(wanted)
        assert "mgn_balance" not in wanted
        wanted_margin = cnequity._needed_aux_columns(plugins, {"两融面"})
        assert "mgn_balance" in wanted_margin


class TestPrunePanelColumns:
    def _panel(self) -> pd.DataFrame:
        idx = pd.MultiIndex.from_product(
            [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["000001.SZ", "000002.SZ"]],
            names=["datetime", "instrument"],
        )
        return pd.DataFrame(
            {
                "close": 1.0,
                "float_cap": 1.0,
                "industry_sw_l1": 48.0,
                "industry_sw_code": 480301.0,
                "ind_asof_days": 4.0,
                "funda_roe": 0.1,
                "mgn_balance": 1.0,
            },
            index=idx,
        )

    def test_prune_keeps_industry_columns(self) -> None:
        out = cnequity._prune_panel_columns(self._panel(), allowed={"价量面"})
        cols = set(out.columns)
        assert set(INDUSTRY_COLUMNS) <= cols
        assert "funda_roe" not in cols
        assert "mgn_balance" not in cols
        assert "close" in cols and "float_cap" in cols

    def test_always_sets_contain_industry_columns(self) -> None:
        assert set(INDUSTRY_COLUMNS) <= set(cnequity._ALWAYS_KEEP_COLUMNS)
        assert set(INDUSTRY_COLUMNS) <= set(cnequity._ALWAYS_WANTED_AUX_COLUMNS)
