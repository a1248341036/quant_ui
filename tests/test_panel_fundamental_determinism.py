# -*- coding: utf-8 -*-
"""``fundamental`` adapter asof 合并的**确定性**回归测试。

背景（P1）：``alphaagent.data.adapters.plugins.fundamental.load()`` 用
``join_asof(backward)`` 按公告日 ``ann_date`` 把季频财报贴到日频网格上。
同一 symbol、同一 ``ann_date`` 可能**同时公告两期**报表（如 000006.SZ 在
2024-04-29 同时披露 2023 年报 + 2024Q1）。修复前 tie-break 取决于右表**物理行序**
（polars ``group_by`` / 多 parquet 文件读取顺序不确定）→ 同一进程连续两次 ``load()``
在 149,292 行中有 10,178 行取值不同（如 ``funda_total_assets`` 259.19 亿 vs 257.65 亿），
面板不可复现。

本测试的语义约定：同一公告日出现多期时，取**报告期更晚**（``end_date`` 更大）的那一期。
这不是前视偏差——两期在当天都已公开；只是把 tie-break 钉死为数据决定的确定值。
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import polars as pl

from alphaagent.data.adapters.plugins import fundamental as funda_plugin

_SYMBOL = "000006.SZ"
_ANN_DATE = datetime.date(2024, 4, 29)  # 两期同一公告日
_END_OLD = datetime.date(2023, 12, 31)  # 2023 年报
_END_NEW = datetime.date(2024, 3, 31)  # 2024Q1
_ASSETS_OLD = 2.5919e10
_ASSETS_NEW = 2.5765e10


def _synthetic_curated(order: str):
    """构造最小 curated 表：同一 symbol / 同一 ann_date / 两个不同 end_date 的报告期。

    ``order`` 控制物理行序（``"fwd"`` / ``"rev"``）——修复前它能直接改变 asof tie-break。
    """

    def fake_read(dataset: str) -> pl.DataFrame:
        if dataset != "balancesheet":
            raise FileNotFoundError(f"测试未提供数据集: {dataset}")
        ends = [_END_OLD, _END_NEW]
        assets = [_ASSETS_OLD, _ASSETS_NEW]
        if order == "rev":
            ends, assets = ends[::-1], assets[::-1]
        return pl.DataFrame(
            {
                "symbol": pl.Series([_SYMBOL] * 2, dtype=pl.Utf8),
                "end_date": pl.Series(ends, dtype=pl.Date),
                "ann_date": pl.Series([_ANN_DATE] * 2, dtype=pl.Date),
                "total_assets": assets,
            }
        )

    return fake_read


def _load(monkeypatch, order: str) -> pd.DataFrame:
    monkeypatch.setattr(funda_plugin, "_read_curated", _synthetic_curated(order))
    return funda_plugin.load("fundamental", start="2024-01-01", end="2024-06-30")


def test_consecutive_loads_are_value_identical(monkeypatch):
    """核心断言：连续两次 load() 结果**逐值一致**（右侧物理行序不得影响结果）。

    为了让"行序不确定"这一真实故障模式在单进程内可复现，第二次 load 的 curated 行序刻意反转
    （模拟 polars group_by / 多 parquet 文件读取的行序漂移）。
    """
    first = _load(monkeypatch, "fwd")
    second = _load(monkeypatch, "rev")

    assert not first.empty, "探针数据应展开出日频行"
    assert list(first.columns) == list(second.columns)

    # 逐值精确一致（含浮点值，不使用容差）
    pd.testing.assert_frame_equal(first, second, check_exact=True)


def test_same_ann_date_takes_later_end_date(monkeypatch):
    """同一公告日两期：取得 ``end_date`` 更大（报告期更晚）那一期。"""
    out = _load(monkeypatch, "fwd")
    df = out.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    # 公告日之前无任何行（两期均在 2024-04-29 才公开）
    assert (df["trade_date"] < pd.Timestamp(_ANN_DATE)).sum() == 0

    announced = df[df["trade_date"] >= pd.Timestamp(_ANN_DATE)]
    assert not announced.empty
    np.testing.assert_allclose(
        announced["funda_total_assets"].to_numpy(dtype=float),
        _ASSETS_NEW,
        rtol=0,
        err_msg="tie-break 取到了报告期更早的那一期（应取 end_date 更大的一期）",
    )
    # 归一化到 datetime.date 再比较（pandas Timestamp 与 date 的 hash 不同，不能用 set 直比）
    for col, expected in (("end_date", _END_NEW), ("ann_date", _ANN_DATE)):
        if col in announced.columns:
            got = pd.to_datetime(announced[col]).dt.date.unique().tolist()
            assert got == [expected], f"{col} 应唯一为 {expected}，实际 {got}"


def test_result_independent_of_physical_row_order(monkeypatch):
    """行序反转不得改变任何一行取值（跨两次 load 逐值比对）。"""
    fwd = _load(monkeypatch, "fwd")
    rev = _load(monkeypatch, "rev")

    pd.testing.assert_frame_equal(fwd, rev, check_exact=True)
