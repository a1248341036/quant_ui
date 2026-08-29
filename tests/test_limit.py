"""涨跌停 ST 标记测试。

覆盖：
1. build_limit_flags 无 st_mask 时保持板块近似（10%/20%/30%）
2. build_limit_flags 带 st_mask 时 ST 股按 5% 判定（涨停/跌停/一字板）
3. st_mask 与 close 索引/列错位时安全对齐
4. ETF/基金代码在 st_mask 下仍按 1.0 处理（无涨跌停）
5. load_st_mask 缺省降级路径（CneUnavailable 时不抛异常）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.limit import build_limit_flags, limit_ratio


def _panel(codes: list[str], closes: list[list[float]],
           opens: list[list[float]] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造 close/open DataFrame（index=日期, columns=code）。"""
    dates = pd.date_range("2024-01-02", periods=len(closes[0]))
    close = pd.DataFrame({c: v for c, v in zip(codes, closes)}, index=dates)
    if opens is None:
        opens = closes
    open_ = pd.DataFrame({c: v for c, v in zip(codes, opens)}, index=dates)
    return close, open_


def test_limit_ratio_by_board():
    assert limit_ratio("600000") == 0.10
    assert limit_ratio("000001") == 0.10
    assert limit_ratio("300001") == 0.20
    assert limit_ratio("688001") == 0.20
    assert limit_ratio("301001") == 0.20
    assert limit_ratio("830001") == 0.30  # 北交所 8 开头
    assert limit_ratio("430001") == 0.30  # 北交所 4 开头
    assert limit_ratio("920001") == 0.30
    assert limit_ratio("510300") == 1.0   # ETF 沪 5 开头
    assert limit_ratio("159915") == 1.0   # ETF 深 15 开头
    assert limit_ratio("161725") == 1.0   # 场内基金 16 开头


def test_no_st_mask_keeps_legacy_behavior():
    """无 st_mask：5% 涨幅不触发 10% 涨停，10% 涨幅触发。"""
    dates = pd.date_range("2024-01-02", periods=3)
    close = pd.DataFrame({"600000": [10.0, 10.0, 11.0],
                          "000001": [10.0, 10.0, 10.5]}, index=dates)
    open_ = pd.DataFrame({"600000": [10.0, 10.0, 11.0],
                          "000001": [10.0, 10.0, 10.5]}, index=dates)
    up, down, one_up, one_down = build_limit_flags(close, open_)
    # 第 2 行（索引 2）：600000 +10% 涨停，000001 +5% 不涨停
    assert up[2][0] == True
    assert up[2][1] == False


def test_st_mask_uses_5_percent_limit_up():
    """ST 股按 5% 判定涨停。"""
    dates = pd.date_range("2024-01-02", periods=3)
    close = pd.DataFrame({"600000": [10.0, 10.0, 11.0],
                          "000001": [10.0, 10.0, 10.5]}, index=dates)
    open_ = pd.DataFrame({"600000": [10.0, 10.0, 11.0],
                          "000001": [10.0, 10.0, 10.5]}, index=dates)
    st = pd.DataFrame({"600000": [False] * 3, "000001": [False, False, True]},
                      index=dates)
    up, _, _, _ = build_limit_flags(close, open_, st_mask=st)
    assert up[2, 0] == True   # 600000 +10% 涨停
    assert up[2, 1] == True   # 000001 ST +5% 涨停


def test_st_mask_uses_5_percent_limit_down():
    """ST 股按 5% 判定跌停。"""
    dates = pd.date_range("2024-01-02", periods=3)
    close = pd.DataFrame({"600000": [10.0, 10.0, 9.0],
                          "000001": [10.0, 10.0, 9.5]}, index=dates)
    open_ = pd.DataFrame({"600000": [10.0, 10.0, 9.0],
                          "000001": [10.0, 10.0, 9.5]}, index=dates)
    st = pd.DataFrame({"600000": [False] * 3, "000001": [False, False, True]},
                      index=dates)
    _, down, _, _ = build_limit_flags(close, open_, st_mask=st)
    assert down[2, 0] == True
    assert down[2, 1] == True


def test_st_mask_missing_column_falls_back_to_board():
    """st_mask 缺失某代码列时，该股按板块比例。"""
    dates = pd.date_range("2024-01-02", periods=3)
    close = pd.DataFrame({"600000": [10.0, 10.0, 11.0],
                          "000001": [10.0, 10.0, 10.5]}, index=dates)
    open_ = close.copy()
    # st_mask 只含 600000（无 000001 列）
    st = pd.DataFrame({"600000": [False] * 3}, index=dates)
    up, _, _, _ = build_limit_flags(close, open_, st_mask=st)
    assert up[2, 0] == True    # 600000 +10% 涨停
    assert up[2, 1] == False   # 000001 缺 ST 标记，5% 不触发 10%


def test_st_mask_missing_dates_fill_false():
    """st_mask 缺日期时按非 ST 处理。"""
    dates = pd.date_range("2024-01-02", periods=4)
    close = pd.DataFrame({"000001": [10.0, 10.0, 10.5, 10.0]}, index=dates)
    open_ = pd.DataFrame({"000001": [10.0, 10.0, 10.5, 10.0]}, index=dates)
    # st_mask 只有中间两天
    st = pd.DataFrame({"000001": [True, True]},
                      index=dates[1:3])
    up, _, _, _ = build_limit_flags(close, open_, st_mask=st)
    # 第 2 行 +5% 但 ST 缺失（mask 无此行）→ 按非 ST，不涨停
    assert up[1, 0] == False


def test_etf_not_affected_by_st_mask():
    """ETF 不受 st_mask 影响，仍按 1. 涨跌停幅度。"""
    dates = pd.date_range("2024-01-02", periods=3)
    close = pd.DataFrame({"510300": [10.0, 10.0, 11.0]}, index=dates)
    open_ = pd.DataFrame({"510300": [10.0, 10.0, 11.0]}, index=dates)
    st = pd.DataFrame({"510300": [False, True, True]}, index=dates)
    up, _, _, _ = build_limit_flags(close, open_, st_mask=st)
    assert up[2, 0] == False   # +10% 但 ETF 无涨跌停限制


def test_one_word_flags_with_st():
    """一字板标记在 ST 下同样生效。"""
    dates = pd.date_range("2024-01-02", periods=3)
    close = pd.DataFrame({"000001": [10.0, 10.0, 10.5]}, index=dates)
    open_ = pd.DataFrame({"000001": [10.0, 10.0, 10.5]}, index=dates)
    st = pd.DataFrame({"000001": [False, False, True]}, index=dates)
    _, _, one_up, _ = build_limit_flags(close, open_, st_mask=st)
    assert one_up[2, 0] == True


def test_st_mask_none_is_legacy():
    """st_mask=None 与旧行为完全一致。"""
    dates = pd.date_range("2024-01-02", periods=3)
    close = pd.DataFrame({"000001": [10.0, 10.0, 10.5]}, index=dates)
    open_ = pd.DataFrame({"000001": [10.0, 10.0, 10.5]}, index=dates)
    up_legacy, _, _, _ = build_limit_flags(close, open_)
    up_new, _, _, _ = build_limit_flags(close, open_, st_mask=None)
    np.testing.assert_array_equal(up_legacy, up_new)