"""core.score_matrix：因子分数 → 引擎 external_scores 矩阵的公共转换测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.score_matrix import scores_to_engine_matrix


def _dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2026-01-05", periods=4)


def _dsl_series() -> pd.Series:
    """模拟 AlphaAgent DSL 求值结果：MultiIndex(datetime, instrument)，带后缀。"""
    dates = _dates()
    mi = pd.MultiIndex.from_arrays(
        [list(dates), ["000001.SZ", "600000.SH", "300001.SZ", "600000.SH"]],
        names=["datetime", "instrument"],
    )
    return pd.Series([1.0, 2.0, 3.0, 4.0], index=mi, dtype="float32")


def test_series_input_with_suffix_alignment():
    """Series 输入：去后缀 + reindex 到目标日历/代码，缺失补 NaN。"""
    out = scores_to_engine_matrix(
        _dsl_series(),
        bt_dates=_dates(),
        bt_codes=["000001", "600000", "300001", "999999"],
    )
    assert list(out.columns) == ["000001", "600000", "300001", "999999"]
    assert out.loc[_dates()[0], "000001"] == 1.0
    assert out.loc[_dates()[1], "600000"] == 2.0
    assert out.loc[_dates()[2], "300001"] == 3.0
    assert np.isnan(out.loc[_dates()[0], "600000"])  # 未对齐日补 NaN
    assert np.isnan(out.loc[_dates()[0], "999999"])  # 不存在的代码补 NaN
    assert out.index.name is None or out.index.name == "datetime"


def test_frame_input_long_table():
    """DataFrame 长表输入：date/code/value 三列 → 矩阵。"""
    dates = _dates()
    long = pd.DataFrame({
        "date": list(dates),
        "code": ["000001", "600000", "300001", "000002"],
        "KMID": [10.0, 20.0, 30.0, 40.0],
    })
    out = scores_to_engine_matrix(long, value_col="KMID")
    assert out.shape == (4, 4)
    assert out.loc[dates[0], "000001"] == 10.0
    assert out.loc[dates[1], "600000"] == 20.0
    assert sorted(out.columns) == ["000001", "000002", "300001", "600000"]


def test_frame_input_duplicate_keeps_last():
    """重复 (date, code) 保留最后一条（与 qweave pivot aggfunc='last' 一致）。"""
    dates = _dates()
    long = pd.DataFrame({
        "date": [dates[0], dates[0], dates[1]],
        "code": ["000001", "000001", "600000"],
        "score": [1.0, 99.0, 2.0],
    })
    out = scores_to_engine_matrix(long)
    assert out.loc[dates[0], "000001"] == 99.0


def test_series_input_duplicate_keeps_last():
    mi = pd.MultiIndex.from_arrays(
        [[pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-05")],
         ["000001.SZ", "000001.SZ"]],
        names=["datetime", "instrument"],
    )
    s = pd.Series([1.0, 2.0], index=mi)
    out = scores_to_engine_matrix(s)
    assert out.loc[pd.Timestamp("2026-01-05"), "000001"] == 2.0


def test_rejects_flat_series():
    with pytest.raises(ValueError, match="scores_requires_multiindex"):
        scores_to_engine_matrix(pd.Series([1.0, 2.0]))


def test_rejects_missing_frame_columns():
    # 有 score 列但缺 date/code，触发缺列校验
    with pytest.raises(ValueError, match="scores_frame_missing_columns"):
        scores_to_engine_matrix(pd.DataFrame({"date": [], "score": []}))


def test_rejects_bad_value_col():
    with pytest.raises(ValueError, match="scores_frame_missing_value_column"):
        scores_to_engine_matrix(pd.DataFrame({"date": [], "code": []}), value_col="nope")


def test_rejects_invalid_datetime():
    mi = pd.MultiIndex.from_arrays(
        [["not-a-date", "2026-01-06"], ["000001.SZ", "600000.SH"]],
        names=["datetime", "instrument"],
    )
    with pytest.raises(ValueError, match="scores_invalid_datetime"):
        scores_to_engine_matrix(pd.Series([1.0, 2.0], index=mi))


def test_no_alignment_keeps_source_calendar():
    """缺省不传 bt_dates/bt_codes：直接 unstack 源日历/代码（引擎侧自行对齐）。"""
    out = scores_to_engine_matrix(_dsl_series())
    assert len(out) == 4
    assert sorted(out.columns) == ["000001", "300001", "600000"]