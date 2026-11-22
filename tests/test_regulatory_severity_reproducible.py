# -*- coding: utf-8 -*-
"""监管事件插件：reg_severity 必须与「请求右端」无关，且等于截至当日的最高严重度。

回归背景（2026-11-22）：events 来自 group_by().agg()，输出顺序不确定；随后
cum_max().over("symbol") 按行序累积，导致同一交易日的 reg_severity 随 end 变化
（实测 end=2026-10-15 得 4、end=2026-10-29 得 3）。修复 = 聚合后显式排序。
"""
import datetime as dt

import polars as pl
import pytest

from alphaagent.data.adapters.plugins import _pitlib
from alphaagent.data.adapters.plugins import regulatory_events as reg


def _events() -> pl.DataFrame:
    # 故意用非时间顺序构造，模拟 group_by 输出乱序
    rows = [
        ("000002.SZ", dt.date(2024, 3, 5), "penalty"),
        ("000001.SZ", dt.date(2024, 2, 20), "disciplinary"),
        ("000001.SZ", dt.date(2024, 1, 10), "warning_letter"),
        ("000002.SZ", dt.date(2024, 1, 5), "warning_letter"),
        ("000001.SZ", dt.date(2024, 4, 1), "regulatory_letter"),
    ]
    return pl.DataFrame(
        {
            "symbol": [r[0] for r in rows],
            "event_date": [r[1] for r in rows],
            "event_type": [r[2] for r in rows],
        },
        schema={"symbol": pl.Utf8, "event_date": pl.Date, "event_type": pl.Utf8},
    )


@pytest.fixture()
def fake_source(monkeypatch):
    def _read(dataset, *, start=None, end=None):
        df = _events()
        s, e = _pitlib.parse_window(start, end)
        return df.filter((pl.col("event_date") >= s) & (pl.col("event_date") <= e))

    monkeypatch.setattr(_pitlib, "read_curated", _read)


def test_severity_is_step_max_of_history(fake_source):
    out = reg.load("regulatory_events", start="2024-01-01", end="2024-06-30")
    got = out.set_index(["trade_date", "symbol"])["reg_severity"]

    # 000001.SZ：01-10 警告(1) → 02-20 纪律处分(4) → 04-01 监管函(2)
    # 阶梯最大值应保持 4，不会被后发生的低severity事件拉低
    v = got.xs("000001.SZ", level="symbol")
    assert v.loc[pd_timestamp("2024-01-10")] == 1
    assert v.loc[pd_timestamp("2024-02-20")] == 4
    assert v.loc[pd_timestamp("2024-06-28")] == 4

    # 000002.SZ：01-05 警告(1) → 03-05 处罚(3)
    v2 = got.xs("000002.SZ", level="symbol")
    assert v2.loc[pd_timestamp("2024-01-05")] == 1
    assert v2.loc[pd_timestamp("2024-03-05")] == 3


def test_severity_independent_of_request_end(fake_source):
    a = reg.load("regulatory_events", start="2024-01-01", end="2024-04-15")
    b = reg.load("regulatory_events", start="2024-01-01", end="2024-06-30")
    m = a.merge(b, on=["trade_date", "symbol"], suffixes=("_a", "_b"), how="inner")
    x, y = m["reg_severity_a"], m["reg_severity_b"]
    bad = m[(x != y) & ~(x.isna() & y.isna())]
    assert len(bad) == 0, bad.head(10).to_string()


def pd_timestamp(s: str):
    import pandas as pd

    return pd.Timestamp(s)
