# 回归：@1w 周线广播在含重复日期的 MultiIndex 上不再崩溃（BUG A）
import numpy as np
import pandas as pd

from alphaagent.dsl.stock.resample import (
    broadcast_timeframe_to_main_freq,
    build_timeframe_panel,
)


def test_weekly_broadcast_unique_level_fix():
    # 构造 10 个交易日 × 3 只股票的日频面板（datetime level 天然大量重复）
    dates = pd.bdate_range("2024-01-02", periods=10)
    insts = ["A", "B", "C"]
    idx = pd.MultiIndex.from_product([dates, insts], names=["datetime", "instrument"])
    rng = np.random.default_rng(7)
    n = len(idx)
    panel = pd.DataFrame(
        {
            "adj_vwap": rng.random(n) + 10,
            "volume": rng.integers(100, 1000, n).astype(float),
            "amount": rng.random(n) * 1e6,
            "adj_close": rng.random(n) + 10,
        },
        index=idx,
    )

    weekly = build_timeframe_panel(panel, target_interval="1w")
    assert isinstance(weekly.index, pd.MultiIndex), weekly.index

    out = broadcast_timeframe_to_main_freq(weekly, panel.index, "1w")
    assert out.index.equals(panel.index), "广播后索引必须与日频目标索引一致"
    assert len(out) == n
    assert out["adj_vwap"].notna().any(), "广播值不应全空"
