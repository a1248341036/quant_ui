# -*- coding: utf-8 -*-
"""walk-forward 折边界平移（多路径评估的切法变体）测试。"""

import pandas as pd

from alphaagent.factor.stacking.model import walk_forward_splits


def _days():
    return pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=520))


def test_shift_months_moves_boundary_later():
    dts = _days()
    base = walk_forward_splits(dts, train_start="2024-01-01", train_months=6,
                               step_months=4, purge_days=5, shift_months=0)
    shifted = walk_forward_splits(dts, train_start="2024-01-01", train_months=6,
                                  step_months=4, purge_days=5, shift_months=2)
    assert base and shifted
    assert shifted[0].oos_dates.min() > base[0].oos_dates.min()
    # 平移只推迟 OOS 起点；train 起点不变（expanding 仍从 train_start 起）
    assert shifted[0].train_dates.min() == base[0].train_dates.min()


def test_shift_months_zero_matches_default():
    dts = _days()
    a = walk_forward_splits(dts, train_start="2024-01-01", train_months=6,
                            step_months=4, purge_days=5)
    b = walk_forward_splits(dts, train_start="2024-01-01", train_months=6,
                            step_months=4, purge_days=5, shift_months=0)
    assert [str(f.oos_dates.min().date()) for f in a] == [str(f.oos_dates.min().date()) for f in b]
