# -*- coding: utf-8 -*-
"""拒单顺延补位（reject_fallback）开关的回归测试。

覆盖：
* 开关默认关闭（配置层）
* make_config 透传
* PortfolioBuilder.last_ordered 暴露完整候选降序（补位队列来源）
* 关闭时 last_ordered 不参与执行（由 simulate 的 rf_on 分支保证，
  这里断言默认值即可）
"""
from __future__ import annotations

import numpy as np
import pytest

from core.engine.config import BacktestConfig, run_backtest
from core.selection import PortfolioBuilder, SelectionPolicy


def test_config_default_is_off():
    """默认关闭 —— 保证与历史回测逐位可比。"""
    import dataclasses
    f = {x.name: x for x in dataclasses.fields(BacktestConfig)}["reject_fallback"]
    assert f.default is False


def test_run_backtest_accepts_switch():
    """run_backtest 必须接受 reject_fallback 参数并默认 False。"""
    import inspect
    sig = inspect.signature(run_backtest)
    assert "reject_fallback" in sig.parameters
    assert sig.parameters["reject_fallback"].default is False


def test_last_ordered_is_full_descending_selection():
    """last_ordered 必须是门控后按分数降序的**完整**候选，供补位顺延。"""
    codes = [f"00000{i}" for i in range(6)]
    builder = PortfolioBuilder(codes=codes)
    policy = SelectionPolicy(top_n=2)
    cand = np.array([0, 1, 2, 3, 4, 5])
    scores = np.array([0.1, 0.9, 0.5, 0.3, 0.8, 0.2])

    chosen, targets = builder.build_targets(policy, cand, scores)
    assert chosen == [1, 4]                     # 分数最高的两只
    assert set(targets) == {1, 4}
    # 全序：门控后 6 只，按分数降序
    assert builder.last_ordered == [1, 4, 2, 3, 5, 0]


def test_last_ordered_empty_when_no_candidate():
    builder = PortfolioBuilder(codes=["000001"])
    policy = SelectionPolicy(top_n=2, min_score=99.0)   # 全被门控拦掉
    chosen, targets = builder.build_targets(
        policy, np.array([0]), np.array([0.5]))
    assert chosen == [] and targets == {}
    assert builder.last_ordered == []


def test_last_ordered_respects_ascending():
    """买小模式：全序也必须是升序（与 rank_select 同序）。"""
    codes = [f"00000{i}" for i in range(4)]
    builder = PortfolioBuilder(codes=codes)
    policy = SelectionPolicy(top_n=2, ascending=True)
    cand = np.array([0, 1, 2, 3])
    scores = np.array([0.9, 0.1, 0.5, 0.3])
    chosen, _ = builder.build_targets(policy, cand, scores)
    assert chosen == [1, 3]
    assert builder.last_ordered == [1, 3, 2, 0]


def test_last_ordered_refreshed_per_call():
    """每次调用刷新，不残留上一次的全序。"""
    codes = [f"00000{i}" for i in range(3)]
    builder = PortfolioBuilder(codes=codes)
    policy = SelectionPolicy(top_n=1)
    builder.build_targets(policy, np.array([0, 1, 2]), np.array([0.9, 0.1, 0.5]))
    assert builder.last_ordered == [0, 2, 1]
    builder.build_targets(policy, np.array([0, 1, 2]), np.array([0.0, 0.2, 0.7]))
    assert builder.last_ordered == [2, 1, 0]
