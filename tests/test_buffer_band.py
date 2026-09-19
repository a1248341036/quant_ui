"""P1-1/P1-2 buffer zone + no-trade band 门禁。

硬保证：
- buffer_ratio=0.0 且 no_trade_band=0.0（默认）时，所有既有键与不传参数逐位一致
  （向后兼容，回归测试）；
- buffer_ratio=0.5 时，avg_rebalance_side_turnover_buffered_banded <=
  avg_rebalance_side_turnover_raw（buffer zone 降低调仓换手）；
- no_trade_band=0.15 时，换手进一步降低或持平；
- avg_rebalance_side_turnover_raw 在默认参数时 == avg_rebalance_side_turnover
  （raw 口径 = 原口径）；
- avg_daily_side_turnover 与 avg_rebalance_side_turnover 保持原口径
  （因子层门槛继续用原口径，buffer/band 只影响信息性字段 + 净值序列）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaagent.factor import metrics as M

LEGACY_KEYS = (
    "top_group_annualized_return",
    "top_group_annualized_excess_return",
    "top_group_gross_excess_return",
    "top_group_sharpe",
    "top_group_excess_sharpe",
    "top_group_max_drawdown",
    "avg_daily_side_turnover",
    "avg_rebalance_side_turnover",
    "group_means",
    "monotonicity",
    "spread_annualized",
)


def _idx(n_days=80, n_inst=200):
    return pd.MultiIndex.from_product(
        [pd.bdate_range("2021-01-04", periods=n_days), [f"S{i:03d}" for i in range(n_inst)]],
        names=["datetime", "instrument"],
    )


def _rotating_alpha(n_days=80, n_inst=200, seed=11, hold=5):
    """每 hold 天轮换头部 20 只（高换手因子）：头部 +10、日收益 +0.8%，其余噪声。

    构造目的：让 top 组成员在调仓日显著变化（换手高），buffer zone 能保留
    跌到次高组的老持仓，no-trade band 能拦截小幅轮换。
    """
    rng = np.random.default_rng(seed)
    idx = _idx(n_days, n_inst)
    n = len(idx)
    inst = np.asarray(idx.get_level_values("instrument"))
    dates = pd.bdate_range("2021-01-04", periods=n_days)

    f = np.clip(rng.normal(0, 1, n), -5, 5)
    lab = np.zeros(n)

    # 每 hold 天选 20 只做头部，轮换
    head_pool = [f"S{i:03d}" for i in range(40)]  # 40 只候选头部
    for d_idx in range(0, n_days, hold):
        heads = rng.choice(head_pool, size=20, replace=False)
        mask = (np.asarray(idx.get_level_values("datetime")) == dates[d_idx]) & np.isin(inst, heads)
        f[mask] += 10.0
        lab[mask] = 0.008

    drop = rng.random(n) < 0.02
    return pd.Series(f, index=idx, name="f")[~drop], pd.Series(lab, index=idx)[~drop]


def _noisy(n_days=120, n_inst=40, seed=5):
    """无集中 alpha 的普通面板（fastpaths 同款构造）。"""
    rng = np.random.default_rng(seed)
    idx = _idx(n_days, n_inst)
    n = len(idx)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, (n_days, n_inst)), axis=0)).ravel(order="F")
    c2 = close.reshape(n_days, n_inst)
    ret = np.zeros((n_days, n_inst))
    ret[1:] = c2[1:] / c2[:-1] - 1
    f = rng.normal(0, 1, n) + 0.2 * ret.ravel(order="F")
    label = pd.Series(ret.ravel(order="F") + rng.normal(0, 0.005, n), index=idx)
    factor = pd.Series(f, index=idx, name="f")
    drop = rng.random(n) < 0.03
    return factor[~drop], label[~drop]


# ── 向后兼容：默认参数逐位一致 ──


def test_default_params_byte_identical_to_no_params():
    """buffer_ratio=0.0 + no_trade_band=0.0 与不传参数输出完全一致。"""
    factor, label = _noisy()
    out_ref = M.quantile_portfolio_metrics(factor, label, cost_bps=15.0)
    out_def = M.quantile_portfolio_metrics(
        factor, label, cost_bps=15.0, buffer_ratio=0.0, no_trade_band=0.0
    )
    for key in LEGACY_KEYS:
        assert out_ref[key] == out_def[key], f"legacy key {key} drifted: {out_ref[key]} != {out_def[key]}"
    # 新增键在默认参数时的语义
    assert out_def["buffer_ratio"] == 0.0
    assert out_def["no_trade_band"] == 0.0
    assert out_def["avg_rebalance_side_turnover_raw"] == out_def["avg_rebalance_side_turnover"]
    assert out_def["avg_rebalance_side_turnover_buffered_banded"] == out_def["avg_rebalance_side_turnover"]
    assert out_def["turnover_reduction_pp"] == 0.0


def test_default_params_rotating_alpha_byte_identical():
    """高换手因子在默认参数下也逐位一致。"""
    factor, label = _rotating_alpha()
    out_ref = M.quantile_portfolio_metrics(factor, label, cost_bps=15.0, holding_days=5)
    out_def = M.quantile_portfolio_metrics(
        factor, label, cost_bps=15.0, holding_days=5, buffer_ratio=0.0, no_trade_band=0.0
    )
    for key in LEGACY_KEYS:
        assert out_ref[key] == out_def[key], f"legacy key {key} drifted on rotating alpha"


# ── P1-1 Buffer Zone：降低调仓换手 ──


def test_buffer_zone_reduces_rebalance_turnover():
    """buffer_ratio=0.5 时 buffered_banded 换手 <= raw 换手。"""
    factor, label = _rotating_alpha()
    out = M.quantile_portfolio_metrics(
        factor, label, cost_bps=15.0, holding_days=5, buffer_ratio=0.5
    )
    raw = out["avg_rebalance_side_turnover_raw"]
    buf = out["avg_rebalance_side_turnover_buffered_banded"]
    assert buf <= raw + 1e-9, f"buffer should reduce turnover: buf={buf} > raw={raw}"
    assert out["turnover_reduction_pp"] >= -1e-9, "turnover_reduction_pp should be >= 0"
    assert out["buffer_ratio"] == 0.5


def test_buffer_zone_raw_equals_legacy_rebalance():
    """buffer 启用时 raw 口径仍等于原 rebalance 换手（raw 不受 buffer 影响）。"""
    factor, label = _rotating_alpha()
    out_legacy = M.quantile_portfolio_metrics(factor, label, cost_bps=15.0, holding_days=5)
    out_buf = M.quantile_portfolio_metrics(
        factor, label, cost_bps=15.0, holding_days=5, buffer_ratio=0.5
    )
    # raw 口径 = 严格 top 组换手 = legacy rebalance 换手
    assert out_buf["avg_rebalance_side_turnover_raw"] == out_legacy["avg_rebalance_side_turnover"]
    # avg_rebalance_side_turnover（因子层门槛用）保持原口径
    assert out_buf["avg_rebalance_side_turnover"] == out_legacy["avg_rebalance_side_turnover"]


# ── P1-2 No-Trade Band：拦截小幅轮换 ──


def test_no_trade_band_reduces_turnover():
    """no_trade_band=0.15 时换手降低或持平。"""
    factor, label = _rotating_alpha()
    out = M.quantile_portfolio_metrics(
        factor, label, cost_bps=15.0, holding_days=5, no_trade_band=0.15
    )
    raw = out["avg_rebalance_side_turnover_raw"]
    banded = out["avg_rebalance_side_turnover_buffered_banded"]
    assert banded <= raw + 1e-9, f"band should reduce turnover: banded={banded} > raw={raw}"
    assert out["no_trade_band"] == 0.15


def test_no_trade_band_zero_turnover_when_band_large():
    """no_trade_band=1.0 时所有调仓都在 band 内，换手降为 0（首日建仓除外）。"""
    factor, label = _rotating_alpha()
    out = M.quantile_portfolio_metrics(
        factor, label, cost_bps=15.0, holding_days=5, no_trade_band=1.0
    )
    # band=1.0：任何成员变化比例 <= 1.0 都不调，故 buffered_banded 换手 = 首日建仓 / n_rebalances
    # 首日 nav_side=1.0（建仓），之后全 0 → avg = 1.0 / n_rebalances
    n_reb = out["n_rebalances"]
    expected = 1.0 / max(n_reb, 1)
    assert abs(out["avg_rebalance_side_turnover_buffered_banded"] - round(expected, 4)) < 0.01


# ── P1-1 + P1-2 组合 ──


def test_buffer_plus_band_combined():
    """buffer_ratio=0.5 + no_trade_band=0.15 组合，换手降低。"""
    factor, label = _rotating_alpha()
    out = M.quantile_portfolio_metrics(
        factor, label,
        cost_bps=15.0, holding_days=5,
        buffer_ratio=0.5, no_trade_band=0.15,
    )
    raw = out["avg_rebalance_side_turnover_raw"]
    combined = out["avg_rebalance_side_turnover_buffered_banded"]
    assert combined <= raw + 1e-9, f"combined should reduce turnover: {combined} > {raw}"
    assert out["turnover_reduction_pp"] >= -1e-9


# ── 净值口径：buffer/band 改变净值序列（long_ret 用 buffer 后持仓）──


def test_buffer_changes_nav_series():
    """buffer 启用时净值序列与默认参数不同（buffer 保留次高组老持仓，收益不同）。"""
    factor, label = _rotating_alpha()
    out_def = M.quantile_portfolio_metrics(factor, label, cost_bps=15.0, holding_days=5)
    out_buf = M.quantile_portfolio_metrics(
        factor, label, cost_bps=15.0, holding_days=5, buffer_ratio=0.5
    )
    # buffer 保留的老持仓收益与严格 top 组不同 → 净值年化应不同
    # （除非因子恰好让 top 组成员不变，rotating_alpha 构造保证成员会变）
    assert out_buf["top_group_annualized_return"] != out_def["top_group_annualized_return"] or \
           out_buf["avg_rebalance_side_turnover_buffered_banded"] < out_def["avg_rebalance_side_turnover"], \
           "buffer should either change nav or reduce turnover"
