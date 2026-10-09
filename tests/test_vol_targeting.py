"""Vol targeting 缩放层单测。

覆盖：
- portfolio_vol 数学正确性（滚动已实现波动、年化、窗口切片、停牌剔除）
  与安全退化（None/空表/缺列/窗口不足/NaN/零收益）；
- SelectionPolicy.vol_scale_for 缩放数学（σ_p 高→<1 降仓、低→hi 封顶、
  恰等→1、功能关/无效 σ_p→1 恒等）；
- build_targets 权重出口集成（缩放生效、开关关=恒等、σ_p=0/NaN/窗口不足
  安全退化、与 regime_scale 叠加后整体 clip、builder 信号日游标通道无前视）；
- BacktestConfig 字段默认值 + run_backtest 门面透传 + 端到端行为差异。
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from core.engine.config import BacktestConfig, run_backtest
from core.portfolio import portfolio_vol
from core.selection import PortfolioBuilder, SelectionPolicy
from conftest import CODES, END, START


# ── 构造辅助 ──────────────────────────────────────────────────────────


def _wide(spec: dict[str, float], n: int = 80,
          start: str = "2024-01-02") -> pd.DataFrame:
    """多票收盘价宽表：每票日收益按全局奇偶交替 ±ret（同相位）。"""
    dates = pd.bdate_range(start, periods=n)
    out = {}
    for code, ret in spec.items():
        rets = np.array([ret if i % 2 == 0 else -ret for i in range(n)])
        out[code] = 10.0 * np.cumprod(1.0 + rets)
    return pd.DataFrame(out, index=dates)


def _run_targets(policy, close, market_adx=None, codes=("a", "b")):
    """经 build_targets 权重出口跑一遍：全选、等权、显式 close_wide 通道。"""
    builder = PortfolioBuilder(list(codes))
    cand = np.arange(len(codes))
    scores = np.array([1.0, 0.5][: len(cand)])
    chosen, targets = builder.build_targets(policy, cand, scores, market_adx,
                                            close_wide=close)
    return chosen, targets


# ── portfolio_vol：σ_p 预测器 ─────────────────────────────────────────


def test_portfolio_vol_math_and_annualization():
    close = _wide({"a": 0.02}, n=80)
    sigma = portfolio_vol(close, window=60)
    assert np.isfinite(sigma)
    # 交替 ±2% 日收益 → σ_p ≈ 0.02·√252
    assert sigma == pytest.approx(0.02 * np.sqrt(252), rel=0.05)


def test_portfolio_vol_multi_code_equal_weight():
    close = _wide({"a": 0.02, "b": 0.06}, n=80)
    only_a = portfolio_vol(close, window=60, codes=["a"])
    assert only_a == pytest.approx(0.02 * np.sqrt(252), rel=0.05)
    # 等权两票（同相位 ±0.02 / ±0.06）→ 组合日收益 ±0.04
    both = portfolio_vol(close, window=60)
    assert both == pytest.approx(0.04 * np.sqrt(252), rel=0.05)


def test_portfolio_vol_uses_trailing_window_only():
    # 前 30 行 ±10% 大波动、后 50 行 ±2%：window=50 只看尾部 → 小 σ
    rets = np.concatenate([
        np.where(np.arange(30) % 2 == 0, 0.10, -0.10),
        np.where(np.arange(50) % 2 == 0, 0.02, -0.02),
    ])
    close = pd.DataFrame(
        {"a": 10.0 * np.cumprod(1.0 + rets)},
        index=pd.bdate_range("2024-01-02", periods=80))
    tail_vol = portfolio_vol(close, window=50)
    assert tail_vol == pytest.approx(0.02 * np.sqrt(252), rel=0.05)
    # 全窗口估计包含大波动段 → 明显更大
    all_vol = portfolio_vol(close, window=79)
    assert all_vol > 2 * tail_vol


def test_portfolio_vol_suspension_days_excluded():
    close = _wide({"a": 0.02}, n=80)
    close.iloc[:40, 0] = np.nan  # 前 40 天停牌
    # 后 40 天有 39 个有效收益（>= min_obs=30）→ 可估 σ_p，NaN 段被剔除
    sigma = portfolio_vol(close, window=60)
    assert np.isfinite(sigma)
    assert sigma == pytest.approx(0.02 * np.sqrt(252), rel=0.05)


def test_portfolio_vol_safe_degradation():
    assert np.isnan(portfolio_vol(None, window=60))
    assert np.isnan(portfolio_vol(pd.DataFrame(), window=60))
    assert np.isnan(portfolio_vol(_wide({"a": 0.02}), window=0))
    close = _wide({"a": 0.02}, n=80)
    assert np.isnan(portfolio_vol(close, window=60, codes=["zzz"]))  # 缺列
    assert np.isnan(portfolio_vol(close.head(20), window=60))        # 窗口不足
    flat = pd.DataFrame({"a": 10.0},
                        index=pd.bdate_range("2024-01-02", periods=80))
    assert portfolio_vol(flat, window=60) == 0.0  # 零收益 → σ_p=0（非 NaN）
    nan_frame = pd.DataFrame({"a": np.nan},
                             index=pd.bdate_range("2024-01-02", periods=80))
    assert np.isnan(portfolio_vol(nan_frame, window=60))  # 全 NaN


# ── SelectionPolicy.vol_scale_for：缩放数学 ──────────────────────────


def test_vol_scale_for_math():
    p = SelectionPolicy(vol_target_annual=0.20)
    assert p.vol_scale_for(0.40) == pytest.approx(0.5)    # σ_p 高 → 降仓
    assert p.vol_scale_for(1.00) == pytest.approx(0.3)    # → clip 到 lo
    assert p.vol_scale_for(0.10) == pytest.approx(1.5)    # σ_p 低 → 封顶 hi
    assert p.vol_scale_for(0.20) == pytest.approx(1.0)    # 恰等 → 恒等


def test_vol_scale_for_invalid_sigma_and_disabled():
    p = SelectionPolicy(vol_target_annual=0.20)
    for bad in (None, float("nan"), float("inf"), 0.0, -0.1):
        assert p.vol_scale_for(bad) == 1.0  # 无效 σ_p → 安全退化恒等
    assert SelectionPolicy().vol_scale_for(10.0) == 1.0     # 功能关 → 恒等


# ── build_targets：权重出口集成 ───────────────────────────────────────


def test_build_targets_scales_down_when_sigma_above_target():
    close = _wide({"a": 0.05, "b": 0.05}, n=80)
    sigma = portfolio_vol(close, window=60)
    # target = 0.4·σ_p → vol_scale = 0.4 → 权重 0.4/2
    policy = SelectionPolicy(vol_target_annual=0.4 * sigma)
    chosen, targets = _run_targets(policy, close)
    assert chosen == [0, 1]
    assert targets[0] == pytest.approx(0.4 / 2)
    assert targets[1] == pytest.approx(0.4 / 2)


def test_build_targets_caps_at_hi_when_sigma_below_target():
    close = _wide({"a": 0.002, "b": 0.002}, n=80)  # σ_p ≈ 0.002·√252
    policy = SelectionPolicy(vol_target_annual=0.20)  # 远高于 σ_p
    _, targets = _run_targets(policy, close)
    assert targets[0] == pytest.approx(1.5 / 2)
    assert targets[1] == pytest.approx(1.5 / 2)


def test_build_targets_identity_when_sigma_equals_target():
    close = _wide({"a": 0.05, "b": 0.05}, n=80)
    sigma = portfolio_vol(close, window=60)
    policy = SelectionPolicy(vol_target_annual=sigma)
    _, targets = _run_targets(policy, close)
    assert targets == {0: pytest.approx(0.5), 1: pytest.approx(0.5)}


def test_build_targets_identity_when_disabled():
    close = _wide({"a": 0.05, "b": 0.05}, n=80)
    # 开关关（默认）：即使传入 close_wide 也完全恒等
    _, targets = _run_targets(SelectionPolicy(), close)
    assert targets == {0: 0.5, 1: 0.5}
    # 开关开但无任何数据通道（builder 未注入 close_history、未传 close_wide）
    # → σ_p=NaN → 安全退化 scale=1
    builder = PortfolioBuilder(["a", "b"])
    _, targets = builder.build_targets(
        SelectionPolicy(vol_target_annual=0.20),
        np.array([0, 1]), np.array([1.0, 0.5]))
    assert targets == {0: 0.5, 1: 0.5}


def test_build_targets_safe_degrade_zero_vol_and_short_window():
    policy = SelectionPolicy(vol_target_annual=0.20)
    # σ_p=0（价格恒定，收益全 0）→ scale=1，而非放大仓位
    flat = pd.DataFrame({"a": 10.0, "b": 12.0},
                        index=pd.bdate_range("2024-01-02", periods=80))
    _, targets = _run_targets(policy, flat)
    assert targets == {0: 0.5, 1: 0.5}
    # 窗口不足（19 个有效收益 < min_obs=30）→ NaN → scale=1
    short = _wide({"a": 0.05, "b": 0.05}, n=20)
    _, targets = _run_targets(policy, short)
    assert targets == {0: 0.5, 1: 0.5}


def test_build_targets_regime_and_vol_combined_clip():
    close = _wide({"a": 0.05, "b": 0.05}, n=80)
    sigma = portfolio_vol(close, window=60)
    # 弱市 regime 0.5 × vol 0.4 = 0.2 → 整体 clip 到 lo=0.3
    policy = SelectionPolicy(regime_adx=20, regime_scale=0.5,
                             vol_target_annual=0.4 * sigma)
    _, targets = _run_targets(policy, close, market_adx=10.0)
    assert targets[0] == pytest.approx(0.3 / 2)
    # 弱市 0.5 × vol hi=1.5 = 0.75（在 [0.3, 1.5] 内，不截）
    close_low = _wide({"a": 0.002, "b": 0.002}, n=80)
    policy2 = SelectionPolicy(regime_adx=20, regime_scale=0.5,
                              vol_target_annual=0.20)
    _, targets2 = _run_targets(policy2, close_low, market_adx=10.0)
    assert targets2[0] == pytest.approx(0.75 / 2)
    # 强市（market_adx >= 阈值）→ regime=1.0 × vol 0.4
    _, targets3 = _run_targets(policy, close, market_adx=50.0)
    assert targets3[0] == pytest.approx(0.4 / 2)


def test_regime_only_unchanged_when_vol_disabled():
    # vol 关：regime_scale 行为与旧版完全一致（0.5，不受 clip 干扰）
    close = _wide({"a": 0.05, "b": 0.05}, n=80)
    policy = SelectionPolicy(regime_adx=20, regime_scale=0.5)
    _, targets = _run_targets(policy, close, market_adx=10.0)
    assert targets == {0: 0.25, 1: 0.25}


# ── builder 信号日游标通道（engine 路径，无前视）─────────────────────


def test_builder_channel_consumes_signal_dates_without_lookahead():
    dates = pd.bdate_range("2024-01-02", periods=100)
    sig1, sig2 = dates[70], dates[90]
    rets = np.where(np.arange(100) % 2 == 0, 0.02, -0.02)
    close = pd.DataFrame({"a": 10.0 * np.cumprod(1.0 + rets)}, index=dates)
    # sig1 之后（第 71 行起）换成 ±20% 剧烈波动：sig1 时点的 σ_p 不得受其后
    # 数据影响（第 70 行的收益 = close[70]/close[69]，仍属安静段）
    post = np.where(np.arange(71, 100) % 2 == 0, 0.20, -0.20)
    close_after = close.copy()
    close_after.iloc[71:, 0] = close.iloc[70, 0] * np.cumprod(1.0 + post)

    policy = SelectionPolicy(vol_target_annual=0.02 * np.sqrt(252))
    builder = PortfolioBuilder(["a"], close_history=close_after,
                               signal_dates=[sig1, sig2])
    # 第一次调用：信号日 sig1 → 只能看到 ≤ sig1 的数据 → σ_p ≈ target → scale≈1
    _, t1 = builder.build_targets(policy, np.array([0]), np.array([1.0]))
    assert t1[0] == pytest.approx(1.0, rel=0.02)
    # 第二次调用：信号日 sig2 → 窗口含剧烈段 → σ_p 远超 target → scale=lo
    _, t2 = builder.build_targets(policy, np.array([0]), np.array([1.0]))
    assert t2[0] == pytest.approx(0.3, rel=1e-6)
    # 游标越界钳制在最后一个信号日（滞后，方向安全，不会前视）
    _, t3 = builder.build_targets(policy, np.array([0]), np.array([1.0]))
    assert t3[0] == pytest.approx(0.3, rel=1e-6)


def test_explicit_close_wide_bypasses_cursor():
    rets = np.where(np.arange(100) % 2 == 0, 0.05, -0.05)
    close = pd.DataFrame({"a": 10.0 * np.cumprod(1.0 + rets)},
                         index=pd.bdate_range("2024-01-02", periods=100))
    builder = PortfolioBuilder(["a"])  # 无 close_history
    policy = SelectionPolicy(vol_target_annual=0.5 * portfolio_vol(close, window=60))
    _, targets = builder.build_targets(policy, np.array([0]),
                                       np.array([1.0]), close_wide=close)
    assert targets[0] == pytest.approx(0.5, rel=1e-6)
    # 显式 close_wide 通道不推进游标
    assert getattr(builder, "_vol_cursor") == 0


# ── 配置字段与门面透传 ────────────────────────────────────────────────


def test_config_defaults_off():
    cfg = BacktestConfig(panel=None, codes=[], factor="x", ascending=False,
                         start="2024-01-01", end="2024-03-01",
                         capital=1e6, top_n=2)
    assert cfg.vol_target_annual is None  # 默认关
    assert cfg.vol_target_lo == 0.3
    assert cfg.vol_target_hi == 1.5
    assert cfg.vol_target_window == 60


def test_run_backtest_facade_accepts_vol_target_params():
    params = inspect.signature(run_backtest).parameters
    for name, default in (("vol_target_annual", None),
                          ("vol_target_lo", 0.3),
                          ("vol_target_hi", 1.5),
                          ("vol_target_window", 60)):
        assert name in params
        assert params[name].default == default


def test_run_backtest_vol_target_end_to_end(panel):
    # warmup_days=400 与模拟盘默认一致：因子全程有效 → 每个信号日都有
    # build_targets 调用，游标与信号日 1:1 对齐，σ_p 从首次调仓即可估。
    common = dict(panel=panel, codes=CODES, factor="mom20", ascending=False,
                  start=START, end=END, capital=1_000_000, top_n=2,
                  freq="monthly", warmup_days=400)
    off = run_backtest(**common)
    assert len(off["nav"]) > 0 and off["nav"].notna().all()
    # 开关关（显式 None）= 不传参数，逐位一致
    off2 = run_backtest(**common, vol_target_annual=None)
    assert np.array_equal(off["nav"].to_numpy(), off2["nav"].to_numpy())
    # σ_p >> target → scale=lo=0.3 降仓 → 现金拖累改变净值（全链路生效）
    low = run_backtest(**common, vol_target_annual=1e-4)
    assert len(low["nav"]) > 0 and low["nav"].notna().all()
    assert not np.allclose(off["nav"].to_numpy(), low["nav"].to_numpy())
