"""BacktestConfig 数据类与 run_backtest / run_backtest_config 门面。"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from .. import trading_config
from ..assets import AssetExecutionProfile, STOCK_PROFILE


@dataclass(frozen=True)
class BacktestConfig:
    """一次回测运行的完整配置（与 run_backtest 的形参一一对应）。

    拆分后的引擎以 BacktestConfig 作为唯一输入载体；run_backtest 作为
    兼容门面保持原有 50 个形参签名不变，内部构造本配置后交给
    run_backtest_config 执行四阶段管线。
    """

    panel: pd.DataFrame
    codes: list[str]
    factor: str
    ascending: bool
    start: str
    end: str
    capital: float
    top_n: int
    freq: str = "monthly"
    buy_cost: float = trading_config.BUY_COST
    sell_cost: float = trading_config.SELL_COST
    amount_q: float = 0.3
    affordable: bool = True
    lot_size: int = 100
    warmup_days: int | None = None
    cash_mode: bool = True
    limit_flags: bool = True
    slippage_bps: float = trading_config.SLIPPAGE_BPS
    max_participation: float = trading_config.MAX_PARTICIPATION
    max_weight: float | None = None
    industry_map: dict[str, str] | None = None
    industry_cap: int | None = None
    factor_builder: Callable | None = None
    external_scores: pd.DataFrame | None = None
    factor_weights: dict[str, float] | None = None
    factor_directions: dict[str, bool] | None = None
    analyze: bool = False
    long_short: bool = False
    short_n: int | None = None
    short_cost_rate: float = 0.0
    industry_neutral: bool = False
    use_financial: bool = False
    risk_neutral: bool = False
    adx_filter: float | None = None
    chandelier_mult: float = 0.0
    chandelier_period: int = 22
    regime_adx: float | None = None
    regime_scale: float = 0.5
    selection_mode: str = "top_n"
    selection_pct: float = 0.10
    min_positions: int = 1
    max_positions: int | None = None
    min_score: float | None = None
    execution_profile: AssetExecutionProfile | None = None
    share_classes: dict[str, str] | None = None
    spread_bps: float | None = None
    min_commission: float | None = None
    impact_coef: float = 0.0
    impact_vol: float = 0.02
    # Screener（regime 感知因子选择）
    use_screener: bool = False
    screener_lookback: int = 10
    screener_min_ic: float = 0.02
    screener_max_corr: float = 0.7
    screener_factors: list[str] | None = None


def run_backtest_config(cfg: BacktestConfig) -> dict:
    """按 BacktestConfig 执行完整回测管线（阶段 1-4）。"""
    from .prepare import _prepare_backtest
    from .factor_matrix import _build_factor_matrix
    from .simulate import _simulate
    from .result import _finalize_result

    prep = _prepare_backtest(cfg)
    fctx = _build_factor_matrix(cfg, prep)
    sim = _simulate(cfg, prep, fctx)
    return _finalize_result(cfg, prep, fctx, sim)


def run_backtest(
    panel: pd.DataFrame,
    codes: list[str],
    factor: str,
    ascending: bool,
    start: str,
    end: str,
    capital: float,
    top_n: int,
    freq: str = "monthly",
    buy_cost: float = trading_config.BUY_COST,
    sell_cost: float = trading_config.SELL_COST,
    amount_q: float = 0.3,
    affordable: bool = True,
    lot_size: int = 100,
    warmup_days: int | None = None,
    cash_mode: bool = True,
    limit_flags: bool = True,
    slippage_bps: float = trading_config.SLIPPAGE_BPS,
    max_participation: float = trading_config.MAX_PARTICIPATION,
    max_weight: float | None = None,
    industry_map: dict[str, str] | None = None,
    industry_cap: int | None = None,
    factor_builder: Callable | None = None,
    external_scores: pd.DataFrame | None = None,
    factor_weights: dict[str, float] | None = None,
    factor_directions: dict[str, bool] | None = None,
    analyze: bool = False,
    long_short: bool = False,
    short_n: int | None = None,
    short_cost_rate: float = 0.0,
    industry_neutral: bool = False,
    use_financial: bool = False,
    risk_neutral: bool = False,
    adx_filter: float | None = None,
    chandelier_mult: float = 0.0,
    chandelier_period: int = 22,
    regime_adx: float | None = None,
    regime_scale: float = 0.5,
    selection_mode: str = "top_n",
    selection_pct: float = 0.10,
    min_positions: int = 1,
    max_positions: int | None = None,
    min_score: float | None = None,
    execution_profile: AssetExecutionProfile | None = None,
    share_classes: dict[str, str] | None = None,
    spread_bps: float | None = None,
    min_commission: float | None = None,
    impact_coef: float = 0.0,
    impact_vol: float = 0.02,
    use_screener: bool = False,
    screener_lookback: int = 10,
    screener_min_ic: float = 0.02,
    screener_max_corr: float = 0.7,
    screener_factors: list[str] | None = None,
) -> dict:
    """事件驱动回测：T+1、一手 100 股、费用、可承载性过滤。

    策略只需提供「每个信号日的因子得分」，引擎负责月度/周度调仓、
    停牌继承、买卖成本与净值计算。

    warmup_days: 因子预热天数。短窗口回测时（如只看近半年），动量/波动类
    因子在窗口起点没有足够历史，会用 start 前 warmup_days 个自然日的数据
    计算因子，但净值仍从 start 开始输出。None 表示不预热（窗口即计算区间）。

    industry_map/industry_cap: 行业分散约束。industry_map 为 {code: 行业}，
    调仓时每个行业最多选 industry_cap 只，选不满 top_n 时按实际数量等权。

    industry_neutral: 行业中性化。选股前把因子得分按行业内截面去均值，
    消除行业/风格暴露，再按中性化后的得分排序。

    use_financial: 使用财务因子（PG fina_indicator/income），因子名在
    FINANCIAL_FACTORS 中或组合权重含财务因子时自动开启。

    risk_neutral: 完整风险中性化（风格+行业）。选股前把因子得分对
    风格/行业暴露回归取残差；同时返回 risk_attribution（期末持仓风险分解）。
    需提供 industry_map。

    long_short: 多空对冲模式。多头买 top_n 只（等权 +1），空头卖最弱
    short_n 只（等权 -1），净敞口为 0（名义多头 = 名义空头 = 1 倍资金）。
    short_cost_rate: 空头年化融券费率（占空头名义的比例/年），默认 0。

    factor_builder: 可选的自定义因子构建函数，签名与 build_factor_frames 一致
    (close, am20, turn20) -> {因子名: 得分矩阵}。默认使用本模块内置实现。

    analyze: 额外计算因子质量（20 日未来收益的 Spearman IC + 5 分组收益），
    返回 factor_quality（含 ic_series / group_table）。默认 False 保持轻量。

    cash_mode: 现金/整手执行模型（与模拟盘同口径，默认 True）。按实际资金、
    100 股整手、先卖后买、费用从现金扣、涨停拒买/跌停拒卖、拒单不补仓，
    每日收盘按 close 估值；False 时保留旧的权重连续模型。

    limit_flags: 是否启用涨跌停过滤（涨停不可买入、跌停不可卖出）。
    slippage_bps: 固定滑点（基点），买入价=开盘×(1+bps/1e4)，卖出反向。
    max_participation: 流动性约束，单笔买入金额 <= 20日均成交额 × 该比例。
    max_weight: 单票权重上限（占组合市值比例），None 表示不限制。
    """

    cfg = BacktestConfig(
        panel=panel, codes=codes, factor=factor, ascending=ascending,
        start=start, end=end, capital=capital, top_n=top_n,
        freq=freq, buy_cost=buy_cost, sell_cost=sell_cost,
        amount_q=amount_q, affordable=affordable, lot_size=lot_size,
        warmup_days=warmup_days, cash_mode=cash_mode, limit_flags=limit_flags,
        slippage_bps=slippage_bps, max_participation=max_participation,
        max_weight=max_weight, industry_map=industry_map,
        industry_cap=industry_cap, factor_builder=factor_builder,
        external_scores=external_scores, factor_weights=factor_weights,
        factor_directions=factor_directions, analyze=analyze,
        long_short=long_short, short_n=short_n,
        short_cost_rate=short_cost_rate, industry_neutral=industry_neutral,
        use_financial=use_financial, risk_neutral=risk_neutral,
        adx_filter=adx_filter, chandelier_mult=chandelier_mult,
        chandelier_period=chandelier_period, regime_adx=regime_adx,
        regime_scale=regime_scale, selection_mode=selection_mode,
        selection_pct=selection_pct, min_positions=min_positions,
        max_positions=max_positions, min_score=min_score,
        execution_profile=execution_profile, share_classes=share_classes,
        spread_bps=spread_bps, min_commission=min_commission,
        impact_coef=impact_coef, impact_vol=impact_vol,
        use_screener=use_screener,
        screener_lookback=screener_lookback,
        screener_min_ic=screener_min_ic,
        screener_max_corr=screener_max_corr,
        screener_factors=screener_factors,
    )
    return run_backtest_config(cfg)
