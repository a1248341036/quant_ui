from __future__ import annotations

import pandas as pd

from .assets import FUND_NAV_ADAPTER, FUND_NAV_PROFILE
from .engine import run_backtest
from .execution import detect_fund_share_class


def build_fund_panel(nav: pd.DataFrame) -> pd.DataFrame:
    """把场外基金净值序列转成回测引擎可用的最小 panel。

    场外基金没有盘中价/成交量，用单位净值同时充当 open/close，
    turnover/amount 置 1，让引擎的流动性过滤全部通过（基金申购无
    成交额约束）。

    T+1 确认近似：signal 日 close = NAV(t)，exec 日 open = NAV(t)，
    引擎按 exec 日 open 成交，即按 T+1 净值确认份额（signal 日是 T，
    exec 日是 T+1）。

    场外基金（尤其 QDII/海外指数）公布净值日历不一致：A 股基金周末无
    净值、海外基金周末仍更新。这里先按全市场交易日历逐代码 ffill，
    保证任一代码在 union 日历上都有可估值净值，避免引擎把缺净值日
    的持仓按 0 估值导致净值曲线跳变。
    """
    return FUND_NAV_ADAPTER.build_panel(nav)


def run_fund_backtest(
    nav: pd.DataFrame,
    codes: list[str],
    factor: str,
    ascending: bool,
    start: str,
    end: str,
    capital: float,
    top_n: int,
    freq: str = "monthly",
    buy_cost: float = 0.0015,
    sell_cost: float = 0.0050,
    amount_q: float = 0.2,
    affordable: bool = True,
    lot_size: int = 1,
    warmup_days: int | None = 400,
    cash_mode: bool = True,
    limit_flags: bool = False,
    slippage_bps: float = 0.0,
    max_participation: float = 0.0,
    max_weight: float | None = None,
    analyze: bool = False,
    factor_weights: dict[str, float] | None = None,
    factor_directions: dict[str, bool] | None = None,
    fund_names: dict[str, str] | None = None,
) -> dict:
    """场外基金回测。

    fund_names: {code: 基金简称}，用于识别 A/C 类决定申购费率。
    传入时构建 share_classes 传给执行适配器；不传时全部按 A 类处理。
    """
    panel = build_fund_panel(nav)

    # 按基金代码构建 A/C 类映射。engine 内部 pivot 后列顺序可能变化，
    # 因此不能用传入 codes 的整数位置传递份额类别。
    share_classes = None
    if fund_names:
        share_classes = {
            str(code): detect_fund_share_class(fund_names.get(code, ""))
            for code in codes
        }

    return run_backtest(
        panel=panel,
        codes=codes,
        factor=factor,
        ascending=ascending,
        start=start,
        end=end,
        capital=capital,
        top_n=top_n,
        freq=freq,
        buy_cost=buy_cost,
        sell_cost=sell_cost,
        amount_q=amount_q,
        affordable=affordable,
        lot_size=lot_size,
        warmup_days=warmup_days,
        cash_mode=cash_mode,
        limit_flags=limit_flags,
        slippage_bps=slippage_bps,
        max_participation=max_participation,
        max_weight=max_weight,
        analyze=analyze,
        factor_weights=factor_weights,
        factor_directions=factor_directions,
        execution_profile=FUND_NAV_PROFILE,
        share_classes=share_classes,
    )
