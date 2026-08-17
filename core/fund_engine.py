from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import run_backtest


def build_fund_panel(nav: pd.DataFrame) -> pd.DataFrame:
    """把场外基金净值序列转成回测引擎可用的最小 panel。

    场外基金没有盘中价/成交量，用单位净值同时充当 open/close，
    turnover/amount 置 1，让引擎的流动性过滤全部通过（基金申购无
    成交额约束）。signal 日收盘净值 = T 日净值，执行日 open = T+1
    净值，正好近似场外基金 T+1 确认。

    场外基金（尤其 QDII/海外指数）公布净值日历不一致：A 股基金周末无
    净值、海外基金周末仍更新。这里先按全市场交易日历逐代码 ffill，
    保证任一代码在 union 日历上都有可估值净值，避免引擎把缺净值日
    的持仓按 0 估值导致净值曲线跳变。
    """
    if nav is None or len(nav) == 0:
        return pd.DataFrame(columns=["date", "open", "close", "turnover",
                                     "amount", "code", "turn20", "am20",
                                     "volume"])
    df = nav[["date", "code", "nav"]].dropna().copy()
    df["date"] = pd.to_datetime(df["date"])
    cal = pd.DatetimeIndex(sorted(df["date"].unique()))
    mat = df.pivot_table(index="date", columns="code", values="nav",
                         aggfunc="last", observed=True)
    mat = mat.reindex(cal).ffill()
    mat.index.name = "date"
    long = mat.stack(future_stack=True).rename("nav").reset_index()
    long = long.dropna(subset=["nav"]).sort_values(["code", "date"])
    panel = long.rename(columns={"nav": "close"}).copy()
    panel["open"] = panel["close"]
    panel["turnover"] = 1.0
    panel["amount"] = 1.0
    panel["volume"] = 1.0
    panel["turn20"] = 1.0
    panel["am20"] = 1.0
    panel["code"] = panel["code"].astype("category")
    return panel


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
) -> dict:
    panel = build_fund_panel(nav)
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
    )
