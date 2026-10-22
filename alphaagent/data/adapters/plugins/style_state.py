# -*- coding: utf-8 -*-
"""风格状态插件：恒载 canonical ``style_state``（市场级每日枚举）。

供给 Panel 的 ``style_state`` 列（``state_conditional_ic`` 的输入，spec
§3.4.1）。背景（2026-10-14 P1）：``rotation_policy.report_state_metrics``
挂载的状态条件 IC 因数据端从无任何插件产出 ``style_state`` 而整晚静默
返回 ``_missing_state``（零日志），风格轮动指标从未算过。

口径与 ``scripts/rotation_phase1_validation.py`` 的 canonical 基线同源
（``core/rotation/style_regime``）：

- size 维         ：``float_cap`` 截面 log 中位数的日变化率（市值中枢扩张/收缩）
- value_growth 维 ：风格日收益 ``value - growth``（价值-成长价差）

状态枚举 ``SMALL_VALUE / SMALL_GROWTH / LARGE_VALUE / LARGE_GROWTH / UNKNOWN``
（数值码 1/2/3/4，UNKNOWN→NaN）。无前视纪律由 style_regime 保证（T 日只用
``(T-window, T-1]`` 及以前数据）。

实现说明：本插件是辅助插件（priority=70），只按 ``start/end`` 从
``stock_daily_wide``（行情/市值/估值）与 ``express``（业绩快报 ``exp_roe``
作成长代理）重算市场级每日状态，并广播到每个标的。``style_state`` 不属任何
数据面（``_column_facets`` 为空），由 cnequity 的恒载列机制保证聚焦 run 也加载。
"""

from __future__ import annotations

import gc
import logging
from typing import Any

import numpy as np
import pandas as pd

from alphaagent.data.adapters.registry import DataSourcePlugin
from core.rotation.compiler import compile_design
from core.rotation.style_regime import canonical_design, compute_states

logger = logging.getLogger(__name__)

# 风格暴露的滚动窗口（天）：与 rotation_phase1_validation --style-window 缺省一致。
STYLE_WINDOW_DAYS = 20

# 状态枚举 → Panel 数值码（float 编码，对齐 industry_sw_l1 的离散码风格；
# UNKNOWN 不在表内 → map 后 NaN，评估端按 NaN 剔除暖机段，不参与状态条件 IC）。
_STATE_CODES: dict[str, float] = {
    "SMALL_VALUE": 1.0,
    "SMALL_GROWTH": 2.0,
    "LARGE_VALUE": 3.0,
    "LARGE_GROWTH": 4.0,
}

PLUGIN = DataSourcePlugin(
    name="style_state",
    dataset="style_state",  # 非单一 CNE 数据集，load 内自行拼装
    join_keys=("trade_date", "symbol"),
    datetime_key="trade_date",
    instrument_key="symbol",
    column_map={"style_state": "style_state"},
    priority=70,
)


def _align(wide: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """把宽表重排到参考宽表 (行=交易日, 列=标的) 的完整网格。"""
    return wide.reindex(index=ref.index, columns=ref.columns).astype(np.float64)


def _state_from_wide(
    close: pd.DataFrame,
    amount: pd.DataFrame,
    turnover: pd.DataFrame,
    float_cap: pd.DataFrame,
    pb: pd.DataFrame,
    roe: pd.DataFrame,
    *,
    style_window: int = STYLE_WINDOW_DAYS,
) -> pd.Series:
    """从对齐的宽表计算逐日 canonical ``style_state`` 数值码（纯函数）。

    与 validation 的 ``_style_beta_from_panel`` 同口径：先做滚动风格暴露 +
    逐日横截面回归得到 value/growth 因子收益，再按 canonical 设计算出状态；
    输出 Series 以交易日为索引，UNKNOWN→NaN。纯函数便于用合成数据做单测。
    """
    if close.empty or close.shape[1] == 0:
        raise ValueError("style_state: 无可用行情数据")

    close = close.astype(np.float64)
    amount = _align(amount, close)
    turnover = _align(turnover, close)
    float_cap = _align(float_cap, close)
    pb = _align(pb, close)
    roe = _align(roe, close)

    minp = max(5, style_window // 2)
    am20 = amount.rolling(style_window, min_periods=minp).mean()
    turn20 = turnover.rolling(style_window, min_periods=minp).mean()
    mom20 = close / close.shift(style_window) - 1.0
    vol20 = close.pct_change(fill_method=None).rolling(style_window, min_periods=minp).std()

    from core.risk_model import build_exposures, factor_returns

    X, names = build_exposures(
        close.to_numpy(dtype=float),
        am20.to_numpy(dtype=float),
        turn20.to_numpy(dtype=float),
        mom20.to_numpy(dtype=float),
        vol20.to_numpy(dtype=float),
        pb=pb.to_numpy(dtype=float),
        growth=roe.to_numpy(dtype=float),  # exp_roe 作成长代理（同 validation）
        industry_map=None,
        codes=list(close.columns),
    )
    rets = close.pct_change(fill_method=None).to_numpy(dtype=float)
    beta, _ = factor_returns(rets, X)
    del X, rets, am20, turn20, mom20, vol20
    gc.collect()
    style_beta = pd.DataFrame(beta, index=close.index, columns=names)

    # size 维：float_cap 截面 log 中位数日变化率（绝对口径不影响日变化率）
    logcap = np.log(np.maximum(float_cap.to_numpy(dtype=float), 1e-9))
    size_proxy = pd.Series(
        np.nanmedian(logcap, axis=1), index=close.index, name="size_proxy"
    ).pct_change()
    if not {"value", "growth"} <= set(style_beta.columns):
        raise ValueError(
            f"style_state: 风格因子收益缺 value/growth（实际: {sorted(style_beta.columns)}）"
        )
    value_growth_spread = style_beta["value"] - style_beta["growth"]
    carrier = pd.DataFrame(
        {"size_proxy": size_proxy, "value_growth_spread": value_growth_spread},
        index=close.index,
    )
    compiled = compile_design(canonical_design(), mode="technical")
    states = compute_states(compiled, carrier).states
    return states.map(_STATE_CODES).astype(np.float64).rename("style_state")


def load(
    dataset: str,
    *,
    start: str | None = None,
    end: str | None = None,
    cne_root: str | None = None,
    cne_config: str | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """重算 canonical 市场级风格状态并广播到每个 (trade_date, symbol)。"""
    from .express import load as _exp_load
    from .stock_daily_wide import load as _swd_load

    swd = _swd_load("stock_daily_wide", start=start, end=end,
                    cne_root=cne_root, cne_config=cne_config)
    exp = _exp_load("express", start=start, end=end)
    if swd is None or swd.empty:
        raise ValueError(f"style_state: stock_daily_wide 空 (start={start}, end={end})")
    if not {"trade_date", "ts_code", "close"} <= set(swd.columns):
        raise ValueError(f"style_state: stock_daily_wide 缺列 {sorted({'trade_date','ts_code','close'} - set(swd.columns))}")

    def _pivot(df: pd.DataFrame, col: str) -> pd.DataFrame:
        sub = df[["trade_date", "ts_code", col]].drop_duplicates(
            ["trade_date", "ts_code"], keep="first"
        )
        return sub.pivot(index="trade_date", columns="ts_code", values=col).sort_index()

    close = _pivot(swd, "close")
    amount = _pivot(swd, "amount")
    turnover = _pivot(swd, "turnover_rate")
    # 原始宽表 circ_mv 为万元口径；绝对值比例不影响 log 截面中位数的日变化率
    float_cap = _pivot(swd, "circ_mv")
    pb = _pivot(swd, "pb")
    roe = _pivot(exp, "exp_roe")

    state = _state_from_wide(close, amount, turnover, float_cap, pb, roe)

    # 广播：市场级状态对当日所有标的取同值（含当日停牌/无收盘价的标的一并覆盖）。
    # 两侧统一归一化到 datetime64[ns]，避免 parquet 的 [s]/[ms] 精度差异导致
    # Index.map 键不匹配而静默全 NaN。
    if int(state.notna().sum()) == 0:
        logger.warning("style_state adapter: 状态序列全为 UNKNOWN/NaN（检查 exp_roe 供给）")
    grid = swd[["trade_date", "ts_code"]].drop_duplicates(
        ["trade_date", "ts_code"], keep="first"
    ).copy()
    state.index = pd.DatetimeIndex(pd.to_datetime(state.index).normalize())
    dates = pd.DatetimeIndex(pd.to_datetime(grid["trade_date"]).dt.normalize())
    grid["symbol"] = grid["ts_code"].astype(str)
    grid["style_state"] = dates.map(state).astype(np.float64)
    out = grid[["trade_date", "symbol", "style_state"]]
    logger.info(
        "style_state adapter: %d rows × %d cols, %d 个状态日 (start=%s end=%s)",
        *out.shape, int(state.notna().sum()), start, end,
    )
    return out
