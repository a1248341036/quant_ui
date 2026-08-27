from __future__ import annotations

"""涨跌停/停牌标记计算。

面板数据是前复权日线，没有现成的涨跌停标记，这里用涨跌幅近似识别。
判定基准是**开盘价相对前收的涨幅**：引擎统一在开盘时点成交，
判定口径应为"此刻还能不能成交"——
- 涨停：开盘涨幅 >= 板块限制 - 容差（开盘即封死，买不进）
- 跌停：开盘跌幅 >= 板块限制 - 容差（开盘封死跌停，卖不出）
- 一字板：开盘封板且收盘价与开盘价一致（全天未开板）
- 停牌：开盘价缺失（引擎已用 valid_open 处理不可交易）

盘中才封板的股票开盘仍可正常成交，不会被提前剔除。

板块限制：沪深主板 10%，创业板/科创板 20%，北交所 30%。
ST 股 5% 因数据缺 ST 标记暂不区分。
"""

import numpy as np
import pandas as pd


def limit_ratio(code: str) -> float:
    """按代码前缀返回涨跌停幅度（小数）。"""
    # 基金/ETF：沪市 5 开头、深市 15/16/18 开头，多数无涨跌幅限制或规则不同，
    # 用 1.0 使其不会被近似识别为涨跌停
    if code.startswith(("5", "15", "16", "18")):
        return 1.0
    if code.startswith(("300", "301", "688", "689")):
        return 0.20
    if code.startswith(("8", "4", "920")):
        return 0.30
    return 0.10


def build_limit_flags(
    close: pd.DataFrame,
    open_: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 (limit_up, limit_down, one_word_up, one_word_down) 布尔矩阵。

    矩阵形状与 close 一致（行=交易日，列=股票），True 表示当日处于该状态。
    涨跌幅基于前复权开盘价/前日收盘价，NaN（停牌/无历史）不会被识别为
    涨跌停。
    """
    codes = close.columns.tolist()
    prev_close = close.shift(1)
    open_ret = open_ / prev_close - 1.0

    ratios = pd.Series([limit_ratio(c) for c in codes], index=codes)
    limit_up = open_ret >= ratios - 0.005
    limit_down = open_ret <= -(ratios - 0.005)

    # 一字板：开盘封板且收盘价与开盘价一致（容忍四舍五入误差）
    one_up = limit_up & (open_ >= close - 1e-6)
    one_down = limit_down & (open_ <= close + 1e-6)

    return (limit_up.values.astype(bool),
            limit_down.values.astype(bool),
            one_up.values.astype(bool),
            one_down.values.astype(bool))
