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
ST 股 5%：当调用方传入逐日 ST 标记（st_mask）时按 5% 判定，
缺标记（无 ST 数据源 / 历史早于 ST 覆盖）退化为按板块比例近似。
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


_ST_RATIO = 0.05


def build_limit_flags(
    close: pd.DataFrame,
    open_: pd.DataFrame,
    st_mask: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 (limit_up, limit_down, one_word_up, one_word_down) 布尔矩阵。

    矩阵形状与 close 一致（行=交易日，列=股票），True 表示当日处于该状态。
    涨跌幅基于前复权开盘价/前日收盘价，NaN（停牌/无历史）不会被识别为
    涨跌停。

    st_mask: 可选，与 close 同形状的布尔 DataFrame（index=交易日,
    columns=代码，True=当日为 ST/*ST）。为 True 的位置涨跌停幅度按 5%
    计算；未传/全 NaN/缺失列时该股按板块比例（limit_ratio）近似。
    """
    codes = close.columns.tolist()
    prev_close = close.shift(1)
    open_ret = open_ / prev_close - 1.0

    base_ratios = pd.Series([limit_ratio(c) for c in codes], index=codes)

    if st_mask is not None and len(st_mask):
        # 对齐到 close 的索引/列：缺失日期/代码按非 ST 处理。
        # 先取交集索引/列，再按 close 全量对齐，避免 object dtype 的 fillna warning。
        st_idx = st_mask.index.intersection(close.index)
        st_cols = st_mask.columns.intersection(close.columns)
        st = pd.DataFrame(False, index=close.index, columns=close.columns)
        if len(st_idx) and len(st_cols):
            sub = st_mask.loc[st_idx, st_cols].astype(bool)
            st.loc[st_idx, st_cols] = sub
        st = st.to_numpy(dtype=bool)
        # ST 5% 只对真实有涨跌停的板块生效；ETF/基金（limit_ratio=1.0）不受影响
        st = st & (base_ratios.values.reshape(1, -1) < 1.0)
        ratios = pd.DataFrame(np.where(st, _ST_RATIO, base_ratios.values.reshape(1, -1)),
                              index=close.index, columns=close.columns)
    else:
        ratios = base_ratios

    limit_up = open_ret >= ratios - 0.005
    limit_down = open_ret <= -(ratios - 0.005)

    # 一字板：开盘封板且收盘价与开盘价一致（容忍四舍五入误差）
    one_up = limit_up & (open_ >= close - 1e-6)
    one_down = limit_down & (open_ <= close + 1e-6)

    return (limit_up.values.astype(bool),
            limit_down.values.astype(bool),
            one_up.values.astype(bool),
            one_down.values.astype(bool))