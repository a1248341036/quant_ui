"""DSL 算子公共工具：面板约定、按品种分组、Numba 友好的一维内核包装。

扩展算子（``aqra/dsl/extensions/`` 或用户模块）应只依赖本模块，勿直接改 ``operators.py``。
"""
from __future__ import annotations

from typing import Callable, Union

import numpy as np
import pandas as pd

Window = Union[int, float, pd.DataFrame]

InstrumentKernel = Callable[[np.ndarray], np.ndarray]
InstrumentKernel2 = Callable[[np.ndarray, np.ndarray], np.ndarray]
DatetimeKernel = Callable[[np.ndarray], np.ndarray]
DatetimeTransform = Callable[[pd.Series], pd.Series]


def is_dynamic_window(w: Window) -> bool:
    return isinstance(w, pd.DataFrame)


def as_int_window(w: Window) -> int:
    if is_dynamic_window(w):
        raise TypeError("此处需要整数窗口，收到 DataFrame（动态窗口请用对应重载）")
    return int(w)


def first_series(df: pd.DataFrame) -> pd.Series:
    return df.iloc[:, 0]


def series_from_group(x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    return x


def gb_instrument(df: pd.DataFrame):
    return df.groupby(level="instrument", sort=False)


def gb_datetime(df: pd.DataFrame):
    return df.groupby(level="datetime", sort=False)


def out_frame(values: pd.Series, template: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns[:1])


def dynamic_window_int_series(win: pd.DataFrame, index: pd.Index) -> np.ndarray:
    s = first_series(win.reindex(index)).to_numpy(dtype=float, copy=False)
    out = np.nan_to_num(s, nan=1.0)
    out = np.clip(np.round(out), 1, None)
    return out.astype(int)


def lag_int_series(lag_df: pd.DataFrame, index: pd.Index) -> np.ndarray:
    s = first_series(lag_df.reindex(index)).to_numpy(dtype=float, copy=False)
    out = np.nan_to_num(s, nan=0.0)
    out = np.clip(np.round(out), 0, None)
    return out.astype(int)


def per_instrument_unary(df: pd.DataFrame, kernel: InstrumentKernel) -> pd.DataFrame:
    """单列面板 → 按 instrument 切 1-D 数组 → kernel → 写回同索引单列。"""
    result = np.full(len(df), np.nan, dtype=np.float32)
    for _, sub in gb_instrument(df):
        idx = sub.index
        vals = sub.iloc[:, 0].to_numpy(dtype=float, copy=False)
        pos = df.index.get_indexer(idx)
        result[pos] = kernel(vals)
    return pd.DataFrame(result, index=df.index, columns=df.columns[:1])


def per_instrument_bivariate(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    kernel: InstrumentKernel2,
) -> pd.DataFrame:
    """双列面板对齐后按 instrument 运行二维 kernel。"""
    result = np.full(len(df1), np.nan, dtype=np.float32)
    for _, sub1 in gb_instrument(df1):
        idx = sub1.index
        sub2 = df2.reindex(idx)
        x = sub1.iloc[:, 0].to_numpy(dtype=float, copy=False)
        y = sub2.iloc[:, 0].to_numpy(dtype=float, copy=False)
        pos = df1.index.get_indexer(idx)
        result[pos] = kernel(x, y)
    return pd.DataFrame(result, index=df1.index, columns=df1.columns[:1])


def per_datetime_unary(df: pd.DataFrame, kernel: DatetimeKernel) -> pd.DataFrame:
    """单列面板 → 按 datetime 切截面向量 → kernel → 写回同索引单列。"""
    result = np.full(len(df), np.nan, dtype=np.float32)
    for _, sub in gb_datetime(df):
        idx = sub.index
        vals = sub.iloc[:, 0].to_numpy(dtype=float, copy=False)
        pos = df.index.get_indexer(idx)
        result[pos] = kernel(vals)
    return pd.DataFrame(result, index=df.index, columns=df.columns[:1])


def per_datetime_transform(
    df: pd.DataFrame,
    transform: DatetimeTransform,
) -> pd.DataFrame:
    """按 datetime 截面 ``groupby.transform``，适合 rank / zscore 等 pandas 操作。"""
    ser = first_series(df)
    out = ser.groupby(level="datetime", sort=False).transform(transform)
    return out_frame(out.astype(np.float32), df)
