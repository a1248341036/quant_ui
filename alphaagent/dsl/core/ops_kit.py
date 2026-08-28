"""DSL 算子公共工具：面板约定、按品种分组、Numba 友好的一维内核包装。

扩展算子（``aqra/dsl/extensions/`` 或用户模块）应只依赖本模块，勿直接改 ``operators.py``。
"""
from __future__ import annotations

from collections import OrderedDict
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


_LAYOUT_CACHE_MAX = 4
_LAYOUT_CACHE: "OrderedDict[int, tuple[pd.Index, np.ndarray, np.ndarray, np.ndarray]]" = OrderedDict()


def _index_instrument_codes(index: pd.Index) -> np.ndarray:
    """取 instrument 层的 factorize 编码。MultiIndex 直接复用 pandas 建索引时
    已算好的 ``codes``（零开销），避免每次算子调用对数百万行做字符串 factorize。"""
    if isinstance(index, pd.MultiIndex):
        names = list(index.names)
        if "instrument" in names:
            lvl = names.index("instrument")
        else:
            lvl = index.nlevels - 1
        return index.codes[lvl]
    codes, _ = pd.factorize(index, sort=False)
    return codes


def instrument_group_layout(index: pd.Index) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """稳定按 instrument 归组 → ``(order, bounds, inv)``，按 index 对象 LRU 缓存。

    同一面板的几十次算子调用只需计算一次 argsort/逆置换（首次约百毫秒级，
    命中缓存后 O(1)）。缓存持有 index 引用并以 ``is`` 校验，上限
    ``_LAYOUT_CACHE_MAX`` 条防止内存滞留。
    """
    key = id(index)
    cached = _LAYOUT_CACHE.get(key)
    if cached is not None and cached[0] is index:
        _LAYOUT_CACHE.move_to_end(key)
        return cached[1], cached[2], cached[3]

    codes = _index_instrument_codes(index)
    order = np.argsort(codes, kind="stable").astype(np.int64, copy=False)
    sorted_codes = codes[order]
    head = np.flatnonzero(
        np.concatenate(([True], sorted_codes[1:] != sorted_codes[:-1]))
    )
    bounds = np.empty(head.size + 1, dtype=np.int64)
    bounds[:-1] = head
    bounds[-1] = codes.size
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)

    _LAYOUT_CACHE[key] = (index, order, bounds, inv)
    while len(_LAYOUT_CACHE) > _LAYOUT_CACHE_MAX:
        _LAYOUT_CACHE.popitem(last=False)
    return order, bounds, inv


def instrument_group_order(index: pd.Index) -> tuple[np.ndarray, np.ndarray]:
    """稳定按 instrument 归组 → ``(order, bounds)``（``instrument_group_layout`` 的双子集）。

    ``order`` 把同一品种的行稳定聚为连续区间（保持品种内原行序，等价于
    ``groupby(level='instrument', sort=False)`` 的分组内容）；``bounds`` 为
    ``[起0, 起1, ..., 末]`` 的区间边界（长度 = 品种数 + 1）。调用方对输入数组做
    ``arr[order]`` 重排，逐区间 ``[bounds[g], bounds[g+1])`` 调一维内核，再用
    ``inverse_permutation(order)`` 逆置换写回原行序——消除逐品种 groupby/reindex/
    get_indexer 的 pandas 开销，并使按品种并行成为可能。
    """
    order, bounds, _ = instrument_group_layout(index)
    return order, bounds


def inverse_permutation(order: np.ndarray) -> np.ndarray:
    """``order`` 的逆置换：``inv[order[k]] = k``。"""
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    return inv


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


def datetime_group_bounds(df: pd.DataFrame) -> np.ndarray:
    """面板按 (datetime, instrument) 排序时，返回每个 datetime 组的起始下标数组。

    利用 datetime 层连续相等构成运行区间：``boundaries[i]`` 是第 i 个 datetime 组
    的起始行下标，最后一个组的下界为 ``len(df)``。供 CS_* 算子避免 groupby 回调
    开销（每组一次 Python 调用 → 一次切片运算）。面板未按 datetime 排序时返回 None。
    """
    dts = df.index.get_level_values("datetime")
    if len(dts) == 0:
        return np.zeros(1, dtype=np.int64)
    # 快速检查 datetime 层是否非递减（sorted 面板的契约）
    dt_np = dts._values
    if len(dt_np) > 1:
        if not (dt_np[1:] >= dt_np[:-1]).all():
            return None
    change = np.flatnonzero(dt_np[1:] != dt_np[:-1]) + 1
    return np.concatenate(([0], change, [len(dt_np)])).astype(np.int64)
