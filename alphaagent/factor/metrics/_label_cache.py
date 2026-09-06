"""label float64 身份缓存（2026-09-06 批量评估改造）。

一次评估内多个 metric 插件（ic / rank_ic / lag1 / decile / fmb / portfolio）
各自把同一 label 列整体转 float64：5.7M 行 ≈ 46MB memcpy/次，纯 GIL 段，
且 float32 源列 ``to_numpy(copy=False)`` 无法免拷贝。一轮 8 个表达式即
~40 次重复转换。

label 的 index 与 panel.index 是**同一对象**，以 ``(索引身份, 列名)`` 为键
做弱引用小 LRU；跨评估、跨插件、跨并发线程共享同一只读数组。前提与
``_day_slices`` 缓存一致：panel 载入后不可变。

测试注入点：``_label_f64_override``（设为函数即绕过缓存，用于一致性对照）。
"""

from __future__ import annotations

import threading
import weakref
from collections import OrderedDict

import numpy as np
import pandas as pd

_label_f64_override = None
_cache: "OrderedDict[tuple[int, str], tuple[weakref.ref, np.ndarray]]" = OrderedDict()
_lock = threading.Lock()
_CACHE_MAX = 4


def label_f64(label: pd.Series) -> np.ndarray:
    """label 的 float64 只读数组，按 (索引身份, 列名) 缓存共享。"""
    if _label_f64_override is not None:
        return _label_f64_override(label)
    index = label.index
    key = (id(index), str(label.name))
    with _lock:
        hit = _cache.get(key)
        if hit is not None:
            ref, arr = hit
            if ref() is index:
                _cache.move_to_end(key)
                return arr
            _cache.pop(key, None)
    arr = np.asarray(label.to_numpy(dtype=np.float64, copy=False))
    try:
        arr.setflags(write=False)
    except (ValueError, AttributeError):
        pass
    with _lock:
        _cache[key] = (weakref.ref(index), arr)
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    return arr
