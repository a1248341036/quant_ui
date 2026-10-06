"""factor float64 身份缓存（2026-10-06 并发评测内存优化）。

与 ``_label_cache.label_f64`` 对称：submit 路径单因子交付最多 8 次
``quantile_portfolio_metrics`` 调用（全窗口 1 + tradable lens 1 + 三段 3 +
stage_one 过线后三段 3），每次 ``factor.to_numpy(dtype=np.float64, copy=False)``
在 float32 源列上**无法免拷贝**——8.2M 行 ≈ 66MB memcpy/次，8 次即 ~528MB
瞬时叠加（并发 12 eval 时 12× 叠加到 ~6GB）。

**键必须是 factor 的「对象身份」** ``id(factor)``，不能是 ``(id(index), name)``：
submit 路径的 ``metric_series = pd.Series(values, index=panel.index)`` **不带 name**
（``str(None) == "None"``），且同一 session 内 ``panel`` 是同一对象 → 只按
(index 身份, 名) 作键时，同一 session 内**所有候选**键完全相同，第 2 个候选起会
命中上一个因子的数组（2026-10-06 OCR 评审实测：换手门按上一个因子的值判定）。
同一次 submit 的 ``:510/:529`` 与同一次 eval 的多插件调用传的是**同一 Series 对象**
（评估引擎的 ``context.factor``），故按键对象身份仍能命中；不同表达式/不同候选天然是新对象。

以 weakref 校验对象身份（对象被 GC 后 id 可复用，裸 ``id`` 会 ABA）；跨调用、跨线程
共享同一只读 float64 数组。前提与 ``_label_cache`` 一致：factor 构造后不可变。

测试注入点：``_factor_f64_override``（设为函数即绕过缓存，用于一致性对照）。
"""

from __future__ import annotations

import threading
import weakref
from collections import OrderedDict

import numpy as np
import pandas as pd

_factor_f64_override = None
_cache: "OrderedDict[int, tuple[weakref.ref, np.ndarray]]" = OrderedDict()
_lock = threading.Lock()
_CACHE_MAX = 4


def factor_f64(factor: pd.Series) -> np.ndarray:
    """factor 的 float64 只读数组，按 factor 对象身份缓存共享。

    float64 源列零拷贝（``to_numpy(copy=False)`` 直接返回底层 buffer）；
    float32/其它 dtype 源列只拷贝一次并缓存，后续同对象调用复用。
    """
    if _factor_f64_override is not None:
        return _factor_f64_override(factor)
    key = id(factor)
    with _lock:
        hit = _cache.get(key)
        if hit is not None:
            ref, arr = hit
            if ref() is factor:
                _cache.move_to_end(key)
                return arr
            _cache.pop(key, None)
    arr = np.asarray(factor.to_numpy(dtype=np.float64, copy=False))
    try:
        arr.setflags(write=False)
    except (ValueError, AttributeError):
        pass
    with _lock:
        _cache[key] = (weakref.ref(factor), arr)
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    return arr
