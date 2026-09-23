"""辅周期 panel 缓存：同一日频 panel 重复 eval 时复用 1w 聚合结果。"""

from __future__ import annotations

from collections import OrderedDict
from typing import MutableMapping

import pandas as pd

from alphaagent.dsl.stock.intervals import normalize_bar_interval
from alphaagent.dsl.stock.resample import build_timeframe_panel

# 键：(id(panel), base_interval, tag) → 辅频 DataFrame
AuxCache = MutableMapping[tuple[int, str, str], pd.DataFrame]

# 外部传入长生命周期 cache 时的上限：id 复用会静默命中陈旧辅表，且无界缓存
# 会让内存随 eval 次数线性增长。超限时淘汰最旧条目。
_MAX_CACHE_ENTRIES = 32


def get_or_build_aux_panel(
    panel: pd.DataFrame,
    tag: str,
    *,
    base_interval: str = "1d",
    cache: AuxCache | None = None,
) -> pd.DataFrame:
    """按 tag 返回辅频 panel；命中 cache 则直接复用。"""
    norm_tag = normalize_bar_interval(tag)
    norm_base = normalize_bar_interval(base_interval)
    key = (id(panel), norm_base, norm_tag)

    if cache is not None and key in cache:
        cached = cache[key]
        # id 复用防御：旧 panel 被 GC 后新对象可能拿到相同 id，校验索引身份
        if getattr(cached, "attrs", {}).get("_aux_panel_id") == id(panel):
            if isinstance(cache, OrderedDict):
                cache.move_to_end(key)
            return cached
        # 命中但身份不符：视为未命中，重建覆盖
        if isinstance(cache, OrderedDict):
            cache.pop(key, None)

    built = build_timeframe_panel(
        panel,
        target_interval=norm_tag,
        base_interval=norm_base,
    )
    built.attrs["_aux_panel_id"] = id(panel)

    if cache is not None:
        cache[key] = built
        if isinstance(cache, OrderedDict) and len(cache) > _MAX_CACHE_ENTRIES:
            cache.popitem(last=False)
    return built
