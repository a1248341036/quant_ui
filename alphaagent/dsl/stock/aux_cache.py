"""辅周期 panel 缓存：同一日频 panel 重复 eval 时复用 1w 聚合结果。"""

from __future__ import annotations

from typing import MutableMapping

import pandas as pd

from alphaagent.dsl.stock.intervals import normalize_bar_interval
from alphaagent.dsl.stock.resample import build_timeframe_panel

# 键：(id(panel), base_interval, tag) → 辅频 DataFrame
AuxCache = MutableMapping[tuple[int, str, str], pd.DataFrame]


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
        return cache[key]

    built = build_timeframe_panel(
        panel,
        target_interval=norm_tag,
        base_interval=norm_base,
    )

    if cache is not None:
        cache[key] = built
    return built
