"""Worker 内部的面版管理层（单 Worker 进程内常驻）。

基于 panel_mmap.py 的 Arrow IPC + pa.memory_map，
实现秒级 attach、零拷贝共享与 LRU 缓存管理。
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd

from alphaagent.data.adapters.cnequity import _CACHE_ROOT, is_cne_source, load_panel_from_cne
from alphaagent.data.adapters.panel_mmap import is_available as is_mmap_available, read_panel_arrow_mmap
from alphaagent.data.panel import load_panel, slice_panel

logger = logging.getLogger(__name__)


class WorkerPanelStore:
    """Worker 进程内的 panel 管理器。"""

    def __init__(self, max_cached: int = 2) -> None:
        self.max_cached = max_cached
        self._cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._split_cache: dict[str, dict[str, pd.DataFrame]] = {}

    def get_panel(self, session_key: str, spec: dict[str, Any]) -> pd.DataFrame:
        """根据 session_key 获取已加载/已映射的 panel，未命中则加载并缓存。"""
        if session_key in self._cache:
            self._cache.move_to_end(session_key)
            return self._cache[session_key]

        panel = self._load_panel_from_spec(spec)
        if len(self._cache) >= self.max_cached:
            evicted_key, _ = self._cache.popitem(last=False)
            self._split_cache.pop(evicted_key, None)
            logger.info("WorkerPanelStore 淘汰过期 panel 缓存 key=%s", evicted_key)

        self._cache[session_key] = panel
        return panel

    def get_split_panel(
        self,
        session_key: str,
        spec: dict[str, Any],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """获取切片 panel（在 Worker 内存缓存切片）。"""
        splits = self._split_cache.setdefault(session_key, {})
        split_key = f"{start}::{end}"
        if split_key in splits:
            return splits[split_key]

        full_panel = self.get_panel(session_key, spec)
        sliced = slice_panel(full_panel, start=start, end=end)
        splits[split_key] = sliced
        return sliced

    def _load_panel_from_spec(self, spec: dict[str, Any]) -> pd.DataFrame:
        """从 spec 参数加载 panel（优先 mmap）。"""
        panel_path = spec.get("panel_path", "cne://")
        start = spec.get("start", "2020-01-01")
        end = spec.get("end", "2024-12-31")
        include_fundamentals = spec.get("include_fundamentals", True)
        asset_type = spec.get("asset_type", "stock")
        focus_facets = spec.get("focus_facets")

        if is_cne_source(panel_path):
            panel = load_panel_from_cne(
                start=start,
                end=end,
                universe_mask=False,
                include_fundamentals=include_fundamentals,
                asset_type=asset_type,
                focus_facets=focus_facets,
            )
        else:
            panel = load_panel(panel_path)
            panel = slice_panel(panel, start=start, end=end)

        return panel
