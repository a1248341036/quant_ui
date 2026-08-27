"""StockEvalSession：会话级 panel 驻内存与 split 懒缓存。"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from alphaagent.data.panel import load_panel, slice_panel
from alphaagent.factor.mining.context import StockEvalContext
from alphaagent.factor.evaluation.candidate import CandidateRegistry


@dataclass
class StockEvalSession:
    session_id: str
    ctx: StockEvalContext
    panel: pd.DataFrame
    meta: dict[str, Any] = field(default_factory=dict)
    candidates: CandidateRegistry = field(default_factory=CandidateRegistry, repr=False)
    created_at: float = field(default_factory=time.time)
    _split_cache: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)
    _split_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def get_split_panel(self, split: str) -> tuple[pd.DataFrame, str, str]:
        """返回 (panel_slice, start, end)。"""
        start, end = self.ctx.split_range(split)
        with self._split_lock:
            cached = self._split_cache.get(split)
            if cached is not None:
                return cached, start, end
            sliced = slice_panel(self.panel, start=start, end=end)
            self._split_cache[split] = sliced
            return sliced, start, end


class SessionStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, StockEvalSession] = {}

    def create(self, ctx: StockEvalContext) -> StockEvalSession:
        t0 = time.perf_counter()
        cov_start, cov_end = ctx.coverage_range()

        # 数据源分支：cne:// 走 adapter 实时构建，否则读 parquet 文件
        from alphaagent.data.adapters.cnequity import is_cne_source, load_panel_from_cne
        if is_cne_source(ctx.panel_path):
            panel = load_panel_from_cne(
                start=cov_start, end=cov_end, universe_mask=False,
            )
        else:
            panel = load_panel(ctx.panel_path)

        dropped_funda = 0
        if not ctx.include_fundamentals:
            funda_cols = [c for c in panel.columns if str(c).startswith("funda_")]
            if funda_cols:
                panel = panel.drop(columns=funda_cols)
                dropped_funda = len(funda_cols)
        panel = slice_panel(panel, start=cov_start, end=cov_end)
        load_ms = (time.perf_counter() - t0) * 1000
        session_id = uuid.uuid4().hex
        session = StockEvalSession(
            session_id=session_id,
            ctx=ctx,
            panel=panel,
            meta={
                "load_ms": load_ms,
                "coverage_start": cov_start,
                "coverage_end": cov_end,
                "include_fundamentals": ctx.include_fundamentals,
                "dropped_fundamental_cols": dropped_fundamental_cols,
            },
        )
        with self._lock:
            self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> StockEvalSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"未知 session_id: {session_id!r}")
        return session
