"""StockEvalSession：会话级 panel 驻内存与 split 懒缓存。"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from alphaagent.data.panel import load_panel, slice_panel
from alphaagent.factor.cache import FactorValueCache, get_default_cache
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
    factor_cache: FactorValueCache = field(default_factory=get_default_cache, repr=False)
    _column_autocorr_cache: dict[str, float] = field(default_factory=dict, repr=False)

    def get_column_autocorr(self, col: str, default_val: float = 0.5) -> float:
        """获取列的截面一阶自相关（优先读会话级缓存，缺失则现算一次并缓存）。"""
        col = str(col).lstrip("$")
        if col in self._column_autocorr_cache:
            return self._column_autocorr_cache[col]
        # 基础常用列的先验基线（秒级命中避免开销）
        _KNOWN_BASE_AUTOCORRS = {
            "close": 0.95, "adj_close": 0.95, "open": 0.94, "adj_open": 0.94,
            "high": 0.94, "adj_high": 0.94, "low": 0.94, "adj_low": 0.94,
            "vwap": 0.95, "adj_vwap": 0.95, "volume": 0.28, "amount": 0.32,
            "turnover": 0.30, "ret": -0.05, "returns": -0.05,
            "float_cap": 0.99, "total_cap": 0.99,
        }
        if col in _KNOWN_BASE_AUTOCORRS:
            val = _KNOWN_BASE_AUTOCORRS[col]
            self._column_autocorr_cache[col] = val
            return val
        if self.panel is not None and col in self.panel.columns:
            try:
                from alphaagent.factor.metrics._core import pearson_autocorr
                # 取 100 天样本快速计算截面自相关
                s = self.panel[col].dropna()
                val = float(pearson_autocorr(s.to_numpy())) if len(s) > 100 else default_val
                if not (np.isfinite(val)):
                    val = default_val
            except Exception:
                val = default_val
        else:
            val = default_val
        self._column_autocorr_cache[col] = val
        return val

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
                include_fundamentals=ctx.include_fundamentals,
                asset_type=ctx.asset_type,
                focus_facets=getattr(ctx, "focus_facets", ()) or None,
            )
        else:
            panel = load_panel(ctx.panel_path)

        dropped_funda = 0
        if not ctx.include_fundamentals:
            funda_cols = [c for c in panel.columns if str(c).startswith("funda_")]
            if funda_cols:
                panel = panel.drop(columns=funda_cols)
                dropped_funda = len(funda_cols)
            else:
                dropped_funda = 0
        else:
            dropped_funda = 0
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
                "asset_type": ctx.asset_type,
                "dropped_fundamental_cols": dropped_funda,
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

    def register(self, session: StockEvalSession) -> None:
        """将已存在的会话注册进存储（供外部缓存复用的会话接入评估服务）。"""
        with self._lock:
            self._sessions[session.session_id] = session

    def remove(self, session_id: str) -> bool:
        """移除会话并释放其 panel 引用（清空 split 缓存、置空 panel）。

        一次性评估场景调用此方法，避免多份全量 panel 常驻内存。
        """
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        try:
            session._split_cache.clear()
            session.panel = None  # type: ignore[assignment]
        except Exception:
            pass
        return True
