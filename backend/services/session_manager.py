"""会话管理器：LRU 缓存策略管理评估会话生命周期。"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alphaagent.factor.mining.service import StockEvalService
    from alphaagent.factor.mining.schemas import SessionCreateRequest
    from alphaagent.factor.mining.session import StockEvalSession


class LRUSessionCache:
    """LRU 缓存池，管理多个评估会话。"""

    def __init__(self, max_size: int = 3):
        """
        初始化 LRU 缓存。

        Args:
            max_size: 最大缓存会话数，默认 3（约 6-15GB 内存）
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, StockEvalSession] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> StockEvalSession | None:
        """
        获取缓存的会话。

        Args:
            key: 会话参数哈希键

        Returns:
            如果命中缓存则返回会话并更新 LRU 顺序，否则返回 None
        """
        with self._lock:
            if key not in self._cache:
                return None
            # 移动到末尾（最近使用）
            self._cache.move_to_end(key)
            return self._cache[key]

    def put(self, key: str, session: StockEvalSession) -> None:
        """
        将会话加入缓存。

        Args:
            key: 会话参数哈希键
            session: 要缓存的会话

        如果缓存已满，会自动淘汰最久未使用的会话。
        """
        with self._lock:
            if key in self._cache:
                # 已存在，更新并移动到末尾
                self._cache.move_to_end(key)
                self._cache[key] = session
                return

            # 缓存已满，淘汰最久未使用的
            if len(self._cache) >= self.max_size:
                oldest_key, oldest_session = self._cache.popitem(last=False)
                self._cleanup_session(oldest_session)

            self._cache[key] = session

    def evict_all(self) -> None:
        """清空所有缓存的会话。"""
        with self._lock:
            for session in self._cache.values():
                self._cleanup_session(session)
            self._cache.clear()

    def _cleanup_session(self, session: StockEvalSession) -> None:
        """
        清理会话，释放内存。

        Args:
            session: 要清理的会话
        """
        try:
            # 清空 split 缓存
            if hasattr(session, "_split_cache"):
                session._split_cache.clear()
            # 释放 panel 引用（依赖 Python GC）
            if hasattr(session, "panel"):
                session.panel = None  # type: ignore
        except Exception:
            # 清理失败不影响主流程
            pass


class SessionManager:
    """会话管理器，提供 LRU 缓存的会话访问。"""

    def __init__(self, max_cached_sessions: int = 3):
        """
        初始化会话管理器。

        Args:
            max_cached_sessions: 最大缓存会话数，默认 3
        """
        self._cache = LRUSessionCache(max_size=max_cached_sessions)
        self._lock = threading.RLock()

    def _hash_params(self, req: SessionCreateRequest) -> str:
        """
        生成会话参数的唯一指纹。

        Args:
            req: 会话创建请求

        Returns:
            SHA256 哈希值的前 16 位字符
        """
        params = {
            "panel_path": str(req.panel_path),
            "train_start": req.train_start,
            "train_end": req.train_end,
            "val_start": req.val_start,
            "val_end": req.val_end,
            "label_col": req.label_col,
            "include_fundamentals": req.include_fundamentals,
        }
        # 生成 SHA256 指纹
        param_str = json.dumps(params, sort_keys=True, default=str)
        return hashlib.sha256(param_str.encode()).hexdigest()[:16]

    def get_or_create_session(
        self, req: SessionCreateRequest
    ) -> StockEvalSession:
        """
        获取或创建评估会话。

        Args:
            req: 会话创建请求

        Returns:
            评估会话实例

        如果参数相同且缓存未过期，直接返回缓存的会话；
        否则创建新会话并加入 LRU 缓存。
        """
        cache_key = self._hash_params(req)

        # 尝试命中缓存
        session = self._cache.get(cache_key)
        if session is not None:
            return session

        # 创建新会话
        from alphaagent.factor.mining.context import StockEvalContext
        from alphaagent.factor.mining.session import SessionStore

        ctx = StockEvalContext(
            panel_path=req.panel_path,
            train_start=req.train_start,
            train_end=req.train_end,
            val_start=req.val_start,
            val_end=req.val_end,
            label_col=req.label_col,
            include_fundamentals=req.include_fundamentals,
        )
        session = SessionStore().create(ctx)

        # 加入缓存
        self._cache.put(cache_key, session)
        return session

    def evict_all(self) -> None:
        """清空所有缓存的会话，释放内存。"""
        self._cache.evict_all()

    def get_cache_stats(self) -> dict:
        """
        获取缓存统计信息。

        Returns:
            包含缓存大小、最大容量等信息的字典
        """
        with self._cache._lock:
            return {
                "current_size": len(self._cache._cache),
                "max_size": self._cache.max_size,
                "keys": list(self._cache._cache.keys()),
            }
