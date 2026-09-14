"""算力工作池客户端（WorkerPoolClient）。

调用方（Web 服务、脚本、智能体）通过该 Client 统一向 WorkerPoolManager 提交评估和回测任务。
提供单例获取机制以及同步/异步提交方法。
"""
from __future__ import annotations

import logging
import threading
from typing import Any

from alphaagent.compute.pool import WorkerPoolManager
from alphaagent.compute.task import ComputeTask

logger = logging.getLogger(__name__)

_GLOBAL_POOL: WorkerPoolManager | None = None
_POOL_LOCK = threading.Lock()


def get_global_worker_pool(num_workers: int | None = None) -> WorkerPoolManager:
    """获取全局单例的 WorkerPoolManager。"""
    global _GLOBAL_POOL
    with _POOL_LOCK:
        if _GLOBAL_POOL is None:
            _GLOBAL_POOL = WorkerPoolManager(num_workers=num_workers)
        return _GLOBAL_POOL


def shutdown_global_worker_pool() -> None:
    """关闭全局 WorkerPoolManager。"""
    global _GLOBAL_POOL
    with _POOL_LOCK:
        if _GLOBAL_POOL is not None:
            _GLOBAL_POOL.shutdown()
            _GLOBAL_POOL = None


class WorkerPoolClient:
    """算力池轻量客户端接口。"""

    def __init__(self, manager: WorkerPoolManager | None = None) -> None:
        self._manager = manager

    @property
    def manager(self) -> WorkerPoolManager:
        if self._manager is not None:
            return self._manager
        return get_global_worker_pool()

    def ping(self, timeout: float = 10.0) -> dict[str, Any]:
        """向工作池发送 ping 检测。"""
        task = ComputeTask(task_type="ping", params={}, priority=1)
        fut = self.manager.submit_task(task)
        return fut.result(timeout=timeout)

    def evaluate_factor(
        self,
        *,
        session_key: str,
        panel_spec: dict[str, Any],
        multi_line_expr: str,
        factor_name: str = "expr",
        profile_id: str = "train_screen",
        label_quantile_n: int = 0,
        include_charts: bool = True,
        include_detail_tables: bool = False,
        priority: int = 1,
        timeout: float = 300.0,
    ) -> dict[str, Any]:
        """同步提交单因子评估任务并等待结果。"""
        params = {
            "session_key": session_key,
            "panel_spec": panel_spec,
            "multi_line_expr": multi_line_expr,
            "factor_name": factor_name,
            "profile_id": profile_id,
            "label_quantile_n": label_quantile_n,
            "include_charts": include_charts,
            "include_detail_tables": include_detail_tables,
        }
        task = ComputeTask(task_type="eval_factor", params=params, priority=priority)
        fut = self.manager.submit_task(task)
        return fut.result(timeout=timeout)

    def run_engine_gate(
        self,
        *,
        session_key: str,
        panel_spec: dict[str, Any],
        multi_line_expr: str,
        val_start: str,
        val_end: str,
        direction: int = 1,
        policy: dict[str, Any] | None = None,
        asset_type: str = "stock",
        priority: int = 1,
        timeout: float = 300.0,
    ) -> dict[str, Any]:
        """同步提交 engine_gate 任务并等待结果。"""
        params = {
            "session_key": session_key,
            "panel_spec": panel_spec,
            "multi_line_expr": multi_line_expr,
            "val_start": val_start,
            "val_end": val_end,
            "direction": direction,
            "policy": policy,
            "asset_type": asset_type,
        }
        task = ComputeTask(task_type="engine_gate", params=params, priority=priority)
        fut = self.manager.submit_task(task)
        return fut.result(timeout=timeout)
