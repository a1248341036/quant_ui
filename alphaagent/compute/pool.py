"""算力工作池管理器（WorkerPoolManager）。

负责管理子进程 Worker 的启动、健康检查、任务调度与分发。
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import threading
import time
from concurrent.futures import Future
from typing import Any

from alphaagent.compute.task import ComputeTask, TaskResult
from alphaagent.compute.worker import _POISON_PILL, _worker_entrypoint

logger = logging.getLogger(__name__)


def default_num_workers() -> int:
    """计算默认 Worker 数：min(物理核数 - 2, 6)，保底 2。"""
    count = os.cpu_count() or 4
    return max(2, min(count - 2, 6))


class WorkerPoolManager:
    """管理一组执行计算任务的子进程 Worker。"""

    def __init__(self, num_workers: int | None = None) -> None:
        self.num_workers = num_workers if (num_workers and num_workers > 0) else default_num_workers()
        self._ctx = mp.get_context("spawn")
        self._task_queue = self._ctx.Queue()
        self._result_queue = self._ctx.Queue()
        self._workers: list[mp.Process] = []
        self._futures: dict[str, Future[dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._running = False
        self._dispatcher_thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        """启动 Worker 进程池及结果分发线程。"""
        with self._lock:
            if self._running:
                return
            logger.info("正在启动算力工作池，Worker 数量=%d", self.num_workers)
            self._workers.clear()
            for i in range(self.num_workers):
                p = self._ctx.Process(
                    target=_worker_entrypoint,
                    args=(i, self._task_queue, self._result_queue),
                    name=f"ComputeWorker-{i}",
                    daemon=True,
                )
                p.start()
                self._workers.append(p)

            self._running = True
            self._dispatcher_thread = threading.Thread(
                target=self._result_dispatcher_loop,
                name="WorkerPool-Dispatcher",
                daemon=True,
            )
            self._dispatcher_thread.start()
            logger.info("算力工作池启动就绪")

    def submit_task(self, task: ComputeTask) -> Future[dict[str, Any]]:
        """向工作池提交任务，返回 Future 对象。"""
        with self._lock:
            if not self._running:
                raise RuntimeError("WorkerPoolManager 尚未启动或已停止")
            fut: Future[dict[str, Any]] = Future()
            self._futures[task.task_id] = fut
            self._task_queue.put(task)
            return fut

    def _result_dispatcher_loop(self) -> None:
        """轮询结果队列并将结果映射回对应 Future。"""
        while self._running:
            try:
                item = self._result_queue.get(timeout=0.5)
                if not isinstance(item, TaskResult):
                    continue

                with self._lock:
                    fut = self._futures.pop(item.task_id, None)

                if fut is not None and not fut.done():
                    if item.ok:
                        fut.set_result(item.result or {})
                    else:
                        fut.set_exception(RuntimeError(f"[{item.error_type}] {item.error}"))
            except mp.queues.Empty:
                continue
            except Exception as e:
                logger.error("结果分发线程异常: %s", e)

    def shutdown(self, timeout: float = 10.0) -> None:
        """优雅关闭 Worker 池。"""
        with self._lock:
            if not self._running:
                return
            self._running = False

            # 发送 poison pill 通知各 worker 退出
            for _ in range(len(self._workers)):
                try:
                    self._task_queue.put(_POISON_PILL)
                except Exception:
                    pass

        t0 = time.time()
        for p in self._workers:
            join_time = max(0.1, timeout - (time.time() - t0))
            p.join(timeout=join_time)
            if p.is_alive():
                logger.warning("Worker %s 优雅停机超时，强制终止", p.name)
                p.terminate()

        # 拒绝仍挂起的 Future
        with self._lock:
            for fut in self._futures.values():
                if not fut.done():
                    fut.set_exception(RuntimeError("WorkerPoolManager 已停机"))
            self._futures.clear()

        logger.info("算力工作池已关闭")

    def status(self) -> dict[str, Any]:
        """获取当前工作池状态统计。"""
        alive_count = sum(1 for p in self._workers if p.is_alive())
        return {
            "running": self._running,
            "total_workers": self.num_workers,
            "alive_workers": alive_count,
            "pending_futures": len(self._futures),
        }
