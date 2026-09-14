"""AlphaAgent 算力工作池模块。"""
from alphaagent.compute.client import WorkerPoolClient, get_global_worker_pool, shutdown_global_worker_pool
from alphaagent.compute.pool import WorkerPoolManager
from alphaagent.compute.task import ComputeTask, TaskResult

__all__ = [
    "ComputeTask",
    "TaskResult",
    "WorkerPoolManager",
    "WorkerPoolClient",
    "get_global_worker_pool",
    "shutdown_global_worker_pool",
]
