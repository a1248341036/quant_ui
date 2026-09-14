"""测试算力工作池（Compute Worker Pool）的生命周期与通信。"""
from __future__ import annotations

import time
import pytest
from alphaagent.compute.pool import WorkerPoolManager
from alphaagent.compute.client import WorkerPoolClient
from alphaagent.compute.task import ComputeTask


def test_worker_pool_ping_lifecycle():
    """测试工作池启动、ping 测试任务分发、以及正常停机。"""
    pool = WorkerPoolManager(num_workers=2)
    try:
        pool.start()
        status = pool.status()
        assert status["running"] is True
        assert status["total_workers"] == 2
        assert status["alive_workers"] == 2

        client = WorkerPoolClient(manager=pool)
        resp = client.ping(timeout=5.0)
        assert resp.get("pong") is True
        assert "worker_id" in resp
        assert "pid" in resp
    finally:
        pool.shutdown(timeout=3.0)
        assert pool.status()["running"] is False
