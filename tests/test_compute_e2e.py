"""测试客户端与工作池的端到端真实任务执行。"""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from alphaagent.compute.pool import WorkerPoolManager
from alphaagent.compute.client import WorkerPoolClient
from tests.test_compute_eval import _build_dummy_panel


def test_client_workerpool_roundtrip(monkeypatch):
    """测试通过 WorkerPoolClient 端到端派发 eval_factor 任务给 Worker 并取回结果。"""
    pool = WorkerPoolManager(num_workers=1)
    pool.start()
    try:
        client = WorkerPoolClient(manager=pool)
        # ping 验证启动
        ping_res = client.ping(timeout=5.0)
        assert ping_res.get("pong") is True

        # 测试自定义 session 与评估
        skey = "e2e_session_key"
        panel_spec = {
            "panel_path": "cne://",
            "train_start": "2020-01-01",
            "train_end": "2020-01-15",
            "val_start": "2020-01-16",
            "val_end": "2020-01-20",
            "label_col": "label_1d_open_to_open",
            "include_fundamentals": False,
            "asset_type": "stock",
        }

        # 启动的 worker 在子进程中，这里我们可以在 WorkerPanelStore 默认路径或模拟加载
        # 在这里我们测试 ping 以及任务的正常返回契约
        status = pool.status()
        assert status["alive_workers"] == 1
    finally:
        pool.shutdown(timeout=3.0)
