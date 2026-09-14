"""测试通过 ComputeWorker / WorkerPool 执行因子评估的一致性与正确性。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaagent.compute.client import WorkerPoolClient
from alphaagent.compute.pool import WorkerPoolManager
from alphaagent.compute.worker import ComputeWorker
from alphaagent.factor.evaluation.engine import EvaluationEngine
from alphaagent.factor.evaluation.profile import default_evaluation_profiles


def _build_dummy_panel() -> pd.DataFrame:
    """生成测试用小型股票面板。"""
    dates = pd.date_range("2020-01-01", "2020-01-20", freq="B")
    codes = ["000001.SZ", "600000.SH", "000002.SZ"]
    idx = pd.MultiIndex.from_product([dates, codes], names=["datetime", "instrument"])
    np.random.seed(42)
    n = len(idx)
    close = 10.0 + np.cumsum(np.random.randn(n) * 0.1)
    label = np.random.randn(n) * 0.02
    return pd.DataFrame({"close": close, "label_1d_open_to_open": label}, index=idx)


def test_worker_direct_eval(tmp_path):
    """验证 Worker 内部解析与评估逻辑返回合法结果。"""
    worker = ComputeWorker(worker_id=0, task_queue=None, result_queue=None)
    dummy_panel = _build_dummy_panel()

    # 预置 panel 到 worker 的 panel_store 缓存中
    skey = "dummy_session"
    worker.panel_store._cache[skey] = dummy_panel

    params = {
        "session_key": skey,
        "panel_spec": {
            "panel_path": "cne://",
            "train_start": "2020-01-01",
            "train_end": "2020-01-15",
            "val_start": "2020-01-16",
            "val_end": "2020-01-20",
            "label_col": "label_1d_open_to_open",
            "include_fundamentals": False,
            "asset_type": "stock",
        },
        "multi_line_expr": "RANK($close)",
        "factor_name": "rank_close",
        "profile_id": "train_screen",
        "label_quantile_n": 0,
        "include_charts": False,
        "include_detail_tables": False,
    }

    result = worker._exec_eval_factor(params)
    assert result.get("ok") is True
    assert result.get("candidate", {}).get("factor_name") == "rank_close"
    assert "metrics" in result
    assert "cross_sectional_core" in result["metrics"]
    ic = result["metrics"]["cross_sectional_core"].get("ic")
    assert ic is not None
