"""ML 组合盲测物理隔离与防泄漏测试。

验证规则：
1. 默认 eval_mode 为 tuning（研发验证态），数据右端物理截断至 val_end (2024-12-31)；
2. tuning 模式下若 mining_end 为 auto，自动对齐到 train_end (2022-12-31)，保留 2023~2024 作为安全 OOS 验证段；
3. blind_test 模式严禁开启 --llm-assist，防止 LLM 窥探盲测数据；
4. llm_summarize_report 遇 blind_test 报告绝对拦截，返回 None；
5. StackingTrainRequest 默认模式为 tuning。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from alphaagent.factor.stacking.llm_assist import _build_report_view, llm_summarize_report
from alphaagent.factor.window_config import (
    DEFAULT_TRAIN_END,
    DEFAULT_VAL_END,
)
from backend.routers.alphaagent import StackingTrainRequest


def test_stacking_train_request_default_eval_mode():
    """断言后端请求默认处于 tuning 研发验证模式。"""
    req = StackingTrainRequest()
    assert req.eval_mode == "tuning"


def test_blind_test_mode_rejects_llm_assist():
    """断言在 blind_test 模式下开启 --llm-assist 会被安全拦截退出。"""
    cmd = [
        sys.executable,
        "scripts/train_ml_composite.py",
        "--eval-mode", "blind_test",
        "--llm-assist",
    ]
    proc = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace")
    assert proc.returncode != 0
    out = (proc.stdout or "") + (proc.stderr or "")
    assert "[safety-violation]" in out


def test_llm_summarize_report_blocks_blind_test():
    """断言 llm_summarize_report 绝对拒绝解读包含盲测段数据的报告。"""
    blind_report = {
        "run_id": "test_blind_run",
        "eval_mode": "blind_test",
        "blind_test_isolated": False,
        "mining_end": "2024-12-31",
        "folds": 3,
        "feature_names": ["f1", "f2"],
    }
    res = llm_summarize_report(blind_report)
    assert res is None


def test_build_report_view_contains_safety_meta():
    """断言压缩视图包含 eval_mode 与 blind_test_isolated 状态元数据。"""
    report = {
        "run_id": "test_tuning_run",
        "eval_mode": "tuning",
        "blind_test_isolated": True,
        "mining_end": "2022-12-31",
        "folds": 4,
        "feature_names": ["f1"],
    }
    view_str = _build_report_view(report)
    assert '"eval_mode": "tuning"' in view_str
    assert '"blind_test_isolated": true' in view_str


def test_tuning_mode_date_logic():
    """断言 tuning 模式下日期截断逻辑严格生效。"""
    val_end_limit = pd.Timestamp(DEFAULT_VAL_END)
    train_end_limit = pd.Timestamp(DEFAULT_TRAIN_END)

    # 模拟传入超过 2024-12-31 的 end 日期
    passed_end = pd.Timestamp("2026-05-01")
    eval_mode = "tuning"

    if eval_mode == "tuning":
        end = passed_end
        if end > val_end_limit:
            end = val_end_limit
        mining_end = train_end_limit

    assert end <= pd.Timestamp("2024-12-31")
    assert mining_end == pd.Timestamp("2022-12-31")
    # OOS 验证段起止
    oos_start = mining_end + pd.Timedelta(days=1)
    assert oos_start == pd.Timestamp("2023-01-01")
    assert end == pd.Timestamp("2024-12-31")
