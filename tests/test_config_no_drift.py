import re
from pathlib import Path
import pytest

from alphaagent.factor.mining.delivery.delivery_criteria import CandidateCriteria, ProductionCriteria
from alphaagent.factor.mining.research_spec import DEFAULT_RESEARCH_SPEC


def test_core_thresholds_alignment():
    """断言 DeliveryCriteria 与 DEFAULT_RESEARCH_SPEC 核心门槛数值严格一致。"""
    from alphaagent.factor.mining.delivery.delivery_criteria import DeliveryCriteria
    cc = DeliveryCriteria.defaults().candidate
    ep = DEFAULT_RESEARCH_SPEC["evaluation_policy"]
    dp_cand = DEFAULT_RESEARCH_SPEC["delivery_policy"]["candidate"]

    # Candidate vs evaluation_policy
    assert cc.min_abs_ic == ep["min_train_abs_ic"] == 0.020
    assert cc.min_icir == ep["min_train_icir"] == 0.28
    assert cc.min_coverage == ep["min_train_coverage"] == 0.85
    assert cc.min_val_abs_ic == ep["min_val_abs_ic"] == 0.012
    assert cc.min_val_ic_retention == ep["min_val_ic_retention_ratio"] == 0.5

    # Candidate vs delivery_policy
    assert cc.min_abs_ic == dp_cand["min_abs_ic"]
    assert cc.min_icir == dp_cand["min_icir"]
    assert cc.min_coverage == dp_cand["min_coverage"]
    assert cc.min_val_abs_ic == dp_cand["min_val_abs_ic"]


def test_no_hardcoded_threshold_drift_in_consumer_files():
    """断言消费侧关键文件中不再出现硬编码门槛 fallback 或手写旧阈值。"""
    repo_root = Path(__file__).resolve().parent.parent

    # 1. factor_reviewer.py: 不得包含硬编码旧 fallback 0.015 或 0.2
    reviewer_file = repo_root / "alphaagent/factor/mining/agent/factor_reviewer.py"
    reviewer_text = reviewer_file.read_text(encoding="utf-8")
    assert 'policy.get("min_train_abs_ic", 0.015)' not in reviewer_text
    assert 'policy.get("min_train_icir", 0.2)' not in reviewer_text

    # 2. population.py: 不得包含写死 icir <= 0.25
    pop_file = repo_root / "alphaagent/factor/mining/agent/population.py"
    pop_text = pop_file.read_text(encoding="utf-8")
    assert "<= 0.25" not in pop_text

    # 3. agentscope_run.py: overfit 判定不得写死 abs(train_ic) >= 0.015
    run_file = repo_root / "alphaagent/factor/mining/agent/agentscope_run.py"
    run_text = run_file.read_text(encoding="utf-8")
    assert "abs(train_ic) >= 0.015" not in run_text

    # 4. agentscope_tools.py: 评估超时由 MiningConfig 统一控制，不得包含 _EVAL_TIMEOUT_SECONDS = 600
    tools_file = repo_root / "alphaagent/factor/mining/agent/agentscope_tools.py"
    tools_text = tools_file.read_text(encoding="utf-8")
    assert "_EVAL_TIMEOUT_SECONDS = 600" not in tools_text
