# -*- coding: utf-8 -*-
"""统一配置中心收口测试（2026-10-26）。

覆盖（AGENTS.md 统一配置中心纪律三件套）：
1. memory/constants.py 新常量存在且值 = 收口前现值（零行为变化）；
2. retrieval.py / ingestion.py 消费侧已接线（源码无裸数字判定）；
3. research_spec.ml_increment_policy 三件套：默认值 / 覆盖生效 / 越界拒绝；
4. scripts/ml_incremental_mine 判定默认取自 spec（源码级断言）。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from alphaagent.factor.mining.memory import constants
from alphaagent.factor.mining.research_spec import build_run_research_spec


def test_memory_constants_values_unchanged():
    """现值收口：新常量数值与收口前硬编码逐位一致（零行为变化）。"""
    assert constants.RETRIEVAL_TURNOVER_SOFT_PENALTY == 0.45
    assert constants.MULTIPLY_REJECT_RATIO == 0.4
    assert constants.EDIT_DELTA_IC_SUCCESS_THRESHOLD == 0.003
    assert constants.SATURATION_FAMILY_SKIP == 0.4
    assert constants.EDIT_PRIOR_RECOMMEND_CONF_DEFAULT == 0.4


def test_retrieval_wired_to_constants():
    """消费侧接线断言：retrieval.py 判定行不再出现裸 0.45/0.4 魔法数。"""
    path = Path(constants.__file__).parent / "retrieval.py"
    text = path.read_text(encoding="utf-8")
    assert "RETRIEVAL_TURNOVER_SOFT_PENALTY" in text
    assert "> 0.45" not in text  # 换手软惩罚已接线
    assert text.count("SATURATION_FAMILY_SKIP") >= 3  # import + 2 处判定
    assert "MULTIPLY_REJECT_RATIO" in text


def test_ingestion_wired_to_constants():
    path = Path(constants.__file__).parent / "ingestion.py"
    text = path.read_text(encoding="utf-8")
    assert "EDIT_DELTA_IC_SUCCESS_THRESHOLD" in text
    assert "IC_SUCCESS_THRESHOLD = 0.003" not in text  # 函数内局部常量已移除


def test_ml_increment_policy_defaults():
    spec = build_run_research_spec()
    pol = spec["ml_increment_policy"]
    assert pol["enabled"] is True
    assert pol["t_threshold"] == 1.28
    assert pol["min_delta_ic"] == 0.001
    assert pol["label"] == "label_20d_close_to_close"
    assert pol["target_library"] == "ml_increment_main"


def test_ml_increment_policy_override_applies():
    spec = build_run_research_spec({"ml_increment_policy": {"t_threshold": 2.0, "enabled": False}})
    pol = spec["ml_increment_policy"]
    assert pol["t_threshold"] == 2.0
    assert pol["enabled"] is False
    # 未覆盖键回落默认
    assert pol["min_delta_ic"] == 0.001


def test_ml_increment_policy_rejects_out_of_range():
    with pytest.raises(ValueError):
        build_run_research_spec({"ml_increment_policy": {"t_threshold": 99.0}})
    with pytest.raises(ValueError):
        build_run_research_spec({"ml_increment_policy": {"min_delta_ic": -1.0}})


def test_mine_script_reads_spec_defaults():
    repo = Path(constants.__file__).resolve().parents[4]
    mine = repo / "scripts" / "ml_incremental_mine.py"
    text = mine.read_text(encoding="utf-8")
    assert "ml_increment_policy" in text
    assert "DEFAULT_RESEARCH_SPEC" in text
    assert "default=1.28" not in text  # argparse 裸默认已移除
