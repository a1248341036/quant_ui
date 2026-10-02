# -*- coding: utf-8 -*-
"""P5 阈值收口回归测试（2026-10-02）。

被收口的 4 处硬编码（原先散落在业务逻辑里，违反项目"阈值收口配置中心"纪律）：
  ① agentscope_tools._ORTHO_MAX_CORR = 0.7      → evaluation_policy.orthogonality_max_corr
  ② _dispatch._PREDICTION_SOFT_LIMIT = 3        → evaluation_policy.prediction_soft_limit
  ③ eval/prediction.py 硬编码 0.003（两处）      → evaluation_policy.prediction_sign_min_abs_ic
  ④ memory/retrieval.py 硬编码 0.4（饱和度）     → memory.constants.SATURATION_FAMILY_SKIP
"""
from __future__ import annotations

from alphaagent.factor.evaluation.defaults import DEFAULT_EVALUATION_POLICY
from alphaagent.factor.mining import research_spec as rs
from alphaagent.factor.mining.eval.prediction import _sign_min_abs_ic
from alphaagent.factor.mining.memory import constants as mc
from alphaagent.factor.mining.tools._dispatch import _prediction_soft_limit


def test_defaults_carry_the_three_keys():
    assert DEFAULT_EVALUATION_POLICY["orthogonality_max_corr"] == 0.7
    assert DEFAULT_EVALUATION_POLICY["prediction_soft_limit"] == 3
    assert DEFAULT_EVALUATION_POLICY["prediction_sign_min_abs_ic"] == 0.003


def test_memory_constant_exists():
    assert mc.SATURATION_FAMILY_SKIP == 0.4


def test_effective_spec_exposes_and_validates_keys():
    ep = rs.build_run_research_spec(rs.DEFAULT_RESEARCH_SPEC)["evaluation_policy"]
    assert ep["orthogonality_max_corr"] == 0.7
    assert ep["prediction_soft_limit"] == 3
    assert ep["prediction_sign_min_abs_ic"] == 0.003
    # 越界拒绝
    import pytest

    with pytest.raises(ValueError):
        rs.build_run_research_spec(
            {"evaluation_policy": {"orthogonality_max_corr": 5.0}})
    with pytest.raises(ValueError):
        rs.build_run_research_spec(
            {"evaluation_policy": {"prediction_soft_limit": 999}})


def test_read_side_helpers_work():
    """读取侧不再吃硬编码：助手函数返回配置中心的值且类型正确。"""
    assert isinstance(_prediction_soft_limit(), int) and _prediction_soft_limit() >= 1
    assert 0 < _sign_min_abs_ic() <= 0.02


def test_agentscope_tools_reads_gate_then_default():
    from alphaagent.factor.mining.agent.agentscope_tools import _ortho_max_corr

    class _T:
        report_reproduce_gate = {"orthogonality_max_corr": 0.55}

    assert _ortho_max_corr(_T()) == 0.55          # run 网关优先（研报模式可覆盖）
    assert _ortho_max_corr(None) == 0.7           # 否则配置中心默认
    class _Bad:
        report_reproduce_gate = {"orthogonality_max_corr": "oops"}

    assert _ortho_max_corr(_Bad()) == 0.7         # 脏值回落默认，不崩
