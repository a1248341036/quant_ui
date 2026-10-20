# -*- coding: utf-8 -*-
"""增量轨（双轨判定）回归测试 —— 2026-10-20。

**背景**：候选池的实际用途是 ML 样本源，但候选门是按"单因子能否独立交易"定的。
实测候选池 47 个因子的组合增量比中位数只有 0.264（illiq/vwap 族几乎纯冗余），
而两套口径的名单重合度仅 1/3。增量轨允许"raw 门未过但对组合有明显增量"的因子
进池，**是加法不是替换**：raw 门通过者照旧放行，行为完全不变。

**为什么必须有 raw_ic 地板**：increment 是比值，分母趋零会爆表 —— 实测
`x_vol_pe`（raw_ic +0.0000 / resid_ic +0.0014）算出 increment **53.26**。
没有地板的话纯噪声因子会因为"残差占比高"而进池。
"""
from __future__ import annotations

import dataclasses

import pytest

from alphaagent.factor.mining.delivery.delivery_checker import StageOneStats
from alphaagent.factor.mining.delivery.delivery_criteria import (
    CandidateCriteria,
    DeliveryCriteria,
)


def _criteria(**cand_overrides) -> DeliveryCriteria:
    base = candidate_default()
    return dataclasses.replace(base, candidate=dataclasses.replace(base.candidate, **cand_overrides))


def candidate_default() -> DeliveryCriteria:
    return DeliveryCriteria.from_spec({})


class TestCriteriaDefaults:
    def test_defaults_present(self):
        c = candidate_default().candidate
        assert c.min_raw_ic_floor == 0.020
        assert c.min_resid_ic == 0.010
        assert c.min_increment_ratio == 0.30
        assert c.enable_increment_track is True

    def test_raw_gate_unchanged(self):
        """增量轨是加法：原有 raw 门阈值一个都不许动。"""
        c = candidate_default().candidate
        assert c.min_abs_ic == 0.020  # 1d 档默认；月频由 mode override 覆盖
        assert c.min_icir == 0.28
        assert c.min_coverage == 0.85


class TestIncrementTrack:
    def _run(self, metrics, **overrides):
        crit = DeliveryCriteria.from_spec({})
        return StageOneStats(crit).run({"metrics": metrics}), crit

    def test_raw_pass_unchanged(self):
        """raw 门全过 → 通过，且不应标记 increment_track。"""
        crit = DeliveryCriteria.from_spec({})
        res = StageOneStats(crit).run({"metrics": {
            "ic": 0.09, "icir": 0.9, "coverage": 0.95,
        }})
        assert res.passed is True
        assert "increment_track" not in res.extra

    def test_rescued_by_increment(self):
        """raw 门不过（IC 低 / ICIR 低）但增量三条满足 → 放行并标记。"""
        crit = DeliveryCriteria.from_spec({})
        res = StageOneStats(crit).run({"metrics": {
            "ic": 0.030, "icir": 0.20, "coverage": 0.93,   # raw 门不过（icir 0.20 < 0.28）
            "resid_ic": 0.015, "increment": 0.55,          # 增量门过
        }})
        assert res.passed is True
        assert res.extra.get("increment_track") is True
        assert "icir" in res.extra.get("raw_gate_failed", [])

    def test_noise_blocked_by_raw_floor(self):
        """分母趋零的噪声因子必须被 raw_ic 地板拦掉（increment 可能爆表 53.26）。"""
        crit = DeliveryCriteria.from_spec({})
        res = StageOneStats(crit).run({"metrics": {
            "ic": 0.0000, "icir": 0.01, "coverage": 0.93,
            "resid_ic": 0.0014, "increment": 53.26,
        }})
        assert res.passed is False
        assert any("increment:raw_below_floor" in r for r in res.fail_reasons)

    def test_resid_below_gate(self):
        crit = DeliveryCriteria.from_spec({})
        res = StageOneStats(crit).run({"metrics": {
            "ic": 0.030, "icir": 0.20, "coverage": 0.93,
            "resid_ic": 0.004, "increment": 0.60,   # 残差过小
        }})
        assert res.passed is False
        assert any("increment:resid_ic" in r for r in res.fail_reasons)

    def test_increment_ratio_below_gate(self):
        crit = DeliveryCriteria.from_spec({})
        res = StageOneStats(crit).run({"metrics": {
            "ic": 0.060, "icir": 0.20, "coverage": 0.93,
            "resid_ic": 0.012, "increment": 0.15,   # 高度重叠
        }})
        assert res.passed is False
        assert any("increment:increment_ratio" in r for r in res.fail_reasons)

    def test_missing_metrics_not_rescued(self):
        """没有增量指标时不允许放行（缺数据 ≠ 通过）。"""
        crit = DeliveryCriteria.from_spec({})
        res = StageOneStats(crit).run({"metrics": {
            "ic": 0.030, "icir": 0.20, "coverage": 0.93,
        }})
        assert res.passed is False
        assert any("increment:missing_metrics" in r for r in res.fail_reasons)

    def test_disabled_falls_back_to_raw_only(self):
        """开关关掉 → 完全退回旧行为（增量指标存在也不放行）。"""
        crit = DeliveryCriteria.from_spec({})
        crit_obj = dataclasses.replace(
            crit, candidate=dataclasses.replace(crit.candidate, enable_increment_track=False)
        )
        res = StageOneStats(crit_obj).run({"metrics": {
            "ic": 0.030, "icir": 0.20, "coverage": 0.93,
            "resid_ic": 0.015, "increment": 0.55,
        }})
        assert res.passed is False
        assert res.fail_reasons == ["icir"]

    def test_negative_ic_rescued_symmetrically(self):
        """方向取绝对值：负 IC 的因子同样可以用增量轨放行。"""
        crit = DeliveryCriteria.from_spec({})
        res = StageOneStats(crit).run({"metrics": {
            "ic": -0.027, "icir": -0.20, "coverage": 0.93,
            "resid_ic": -0.0123, "increment": 0.458,
        }})
        assert res.passed is True
        assert res.extra.get("increment_track") is True
