"""盲测终审门禁（BlindTestStage）单测。

覆盖：
- enabled=False 时跳过
- IC 保留比 ≥ 阈值 + 方向一致 → 通过
- IC 保留比 < 阈值 → blind_test_ic_retention
- 方向反转 → blind_test_sign_flip
- IC 缺失 → blind_test_ic_missing
- train IC 近零 → blind_test_train_ic_near_zero
"""

from __future__ import annotations

import pytest

from alphaagent.factor.mining.delivery_criteria import BlindTestCriteria, DeliveryCriteria
from alphaagent.factor.mining.delivery_checker import BlindTestStage, DeliveryChecker


@pytest.fixture()
def checker() -> DeliveryChecker:
    return DeliveryChecker(DeliveryCriteria())


class TestBlindTestStage:
    def test_disabled_skips(self) -> None:
        c = DeliveryCriteria(blind_test=BlindTestCriteria(enabled=False))
        stage = BlindTestStage(c)
        result = stage.run({
            "train_metrics": {"ic": 0.05},
            "test_metrics": {"ic": -0.10},
        })
        assert result.passed is True
        assert result.fail_reasons == []

    def test_pass_when_retention_ok_and_sign_consistent(self, checker) -> None:
        """train IC=0.05, test IC=0.03 → 保留比 0.6 ≥ 0.50, 方向一致 → 通过。"""
        result = checker.blind_test(
            {"ic": 0.05},
            {"ic": 0.03},
        )
        assert result.passed is True
        assert result.fail_reasons == []

    def test_fail_when_retention_too_low(self, checker) -> None:
        """train IC=0.05, test IC=0.01 → 保留比 0.2 < 0.50 → 拦截。"""
        result = checker.blind_test(
            {"ic": 0.05},
            {"ic": 0.01},
        )
        assert result.passed is False
        assert "blind_test_ic_retention" in result.fail_reasons

    def test_fail_when_sign_flip(self, checker) -> None:
        """train IC=0.05, test IC=-0.03 → 保留比 0.6 但方向反转 → 拦截。"""
        result = checker.blind_test(
            {"ic": 0.05},
            {"ic": -0.03},
        )
        assert result.passed is False
        assert "blind_test_sign_flip" in result.fail_reasons

    def test_fail_when_both_retention_and_sign(self, checker) -> None:
        """train IC=0.05, test IC=-0.01 → 保留比 0.2 + 方向反转 → 两项都拦。"""
        result = checker.blind_test(
            {"ic": 0.05},
            {"ic": -0.01},
        )
        assert result.passed is False
        assert "blind_test_sign_flip" in result.fail_reasons
        assert "blind_test_ic_retention" in result.fail_reasons

    def test_fail_when_test_ic_missing(self, checker) -> None:
        result = checker.blind_test(
            {"ic": 0.05},
            {},
        )
        assert result.passed is False
        assert result.fail_reasons == ["blind_test_ic_missing"]

    def test_fail_when_train_ic_missing(self, checker) -> None:
        result = checker.blind_test(
            {},
            {"ic": 0.03},
        )
        assert result.passed is False
        assert result.fail_reasons == ["blind_test_ic_missing"]

    def test_fail_when_train_ic_near_zero(self, checker) -> None:
        """train IC ≈ 0 时无法算保留比 → blind_test_train_ic_near_zero。"""
        result = checker.blind_test(
            {"ic": 1e-13},
            {"ic": 0.03},
        )
        assert result.passed is False
        assert "blind_test_train_ic_near_zero" in result.fail_reasons

    def test_negative_ic_passes_when_consistent(self, checker) -> None:
        """负 IC 因子：train=-0.05, test=-0.03 → 方向一致，保留比 0.6 → 通过。"""
        result = checker.blind_test(
            {"ic": -0.05},
            {"ic": -0.03},
        )
        assert result.passed is True

    def test_custom_threshold(self) -> None:
        """自定义保留比阈值 0.60。"""
        c = DeliveryCriteria(blind_test=BlindTestCriteria(min_ic_retention=0.60))
        stage = BlindTestStage(c)
        # 保留比 = 0.03/0.05 = 0.6, 恰好等于阈值 → 通过（>= 比较）
        result = stage.run({
            "train_metrics": {"ic": 0.05},
            "test_metrics": {"ic": 0.03},
        })
        assert result.passed is True
        # 保留比 = 0.02/0.05 = 0.4 < 0.60 → 拦截
        result2 = stage.run({
            "train_metrics": {"ic": 0.05},
            "test_metrics": {"ic": 0.02},
        })
        assert result2.passed is False
        assert "blind_test_ic_retention" in result2.fail_reasons

    def test_sign_consistency_disabled(self) -> None:
        """require_sign_consistency=False 时方向反转不拦截。"""
        c = DeliveryCriteria(
            blind_test=BlindTestCriteria(require_sign_consistency=False)
        )
        stage = BlindTestStage(c)
        # 方向反转但保留比足够 → 只报保留比通过，方向不拦
        result = stage.run({
            "train_metrics": {"ic": 0.05},
            "test_metrics": {"ic": -0.03},
        })
        assert result.passed is True
        assert result.fail_reasons == []


class TestBlindTestCriteriaSerialization:
    """确保 BlindTestCriteria 在 DeliveryCriteria.from_spec / to_spec_dict 中正确往返。"""

    def test_defaults_roundtrip(self) -> None:
        d = DeliveryCriteria.defaults()
        spec = d.to_spec_dict()
        assert "blind_test" in spec
        assert spec["blind_test"]["enabled"] is True
        assert spec["blind_test"]["min_ic_retention"] == 0.50
        assert spec["blind_test"]["require_sign_consistency"] is True

    def test_from_spec_partial(self) -> None:
        """只传部分字段，其余回落默认。"""
        d = DeliveryCriteria.from_spec({
            "blind_test": {"min_ic_retention": 0.65}
        })
        assert d.blind_test.min_ic_retention == 0.65
        assert d.blind_test.enabled is True
        assert d.blind_test.require_sign_consistency is True

    def test_from_spec_disabled(self) -> None:
        d = DeliveryCriteria.from_spec({
            "blind_test": {"enabled": False}
        })
        assert d.blind_test.enabled is False
