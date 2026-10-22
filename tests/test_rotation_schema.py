# -*- coding: utf-8 -*-
"""rotation_strategy schema 结构校验测试（Phase 1，spec §6）。"""

from __future__ import annotations

import pytest

from alphaagent.factor.mining.rotation.schema import (
    BOOST_MAX,
    BOOST_MIN,
    MAX_DAILY_TURNOVER_MAX,
    MAX_DAILY_TURNOVER_MIN,
    WINDOW_DAYS_MAX,
    WINDOW_DAYS_MIN,
    legal_values,
    validate_design,
)


def design(**overrides) -> dict:
    base = {
        "version": 1,
        "thesis": "测试：小市值/价值阶段偏好反转",
        "state_dimensions": [
            {"dim": "size", "carrier": "style_factor", "window_days": 120,
             "threshold_mode": "quantile", "params": {"q": 0.5}},
        ],
        "quantile_benchmark": "expanding",
        "mapping": {"SMALL": {"reversal": 1.1, "liquidity": 1.1}},
        "allocation": None,
        "guardrails": {"max_daily_turnover": 0.2, "min_state_duration_days": 5, "no_lookahead": True},
        "evidence": "2021-2023 风格回归 beta 季度滚动相关性（测试）",
    }
    for k, v in overrides.items():
        if v is None and k in base:
            del base[k]
        else:
            base[k] = v
    return base


class TestValidDesign:
    def test_minimal_valid(self) -> None:
        rep = validate_design(design())
        assert rep.ok, rep.message()
        assert rep.errors == []

    def test_two_dimension_valid(self) -> None:
        d = design()
        d["state_dimensions"].append(
            {"dim": "value_growth", "carrier": "style_factor", "window_days": 60,
             "threshold_mode": "sign", "params": {}}
        )
        assert validate_design(d).ok


class TestRequiredFields:
    def test_missing_thesis_rejected(self) -> None:
        rep = validate_design(design(thesis="  "))
        assert not rep.ok
        assert any("thesis" in e for e in rep.errors)

    def test_missing_evidence_rejected(self) -> None:
        rep = validate_design(design(evidence=""))
        assert not rep.ok
        assert any("evidence" in e for e in rep.errors)

    def test_empty_state_dimensions_rejected(self) -> None:
        rep = validate_design(design(state_dimensions=[]))
        assert not rep.ok
        assert any("state_dimensions" in e for e in rep.errors)

    def test_missing_mapping_rejected(self) -> None:
        rep = validate_design(design(mapping={}))
        assert not rep.ok
        assert any("mapping" in e for e in rep.errors)


class TestBounds:
    @pytest.mark.parametrize("wd", [WINDOW_DAYS_MIN - 1, WINDOW_DAYS_MAX + 1, 0])
    def test_window_days_out_of_range(self, wd: int) -> None:
        d = design()
        d["state_dimensions"][0]["window_days"] = wd
        assert not validate_design(d).ok

    @pytest.mark.parametrize("b", [BOOST_MIN - 0.01, BOOST_MAX + 0.01, -0.5])
    def test_boost_out_of_range(self, b: float) -> None:
        d = design()
        d["mapping"] = {"SMALL": {"reversal": b}}
        assert not validate_design(d).ok

    @pytest.mark.parametrize("t", [MAX_DAILY_TURNOVER_MIN - 0.01, MAX_DAILY_TURNOVER_MAX + 0.01])
    def test_max_daily_turnover_out_of_range(self, t: float) -> None:
        d = design()
        d["guardrails"]["max_daily_turnover"] = t
        assert not validate_design(d).ok

    def test_min_state_duration_below_floor(self) -> None:
        d = design()
        d["guardrails"]["min_state_duration_days"] = 1
        assert not validate_design(d).ok


class TestWhitelist:
    def test_unknown_dim_rejected_with_legal_set(self) -> None:
        d = design()
        d["state_dimensions"][0]["dim"] = "quantile"  # 非状态维度
        rep = validate_design(d)
        assert not rep.ok
        text = rep.message()
        assert "dim" in text and "size" in text and "market_regime" in text

    def test_unknown_carrier_rejected_with_legal_set(self) -> None:
        d = design()
        d["state_dimensions"][0]["carrier"] = "analyst_ratio"
        rep = validate_design(d)
        assert not rep.ok
        assert "carrier" in rep.message() and "northbound_chg" in rep.message()

    def test_duplicate_dim_rejected(self) -> None:
        d = design()
        d["state_dimensions"].append(
            {"dim": "size", "carrier": "index_spread", "window_days": 60,
             "threshold_mode": "sign", "params": {}}
        )
        rep = validate_design(d)
        assert not rep.ok
        assert any("重复定义" in e for e in rep.errors)


class TestThresholdMode:
    def test_quantile_without_fallback_benchmark_rejected(self) -> None:
        d = design()
        del d["quantile_benchmark"]
        rep = validate_design(d)
        assert not rep.ok
        assert any("quantile_benchmark" in e for e in rep.errors)

    def test_level_on_non_stationary_carrier_rejected(self) -> None:
        d = design()
        d["state_dimensions"][0]["threshold_mode"] = "level"
        d["state_dimensions"][0]["params"] = {"threshold": 0.0}
        # style_factor 是水平量：level 模式必须拒绝（只允许 margin_chg/northbound_chg）
        rep = validate_design(d)
        assert not rep.ok
        assert any("level" in e for e in rep.errors)
        assert any("margin_chg" in e for e in rep.errors)

    def test_level_on_margin_chg_accepted(self) -> None:
        d = design()
        d["state_dimensions"][0] = {
            "dim": "market_regime", "carrier": "margin_chg", "window_days": 60,
            "threshold_mode": "level", "params": {"threshold": 0.0},
        }
        # mapping 键需匹配 market_regime 状态枚举；此处只验 schema 结构
        d["mapping"] = {"RISK_ON": {"momentum": 1.1}}
        assert validate_design(d).ok


class TestVersion:
    def test_future_version_rejected_with_migration_hint(self) -> None:
        rep = validate_design(design(version=2))
        assert not rep.ok
        assert any("迁移" in e for e in rep.errors)

    def test_old_version_still_compilable_shape(self) -> None:
        # version=1 是当前版本；缺省即 1
        assert validate_design(design()).ok


class TestMetaFields:
    def test_allocation_value_is_warning_only(self) -> None:
        rep = validate_design(design(allocation={"industry": ["银行"]}))
        assert rep.ok  # P3 预留：警告不阻断
        assert any("P3" in w for w in rep.warnings)

    def test_no_lookahead_false_rejected(self) -> None:
        d = design()
        d["guardrails"]["no_lookahead"] = False
        assert not validate_design(d).ok

    def test_legal_values_export_complete(self) -> None:
        legal = legal_values()
        assert {"size", "value_growth", "momentum", "market_regime"} == legal["dim"]
        assert {"sign", "quantile", "level"} == legal["threshold_mode"]


class TestUntrustedInputGuard:
    """OCR(high,20261013)：LLM 交付不可信 JSON，白名单判断须兼容不可哈希类型。"""

    def test_unhashable_dim_returns_report_not_raise(self) -> None:
        d = design()
        d["state_dimensions"][0]["dim"] = ["size"]
        rep = validate_design(d)
        assert not rep.ok
        assert any("dim 必须" in e for e in rep.errors)

    def test_unhashable_carrier_returns_report_not_raise(self) -> None:
        d = design()
        d["state_dimensions"][0]["carrier"] = {"x": 1}
        rep = validate_design(d)
        assert not rep.ok
        assert any("carrier 必须" in e for e in rep.errors)

    def test_unhashable_threshold_mode_returns_report_not_raise(self) -> None:
        d = design()
        d["state_dimensions"][0]["threshold_mode"] = ["quantile"]
        rep = validate_design(d)
        assert not rep.ok
        assert any("threshold_mode 必须" in e for e in rep.errors)

    def test_unhashable_quantile_benchmark_returns_report_not_raise(self) -> None:
        d = design()
        d["quantile_benchmark"] = ["expanding"]
        rep = validate_design(d)
        assert not rep.ok
        assert any("quantile_benchmark" in e for e in rep.errors)

    def test_non_integer_version_rejected(self) -> None:
        rep = validate_design(design(version=1.5))
        assert not rep.ok
        assert any("version" in e for e in rep.errors)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_version_returns_report_not_raise(self, bad: float) -> None:
        # json.loads 默认接受 NaN/Infinity 字面量：非有限数必须进报错集合而非抛异常，
        # 否则"收集 errors 不抛"契约被击穿，LLM 拿不到合法全集无法自纠。
        rep = validate_design(design(version=bad))
        assert not rep.ok
        assert any("version" in e for e in rep.errors)

    @pytest.mark.parametrize(
        "dim_cfg",
        [
            {"dim": "size", "carrier": "style_factor", "window_days": 120,
             "threshold_mode": "quantile", "params": {"q": float("nan")}},
            {"dim": "market_regime", "carrier": "margin_chg", "window_days": 60,
             "threshold_mode": "level", "params": {"threshold": float("inf")}},
        ],
    )
    def test_non_finite_numeric_params_returns_report_not_raise(self, dim_cfg) -> None:  # noqa: ANN001
        d = design()
        d["state_dimensions"][0] = dim_cfg
        rep = validate_design(d)
        assert not rep.ok

    def test_non_number_level_threshold_rejected(self) -> None:
        d = design()
        d["state_dimensions"][0] = {
            "dim": "market_regime", "carrier": "margin_chg", "window_days": 60,
            "threshold_mode": "level", "params": {"threshold": "abc"},
        }
        rep = validate_design(d)
        assert not rep.ok
        assert any("threshold" in e for e in rep.errors)
