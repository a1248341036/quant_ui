# -*- coding: utf-8 -*-
"""rotation_policy / prioritize_method_types 配置中心三件套测试（P1）。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from alphaagent.factor.mining.research_spec import (
    DEFAULT_RESEARCH_SPEC,
    METHOD_TYPE_LABELS,
    build_run_research_spec,
    normalize_research_spec,
)


def _load_classify_report_labels() -> frozenset[str]:
    """从 scripts/classify_report_methods.py 直接加载 LABELS，避免依赖 scripts 为包。"""
    src = Path(__file__).resolve().parents[1] / "scripts" / "classify_report_methods.py"
    spec = importlib.util.spec_from_file_location("classify_report_methods", src)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return frozenset(module.LABELS)


class TestRotationPolicyDefaults:
    def test_default_keys_complete(self) -> None:
        rp = DEFAULT_RESEARCH_SPEC["rotation_policy"]
        assert rp["enabled"] is False
        assert rp["industry_field"] == "industry_sw_l1"
        assert rp["min_industry_cover"] == 15
        assert rp["min_industry_days"] == 20
        assert rp["report_industry_metrics"] is True

    def test_state_layer_defaults(self) -> None:
        rp = DEFAULT_RESEARCH_SPEC["rotation_policy"]
        assert rp["dims"] == ["size", "value_growth"]
        assert rp["style_lookback_days"] == 120
        assert rp["min_state_days"] == 20
        assert rp["boost_cap"] == 1.25
        assert rp["boost_source"] == "prior"
        assert rp["max_daily_turnover"] == 0.20
        assert rp["strategy_dir"] == "artifacts/alphaagent/rotation_strategies"
        assert rp["max_test_submissions"] == 2
        assert rp["max_strategy_submissions"] == 10
        assert rp["cost_bps"] == 20
        assert rp["report_state_metrics"] is True

    def test_normalize_empty_rotation_keeps_defaults(self) -> None:
        spec = normalize_research_spec({"rotation_policy": {}})
        rp = spec["rotation_policy"]
        assert rp["enabled"] is False
        assert rp["industry_field"] == "industry_sw_l1"
        assert rp["min_industry_cover"] == 15
        assert rp["min_industry_days"] == 20
        assert rp["report_industry_metrics"] is True
        assert rp["style_lookback_days"] == 120
        assert rp["min_state_days"] == 20
        assert rp["boost_source"] == "prior"
        assert rp["report_state_metrics"] is True

    def test_prioritize_defaults_empty(self) -> None:
        spec = normalize_research_spec({})
        assert spec["report_policy"]["prioritize_method_types"] == []


class TestRotationPolicyOverride:
    def test_enabled_override_roundtrip(self) -> None:
        spec = normalize_research_spec({
            "rotation_policy": {
                "enabled": True,
                "industry_field": "industry_sw_custom",
                "min_industry_cover": 30,
                "min_industry_days": 25,
                "report_industry_metrics": False,
            },
        })
        rp = spec["rotation_policy"]
        assert rp["enabled"] is True
        assert rp["industry_field"] == "industry_sw_custom"
        assert rp["min_industry_cover"] == 30
        assert rp["min_industry_days"] == 25
        assert rp["report_industry_metrics"] is False

    def test_state_layer_override_roundtrip(self) -> None:
        spec = normalize_research_spec({
            "rotation_policy": {
                "dims": ["size", "momentum"],
                "style_lookback_days": 90,
                "min_state_days": 30,
                "boost_cap": 1.5,
                "boost_source": "stat",
                "max_daily_turnover": 0.3,
                "strategy_dir": "artifacts/x/rotation",
                "max_test_submissions": 3,
                "max_strategy_submissions": 5,
                "cost_bps": 30,
                "report_state_metrics": False,
            },
        })
        rp = spec["rotation_policy"]
        assert rp["dims"] == ["size", "momentum"]
        assert rp["style_lookback_days"] == 90
        assert rp["min_state_days"] == 30
        assert rp["boost_cap"] == 1.5
        assert rp["boost_source"] == "stat"
        assert rp["max_daily_turnover"] == 0.3
        assert rp["strategy_dir"] == "artifacts/x/rotation"
        assert rp["max_test_submissions"] == 3
        assert rp["max_strategy_submissions"] == 5
        assert rp["cost_bps"] == 30
        assert rp["report_state_metrics"] is False

    def test_build_run_carries_rotation_enabled(self) -> None:
        run = build_run_research_spec({"rotation_policy": {"enabled": True}})
        assert run["rotation_policy"]["enabled"] is True


class TestRotationStatePolicyValidation:
    @pytest.mark.parametrize("dims", [["not_a_dim"], ["size", "bogus"], "size"])
    def test_unknown_dim_or_non_list_rejected(self, dims) -> None:  # noqa: ANN001
        with pytest.raises(ValueError):
            normalize_research_spec({"rotation_policy": {"dims": dims}})

    @pytest.mark.parametrize("v", [10, 600])
    def test_style_lookback_out_of_range_rejected(self, v: int) -> None:
        with pytest.raises(ValueError):
            normalize_research_spec({"rotation_policy": {"style_lookback_days": v}})

    @pytest.mark.parametrize("v", [3, 600])
    def test_min_state_days_out_of_range_rejected(self, v: int) -> None:
        with pytest.raises(ValueError):
            normalize_research_spec({"rotation_policy": {"min_state_days": v}})

    @pytest.mark.parametrize("v", [0.9, 2.5])
    def test_boost_cap_out_of_range_rejected(self, v: float) -> None:
        with pytest.raises(ValueError):
            normalize_research_spec({"rotation_policy": {"boost_cap": v}})

    @pytest.mark.parametrize("v", ["random", "STAT", 1])
    def test_boost_source_invalid_rejected(self, v) -> None:  # noqa: ANN001
        with pytest.raises(ValueError):
            normalize_research_spec({"rotation_policy": {"boost_source": v}})

    @pytest.mark.parametrize("v", [0.01, 0.6])
    def test_max_daily_turnover_out_of_range_rejected(self, v: float) -> None:
        with pytest.raises(ValueError):
            normalize_research_spec({"rotation_policy": {"max_daily_turnover": v}})

    @pytest.mark.parametrize("v", [0, 11])
    def test_max_test_submissions_out_of_range_rejected(self, v: int) -> None:
        with pytest.raises(ValueError):
            normalize_research_spec({"rotation_policy": {"max_test_submissions": v}})

    @pytest.mark.parametrize("v", [0, 101])
    def test_max_strategy_submissions_out_of_range_rejected(self, v: int) -> None:
        with pytest.raises(ValueError):
            normalize_research_spec({"rotation_policy": {"max_strategy_submissions": v}})

    @pytest.mark.parametrize("v", [-1, 501])
    def test_cost_bps_out_of_range_rejected(self, v: int) -> None:
        with pytest.raises(ValueError):
            normalize_research_spec({"rotation_policy": {"cost_bps": v}})

    @pytest.mark.parametrize("key", [
        "style_lookback_days", "min_state_days", "max_test_submissions",
        "max_strategy_submissions", "cost_bps",
    ])
    def test_fractional_integer_key_rejected_no_truncation(self, key: str) -> None:
        # OCR(low,20261013)：整数语义键（天/次/bp）小数入参须 fail-fast，
        # 禁止 int() 静默截断——cost_bps=0.5 若变成 0 会把"0.5bp 成本"
        # 悄悄改成"零成本回测"，与配置中心三件套"越界被拒"纪律相悖。
        with pytest.raises(ValueError):
            normalize_research_spec({"rotation_policy": {key: 0.5}})

    def test_empty_dims_rejected(self) -> None:
        with pytest.raises(ValueError):
            normalize_research_spec({"rotation_policy": {"dims": []}})

    def test_dims_deduplicated_preserving_order(self) -> None:
        spec = normalize_research_spec({"rotation_policy": {"dims": ["size", "size", "momentum"]}})
        assert spec["rotation_policy"]["dims"] == ["size", "momentum"]

    @pytest.mark.parametrize("v", [1, "yes", None])
    def test_report_state_metrics_non_bool_rejected(self, v) -> None:  # noqa: ANN001
        with pytest.raises(ValueError):
            normalize_research_spec({"rotation_policy": {"report_state_metrics": v}})


class TestRotationPolicyValidation:
    @pytest.mark.parametrize("cover", [1, 4, 201, 500])
    def test_min_industry_cover_out_of_range_rejected(self, cover: int) -> None:
        with pytest.raises(ValueError):
            normalize_research_spec({"rotation_policy": {"min_industry_cover": cover}})

    @pytest.mark.parametrize("days", [0, 4, 501])
    def test_min_industry_days_out_of_range_rejected(self, days: int) -> None:
        with pytest.raises(ValueError):
            normalize_research_spec({"rotation_policy": {"min_industry_days": days}})

    @pytest.mark.parametrize("enabled", ["yes", 1, None])
    def test_enabled_non_bool_rejected(self, enabled) -> None:  # noqa: ANN001
        with pytest.raises(ValueError):
            normalize_research_spec({"rotation_policy": {"enabled": enabled}})


class TestPrioritizeMethodTypes:
    def test_whitelist_accepted(self) -> None:
        spec = normalize_research_spec({
            "report_policy": {"prioritize_method_types": ["timing_rotation", "portfolio_combo"]},
        })
        assert spec["report_policy"]["prioritize_method_types"] == ["timing_rotation", "portfolio_combo"]

    def test_unknown_method_type_rejected(self) -> None:
        with pytest.raises(ValueError):
            normalize_research_spec({"report_policy": {"prioritize_method_types": ["not_a_label"]}})

    def test_non_string_list_rejected(self) -> None:
        with pytest.raises(ValueError):
            normalize_research_spec({"report_policy": {"prioritize_method_types": "timing_rotation"}})


class TestPrioritizeWhitelistSingleSource:
    # OCR (P1) low：research_spec 白名单是 classify_report_methods.LABELS 的静态拷贝，
    # 防止"新增/改名 method_type → 白名单忘记同步 → 合法优先级被误拒"或反向漂移。
    def test_whitelist_equals_classify_labels(self) -> None:
        assert METHOD_TYPE_LABELS == _load_classify_report_labels()

    def test_every_classify_label_accepted(self) -> None:
        labels = _load_classify_report_labels()
        spec = normalize_research_spec({
            "report_policy": {"prioritize_method_types": sorted(labels)},
        })
        assert set(spec["report_policy"]["prioritize_method_types"]) == labels
