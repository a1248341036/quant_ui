# -*- coding: utf-8 -*-
"""rotation_policy / prioritize_method_types 配置中心三件套测试（P1）。"""

from __future__ import annotations

import pytest

from alphaagent.factor.mining.research_spec import (
    DEFAULT_RESEARCH_SPEC,
    build_run_research_spec,
    normalize_research_spec,
)


class TestRotationPolicyDefaults:
    def test_default_keys_complete(self) -> None:
        rp = DEFAULT_RESEARCH_SPEC["rotation_policy"]
        assert rp["enabled"] is False
        assert rp["industry_field"] == "industry_sw_l1"
        assert rp["min_industry_cover"] == 15
        assert rp["min_industry_days"] == 20
        assert rp["report_industry_metrics"] is True

    def test_normalize_empty_rotation_keeps_defaults(self) -> None:
        spec = normalize_research_spec({"rotation_policy": {}})
        rp = spec["rotation_policy"]
        assert rp["enabled"] is False
        assert rp["industry_field"] == "industry_sw_l1"
        assert rp["min_industry_cover"] == 15
        assert rp["min_industry_days"] == 20
        assert rp["report_industry_metrics"] is True

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

    def test_build_run_carries_rotation_enabled(self) -> None:
        run = build_run_research_spec({"rotation_policy": {"enabled": True}})
        assert run["rotation_policy"]["enabled"] is True


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
