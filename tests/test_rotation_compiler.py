# -*- coding: utf-8 -*-
"""轮动策略确定性编译器测试（Phase 1，spec §3.2 / §6）。"""

from __future__ import annotations

import pytest

from core.rotation.compiler import (
    compile_design,
    default_blind_start,
    probe_carrier_health,
    state_enum_names,
    worst_case_turnover,
)
from core.rotation.style_regime import canonical_design


def base_design(**overrides) -> dict:
    d = canonical_design(mode="technical")
    for k, v in overrides.items():
        d[k] = v
    return d


class TestDeterministicHash:
    def test_same_design_same_strategy_id(self) -> None:
        a = compile_design(base_design(), mode="technical")
        b = compile_design(base_design(), mode="technical")
        assert a.strategy_id == b.strategy_id
        assert a.design_fp == b.design_fp
        assert a.compile_fp == b.compile_fp

    def test_change_design_changes_id(self) -> None:
        a = compile_design(base_design(), mode="technical")
        d2 = base_design()
        d2["state_dimensions"][0]["window_days"] = 125
        b = compile_design(d2, mode="technical")
        assert a.strategy_id != b.strategy_id

    def test_thesis_not_in_fingerprint(self) -> None:
        # thesis/evidence 是评审元数据，不进 canonical => 改措辞不换 id
        a = compile_design(base_design(), mode="technical")
        d2 = base_design()
        d2["thesis"] = "完全不同的经济叙述"
        b = compile_design(d2, mode="technical")
        assert a.strategy_id == b.strategy_id


class TestModeFamilyValidation:
    def test_technical_rejects_fundamental_family(self) -> None:
        d = base_design()
        d["mapping"] = {"SMALL_VALUE": {"quality": 1.1}, "SMALL_GROWTH": {}, "LARGE_VALUE": {}, "LARGE_GROWTH": {}}
        with pytest.raises(ValueError) as ei:
            compile_design(d, mode="technical")
        assert "quality" in str(ei.value)
        assert "可用因子族" in str(ei.value)
        assert "volatility" in str(ei.value) or "momentum" in str(ei.value)

    def test_fundamental_mode_accepts_value_growth(self) -> None:
        d = canonical_design()
        d["mapping"] = {
            "SMALL_VALUE": {"value": 1.1},
            "SMALL_GROWTH": {"growth": 1.1},
            "LARGE_VALUE": {"quality": 1.1},
            "LARGE_GROWTH": {"revision": 1.1},
        }
        c = compile_design(d, mode="fundamental")
        assert c.strategy_id.startswith("rot_")
        assert c.families_used == frozenset({"value", "growth", "quality", "revision"})


class TestStateEnum:
    def test_mapping_key_outside_enum_rejected(self) -> None:
        d = base_design()
        d["mapping"] = {"NOT_A_STATE": {"momentum": 1.1}, "SMALL_VALUE": {}, "SMALL_GROWTH": {}, "LARGE_VALUE": {}, "LARGE_GROWTH": {}}
        with pytest.raises(ValueError) as ei:
            compile_design(d, mode="technical")
        assert "状态枚举全集" in str(ei.value)
        assert "SMALL_VALUE" in str(ei.value)

    def test_partial_mapping_ok(self) -> None:
        d = base_design()
        d["mapping"] = {"SMALL_VALUE": {"reversal": 1.1}}
        c = compile_design(d, mode="technical")
        assert c.state_enums == ("LARGE_GROWTH", "LARGE_VALUE", "SMALL_GROWTH", "SMALL_VALUE")

    def test_unknown_dim_carrier_pair_rejected_with_legal_pairs(self) -> None:
        d = base_design()
        d["state_dimensions"][0] = {"dim": "momentum", "carrier": "margin_chg", "window_days": 60,
                                    "threshold_mode": "sign", "params": {}}
        with pytest.raises(ValueError) as ei:
            compile_design(d, mode="technical")
        assert "可编译组合" in str(ei.value)


class TestWorstCaseTurnover:
    def test_formula_uses_upper_frequency_bound_and_pair_delta(self) -> None:
        enums = ("A", "B")
        mapping = {"A": {"momentum": 1.25, "reversal": 0.8}, "B": {"momentum": 0.8, "reversal": 1.25}}
        # 单次切换最大权重变化 = |1.25-0.8|+|0.8-1.25| = 0.9；频率上界 = 1/2
        w = worst_case_turnover(min_state_duration_days=2, mapping=mapping, state_enums=enums)
        assert abs(w - 0.45) < 1e-9

    def test_lower_bound_zero_never_triggers(self) -> None:
        mapping = {"A": {"reversal": 1.1}, "B": {"reversal": 1.1}}
        w = worst_case_turnover(min_state_duration_days=1000, mapping=mapping, state_enums=("A", "B"))
        assert w == 0.0

    def test_pre_gate_rejects_high_turnover_design(self) -> None:
        d = base_design()
        d["guardrails"] = {"max_daily_turnover": 0.15, "min_state_duration_days": 2, "no_lookahead": True}
        with pytest.raises(ValueError) as ei:
            compile_design(d, mode="technical")
        assert "最坏换手" in str(ei.value)

    def test_pre_gate_passes_with_long_duration(self) -> None:
        d = base_design()
        d["guardrails"] = {"max_daily_turnover": 0.15, "min_state_duration_days": 10, "no_lookahead": True}
        c = compile_design(d, mode="technical")
        assert c.worst_case_turnover <= 0.15


class TestCarrierHealth:
    def test_health_unreported_passes_as_pure_design(self) -> None:
        c = compile_design(base_design(), mode="technical")
        assert c.carrier_health_reported is False

    def test_missing_carrier_health_entry_rejected(self) -> None:
        # 体检 dict 缺少设计所引用 carrier => 拒（只提供别的 carrier 即触发缺项）
        with pytest.raises(ValueError) as ei:
            compile_design(
                base_design(), mode="technical",
                carrier_health={"margin_chg": {"available": True}},
            )
        assert "无体检数据" in str(ei.value)

    def test_unavailable_carrier_rejected(self) -> None:
        d = base_design()
        with pytest.raises(ValueError) as ei:
            compile_design(
                d, mode="technical",
                carrier_health={"style_factor": {"available": False, "reason": "数据源断供"}},
            )
        assert "不可用" in str(ei.value) and "style_factor" in str(ei.value)

    def test_no_blind_coverage_rejected(self) -> None:
        with pytest.raises(ValueError) as ei:
            compile_design(
                base_design(), mode="technical",
                carrier_health={"style_factor": {"available": True, "blind_coverage": False}},
            )
        assert "盲测段" in str(ei.value)

    def test_probe_carrier_health_detects_missing_and_stale(self) -> None:
        import numpy as np
        import pandas as pd

        idx = pd.date_range("2022-01-03", periods=250, freq="B")
        signals = pd.DataFrame({"size_proxy": np.arange(250, dtype=float)}, index=idx)
        # 250 个工作日止于 2022-12-16；盲测起点取段内日期验证有覆盖
        health = probe_carrier_health(signals, "2022-12-05")
        assert health["style_factor"]["available"] is True
        assert health["style_factor"]["blind_coverage"] is True
        # northbound 系列列缺失 => 不可用
        assert health["northbound_chg"]["available"] is False
        # 盲测段（2025 起）无覆盖
        health2 = probe_carrier_health(signals, "2025-01-01")
        assert health2["style_factor"]["blind_coverage"] is False


class TestMisc:
    def test_default_blind_start_derived_from_window_config(self) -> None:
        assert default_blind_start() == "2025-01-01"

    def test_state_enum_names_deterministic(self) -> None:
        c = compile_design(base_design(), mode="technical")
        assert c.state_enums == ("LARGE_GROWTH", "LARGE_VALUE", "SMALL_GROWTH", "SMALL_VALUE")
        assert state_enum_names(c.dimensions) == c.state_enums

    def test_to_dict_roundtrip_fingerprints(self) -> None:
        c = compile_design(base_design(), mode="technical")
        d = c.to_dict()
        assert d["strategy_id"] == c.strategy_id
        assert d["fingerprints"]["design"] == c.design_fp
