# -*- coding: utf-8 -*-
"""label 与调仓频率**强制一致**（2026-10-04 用户定调：「调仓频率是多少，label 就应该多少」）。

规则：`engine_gate.freq` 的持有期 == `recommended_label_col` 的持有期
（daily↔1d、weekly↔5d、monthly↔20d）。覆盖三件事：
1. 注册表里每个档位都是对齐的（不变量：防止后人再加解耦档位）；
2. 裸 spec / 未显式声明 freq 时由 label 派生（自洽）；
3. 显式矛盾 → fail-closed（构建期与 API 入口共用 `ensure_label_freq_consistency`）。
"""
from __future__ import annotations

import pytest

from alphaagent.factor.mining.research_spec import (
    DEFAULT_RESEARCH_SPEC,
    default_research_spec,
    ensure_label_freq_consistency,
    label_hold_days,
    normalize_research_spec,
)
from core.research_modes import RESEARCH_MODES

_FREQ_HOLD = {"daily": 1, "weekly": 5, "monthly": 20}


def _freq_of(spec: dict) -> str:
    return str(spec["delivery_policy"]["production"]["engine_gate"]["freq"]).lower()


def test_label_hold_days_parsing() -> None:
    assert label_hold_days("label_1d_open_to_open") == 1
    assert label_hold_days("label_5d_close_to_close") == 5
    assert label_hold_days("label_20d_close_to_close") == 20
    assert label_hold_days("") == 1          # 无数字视为 1d（与 engine/submit 侧一致）
    assert label_hold_days(None) == 1


def test_registry_modes_are_all_aligned() -> None:
    """不变量：注册表里每个档位的 label 持有期 == freq 持有期。"""
    bad: list[str] = []
    for mode_id, mode_spec in RESEARCH_MODES.items():
        freq = str((mode_spec.engine_gate_overrides or {}).get("freq") or "").lower()
        hold = label_hold_days(mode_spec.recommended_label_col)
        if _FREQ_HOLD.get(freq) != hold:
            bad.append(f"{mode_id}: label={mode_spec.recommended_label_col}({hold}d) vs freq={freq}")
    assert not bad, "以下档位 label 与调仓频率不一致（违反 2026-10-04 强制一致定调）：" + "; ".join(bad)


@pytest.mark.parametrize(
    ("mode", "freq"),
    [
        ("technical", "daily"),
        ("report", "daily"),
        ("technical_daily", "daily"),
        ("technical_weekly", "weekly"),
        ("technical_monthly", "monthly"),
        ("fundamental", "monthly"),
    ],
)
def test_default_specs_are_aligned(mode: str, freq: str) -> None:
    spec = normalize_research_spec(default_research_spec(mode))
    assert _freq_of(spec) == freq
    assert spec["delivery_policy"]["production"]["engine_gate"]["allowed_freqs"] == [freq] or freq in (
        spec["delivery_policy"]["production"]["engine_gate"]["allowed_freqs"]
    )
    ensure_label_freq_consistency(spec)  # 不抛异常


def test_bare_spec_derives_freq_from_label() -> None:
    """裸 spec（无模式、无显式 freq）应派生出与 label 对应的 daily。"""
    spec = normalize_research_spec({})
    assert label_hold_days(spec["recommended_label_col"]) == 1
    assert _freq_of(spec) == "daily"


def test_normalize_coerces_mismatched_freq_to_label() -> None:
    """归一化阶段以 label 为准收口：1d label + 显式 weekly → daily 且白名单收成单值。"""
    spec = normalize_research_spec(
        {
            "recommended_label_col": "label_1d_open_to_open",
            "delivery_policy": {"production": {"engine_gate": {"freq": "weekly", "allowed_freqs": ["daily", "weekly"]}}},
        }
    )
    eg = spec["delivery_policy"]["production"]["engine_gate"]
    assert eg["freq"] == "daily"
    assert eg["allowed_freqs"] == ["daily"]


def test_ensure_raises_on_explicit_mismatch() -> None:
    """显式矛盾必须报错（API 入口在用户覆盖 freq 后调它，防止覆盖绕过校验）。"""
    spec = default_research_spec("technical")
    spec["delivery_policy"]["production"]["engine_gate"]["freq"] = "weekly"
    with pytest.raises(ValueError) as exc:
        ensure_label_freq_consistency(spec)
    assert "label_freq_consistency.mismatch" in str(exc.value)
    assert "label_1d_open_to_open" in str(exc.value)


def test_enforce_false_is_legacy_escape_hatch() -> None:
    spec = default_research_spec("technical")
    spec["delivery_policy"]["production"]["engine_gate"]["freq"] = "weekly"
    spec["label_freq_consistency"] = {"enforce": False}
    ensure_label_freq_consistency(spec)  # 不抛异常（迁移/诊断期临时放行）


def test_default_config_center_declares_rule() -> None:
    lfc = DEFAULT_RESEARCH_SPEC["label_freq_consistency"]
    assert lfc["enforce"] is True
    assert lfc["freq_hold_days"] == {"daily": 1, "weekly": 5, "monthly": 20}


def test_unknown_freq_is_rejected() -> None:
    spec = default_research_spec("technical")
    spec["delivery_policy"]["production"]["engine_gate"]["freq"] = "hourly"
    spec["delivery_policy"]["production"]["engine_gate"]["allowed_freqs"] = ["hourly"]
    with pytest.raises(ValueError) as exc:
        ensure_label_freq_consistency(spec)
    assert "freq_unknown" in str(exc.value)


def test_bad_freq_hold_map_is_rejected() -> None:
    with pytest.raises(ValueError):
        normalize_research_spec({"label_freq_consistency": {"freq_hold_days": {"daily": 0}}})


def test_start_run_rejects_mismatch_before_spawning() -> None:
    """所有 API/脚本/monitor 启动路径的唯一收口：start_run 在 spawn 之前拒绝不一致组合。"""
    from backend import alphaagent_service as svc

    spec = default_research_spec("technical")  # label_1d + daily
    spec["delivery_policy"]["production"]["engine_gate"]["freq"] = "weekly"  # 人为制造不一致
    with pytest.raises(svc.RunAdmissionError) as exc:
        svc.start_run({"label_col": "label_1d_open_to_open", "research_spec": spec})
    assert "label_freq_consistency.mismatch" in str(exc.value)

