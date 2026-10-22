# -*- coding: utf-8 -*-
"""旋转策略库 registry 测试（Phase 1，spec §3.9）。"""

from __future__ import annotations

import pytest

from alphaagent.factor.mining.rotation.registry import (
    RotationRegistry,
    RotationRegistryError,
)


def _series(n: int = 100, fill: str = "A") -> list[str]:
    return [fill] * n


def _series_with_diff(n: int, diff_positions: list[int]) -> list[str]:
    out = ["A"] * n
    for p in diff_positions:
        out[p] = "B"
    return out


def _register(reg: RotationRegistry, sid: str, mode: str = "technical", series: list[str] | None = None) -> dict:  # noqa: ANN001
    return reg.register(
        strategy_id=sid,
        design_fp=f"fp:{sid}",
        compile_fp=f"c:{sid}",
        mode=mode,
        meta={"note": "test"},
        state_series=series,
    )


class TestRoundtrip:
    def test_register_get_list(self, tmp_path) -> None:
        reg = RotationRegistry(tmp_path)
        _register(reg, "rot_a", series=_series())
        assert reg.get("rot_a")["strategy_id"] == "rot_a"
        assert [e["strategy_id"] for e in reg.list()] == ["rot_a"]

    def test_list_filters_by_mode(self, tmp_path) -> None:
        reg = RotationRegistry(tmp_path)
        _register(reg, "rot_t", mode="technical")
        _register(reg, "rot_f", mode="fundamental")
        assert [e["strategy_id"] for e in reg.list(mode="technical")] == ["rot_t"]
        assert [e["strategy_id"] for e in reg.list(mode="fundamental")] == ["rot_f"]
        assert len(reg.list()) == 2

    def test_hash_for(self, tmp_path) -> None:
        reg = RotationRegistry(tmp_path)
        _register(reg, "rot_a")
        assert reg.hash_for("rot_a")["design"] == "fp:rot_a"
        assert reg.hash_for("rot_nope") is None


class TestHashConflict:
    def test_idempotent_same_fp_ok(self, tmp_path) -> None:
        reg = RotationRegistry(tmp_path)
        _register(reg, "rot_a")
        e2 = reg.register(
            strategy_id="rot_a", design_fp="fp:rot_a", compile_fp="c:rot_a", mode="technical"
        )
        assert e2["strategy_id"] == "rot_a"

    def test_same_id_different_fp_rejected(self, tmp_path) -> None:
        reg = RotationRegistry(tmp_path)
        _register(reg, "rot_a")
        with pytest.raises(RotationRegistryError) as ei:
            reg.register(
                strategy_id="rot_a", design_fp="DIFFERENT", compile_fp="c:rot_a", mode="technical"
            )
        assert "hash 冲突" in str(ei.value)


class TestNearDuplicate:
    def test_near_duplicate_rejected(self, tmp_path) -> None:
        reg = RotationRegistry(tmp_path)
        _register(reg, "rot_base", series=_series(100))
        dup = _series_with_diff(100, [3, 4, 7, 9])  # 96/100 一致 > 0.95
        with pytest.raises(RotationRegistryError) as ei:
            _register(reg, "rot_dup", series=dup)
        assert "近重复" in str(ei.value)
        assert "rot_base" in str(ei.value)

    def test_distinct_series_accepted(self, tmp_path) -> None:
        reg = RotationRegistry(tmp_path)
        _register(reg, "rot_base", series=_series(100))
        distinct = ["A" if i % 2 == 0 else "B" for i in range(100)]  # 50% 一致
        entry = _register(reg, "rot_distinct", series=distinct)
        assert entry["strategy_id"] == "rot_distinct"

    def test_allow_variant_of_registers_with_tag(self, tmp_path) -> None:
        reg = RotationRegistry(tmp_path)
        _register(reg, "rot_base", series=_series(100))
        dup = _series_with_diff(100, [3, 4, 7, 9])
        entry = reg.register(
            strategy_id="rot_v", design_fp="fp:rot_v", compile_fp="c:rot_v",
            mode="technical", state_series=dup, allow_variant_of=True,
        )
        assert entry["variant_of"] == "rot_base"

    def test_canonical_baseline_is_dedup_target(self, tmp_path) -> None:
        reg = RotationRegistry(tmp_path)
        reg.register_canonical_baseline(_series(100))
        dup = _series_with_diff(100, [1, 2, 3, 5])
        with pytest.raises(RotationRegistryError) as ei:
            reg.register(
                strategy_id="rot_x", design_fp="fp", compile_fp="c",
                mode="technical", state_series=dup,
            )
        assert "system_canonical" in str(ei.value)


class TestActive:
    def test_single_active_per_mode(self, tmp_path) -> None:
        reg = RotationRegistry(tmp_path)
        _register(reg, "rot_a", mode="technical")
        _register(reg, "rot_b", mode="technical")
        reg.set_active("rot_a")
        assert reg.active("technical") == "rot_a"
        reg.set_active("rot_b")
        assert reg.active("technical") == "rot_b"

    def test_active_modes_independent(self, tmp_path) -> None:
        reg = RotationRegistry(tmp_path)
        _register(reg, "rot_t", mode="technical")
        _register(reg, "rot_f", mode="fundamental")
        reg.set_active("rot_t")
        reg.set_active("rot_f")
        assert reg.active("technical") == "rot_t"
        assert reg.active("fundamental") == "rot_f"

    def test_set_active_unknown_id_raises(self, tmp_path) -> None:
        reg = RotationRegistry(tmp_path)
        with pytest.raises(RotationRegistryError):
            reg.set_active("rot_nope")


class TestUnregister:
    def test_unregister_removes_entry_and_active(self, tmp_path) -> None:
        reg = RotationRegistry(tmp_path)
        _register(reg, "rot_a")
        reg.set_active("rot_a")
        assert reg.unregister("rot_a") is True
        assert reg.get("rot_a") is None
        assert reg.active("technical") is None
        assert reg.unregister("rot_a") is False


class TestModeBinding:
    """OCR(medium,20261013)：mode 绑定契约缺口补测。"""

    def test_register_same_id_different_mode_rejected(self, tmp_path) -> None:
        reg = RotationRegistry(tmp_path)
        _register(reg, "rot_a", mode="technical")
        with pytest.raises(RotationRegistryError) as ei:
            reg.register(
                strategy_id="rot_a", design_fp="fp:rot_a", compile_fp="c:rot_a",
                mode="fundamental",
            )
        assert "mode 冲突" in str(ei.value)

    def test_set_active_with_mismatched_mode_rejected(self, tmp_path) -> None:
        reg = RotationRegistry(tmp_path)
        _register(reg, "rot_a", mode="technical")
        reg.set_active("rot_a")  # mode=None 解析自身 mode，正常
        assert reg.active("technical") == "rot_a"
        with pytest.raises(RotationRegistryError) as ei:
            reg.set_active("rot_a", mode="fundamental")
        assert "拒绝跨 mode" in str(ei.value)
