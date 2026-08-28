"""研究规范门槛持久化（每模式 JSON 覆盖文件）测试。

覆盖：load/save/reset、effective vs 纯默认、运行口径合并、增量 diff、
normalize/resolve_profiles 的默认值对齐单一来源。
"""

from __future__ import annotations

import json

import pytest

import alphaagent.factor.mining.research_spec as rs


@pytest.fixture(autouse=True)
def isolated_specs_dir(tmp_path, monkeypatch):
    """每个测试独立存储目录，避免污染 artifacts/alphaagent/research_specs。"""
    monkeypatch.setattr(rs, "RESEARCH_SPECS_DIR", tmp_path)
    return tmp_path


class TestPersistence:
    def test_no_overrides_returns_empty(self) -> None:
        assert rs.load_saved_overrides("technical") == {}

    def test_save_load_roundtrip(self) -> None:
        rs.save_research_spec_overrides("fundamental", {"evaluation_policy": {"min_train_abs_ic": 0.01}})
        loaded = rs.load_saved_overrides("fundamental")
        assert loaded["evaluation_policy"]["min_train_abs_ic"] == 0.01

    def test_save_writes_json_file(self) -> None:
        rs.save_research_spec_overrides("technical", {"review_policy": {"minimum_novelty": "medium"}})
        raw = json.loads((rs.RESEARCH_SPECS_DIR / "technical.json").read_text(encoding="utf-8"))
        assert raw["review_policy"]["minimum_novelty"] == "medium"

    def test_reset_removes_file(self) -> None:
        rs.save_research_spec_overrides("technical", {"evaluation_policy": {"min_train_abs_ic": 0.01}})
        assert rs.reset_research_spec_overrides("technical") is True
        assert not (rs.RESEARCH_SPECS_DIR / "technical.json").exists()
        assert rs.load_saved_overrides("technical") == {}

    def test_reset_without_file_returns_false(self) -> None:
        assert rs.reset_research_spec_overrides("fundamental") is False

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError):
            rs.save_research_spec_overrides("macro", {})


class TestEffectiveSpec:
    def test_effective_defaults_when_no_overrides(self) -> None:
        eff = rs.effective_research_spec("technical")
        assert eff["evaluation_policy"]["min_train_abs_ic"] == 0.02

    def test_effective_applies_saved_overrides(self) -> None:
        rs.save_research_spec_overrides("fundamental", {"evaluation_policy": {"min_train_abs_ic": 0.01}})
        eff = rs.effective_research_spec("fundamental")
        assert eff["evaluation_policy"]["min_train_abs_ic"] == 0.01
        # 未改的键仍跟随注册表默认
        assert eff["delivery_policy"]["candidate"]["min_icir"] == 0.20

    def test_default_research_spec_stays_pure(self) -> None:
        """纯默认函数不受保存覆盖影响（测试/调用方语义稳定）。"""
        rs.save_research_spec_overrides("fundamental", {"evaluation_policy": {"min_train_abs_ic": 0.01}})
        pure = rs.default_research_spec("fundamental")
        assert pure["evaluation_policy"]["min_train_abs_ic"] == 0.015


class TestRunSpecMerge:
    def test_build_run_without_explicit(self) -> None:
        rs.save_research_spec_overrides("technical", {"evaluation_policy": {"min_train_icir": 0.3}})
        run = rs.build_run_research_spec()
        assert run["evaluation_policy"]["min_train_icir"] == 0.3

    def test_build_run_saved_fills_missing_keys(self) -> None:
        rs.save_research_spec_overrides("fundamental", {"evaluation_policy": {"min_train_abs_ic": 0.01}})
        # 显式 spec 只给 research_mode，缺键由保存覆盖补齐
        run = rs.build_run_research_spec({"research_mode": "fundamental"})
        assert run["evaluation_policy"]["min_train_abs_ic"] == 0.01

    def test_explicit_beats_saved(self) -> None:
        rs.save_research_spec_overrides("technical", {"evaluation_policy": {"min_train_abs_ic": 0.01}})
        run = rs.build_run_research_spec({"research_mode": "technical", "evaluation_policy": {"min_train_abs_ic": 0.05}})
        assert run["evaluation_policy"]["min_train_abs_ic"] == 0.05

    def test_load_research_spec_merges_saved(self, tmp_path) -> None:
        rs.save_research_spec_overrides("fundamental", {"evaluation_policy": {"min_train_abs_ic": 0.009}})
        f = tmp_path / "spec.json"
        f.write_text(json.dumps({"research_mode": "fundamental"}), encoding="utf-8")
        loaded = rs.load_research_spec(f)
        assert loaded["evaluation_policy"]["min_train_abs_ic"] == 0.009

    def test_load_research_spec_none(self) -> None:
        loaded = rs.load_research_spec(None)
        assert loaded["research_mode"] == "technical"
        assert loaded["evaluation_policy"]["min_train_abs_ic"] == 0.02


class TestDiff:
    def test_diff_only_changed_keys(self) -> None:
        defaults = {"a": 1, "b": {"x": 1, "y": 2}, "c": [1, 2]}
        edited = {"a": 1, "b": {"x": 9, "y": 2}, "c": [1, 2]}
        assert rs.compute_spec_overrides(defaults, edited) == {"b": {"x": 9}}

    def test_diff_new_keys(self) -> None:
        defaults = {"a": 1}
        edited = {"a": 1, "z": 10}
        assert rs.compute_spec_overrides(defaults, edited) == {"z": 10}

    def test_diff_deleted_keys_restore_default(self) -> None:
        """编辑版删掉的键 = 恢复默认，不写入覆盖。"""
        defaults = {"a": 1, "b": 2}
        edited = {"a": 1}
        assert rs.compute_spec_overrides(defaults, edited) == {}

    def test_diff_list_replaced_wholesale(self) -> None:
        defaults = {"families": ["a", "b"]}
        edited = {"families": ["a", "b", "c"]}
        assert rs.compute_spec_overrides(defaults, edited) == {"families": ["a", "b", "c"]}


class TestProfileFallbackAlignment:
    def test_resolve_profiles_without_spec_uses_canonical(self) -> None:
        """无 evaluation_policy 时 train_screen 规则等于 DEFAULT_RESEARCH_SPEC 数值。"""
        from alphaagent.factor.evaluation.profile import resolve_profiles
        canonical = rs.DEFAULT_RESEARCH_SPEC["evaluation_policy"]
        profiles = resolve_profiles({})
        train_rules = {
            r["metric"]: r["value"]
            for r in profiles["train_screen"].rules
        }
        assert train_rules["cross_sectional_core.ic"] == canonical["min_train_abs_ic"]
        assert train_rules["cross_sectional_core.icir"] == canonical["min_train_icir"]
        assert train_rules["cross_sectional_core.factor_coverage"] == canonical["min_train_coverage"]
        val_rules = {r["metric"]: r["value"] for r in profiles["validation"].rules}
        assert val_rules["cross_sectional_core.ic"] == canonical["min_val_abs_ic"]

    def test_resolve_profiles_production_delivery_canonical(self) -> None:
        from alphaagent.factor.evaluation.profile import resolve_profiles
        canonical_prod = rs.DEFAULT_RESEARCH_SPEC["delivery_policy"]["production"]
        profiles = resolve_profiles({})
        prod_rules = {r["metric"]: r["value"] for r in profiles["production_delivery"].rules}
        assert prod_rules["cross_sectional_core.ic"] == canonical_prod["min_train_abs_ic"]
        assert prod_rules["cross_sectional_core.icir"] == canonical_prod["min_train_icir"]