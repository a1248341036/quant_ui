"""core.factor_registry：引擎因子元数据注册表测试。"""

from __future__ import annotations

from core.composites import FACTOR_OPTIONS
from core.factor_registry import factor_options, known_factor, validate_factor_refs
from strategies.registry import STRATEGIES, validate_registry_factors


def test_options_contains_all_legacy_factors():
    """注册表派生出的 options 覆盖旧 FACTOR_OPTIONS 全部条目。"""
    legacy_names = {
        "mom20", "mom60", "vol20", "ma_cross5_10", "ma_cross5_20",
        "ma_cross10_30", "ma_cross20_60", "turn20", "am20",
        "mdd20", "mdd60", "sharpe20", "sharpe60", "sortino20",
        "mom_accel", "nav_stability",
        "pb", "ep", "roe", "gross_margin", "rev_yoy", "np_yoy",
    }
    names = {o["name"] for o in FACTOR_OPTIONS}
    assert legacy_names <= names


def test_non_composable_hidden_from_options():
    """composite/pred 不作为组合编辑器的成员因子。"""
    names = {o["name"] for o in factor_options()}
    assert "composite" not in names
    assert "pred" not in names


def test_registry_typing_contract():
    """每个 option 必须有 name/label/desc/types。"""
    for o in factor_options():
        assert o["name"]
        assert o["label"]
        assert o["desc"]
        assert "types" in o and o["types"]


def test_strategy_factor_refs_all_valid():
    """策略注册表引用的因子全部在引擎因子注册表中。"""
    assert validate_registry_factors() == []


def test_validate_factor_refs_catches_unknown():
    assert validate_factor_refs(["mom20", "nope"]) == ["nope"]
    assert known_factor("mom20")
    assert not known_factor("nope")


def test_strategy_types_are_known_asset_types():
    allowed = {"stock", "etf", "fund"}
    for name, cfg in STRATEGIES.items():
        types = cfg.get("types", ["stock", "etf", "fund"])
        assert all(t in allowed for t in types), f"{name}: {types}"