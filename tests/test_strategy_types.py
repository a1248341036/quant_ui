"""core.strategy_types / core.strategy_pool：统一策略定义模型测试。"""

from __future__ import annotations

import pytest

from core.strategy_types import StrategyDefinition, from_dsl_factor, from_legacy_dict
from core.strategy_pool import resolve_strategy, resolve_strategy_def
from strategies.registry import STRATEGIES


def test_from_legacy_dict_parses_registry_item():
    d = STRATEGIES["低换手冷门"]
    sd = from_legacy_dict("低换手冷门", d, source="registry")
    assert sd.kind == "factor"
    assert sd.factor == d["factor"]
    assert sd.ascending is True
    assert sd.source == "registry"
    assert sd.types == ("stock", "etf")


def test_to_dict_matches_legacy_shape():
    d = STRATEGIES["动量 20 日"]
    sd = from_legacy_dict("动量 20 日", d)
    out = sd.to_dict()
    assert out["factor"] == d["factor"]
    assert out["ascending"] == d["ascending"]
    assert out["group"] == d["group"]
    assert out["desc"] == d["desc"]
    assert "types" in out and out["types"] == ["stock", "etf", "fund"]


def test_legacy_params_promoted_into_to_dict():
    sd = from_legacy_dict("x", {
        "factor": "am20", "ascending": True,
        "industry_cap": 1, "adx_filter": 25.0,
    })
    out = sd.to_dict()
    assert out["industry_cap"] == 1
    assert out["adx_filter"] == 25.0


def test_fingerprint_stable_and_sensitive():
    a = from_legacy_dict("s", {"factor": "mom20", "ascending": False})
    b = from_legacy_dict("s", {"factor": "mom20", "ascending": False})
    c = from_legacy_dict("s", {"factor": "mom20", "ascending": True})
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()
    assert len(a.fingerprint()) == 16


def test_from_dsl_factor():
    sd = from_dsl_factor("mom_20d", name="动量20", dsl_expr="TS_MEAN($ret, 20)")
    assert sd.kind == "dsl"
    assert sd.dsl_expr == "TS_MEAN($ret, 20)"
    assert sd.types == ("stock",)
    assert sd.source == "dsl"


def test_resolve_strategy_def_registry_source():
    sd = resolve_strategy_def("低换手冷门")
    assert sd.kind == "factor"
    assert sd.source == "registry"
    assert sd.factor == "turn20"


def test_resolve_strategy_dict_compat():
    # 旧调用面继续可用
    d = resolve_strategy("低换手冷门")
    assert d["factor"] == "turn20"
    assert d["ascending"] is True


def test_resolve_strategy_dict_preserves_all_extra_params():
    """旧 resolve_strategy 的 dict 必须扁平展开全部附加参数（不丢参）。"""
    for name, cfg in STRATEGIES.items():
        out = resolve_strategy(name)
        for k, v in cfg.items():
            if k in ("factor", "ascending", "group", "desc", "types"):
                continue
            # 附加参数（industry_cap/adx_filter/long_short 等）必须出现在输出 dict
            assert out.get(k) == v, f"{name}: 参数 {k} 丢失或变化 {out.get(k)!r} vs {v!r}"


def test_resolve_strategy_dict_has_legacy_core_keys():
    """输出 dict 必须包含旧调用方依赖的核心键。"""
    for name in ("低换手冷门", "动量 20 日", "双均线多头 20/30 ADX25",
                 "多空动量 20 日", "冷门+行业分散"):
        out = resolve_strategy(name)
        for key in ("factor", "ascending", "group", "desc"):
            assert key in out, f"{name}: 缺少 {key}"


def test_resolve_unknown_raises():
    with pytest.raises(KeyError):
        resolve_strategy_def("不存在的策略xyz")


def test_resolve_strategy_def_returns_immutable_like():
    sd = resolve_strategy_def("动量 20 日")
    assert isinstance(sd, StrategyDefinition)
    assert sd.fingerprint()