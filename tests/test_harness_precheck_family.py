# -*- coding: utf-8 -*-
"""scripts/harness_precheck_family.py 单测：评估前同族/重复预检三层判定。

背景（2026-10-09）：整夜重挖 7 候选全同源 → harness 直驱评估前补族内去重建议。
覆盖：
1. EXACT（批内同指纹——AST 指纹归一化变量/数字，**换窗长/换变量也命中**）；
2. EXACT（撞 candidate registry——已评估过=重复劳动）；
3. FAMILY（算子集相同但指纹不同=同族变体组，每组只保留代表）；
4. REGISTRY 关联（同算子集信息性提示）；
5. 无冲突 → strict 放行。
"""
from __future__ import annotations

import json
import tempfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.harness_precheck_family import (  # noqa: E402
    format_report,
    load_factors,
    load_registry,
    precheck,
)


def _mk_factors(*pairs: tuple[str, str]) -> list[dict]:
    return [{"name": n, "expr": e} for n, e in pairs]


def test_exact_batch_window_variant() -> None:
    """同构换窗长 → AST 指纹相同 → 批内 EXACT（正是要拦的重复劳动）。"""
    factors = _mk_factors(
        ("mom5", "TS_MEAN($close, 5)"),
        ("mom20", "TS_MEAN($close, 20)"),
    )
    rep = precheck(factors, registry=[])
    assert rep["exact_batch"] == [["mom5", "mom20"]]
    assert rep["exact_registry"] == []
    assert rep["family_groups"] == []


def test_exact_registry_collision() -> None:
    """新因子结构指纹在 registry 已存在 → EXACT 撞库。"""
    factors = _mk_factors(("c2", "TS_MEAN($close, 20)"))
    registry = [("c1", "TS_MEAN($close, 5)")]
    rep = precheck(factors, registry)
    assert rep["exact_batch"] == []
    assert rep["exact_registry"] == [{"factor": "c2", "registry": ["c1"]}]


def test_family_group_same_ops_diff_topology() -> None:
    """算子集相同但结构不同（嵌套深度）→ 强同族组（指纹不同，非 EXACT）。"""
    factors = _mk_factors(
        ("flat", "TS_MEAN($close, 5)"),
        ("nested", "TS_MEAN(TS_MEAN($close, 5), 20)"),
    )
    rep = precheck(factors, registry=[])
    assert rep["exact_batch"] == []
    assert rep["family_groups"] == [["flat", "nested"]]


def test_registry_links_same_ops() -> None:
    """与 registry 同算子集（结构不同）→ 信息性关联（非 EXACT）。"""
    factors = _mk_factors(("new", "TS_MEAN(TS_MEAN($close, 5), 20)"))
    registry = [("old", "TS_MEAN($close, 5)")]
    rep = precheck(factors, registry)
    assert rep["exact_registry"] == []
    assert rep["registry_links"] == [{"factor": "new", "registry": ["old"]}]


def test_no_conflict_strict_pass() -> None:
    """无关因子 + 无关 registry → 无冲突（strict 会放行，exit 0）。"""
    factors = _mk_factors(
        ("a", "TS_RANK($close, 20)"),
        ("b", "CS_ZSCORE($open)"),
    )
    registry = [("x", "TS_STD($high, 60)")]
    rep = precheck(factors, registry)
    assert rep == {
        "exact_batch": [],
        "exact_registry": [],
        "family_groups": [],
        "registry_links": [],
    }


def test_load_factors_requires_nonempty_expr() -> None:
    """expr 为空 → ValueError（防无效因子混入批次）。"""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "bad.json"
        p.write_text(json.dumps([{"name": "x", "expr": "  "}]), encoding="utf-8")
        try:
            load_factors(p)
            raise AssertionError("应抛出 ValueError")
        except ValueError:
            pass


def test_exact_batch_three_same_fp_one_group() -> None:
    """三个同指纹因子 → 单个组（不是多对拆分，便于保留一个即可的决策）。"""
    factors = _mk_factors(
        ("x1", "TS_MEAN($close, 5)"),
        ("x2", "TS_MEAN($close, 20)"),
        ("x3", "TS_MEAN($close, 60)"),
    )
    rep = precheck(factors, registry=[])
    assert rep["exact_batch"] == [["x1", "x2", "x3"]]


def test_load_registry_filters_missing_expr() -> None:
    """registry 加载：无 expr 条目跳过、文件不存在（默认路径）返回空。"""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "reg.json"
        p.write_text(
            json.dumps({
                "ok": {"expr": "TS_MEAN($close, 5)"},
                "no_expr": {"metrics": {}},
            }),
            encoding="utf-8",
        )
        assert load_registry(p) == [("ok", "TS_MEAN($close, 5)")]
    assert load_registry(Path(d) / "reg.json") == []  # 目录已删 → 文件不存在（降级）


def test_load_registry_expect_file_missing_raises() -> None:
    """显式传入但文件不存在 → FileNotFoundError（防"EXACT 0 处"假阴性 fail-open）。"""
    try:
        load_registry(Path("C:/definitely/not/here/reg.json"), expect_file=True)
        raise AssertionError("应抛出 FileNotFoundError")
    except FileNotFoundError:
        pass


def test_load_registry_bad_json_raises() -> None:
    """文件存在但 JSON 损坏 / 顶层非 dict → ValueError（不静默返回空）。"""
    import pytest as _pytest

    with tempfile.TemporaryDirectory() as d:
        bad = Path(d) / "bad.json"
        bad.write_text("{ not json !!", encoding="utf-8")
        with _pytest.raises(ValueError):
            load_registry(bad)
        not_dict = Path(d) / "list.json"
        not_dict.write_text(json.dumps([{"expr": "x"}]), encoding="utf-8")
        with _pytest.raises(ValueError):
            load_registry(not_dict)


def test_format_report_smoke() -> None:
    """报告格式化：三类命中行齐全、不抛异常。"""
    factors = _mk_factors(
        ("dup1", "TS_MEAN($close, 5)"),
        ("dup2", "TS_MEAN($close, 10)"),
        ("nested", "TS_MEAN(TS_MEAN($close, 5), 20)"),
    )
    rep = precheck(factors, registry=[])
    out = format_report(rep, len(factors))
    assert "EXACT_DUPLICATE" in out
    assert "强同族组" in out
    assert "无冲突" not in out


def test_format_report_no_conflict() -> None:
    """无冲突时报告明示可放心评估。"""
    factors = _mk_factors(("a", "TS_RANK($close, 20)"))
    out = format_report(precheck(factors, registry=[]), len(factors))
    assert "无冲突" in out