# -*- coding: utf-8 -*-
"""统一大库（Phase 2）回归：两模式共享 candidate_main/production_main +
factors API facet 筛选（含 "融合" 过滤与老条目表达式兜底）。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import factor_categories
from core.research_modes import RESEARCH_MODES


def test_unified_library_routing():
    """两模式全部指向统一大库目录。"""
    for mode in RESEARCH_MODES:
        assert factor_categories.candidate_dir(mode).name == "candidate_main", mode
        assert factor_categories.production_dir(mode).name == "production_main", mode
    # 且确实只有一个物理候选/正式库
    cand = {factor_categories.candidate_dir(m) for m in RESEARCH_MODES}
    prod = {factor_categories.production_dir(m) for m in RESEARCH_MODES}
    assert len(cand) == 1 and len(prod) == 1


def test_facet_filter_on_candidate_registry(tmp_path, monkeypatch):
    """facet 筛选：facets 交集、"融合" 关键字、老条目表达式兜底。"""
    import backend.alphaagent_service as svc

    registry = {
        "fusion_one": {
            "name": "funda_x_price",
            "expr": "RANK(MULTIPLY($funda_roe, TS_STD($close,20)))",
            "facets": ["价量面", "基本面"],
            "is_fusion": True,
        },
        "pure_vol": {
            "name": "vol_only",
            "expr": "TS_STD($close,20)",
            "facets": ["价量面"],
            "is_fusion": False,
        },
        # 老条目：无 facets 字段 → 按表达式现算兜底
        "legacy_entry": {
            "name": "old_funda_factor",
            "expr": "RANK($funda_roe)",
        },
        # 无 facets 且无 expr → 按 expression_file 读取（缺文件则无面 → 不匹配）
        "no_data": {"name": "nothing"},
    }
    monkeypatch.setattr(svc, "_candidate_registry", lambda category: dict(registry))

    fusion = svc.list_factors(library="candidate", facet="融合")
    assert sorted(f["factor_id"] for f in fusion["factors"]) == ["fusion_one"]

    funda = svc.list_factors(library="candidate", facet="基本面")
    assert sorted(f["factor_id"] for f in funda["factors"]) == ["fusion_one", "legacy_entry"]

    none = svc.list_factors(library="candidate", facet="资金面")
    assert none["n_factors"] == 0


def test_removed_library_list(tmp_path, monkeypatch):
    """已删除因子回收站（removed_main）：列表含删除元数据，表达式已归档。"""
    import backend.alphaagent_service as svc

    removed = {
        "vsi_dblil_ema3": {
            "name": "vsi_dblil_ema3",
            "expr": "$close\n$open",
            "removed_at": "2026-10-11T18:30:00+08:00",
            "removed_from": "candidate",
            "removed_reason": "dedup_20261011_threshold_0.9",
        }
    }
    monkeypatch.setattr(svc, "_removed_registry", lambda category: dict(removed))
    monkeypatch.setattr(factor_categories, "removed_dir", lambda mode: tmp_path / "removed_main")

    res = svc.list_factors(library="removed")
    assert res["library"] == "removed"
    assert res["n_factors"] == 1
    f = res["factors"][0]
    assert f["name"] == "vsi_dblil_ema3"
    assert f["removed_at"].startswith("2026-10-11")
    assert f["removed_from"] == "candidate"
    assert f["removed_reason"] == "dedup_20261011_threshold_0.9"
    assert f["expr"] == "$close\n$open"  # 删除库表达式已随归档落盘，不走原库 DSL 兜底

    detail = svc.get_factor_detail("vsi_dblil_ema3", library="removed")
    assert detail["removed_reason"] == "dedup_20261011_threshold_0.9"
    assert detail["registry_entry"]["removed_from"] == "candidate"
    assert svc.get_factor_detail("missing", library="removed")["error"] == "factor_not_found"


def test_archive_to_removed_idempotent(tmp_path, monkeypatch):
    """删除归档：条目写入 removed registry + DSL 落盘，重复删除幂等不覆盖。"""
    import backend.alphaagent_service as svc

    monkeypatch.setattr(factor_categories, "removed_dir", lambda mode: tmp_path / "removed_main")
    monkeypatch.setattr(factor_categories, "removed_registry_path", lambda mode: tmp_path / "removed_main" / "mining_removed_registry.json")

    svc._archive_to_removed(
        "demo_factor",
        {"name": "demo", "expr": "$close"},
        category="technical", source_library="candidate", reason="test",
    )
    reg = json.loads((tmp_path / "removed_main" / "mining_removed_registry.json").read_text(encoding="utf-8"))
    entry = reg["demo_factor"]
    assert entry["removed_from"] == "candidate"
    assert entry["removed_reason"] == "test"
    assert entry["removed_at"]
    assert (tmp_path / "removed_main" / "expressions" / "demo_factor.dsl").is_file()
    assert not entry["expr"].endswith("+")  # 不追加

    # 幂等：再次归档同一因子不覆盖（保留原 removed_at）
    first_removed_at = entry["removed_at"]
    svc._archive_to_removed(
        "demo_factor",
        {"name": "demo_v2"},
        category="technical", source_library="candidate", reason="test2",
    )
    reg = json.loads((tmp_path / "removed_main" / "mining_removed_registry.json").read_text(encoding="utf-8"))
    assert reg["demo_factor"]["removed_at"] == first_removed_at
    assert reg["demo_factor"]["name"] == "demo"
