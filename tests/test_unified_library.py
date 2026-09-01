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
