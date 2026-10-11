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


def test_delete_factor_archives_to_removed(tmp_path, monkeypatch):
    """delete_factor(candidate) 先归档再删（registry/DSL），回收站含完整条目。"""
    import backend.alphaagent_service as svc

    cand_reg = {
        "del_me": {
            "name": "del_me",
            "expr": "$close",
            "factor_uid": "uid-1",
            "expression_file": "artifacts/x/del_me.dsl",
        }
    }
    removed_files: dict = {}

    monkeypatch.setattr(svc, "_candidate_registry", lambda category: dict(cand_reg))
    monkeypatch.setattr(svc, "_removed_registry", lambda category: dict(removed_files))
    monkeypatch.setattr(factor_categories, "removed_dir", lambda mode: tmp_path / "removed_main")
    monkeypatch.setattr(factor_categories, "removed_registry_path",
                        lambda mode: tmp_path / "removed_main" / "mining_removed_registry.json")
    # 原库根目录用真实路径会碰真库 → 指向 tmp（candidate_root 仅用于算 registry 写盘路径）
    monkeypatch.setattr(svc, "_candidate_root", lambda category: tmp_path / "candidate_main")
    (tmp_path / "candidate_main").mkdir(parents=True, exist_ok=True)

    # 命中 candidate 分支：进入即 pop + 归档；research memory / index 属副作用，
    # 用假类跳过真实 db（delete_factor 内函数级 import，patch 模块属性即可生效）
    class _FakeMemory:
        def __init__(self, path):
            self.path = path

        def purge_factor(self, **kw):
            return 0

    monkeypatch.setattr(
        "alphaagent.factor.mining.research_memory.ResearchMemoryStore", _FakeMemory
    )
    monkeypatch.setattr(
        "alphaagent.factor.index.get_factor_index",
        lambda: type("Idx", (), {"delete_factor": lambda self, uid: None})(),
    )

    res = svc.delete_factor("del_me", library="candidate", category="technical", reason="与保留因子相关性 0.95，冗余")
    assert res["ok"] is True
    # 回收站已归档（含用户填写原因）
    got = json.loads((tmp_path / "removed_main" / "mining_removed_registry.json").read_text(encoding="utf-8"))
    assert got["del_me"]["name"] == "del_me"
    assert got["del_me"]["removed_from"] == "candidate"
    assert got["del_me"]["removed_reason"] == "与保留因子相关性 0.95，冗余"
    assert got["del_me"]["removed_at"]
    # DSL 已在归档时落盘
    assert (tmp_path / "removed_main" / "expressions" / "del_me.dsl").is_file()
    # 候选 registry 已写盘（pop 后为空）
    cand_on_disk = json.loads(
        (tmp_path / "candidate_main" / "mining_candidate_registry.json").read_text(encoding="utf-8")
    )
    assert cand_on_disk == {}


def test_delete_factor_reason_fallback(tmp_path, monkeypatch):
    """reason 为空时回落默认标记（脚本/存量调用兼容）。"""
    import backend.alphaagent_service as svc

    cand_reg = {"legacy_x": {"name": "legacy_x", "expr": "$close"}}
    monkeypatch.setattr(svc, "_candidate_registry", lambda category: dict(cand_reg))
    monkeypatch.setattr(svc, "_removed_registry", lambda category: {})
    monkeypatch.setattr(factor_categories, "removed_dir", lambda mode: tmp_path / "removed_main")
    monkeypatch.setattr(factor_categories, "removed_registry_path",
                        lambda mode: tmp_path / "removed_main" / "mining_removed_registry.json")
    monkeypatch.setattr(svc, "_candidate_root", lambda category: tmp_path / "candidate_main")
    (tmp_path / "candidate_main").mkdir(parents=True, exist_ok=True)

    class _FakeMemory2:
        def __init__(self, path):
            self.path = path

        def purge_factor(self, **kw):
            return 0

    monkeypatch.setattr(
        "alphaagent.factor.mining.research_memory.ResearchMemoryStore", _FakeMemory2
    )
    monkeypatch.setattr(
        "alphaagent.factor.index.get_factor_index",
        lambda: type("Idx", (), {"delete_factor": lambda self, uid: None})(),
    )

    svc.delete_factor("legacy_x", library="candidate", category="technical")  # 不传 reason
    got = json.loads((tmp_path / "removed_main" / "mining_removed_registry.json").read_text(encoding="utf-8"))
    assert got["legacy_x"]["removed_reason"] == "deleted_from_candidate"


def test_delete_factor_rejects_removed_library(tmp_path, monkeypatch):
    """回收站只读：delete_factor 拒绝 removed（前端已禁用，后端防御）。"""
    import backend.alphaagent_service as svc

    res = svc.delete_factor("anything", library="removed", category="technical")
    assert res == {"error": "readonly_library"}
