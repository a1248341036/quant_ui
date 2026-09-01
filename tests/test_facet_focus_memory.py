# -*- coding: utf-8 -*-
"""数据面聚焦 facet_focus 插件模块 + 白名单豁免 + facets 分类/亲和回归。

覆盖总计划 Phase 1 验收点：
- FACET_GROUPS 跨组融合判定（假融合防线）与 $float_cap 移出股东面；
- classify_family_ex 面对组合键 / 单面旧行为不变；
- memory_entries.facets_json 迁移幂等 + 读取兜底；
- 检索亲和两档（family 精确 +0.3 / facets 重叠 +0.15）；
- facet_focus 板块 enabled/render（单面/多面/豁免句）与装配报告。
"""
import json
import sqlite3

import pytest

from alphaagent.factor.mining.memory.expressions import (
    FACET_DEFS,
    classify_family,
    classify_family_ex,
    expr_facets,
    facet_groups,
    fusion_family_key,
    is_cross_group_fusion,
)
from alphaagent.factor.mining.research_memory import ResearchMemoryStore


def _eval_row(name, expr, factor_name, ic=None, icir=0.4, coverage=0.9, error=None, extra_args=None):
    args = {"multi_line_expr": expr, "factor_name": factor_name}
    if extra_args:
        args.update(extra_args)
    result = {"ok": error is None, "split": "train"}
    if ic is not None:
        result["metrics"] = {"ic": ic, "icir": icir, "factor_coverage": coverage}
    if error:
        result["error"] = error
    return {"name": name, "arguments_raw": json.dumps(args), "result": result}


# ── 1.1 假融合防线 ──

def test_same_group_multi_facet_is_not_fusion():
    """价量+量能同属行情组：触及 2 面但不算融合。"""
    facets = expr_facets("RANK(DIVIDE($close, $volume))")
    assert {"价量面", "量能面"} <= facets
    assert not is_cross_group_fusion(facets)


def test_float_cap_no_longer_maps_to_holder_facet():
    """$float_cap 是行情面板列，移出股东面识别键。"""
    facets = expr_facets("CS_NEUTRALIZE(RANK($close), LOG($float_cap))")
    assert "股东面" not in facets


def test_cross_group_fusion_detected():
    """funda_ + $close 跨行情/基本面两组 → 融合。"""
    facets = expr_facets("RANK(MULTIPLY($funda_roe_ttm, TS_STD($close,20)))")
    assert is_cross_group_fusion(facets)
    assert facet_groups(facets) == {"行情组", "基本面组"}


# ── 1.2 classify_family_ex ──

def test_classify_family_ex_fusion_pair_key():
    """跨组融合 → 面对键，键内面名按 FACET_DEFS 稳定排序。

    这是改造前被误分为 volatility 的实锤用例（std 关键词排在 funda_ 前）。
    """
    fam, facets = classify_family_ex(
        "funda_mom_x_price", "RANK(MULTIPLY($funda_roe_ttm, TS_STD($close,20)))"
    )
    assert fam == fusion_family_key(facets) == "价量面×基本面"


def test_classify_family_single_facet_unchanged():
    """单面因子走旧规则，与历史分类完全一致。"""
    assert classify_family("vwap_close_dev", "RANK(DIVIDE($vwap, $close))") == "vwap"
    assert classify_family("", "RANK($funda_roe)") == "fundamental"
    assert classify_family("chip_peak_dev", "CHIP_PEAK_LOC($close,20)") == "chip"
    assert classify_family("unknown_thing", "RANK($nonexistent_col)") == "other"


def test_facet_order_stable():
    order = {name: i for i, (name, _) in enumerate(FACET_DEFS)}
    assert order["价量面"] < order["基本面"]
    assert fusion_family_key({"基本面", "价量面"}) == "价量面×基本面"
    assert fusion_family_key({"股东面", "量能面"}) == "量能面×股东面"


# ── 1.3 facets_json 持久化 + 读取兜底 ──

def test_facets_json_persisted(tmp_path):
    store = ResearchMemoryStore(tmp_path / "m.db")
    store.record_tool_result(
        run_id="r1",
        row=_eval_row(
            "evaluate_factor",
            "RANK(MULTIPLY($funda_roe_ttm, TS_STD($close,20)))",
            "funda_x_price",
            ic=0.03,
        ),
    )
    with store._open() as conn:
        rec = conn.execute(
            "SELECT family, facets_json FROM memory_entries WHERE factor_name = ?",
            ("funda_x_price",),
        ).fetchone()
    assert rec["family"] == "价量面×基本面"
    assert json.loads(rec["facets_json"]) == ["价量面", "基本面"]


def test_facets_column_migration_idempotent(tmp_path):
    """打开库即补 facets_json 列 + data_version=v4，重复打开幂等。"""
    db = tmp_path / "m.db"
    for _ in range(2):
        store = ResearchMemoryStore(db)
        with store._open() as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(memory_entries)").fetchall()}
            ver = conn.execute("SELECT v FROM store_meta WHERE k='data_version'").fetchone()[0]
        assert "facets_json" in cols
        assert ver == "4"


def test_legacy_row_facets_fallback_and_recall(tmp_path):
    """老行（无 facets_json / family 过时）→ 读取侧现算兜底，仍可被 facets 亲召回。"""
    store = ResearchMemoryStore(tmp_path / "m.db")
    with store._open() as conn:
        conn.execute(
            "INSERT INTO memory_entries (id, factor_name, expression, verdict, family, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            ("legacy1", "old_factor", "RANK(MULTIPLY($funda_roe, $close))", "weak", "volatility", "2026-09-01T00:00:00"),
        )
        conn.commit()
    entries = store._retrieval_candidates("RANK($funda_roe)", True)
    match = [e for e in entries if e.get("id") == "legacy1"]
    assert match, "老条目应能被检索"
    assert "基本面" in match[0]["facets"]


# ── 1.3 亲和两档 ──

def _static_entry():
    return {
        "family": "volatility",
        "facets": {"基本面", "价量面"},
        "expression": "RANK(MULTIPLY($funda_roe, TS_STD($close,20)))",
        "verdict": "weak",
        "updated_at": "2026-09-01T00:00:00",
        "_bm25": 0.0,
    }


def test_hybrid_score_facet_overlap_tier():
    entry = _static_entry()
    base = dict(query_ops=set(), now_ts=1.7e9)
    no_match = ResearchMemoryStore._hybrid_score(entry, focus_families=set(), **base)
    overlap = ResearchMemoryStore._hybrid_score(entry, focus_families=set(), query_facets={"基本面"}, **base)
    exact = ResearchMemoryStore._hybrid_score(entry, focus_families={"volatility"}, **base)
    assert overlap - no_match == pytest.approx(0.15)
    assert exact - no_match == pytest.approx(0.30)
    # family 精确命中优先，不再叠加 facets 档
    both = ResearchMemoryStore._hybrid_score(entry, focus_families={"volatility"}, query_facets={"基本面"}, **base)
    assert both - no_match == pytest.approx(0.30)


# ── 1.5 facet_focus 板块 ──

def test_facet_focus_disabled_without_focus():
    from alphaagent.factor.mining.prompt.prompt_modules import PromptContext, assemble_system_prompt
    from alphaagent.factor.mining.prompt.modules import DEFAULT_MODULES

    text, report = assemble_system_prompt(DEFAULT_MODULES, PromptContext())
    row = next(r for r in report if r["module"] == "facet_focus")
    assert row["enabled"] is False and row["chars"] == 0
    assert "数据面聚焦" not in text


def test_facet_focus_multi_facet_render_with_exemption():
    from alphaagent.factor.mining.prompts import build_system_prompt

    text = build_system_prompt(focus_facets=["基本面", "价量面"])
    assert "跨面融合" in text
    assert "MULTIPLY" in text
    assert "_x_" in text
    # 白名单豁免句必须存在（ResearchSpec extra_instructions 注入了族白名单）
    assert "信号族白名单对本指令豁免" in text


def test_facet_focus_single_facet_render():
    from alphaagent.factor.mining.prompts import build_system_prompt

    text = build_system_prompt(focus_facets=["价量面"])
    assert "单面聚焦不要求跨面融合" in text
    assert "信号族白名单对本指令豁免" in text


def test_facet_focus_in_assembly_report():
    from alphaagent.factor.mining.prompts import build_system_prompt, last_assembly_report

    build_system_prompt(focus_facets=["基本面", "价量面"])
    rows = {r["module"]: r for r in last_assembly_report}
    assert rows["facet_focus"]["enabled"] is True
    assert rows["facet_focus"]["chars"] > 0
    # ORDER=135：排在 population_mode(130) 之后、extra_instructions(140) 之前
    assert 130 < rows["facet_focus"]["order"] < 140
