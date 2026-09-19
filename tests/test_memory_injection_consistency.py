# -*- coding: utf-8 -*-
"""测试记忆注入层一致性与门控（对应 docs/specs/alphaagent_prompt_contradictions_repair_spec_v2.md D1~D4）。"""

import sqlite3
import pytest
from pathlib import Path
from alphaagent.factor.mining.research_memory import ResearchMemoryStore


def _create_mock_memory_db(db_path: Path):
    store = ResearchMemoryStore(db_path)
    with store._open() as conn:
        # 插入 vwap 族（大量记录，饱和度计算会 > 0.4）
        for i in range(10):
            conn.execute(
                """
                INSERT INTO memory_entries (
                    id, factor_name, expression, family, verdict, created_at, updated_at
                ) VALUES (?, ?, 'RANK($vwap)', 'vwap', 'promising', '2026-09-01', '2026-09-01')
                """,
                (f"e_vwap_{i}", f"vwap_f_{i}")
            )
        for i in range(3):
            conn.execute(
                """
                INSERT INTO memory_entries (
                    id, factor_name, expression, family, verdict, created_at, updated_at
                ) VALUES (?, ?, 'RANK($vwap)', 'vwap', 'validated', '2026-09-01', '2026-09-01')
                """,
                (f"e_vwap_val_{i}", f"vwap_val_f_{i}")
            )

        # 插入 chip 族（未拥挤族：1 个 promising，0 个 validated，饱和度 <= 0.4）
        conn.execute(
            """
            INSERT INTO memory_entries (
                id, factor_name, expression, family, verdict, created_at, updated_at
            ) VALUES ('e_chip_0', 'chip_f_0', 'RANK($chip)', 'chip', 'promising', '2026-09-01', '2026-09-01')
            """
        )

        # 插入 memory_cells
        # vwap 族有正残差
        conn.execute(
            """
            INSERT OR REPLACE INTO memory_cells (
                family, motif, parent_bucket,
                explicit_s, explicit_f, implicit_s, implicit_f,
                residuals_json, updated_at
            ) VALUES (
                'vwap', 'operator_substitute', 'high',
                5.0, 1.0, 0.0, 0.0, '[0.01, 0.015, 0.008]', '2026-09-01'
            )
            """
        )
        # chip 族有正残差
        conn.execute(
            """
            INSERT OR REPLACE INTO memory_cells (
                family, motif, parent_bucket,
                explicit_s, explicit_f, implicit_s, implicit_f,
                residuals_json, updated_at
            ) VALUES (
                'chip', 'window_rescale', 'high',
                4.0, 1.0, 0.0, 0.0, '[0.012, 0.018, 0.009]', '2026-09-01'
            )
            """
        )
    return store


def test_recommend_edits_skips_saturated_family(tmp_path):
    """D2: 饱和度 > 0.4 的拥挤族（vwap）严禁被 recommend_edits 推荐。"""
    db_file = tmp_path / "test_mem.db"
    retrieval = _create_mock_memory_db(db_file)
    sat = retrieval.compute_saturation()
    assert sat.get("vwap", {}).get("saturation_score", 0) > 0.4
    assert sat.get("chip", {}).get("saturation_score", 0) <= 0.4

    # 模拟 _best_family_parent 避免依赖复杂 parent 表查询
    retrieval._best_family_parent = lambda fam: {
        "parent_factor": f"{fam}_parent",
        "parent_expression": f"RANK(${fam})",
        "parent_ic": 0.025,
    }

    recs = retrieval.recommend_edits(k=2)
    assert len(recs) > 0
    # 推荐列表中绝对不能出现 vwap
    fams = [r["family"] for r in recs]
    assert "vwap" not in fams
    assert "chip" in fams


def test_recommend_edits_respects_excluded_families(tmp_path):
    """D4: 传入 excluded_families 时，该族严禁出现在推荐中。"""
    db_file = tmp_path / "test_mem.db"
    retrieval = _create_mock_memory_db(db_file)
    retrieval._best_family_parent = lambda fam: {
        "parent_factor": f"{fam}_parent",
        "parent_expression": f"RANK(${fam})",
        "parent_ic": 0.025,
    }

    recs = retrieval.recommend_edits(k=2, excluded_families={"chip"})
    fams = [r["family"] for r in recs]
    assert "chip" not in fams
