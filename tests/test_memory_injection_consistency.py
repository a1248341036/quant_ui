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


# ── D1: _edit_prior_block 饱和族推荐降级 ──────────────────────────


def _create_mock_memory_db_d1(db_path: Path):
    """D1 测试库：vwap 饱和（>0.4）+ chip 未饱和，两族都有 recommend 档 cells。"""
    store = ResearchMemoryStore(db_path)
    with store._open() as conn:
        # vwap 族：10 promising + 3 validated → 饱和度 > 0.4
        for i in range(10):
            conn.execute(
                "INSERT INTO memory_entries (id, factor_name, expression, family, verdict, "
                "created_at, updated_at) VALUES (?, ?, 'RANK($vwap)', 'vwap', 'promising', "
                "'2026-09-01', '2026-09-01')",
                (f"d1_vwap_{i}", f"vwap_f_{i}"),
            )
        for i in range(3):
            conn.execute(
                "INSERT INTO memory_entries (id, factor_name, expression, family, verdict, "
                "created_at, updated_at) VALUES (?, ?, 'RANK($vwap)', 'vwap', 'validated', "
                "'2026-09-01', '2026-09-01')",
                (f"d1_vwap_val_{i}", f"vwap_val_f_{i}"),
            )
        # chip 族：1 promising → 饱和度 <= 0.4
        conn.execute(
            "INSERT INTO memory_entries (id, factor_name, expression, family, verdict, "
            "created_at, updated_at) VALUES ('d1_chip_0', 'chip_f_0', 'RANK($chip)', 'chip', "
            "'promising', '2026-09-01', '2026-09-01')"
        )

        # vwap soft_recommend：8 个一致正残差 → conf ≈ 0.5 > 0.4
        conn.execute(
            "INSERT OR REPLACE INTO memory_cells (family, motif, parent_bucket, "
            "explicit_s, explicit_f, implicit_s, implicit_f, residuals_json, updated_at) "
            "VALUES ('vwap', 'window_rescale', 'high', 4.0, 1.0, 0.0, 0.0, "
            "'[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]', '2026-09-01')"
        )
        # vwap hard_recommend：20 个一致正残差 → conf ≈ 0.714 > 0.7
        conn.execute(
            "INSERT OR REPLACE INTO memory_cells (family, motif, parent_bucket, "
            "explicit_s, explicit_f, implicit_s, implicit_f, residuals_json, updated_at) "
            "VALUES ('vwap', 'operator_substitute', 'high', 12.0, 1.0, 0.0, 0.0, "
            "'[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, "
            "0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]', '2026-09-01')"
        )
        # chip soft_recommend（未饱和，应正常注入）
        conn.execute(
            "INSERT OR REPLACE INTO memory_cells (family, motif, parent_bucket, "
            "explicit_s, explicit_f, implicit_s, implicit_f, residuals_json, updated_at) "
            "VALUES ('chip', 'window_rescale', 'high', 4.0, 1.0, 0.0, 0.0, "
            "'[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]', '2026-09-01')"
        )
    return store


def test_edit_prior_block_soft_recommend_skipped_when_saturated(tmp_path):
    """D1: 饱和族（>0.4）的 soft_recommend 档应被跳过（不注入）。

    vwap 的 window_rescale（soft_recommend，conf 50%）在饱和度 1.0 下应被 continue 跳过。
    验证：block 中不应同时出现 vwap 族 + 调整窗口参数 + 优先尝试（soft_recommend 标签）。
    """
    retrieval = _create_mock_memory_db_d1(tmp_path / "d1_mem.db")
    block = retrieval._edit_prior_block()
    # soft_recommend 的标签是"优先尝试"，hard_recommend 是"优先采用"
    # vwap 的 window_rescale 是 soft_recommend，应被跳过
    # 检查：不出现"VWAP"与"调整窗口参数"同行的 soft_recommend
    assert "VWAP" in block  # vwap 的 hard_recommend 行仍在（降级标注）
    # soft_recommend 行不应出现 vwap：查找所有"优先尝试"行，不应含 VWAP
    soft_lines = [ln for ln in block.splitlines() if "优先尝试" in ln]
    assert not any("VWAP" in ln for ln in soft_lines), \
        "饱和族 vwap 的 soft_recommend 应被跳过，不应出现在'优先尝试'行"


def test_edit_prior_block_hard_recommend_downgraded_when_saturated(tmp_path):
    """D1: 饱和族（>0.4）的 hard_recommend 档应降级标注"饱和拥挤区"，不直接跳过。"""
    retrieval = _create_mock_memory_db_d1(tmp_path / "d1_mem.db")
    block = retrieval._edit_prior_block()
    # hard_recommend 降级后仍注入，但带"饱和拥挤区"标注
    assert "饱和拥挤区" in block
    # vwap 的 operator_substitute（hard_recommend）应存在，用中文标签"替换核心算子"
    assert "替换核心算子" in block
    # 该行应是"优先采用"档（hard_recommend 标签）
    hard_lines = [ln for ln in block.splitlines() if "优先采用" in ln]
    assert any("VWAP" in ln and "饱和拥挤区" in ln for ln in hard_lines), \
        "vwap hard_recommend 应降级为带'饱和拥挤区'标注的'优先采用'行"


def test_edit_prior_block_unsaturated_family_recommend_kept(tmp_path):
    """D1: 未饱和族（<=0.4）的 recommend 档应正常注入，无降级标注。"""
    retrieval = _create_mock_memory_db_d1(tmp_path / "d1_mem.db")
    block = retrieval._edit_prior_block()
    # chip 未饱和（0.025），其 soft_recommend 应正常注入（中文族名"筹码分布类"）
    assert "筹码分布" in block
    # chip 行不应带"饱和拥挤区"标注（该标注只针对 vwap）
    chip_lines = [ln for ln in block.splitlines() if "筹码分布" in ln]
    assert chip_lines, "chip 族 recommend 行应存在"
    assert not any("饱和拥挤区" in ln for ln in chip_lines), \
        "未饱和族 chip 不应带'饱和拥挤区'标注"


# ── D3: recommend_edits 跳过历史零产出族 ──────────────────────────


def _create_mock_memory_db_d3(db_path: Path):
    """D3 测试库：dead 族 30+ 条全 rejected（0 过线）+ 有正残差 cells，chip 族正常。"""
    store = ResearchMemoryStore(db_path)
    with store._open() as conn:
        # dead 族：35 条全 rejected/weak，0 条 positive → zero_yield_fams
        for i in range(35):
            verdict = "rejected" if i % 2 == 0 else "weak"
            conn.execute(
                "INSERT INTO memory_entries (id, factor_name, expression, family, verdict, "
                "created_at, updated_at) VALUES (?, ?, 'RANK($dead)', 'dead', ?, "
                "'2026-09-01', '2026-09-01')",
                (f"d3_dead_{i}", f"dead_f_{i}", verdict),
            )
        # chip 族：1 promising → 未饱和、非零产出
        conn.execute(
            "INSERT INTO memory_entries (id, factor_name, expression, family, verdict, "
            "created_at, updated_at) VALUES ('d3_chip_0', 'chip_f_0', 'RANK($chip)', 'chip', "
            "'promising', '2026-09-01', '2026-09-01')"
        )

        # dead 族有正残差 cells（否则 recommend_edits 不会选它）
        conn.execute(
            "INSERT OR REPLACE INTO memory_cells (family, motif, parent_bucket, "
            "explicit_s, explicit_f, implicit_s, implicit_f, residuals_json, updated_at) "
            "VALUES ('dead', 'window_rescale', 'high', 5.0, 1.0, 0.0, 0.0, "
            "'[0.01, 0.012, 0.011, 0.013, 0.01, 0.012]', '2026-09-01')"
        )
        # chip 族正残差 cells
        conn.execute(
            "INSERT OR REPLACE INTO memory_cells (family, motif, parent_bucket, "
            "explicit_s, explicit_f, implicit_s, implicit_f, residuals_json, updated_at) "
            "VALUES ('chip', 'window_rescale', 'high', 4.0, 1.0, 0.0, 0.0, "
            "'[0.01, 0.012, 0.011, 0.013]', '2026-09-01')"
        )
    return store


def test_recommend_edits_skips_zero_yield_family(tmp_path):
    """D3: 历史 30+ 次评估全无过线（n_pass==0）的族应被跳过，不推荐。"""
    retrieval = _create_mock_memory_db_d3(tmp_path / "d3_mem.db")
    retrieval._best_family_parent = lambda fam: {
        "parent_factor": f"{fam}_parent",
        "parent_expression": f"RANK(${fam})",
        "parent_ic": 0.025,
    }

    recs = retrieval.recommend_edits(k=2)
    fams = [r["family"] for r in recs]
    # dead 族 35 条全 rejected/weak，应被 zero_yield_fams 跳过
    assert "dead" not in fams
    # chip 族有 promising，应正常推荐
    assert "chip" in fams
