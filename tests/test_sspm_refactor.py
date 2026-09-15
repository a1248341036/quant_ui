# -*- coding: utf-8 -*-
"""SSPM 编辑统计层重构验证测试 (Tests for SSPM Refactor Spec)."""

from __future__ import annotations

import json
import sqlite3
import pytest
from pathlib import Path

from alphaagent.dsl.core.ast import classify_family_coarse
from alphaagent.factor.mining.memory.expressions import extract_edit_motif
from alphaagent.factor.mining.memory import ResearchMemoryStore


def test_sspm_coarse_family_mapping():
    """验证细族与融合族键正确聚合为 9 大粗族。"""
    # A1. 细族映射
    assert classify_family_coarse("momentum") == "momentum_reversal"
    assert classify_family_coarse("reversal") == "momentum_reversal"
    assert classify_family_coarse("gap_overnight") == "momentum_reversal"
    assert classify_family_coarse("volatility") == "volatility"
    assert classify_family_coarse("kline") == "volatility"
    assert classify_family_coarse("volume") == "volume_liquidity"
    assert classify_family_coarse("liquidity") == "volume_liquidity"
    assert classify_family_coarse("correlation") == "correlation"
    assert classify_family_coarse("chip") == "chip"
    assert classify_family_coarse("fundamental") == "fundamental"
    assert classify_family_coarse("gated") == "interaction"
    assert classify_family_coarse("piecewise") == "interaction"
    assert classify_family_coarse("vwap") == "vwap"

    # A2. 融合族键（最慢信息面主导）
    assert classify_family_coarse("价量面×基本面") == "fundamental"
    assert classify_family_coarse("价量面×股东面") == "fundamental"
    assert classify_family_coarse("价量面×筹码面") == "chip"
    assert classify_family_coarse("筹码面×事件面") == "chip"
    assert classify_family_coarse("价量面×资金面") == "interaction"
    assert classify_family_coarse("行情内部跨面", facets=["价量面", "量能面"], expression="TS_PCTCHANGE($close, 5)") == "momentum_reversal"


def test_sspm_motif_refinement():
    """验证 motif 细化逻辑（窗口长短/算子类型/修饰细类）。"""
    # 窗口延长 vs 缩短
    p_expr = "RANK(TS_MEAN($close, 20))"
    c_extend = "RANK(TS_MEAN($close, 40))"
    c_shorten = "RANK(TS_MEAN($close, 10))"
    assert extract_edit_motif(p_expr, c_extend) == "window_extend"
    assert extract_edit_motif(p_expr, c_shorten) == "window_shorten"

    # 字段替换细化
    c_funda = "RANK(TS_MEAN($funda_roe_ttm, 20))"
    c_chip = "RANK(TS_MEAN($chip_peak_loc, 20))"
    c_price = "RANK(TS_MEAN($volume, 20))"
    assert extract_edit_motif(p_expr, c_funda) == "feature_swap_funda"
    assert extract_edit_motif(p_expr, c_chip) == "feature_swap_chip"
    assert extract_edit_motif(p_expr, c_price) == "feature_swap_price"

    # 算子替换细化
    c_smooth_swap = "RANK(WMA($close, 20))"
    c_trans_swap = "CS_ZSCORE(TS_MEAN($close, 20))"
    assert extract_edit_motif(p_expr, c_smooth_swap) == "operator_swap_smoothing"
    assert extract_edit_motif(p_expr, c_trans_swap) == "operator_swap_transform"

    # 外层修饰细化
    base = "TS_MEAN($close, 20)"
    add_norm = "RANK(TS_MEAN($close, 20))"
    add_smooth = "EMA(TS_MEAN($close, 20), 5)"
    add_neut = "CS_NEUTRALIZE(TS_MEAN($close, 20))"
    assert extract_edit_motif(base, add_norm) == "composition_add_normalize"
    assert extract_edit_motif(base, add_smooth) == "composition_add_smoothing"
    assert extract_edit_motif(base, add_neut) == "composition_add_neutralize"


def test_sspm_technical_error_skipped(tmp_path: Path):
    """改造 C 测试：纯技术性报错不计入 cell 失败观测，不污染失败率。"""
    store = ResearchMemoryStore(tmp_path / "m.db")
    parent_expr = "RANK(TS_MEAN($close, 20))"
    child_expr = "RANK(TS_MEAN($close, 40))"

    # 1. 存入父本
    p_row = {
        "name": "eval_on_train_set",
        "arguments_raw": json.dumps({"multi_line_expr": parent_expr, "factor_name": "p_factor"}),
        "result": {"ok": True, "split": "train", "summary": {"ic": 0.025}},
    }
    store.record_tool_result(run_id="r1", row=p_row)
    store.flush_writes()

    # 2. 存入一个技术性报错子代 (例如 dsl_compile_failed)
    c_row_tech_err = {
        "name": "eval_on_train_set",
        "arguments_raw": json.dumps({
            "multi_line_expr": child_expr,
            "factor_name": "c_factor",
            "parent_factor": "p_factor",
            "edit_note": "edit=window_extend 20->40",
        }),
        "result": {
            "ok": False,
            "error": "dsl_compile_failed: syntax error on line 1",
            "split": "train",
        },
    }
    store.record_tool_result(run_id="r1", row=c_row_tech_err)
    store.flush_writes()

    # 验证 cells 表中没有生成因该技术失败记账的失败观测
    with store._open() as conn:
        cell = conn.execute("SELECT explicit_f FROM memory_cells").fetchone()
        # 由于技术错误跳过，cell 应不存在或者 explicit_f == 0
        assert cell is None or cell["explicit_f"] == 0.0


def test_sspm_actionable_injection(tmp_path: Path):
    """改造 D 测试：否决先验行包含具体替代方向指导，而非抽象口号。"""
    store = ResearchMemoryStore(tmp_path / "m.db")
    # 模拟在 cells 表写入一条高置信度失败 cell 和一条同族高胜率 cell
    with store._open() as conn:
        # momentum_reversal 下 window_shorten 全败
        conn.execute(
            """
            INSERT INTO memory_cells (family, motif, parent_bucket, explicit_s, explicit_f, residuals_json, updated_at)
            VALUES ('momentum_reversal', 'window_shorten', 'medium', 0.0, 5.0, '[0.01, 0.02, 0.01, 0.02, 0.01]', '2026-09-15')
            """
        )
        # momentum_reversal 下 operator_swap_smoothing 胜率较高
        conn.execute(
            """
            INSERT INTO memory_cells (family, motif, parent_bucket, explicit_s, explicit_f, residuals_json, updated_at)
            VALUES ('momentum_reversal', 'operator_swap_smoothing', 'medium', 4.0, 1.0, '[0.02, 0.02, 0.03, 0.02, 0.02]', '2026-09-15')
            """
        )

    block = store._edit_prior_block()
    assert block != ""
    assert "替代方向" in block
    assert "更换平滑算子" in block
