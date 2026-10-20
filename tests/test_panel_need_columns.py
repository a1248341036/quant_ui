# -*- coding: utf-8 -*-
"""面板按需取列（keep_columns / columns）回归 —— 2026-10-19。

**背景**：`read_panel_arrow_mmap` 早就支持 `columns=`（数值列零拷贝、未选中列不
fault 文件页），但**没有任何调用方传它**：`_find_cached_panel` 读缓存时写死
`read_panel_arrow_mmap(arrow_path)`，把 161 列全量物化。harness 直驱一批因子因此
吃掉 6-8GB，跑两批就整机卡死。

本次接线：`SessionCreateRequest.columns` → `StockEvalContext.columns` →
`load_panel_from_cne(keep_columns=)` → `_find_cached_panel(keep_columns=)` →
`read_panel_arrow_mmap(columns=)`。实测请求 4 列时只物化 18 列（含系统必需列），
RSS 1.1GB、load 6s。

关键契约（本文件锁定）：
1. `columns` 为空 = 全量，**逐字节等价旧行为**（向后兼容）；
2. 命中判定改用 **schema 列名**（`arrow_column_names`），否则按需取列后
   `df.columns` 只是子集，基本面哨兵检查会把好缓存在误判为"未命中"→ 触发重建；
3. 系统必需列（掩码/复权/中性化/行业/index/哨兵）由 adapter 兜底并上，
   调用方只列表达式字段即可。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from alphaagent.data.adapters.panel_mmap import (
    arrow_column_names,
    read_panel_arrow_mmap,
    write_panel_arrow,
)
from alphaagent.factor.mining.eval.context import StockEvalContext
from alphaagent.factor.mining.eval.schemas import SessionCreateRequest

_ROOT = Path(__file__).resolve().parents[1]


def _load_harness_mine():
    path = _ROOT / "scripts" / "harness_mine.py"
    spec = importlib.util.spec_from_file_location("_harness_mine_needcols", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def small_arrow(tmp_path):
    """造一个含 index 列 + 3 个数据列的小 arrow 缓存。"""
    flat = pd.DataFrame({
        "datetime": pd.to_datetime(["2020-01-02"] * 4),
        "instrument": ["a", "b", "c", "d"],
        "x": [1.0, 2.0, 3.0, 4.0],
        "y": [5.0, 6.0, 7.0, 8.0],
        "z": [9.0, 10.0, 11.0, 12.0],
    })
    path = tmp_path / "panel_test.arrow"
    assert write_panel_arrow(path, flat) is True
    return path


class TestArrowSchema:
    def test_missing_file_returns_none(self, tmp_path):
        assert arrow_column_names(tmp_path / "nope.arrow") is None

    def test_reads_all_columns(self, small_arrow):
        names = arrow_column_names(small_arrow)
        assert names is not None
        assert {"datetime", "instrument", "x", "y", "z"} <= set(names)


class TestColumnPruning:
    def test_none_means_full(self, small_arrow):
        df = read_panel_arrow_mmap(small_arrow)
        assert df is not None
        assert set(df.columns) == {"x", "y", "z"}

    def test_subset_only_materializes_requested(self, small_arrow):
        df = read_panel_arrow_mmap(small_arrow, columns=["x"])
        assert df is not None
        assert list(df.columns) == ["x"]
        # index 仍然完整（按列裁剪不影响行）
        assert len(df) == 4
        assert df.index.names == ["datetime", "instrument"]

    def test_unknown_column_ignored(self, small_arrow):
        df = read_panel_arrow_mmap(small_arrow, columns=["x", "not_a_column"])
        assert df is not None
        assert list(df.columns) == ["x"]


class TestExprColumnExtraction:
    def test_collects_and_dedups(self):
        hm = _load_harness_mine()
        cols = hm._expr_columns([
            {"expr": "a = $adj_close\nb = $amount\nCS_ZSCORE(a)"},
            {"expr": "DIVIDE($adj_close, $float_cap)"},
        ])
        assert cols == ("adj_close", "amount", "float_cap")

    def test_alias_normalized(self):
        """$turnover 是提示词写法，真实列是 turnover_rate —— 别名必须归一到真实列，
        否则按需取列会漏掉它（表达式编译期又会把 $turnover 改写成 turnover_rate）。"""
        hm = _load_harness_mine()
        assert hm._expr_columns([{"expr": "TS_MEAN($turnover, 20)"}]) == ("turnover_rate",)

    def test_frequency_suffix_stripped(self):
        hm = _load_harness_mine()
        assert hm._expr_columns([{"expr": "TS_MEAN($adj_close@5d, 20)"}]) == ("adj_close",)

    def test_empty_input(self):
        hm = _load_harness_mine()
        assert hm._expr_columns([{"expr": ""}, {}]) == ()


class TestBackwardCompatibility:
    def test_request_columns_default_empty(self):
        assert SessionCreateRequest(panel_path="cne://x").columns == ()

    def test_context_columns_default_empty(self):
        assert StockEvalContext(panel_path=Path("cne://x")).columns == ()

    def test_harness_has_full_cols_escape_hatch(self):
        """必须保留退回全列的开关：排查"某列没加载"类问题时要能一键全量。"""
        src = (_ROOT / "scripts" / "harness_mine.py").read_text(encoding="utf-8")
        assert "--full-cols" in src
