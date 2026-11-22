# -*- coding: utf-8 -*-
"""面板缓存分块写测试：产物与整份写等价，且分块边界不丢行/不串列。

覆盖 2026-11-22 内存事故修复（重建峰值 16GB → 面板本体 + 一个分块）。
"""
import numpy as np
import pandas as pd
import pytest

from alphaagent.data.adapters.cnequity import _write_panel_parquet_chunked
from alphaagent.data.adapters.panel_mmap import write_panel_arrow_chunked


def _panel(rows: int = 120_000, cols: int = 12) -> pd.DataFrame:
    dts = pd.bdate_range("2024-01-01", periods=rows // 3)
    codes = ["000001", "000002", "600000"]
    idx = pd.MultiIndex.from_product([dts, codes], names=["datetime", "instrument"])
    rng = np.random.default_rng(7)
    data = {f"f{i}": rng.normal(size=len(idx)).astype("float64") for i in range(cols)}
    # 尾部 20% 行注入 NaN，模拟真实面板稀疏
    for k in data:
        data[k][int(len(idx) * 0.8):] = np.nan
    # 只在第一个分块有值的列（检验分块 dtype 退化后的 cast 路径）
    first_only = np.full(len(idx), np.nan)
    first_only[:1000] = 1.5
    data["first_chunk_only"] = first_only
    data["int_flag"] = np.zeros(len(idx), dtype="int8")
    df = pd.DataFrame(data, index=idx)
    return df.iloc[:rows]


@pytest.mark.parametrize("chunk_rows", [10_000, 50_000])
def test_parquet_chunked_matches_full(tmp_path, chunk_rows):
    panel = _panel()
    out = tmp_path / "p.parquet"
    _write_panel_parquet_chunked(panel, str(out), chunk_rows)
    got = pd.read_parquet(out)
    want = panel.reset_index()
    assert list(got.columns) == list(want.columns)
    assert len(got) == len(want)
    pd.testing.assert_frame_equal(
        got.reset_index(drop=True), want.reset_index(drop=True), check_dtype=False
    )


@pytest.mark.parametrize("chunk_rows", [10_000, 50_000])
def test_arrow_chunked_matches_full(tmp_path, chunk_rows):
    panel = _panel()
    out = tmp_path / "p.arrow"
    assert write_panel_arrow_chunked(out, panel, chunk_rows) is True

    import pyarrow as pa
    with pa.memory_map(str(out), "r") as source:
        table = pa.ipc.open_file(source).read_all()
    got = table.to_pandas()
    assert len(got) == len(panel)
    assert set(got.columns) == set(panel.reset_index().columns)
    want = panel.reset_index()
    pd.testing.assert_series_equal(
        got["f0"].reset_index(drop=True),
        want["f0"].reset_index(drop=True),
        check_dtype=False,
        check_names=False,
    )


def test_chunked_write_single_chunk(tmp_path):
    """行数小于块大小时只写一个 row group，产物仍完整。"""
    panel = _panel(rows=3_000)
    out = tmp_path / "small.parquet"
    _write_panel_parquet_chunked(panel, str(out), 400_000)
    got = pd.read_parquet(out)
    assert len(got) == len(panel)
