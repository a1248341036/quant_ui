# -*- coding: utf-8 -*-
"""分块写 parquet：跨块 schema 不一致时必须自动对齐。

回归背景（2026-11-25）：分批重建 6 年面板缓存时，某个分块的构建结果里
有一列在所有分块中的类型推断不同（块 A 全 NaN → null 类型，块 B 有值 → double），
pyarrow 的 ParquetWriter 直接抛 "Table schema does not match schema used to create file"，
前 34 个分片白建。
"""
import numpy as np
import pandas as pd
import pytest

from alphaagent.data.adapters.cnequity import _write_panel_parquet_chunked


def _panel(rows: int = 120_000) -> pd.DataFrame:
    dts = pd.bdate_range("2024-01-01", periods=rows // 3)
    codes = ["000001", "000002", "600000"]
    idx = pd.MultiIndex.from_product([dts, codes], names=["datetime", "instrument"])
    rng = np.random.default_rng(11)
    data = {f"f{i}": rng.normal(size=len(idx)) for i in range(6)}
    # 一个只在前 1/3 行有值的 object 列：不同块的类型推断会不一致
    sparse = np.full(len(idx), None, dtype=object)
    sparse[: len(idx) // 3] = "abc"
    data["sparse_obj"] = sparse
    return pd.DataFrame(data, index=idx)[:rows]


def test_chunked_write_survives_schema_drift(tmp_path):
    panel = _panel()
    out = tmp_path / "p.parquet"
    # 小分块 → 必然跨越 sparse_obj 的空/非空边界
    _write_panel_parquet_chunked(panel, str(out), 20_000)

    got = pd.read_parquet(out)
    assert len(got) == len(panel)
    assert set(got.columns) >= set(panel.reset_index().columns) - {"sparse_obj"}
    # 行序与数值不变
    pd.testing.assert_series_equal(
        got["f0"].reset_index(drop=True),
        panel.reset_index()["f0"].reset_index(drop=True),
        check_dtype=False,
        check_names=False,
    )


def test_chunked_write_normal_case(tmp_path):
    panel = _panel()
    panel = panel.drop(columns=["sparse_obj"])
    out = tmp_path / "q.parquet"
    _write_panel_parquet_chunked(panel, str(out), 20_000)
    assert len(pd.read_parquet(out)) == len(panel)
