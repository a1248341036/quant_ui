# -*- coding: utf-8 -*-
"""panel arrow mmap 共享层测试。

覆盖：
- write→mmap 读回的值/dtype/index 一致性
- 数值列零拷贝（numpy 数组基于 mmap buffer，无独立拷贝）
- 损坏 arrow 文件回退 None（调用方走 parquet 路径）
- 跨进程物理页共享（spawn 子进程 attach 同一文件，系统可用内存下降 ≈ 页表级）
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphaagent.data.adapters.panel_mmap import (
    is_available,
    read_panel_arrow_mmap,
    write_panel_arrow,
)
from alphaagent.data.panel import load_panel, panel_columns

pytestmark = pytest.mark.skipif(
    not is_available(), reason="pyarrow unavailable"
)


def _sample_flat(n_days: int = 40, n_insts: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    insts = [f"S{i:03d}" for i in range(n_insts)]
    idx = pd.MultiIndex.from_product([dates, insts], names=["datetime", "instrument"])
    rng = np.random.default_rng(42)
    n = len(idx)
    return pd.DataFrame(
        {
            "adj_close": rng.normal(10, 1, n).astype(np.float32),
            "volume": rng.integers(1e5, 1e6, n).astype(np.float64),
            "funda_ocf": rng.normal(0, 1, n).astype(np.float32),
        },
        index=idx,
    ).reset_index()


def test_roundtrip_values_dtypes_index(tmp_path: Path):
    flat = _sample_flat()
    arrow = tmp_path / "panel_v3_test.arrow"
    assert write_panel_arrow(arrow, flat) is True

    panel = read_panel_arrow_mmap(arrow)
    assert panel is not None
    assert isinstance(panel.index, pd.MultiIndex)
    assert panel.index.names == ["datetime", "instrument"]
    assert len(panel) == len(flat)

    src = flat.set_index(["datetime", "instrument"])
    # 值逐列一致
    for col in src.columns:
        np.testing.assert_allclose(panel[col].to_numpy(), src[col].to_numpy(), rtol=1e-6)
    # dtype 保持（float32 列不被静默升为 float64）
    assert panel["adj_close"].dtype == np.float32
    assert panel["volume"].dtype == np.float64


def test_numeric_columns_are_zero_copy_views(tmp_path: Path):
    flat = _sample_flat()
    arrow = tmp_path / "panel_v3_zero.arrow"
    write_panel_arrow(arrow, flat)
    panel = read_panel_arrow_mmap(arrow)
    # 数值列 numpy 数组应是 mmap buffer 的视图：写入保护（只读 buffer）特征
    # —— 底层数据不可写（pa.memory_map("r")）
    arr = panel["adj_close"].to_numpy(copy=False)
    assert arr.flags.writeable is False


def test_corrupt_arrow_returns_none(tmp_path: Path):
    arrow = tmp_path / "panel_v3_bad.arrow"
    arrow.write_bytes(b"not-an-arrow-file" * 100)
    assert read_panel_arrow_mmap(arrow) is None


def test_missing_file_returns_none(tmp_path: Path):
    assert read_panel_arrow_mmap(tmp_path / "nope.arrow") is None


def _available_mb(samples: int = 5) -> float:
    """系统可用内存（MB），多次采样取中位数降噪。"""
    import psutil

    vals = sorted(psutil.virtual_memory().available / 1e6 for _ in range(samples))
    return vals[samples // 2]


def test_two_process_zero_copy_contract(tmp_path: Path):
    """零拷贝契约验收（两个独立进程各映射同一文件）。

    判定依据：``to_numpy(zero_copy_only=True)`` 成功 → numpy 数组是 arrow
    mmap buffer 的**只读别名**（pyarrow 契约：一旦发生拷贝结果可写，此断言即失败）。
    任一进程出现私有拷贝（如旧版 NaN→null 导致 ArrowInvalid 回落
    ``zero_copy_only=False``）都会让 ``writeable`` 变 True → 测试失败。

    旧版测试曾用「系统 available 内存下降量」判定共享——在 Windows 上不可靠：
    A 退出/工作集被修剪后其页进入 standby（仍计为 available），B 重新激活时
    available 照样下降 ≈ 文件体积，与是否共享无关，存在周期性误报。
    """
    flat = _sample_flat(n_days=2000, n_insts=3000)
    arrow = tmp_path / "panel_v5_share.arrow"
    assert write_panel_arrow(arrow, flat)
    file_mb = arrow.stat().st_size / 1e6
    assert file_mb > 100, f"test fixture too small: {file_mb:.0f}MB"

    child = r"""
import sys
from pathlib import Path
from alphaagent.data.adapters.panel_mmap import read_panel_arrow_mmap

panel = read_panel_arrow_mmap(Path(sys.argv[1]))
assert panel is not None
bad = [c for c in panel.columns if panel[c].to_numpy(copy=False).flags.writeable]
assert not bad, f"non-zero-copy columns: {bad}"
total = 0.0
for col in panel.columns:
    total += float(panel[col].sum())
print(f"CHILD_OK {total:.3f}", flush=True)
"""
    proc = subprocess.run(
        [sys.executable, "-c", child, str(arrow)],
        capture_output=True, text=True, timeout=300,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert proc.returncode == 0, proc.stderr[-500:]
    assert "CHILD_OK" in proc.stdout

    # 本测试进程（B）：独立进程再次映射，同样必须全部零拷贝
    panel = read_panel_arrow_mmap(arrow)
    assert panel is not None
    non_zero_copy = [
        c for c in panel.columns if panel[c].to_numpy(copy=False).flags.writeable
    ]
    assert not non_zero_copy, f"non-zero-copy columns: {non_zero_copy}"
    src = flat.set_index(["datetime", "instrument"])
    np.testing.assert_allclose(
        panel["adj_close"].to_numpy(), src["adj_close"].to_numpy(), rtol=1e-6
    )


def _write_sibling(flat: pd.DataFrame, tmp_path: Path, name: str):
    """写出同 key 的 .parquet + .arrow 双格式缓存（与 cnequity 落盘形态一致）。"""
    pq = tmp_path / f"{name}.parquet"
    flat.set_index(["datetime", "instrument"]).to_parquet(pq)
    arrow = pq.with_suffix(".arrow")
    assert write_panel_arrow(arrow, flat) is True
    return pq


def test_load_panel_prefers_arrow_mmap(tmp_path: Path):
    """同 key 存在 .arrow 时 load_panel 走 mmap：数值列只读（零拷贝视图）。"""
    flat = _sample_flat()
    pq = _write_sibling(flat, tmp_path, "panel_v9_prefer")
    panel = load_panel(pq)
    assert isinstance(panel.index, pd.MultiIndex)
    assert panel.index.names == ["datetime", "instrument"]
    src = flat.set_index(["datetime", "instrument"])
    for col in src.columns:
        np.testing.assert_allclose(panel[col].to_numpy(), src[col].to_numpy(), rtol=1e-6)
    assert panel["adj_close"].dtype == np.float32
    # mmap 契约：数值列是只读视图（parquet 拷贝路径是可写的）
    assert panel["adj_close"].to_numpy(copy=False).flags.writeable is False


def test_load_panel_falls_back_to_parquet_without_arrow(tmp_path: Path):
    """无 .arrow 兄弟文件时回退 pd.read_parquet（可写拷贝，行为不变）。"""
    flat = _sample_flat()
    pq = tmp_path / "panel_v9_noarrow.parquet"
    flat.set_index(["datetime", "instrument"]).to_parquet(pq)
    panel = load_panel(pq)
    src = flat.set_index(["datetime", "instrument"])
    for col in src.columns:
        np.testing.assert_allclose(panel[col].to_numpy(), src[col].to_numpy(), rtol=1e-6)
    assert panel["adj_close"].to_numpy(copy=False).flags.writeable is True


def test_load_panel_falls_back_when_arrow_corrupt(tmp_path: Path):
    """.arrow 损坏时 read_panel_arrow_mmap 返回 None → 回退 parquet，值一致。"""
    flat = _sample_flat()
    pq = _write_sibling(flat, tmp_path, "panel_v9_bad")
    pq.with_suffix(".arrow").write_bytes(b"not-an-arrow-file" * 200)
    panel = load_panel(pq)
    src = flat.set_index(["datetime", "instrument"])
    np.testing.assert_allclose(
        panel["adj_close"].to_numpy(), src["adj_close"].to_numpy(), rtol=1e-6
    )


def test_load_panel_columns_subset(tmp_path: Path):
    """columns= 只返回选定列：mmap 路径过滤构造，索引与 dtype 保持。"""
    flat = _sample_flat()
    pq = _write_sibling(flat, tmp_path, "panel_v9_cols")
    panel = load_panel(pq, columns=["adj_close"])
    assert list(panel.columns) == ["adj_close"]
    assert panel.index.names == ["datetime", "instrument"]
    src = flat.set_index(["datetime", "instrument"])
    np.testing.assert_allclose(
        panel["adj_close"].to_numpy(), src["adj_close"].to_numpy(), rtol=1e-6
    )
    assert panel["adj_close"].dtype == np.float32
    # 仍是 mmap 零拷贝视图（未被整表 drop 复制）
    assert panel["adj_close"].to_numpy(copy=False).flags.writeable is False


def test_panel_columns_metadata(tmp_path: Path):
    """panel_columns 元数据级取全量列名（含索引列），不物化数据。"""
    flat = _sample_flat()
    pq = _write_sibling(flat, tmp_path, "panel_v9_meta")
    cols = panel_columns(pq)
    assert cols is not None
    assert set(cols) == {"datetime", "instrument", "adj_close", "volume", "funda_ocf"}


def test_load_panel_columns_parquet_fallback_keeps_index(tmp_path: Path):
    """columns= 落到 parquet 回退路径：reset 列形态补索引列、index 形态不误补，
    MultiIndex 都不被丢。"""
    flat = _sample_flat()
    src = flat.set_index(["datetime", "instrument"])
    # 形态 1：datetime/instrument 是普通列（CNE 缓存 reset_index 落盘形态）
    pq = tmp_path / "panel_v9_cols_reset.parquet"
    flat.to_parquet(pq)
    panel = load_panel(pq, columns=["adj_close"])
    assert list(panel.columns) == ["adj_close"]
    assert panel.index.names == ["datetime", "instrument"]
    np.testing.assert_allclose(
        panel["adj_close"].to_numpy(), src["adj_close"].to_numpy(), rtol=1e-6
    )
    # 形态 2：datetime/instrument 是 parquet index
    pq2 = tmp_path / "panel_v9_cols_index.parquet"
    src.to_parquet(pq2)
    panel2 = load_panel(pq2, columns=["adj_close"])
    assert list(panel2.columns) == ["adj_close"]
    assert panel2.index.names == ["datetime", "instrument"]
    np.testing.assert_allclose(
        panel2["adj_close"].to_numpy(), src["adj_close"].to_numpy(), rtol=1e-6
    )


def test_load_panel_falls_back_when_arrow_missing_columns(tmp_path: Path):
    """arrow 是旧 schema（缺请求列）时回退 parquet，避免静默缺列。"""
    flat = _sample_flat()
    pq = _write_sibling(flat, tmp_path, "panel_v9_stale")
    # 覆盖 arrow 成旧 schema：只剩 datetime/instrument/adj_close
    stale = flat[["datetime", "instrument", "adj_close"]]
    assert write_panel_arrow(pq.with_suffix(".arrow"), stale) is True
    panel = load_panel(pq, columns=["adj_close", "volume"])
    assert sorted(panel.columns) == ["adj_close", "volume"]
    assert panel.index.names == ["datetime", "instrument"]
    src = flat.set_index(["datetime", "instrument"])
    np.testing.assert_allclose(
        panel["volume"].to_numpy(), src["volume"].to_numpy(), rtol=1e-6
    )


def test_load_panel_sorts_when_arrow_index_not_fully_ordered(tmp_path: Path):
    """arrow 内 datetime 单调但 (datetime, instrument) 非全序时退化排序，与 parquet
    路径行序语义一致。"""
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    insts = [f"S{i:03d}" for i in range(4)]
    rng = np.random.default_rng(7)
    rows = []
    for d in dates:
        order = list(insts)
        rng.shuffle(order)
        for i, inst in enumerate(order):
            rows.append(
                {"datetime": d, "instrument": inst, "adj_close": float(i)}
            )
    flat = pd.DataFrame(rows)  # datetime 层单调，但同日 instrument 乱序
    pq = tmp_path / "panel_v9_unsorted.parquet"
    flat.set_index(["datetime", "instrument"]).to_parquet(pq)
    write_panel_arrow(pq.with_suffix(".arrow"), flat)

    panel = load_panel(pq)
    assert panel.index.is_monotonic_increasing  # 退化 sort_index 已全序
    ref = flat.set_index(["datetime", "instrument"]).sort_index()
    np.testing.assert_allclose(
        panel["adj_close"].to_numpy(), ref["adj_close"].to_numpy(), rtol=1e-6
    )
