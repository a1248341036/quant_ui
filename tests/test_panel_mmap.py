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
