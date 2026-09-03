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


def test_two_process_physical_sharing(tmp_path: Path):
    """物理页共享验收：子进程 A attach并触碰全部页后，进程 B attach 触碰时
    系统可用内存下降应远小于文件体积（独立拷贝则 ≈ 文件体积）。"""
    # ~170MB 文件：独立拷贝 drop≈170MB vs 共享 drop≈<30MB，远超系统噪声
    flat = _sample_flat(n_days=2000, n_insts=3000)
    arrow = tmp_path / "panel_v3_share.arrow"
    assert write_panel_arrow(arrow, flat)
    file_mb = arrow.stat().st_size / 1e6
    assert file_mb > 100, f"test fixture too small: {file_mb:.0f}MB"

    child = r"""
import sys
import psutil
from pathlib import Path
from alphaagent.data.adapters.panel_mmap import read_panel_arrow_mmap

path = Path(sys.argv[1])
before = psutil.virtual_memory().available
panel = read_panel_arrow_mmap(path)
assert panel is not None
# 触碰全部数值页（强制换入物理内存）
total = 0.0
for col in panel.columns:
    total += float(panel[col].sum())
after = psutil.virtual_memory().available
drop_mb = (before - after) / 1e6
print(f"CHILD_OK {drop_mb:.1f} {total:.3f}")
"""
    # 子进程 A：首次 attach + 触碰（建立共享物理页）
    env_proc = subprocess.run(
        [sys.executable, "-c", child, str(arrow)],
        capture_output=True, text=True, timeout=300,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert env_proc.returncode == 0, env_proc.stderr[-500:]
    assert "CHILD_OK" in env_proc.stdout

    # 本测试进程（B）：attach + 触碰，量测 available 下降（中位数降噪）
    before = _available_mb()
    panel = read_panel_arrow_mmap(arrow)
    assert panel is not None
    for col in panel.columns:
        _ = float(panel[col].sum())
    after = _available_mb()
    drop_mb = before - after

    # 共享生效：B 触碰全部页后系统可用内存下降应远小于文件体积的 1/2
    # （若独立拷贝，drop ≈ file_mb；共享时只有 index/结构开销 ≪ file_mb/2）
    assert drop_mb < file_mb / 2, (
        f"physical sharing failed: drop={drop_mb:.1f}MB file={file_mb:.1f}MB"
    )
