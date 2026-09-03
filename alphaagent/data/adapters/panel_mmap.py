# -*- coding: utf-8 -*-
"""panel 只读内存映射共享层（跨进程物理页共享）。

解决多会话（多挖掘 run 子进程）各自完整加载 3.6GB panel 的内存膨胀：
同一 (start, end, include_fundamentals) 缓存数据被 N 个进程并发使用时，
parquet 解压后的 pandas DataFrame 是每进程独立拷贝（N × 3.6GB）；
本模块提供 Arrow IPC（未压缩）+ ``pa.memory_map`` 的只读映射路径——
数值列 numpy 数组零拷贝指向同一物理页（OS page cache 多进程共享），
第 2..N 个进程 attach 成本 ≈ 秒级索引构建，无解压。

内存语义
--------
- 数值列（float/int）：``Array.to_numpy(zero_copy_only=True)`` → numpy 视图
  指向 mmap 物理页，N 个进程共享同一份物理内存；
- datetime 列：转 ns 精度构建 DatetimeIndex（一次性小拷贝 ~64MB/进程）；
- instrument 字符串列：object 化每进程独立（~0.4GB/进程，pandas 字符串
  无法零拷贝）——大头数值列已共享；
- 文件生命周期：mmap 只读，Windows 下被映射期间缓存淘汰（unlink）会失败
  → 沿用 cnequity._purge_old_cache 的 try/except 兜底（警告 + 跳过）。

写入语义
--------
``write_panel_arrow()`` 由 cnequity 缓存层在首次构建 panel 落盘时顺带调用：
parquet（压缩、人可查）+ arrow（未压缩、可 mmap）双格式共存，arrow 写入
失败不影响 parquet 主缓存。
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_INDEX_COLS = ["datetime", "instrument"]

try:
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import pyarrow.lib as pl_lib

    _ARROW_AVAILABLE = True
except Exception:  # noqa: BLE001 — pyarrow 缺失时回退 parquet 路径
    _ARROW_AVAILABLE = False


def is_available() -> bool:
    return _ARROW_AVAILABLE


def write_panel_arrow(arrow_path: Path, panel_flat: pd.DataFrame) -> bool:
    """把平表（datetime/instrument 两列 + 数值列，MultiIndex reset 后的形态）
    写成未压缩 Arrow IPC 文件。失败返回 False（不阻断 parquet 主缓存）。

    panel_flat 要求：``datetime`` 列为 datetime64、``instrument`` 为字符串、
    其余列数值（与 cnequity._save_cached_panel 的 reset_index 产物一致）。
    """
    if not _ARROW_AVAILABLE:
        return False
    try:
        table = pa.Table.from_pandas(panel_flat, preserve_index=False)
        sink = pa.BufferOutputStream()
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
        buf = sink.getvalue()
        arrow_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(arrow_path.parent), prefix=".panel.", suffix=".arrow.tmp")
        os.close(fd)
        try:
            with open(tmp, "wb") as fh:
                fh.write(buf)
            os.replace(tmp, arrow_path)
        except BaseException:  # noqa: BLE001
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("panel arrow 缓存写入失败（不影响 parquet 主缓存）: %s", exc)
        return False


def read_panel_arrow_mmap(arrow_path: Path) -> pd.DataFrame | None:
    """memory-map 读取 arrow 缓存 → pandas panel（MultiIndex）。

    数值列零拷贝共享物理页；失败返回 None（调用方回退 parquet 路径）。
    """
    if not _ARROW_AVAILABLE or not arrow_path.is_file():
        return None
    try:
        source = pa.memory_map(str(arrow_path), "r")
        table = ipc.open_file(source).read_all()

        dt_col = table.column("datetime").combine_chunks()
        dt_values = dt_col.to_numpy(zero_copy_only=False).astype("datetime64[ns]")
        inst_col = table.column("instrument").combine_chunks()
        inst_values = inst_col.to_numpy(zero_copy_only=False)
        index = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex(dt_values), pd.Index(inst_values, dtype="object")],
            names=_INDEX_COLS,
        )

        data: dict[str, np.ndarray] = {}
        for field in table.schema:
            name = field.name
            if name in _INDEX_COLS:
                continue
            col = table.column(name).combine_chunks()
            try:
                arr = col.to_numpy(zero_copy_only=True)
            except (pa.ArrowInvalid, pl_lib.ArrowInvalid):
                arr = col.to_numpy(zero_copy_only=False)
            data[name] = arr

        panel = pd.DataFrame(data, index=index, copy=False)
        logger.info(
            "CNE adapter: panel 命中 arrow mmap 缓存 %s (%d 行, 数值列零拷贝共享)",
            arrow_path.name, len(panel),
        )
        return panel
    except Exception as exc:  # noqa: BLE001
        logger.warning("panel arrow mmap 读取失败（回退 parquet）: %s", exc)
        return None
