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
from typing import Sequence

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


def _column_to_arrow(values: "pd.Series") -> "pa.Array":
    """单列 → Arrow 数组，**数值列保证零 null**（零拷贝共享的前提）。

    历史坑（2026-09-10 实测定位）：此前用 ``pa.Table.from_pandas(panel_flat)``
    整表转换，pandas 的 NaN 会被当作 Arrow **null**（带 validity bitmap），
    于是 ``to_numpy(zero_copy_only=True)`` 报
    ``ArrowInvalid: Needed to copy ... with 428820 nulls``，读取路径静默回落
    到 ``zero_copy_only=False`` → **每列做一次完整私有拷贝**，多进程共享彻底失效
    （实测 shared=0.00 GiB、USS≈RSS，单进程 4.9GB）。

    改为逐列 ``pa.array(numpy_array, from_pandas=False)``：NaN 作为真实浮点值
    写入（null_count=0），``to_numpy(zero_copy_only=True)`` 返回指向 mmap 页的视图。
    """
    arr = values.to_numpy()
    kind = arr.dtype.kind
    if kind in "iufb" or kind == "M":
        return pa.array(arr, from_pandas=False)
    # 字符串/对象列（instrument 等）：Arrow 侧只能带 null，读取时必然拷贝，
    # 走 from_pandas=True 保证缺失值语义正确。
    return pa.array(values.to_numpy(dtype=object), from_pandas=True)


def write_panel_arrow(arrow_path: Path, panel_flat: pd.DataFrame) -> bool:
    """把平表（datetime/instrument 两列 + 数值列，MultiIndex reset 后的形态）
    写成未压缩 Arrow IPC 文件。失败返回 False（不阻断 parquet 主缓存）。

    panel_flat 要求：``datetime`` 列为 datetime64、``instrument`` 为字符串、
    其余列数值（与 cnequity._save_cached_panel 的 reset_index 产物一致）。
    """
    if not _ARROW_AVAILABLE:
        return False
    try:
        names = list(panel_flat.columns)
        arrays = [_column_to_arrow(panel_flat[name]) for name in names]
        schema = pa.schema([pa.field(n, a.type) for n, a in zip(names, arrays)])
        numeric_nulls = sum(
            a.null_count for a, n in zip(arrays, names) if panel_flat[n].dtype.kind in "iufb"
        )
        if numeric_nulls:
            logger.warning(
                "panel arrow 写入：数值列仍含 %d 个 null（将退化为私有拷贝，共享失效）",
                numeric_nulls,
            )
        batch = pa.record_batch(arrays, schema=schema)
        arrow_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(arrow_path.parent), prefix=".panel.", suffix=".arrow.tmp")
        os.close(fd)
        try:
            # 直接流式写文件：此前用 BufferOutputStream + getvalue() 会把整份
            # arrow（全量面板 4.4GB）在内存里再物化一遍，构建峰值多出 ~1.5-4GB。
            with pa.OSFile(tmp, "wb") as sink, ipc.new_file(sink, schema) as writer:
                writer.write_batch(batch)
            os.replace(tmp, arrow_path)
        except BaseException:  # noqa: BLE001
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        logger.info(
            "panel arrow 缓存写入完成: %s (%d 列, %d 行, 数值列 null=%d)",
            arrow_path.name, len(names), len(panel_flat), numeric_nulls,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("panel arrow 缓存写入失败（不影响 parquet 主缓存）: %s", exc)
        return False


def read_panel_arrow_mmap(
    arrow_path: Path,
    *,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame | None:
    """memory-map 读取 arrow 缓存 → pandas panel（MultiIndex）。

    数值列零拷贝共享物理页；失败返回 None（调用方回退 parquet 路径）。

    ``columns``：只保留这些数据列（None = 全部）。用于数据面聚焦时按需取列，
    未选中的列不会构造 numpy 视图（也不会为此 fault 对应文件页）。
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

        keep = None if columns is None else {str(c) for c in columns}
        data: dict[str, np.ndarray] = {}
        copied = 0
        for field in table.schema:
            name = field.name
            if name in _INDEX_COLS:
                continue
            if keep is not None and name not in keep:
                continue
            col = table.column(name).combine_chunks()
            try:
                arr = col.to_numpy(zero_copy_only=True)
            except (pa.ArrowInvalid, pl_lib.ArrowInvalid):
                arr = col.to_numpy(zero_copy_only=False)
                copied += 1
            data[name] = arr

        panel = pd.DataFrame(data, index=index, copy=False)
        logger.info(
            "CNE adapter: panel 命中 arrow mmap 缓存 %s (%d 行, %d/%d 列零拷贝共享)",
            arrow_path.name, len(panel), len(data) - copied, len(data),
        )
        return panel
    except Exception as exc:  # noqa: BLE001
        logger.warning("panel arrow mmap 读取失败（回退 parquet）: %s", exc)
        return None
