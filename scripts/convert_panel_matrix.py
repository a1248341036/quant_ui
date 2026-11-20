# -*- coding: utf-8 -*-
"""把已有 parquet 面板缓存流式转换成行主序矩阵缓存（.matrix.*）。

为什么单独写这个脚本
--------------------
``panel_matrix.write_panel_matrix`` 需要一份**内存里的**完整 DataFrame；
而 7.95M 行 × 230 列的 parquet（2.25GB 压缩）一次性加载峰值约 15GB ——
2026-11-20 就是这么把机器压到 Available 2MB 的。

本脚本用 ``ParquetFile.iter_batches`` **流式**读：每批 20 万行（约 184MB），
逐批写入目标矩阵，峰值内存 < 1GB，代价是两遍扫描（约 1–2 分钟）。

两遍的原因：类别码需要**全局词表**，第一遍收集 instrument 与字符串列的
唯一值（顺序取 sorted，保证可复现），第二遍才编码写盘。

用法
----
    python scripts/convert_panel_matrix.py <cache.parquet> [--chunk 200000] [--force]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

GB = 1024 ** 3


def _mem_text() -> str:
    try:
        import psutil

        m = psutil.Process().memory_info()
        vm = psutil.virtual_memory()
        return (
            f"self rss={m.rss / GB:.2f}GB priv={getattr(m, 'private', 0) / GB:.2f}GB "
            f"peak={getattr(m, 'peak_wset', 0) / GB:.2f}GB | avail={vm.available / GB:.1f}GB"
        )
    except Exception:  # noqa: BLE001
        return "(psutil unavailable)"


def main() -> int:
    ap = argparse.ArgumentParser(description="parquet → matrix 流式转换")
    ap.add_argument("parquet", help="同 key 的 .parquet 缓存路径")
    ap.add_argument("--chunk", type=int, default=200_000, help="每批行数（默认 20 万）")
    ap.add_argument("--force", action="store_true", help="目标 matrix 已存在也覆盖")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    from alphaagent.data.adapters.panel_matrix import (
        _INDEX_NAMES,
        _MATRIX_SCHEMA,
        _open_memmap,
        matrix_paths,
    )

    src = Path(args.parquet)
    if not src.is_file():
        print(f"ERROR: parquet 不存在: {src}")
        return 2
    data_p, meta_p, idx_p = matrix_paths(src)
    if data_p.is_file() and not args.force:
        print(f"目标已存在（加 --force 覆盖）: {data_p.name}")
        return 0

    t0 = time.perf_counter()
    pf = pq.ParquetFile(str(src))
    names = [str(n) for n in pf.schema_arrow.names]
    n_rows = int(pf.metadata.num_rows)
    idx_cols = [c for c in _INDEX_NAMES if c in names]
    if len(idx_cols) != 2:
        print(f"ERROR: parquet 缺索引列 {_INDEX_NAMES}（实际列: {names[:6]}...）")
        return 2
    data_cols = [c for c in names if c not in _INDEX_NAMES]

    # 判定数值 / 字符串列
    types = {f.name: f.type for f in pf.schema_arrow}
    import pyarrow as pa

    numeric_cols, string_cols = [], []
    for c in data_cols:
        t = types[c]
        (numeric_cols if (pa.types.is_integer(t) or pa.types.is_floating(t)) else string_cols).append(c)

    print(f"[convert] {src.name}: rows={n_rows} numeric={len(numeric_cols)} "
          f"strings={len(string_cols)} {string_cols}", flush=True)
    print(f"[convert] {_mem_text()}", flush=True)

    # ---- 第一遍：收集全局词表 ----
    inst_uniq: set[str] = set()
    str_uniq: dict[str, set[str]] = {c: set() for c in string_cols}
    cols_scan = ["instrument"] + string_cols
    n_batches = 0
    for batch in pf.iter_batches(batch_size=args.chunk, columns=cols_scan):
        d = batch.to_pandas()
        inst_uniq.update(d["instrument"].astype(str).unique().tolist())
        for c in string_cols:
            str_uniq[c].update(d[c].dropna().astype(str).unique().tolist())
        n_batches += 1
    inst_cats = sorted(inst_uniq)
    str_cats = {c: sorted(v) for c, v in str_uniq.items()}
    inst_map = {v: i for i, v in enumerate(inst_cats)}
    str_maps = {c: {v: i for i, v in enumerate(str_cats[c])} for c in string_cols}
    print(f"[convert] 词表: instrument={len(inst_cats)} "
          f"strings={ {c: len(v) for c, v in str_cats.items()} } batches={n_batches}", flush=True)
    print(f"[convert] {_mem_text()}", flush=True)

    # ---- 第二遍：流式写矩阵与索引 ----
    data_p.parent.mkdir(parents=True, exist_ok=True)
    tmp_data = data_p.with_name(data_p.name + ".tmp")
    dt_ns = np.empty(n_rows, dtype="int64")
    inst_codes = np.empty(n_rows, dtype="int32")
    cat_codes = {c: np.empty(n_rows, dtype="int32") for c in string_cols}

    try:
        arr = _open_memmap(tmp_data, mode="w+", dtype=np.float32, shape=(n_rows, len(numeric_cols)))
        try:
            off = 0
            for batch in pf.iter_batches(batch_size=args.chunk):
                d = batch.to_pandas()
                n = len(d)
                hi = off + n
                if n and numeric_cols:
                    arr[off:hi, :] = d[numeric_cols].to_numpy(
                        dtype=np.float32, copy=False, na_value=np.nan
                    )
                dt = d["datetime"]
                if not np.issubdtype(dt.dtype, np.datetime64):
                    import pandas as pd

                    dt = pd.to_datetime(dt)
                dt_ns[off:hi] = dt.to_numpy(dtype="datetime64[ns]").view("int64")
                # .map(dict) 是 C 级映射，比逐元素 lambda 快一到两个数量级
                inst_codes[off:hi] = (
                    d["instrument"].map(inst_map).fillna(-1).to_numpy(dtype="int32")
                )
                for c in string_cols:
                    cat_codes[c][off:hi] = (
                        d[c].map(str_maps[c]).fillna(-1).to_numpy(dtype="int32")
                    )
                off = hi
                if off % (args.chunk * 5) == 0 or off == n_rows:
                    print(f"[convert]   {off}/{n_rows} {_mem_text()}", flush=True)
            if off != n_rows:
                print(f"ERROR: 行数不符 off={off} n_rows={n_rows}")
                raise RuntimeError("row count mismatch")
            arr.flush()
        finally:
            del arr
        os.replace(tmp_data, data_p)
    except BaseException:
        try:
            tmp_data.unlink()
        except OSError:
            pass
        raise

    # ---- 索引与元数据 ----
    payload: dict[str, np.ndarray] = {"dt": dt_ns, "codes": inst_codes}
    for c in string_cols:
        payload[f"cat_{c}"] = cat_codes[c]
    tmp_idx = idx_p.with_name(idx_p.name + ".tmp")
    with open(tmp_idx, "wb") as fh:
        np.savez(fh, **payload)
    os.replace(tmp_idx, idx_p)

    meta = {
        "schema": _MATRIX_SCHEMA,
        "columns": data_cols,
        "numeric_columns": numeric_cols,
        "n_rows": n_rows,
        "n_cols": len(numeric_cols),
        "dtype": "float32",
        "index_names": list(_INDEX_NAMES),
        "categories": inst_cats,
        "categorical_columns": str_cats,
    }
    tmp_meta = meta_p.with_name(meta_p.name + ".tmp")
    tmp_meta.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_meta, meta_p)

    print(f"[convert] 完成 {data_p.name} "
          f"({data_p.stat().st_size / GB:.2f}GB) idx {idx_p.stat().st_size / GB:.2f}GB "
          f"耗时 {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"[convert] {_mem_text()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
