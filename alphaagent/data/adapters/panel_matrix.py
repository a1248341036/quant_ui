# -*- coding: utf-8 -*-
"""panel 行主序矩阵缓存（``.npy`` memmap，**单 block 真零拷贝**）。

为什么不用 arrow
----------------
`panel_mmap.read_panel_arrow_mmap` 逐列 ``combine_chunks().to_numpy(zero_copy_only=True)``，
2026-11-14 实测：**每列仍产生一份私有拷贝**（29.7MB = 7,782,511 行 × 4B），
230 列累计 6.83GB 私有内存；同时 mmap 页被 fault（6.76GB RSS）——
**私有副本与 mmap 页双份并存**，加载期峰值 14.5GB（7.78M 行 × 230 列面板）。
多进程共享只在"页"层面成立，每个进程依旧完整拷一份。

本模块的做法
------------
把面板落成**行主序 float32 连续矩阵**（``.npy``），读取时 ``np.memmap`` +
``pd.DataFrame(arr, columns=..., index=..., copy=False)``。已实测（2026-11-14）：

- pandas 侧得到 **1 个 block**（``_mgr.blocks`` 长度 1），``shares_memory`` 为 True；
- ``df[col]`` 单列取值、``df.iloc[lo:hi]`` 行切片（``slice_panel`` 用的就是它）都是**视图**；
- **新增列不触发整块重排**（原 block 仍与文件共享物理页）——
  这正是 ``_enrich_trade_flags`` 写 ``is_trade/is_st/not_st`` 时需要的性质；
- 私有内存 ≈ index 部分（datetime int64 + instrument codes）+ 实际被访问列的页。

裸 ``.npy`` 而不用 ``.arrow`` 的原因：renders arrow 的列式缓冲是 230 段独立内存，
pandas 无法把它们拼成单 block（只能逐列持有 → 正是历史拷贝的来源）；
而行主序连续矩阵天然就是一个 block。

落盘三件套（与 parquet / arrow **同 key 共存**，互不覆盖）
---------------------------------------------------------
``<stem>.matrix.npy``      float32, shape=(n_rows, n_cols)，C 序（行主序）
``<stem>.matrix.json``     列名顺序 / 行数 / 索引层名 / instrument 词表等元数据
``<stem>.matrix.idx.npz``  索引层：``dt``(int64 纳秒) + ``codes``(int32 → 词表下标)

设计约束
--------
1. 所有列必须是数值列（面板本身全是 float32/int8），统一按 float32 落盘——
   单 block 的前提是同 dtype。``int8`` 标记列（is_trade/is_st/not_st）读回变 float32，
   数值语义等价（下游用 ``== 0`` / ``== 1`` 判定，DSL 位运算先 ``astype(bool)``）。
2. 返回的 DataFrame **只读**（numpy memmap ``mode="r"``）：任何原地写会立刻抛错，
   而不是静默写穿缓存文件。加列不受影响。
3. 任何一步失败都返回 ``None`` / ``False``，由调用方回退 arrow / parquet，绝不阻断主路径。
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MATRIX_SUFFIX = ".matrix.npy"
META_SUFFIX = ".matrix.json"
INDEX_SUFFIX = ".matrix.idx.npz"

# 分块写入的行数：200_000 × 230 列 × 4B ≈ 184MB 临时块，
# 避免"一次性把整表 to_numpy() 再写"导致额外 6.8GB 峰值
_CHUNK_ROWS = 200_000

# matrix 元数据版本（与 cnequity._CACHE_SCHEMA_VERSION 独立演进）
_MATRIX_SCHEMA = 1

_INDEX_NAMES = ("datetime", "instrument")


def _open_memmap(path: Path, *, mode: str, dtype: Any, shape: tuple[int, int]) -> np.memmap:
    """裸 memmap（**不是** ``np.lib.format.open_memmap``）。

    踩坑记录（2026-11-14 单测暴露）：写入用 ``np.lib.format.open_memmap`` 会写带
    128 字节 header 的 ``.npy``，而读取侧用裸 ``np.memmap`` 从文件首字节按数据解释 →
    整个矩阵错位（读出 1e19 量级垃圾值）。两侧必须用同一种裸格式；
    shape/dtype/列名由旁注 json 提供，不需要 ``.npy`` 自描述。
    """
    return np.memmap(path, dtype=dtype, mode=mode, shape=shape)


def matrix_paths(cache_path: Path | str) -> tuple[Path, Path, Path]:
    """由同 key 的 parquet / arrow 缓存路径推导 matrix 三件套路径。

    ``panel_v11_<key>_all.parquet`` → ``panel_v11_<key>_all.matrix.{npy,json,idx.npz}``
    """
    p = Path(cache_path)
    base = p.with_suffix("")  # 去掉 .parquet / .arrow
    stem = base.name
    return (
        base.parent / f"{stem}{MATRIX_SUFFIX}",
        base.parent / f"{stem}{META_SUFFIX}",
        base.parent / f"{stem}{INDEX_SUFFIX}",
    )


def is_available() -> bool:
    """matrix 路径只依赖 numpy，恒可用（保留接口以对齐 panel_mmap）。"""
    return True


def _read_meta(cache_path: Path | str) -> dict[str, Any] | None:
    _, meta_p, _ = matrix_paths(cache_path)
    if not meta_p.is_file():
        return None
    try:
        data = json.loads(meta_p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 — 元数据损坏按未命中处理
        logger.warning("panel matrix 元数据读取失败 %s: %s", meta_p.name, exc)
        return None
    return data if isinstance(data, dict) else None


def matrix_column_names(cache_path: Path | str) -> list[str] | None:
    """只读 json 取列名，**不触碰任何数值数据**。

    对齐 ``panel_mmap.arrow_column_names`` 的用途：缓存命中判定（基本面哨兵、开关一致性、
    请求列是否齐全）只需要列名，不应为此物化数 GB 的数值列。
    """
    meta = _read_meta(cache_path)
    if meta is None:
        return None
    cols = meta.get("columns")
    if not isinstance(cols, list):
        return None
    return [str(c) for c in cols]


def write_panel_matrix(cache_path: Path | str, panel: pd.DataFrame) -> bool:
    """把面板落成行主序 float32 矩阵三件套。失败返回 False（不阻断 parquet 主缓存）。

    ``cache_path`` 是**同 key 的 parquet 路径**（只用于推导文件名，不读写它）。
    """
    try:
        if not isinstance(panel.index, pd.MultiIndex):
            logger.warning("panel matrix 写入跳过：索引不是 MultiIndex")
            return False
        names = list(panel.index.names)
        if _INDEX_NAMES[0] not in names or _INDEX_NAMES[1] not in names:
            logger.warning(
                "panel matrix 写入跳过：索引缺 %s 层（实际 %s）", list(_INDEX_NAMES), names
            )
            return False

        cols = [str(c) for c in panel.columns]
        # 非数值列（实际存在：``suspend_timing`` / ``suspend_type``，停牌类型字符串）
        # 以**类别码 float32** 存进矩阵对应列位，读取时还原成 object 列。
        # 为什么不用"排除该列 + 读取时从 parquet 补"：这两列位于列序中段（38/40），
        # 补列后重排会对 6.8GB 数值块整块拷贝。编码成码位则列序与列集完全不变，
        # 且实测替换中段列只会把单 block **split 成视图**，数值列仍与文件共享（零拷贝）。
        categorical: dict[str, list[str]] = {}
        cat_codes: dict[str, np.ndarray] = {}
        for c in panel.columns:
            col = panel[c]
            if pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
                continue
            codes, cats = pd.factorize(np.asarray(col, dtype=object))
            categorical[str(c)] = [str(x) for x in cats]
            cat_codes[str(c)] = codes.astype(np.int64, copy=False)

        numeric_cols = [c for c in cols if c not in categorical]
        if not numeric_cols:
            logger.warning("panel matrix 写入跳过：没有可写入的数值列")
            return False

        n_rows, n_cols = int(len(panel)), len(cols)
        if n_rows == 0 or n_cols == 0:
            return False

        data_p, meta_p, idx_p = matrix_paths(cache_path)
        data_p.parent.mkdir(parents=True, exist_ok=True)

        # ---- ① 数值矩阵（分块写，避免整表一次性物化）----
        numeric_pos = np.array(
            [i for i, c in enumerate(cols) if c not in categorical], dtype=np.intp
        )
        cat_pos = {name: cols.index(name) for name in cat_codes}
        tmp_data = data_p.with_name(data_p.name + ".tmp")
        try:
            arr = _open_memmap(tmp_data, mode="w+", dtype=np.float32, shape=(n_rows, n_cols))
            try:
                for lo in range(0, n_rows, _CHUNK_ROWS):
                    hi = min(lo + _CHUNK_ROWS, n_rows)
                    block = panel.iloc[lo:hi]
                    vals = block[numeric_cols].to_numpy(
                        dtype=np.float32, copy=False, na_value=np.nan
                    )
                    arr[lo:hi, numeric_pos] = vals
                    for name, codes in cat_codes.items():
                        arr[lo:hi, cat_pos[name]] = codes[lo:hi].astype(np.float32)
                arr.flush()
            finally:
                del arr  # Windows 下必须先关句柄才能 rename
            os.replace(tmp_data, data_p)
        except BaseException:
            try:
                tmp_data.unlink()
            except OSError:
                pass
            raise

        # ---- ② 索引层（datetime → int64 ns；instrument → int32 codes + 词表）----
        dt_idx = pd.DatetimeIndex(panel.index.get_level_values(_INDEX_NAMES[0]))
        dt_ns = dt_idx.to_numpy(dtype="datetime64[ns]").view("int64")

        inst_values = panel.index.get_level_values(_INDEX_NAMES[1])
        codes, categories = pd.factorize(np.asarray(inst_values, dtype=object))
        if codes.min(initial=0) < 0:
            # factorize 遇到缺失值会给 -1；索引层不应有缺失，出现即放弃矩阵路径
            logger.warning("panel matrix 写入跳过：instrument 索引含缺失值")
            try:
                data_p.unlink()
            except OSError:
                pass
            return False
        codes_i32 = codes.astype(np.int32, copy=False)

        tmp_idx = idx_p.with_name(idx_p.name + ".tmp")
        try:
            with open(tmp_idx, "wb") as fh:
                np.savez(fh, dt=dt_ns, codes=codes_i32)
            os.replace(tmp_idx, idx_p)
        except BaseException:
            try:
                tmp_idx.unlink()
            except OSError:
                pass
            raise

        # ---- ③ 元数据（列名顺序 + 词表；只读列名时不必碰数据）----
        meta = {
            "schema": _MATRIX_SCHEMA,
            "columns": cols,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "dtype": "float32",
            "index_names": list(_INDEX_NAMES),
            "categories": [str(c) for c in categories],
            # 字符串列 → 类别词表（矩阵里存的是码位，读回需还原为 object）
            "categorical_columns": categorical,
        }
        tmp_meta = meta_p.with_name(meta_p.name + ".tmp")
        tmp_meta.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_meta, meta_p)

        logger.info(
            "panel matrix 写入完成: %s (%d 行, %d 列, float32 行主序 %.2fGB)",
            data_p.name, n_rows, n_cols, n_rows * n_cols * 4 / 1024 ** 3,
        )
        return True
    except Exception as exc:  # noqa: BLE001 — 落盘失败不影响 parquet 主缓存
        logger.warning("panel matrix 写入失败（不影响 parquet 主缓存）: %s", exc)
        return False


def read_panel_matrix(
    cache_path: Path | str,
    *,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame | None:
    """memory-map 读取矩阵缓存 → pandas panel（MultiIndex）。

    成功返回**只读** DataFrame（数值列零拷贝指向 mmap 页，单 block）；失败返回 None
    （调用方回退 arrow / parquet）。

    ``columns``：只保留这些数据列（None = 全部）。单 block 上做列选择会产生一次
    选中列的拷贝（选中列数 × 29.7MB）——仍远小于 arrow 路径的全列私有拷贝。
    """
    data_p, meta_p, idx_p = matrix_paths(cache_path)
    if not (data_p.is_file() and meta_p.is_file() and idx_p.is_file()):
        return None

    meta = _read_meta(cache_path)
    if meta is None:
        return None

    try:
        if int(meta.get("schema", 0)) != _MATRIX_SCHEMA:
            logger.info(
                "panel matrix 版本不符（%s != %s），按未命中处理: %s",
                meta.get("schema"), _MATRIX_SCHEMA, data_p.name,
            )
            return None

        cols = [str(c) for c in meta["columns"]]
        n_rows = int(meta["n_rows"])
        n_cols = int(meta["n_cols"])
        if len(cols) != n_cols or n_rows <= 0 or n_cols <= 0:
            return None

        arr = _open_memmap(data_p, mode="r", dtype=np.float32, shape=(n_rows, n_cols))

        with np.load(idx_p) as npz:
            dt_ns = npz["dt"]
            codes = npz["codes"]
        if dt_ns.shape[0] != n_rows or codes.shape[0] != n_rows:
            logger.warning(
                "panel matrix 索引长度与矩阵不符（dt=%d codes=%d rows=%d），按未命中处理",
                dt_ns.shape[0], codes.shape[0], n_rows,
            )
            return None

        dt_index = pd.DatetimeIndex(dt_ns.view("datetime64[ns]"))
        categories = [str(c) for c in (meta.get("categories") or [])]
        if not categories:
            return None
        # take 出 object 数组（指针层复用词表里的 str 对象，不重复分配字符串）
        inst_values = np.asarray(categories, dtype=object)[codes]

        index = pd.MultiIndex.from_arrays(
            [dt_index, pd.Index(inst_values, dtype="object")],
            names=list(meta.get("index_names") or _INDEX_NAMES),
        )

        panel = pd.DataFrame(arr, columns=cols, index=index, copy=False)

        # 字符串列还原：矩阵里存的是类别码（float32），读回替换成 object 列。
        # 实测替换中段列只把单 block split 成视图，数值列仍与 mmap 页共享（零拷贝）。
        cat_meta = meta.get("categorical_columns") or {}
        for name, cats in cat_meta.items():
            if name not in panel.columns or not cats:
                continue
            j = cols.index(name)
            codes_f = arr[:, j]
            ci = np.rint(codes_f).astype(np.int64, copy=False)
            bad = ~np.isfinite(codes_f) | (ci < 0)
            restored = np.empty(n_rows, dtype=object)
            good = ~bad
            if good.any():
                restored[good] = np.asarray(cats, dtype=object)[ci[good]]
            if bad.any():
                restored[bad] = np.nan
            panel[name] = restored

        keep = None if columns is None else {str(c) for c in columns}
        if keep is not None:
            selected = [c for c in cols if c in keep]
            if selected and len(selected) < n_cols:
                # 显式 copy=True：memmap 是跨列连续的行主序，横向切列无法返回视图，
                # 与其让 pandas 隐式拷贝，不如写明语义（选中列通常只有几列）
                panel = panel[selected].copy()

        logger.info(
            "panel matrix 命中 mmap 缓存 %s (%d 行, %d 列, 单 block 零拷贝)",
            data_p.name, len(panel), panel.shape[1],
        )
        return panel
    except Exception as exc:  # noqa: BLE001 — 读取失败一律回退
        logger.warning("panel matrix 读取失败（回退 arrow/parquet）: %s", exc)
        return None


def remove_panel_matrix(cache_path: Path | str) -> None:
    """删除 matrix 三件套（缓存淘汰时调用；单个失败不阻断）。"""
    for p in matrix_paths(cache_path):
        if p.is_file():
            try:
                p.unlink()
            except OSError as exc:
                logger.warning("panel matrix 附属文件删除失败: %s (%s)", p.name, exc)
