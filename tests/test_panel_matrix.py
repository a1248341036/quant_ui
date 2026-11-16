# -*- coding: utf-8 -*-
"""行主序矩阵面板缓存（panel_matrix）单元测试。"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from alphaagent.data.adapters.panel_matrix import (
    META_SUFFIX,
    MATRIX_SUFFIX,
    INDEX_SUFFIX,
    matrix_column_names,
    matrix_paths,
    read_panel_matrix,
    remove_panel_matrix,
    write_panel_matrix,
)


def _make_panel(n_days: int = 30, n_inst: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D").repeat(n_inst)
    insts = np.tile([f"{i:06d}" for i in range(n_inst)], n_days)
    index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex(dates), pd.Index(insts, dtype="object")],
        names=["datetime", "instrument"],
    )
    data = {
        "ret": rng.standard_normal(len(index)).astype("float32"),
        "close": (rng.random(len(index)) * 100).astype("float32"),
        "is_trade": np.ones(len(index), dtype="int8"),
        "not_st": np.ones(len(index), dtype="int8"),
    }
    return pd.DataFrame(data, index=index, copy=False)


def _cache_path(tmp_path) -> "object":
    # 用同 key 的 parquet 路径做锚点（只用于推导文件名，不要求文件存在）
    return tmp_path / "panel_v11_testkey_all.parquet"


class TestPaths:
    def test_matrix_paths_derived_from_parquet(self, tmp_path):
        data, meta, idx = matrix_paths(_cache_path(tmp_path))
        assert data.name == "panel_v11_testkey_all" + MATRIX_SUFFIX
        assert meta.name == "panel_v11_testkey_all" + META_SUFFIX
        assert idx.name == "panel_v11_testkey_all" + INDEX_SUFFIX

    def test_matrix_paths_from_arrow_same_key(self, tmp_path):
        p = _cache_path(tmp_path)
        assert matrix_paths(p) == matrix_paths(p.with_suffix(".arrow"))


class TestRoundTrip:
    def test_roundtrip_values_and_index(self, tmp_path):
        src = _make_panel()
        assert write_panel_matrix(_cache_path(tmp_path), src) is True

        out = read_panel_matrix(_cache_path(tmp_path))
        assert out is not None
        assert list(out.columns) == list(src.columns)
        assert out.index.names == src.index.names
        assert len(out) == len(src)
        # 索引逐元素一致（顺序敏感）
        pd.testing.assert_index_equal(out.index, src.index)
        # 数值一致（统一 float32）
        for col in src.columns:
            np.testing.assert_allclose(
                out[col].to_numpy(), src[col].to_numpy(dtype="float32"), rtol=0, atol=0
            )

    def test_int8_flags_read_back_as_float32_same_values(self, tmp_path):
        src = _make_panel()
        write_panel_matrix(_cache_path(tmp_path), src)
        out = read_panel_matrix(_cache_path(tmp_path))
        assert out is not None
        # 单 block 要求同 dtype → 标记列读回 float32，但 0/1 语义不变
        assert str(out["is_trade"].dtype) == "float32"
        np.testing.assert_array_equal(out["is_trade"].to_numpy(), np.ones(len(out)))
        assert bool((out["not_st"] == 1).all())

    def test_nan_preserved(self, tmp_path):
        src = _make_panel()
        src.loc[src.index[:10], "ret"] = np.nan
        write_panel_matrix(_cache_path(tmp_path), src)
        out = read_panel_matrix(_cache_path(tmp_path))
        assert out is not None
        assert int(out["ret"].isna().sum()) == 10

    def test_column_subset(self, tmp_path):
        src = _make_panel()
        write_panel_matrix(_cache_path(tmp_path), src)
        out = read_panel_matrix(_cache_path(tmp_path), columns=["ret", "close"])
        assert out is not None
        assert list(out.columns) == ["ret", "close"]
        np.testing.assert_allclose(
            out["ret"].to_numpy(), src["ret"].to_numpy(dtype="float32"), rtol=0, atol=0
        )


class TestZeroCopy:
    def test_single_block_and_shared_backing(self, tmp_path):
        src = _make_panel()
        write_panel_matrix(_cache_path(tmp_path), src)
        out = read_panel_matrix(_cache_path(tmp_path))
        assert out is not None
        # 关键性质：pandas 侧只有 1 个数值 block（而非每列一个）
        assert len(out._mgr.blocks) == 1
        # 内存映射只读 → 不允许原地写穿缓存文件
        with pytest.raises((ValueError, TypeError)):
            out.iloc[0, 0] = 123.0

    def test_new_column_does_not_break_original_block(self, tmp_path):
        src = _make_panel()
        write_panel_matrix(_cache_path(tmp_path), src)
        out = read_panel_matrix(_cache_path(tmp_path))
        assert out is not None
        before = out["ret"].to_numpy().copy()
        out["extra_flag"] = np.ones(len(out), dtype="int8")
        assert len(out._mgr.blocks) == 2  # 原 block 仍在，新增独立 block
        np.testing.assert_array_equal(out["ret"].to_numpy(), before)


class TestStringColumns:
    """面板里真实存在字符串列（suspend_timing / suspend_type），必须无损往返且不破坏零拷贝。"""

    def _panel_with_strings(self) -> pd.DataFrame:
        src = _make_panel()
        idx = np.arange(len(src))
        src.insert(1, "suspend_type", np.where(idx % 7 == 0, "S", ""))
        src.insert(3, "suspend_timing", np.where(idx % 11 == 0, "全天", ""))
        return src

    def test_string_columns_roundtrip_and_order(self, tmp_path):
        src = self._panel_with_strings()
        assert write_panel_matrix(_cache_path(tmp_path), src) is True
        out = read_panel_matrix(_cache_path(tmp_path))
        assert out is not None
        assert list(out.columns) == list(src.columns)  # 列序完全不变
        for col in ("suspend_type", "suspend_timing"):
            assert out[col].dtype == object
            np.testing.assert_array_equal(
                out[col].to_numpy(), src[col].to_numpy(dtype=object)
            )

    def test_string_columns_keep_float_zero_copy(self, tmp_path):
        src = self._panel_with_strings()
        write_panel_matrix(_cache_path(tmp_path), src)
        out = read_panel_matrix(_cache_path(tmp_path))
        assert out is not None

        # 还原字符串列只 split 视图；数值列仍指向 memmap（未拷贝）。
        # 注意不能拿测试里二次 np.memmap 的数组去 shares_memory——那是另一个
        # 虚拟地址映射（物理页相同、地址不同），必然返回 False。改查 base 链。
        def _memmap_backed(a) -> bool:
            for _ in range(8):
                if isinstance(a, np.memmap):
                    return True
                a = getattr(a, "base", None)
                if a is None:
                    return False
            return False

        assert _memmap_backed(out["ret"].to_numpy())
        # float 部分应保持极少数 block（而非每列一个）
        float_blocks = [b for b in out._mgr.blocks if str(b.values.dtype) == "float32"]
        assert len(float_blocks) <= 4

    def test_string_column_missing_values(self, tmp_path):
        src = self._panel_with_strings()
        src.loc[src.index[:5], "suspend_type"] = np.nan
        write_panel_matrix(_cache_path(tmp_path), src)
        out = read_panel_matrix(_cache_path(tmp_path))
        assert out is not None
        assert int(out["suspend_type"].isna().sum()) == 5


class TestFallbacks:
    def test_missing_files_returns_none(self, tmp_path):
        assert read_panel_matrix(_cache_path(tmp_path)) is None

    def test_non_multiindex_rejected(self, tmp_path):
        df = pd.DataFrame({"ret": np.zeros(5, dtype="float32")})
        assert write_panel_matrix(_cache_path(tmp_path), df) is False

    def test_schema_mismatch_returns_none(self, tmp_path):
        src = _make_panel()
        write_panel_matrix(_cache_path(tmp_path), src)
        _, meta_p, _ = matrix_paths(_cache_path(tmp_path))
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        meta["schema"] = 999
        meta_p.write_text(json.dumps(meta), encoding="utf-8")
        assert read_panel_matrix(_cache_path(tmp_path)) is None

    def test_index_length_mismatch_returns_none(self, tmp_path):
        src = _make_panel()
        write_panel_matrix(_cache_path(tmp_path), src)
        _, _, idx_p = matrix_paths(_cache_path(tmp_path))
        np.savez(idx_p, dt=np.zeros(3, dtype="int64"), codes=np.zeros(3, dtype="int32"))
        assert read_panel_matrix(_cache_path(tmp_path)) is None

    def test_corrupt_meta_returns_none(self, tmp_path):
        src = _make_panel()
        write_panel_matrix(_cache_path(tmp_path), src)
        _, meta_p, _ = matrix_paths(_cache_path(tmp_path))
        meta_p.write_text("{not json", encoding="utf-8")
        assert read_panel_matrix(_cache_path(tmp_path)) is None


class TestColumnNamesAndRemoval:
    def test_matrix_column_names_without_touching_data(self, tmp_path):
        src = _make_panel()
        write_panel_matrix(_cache_path(tmp_path), src)
        cols = matrix_column_names(_cache_path(tmp_path))
        assert cols == [str(c) for c in src.columns]

    def test_column_names_none_when_absent(self, tmp_path):
        assert matrix_column_names(_cache_path(tmp_path)) is None

    def test_remove_deletes_all_three(self, tmp_path):
        src = _make_panel()
        write_panel_matrix(_cache_path(tmp_path), src)
        paths = matrix_paths(_cache_path(tmp_path))
        assert all(p.is_file() for p in paths)
        remove_panel_matrix(_cache_path(tmp_path))
        assert not any(p.exists() for p in paths)

    def test_remove_is_idempotent(self, tmp_path):
        remove_panel_matrix(_cache_path(tmp_path))  # 不存在也不报错


class TestAtomicity:
    def test_failed_write_leaves_no_partial_matrix(self, tmp_path, monkeypatch):
        src = _make_panel()

        # 让矩阵写入中途抛错，确认不留半成品（read 应返回 None）
        from alphaagent.data.adapters import panel_matrix as pm_mod

        def _boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(pm_mod, "_open_memmap", _boom)
        assert write_panel_matrix(_cache_path(tmp_path), src) is False
        monkeypatch.undo()
        assert read_panel_matrix(_cache_path(tmp_path)) is None
