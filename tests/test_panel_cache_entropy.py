# -*- coding: utf-8 -*-
"""panel 缓存熵增治理（2026-10-23）四项行为回归：

1. 缓存 key 不含 end：不同请求 end 命中同一文件（覆盖写而非每日新文件）。
2. 淘汰时 .meta.json 成对删除（孤儿 meta 修复）。
3. 磁盘配额双闸：parquet+arrow 总量超 _CACHE_MAX_GB 时最旧先淘汰。
4. arrow 副本默认不写（ALPHA_PANEL_ARROW 开关）。

背景：TEST_END_CAP（e6087ec）锁死右端后，end 仍参与缓存 key——cap 推进、
或任何窗口变化都会生成新文件，此前 9 天积出 8 个全量变体 ~40GB。
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from alphaagent.data.adapters import cnequity


def _panel(days: int = 6, codes=("000001", "000002")) -> pd.DataFrame:
    idx = pd.MultiIndex.from_product(
        [pd.bdate_range("2024-01-01", periods=days), codes],
        names=["datetime", "instrument"],
    )
    n = len(idx)
    return pd.DataFrame(
        {
            "adj_close": np.linspace(10.0, 20.0, n).astype(np.float32),
            "float_cap": np.full(n, 1e9, dtype=np.float32),
        },
        index=idx,
    )


def _write_cache(start: str, end: str, *, with_arrow=False, with_meta=True):
    """手工落一份假缓存（绕过构建路径），返回 parquet 路径。

    注意：key 不含 end，**同 start 的两次写落同一文件**——需要多个缓存文件时
    必须用不同 start。
    """
    path = cnequity._cache_path(start, False, cnequity._facet_signature(None))
    panel = _panel()
    panel.to_parquet(path)
    if with_meta:
        cnequity._meta_path(path).write_text(
            json.dumps({
                "signature": cnequity._facet_signature(None),
                "include_fundamentals": False,
                "board_filter": cnequity._board_filter_flag(),
                "rows": int(len(panel)),
                "start": start,
                "end": end,
            }),
            encoding="utf-8",
        )
    if with_arrow:
        path.with_suffix(".arrow").write_bytes(b"x" * 16)
    return path


def test_cache_key_stable_across_end(tmp_path, monkeypatch):
    """key 不含 end：不同请求右端 → 同一文件路径（覆盖写前提）。"""
    monkeypatch.setattr(cnequity, "_CACHE_ROOT", tmp_path)
    p1 = cnequity._cache_path("2024-01-01", False, "all")
    p2 = cnequity._cache_path("2024-01-01", False, "all")
    assert p1 == p2
    # 同 start 不同 end 的请求共享同一 key（旧实现产生两个文件）
    _write_cache("2024-01-01", "2024-01-05")
    _write_cache("2024-01-01", "2024-01-10")
    parquets = list(tmp_path.glob("panel_*.parquet"))
    assert len(parquets) == 1, "end 不应参与 key，两次写应落在同一文件"


def test_purge_removes_orphan_meta(tmp_path, monkeypatch):
    """淘汰时 .meta.json 必须成对删除（此前漏删留孤儿 meta）。"""
    monkeypatch.setattr(cnequity, "_CACHE_ROOT", tmp_path)
    old = _write_cache("2023-01-01", "2023-01-05", with_arrow=True)
    old_t = old.stat().st_mtime - 10_000
    os.utime(old, (old_t, old_t))          # 排到淘汰序最前
    meta_p = old.with_suffix(".meta.json")
    assert meta_p.is_file()

    monkeypatch.setattr(cnequity, "_CACHE_MAX_GB", 0.0000001)
    keep = _write_cache("2024-01-01", "2024-01-09")   # 不同 start → 不同文件
    cnequity._purge_old_cache(keep)

    assert not old.exists(), "最旧缓存应被淘汰"
    assert not old.with_suffix(".arrow").exists(), "arrow 应成对淘汰"
    assert not meta_p.exists(), "meta.json 应成对淘汰（孤儿修复）"
    assert keep.exists(), "keep 自身不淘汰"


def test_gb_quota_evicts_oldest_first(tmp_path, monkeypatch):
    """超 _CACHE_MAX_GB 时按最旧优先淘汰；新缓存（keep）保留。"""
    monkeypatch.setattr(cnequity, "_CACHE_ROOT", tmp_path)
    old1 = _write_cache("2022-01-01", "2022-01-05")
    old2 = _write_cache("2023-01-01", "2023-01-05")
    t1 = old1.stat().st_mtime - 20_000
    os.utime(old1, (t1, t1))               # old1 最旧
    t2 = old2.stat().st_mtime - 10_000
    os.utime(old2, (t2, t2))               # old2 次旧
    keep = _write_cache("2024-01-01", "2024-01-09")

    # 配额压到 ~0：keep 之外的旧缓存全部淘汰
    monkeypatch.setattr(cnequity, "_CACHE_MAX_GB", 0.0000001)
    cnequity._purge_old_cache(keep)

    assert not old1.exists() and not old2.exists()
    assert keep.exists()


def test_arrow_write_disabled_by_default(tmp_path, monkeypatch):
    """默认不写 arrow 副本（单进程场景 6.4GB 未压缩页是负债）。"""
    monkeypatch.setattr(cnequity, "_CACHE_ROOT", tmp_path)
    monkeypatch.setattr(cnequity, "_ARROW_MMAP_WRITE", False)
    panel = _panel()
    path = tmp_path / "panel_v11_test.parquet"
    cnequity._save_cached_panel(path, panel)
    assert path.is_file()
    assert not path.with_suffix(".arrow").exists(), "默认不应产生 arrow 副本"


def test_arrow_write_enabled_by_env(tmp_path, monkeypatch):
    """ALPHA_PANEL_ARROW=1（多会话并发场景）时写 arrow 副本。"""
    monkeypatch.setattr(cnequity, "_CACHE_ROOT", tmp_path)
    monkeypatch.setattr(cnequity, "_ARROW_MMAP_WRITE", True)
    panel = _panel()
    path = tmp_path / "panel_v11_test.parquet"
    cnequity._save_cached_panel(path, panel)
    assert path.is_file()
    assert path.with_suffix(".arrow").is_file()
