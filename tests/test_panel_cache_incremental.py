# -*- coding: utf-8 -*-
"""面板缓存增量补尾/补头测试（不依赖真实数据湖）。

用假 registry 替代插件构建，验证：
  尾部补齐：替换接缝 warmup 天、保留更早的行、全程无重复日期、升序
  头部补齐：替换接缝 warmup 天、保留更晚的行
  缺失占比过大 → 放弃增量（返回 None）
  两头都缺 → 放弃增量
  列集合不一致 → 放弃增量
"""
import json

import numpy as np
import pandas as pd
import pytest

from alphaagent.data.adapters import cnequity as cne


class FakeRegistry:
    """按区间生成确定性面板：close = 日期序号，便于验证替换边界。"""

    def __init__(self, cols=4, codes=("000001", "600000")):
        self.cols = cols
        self.codes = codes
        self.calls = []

    def build_panel(self, *, start=None, end=None, universe_mask=False,
                    include_fundamentals=False, asset_type="stock", include_columns=None):
        self.calls.append((start, end))
        dts = pd.bdate_range(start, end)
        idx = pd.MultiIndex.from_product([dts, list(self.codes)],
                                         names=["datetime", "instrument"])
        day_no = pd.Series(dts, index=dts).rank().astype(float)
        data = {f"f{i}": np.repeat(day_no.to_numpy(), len(self.codes)) + i
                for i in range(self.cols)}
        return pd.DataFrame(data, index=idx)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(cne, "_CACHE_ROOT", tmp_path)
    monkeypatch.setattr(cne, "_ARROW_MMAP_WRITE", False)
    reg = FakeRegistry()
    return tmp_path, reg


def _write_cache(root, reg, start, end, include_fundamentals=False):
    panel = reg.build_panel(start=start, end=end)
    cne._enrich_trade_flags(panel)
    sig = cne._facet_signature(None)
    path = cne._cache_path(start, include_fundamentals, sig)
    cne._save_cached_panel(path, panel, meta={
        "schema": cne._CACHE_SCHEMA_VERSION, "signature": sig,
        "focus_facets": [], "include_fundamentals": include_fundamentals,
        "board_filter": cne._board_filter_flag(), "start": start, "end": end,
        "columns": int(panel.shape[1]), "rows": int(len(panel)),
    })
    return path, panel


def test_find_extendable_detects_tail(env):
    root, reg = env
    _write_cache(root, reg, "2026-01-01", "2026-06-30")
    ext = cne._find_extendable_cache("2026-01-01", "2026-07-10", False, None)
    assert ext is not None
    path, cmin, cmax, need_head, need_tail = ext
    assert need_tail is True and need_head is False
    assert pd.Timestamp(cmax) == pd.Timestamp("2026-06-30")


def test_find_extendable_detects_head(env):
    root, reg = env
    _write_cache(root, reg, "2026-03-02", "2026-06-30")
    ext = cne._find_extendable_cache("2026-01-05", "2026-06-30", False, None)
    assert ext is not None
    _, cmin, _, need_head, need_tail = ext
    assert need_head is True and need_tail is False
    assert pd.Timestamp(cmin) == pd.Timestamp("2026-03-02")


def test_extend_tail_appends_without_touching_cache(env):
    root, reg = env
    path, cached = _write_cache(root, reg, "2026-01-01", "2026-06-30")
    ext = cne._find_extendable_cache("2026-01-01", "2026-07-10", False, None)
    _, cmin, cmax, need_head, need_tail = ext
    merged = cne._extend_cached_panel(
        path=path, cov_min=cmin, cov_max=cmax, need_head=need_head, need_tail=need_tail,
        req_start=pd.Timestamp("2026-01-01"), req_end=pd.Timestamp("2026-07-10"),
        registry=reg, include_fundamentals=False)

    assert merged is not None and not merged.empty
    dts = merged.index.get_level_values("datetime")
    assert dts.max() == pd.Timestamp("2026-07-10")
    assert not merged.index.duplicated().any()
    assert dts.is_monotonic_increasing

    # 纯追加：缓存原有行必须逐值原样保留（不替换接缝）
    old = merged[merged.index.get_level_values("datetime") <= pd.Timestamp("2026-06-30")]
    pd.testing.assert_frame_equal(old[cached.columns], cached[cached.columns], check_dtype=False)
    # 新增行必须真的存在
    assert (dts > pd.Timestamp("2026-06-30")).sum() > 0

    # 缓存文件已被原子覆盖，且 meta 的 end 推进
    meta = cne._read_cache_meta(path)
    assert meta["end"] == "2026-07-10"
    assert meta["rows"] == len(merged)


def test_extend_head_replaces_seam_and_keeps_later(env):
    root, reg = env
    path, cached = _write_cache(root, reg, "2026-03-02", "2026-06-30")
    ext = cne._find_extendable_cache("2026-01-05", "2026-06-30", False, None)
    _, cmin, cmax, need_head, need_tail = ext
    merged = cne._extend_cached_panel(
        path=path, cov_min=cmin, cov_max=cmax, need_head=need_head, need_tail=need_tail,
        req_start=pd.Timestamp("2026-01-05"), req_end=pd.Timestamp("2026-06-30"),
        registry=reg, include_fundamentals=False)

    assert merged is not None
    dts = merged.index.get_level_values("datetime")
    assert dts.min() == pd.Timestamp("2026-01-05")
    assert dts.max() == pd.Timestamp("2026-06-30")
    assert not merged.index.duplicated().any()

    warm = cne._CACHE_EXTEND_WARMUP_TPD
    dates = pd.DatetimeIndex(cached.index.get_level_values("datetime").unique()).sort_values()
    cut = dates[warm] if len(dates) > warm else dates[-1]
    old = cached[cached.index.get_level_values("datetime") >= cut]
    new = merged[merged.index.get_level_values("datetime") >= cut]
    pd.testing.assert_frame_equal(old[cached.columns], new[cached.columns], check_dtype=False)


def test_extend_rejects_when_missing_ratio_too_large(env):
    root, reg = env
    path, _ = _write_cache(root, reg, "2026-01-01", "2026-01-10")
    ext = cne._find_extendable_cache("2026-01-01", "2026-12-31", False, None)
    _, cmin, cmax, need_head, need_tail = ext
    merged = cne._extend_cached_panel(
        path=path, cov_min=cmin, cov_max=cmax, need_head=need_head, need_tail=need_tail,
        req_start=pd.Timestamp("2026-01-01"), req_end=pd.Timestamp("2026-12-31"),
        registry=reg, include_fundamentals=False)
    assert merged is None


def test_extend_rejects_both_gaps(env):
    root, reg = env
    path, _ = _write_cache(root, reg, "2026-03-02", "2026-04-30")
    merged = cne._extend_cached_panel(
        path=path, cov_min=pd.Timestamp("2026-03-02"), cov_max=pd.Timestamp("2026-04-30"),
        need_head=True, need_tail=True,
        req_start=pd.Timestamp("2026-01-05"), req_end=pd.Timestamp("2026-06-30"),
        registry=reg, include_fundamentals=False)
    assert merged is None


def test_extend_rejects_on_column_mismatch(env):
    root, reg = env
    path, _ = _write_cache(root, reg, "2026-01-01", "2026-06-30")

    class Narrow(FakeRegistry):
        def build_panel(self, **kw):
            df = super().build_panel(**kw)
            return df[[c for c in df.columns if c != "f3"]]

    ext = cne._find_extendable_cache("2026-01-01", "2026-07-10", False, None)
    _, cmin, cmax, need_head, need_tail = ext
    merged = cne._extend_cached_panel(
        path=path, cov_min=cmin, cov_max=cmax, need_head=need_head, need_tail=need_tail,
        req_start=pd.Timestamp("2026-01-01"), req_end=pd.Timestamp("2026-07-10"),
        registry=Narrow(), include_fundamentals=False)
    assert merged is None
