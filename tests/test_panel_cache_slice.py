# -*- coding: utf-8 -*-
"""panel 磁盘缓存命中裁剪测试：宽缓存命中必须按请求区间切片返回。

2026-10-08 实锤回归：tuning 请求 end=2024-12-31 命中 blind_test 构建的
[2020-01-01 ~ 2026-09-30] 宽缓存后整个返回（命中语义"缓存覆盖 ⊇ 请求"但
返回不裁剪），walk-forward 折铺到 2026-09，OOS 消费 2025+ 盲测段——
"盲测物理隔离"宣称失效。修复：命中后按请求 [start, end] 闭区间裁剪。
"""

import json

import numpy as np
import pandas as pd

from alphaagent.data.adapters import cnequity


def _make_panel(dates, codes=("000001", "000002", "000003")):
    idx = pd.MultiIndex.from_product(
        [pd.DatetimeIndex(dates), codes], names=["datetime", "instrument"]
    )
    n = len(idx)
    return pd.DataFrame(
        {
            "adj_close": np.linspace(10.0, 20.0, n).astype(np.float32),
            "float_cap": np.full(n, 1e9, dtype=np.float32),
        },
        index=idx,
    )


def test_wide_cache_hit_is_sliced_to_requested_range(tmp_path, monkeypatch):
    """请求 [2024-01-01 ~ 2024-01-15] 命中覆盖到 2025-01 的宽缓存：
    返回必须裁到请求区间，2025 段（模拟盲测段）不得回流。"""
    wide = _make_panel(
        list(pd.bdate_range("2024-01-01", periods=25))
        + list(pd.bdate_range("2025-01-02", periods=6))
    )
    monkeypatch.setattr(cnequity, "_CACHE_ROOT", tmp_path)
    req_start, req_end = "2024-01-01", "2024-01-15"
    path = cnequity._cache_path(req_start, req_end, False, cnequity._facet_signature(None))
    wide.to_parquet(path)
    meta = {
        "signature": cnequity._facet_signature(None),
        "include_fundamentals": False,
        "board_filter": cnequity._board_filter_flag(),
        "rows": int(len(wide)),
        "start": "2024-01-01",
        "end": "2025-01-09",
    }
    cnequity._meta_path(path).write_text(json.dumps(meta), encoding="utf-8")

    df = cnequity.load_panel_from_cne(
        start=req_start, end=req_end, include_fundamentals=False
    )

    dts = df.index.get_level_values("datetime")
    assert dts.min() >= pd.Timestamp(req_start)
    assert dts.max() <= pd.Timestamp(req_end)
    assert dts.max() < pd.Timestamp("2025-01-01")
