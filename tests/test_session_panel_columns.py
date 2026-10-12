# -*- coding: utf-8 -*-
"""评估会话按需取列加载测试（内存修复配套）。

``include_fundamentals=False`` 时 `session.create` 应通过 ``load_panel(columns=...)``
直接排除 ``funda_*`` 列，而不是整表 mmap/parquet 后再 ``drop`` —— drop(columns=...)
对数值块整块复制 ~6GB 私有拷贝（pandas 未开 CoW 实测），是批量技术档内存 16-17GB 的
又一来源。此测试断言：会话面板数值列仍是 mmap 只读视图（未被 drop 复制）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphaagent.data.adapters.panel_mmap import is_available, write_panel_arrow
from alphaagent.factor.mining.context import StockEvalContext
from alphaagent.factor.mining.eval.session import SessionStore

pytestmark = pytest.mark.skipif(not is_available(), reason="pyarrow unavailable")


def _flat(n_days: int = 40, n_insts: int = 30) -> pd.DataFrame:
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


def test_session_no_fundamentals_loads_subset_no_drop_copy(tmp_path: Path):
    flat = _flat()
    pq = tmp_path / "panel_v9_cols.parquet"
    flat.to_parquet(pq)  # reset_index 列形态（与 CNE 缓存一致）
    write_panel_arrow(pq.with_suffix(".arrow"), flat)

    ctx = StockEvalContext(panel_path=pq, include_fundamentals=False)
    session = SessionStore().create(ctx)

    # 会话面板不含 funda_* 列,按需取列路径如实记录排除列数（本夹具 1 个 funda_ocf）
    assert not any(str(c).startswith("funda_") for c in session.panel.columns)
    assert session.meta["dropped_fundamental_cols"] == 1
    # 数值列仍是 mmap 零拷贝只读视图 —— 若发生了 drop 整块复制，这里会是可写
    arr = session.panel["adj_close"].to_numpy(copy=False)
    assert arr.flags.writeable is False
    np.testing.assert_allclose(
        arr, flat.set_index(["datetime", "instrument"])["adj_close"].to_numpy(), rtol=1e-6
    )
