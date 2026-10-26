# -*- coding: utf-8 -*-
"""B-01 修复测试：include_fundamentals 降级加载 + 全量重建护栏。

覆盖 docs/specs/alphaagent_known_bugs_spec.md B-01：
1. session 层降级——include_fundamentals=True 但请求列均不含 funda_* 时，
   实际按 include_fundamentals=False 调 CNE adapter（可命中 false 版缓存，
   避免 8GB 全量重建），并在 session.meta 记 fundamentals_demoted=True；
2. 请求列含 funda_* 或未给请求列时不降级（保持 True，不猜需求）；
3. cnequity._warn_panel_rebuild_cost 护栏：有历史缓存规模 → WARNING 带预估
   GB；无缓存 → WARNING 首次重建提示；内部异常绝不抛出。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from alphaagent.data.adapters import cnequity
from alphaagent.factor.mining.context import StockEvalContext
from alphaagent.factor.mining.eval.session import SessionStore


def _panel_df(n_days: int = 6, n_insts: int = 5) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    insts = [f"S{i:03d}" for i in range(n_insts)]
    idx = pd.MultiIndex.from_product([dates, insts], names=["datetime", "instrument"])
    rng = np.random.default_rng(7)
    n = len(idx)
    return pd.DataFrame(
        {
            "close": rng.normal(10, 1, n).astype(np.float32),
            "adj_close": rng.normal(10, 1, n).astype(np.float32),
            "volume": rng.integers(1e5, 1e6, n).astype(np.float64),
            "label_1d_open_to_open": rng.normal(0, 0.01, n).astype(np.float32),
        },
        index=idx,
    )


def _cne_ctx(columns: tuple[str, ...] = ()) -> StockEvalContext:
    # test_end 显式固定，避免动态解析数据源最新交易日（测试环境无 CNE 数据集）
    return StockEvalContext(
        panel_path="cne://",
        include_fundamentals=True,
        columns=columns,
        test_end="2024-06-30",
    )


def test_session_demotes_fundamentals_when_no_funda_requested(monkeypatch):
    captured: dict = {}

    def fake_load(**kw):
        captured.update(kw)
        return _panel_df()

    monkeypatch.setattr(cnequity, "is_cne_source", lambda p: True)
    monkeypatch.setattr(cnequity, "load_panel_from_cne", fake_load)

    session = SessionStore().create(_cne_ctx(columns=("close", "volume")))

    assert captured["include_fundamentals"] is False
    assert session.meta["fundamentals_demoted"] is True
    # 请求列语义不变：表达式列与 label 列照常传下去
    keep = captured["keep_columns"] or ()
    assert "close" in keep
    assert "label_1d_open_to_open" in keep


def test_session_keeps_fundamentals_when_funda_requested(monkeypatch):
    captured: dict = {}

    def fake_load(**kw):
        captured.update(kw)
        return _panel_df()

    monkeypatch.setattr(cnequity, "is_cne_source", lambda p: True)
    monkeypatch.setattr(cnequity, "load_panel_from_cne", fake_load)

    session = SessionStore().create(_cne_ctx(columns=("close", "funda_ocf")))

    assert captured["include_fundamentals"] is True
    assert session.meta["fundamentals_demoted"] is False


def test_session_keeps_fundamentals_without_columns(monkeypatch):
    """无请求列（全量请求）时不降级——表达式是否引用 funda_* 未知，不猜。"""
    captured: dict = {}

    def fake_load(**kw):
        captured.update(kw)
        return _panel_df()

    monkeypatch.setattr(cnequity, "is_cne_source", lambda p: True)
    monkeypatch.setattr(cnequity, "load_panel_from_cne", fake_load)

    session = SessionStore().create(_cne_ctx(columns=()))

    assert captured["include_fundamentals"] is True
    assert session.meta["fundamentals_demoted"] is False


def test_rebuild_guard_warns_with_estimate(tmp_path: Path, caplog):
    # 伪造缓存目录与 meta：400 万行 × 1300 列（历史全量面板量级）
    pq = tmp_path / "panel_v11_fake_all.parquet"
    pq.write_bytes(b"x")
    meta = {
        "rows": 4_000_000,
        "columns": 1300,
        "signature": "all",
        "include_fundamentals": False,
        "board_filter": cnequity._board_filter_flag(),
    }
    cnequity._meta_path(pq).write_text(json.dumps(meta), encoding="utf-8")
    with mock.patch.object(cnequity, "_CACHE_ROOT", tmp_path):
        with caplog.at_level(logging.WARNING, logger="alphaagent.data.adapters.cnequity"):
            cnequity._warn_panel_rebuild_cost(
                start="2024-01-01", include_fundamentals=True, focus=[]
            )
    assert any("全量重建" in r.message and "GB" in r.message for r in caplog.records)


def test_rebuild_guard_warns_first_build(tmp_path: Path, caplog):
    empty = tmp_path / "empty"
    empty.mkdir()
    with mock.patch.object(cnequity, "_CACHE_ROOT", empty):
        with caplog.at_level(logging.WARNING, logger="alphaagent.data.adapters.cnequity"):
            cnequity._warn_panel_rebuild_cost(
                start=None, include_fundamentals=True, focus=["业绩面"]
            )
    assert any("首次全量重建" in r.message for r in caplog.records)


def test_rebuild_guard_never_raises(tmp_path: Path):
    class Boom:
        def is_dir(self):  # noqa: ANN001
            raise RuntimeError("boom")

    with mock.patch.object(cnequity, "_CACHE_ROOT", Boom()):
        # 内部异常必须被吞掉（护栏绝不阻塞主路径）——不抛即过
        cnequity._warn_panel_rebuild_cost(
            start=None, include_fundamentals=False, focus=[]
        )
