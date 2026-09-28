"""数据插件窗口化读取与分红 PIT 展开的回归测试（2026-09-28）。

两个真实故障（整夜 run 日志实证）：

1. CNE curated 的日频数据集按 ``trade_date=YYYY-MM-DD`` 每天一个 parquet
   （融资融券 2607 个 / 估值 2608 个），``_pitlib.read_curated`` 的 512 文件守卫
   一律拒绝 → margin 插件整条不可用，面板缺 ``mgn_*`` 列并被哨兵校验拒绝落盘
   （每次 run 重建面板）。
2. dividend 插件在 ``with_columns`` 里用 ``pl.date_range(..., eager=True)``
   （非元素级 API，polars 1.43 下先在空 frame 上求值），必然抛
   ``ColumnNotFoundError: unable to find column "imp_ann_date"; valid columns: []``
   → ``div_*`` 整族缺失。
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from alphaagent.data.adapters.plugins import _pitlib, dividend


def _write_partition(root: Path, dataset: str, day: str, rows: int = 2) -> None:
    d = root / dataset / f"trade_date={day}"
    d.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "symbol": [f"{i:06d}.SZ" for i in range(rows)],
            "trade_date": [dt.date.fromisoformat(day)] * rows,
            "margin_balance": [1.0] * rows,
        }
    ).write_parquet(d / "part-merged.parquet")


def test_read_curated_prunes_date_partitions(tmp_path, monkeypatch):
    for day in ("2024-01-02", "2024-01-03", "2030-01-02"):
        _write_partition(tmp_path, "margin_trading", day)
    monkeypatch.setattr(_pitlib, "_curated_root", lambda: tmp_path)

    out = _pitlib.read_curated("margin_trading", start="2024-01-01", end="2024-01-31")

    assert out.height == 4
    assert sorted(out["trade_date"].unique().to_list()) == [
        dt.date(2024, 1, 2),
        dt.date(2024, 1, 3),
    ]


def test_read_curated_streams_when_window_exceeds_eager_cap(tmp_path, monkeypatch):
    """裁剪后仍超上限：走 scan 流式读取而不是整包拒绝。"""
    for day in ("2024-01-02", "2024-01-03", "2024-01-04", "2030-01-02"):
        _write_partition(tmp_path, "margin_trading", day)
    monkeypatch.setattr(_pitlib, "_curated_root", lambda: tmp_path)
    monkeypatch.setattr(_pitlib, "_MAX_EAGER_FILES", 2)

    out = _pitlib.read_curated("margin_trading", start="2024-01-01", end="2024-01-31")

    assert out.height == 6  # 窗口内 3 档 × 2 行；窗口外 2030 档不读
    assert out["trade_date"].max() == dt.date(2024, 1, 4)


def test_read_curated_without_window_keeps_guard(tmp_path, monkeypatch):
    for day in ("2024-01-02", "2024-01-03", "2024-01-04"):
        _write_partition(tmp_path, "margin_trading", day)
    monkeypatch.setattr(_pitlib, "_curated_root", lambda: tmp_path)
    monkeypatch.setattr(_pitlib, "_MAX_EAGER_FILES", 2)

    with pytest.raises(ValueError, match="文件数异常"):
        _pitlib.read_curated("margin_trading")


def test_read_curated_empty_window_is_explicit(tmp_path, monkeypatch):
    _write_partition(tmp_path, "margin_trading", "2024-01-02")
    monkeypatch.setattr(_pitlib, "_curated_root", lambda: tmp_path)

    with pytest.raises(ValueError, match="无分区数据"):
        _pitlib.read_curated("margin_trading", start="2030-01-01", end="2030-12-31")


def test_dividend_load_expands_ex_window(tmp_path, monkeypatch):
    """分红实施事件的 window 展开：deps 里 pl.date_ranges 回归点。"""
    pg = tmp_path / "dividend.parquet"
    pl.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "end_date": [dt.date(2023, 12, 31)],
            "div_proc": ["实施"],
            "cash_div": [0.5],
            "imp_ann_date": [dt.date(2024, 3, 1)],
            "ex_date": [dt.date(2024, 3, 5)],
        }
    ).write_parquet(pg)
    monkeypatch.setattr(dividend, "_DIVIDEND_PG", pg)

    out = dividend.load("dividend", start="2024-02-01", end="2024-03-31")

    assert isinstance(out, pd.DataFrame)
    assert {"div_cash_div", "div_days_since_ann", "div_days_to_ex"} <= set(out.columns)
    # 首事件前的行被丢弃（PIT：公告前不可见）
    assert out["trade_date"].min() == pd.Timestamp("2024-03-01")

    on_ann = out[out["trade_date"] == pd.Timestamp("2024-03-01")].iloc[0]
    assert on_ann["div_days_since_ann"] == 0
    assert on_ann["div_days_to_ex"] == 4
    assert on_ann["div_cash_div"] == pytest.approx(0.5)

    # [实施公告日, 除息日) 内倒计时；除息日起为空
    before_ex = out[out["trade_date"] == pd.Timestamp("2024-03-04")].iloc[0]
    assert before_ex["div_days_to_ex"] == 1
    after_ex = out[out["trade_date"] == pd.Timestamp("2024-03-05")].iloc[0]
    assert pd.isna(after_ex["div_days_to_ex"])
