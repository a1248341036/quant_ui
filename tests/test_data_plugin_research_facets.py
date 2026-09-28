"""研报数据面插件回归：行业 PIT、公告密度/打标、财报明细 PIT（合成数据，Hermetic）。

背景：1200 道研报课题点名 industry/announcement/一致预期/财报明细等数据面，而这些列
此前在挖掘面板里一个都没有。本文件只验证 PIT 与口径，不依赖本地 CNE 数据。
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from alphaagent.data.adapters.plugins import _pitlib, announcement, financial_items, industry


def _write(root: Path, dataset: str, partition: str, df: pl.DataFrame) -> None:
    d = root / dataset / partition
    d.mkdir(parents=True, exist_ok=True)
    df.write_parquet(d / "part-merged.parquet")


def test_industry_is_pit_and_prefix_hierarchical(tmp_path, monkeypatch):
    _write(
        tmp_path, "industry_members", "as_of_date=2024-01-10",
        pl.DataFrame({
            "symbol": ["000001.SZ"], "classification_system": ["sw"], "industry_code": ["480301"],
            "industry_name": ["480301"], "as_of_date": [dt.date(2024, 1, 10)],
        }),
    )
    monkeypatch.setattr(_pitlib, "_curated_root", lambda: tmp_path)

    out = industry.load("industry_members", start="2024-01-01", end="2024-01-20")

    # 首次快照之前的行整体丢弃（PIT：快照前不可见）
    assert out["trade_date"].min() == pd.Timestamp("2024-01-10")
    assert (out["industry_sw_l1"] == 48.0).all()          # 6 位码前 2 位 = 一级行业
    assert (out["industry_sw_code"] == 480301.0).all()
    assert (out["ind_asof_days"] >= 0).all()


def test_announcement_counts_and_flags(tmp_path, monkeypatch):
    _write(
        tmp_path, "announcement_index", "announce_date=2024-01-05",
        pl.DataFrame({
            "symbol": ["000001.SZ"], "announce_date": [dt.date(2024, 1, 5)],
            "title": ["关于股东减持股份的公告"], "category": ["股东/实际控制人股份减持"],
        }),
    )
    _write(
        tmp_path, "announcement_index", "announce_date=2024-01-08",
        pl.DataFrame({
            "symbol": ["000001.SZ"], "announce_date": [dt.date(2024, 1, 8)],
            "title": ["2023年度业绩预告"], "category": ["业绩预告"],
        }),
    )
    monkeypatch.setattr(_pitlib, "_curated_root", lambda: tmp_path)

    out = announcement.load("announcement_index", start="2024-01-01", end="2024-01-12").set_index("trade_date")

    d5, d8 = out.loc["2024-01-05"], out.loc["2024-01-08"]
    assert d5["ann_cnt_5d"] == 1 and d5["ann_flag_reduce"] == 1 and d5["ann_days_since"] == 0
    # 计数逐日累计；打标在 20 日窗口内保持（01-05 的减持在 01-08 仍记为 1）
    assert d8["ann_cnt_5d"] == 2 and d8["ann_cnt_20d"] == 2
    assert d8["ann_flag_forecast"] == 1 and d8["ann_flag_reduce"] == 1
    # 首个公告之前无"距最近公告"可言（保持 NaN，不向前回填）
    assert pd.isna(out.loc["2024-01-02", "ann_days_since"])
    assert out.loc["2024-01-02", "ann_cnt_5d"] == 0


def test_financial_items_pit_anchor_is_announce_date(tmp_path, monkeypatch):
    _write(
        tmp_path, "financial_statement_items", "report_period=2023Q4",
        pl.DataFrame({
            "symbol": ["000001.SZ"], "report_period": ["2023Q4"], "statement_type": ["income"],
            "item_code": ["net_profit"], "item_value": [39_635_000_000.0],
            "announce_date": [dt.date(2024, 1, 15)],
        }),
    )
    monkeypatch.setattr(_pitlib, "_curated_root", lambda: tmp_path)

    out = financial_items.load("financial_statement_items", start="2024-01-01", end="2024-01-20")

    # PIT 锚点是公告日而非报告期期末；未公告前无行
    assert out["trade_date"].min() == pd.Timestamp("2024-01-15")
    assert out["fsi_net_profit"].iloc[0] == pytest.approx(39_635_000_000.0)
    assert out["fsi_days_since"].iloc[0] == 0
