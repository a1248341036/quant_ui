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

from alphaagent.data.adapters.plugins import (
    _pitlib,
    announcement,
    financial_items,
    index_membership,
    industry,
    research_views,
)


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


def test_industry_uses_pre_window_snapshot(tmp_path, monkeypatch):
    """R1 回归：窗口起点之前的最近快照必须当初始状态，否则窗口头部缺行业值。"""
    _write(tmp_path, "industry_members", "as_of_date=2023-12-29", pl.DataFrame({
        "symbol": ["000001.SZ"], "classification_system": ["sw"], "industry_code": ["480301"],
        "industry_name": ["480301"], "as_of_date": [dt.date(2023, 12, 29)],
    }))
    _write(tmp_path, "industry_members", "as_of_date=2024-01-31", pl.DataFrame({
        "symbol": ["000001.SZ"], "classification_system": ["sw"], "industry_code": ["630802"],
        "industry_name": ["630802"], "as_of_date": [dt.date(2024, 1, 31)],
    }))
    monkeypatch.setattr(_pitlib, "_curated_root", lambda: tmp_path)

    out = industry.load("industry_members", start="2024-01-02", end="2024-01-31")

    assert out["trade_date"].min() == pd.Timestamp("2024-01-02")          # 用 2023-12-29 快照播种
    assert (out[out["trade_date"] < "2024-01-31"]["industry_sw_l1"] == 48.0).all()
    assert (out[out["trade_date"] == "2024-01-31"]["industry_sw_l1"] == 63.0).all()  # 新快照生效


def test_announcement_counts_include_pre_window_events(tmp_path, monkeypatch):
    """R1 回归：窗口起点之前的公告要计入窗口首日的 5d/20d 与 ann_days_since。"""
    _write(tmp_path, "announcement_index", "announce_date=2023-12-28", pl.DataFrame({
        "symbol": ["000001.SZ"], "announce_date": [dt.date(2023, 12, 28)],
        "title": ["关于股东减持股份的公告"], "category": ["股东/实际控制人股份减持"],
    }))
    monkeypatch.setattr(_pitlib, "_curated_root", lambda: tmp_path)

    out = announcement.load("announcement_index", start="2024-01-02", end="2024-01-12").set_index("trade_date")

    d0 = out.loc["2024-01-02"]
    assert d0["ann_cnt_5d"] == 1 and d0["ann_cnt_20d"] == 1
    assert d0["ann_flag_reduce"] == 1
    assert d0["ann_days_since"] == 5          # 2023-12-28 → 2024-01-02
    # 回看段本身不输出
    assert out.index.min() == pd.Timestamp("2024-01-02")


def test_index_membership_pit_flags(tmp_path, monkeypatch):
    """快照切换按 PIT 生效：新快照覆盖旧成分（标记翻回 0）。"""
    _write(tmp_path, "index_constituents", "as_of_date=2024-01", pl.DataFrame({
        "index_symbol": ["399001.SZ"], "symbol": ["000001.SZ"],
        "as_of_date": [dt.date(2024, 1, 31)], "weight": [0.0],
    }))
    _write(tmp_path, "index_constituents", "as_of_date=2024-02", pl.DataFrame({
        "index_symbol": ["399006.SZ"], "symbol": ["000001.SZ"],
        "as_of_date": [dt.date(2024, 2, 29)], "weight": [0.0],
    }))
    monkeypatch.setattr(_pitlib, "_curated_root", lambda: tmp_path)

    out = index_membership.load("index_constituents", start="2024-01-02", end="2024-03-08").set_index("trade_date")

    assert out.index.min() == pd.Timestamp("2024-01-31")          # 首快照前无行
    assert out.loc["2024-01-31", "idx_szcomp"] == 1.0 and out.loc["2024-01-31", "idx_chinext"] == 0.0
    assert out.loc["2024-01-31", "idx_in_index"] == 1.0
    assert out.loc["2024-02-29", "idx_szcomp"] == 0.0 and out.loc["2024-02-29", "idx_chinext"] == 1.0


def test_research_views_dual_source(tmp_path, monkeypatch):
    """一致预期 + 研报预测双源合并：年度取最新、评级映射、90 交易日覆盖密度。"""
    _write(tmp_path, "analyst_consensus", "forecast_date=2024-01-10", pl.DataFrame({
        "symbol": ["000001.SZ"] * 2, "forecast_date": [dt.date(2024, 1, 10)] * 2,
        "forecast_year": ["2024", "2025"], "eps_forecast": [1.0, 2.0],
        "pe_forecast": [None, None], "target_price": [None, 12.5],
        "rating": ["overweight", "buy"], "analyst_count": [3, 5],
    }))
    _write(tmp_path, "report_rc", "report_date=2024", pl.DataFrame({
        "symbol": ["000001.SZ"] * 2, "report_date": [dt.date(2024, 1, 8), dt.date(2024, 1, 9)],
        "eps": [1.1, 1.3], "pe": [10.0, 9.0], "roe": [11.0, 12.0], "tp": [100.0, 200.0],
        "rating": ["买入", "增持"], "max_price": [None, None], "min_price": [None, None],
    }))
    monkeypatch.setattr(_pitlib, "_curated_root", lambda: tmp_path)

    out = research_views.load("analyst_consensus", start="2024-01-02", end="2024-01-31").set_index("trade_date")

    d10 = out.loc["2024-01-10"]
    # 同 (symbol, forecast_date) 取最近预测年度 → 2025 的 eps 2.0 / target 12.5 / buy=2 分
    assert d10["ac_eps_fy"] == pytest.approx(2.0) and d10["ac_target_price"] == pytest.approx(12.5)
    assert d10["ac_rating"] == 2.0 and d10["ac_analyst_count"] == 5.0 and d10["ac_days_since"] == 0
    # 研报侧：1/8 与 1/9 两份 → 覆盖数累计 2；asof 取最近一份（1.3 / 买入=2 分）
    d8 = out.loc["2024-01-08"]
    assert d8["rc_cnt_90d"] == 1 and d8["rc_rating"] == 2.0          # 买入 → 2 分
    d9 = out.loc["2024-01-09"]
    assert d9["rc_cnt_90d"] == 2 and d9["rc_eps_y"] == pytest.approx(1.3)
    assert d9["rc_rating"] == 1.0                                     # 增持 → 1 分（asof 取最近一份）
