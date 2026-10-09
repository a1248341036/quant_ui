# -*- coding: utf-8 -*-
"""研报字段接入回归：fundamental 扩展列 / 机构分类型持仓 / 解禁插件（2026-10-08）。

背景：读三篇金工研报（招商市值、海通五因子、天风电子基本面）时发现，研报因子表里
高频需要的字段在 CNE 里**已有**、但没进面板或没映射：
- ``fina_indicator`` 的单季口径（q_roe / q_sales_yoy / dt_netprofit_yoy）与周转（fa_turn）
- ``balancesheet`` 的在建工程（cip_total）与无形资产（intan_assets）
- ``institutional_holdings`` 的分类型持仓（基金/QFII/保险/社保…，研报"基金增持"事件依赖它）
- ``share_unlock_schedule`` 解禁事件（面板此前完全没有该族列）
本文件锁定这三处的映射与 PIT 口径，防止后续静默丢列。

夹具说明：``fundamental`` 自己缓存 curated 根（``_curated_root_cache``）而读文件走
``_pitlib.read_curated``，两处都要指向临时目录；在 worktree 里运行时其默认回退路径
（``<worktree>/CNEquity/...``）并不存在，故必须显式 patch。
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from alphaagent.data.adapters.plugins import _pitlib, fundamental, institutional, share_unlock


def _patch_roots(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(_pitlib, "_curated_root", lambda: tmp_path)
    monkeypatch.setattr(fundamental, "_curated_root_cache", tmp_path)


def _write_curated(root: Path, dataset: str, df: pl.DataFrame) -> None:
    d = root / dataset
    d.mkdir(parents=True, exist_ok=True)
    df.write_parquet(d / "part-merged.parquet")


# ── 1. fundamental：单季口径 / 周转 / 在建工程 / 无形资产 ─────────────


def test_fundamental_maps_report_extra_columns(tmp_path, monkeypatch):
    """新映射列必须出现在输出里，并遵守 ann_date PIT（公告前不产出该行）。"""
    _patch_roots(monkeypatch, tmp_path)
    ann = dt.date(2024, 4, 25)
    _write_curated(tmp_path, "fina_indicator", pl.DataFrame({
        "symbol": ["000001.SZ"], "ann_date": [ann], "end_date": [dt.date(2024, 3, 31)],
        "roe": [12.5], "q_roe": [3.1], "q_dt_roe": [2.9], "q_sales_yoy": [18.4],
        "dt_netprofit_yoy": [22.2], "fa_turn": [3.7], "assets_turn": [0.8],
        "ar_turn": [5.5], "roe_waa": [3.3], "roe_dt": [2.8], "or_yoy": [15.0],
    }))
    _write_curated(tmp_path, "balancesheet", pl.DataFrame({
        "symbol": ["000001.SZ"], "ann_date": [ann], "end_date": [dt.date(2024, 3, 31)],
        "total_assets": [1.0e11], "cip_total": [2.5e9], "intan_assets": [4.0e8],
        "fix_assets": [3.0e9],
    }))
    _write_curated(tmp_path, "income", pl.DataFrame({
        "symbol": ["000001.SZ"], "ann_date": [ann], "end_date": [dt.date(2024, 3, 31)],
        "n_income_attr_p": [5.0e9], "rd_exp": [1.2e9],
    }))
    _write_curated(tmp_path, "cashflow", pl.DataFrame({
        "symbol": ["000001.SZ"], "ann_date": [ann], "end_date": [dt.date(2024, 3, 31)],
        "n_cashflow_act": [6.0e9],
    }))

    out = fundamental.load("fundamental", start="2024-03-01", end="2024-05-31")

    for col in ("funda_q_roe", "funda_q_dt_roe", "funda_q_sales_yoy", "funda_dt_netprofit_yoy",
                "funda_fa_turn", "funda_assets_turn", "funda_ar_turn", "funda_roe_waa",
                "funda_roe_dt", "funda_cip", "funda_intangible_assets", "funda_rd_expense"):
        assert col in out.columns, f"缺列 {col}"

    idx = out.set_index(["trade_date", "ts_code"]).sort_index()
    # 公告日之前不产出（fundamental 丢弃首份财报披露前的行）
    assert (pd.Timestamp("2024-04-01"), "000001.SZ") not in idx.index
    after = idx.loc[(pd.Timestamp("2024-05-06"), "000001.SZ")]
    assert float(after["funda_q_roe"]) == pytest.approx(3.1)
    assert float(after["funda_q_sales_yoy"]) == pytest.approx(18.4)
    assert float(after["funda_fa_turn"]) == pytest.approx(3.7)
    assert float(after["funda_cip"]) == pytest.approx(2.5e9)
    assert float(after["funda_intangible_assets"]) == pytest.approx(4.0e8)
    assert float(after["funda_rd_expense"]) == pytest.approx(1.2e9)


# ── 2. institutional：分类型持仓 ───────────────────────────────────────


def test_institutional_per_type_columns(tmp_path, monkeypatch):
    """基金/QFII/保险 分类型比例与家数进面板；缺失类型留空而不是填 0。"""
    _patch_roots(monkeypatch, tmp_path)
    _write_curated(tmp_path, "institutional_holdings", pl.DataFrame({
        "symbol": ["600000.SH"] * 3,
        "holder_type": ["summary", "fund", "qfii"],
        "report_period": ["2024Q1"] * 3,
        "holding_shares": [30.0, 25.0, 5.0],
        "holding_ratio": [12.5, 10.0, 2.5],
        "holding_mv": [1.0e9, 8.0e8, 2.0e8],
    }))
    _write_curated(tmp_path, "earnings_disclosure_schedule", pl.DataFrame({
        "symbol": ["600000.SH"], "report_period": ["2024Q1"],
        "actual_date": [dt.date(2024, 4, 29)],
    }))

    out = institutional.load("institutional_holdings", start="2024-04-01", end="2024-05-31")
    for col in ("inst_fund_ratio", "inst_fund_count", "inst_qfii_ratio", "inst_insurance_ratio"):
        assert col in out.columns, f"缺列 {col}"

    row = out.set_index(["trade_date", "ts_code"]).sort_index().loc[
        (pd.Timestamp("2024-05-06"), "600000.SH")
    ]
    assert float(row["inst_fund_ratio"]) == pytest.approx(10.0)
    assert float(row["inst_fund_count"]) == pytest.approx(25.0)
    assert float(row["inst_qfii_ratio"]) == pytest.approx(2.5)
    # 该期没有保险持仓行 → 留空（不得填 0，避免"未披露"被当作"无持仓"）
    assert pd.isna(row["inst_insurance_ratio"])


# ── 3. share_unlock：解禁事件 PIT 展开 ─────────────────────────────────


def test_share_unlock_pit_columns(tmp_path, monkeypatch):
    """最近一次解禁、过去 180 天解禁合计、下一次计划解禁；首次解禁前不产出。"""
    _patch_roots(monkeypatch, tmp_path)
    _write_curated(tmp_path, "share_unlock_schedule", pl.DataFrame({
        "symbol": ["000002.SZ", "000002.SZ"],
        "unlock_date": [dt.date(2024, 3, 1), dt.date(2024, 9, 2)],
        "unlock_ratio": [10.0, 20.0],
    }))

    out = share_unlock.load("share_unlock_schedule", start="2024-01-01", end="2024-12-31")
    idx = out.set_index(["trade_date", "ts_code"]).sort_index()

    # 首次解禁之前不产出
    assert (pd.Timestamp("2024-02-01"), "000002.SZ") not in idx.index

    d1 = pd.Timestamp("2024-06-03")   # 距 3/1 已 94 天，下一次 9/2
    r1 = idx.loc[(d1, "000002.SZ")]
    assert float(r1["unlock_days_since"]) == float((d1.date() - dt.date(2024, 3, 1)).days)
    assert float(r1["unlock_ratio_last"]) == pytest.approx(10.0)
    assert float(r1["unlock_ratio_past180"]) == pytest.approx(10.0)
    assert float(r1["unlock_days_to_next"]) == float((dt.date(2024, 9, 2) - d1.date()).days)
    assert float(r1["unlock_ratio_next"]) == pytest.approx(20.0)

    d2 = pd.Timestamp("2024-08-30")  # 3/1 已超出 180 天窗口，9/2 尚未发生
    r2 = idx.loc[(d2, "000002.SZ")]
    assert float(r2["unlock_ratio_past180"]) == pytest.approx(0.0)
    assert float(r2["unlock_days_to_next"]) == float((dt.date(2024, 9, 2) - d2.date()).days)

    d3 = pd.Timestamp("2024-09-10")  # 两次解禁中只有 9/2 落在 180 天窗口内
    r3 = idx.loc[(d3, "000002.SZ")]
    assert float(r3["unlock_ratio_past180"]) == pytest.approx(20.0)
    assert float(r3["unlock_ratio_last"]) == pytest.approx(20.0)
    assert pd.isna(r3["unlock_days_to_next"])
