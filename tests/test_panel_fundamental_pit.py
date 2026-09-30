# -*- coding: utf-8 -*-
"""``funda_ebitda`` 接入面板的 PIT（point-in-time）回归测试。

验收核心：**PIT 锚点必须是实际公告日 ``ann_date``，绝不是报告期 ``end_date``**。
构造"报告期已结束、公告日尚未到"的用例——报告期 2023-03-31，公告日 2023-04-25——
断言公告日**之前**取不到该值（无行 / NaN），且公告日之前旧报告期的值继续生效，
不允许出现"用 end_date 当锚点导致提前泄露"的前视偏差。

覆盖两条真实字段路径：
1. 挖掘链路（CNE adapter）``alphaagent.data.adapters.plugins.fundamental.load``
   ——curated ``fina_indicator.ebitda`` → ``funda_ebitda``，按 ann_date 广播。
2. 离线 enrich ``alphaagent.data.fundamental.enrich_panel_fundamentals``
   ——季频缓存 ``funda_ebitda`` + 披露日历 → 严格 PIT 日频（公告日 D 不可用，
   D 的下一交易日起可用）。
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import polars as pl

from alphaagent.data import fundamental as funda_mod
from alphaagent.data.adapters.plugins import fundamental as funda_plugin

# 报告期（end_date）与公告日（ann_date）刻意分开：报告期早、公告日明显靠后
_END_DATE = datetime.date(2023, 3, 31)
_ANN_DATE = datetime.date(2023, 4, 25)
_EBITDA = 3.3e9


# ---------------------------------------------------------------------------
# 路径 1：CNE adapter 插件（挖掘链路）
# ---------------------------------------------------------------------------
def _synthetic_curated(*, end_dates, ann_dates, ebitdas):
    """构造最小 curated 表：balancesheet(提供规模哨兵列) + fina_indicator(含 ebitda)。"""
    bal = pl.DataFrame(
        {
            "symbol": pl.Series(["000001.SZ"] * len(end_dates), dtype=pl.Utf8),
            "end_date": pl.Series(list(end_dates), dtype=pl.Date),
            "ann_date": pl.Series(list(ann_dates), dtype=pl.Date),
            "total_assets": [1.0e11] * len(end_dates),
        }
    )
    fina = pl.DataFrame(
        {
            "symbol": pl.Series(["000001.SZ"] * len(end_dates), dtype=pl.Utf8),
            "end_date": pl.Series(list(end_dates), dtype=pl.Date),
            "ann_date": pl.Series(list(ann_dates), dtype=pl.Date),
            "ebitda": list(ebitdas),
        }
    )

    def fake_read(dataset: str) -> pl.DataFrame:
        if dataset == "balancesheet":
            return bal
        if dataset == "fina_indicator":
            return fina
        raise FileNotFoundError(f"测试未提供数据集: {dataset}")

    return fake_read


def test_cne_adapter_ebitda_not_visible_before_ann_date(monkeypatch):
    """公告日之前取不到该值；报告期 end_date 不得作为 PIT 锚点。"""
    # 报告期 2023-03-31 于 2023-04-25 公告；上一期 2022-12-31 于 2023-04-20 公告
    prev_end, prev_ann, prev_ebitda = (
        datetime.date(2022, 12, 31),
        datetime.date(2023, 4, 20),
        1.0e9,
    )
    monkeypatch.setattr(
        funda_plugin,
        "_read_curated",
        _synthetic_curated(
            end_dates=[prev_end, _END_DATE],
            ann_dates=[prev_ann, _ANN_DATE],
            ebitdas=[prev_ebitda, _EBITDA],
        ),
    )

    out = funda_plugin.load("fundamental", start="2023-01-02", end="2023-05-31")

    assert "funda_ebitda" in out.columns, f"面板缺 funda_ebitda: {sorted(out.columns)}"
    df = out.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    assert (df["trade_date"] <= pd.Timestamp("2023-05-31")).all()

    # ① 上一期公告日之前：该股票完全没有基本面行 → 取不到值
    assert (df["trade_date"] < pd.Timestamp(prev_ann)).sum() == 0

    # ② 上一期公告日 ~ 本期公告日之间：只允许旧值（报告期 2023-03-31 已结束但未公告）
    gap = df[(df["trade_date"] >= pd.Timestamp(prev_ann)) & (df["trade_date"] < pd.Timestamp(_ANN_DATE))]
    assert not gap.empty, "gap 窗口应有行（上一期已公告）"
    np.testing.assert_allclose(
        gap["funda_ebitda"].to_numpy(dtype=float),
        prev_ebitda,
        rtol=1e-6,
        err_msg="公告日之前泄露了本期 ebitda（end_date 被当成了 PIT 锚点）",
    )

    # ③ 本期公告日起：切换为本期值
    after = df[df["trade_date"] >= pd.Timestamp(_ANN_DATE)]
    assert not after.empty
    np.testing.assert_allclose(after["funda_ebitda"].to_numpy(dtype=float), _EBITDA, rtol=1e-6)


def test_cne_adapter_ebitda_absent_entirely_before_first_announcement(monkeypatch):
    """只有一期报表时：公告日之后才有行，公告日之前没有任何 funda_ebitda 取值。"""
    monkeypatch.setattr(
        funda_plugin,
        "_read_curated",
        _synthetic_curated(end_dates=[_END_DATE], ann_dates=[_ANN_DATE], ebitdas=[_EBITDA]),
    )

    out = funda_plugin.load("fundamental", start="2023-01-02", end="2023-05-31")
    df = out.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    # 报告期 2023-03-31 早于公告日：报告期当天及之前一律无值
    assert (df["trade_date"] < pd.Timestamp(_ANN_DATE)).sum() == 0
    assert (df["trade_date"] >= pd.Timestamp(_END_DATE)).all()
    np.testing.assert_allclose(
        df.loc[df["trade_date"] >= pd.Timestamp(_ANN_DATE), "funda_ebitda"].to_numpy(dtype=float),
        _EBITDA,
        rtol=1e-6,
    )


# ---------------------------------------------------------------------------
# 路径 2：离线 build_panel --with-fundamentals（季频缓存 + 披露日历）
# ---------------------------------------------------------------------------
def _write_synthetic_caches(tmp_path):
    trade_dates = pd.bdate_range("2023-04-03", "2023-05-31")
    inst = "000001.SZ"
    panel = pd.DataFrame(
        {"close": 1.0},
        index=pd.MultiIndex.from_product(
            [trade_dates, [inst]], names=["datetime", "instrument"]
        ),
    )

    quarterly = pd.DataFrame(
        {"funda_ebitda": [_EBITDA]},
        index=pd.MultiIndex.from_arrays(
            [pd.to_datetime([_END_DATE]), [inst]], names=["report_end", "instrument"]
        ),
    )
    quarterly_path = tmp_path / "quarterly.parquet"
    quarterly.to_parquet(quarterly_path)

    disclosure = pd.DataFrame(
        {inst: [pd.Timestamp(_ANN_DATE)]},
        index=pd.DatetimeIndex(pd.to_datetime([_END_DATE]), name="report_end"),
    )
    disclosure_path = tmp_path / "disclosure.parquet"
    disclosure.to_parquet(disclosure_path)
    return panel, quarterly_path, disclosure_path


def test_offline_enrich_ebitda_pit_uses_next_trade_date(tmp_path):
    """公告日 D 当天不可用，D 的下一交易日起才可引用（严格 PIT）。"""
    panel, quarterly_path, disclosure_path = _write_synthetic_caches(tmp_path)

    out = funda_mod.enrich_panel_fundamentals(
        panel,
        quarterly_path=quarterly_path,
        disclosure_path=disclosure_path,
        include_disclosure_features=False,
    )

    assert "funda_ebitda" in out.columns, f"离线 enrich 缺 funda_ebitda: {sorted(out.columns)}"
    values = out["funda_ebitda"]
    days = values.index.get_level_values("datetime")

    effective = pd.Timestamp(_ANN_DATE) + pd.tseries.offsets.BDay(1)
    before = values[days < effective]
    assert not before.empty
    assert before.isna().all(), "公告日（含）之前就取到了 ebitda → 前视偏差"

    after = values[days >= effective]
    assert not after.empty
    np.testing.assert_allclose(after.to_numpy(dtype=float), _EBITDA, rtol=1e-6)


# ---------------------------------------------------------------------------
# 登记检查：prompt 数据面清单必须能看见该字段
# ---------------------------------------------------------------------------
def test_data_fields_prompt_registers_funda_ebitda():
    from alphaagent.factor.mining.prompt.modules import data_fields

    assert "$funda_ebitda" in data_fields._FUNDAMENTAL_SECTION_MD
