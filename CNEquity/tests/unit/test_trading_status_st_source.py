"""trading_status 的 ST 回填源切换（baostock ↔ tushare）+ 覆盖率 receipt 白名单。

背景：baostock 逐票扫全 A ≈ 11 小时（免费层一个会话约 43 次查询就进黑名单，只能
20 票/会话 + 120s 冷却）；Tushare 中间件 `stock_st` 返回按日全市场名单，整窗约 15
分钟。本文件锁住：来源可配置且写入 scope/receipt、非白名单来源被拒、名单源的零行
标的算有效证据、receipt 校验认新来源。
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

import cnequity.steps  # noqa: F401
from cnequity.config import Config
from cnequity.config.loader import ST_BACKFILL_SOURCES
from cnequity.external import tushare_stock_st
from cnequity.quality import st_coverage
from cnequity.steps import reference


@pytest.fixture
def cfg(tmp_path):
    c = Config(data_root=tmp_path / "data")
    c.staging_root.mkdir(parents=True, exist_ok=True)
    c.curated_root.mkdir(parents=True, exist_ok=True)
    return c


# ─── 来源配置 ─────────────────────────────────────────────────────────


def test_source_whitelist_covers_both_backfill_sources():
    assert set(ST_BACKFILL_SOURCES) == {"baostock", "tushare"}
    assert Config(data_root="x").trading_status_st_backfill_source == "baostock"


def test_loader_rejects_unknown_source(tmp_path):
    from cnequity.config.loader import load_config

    toml = tmp_path / "cnequity.toml"
    toml.write_text(
        '[data]\nroot = "lake"\n[datasets.trading_status]\nst_backfill_source = "akshare"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="不在白名单"):
        load_config(toml)


def test_loader_accepts_tushare_source(tmp_path):
    from cnequity.config.loader import load_config

    toml = tmp_path / "cnequity.toml"
    toml.write_text(
        '[data]\nroot = "lake"\n[datasets.trading_status]\nst_backfill_source = "tushare"\n',
        encoding="utf-8",
    )
    assert load_config(toml).trading_status_st_backfill_source == "tushare"


def test_scope_records_the_source():
    baostock = st_coverage.build_st_scope(["000001.SZ"], date(2024, 1, 1), date(2024, 6, 28), universe="all_a")
    tushare = st_coverage.build_st_scope(
        ["000001.SZ"], date(2024, 1, 1), date(2024, 6, 28), universe="all_a", source="tushare"
    )
    assert baostock["source"] == "baostock"
    assert tushare["source"] == "tushare"
    # 来源不同 → scope 身份不同，不会互相复用 checkpoint
    assert baostock["scope_id"] != tushare["scope_id"]


# ─── Tushare ST 抓取 → trading_status 行契约 ──────────────────────────


def test_fetch_tushare_st_maps_to_trading_status_contract(monkeypatch):
    raw = pl.DataFrame(
        {
            "ts_code": ["600265.SH", "000732.SZ", "600265.SH"],
            "name": ["*ST景谷", "ST泰禾", "*ST景谷"],
            # 中间件实际返回的是 YYYYMMDD 字符串（实测），不是 date 对象
            "trade_date": ["20220601", "20220601", "20220601"],
            "type": ["ST", "ST", "ST"],
            "type_name": ["风险警示板"] * 3,
        }
    )
    monkeypatch.setattr(tushare_stock_st, "_get_pro", lambda config: object())
    monkeypatch.setattr(tushare_stock_st, "fetch_range_paged", lambda *a, **k: raw)

    cfg = Config(data_root="x", external_tushare_wide_interval=0.0)
    out = tushare_stock_st.fetch_st_history(date(2022, 6, 1), date(2022, 6, 1), config=cfg)

    assert out.columns == ["symbol", "trade_date", "is_trading", "status"]
    assert out.height == 2  # 同 (symbol, trade_date) 去重
    assert out["status"].unique().to_list() == ["st"]
    assert out["is_trading"].unique().to_list() == [True]
    assert out["trade_date"].to_list() == [date(2022, 6, 1), date(2022, 6, 1)]


def test_fetch_tushare_st_accepts_iso_date_strings(monkeypatch):
    """date 列既可能是 YYYYMMDD 也可能是 ISO 字符串，两种都要能解析。"""
    raw = pl.DataFrame(
        {
            "ts_code": ["600265.SH", "000732.SZ"],
            "trade_date": ["2022-06-01", "2022-06-02"],
        }
    )
    monkeypatch.setattr(tushare_stock_st, "_get_pro", lambda config: object())
    monkeypatch.setattr(tushare_stock_st, "fetch_range_paged", lambda *a, **k: raw)

    cfg = Config(data_root="x", external_tushare_wide_interval=0.0)
    out = tushare_stock_st.fetch_st_history(date(2022, 6, 1), date(2022, 6, 2), config=cfg)
    assert out["trade_date"].to_list() == [date(2022, 6, 1), date(2022, 6, 2)]


def test_fetch_tushare_st_filters_to_requested_symbols(monkeypatch):
    raw = pl.DataFrame(
        {
            "ts_code": ["600265.SH", "000732.SZ"],
            "trade_date": [date(2022, 6, 1)] * 2,
        }
    )
    monkeypatch.setattr(tushare_stock_st, "_get_pro", lambda config: object())
    monkeypatch.setattr(tushare_stock_st, "fetch_range_paged", lambda *a, **k: raw)

    cfg = Config(data_root="x", external_tushare_wide_interval=0.0)
    out = tushare_stock_st.fetch_st_history(
        date(2022, 6, 1), date(2022, 6, 1), symbols=["000732.SZ"], config=cfg
    )
    assert out["symbol"].to_list() == ["000732.SZ"]


def test_fetch_tushare_st_requires_config():
    with pytest.raises(ValueError, match="需要 config"):
        tushare_stock_st.fetch_st_history(date(2022, 6, 1), date(2022, 6, 1))


# ─── 回填步骤：tushare 源一次拉取 + 零行标的算有效证据 ─────────────────


def test_tushare_backfill_marks_never_st_symbols_complete(cfg, monkeypatch):
    """名单源：整窗一次拉取；名单里没有的标的记 0 行证据并标记完成。"""
    cfg.trading_status_st_backfill_source = "tushare"
    cfg._backfill = True
    cfg._backfill_start = date(2022, 6, 1)
    cfg._backfill_end = date(2022, 6, 30)
    # 直接给定 universe：避免测试去打 TDX/湖取标的清单
    monkeypatch.setattr(reference, "current_st_universe", lambda config: ["000732.SZ", "600519.SH"])

    market = pl.DataFrame(
        {
            "symbol": ["000732.SZ"],
            "trade_date": [date(2022, 6, 1)],
            "is_trading": [True],
            "status": ["st"],
        }
    )
    calls: list[tuple] = []

    def fake_fetch(start, end, *, config=None, symbols=None):
        calls.append((start, end))
        return market

    monkeypatch.setattr(tushare_stock_st, "fetch_st_history", fake_fetch)
    # 步骤内是 `from ... import fetch_st_history as fetch_tushare_st`，patch 模块属性即可
    result = reference._backfill_trading_status_st(cfg, date(2022, 6, 30), "run-tushare")

    assert calls == [(date(2022, 6, 1), date(2022, 6, 30))]  # 整窗只拉一次
    assert result["completed_symbols"] == 2
    assert result["expected_symbols"] == 2
    assert result["coverage_pending_compact"] is True

    ckpt_path = cfg.meta_root / "state" / "historical_st_evidence" / "v2" / f"{result['scope_id']}.json"
    ckpt = __import__("json").loads(ckpt_path.read_text(encoding="utf-8"))
    assert ckpt["scope"]["source"] == "tushare"
    assert ckpt["evidence_rows_by_symbol"] == {"000732.SZ": 1, "600519.SH": 0}
    assert ckpt["status"] == "complete"


def test_receipt_accepts_tushare_source(cfg, monkeypatch):
    """receipt 校验必须认 tushare 来源，否则新源永远签不出覆盖率凭证。"""
    cfg.curated_root.mkdir(parents=True, exist_ok=True)
    part = cfg.curated_root / "trading_status" / "trade_date=2022-06"
    part.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "symbol": ["000732.SZ"],
            "trade_date": [date(2022, 6, 1)],
            "is_trading": [True],
            "status": ["st"],
            "source": ["tushare"],
        }
    ).write_parquet(part / "part-merged.parquet")

    scope = st_coverage.build_st_scope(
        ["000732.SZ"], date(2022, 6, 1), date(2022, 6, 30), universe="all_a", source="tushare"
    )
    checkpoint = {
        "scope": scope,
        "status": "complete",
        "completed_symbols": ["000732.SZ"],
        "evidence_rows_by_symbol": {"000732.SZ": 1},
        "unresolved_symbols": [],
    }
    st_coverage.publish_st_coverage_receipt(cfg, checkpoint)

    monkeypatch.setattr(st_coverage, "current_st_universe", lambda config: ["000732.SZ"])
    report = st_coverage.st_evidence_coverage_report(cfg, date(2022, 6, 1), date(2022, 6, 30))
    assert report["verified"] is True
    assert report["scope_id"] == scope["scope_id"]
