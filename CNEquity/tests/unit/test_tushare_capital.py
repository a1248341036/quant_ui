"""Unit tests: Tushare capital fetchers map onto the existing lake schemas.

The fetchers must emit rows with the SAME units as the EastMoney history they
join (fund_flow 元, dragon_tiger 元, block_trades 万股/万元, holder counts
户/%/股/元) so curated history and Tushare increments coexist in one schema.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import polars as pl
import pytest

import cnequity.external.tushare_capital as tc

_CFG = SimpleNamespace(external_tushare_wide_interval=0.0)


def _md(df: pl.DataFrame, symbol: str) -> dict:
    return df.filter(pl.col("symbol") == symbol).row(0, named=True)


# ── fund_flow ────────────────────────────────────────────────────────────


def test_fund_flow_maps_yuan_units_from_buy_sell(monkeypatch):
    raw = pl.DataFrame(
        {
            "ts_code": ["000001.SZ", "600000.SH"],
            "trade_date": ["20260828", "20260828"],
            "buy_elg_amount": [100.0, 0.0],
            "sell_elg_amount": [40.0, 0.0],
            "buy_lg_amount": [50.0, 10.0],
            "sell_lg_amount": [20.0, 30.0],
            "buy_md_amount": [5.0, 5.0],
            "sell_md_amount": [5.0, 5.0],
            "buy_sm_amount": [1.0, 1.0],
            "sell_sm_amount": [3.0, 1.0],
        }
    )
    monkeypatch.setattr(tc, "_fetch", lambda config, api, **kw: raw)
    out = tc.fetch_fund_flow_tushare(date(2026, 8, 28), config=None)

    assert out.columns == [
        "symbol",
        "trade_date",
        "main_net_inflow",
        "super_large_net_inflow",
        "large_net_inflow",
        "medium_net_inflow",
        "small_net_inflow",
    ]
    row = _md(out, "000001.SZ")
    # 万元 → 元：净超大 60万 → 60e4；净大单 30万 → 30e4；主力 = 90e4
    assert row["super_large_net_inflow"] == 60 * 1e4
    assert row["large_net_inflow"] == 30 * 1e4
    assert row["main_net_inflow"] == 90 * 1e4
    assert row["medium_net_inflow"] == 0.0
    assert row["small_net_inflow"] == -2 * 1e4
    assert out["trade_date"].dtype == pl.Date
    assert _md(out, "600000.SH")["main_net_inflow"] == -20 * 1e4


def test_fund_flow_prefers_buy_sell_over_net_columns(monkeypatch):
    raw = pl.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": ["20260828"],
            "buy_elg_amount": [10.0],
            "sell_elg_amount": [4.0],
            "buy_lg_amount": [6.0],
            "sell_lg_amount": [2.0],
            "buy_md_amount": [1.0],
            "sell_md_amount": [1.0],
            "buy_sm_amount": [1.0],
            "sell_sm_amount": [1.0],
            # vendor net 故意给错值，映射必须用 buy/sell 差值
            "net_elg_amount": [999.0],
            "net_lg_amount": [999.0],
        }
    )
    monkeypatch.setattr(tc, "_fetch", lambda config, api, **kw: raw)
    out = tc.fetch_fund_flow_tushare(date(2026, 8, 28), config=None)
    row = _md(out, "000001.SZ")
    assert row["main_net_inflow"] == 10 * 1e4  # (10-4)+(6-2) 万元


def test_fund_flow_empty_response(monkeypatch):
    monkeypatch.setattr(tc, "_fetch", lambda config, api, **kw: pl.DataFrame())
    out = tc.fetch_fund_flow_tushare(date(2026, 8, 28), config=None)
    assert out.is_empty()


# ── dragon_tiger ─────────────────────────────────────────────────────────


def test_dragon_tiger_maps_top_list(monkeypatch):
    raw = pl.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ", "600000.SH"],
            "trade_date": ["20260828"] * 3,
            "name": ["平安银行"] * 3,
            "reason": ["日涨幅偏离值达7%的证券", "换手率达20%", ""],
            "l_buy": [2.0e8, 1.0e8, 5.0e7],
            "l_sell": [1.5e8, 1.2e8, 6.0e7],
            "net_amount": [5.0e7, -2.0e7, -1.0e7],
        }
    )
    monkeypatch.setattr(tc, "_fetch", lambda config, api, **kw: raw)
    out = tc.fetch_dragon_tiger_tushare(date(2026, 8, 28), config=None)

    assert out.columns == [
        "symbol",
        "trade_date",
        "reason",
        "buy_amount",
        "sell_amount",
        "net_amount",
    ]
    # 空白 reason 无法入库（主键 + 非空校验），必须被丢弃
    assert out.height == 2
    # unique(keep="last") 不保证行序（多线程哈希去重），按 reason 精确取行
    row = out.filter(pl.col("reason") == "日涨幅偏离值达7%的证券").row(0, named=True)
    assert row["buy_amount"] == 2.0e8  # 元直通，不换算
    assert row["net_amount"] == 5.0e7
    other = out.filter(pl.col("reason") == "换手率达20%").row(0, named=True)
    assert other["buy_amount"] == 1.0e8
    assert other["net_amount"] == -2.0e7


def test_dragon_tiger_computes_net_when_missing(monkeypatch):
    raw = pl.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": ["20260828"],
            "reason": ["日涨幅偏离值达7%的证券"],
            "l_buy": [2.0e8],
            "l_sell": [1.5e8],
        }
    )
    monkeypatch.setattr(tc, "_fetch", lambda config, api, **kw: raw)
    out = tc.fetch_dragon_tiger_tushare(date(2026, 8, 28), config=None)
    assert _md(out, "000001.SZ")["net_amount"] == 5.0e7


# ── block_trades ─────────────────────────────────────────────────────────


def _wide_frame(rows: list[tuple[str, float | None]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [r[0] for r in rows],
            "trade_date": [date(2026, 8, 28)] * len(rows),
            "close": [r[1] for r in rows],
            "float_share": [None] * len(rows),
            "circ_mv": [None] * len(rows),
        },
        schema={"symbol": pl.Utf8, "trade_date": pl.Date, "close": pl.Float64,
                "float_share": pl.Float64, "circ_mv": pl.Float64},
    )


def test_block_trades_units_and_premium(monkeypatch):
    raw = pl.DataFrame(
        {
            "ts_code": ["000001.SZ", "600000.SH"],
            "trade_date": ["20260828"] * 2,
            "price": [10.0, 8.0],
            "vol": [100.0, 50.0],  # 万股
            "amount": [1000.0, 400.0],  # 万元
        }
    )
    monkeypatch.setattr(tc, "_fetch", lambda config, api, **kw: raw)
    monkeypatch.setattr(tc, "_wide_daily", lambda config, s, e: _wide_frame([("000001.SZ", 12.5), ("600000.SH", 8.0)]))
    out = tc.fetch_block_trades_tushare(date(2026, 8, 28), config=None)

    assert out.columns == ["symbol", "trade_date", "price", "volume", "amount", "premium_ratio"]
    row = _md(out, "000001.SZ")
    assert row["volume"] == 100.0  # 万股直通
    assert row["amount"] == 1000.0  # 万元直通
    assert abs(row["premium_ratio"] - (10.0 / 12.5 - 1.0)) < 1e-12
    # 折价 20%：(8/8 - 1)
    assert _md(out, "600000.SH")["premium_ratio"] == 0.0


def test_block_trades_premium_null_when_wide_missing(monkeypatch):
    raw = pl.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": ["20260828"],
            "price": [10.0],
            "vol": [100.0],
            "amount": [1000.0],
        }
    )
    monkeypatch.setattr(tc, "_fetch", lambda config, api, **kw: raw)
    monkeypatch.setattr(tc, "_wide_daily", lambda config, s, e: _wide_frame([]))
    out = tc.fetch_block_trades_tushare(date(2026, 8, 28), config=None)
    assert out["premium_ratio"][0] is None
    assert out.height == 1


# ── shareholder_counts ───────────────────────────────────────────────────


def _holder_raw(rows: list[tuple[str, str, str, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts_code": [r[0] for r in rows],
            "ann_date": [r[1] for r in rows],
            "end_date": [r[2] for r in rows],
            "holder_num": [r[3] for r in rows],
        }
    )


def test_holder_counts_enrichment_chain_and_units(monkeypatch):
    # 窗口内两批公告：A 首披露 900 户，第二期 810 户（环比 -10%）；
    # B 首披露（curated 无历史 → chg 为 null）
    raw = _holder_raw(
        [
            ("000001.SZ", "20260711", "20260710", 900.0),
            ("000001.SZ", "20260721", "20260720", 810.0),
            ("600000.SH", "20260711", "20260710", 5000.0),
        ]
    )
    monkeypatch.setattr(tc, "_get_pro", lambda config: None)
    monkeypatch.setattr(tc, "_fetch_with_retry", lambda pro, api, interval=0.0, **kw: raw)
    prior = pl.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "count_date": [date(2026, 6, 30)],
            "holder_count": [1000.0],
        },
        schema={"symbol": pl.Utf8, "count_date": pl.Date, "holder_count": pl.Float64},
    )
    monkeypatch.setattr(tc, "_curated_holder_history", lambda config: prior)
    wide = pl.DataFrame(
        {
            "symbol": ["000001.SZ", "600000.SH"],
            "trade_date": [date(2026, 7, 10), date(2026, 7, 10)],
            "close": [12.0, 9.0],
            "float_share": [5000.0, 10000.0],  # 万股
            "circ_mv": [60000.0, 90000.0],  # 万元
        },
        schema={"symbol": pl.Utf8, "trade_date": pl.Date, "close": pl.Float64,
                "float_share": pl.Float64, "circ_mv": pl.Float64},
    )
    monkeypatch.setattr(tc, "_wide_daily", lambda config, s, e: wide)

    out = tc.fetch_holder_counts_tushare(date(2026, 7, 10), date(2026, 7, 21), config=_CFG)
    assert out.columns == tc._HOLDER_SCHEMA_COLS

    first = out.filter((pl.col("symbol") == "000001.SZ") & (pl.col("count_date") == date(2026, 7, 10))).row(0, named=True)
    assert first["holder_count"] == 900.0
    assert first["holder_count_change_pct"] == pytest.approx(-10.0)  # vs curated 1000 户
    assert abs(first["avg_float_shares"] - 5000.0 * 1e4 / 900.0) < 1e-6  # 股/户
    assert abs(first["avg_holding_value"] - 60000.0 * 1e4 / 900.0) < 1e-6  # 元/户
    assert first["announce_date"] == date(2026, 7, 11)

    # 同窗口内第二期链上第一期：810/900 - 1 = -10%
    second = out.filter((pl.col("symbol") == "000001.SZ") & (pl.col("count_date") == date(2026, 7, 20))).row(0, named=True)
    assert second["holder_count_change_pct"] == pytest.approx(-10.0)

    # curated 无历史 → 环比为 null
    b = _md(out, "600000.SH")
    assert b["holder_count_change_pct"] is None
    assert abs(b["avg_holding_value"] - 90000.0 * 1e4 / 5000.0) < 1e-6


def test_holder_counts_counts_daily_ann_date_calls(monkeypatch):
    calls: list[str] = []

    def fake_fetch_with_retry(pro, api, interval=0.0, **kw):
        calls.append(kw["ann_date"])
        return pl.DataFrame()

    monkeypatch.setattr(tc, "_get_pro", lambda config: None)
    monkeypatch.setattr(tc, "_fetch_with_retry", fake_fetch_with_retry)
    out = tc.fetch_holder_counts_tushare(date(2026, 8, 1), date(2026, 8, 3), config=_CFG)
    assert out.is_empty()
    assert calls == ["20260801", "20260802", "20260803"]


# ── step wiring: Tushare primary, EastMoney fallback ─────────────────────


def _fund_flow_frame(day: date) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["600519.SH"],
            "trade_date": [day],
            "main_net_inflow": [1.0],
            "super_large_net_inflow": [0.0],
            "large_net_inflow": [0.0],
            "medium_net_inflow": [0.0],
            "small_net_inflow": [0.0],
        }
    )


def _seed_trading_calendar(cfg, start: date, end: date) -> None:
    rows = []
    d = start
    while d <= end:
        rows.append({"trade_date": d, "is_trading": d.weekday() < 5})
        d = date.fromordinal(d.toordinal() + 1)
    path = cfg.curated_root / "trading_calendar" / "part-merged.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(path)


def test_step_fund_flow_tushare_primary_skips_eastmoney(monkeypatch, tmp_path):
    from cnequity.config import Config
    from cnequity.steps import capital as cap
    from cnequity.storage.state import StateStore

    cfg = Config(data_root=tmp_path / "data")
    _seed_trading_calendar(cfg, date(2024, 6, 27), date(2024, 6, 28))
    StateStore(cfg.meta_root).set_date("fund_flow", date(2024, 6, 27))
    em_calls: list[date] = []
    ts_calls: list[date] = []
    monkeypatch.setattr(cap, "_tushare_capital_ready", lambda config: True)
    monkeypatch.setattr(
        cap, "fetch_fund_flow", lambda d, **kw: em_calls.append(d) or pl.DataFrame()
    )
    monkeypatch.setattr(
        tc,
        "fetch_fund_flow_tushare",
        lambda d, *, config: ts_calls.append(d) or _fund_flow_frame(d),
    )
    cfg.staging_root.mkdir(parents=True)
    result = cap.step_fund_flow(cfg, date(2024, 6, 28), "run-ts", {})
    assert ts_calls == [date(2024, 6, 28)]
    assert em_calls == []
    assert result["rows_written"] == 1
    staged = list(cfg.staging_root.glob("fund_flow/**/*.parquet"))
    assert staged and "tushare" in pl.read_parquet(staged[0])["source"][0]


def test_step_fund_flow_falls_back_to_eastmoney_on_tushare_failure(
    monkeypatch, tmp_path
):
    from cnequity.config import Config
    from cnequity.steps import capital as cap
    from cnequity.storage.state import StateStore

    cfg = Config(data_root=tmp_path / "data")
    _seed_trading_calendar(cfg, date(2024, 6, 27), date(2024, 6, 28))
    StateStore(cfg.meta_root).set_date("fund_flow", date(2024, 6, 27))
    em_calls: list[date] = []
    monkeypatch.setattr(cap, "_tushare_capital_ready", lambda config: True)

    def boom(d, *, config):
        raise RuntimeError("middleware down")

    monkeypatch.setattr(tc, "fetch_fund_flow_tushare", boom)
    monkeypatch.setattr(
        cap, "fetch_fund_flow", lambda d, **kw: em_calls.append(d) or _fund_flow_frame(d)
    )
    cfg.staging_root.mkdir(parents=True)
    result = cap.step_fund_flow(cfg, date(2024, 6, 28), "run-fb", {})
    assert em_calls == [date(2024, 6, 28)]
    assert result["rows_written"] == 1
