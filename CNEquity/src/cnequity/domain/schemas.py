from __future__ import annotations

from datetime import datetime, timezone

import polars as pl

PROVENANCE = ["source", "data_version", "fetched_at"]

FETCHED_AT_DTYPE = pl.Datetime(time_unit="us", time_zone="UTC")

# Rows carrying this source value are synthetic and must never be trusted
# downstream; audit raises an error finding whenever they reach curated.
MOCK_SOURCE = "mock"

DEFAULT_DATA_VERSION = "v1"

# Datasets whose stored values changed *meaning* — not shape. Adding a column
# is a schema change and leaves `data_version` alone; reinterpreting a value
# already written does not, because old and new rows are then not comparable
# and a reader has no other way to tell them apart.
#
# daily_bars v2: `volume` is 股 for every source. v1 rows are 手 from
# tdx_protocol and sina, 股 from ths and baostock — see
# `cnequity.domain.units` and docs/datasets/schema.md.
DATASET_DATA_VERSION = {
    "daily_bars": "v2",
}


def data_version_for(dataset: str) -> str:
    return DATASET_DATA_VERSION.get(dataset, DEFAULT_DATA_VERSION)


DAILY_BARS_SCHEMA = {
    "symbol": pl.Utf8,
    "trade_date": pl.Date,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Int64,
    "amount": pl.Float64,
    "source": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
}

# Intraday bars. One registered dataset holds exactly one frequency: TDX serves
# 95 trading days of 1m but 491 of 5m and coarser, and a dataset carries a
# single watermark and a single coverage_start — mixing the two horizons under
# one name would make both of them lie. `frequency` is still in the schema and
# in the PK so a second frequency can be added without a breaking change.
#
# bar_time is the bar's CLOSING minute, which is how TDX labels them: a session
# runs 09:31…11:30 and 13:01…15:00, 240 bars, no lunch bars, and the 15:00 bar
# carries the closing auction. It is a naive Asia/Shanghai wall clock, matching
# the convention that only fetched_at is stored tz-aware.
#
# A-shares have no overnight session, so trade_date == bar_time.date() always;
# it is stored anyway because it is the partition column.
MINUTE_BARS_SCHEMA = {
    "symbol": pl.Utf8,
    "trade_date": pl.Date,
    "bar_time": pl.Datetime(time_unit="us"),
    "frequency": pl.Utf8,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    # 股, like every other dataset (cnequity.domain.units). TDX reports
    # intraday bars in 股 natively — unlike its daily K, which is 手 — so this
    # path must NOT reuse the daily lots_to_shares conversion.
    "volume": pl.Int64,
    "amount": pl.Float64,
    "source": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
}

# Transaction records (分笔). Not tick data — A-share Level-1 is a 3-second
# snapshot, so one row aggregates however many real trades landed in one frame
# (6–33 on average, measured). See adapters/tdx_protocol/trade_ticks.py.
TRADE_TICKS_SCHEMA = {
    "symbol": pl.Utf8,
    "trade_date": pl.Date,
    # Position in the session, ascending in time, dense from 0. This is the
    # identity of a row: the wire timestamp has no seconds, so up to twenty
    # records share a `trade_time` and it cannot key anything. Safe as a key
    # because a settled session is frozen — refetched twice, 4,308 rows came
    # back identical field for field.
    "tick_seq": pl.Int32,
    # Minute precision. The seconds are always :00 — they are not truncated
    # from a finer timestamp, the protocol never carried them.
    "trade_time": pl.Datetime(time_unit="us"),
    "price": pl.Float64,
    # 股, like every other dataset (cnequity.domain.units). The wire reports
    # 手; the adapter multiplies by 100, which reconciliation against
    # daily_bars confirms rather than assumes.
    "volume": pl.Int64,
    # buy / sell / neutral / after_hours. TDX's own tick-rule inference, not an
    # exchange field. `after_hours` is the 15:05–15:30 fixed-price session and
    # is *not* in the exchange's daily volume — exclude it before reconciling.
    # No `amount`: the source does not carry one, and a stored price × volume
    # would look like a fact while being an approximation (one representative
    # price stands for every trade folded into the frame).
    "direction": pl.Utf8,
    "source": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
}

# Domestic commodity futures main-continuous daily bars (东财主连).
# symbol = {ROOT}0.{EXCH} e.g. AU0.SHF / I0.DCE — not A-share .SH/.SZ.
COMMODITY_BARS_SCHEMA = {
    "symbol": pl.Utf8,
    "name": pl.Utf8,
    "exchange": pl.Utf8,
    "trade_date": pl.Date,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Int64,
    "amount": pl.Float64,
    "open_interest": pl.Float64,
    "source": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
}

INSTRUMENTS_SCHEMA = {
    "symbol": pl.Utf8,
    "name": pl.Utf8,
    "exchange": pl.Utf8,
    "asset_type": pl.Utf8,
    "list_date": pl.Date,
    "delist_date": pl.Date,
    "prev_symbol": pl.Utf8,
    "source": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
}

TRADING_CALENDAR_SCHEMA = {
    "trade_date": pl.Date,
    "is_trading": pl.Boolean,
    "source": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
}

TRADING_STATUS_SCHEMA = {
    "symbol": pl.Utf8,
    "trade_date": pl.Date,
    "is_trading": pl.Boolean,
    "status": pl.Utf8,
    "source": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
}

# UNIT CONTRACT (per-share): every ratio/amount below is per ONE held share,
# NOT the "每10股" convention Chinese sources quote raw. Adapters divide raw
# per-10-share source values by 10 before staging. So "10派8.5元" → 0.85,
# "10送8股" → 0.8, "10转4股" → 0.4, "10配3股" → 0.3. Downstream real-share
# accounting is uniform: shares_after = shares * (1 + bonus_ratio +
# transfer_ratio); cash = shares * cash_dividend. No /10 magic numbers.
# allotment_price stays a per-share price (yuan paid per allotted share).
CORPORATE_ACTIONS_SCHEMA = {
    "symbol": pl.Utf8,
    "ex_date": pl.Date,
    "action_type": pl.Utf8,
    "cash_dividend": pl.Float64,  # per share (yuan), pretax
    "bonus_ratio": pl.Float64,  # per share (送股: new shares per held share)
    "transfer_ratio": pl.Float64,  # per share (转股: new shares per held share)
    "allotment_ratio": pl.Float64,  # per share (配股: offered shares per held share)
    "allotment_price": pl.Float64,  # per allotted share (yuan), NOT a ratio
    "source": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
}

ADJ_FACTORS_SCHEMA = {
    "symbol": pl.Utf8,
    "trade_date": pl.Date,
    "adjust_type": pl.Utf8,
    "factor": pl.Float64,
    "source": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
}

FINANCIAL_STATEMENT_ITEMS_SCHEMA = {
    "symbol": pl.Utf8,
    "report_period": pl.Utf8,
    "statement_type": pl.Utf8,
    "item_code": pl.Utf8,
    "item_value": pl.Float64,
    "announce_date": pl.Date,
    "source": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
}

# 股本结构. Keyed by the date the structure *changed*, not a report period: a
# company can restructure twice in one quarter and both rows matter, which is
# why `change_reason` rides along rather than being dropped as prose.
SHARE_STRUCTURE_SCHEMA = {
    "symbol": pl.Utf8,
    "change_date": pl.Date,
    "total_shares": pl.Float64,
    "float_shares": pl.Float64,
    "restricted_shares": pl.Float64,
    # 自由流通股 — float minus strategic/locked holdings. The denominator an
    # index uses for free-float weighting; not the same as `float_shares`.
    "free_float_shares": pl.Float64,
    "change_reason": pl.Utf8,
    "announce_date": pl.Date,
    "source": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
}

# 股东户数. The input to 筹码集中度 factors: a falling holder count against a
# flat share count means concentration.
SHAREHOLDER_COUNTS_SCHEMA = {
    "symbol": pl.Utf8,
    # The date the count is as of, NOT a report period. Companies disclose
    # 股东户数 at 旬末/月末 as well as quarter-ends — 2025-07-10 carries 894 rows
    # and 2025-07-31 another 1,162 — and those interim counts are the timely
    # half of the signal. A quarter label would collapse them onto each other.
    "count_date": pl.Date,
    "holder_count": pl.Float64,
    "holder_count_change_pct": pl.Float64,
    "avg_float_shares": pl.Float64,
    "avg_holding_value": pl.Float64,
    "announce_date": pl.Date,
    "source": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
}

# 前十大股东 / 前十大流通股东 — one table, scope discriminator. Both are the same
# shape from the same disclosure: a ranked repeating group of ten. That is what
# the long-format statement table genuinely cannot express, and why this is its
# own dataset rather than more `item_code` rows.
#
# `holding_pct` means different things per scope, deliberately: for `total` it
# is a share of total shares, for `float` a share of the float. They are not
# comparable across scopes, and averaging them would invent a number neither
# source published.
TOP_HOLDERS_SCHEMA = {
    "symbol": pl.Utf8,
    # The list's as-of date. Mostly quarter-ends, but not only: 2025 Q3 has
    # 10,749 total-scope rows dated to something else (prospectuses, 权益变动).
    "record_date": pl.Date,
    "holder_scope": pl.Utf8,
    "holder_rank": pl.Int32,
    "holder_name": pl.Utf8,
    "holding_shares": pl.Float64,
    "holding_pct": pl.Float64,
    "is_institution": pl.Boolean,
    "holder_type": pl.Utf8,
    "announce_date": pl.Date,
    "source": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
}

FUND_FLOW_SCHEMA = {
    "symbol": pl.Utf8,
    "trade_date": pl.Date,
    "main_net_inflow": pl.Float64,
    "super_large_net_inflow": pl.Float64,
    "large_net_inflow": pl.Float64,
    "medium_net_inflow": pl.Float64,
    "small_net_inflow": pl.Float64,
    "source": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
}

MARGIN_TRADING_SCHEMA = {
    "symbol": pl.Utf8,
    "trade_date": pl.Date,
    "margin_balance": pl.Float64,
    "margin_buy": pl.Float64,
    "short_balance": pl.Float64,
    "short_sell_volume": pl.Float64,
    "source": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
}

NORTHBOUND_HOLDINGS_SCHEMA = {
    "symbol": pl.Utf8,
    "trade_date": pl.Date,
    "channel": pl.Utf8,
    "holding_shares": pl.Float64,
    "holding_mv": pl.Float64,
    "holding_ratio": pl.Float64,
    "source": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
}

NORTHBOUND_FLOWS_SCHEMA = {
    "trade_date": pl.Date,
    "channel": pl.Utf8,
    "net_buy": pl.Float64,
    "buy_amount": pl.Float64,
    "sell_amount": pl.Float64,
    "source": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
}

VALUATION_METRICS_SCHEMA = {
    "symbol": pl.Utf8,
    "trade_date": pl.Date,
    "pe_ttm": pl.Float64,
    "pb": pl.Float64,
    "ps_ttm": pl.Float64,
    "total_mv": pl.Float64,
    "float_mv": pl.Float64,
    "source": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
}

SECTOR_MEMBERS_SCHEMA = {
    "symbol": pl.Utf8,
    "sector_code": pl.Utf8,
    "sector_name": pl.Utf8,
    "as_of_date": pl.Date,
    "source": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
}

ANNOUNCEMENT_INDEX_SCHEMA = {
    "announcement_id": pl.Utf8,
    "symbol": pl.Utf8,
    "title": pl.Utf8,
    "announce_date": pl.Date,
    "category": pl.Utf8,
    "url": pl.Utf8,
    "source": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
}

# Scheduled disclosure dates (预约披露) for periodic reports. Current-state:
# a revision overwrites scheduled_date in place; first_scheduled_date keeps the
# original appointment and actual_date stays null until the report is published.
EARNINGS_DISCLOSURE_SCHEDULE_SCHEMA = {
    "symbol": pl.Utf8,
    "report_period": pl.Utf8,
    "scheduled_date": pl.Date,
    "first_scheduled_date": pl.Date,
    "actual_date": pl.Date,
    "source": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
}

DRAGON_TIGER_SCHEMA = {
    "symbol": pl.Utf8,
    "trade_date": pl.Date,
    "reason": pl.Utf8,
    "buy_amount": pl.Float64,
    "sell_amount": pl.Float64,
    "net_amount": pl.Float64,
    "source": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
}

BLOCK_TRADES_SCHEMA = {
    "symbol": pl.Utf8,
    "trade_date": pl.Date,
    "price": pl.Float64,
    "volume": pl.Float64,
    "amount": pl.Float64,
    "premium_ratio": pl.Float64,
    "source": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
}

INDEX_CONSTITUENTS_SCHEMA = {
    "index_symbol": pl.Utf8,
    "symbol": pl.Utf8,
    "as_of_date": pl.Date,
    "weight": pl.Float64,
    "source": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
}

INDUSTRY_MEMBERS_SCHEMA = {
    "symbol": pl.Utf8,
    "classification_system": pl.Utf8,
    "industry_code": pl.Utf8,
    "industry_name": pl.Utf8,
    "as_of_date": pl.Date,
    "source": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
}

MACRO_INDICATORS_SCHEMA = {
    "indicator_id": pl.Utf8,
    "obs_date": pl.Date,
    "value": pl.Float64,
    "frequency": pl.Utf8,
    "source": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
}

MARKET_BREADTH_SCHEMA = {
    "trade_date": pl.Date,
    "metric_id": pl.Utf8,
    "value": pl.Float64,
    "source": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
}

SHARE_UNLOCK_SCHEDULE_SCHEMA = {
    "symbol": pl.Utf8,
    "unlock_date": pl.Date,
    "unlock_shares": pl.Float64,
    "unlock_ratio": pl.Float64,
    "unlock_type": pl.Utf8,
    "source": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
}

REGULATORY_EVENTS_SCHEMA = {
    "event_id": pl.Utf8,
    "symbol": pl.Utf8,
    "event_date": pl.Date,
    "event_type": pl.Utf8,
    "title": pl.Utf8,
    "source": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
}

INSTITUTIONAL_HOLDINGS_SCHEMA = {
    "symbol": pl.Utf8,
    "holder_type": pl.Utf8,
    "report_period": pl.Utf8,
    "holding_shares": pl.Float64,
    "holding_ratio": pl.Float64,
    "holding_mv": pl.Float64,
    "source": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
}

ANALYST_CONSENSUS_SCHEMA = {
    "symbol": pl.Utf8,
    "forecast_date": pl.Date,
    "forecast_year": pl.Int64,
    "eps_forecast": pl.Float64,
    "pe_forecast": pl.Float64,
    "target_price": pl.Float64,
    "rating": pl.Utf8,
    "analyst_count": pl.Int64,
    "source": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
}

SENTIMENT_SCORES_SCHEMA = {
    "symbol": pl.Utf8,
    "trade_date": pl.Date,
    "score_channel": pl.Utf8,
    "sentiment_score": pl.Float64,
    "headline_count": pl.Int64,
    "source": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
}

HOT_RANK_SCHEMA = {
    "symbol": pl.Utf8,
    "trade_date": pl.Date,
    "rank": pl.Int64,
    "rank_change": pl.Int64,
    "hist_rank": pl.Int64,
    "source": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
}

SECTOR_BARS_SCHEMA = {
    "sector_code": pl.Utf8,
    "sector_name": pl.Utf8,
    "board_type": pl.Utf8,
    "trade_date": pl.Date,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Int64,
    "amount": pl.Float64,
    "change_pct": pl.Float64,
    "source": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
}

SECTOR_FUND_FLOW_SCHEMA = {
    "sector_code": pl.Utf8,
    "sector_name": pl.Utf8,
    "board_type": pl.Utf8,
    "trade_date": pl.Date,
    "main_net_inflow": pl.Float64,
    "change_pct": pl.Float64,
    "turnover_pct": pl.Float64,
    "source": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
}

NEWS_HEADLINES_SCHEMA = {
    "news_id": pl.Utf8,
    "publish_date": pl.Date,
    "publish_time": pl.Utf8,
    "title": pl.Utf8,
    "summary": pl.Utf8,
    "related_symbols": pl.Utf8,
    "channel": pl.Utf8,
    "source": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
}

FLASH_NEWS_WIRE_SCHEMA = {
    "wire_id": pl.Utf8,
    "wire_source": pl.Utf8,
    "item_hash": pl.Utf8,
    "publish_date": pl.Date,
    "publish_time": pl.Utf8,
    "title": pl.Utf8,
    "summary": pl.Utf8,
    "related_symbols": pl.Utf8,
    "importance": pl.Int8,
    "channel": pl.Utf8,
    "source": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
}

# --- sentiment-mvp 接入 ---

SENTIMENT_ARTICLES_SCHEMA = {
    "article_id": pl.Utf8,         # articles.key (sha1)
    "symbol": pl.Utf8,             # articles.code (补零到 6 位)
    "publish_date": pl.Date,       # substr(publish_time,1,10)
    "publish_time": pl.Utf8,       # articles.publish_time
    "media": pl.Utf8,              # 来源媒体
    "title": pl.Utf8,
    "summary": pl.Utf8,            # articles.content 截取前 500 字
    "url": pl.Utf8,
    "source": pl.Utf8,             # em / guba / cls / hot
    "label": pl.Utf8,              # positive / negative / neutral
    "score": pl.Float64,           # 情感分 [-1, 1]
    "pos_hits": pl.Float64,
    "neg_hits": pl.Float64,
    "fetched_at": pl.Utf8,         # articles.fetched_at
    "data_version": pl.Utf8,
}

ECONOMIC_CALENDAR_SCHEMA = {
    "event_id": pl.Utf8,
    "event_date": pl.Date,
    "event_time": pl.Utf8,
    "country": pl.Utf8,
    "indicator": pl.Utf8,
    "importance": pl.Int8,
    "forecast": pl.Float64,
    "previous": pl.Float64,
    "actual": pl.Float64,
    "unit": pl.Utf8,
    "source": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
}

# One row per recovered delisting, describing how its price series *ends*.
#
# Whether a series runs through the 退市整理期 decides whether a backtest can
# realise the final loss or marks the position at its last pre-suspension price.
# Measured on this lake, that period is worth -27% to -92%, so the distinction is
# not cosmetic — and it cannot be assumed, because trading-rule delistings
# (面值/市值) legitimately have no consolidation period while a truncated vendor
# series looks identical. Recording the shape lets research separate them
# instead of silently treating both as complete.
INDUSTRY_INDEX_SCHEMA = {
    "trade_date": pl.Date,
    "industry_code": pl.Utf8,
    # L1 / L2 / L3 — the 申万 code is prefix-hierarchical, so one membership
    # series yields all three depths.
    "level": pl.Utf8,
    # equal | amount. Both are stored because free-float cap, the 申万
    # convention, is only ~69% populated in valuation_metrics.
    "weighting": pl.Utf8,
    "ret": pl.Float64,
    # Members known that day, members that actually had a priced bar, and the
    # difference — names without an adjustment factor (北交所) cannot enter the
    # index, and which industries that distorts has to stay visible.
    "n_members": pl.Int64,
    "n_priced": pl.Int64,
    "n_excluded": pl.Int64,
    "amount": pl.Float64,
    "source": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
}

DELISTING_EVENTS_SCHEMA = {
    "symbol": pl.Utf8,
    "first_trade_date": pl.Date,
    "last_trade_date": pl.Date,
    # consolidation | abrupt_decline | abrupt_stable | insufficient
    "ending_pattern": pl.Utf8,
    "final_close": pl.Float64,
    "halt_gap_days": pl.Int64,
    "worst_final_return": pl.Float64,
    "final_window_return": pl.Float64,
    "bars": pl.Int64,
    "source": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
}

# Tushare fundamentals/events tables. Schemas mirror the curated parquet
# columns produced daily by their steps (verified against live partitions).

NAMECHANGE_SCHEMA = {
    "symbol": pl.Utf8,
    "name": pl.Utf8,
    "start_date": pl.Date,
    "end_date": pl.Date,
    "ann_date": pl.Date,
    "change_reason": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
    "source": pl.Utf8,
}

DIVIDEND_SCHEMA = {
    "symbol": pl.Utf8,
    "end_date": pl.Date,
    "ann_date": pl.Date,
    "div_proc": pl.Utf8,
    "stk_div": pl.Float64,
    "stk_bo_rate": pl.Float64,
    "stk_co_rate": pl.Float64,
    "cash_div": pl.Float64,
    "cash_div_tax": pl.Float64,
    "record_date": pl.Date,
    "ex_date": pl.Date,
    "pay_date": pl.Date,
    "div_listdate": pl.Date,
    "imp_ann_date": pl.Date,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
    "source": pl.Utf8,
}

BALANCESHEET_SCHEMA = {
    "symbol": pl.Utf8,
    "ann_date": pl.Date,
    "f_ann_date": pl.Date,
    "end_date": pl.Date,
    "report_type": pl.Float64,
    "comp_type": pl.Float64,
    "end_type": pl.Float64,
    "total_share": pl.Float64,
    "cap_rese": pl.Float64,
    "undistr_porfit": pl.Float64,
    "surplus_rese": pl.Float64,
    "special_rese": pl.Float64,
    "money_cap": pl.Float64,
    "trad_asset": pl.Float64,
    "notes_receiv": pl.Float64,
    "accounts_receiv": pl.Float64,
    "oth_receiv": pl.Float64,
    "prepayment": pl.Float64,
    "div_receiv": pl.Float64,
    "int_receiv": pl.Float64,
    "inventories": pl.Float64,
    "amor_exp": pl.Float64,
    "nca_within_1y": pl.Float64,
    "sett_rsrv": pl.Float64,
    "loanto_oth_bank_fi": pl.Float64,
    "premium_receiv": pl.Float64,
    "reinsur_receiv": pl.Float64,
    "reinsur_res_receiv": pl.Float64,
    "pur_resale_fa": pl.Float64,
    "oth_cur_assets": pl.Float64,
    "total_cur_assets": pl.Float64,
    "fa_avail_for_sale": pl.Float64,
    "htm_invest": pl.Float64,
    "lt_eqt_invest": pl.Float64,
    "invest_real_estate": pl.Float64,
    "time_deposits": pl.Float64,
    "oth_assets": pl.Float64,
    "lt_rec": pl.Float64,
    "fix_assets": pl.Float64,
    "cip": pl.Float64,
    "const_materials": pl.Float64,
    "fixed_assets_disp": pl.Float64,
    "produc_bio_assets": pl.Float64,
    "oil_and_gas_assets": pl.Float64,
    "intan_assets": pl.Float64,
    "r_and_d": pl.Float64,
    "goodwill": pl.Float64,
    "lt_amor_exp": pl.Float64,
    "defer_tax_assets": pl.Float64,
    "decr_in_disbur": pl.Float64,
    "oth_nca": pl.Float64,
    "total_nca": pl.Float64,
    "cash_reser_cb": pl.Float64,
    "depos_in_oth_bfi": pl.Float64,
    "prec_metals": pl.Float64,
    "deriv_assets": pl.Float64,
    "rr_reins_une_prem": pl.Float64,
    "rr_reins_outstd_cla": pl.Float64,
    "rr_reins_lins_liab": pl.Float64,
    "rr_reins_lthins_liab": pl.Float64,
    "refund_depos": pl.Float64,
    "ph_pledge_loans": pl.Float64,
    "refund_cap_depos": pl.Float64,
    "indep_acct_assets": pl.Float64,
    "client_depos": pl.Float64,
    "client_prov": pl.Float64,
    "transac_seat_fee": pl.Float64,
    "invest_as_receiv": pl.Float64,
    "total_assets": pl.Float64,
    "lt_borr": pl.Float64,
    "st_borr": pl.Float64,
    "cb_borr": pl.Float64,
    "depos_ib_deposits": pl.Float64,
    "loan_oth_bank": pl.Float64,
    "trading_fl": pl.Float64,
    "notes_payable": pl.Float64,
    "acct_payable": pl.Float64,
    "adv_receipts": pl.Float64,
    "sold_for_repur_fa": pl.Float64,
    "comm_payable": pl.Float64,
    "payroll_payable": pl.Float64,
    "taxes_payable": pl.Float64,
    "int_payable": pl.Float64,
    "div_payable": pl.Float64,
    "oth_payable": pl.Float64,
    "acc_exp": pl.Float64,
    "deferred_inc": pl.Float64,
    "st_bonds_payable": pl.Float64,
    "payable_to_reinsurer": pl.Float64,
    "rsrv_insur_cont": pl.Float64,
    "acting_trading_sec": pl.Float64,
    "acting_uw_sec": pl.Float64,
    "non_cur_liab_due_1y": pl.Float64,
    "oth_cur_liab": pl.Float64,
    "total_cur_liab": pl.Float64,
    "bond_payable": pl.Float64,
    "lt_payable": pl.Float64,
    "specific_payables": pl.Float64,
    "estimated_liab": pl.Float64,
    "defer_tax_liab": pl.Float64,
    "defer_inc_non_cur_liab": pl.Float64,
    "oth_ncl": pl.Float64,
    "total_ncl": pl.Float64,
    "depos_oth_bfi": pl.Float64,
    "deriv_liab": pl.Float64,
    "depos": pl.Float64,
    "agency_bus_liab": pl.Float64,
    "oth_liab": pl.Float64,
    "prem_receiv_adva": pl.Float64,
    "depos_received": pl.Float64,
    "ph_invest": pl.Float64,
    "reser_une_prem": pl.Float64,
    "reser_outstd_claims": pl.Float64,
    "reser_lins_liab": pl.Float64,
    "reser_lthins_liab": pl.Float64,
    "indept_acc_liab": pl.Float64,
    "pledge_borr": pl.Float64,
    "indem_payable": pl.Float64,
    "policy_div_payable": pl.Float64,
    "total_liab": pl.Float64,
    "treasury_share": pl.Float64,
    "ordin_risk_reser": pl.Float64,
    "forex_differ": pl.Float64,
    "invest_loss_unconf": pl.Float64,
    "minority_int": pl.Float64,
    "total_hldr_eqy_exc_min_int": pl.Float64,
    "total_hldr_eqy_inc_min_int": pl.Float64,
    "total_liab_hldr_eqy": pl.Float64,
    "lt_payroll_payable": pl.Float64,
    "oth_comp_income": pl.Float64,
    "oth_eqt_tools": pl.Float64,
    "oth_eqt_tools_p_shr": pl.Float64,
    "lending_funds": pl.Float64,
    "acc_receivable": pl.Float64,
    "st_fin_payable": pl.Float64,
    "payables": pl.Float64,
    "hfs_assets": pl.Float64,
    "hfs_sales": pl.Float64,
    "cost_fin_assets": pl.Float64,
    "fair_value_fin_assets": pl.Float64,
    "cip_total": pl.Float64,
    "oth_pay_total": pl.Float64,
    "long_pay_total": pl.Float64,
    "debt_invest": pl.Float64,
    "oth_debt_invest": pl.Float64,
    "oth_eq_invest": pl.Float64,
    "oth_illiq_fin_assets": pl.Float64,
    "oth_eq_ppbond": pl.Float64,
    "receiv_financing": pl.Float64,
    "use_right_assets": pl.Float64,
    "lease_liab": pl.Float64,
    "contract_assets": pl.Float64,
    "contract_liab": pl.Float64,
    "accounts_receiv_bill": pl.Float64,
    "accounts_pay": pl.Float64,
    "oth_rcv_total": pl.Float64,
    "fix_assets_total": pl.Float64,
    "update_flag": pl.Date,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
    "source": pl.Utf8,
}

INCOME_SCHEMA = {
    "symbol": pl.Utf8,
    "ann_date": pl.Date,
    "f_ann_date": pl.Date,
    "end_date": pl.Date,
    "report_type": pl.Float64,
    "comp_type": pl.Float64,
    "end_type": pl.Float64,
    "basic_eps": pl.Float64,
    "diluted_eps": pl.Float64,
    "total_revenue": pl.Float64,
    "revenue": pl.Float64,
    "int_income": pl.Float64,
    "prem_earned": pl.Float64,
    "comm_income": pl.Float64,
    "n_commis_income": pl.Float64,
    "n_oth_income": pl.Float64,
    "n_oth_b_income": pl.Float64,
    "prem_income": pl.Float64,
    "out_prem": pl.Float64,
    "une_prem_reser": pl.Float64,
    "reins_income": pl.Float64,
    "n_sec_tb_income": pl.Float64,
    "n_sec_uw_income": pl.Float64,
    "n_asset_mg_income": pl.Float64,
    "oth_b_income": pl.Float64,
    "fv_value_chg_gain": pl.Float64,
    "invest_income": pl.Float64,
    "ass_invest_income": pl.Float64,
    "forex_gain": pl.Float64,
    "total_cogs": pl.Float64,
    "oper_cost": pl.Float64,
    "int_exp": pl.Float64,
    "comm_exp": pl.Float64,
    "biz_tax_surchg": pl.Float64,
    "sell_exp": pl.Float64,
    "admin_exp": pl.Float64,
    "fin_exp": pl.Float64,
    "assets_impair_loss": pl.Float64,
    "prem_refund": pl.Float64,
    "compens_payout": pl.Float64,
    "reser_insur_liab": pl.Float64,
    "div_payt": pl.Float64,
    "reins_exp": pl.Float64,
    "oper_exp": pl.Float64,
    "compens_payout_refu": pl.Float64,
    "insur_reser_refu": pl.Float64,
    "reins_cost_refund": pl.Float64,
    "other_bus_cost": pl.Float64,
    "operate_profit": pl.Float64,
    "non_oper_income": pl.Float64,
    "non_oper_exp": pl.Float64,
    "nca_disploss": pl.Float64,
    "total_profit": pl.Float64,
    "income_tax": pl.Float64,
    "n_income": pl.Float64,
    "n_income_attr_p": pl.Float64,
    "minority_gain": pl.Float64,
    "oth_compr_income": pl.Float64,
    "t_compr_income": pl.Float64,
    "compr_inc_attr_p": pl.Float64,
    "compr_inc_attr_m_s": pl.Float64,
    "ebit": pl.Float64,
    "ebitda": pl.Float64,
    "insurance_exp": pl.Float64,
    "undist_profit": pl.Float64,
    "distable_profit": pl.Float64,
    "rd_exp": pl.Float64,
    "fin_exp_int_exp": pl.Float64,
    "fin_exp_int_inc": pl.Float64,
    "transfer_surplus_rese": pl.Float64,
    "transfer_housing_imprest": pl.Float64,
    "transfer_oth": pl.Float64,
    "adj_lossgain": pl.Float64,
    "withdra_legal_surplus": pl.Float64,
    "withdra_legal_pubfund": pl.Float64,
    "withdra_biz_devfund": pl.Float64,
    "withdra_rese_fund": pl.Float64,
    "withdra_oth_ersu": pl.Float64,
    "workers_welfare": pl.Float64,
    "distr_profit_shrhder": pl.Float64,
    "prfshare_payable_dvd": pl.Float64,
    "comshare_payable_dvd": pl.Float64,
    "capit_comstock_div": pl.Float64,
    "update_flag": pl.Date,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
    "source": pl.Utf8,
}

CASHFLOW_SCHEMA = {
    "symbol": pl.Utf8,
    "ann_date": pl.Date,
    "f_ann_date": pl.Date,
    "end_date": pl.Date,
    "comp_type": pl.Float64,
    "report_type": pl.Float64,
    "end_type": pl.Float64,
    "net_profit": pl.Float64,
    "finan_exp": pl.Float64,
    "c_fr_sale_sg": pl.Float64,
    "recp_tax_rends": pl.Float64,
    "n_depos_incr_fi": pl.Float64,
    "n_incr_loans_cb": pl.Float64,
    "n_inc_borr_oth_fi": pl.Float64,
    "prem_fr_orig_contr": pl.Float64,
    "n_incr_insured_dep": pl.Float64,
    "n_reinsur_prem": pl.Float64,
    "n_incr_disp_tfa": pl.Float64,
    "ifc_cash_incr": pl.Float64,
    "n_incr_disp_faas": pl.Float64,
    "n_incr_loans_oth_bank": pl.Float64,
    "n_cap_incr_repur": pl.Float64,
    "c_fr_oth_operate_a": pl.Float64,
    "c_inf_fr_operate_a": pl.Float64,
    "c_paid_goods_s": pl.Float64,
    "c_paid_to_for_empl": pl.Float64,
    "c_paid_for_taxes": pl.Float64,
    "n_incr_clt_loan_adv": pl.Float64,
    "n_incr_dep_cbob": pl.Float64,
    "c_pay_claims_orig_inco": pl.Float64,
    "pay_handling_chrg": pl.Float64,
    "pay_comm_insur_plcy": pl.Float64,
    "oth_cash_pay_oper_act": pl.Float64,
    "st_cash_out_act": pl.Float64,
    "n_cashflow_act": pl.Float64,
    "oth_recp_ral_inv_act": pl.Float64,
    "c_disp_withdrwl_invest": pl.Float64,
    "c_recp_return_invest": pl.Float64,
    "n_recp_disp_fiolta": pl.Float64,
    "n_recp_disp_sobu": pl.Float64,
    "stot_inflows_inv_act": pl.Float64,
    "c_pay_acq_const_fiolta": pl.Float64,
    "c_paid_invest": pl.Float64,
    "n_disp_subs_oth_biz": pl.Float64,
    "oth_pay_ral_inv_act": pl.Float64,
    "n_incr_pledge_loan": pl.Float64,
    "stot_out_inv_act": pl.Float64,
    "n_cashflow_inv_act": pl.Float64,
    "c_recp_borrow": pl.Float64,
    "proc_issue_bonds": pl.Float64,
    "oth_cash_recp_ral_fnc_act": pl.Float64,
    "stot_cash_in_fnc_act": pl.Float64,
    "free_cashflow": pl.Float64,
    "c_prepay_amt_borr": pl.Float64,
    "c_pay_dist_dpcp_int_exp": pl.Float64,
    "incl_dvd_profit_paid_sc_ms": pl.Float64,
    "oth_cashpay_ral_fnc_act": pl.Float64,
    "stot_cashout_fnc_act": pl.Float64,
    "n_cash_flows_fnc_act": pl.Float64,
    "eff_fx_flu_cash": pl.Float64,
    "n_incr_cash_cash_equ": pl.Float64,
    "c_cash_equ_beg_period": pl.Float64,
    "c_cash_equ_end_period": pl.Float64,
    "c_recp_cap_contrib": pl.Float64,
    "incl_cash_rec_saims": pl.Float64,
    "uncon_invest_loss": pl.Float64,
    "prov_depr_assets": pl.Float64,
    "depr_fa_coga_dpba": pl.Float64,
    "amort_intang_assets": pl.Float64,
    "lt_amort_deferred_exp": pl.Float64,
    "decr_deferred_exp": pl.Float64,
    "incr_acc_exp": pl.Float64,
    "loss_disp_fiolta": pl.Float64,
    "loss_scr_fa": pl.Float64,
    "loss_fv_chg": pl.Float64,
    "invest_loss": pl.Float64,
    "decr_def_inc_tax_assets": pl.Float64,
    "incr_def_inc_tax_liab": pl.Float64,
    "decr_inventories": pl.Float64,
    "decr_oper_payable": pl.Float64,
    "incr_oper_payable": pl.Float64,
    "others": pl.Float64,
    "im_net_cashflow_oper_act": pl.Float64,
    "conv_debt_into_cap": pl.Float64,
    "conv_copbonds_due_within_1y": pl.Float64,
    "fa_fnc_leases": pl.Float64,
    "im_n_incr_cash_equ": pl.Float64,
    "net_dism_capital_add": pl.Float64,
    "net_cash_rece_sec": pl.Float64,
    "credit_impa_loss": pl.Float64,
    "use_right_asset_dep": pl.Float64,
    "oth_loss_asset": pl.Float64,
    "end_bal_cash": pl.Float64,
    "beg_bal_cash": pl.Float64,
    "end_bal_cash_equ": pl.Float64,
    "beg_bal_cash_equ": pl.Float64,
    "update_flag": pl.Date,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
    "source": pl.Utf8,
}

FINAINDICATOR_SCHEMA = {
    "symbol": pl.Utf8,
    "ann_date": pl.Date,
    "end_date": pl.Date,
    "eps": pl.Float64,
    "dt_eps": pl.Float64,
    "total_revenue_ps": pl.Float64,
    "revenue_ps": pl.Float64,
    "capital_rese_ps": pl.Float64,
    "surplus_rese_ps": pl.Float64,
    "undist_profit_ps": pl.Float64,
    "extra_item": pl.Float64,
    "profit_dedt": pl.Float64,
    "gross_margin": pl.Float64,
    "current_ratio": pl.Float64,
    "quick_ratio": pl.Float64,
    "cash_ratio": pl.Float64,
    "ar_turn": pl.Float64,
    "ca_turn": pl.Float64,
    "fa_turn": pl.Float64,
    "assets_turn": pl.Float64,
    "op_income": pl.Float64,
    "ebit": pl.Float64,
    "ebitda": pl.Float64,
    "fcff": pl.Float64,
    "fcfe": pl.Float64,
    "current_exint": pl.Float64,
    "noncurrent_exint": pl.Float64,
    "interestdebt": pl.Float64,
    "netdebt": pl.Float64,
    "tangible_asset": pl.Float64,
    "working_capital": pl.Float64,
    "networking_capital": pl.Float64,
    "invest_capital": pl.Float64,
    "retained_earnings": pl.Float64,
    "diluted2_eps": pl.Float64,
    "bps": pl.Float64,
    "ocfps": pl.Float64,
    "retainedps": pl.Float64,
    "cfps": pl.Float64,
    "ebit_ps": pl.Float64,
    "fcff_ps": pl.Float64,
    "fcfe_ps": pl.Float64,
    "netprofit_margin": pl.Float64,
    "grossprofit_margin": pl.Float64,
    "cogs_of_sales": pl.Float64,
    "expense_of_sales": pl.Float64,
    "profit_to_gr": pl.Float64,
    "saleexp_to_gr": pl.Float64,
    "adminexp_of_gr": pl.Float64,
    "finaexp_of_gr": pl.Float64,
    "impai_ttm": pl.Float64,
    "gc_of_gr": pl.Float64,
    "op_of_gr": pl.Float64,
    "ebit_of_gr": pl.Float64,
    "roe": pl.Float64,
    "roe_waa": pl.Float64,
    "roe_dt": pl.Float64,
    "roa": pl.Float64,
    "npta": pl.Float64,
    "roic": pl.Float64,
    "roe_yearly": pl.Float64,
    "roa2_yearly": pl.Float64,
    "debt_to_assets": pl.Float64,
    "assets_to_eqt": pl.Float64,
    "dp_assets_to_eqt": pl.Float64,
    "ca_to_assets": pl.Float64,
    "nca_to_assets": pl.Float64,
    "tbassets_to_totalassets": pl.Float64,
    "int_to_talcap": pl.Float64,
    "eqt_to_talcapital": pl.Float64,
    "currentdebt_to_debt": pl.Float64,
    "longdeb_to_debt": pl.Float64,
    "ocf_to_shortdebt": pl.Float64,
    "debt_to_eqt": pl.Float64,
    "eqt_to_debt": pl.Float64,
    "eqt_to_interestdebt": pl.Float64,
    "tangibleasset_to_debt": pl.Float64,
    "tangasset_to_intdebt": pl.Float64,
    "tangibleasset_to_netdebt": pl.Float64,
    "ocf_to_debt": pl.Float64,
    "turn_days": pl.Float64,
    "roa_yearly": pl.Float64,
    "roa_dp": pl.Float64,
    "fixed_assets": pl.Float64,
    "profit_to_op": pl.Float64,
    "q_saleexp_to_gr": pl.Float64,
    "q_gc_to_gr": pl.Float64,
    "q_roe": pl.Float64,
    "q_dt_roe": pl.Float64,
    "q_npta": pl.Float64,
    "q_ocf_to_sales": pl.Float64,
    "basic_eps_yoy": pl.Float64,
    "dt_eps_yoy": pl.Float64,
    "cfps_yoy": pl.Float64,
    "op_yoy": pl.Float64,
    "ebt_yoy": pl.Float64,
    "netprofit_yoy": pl.Float64,
    "dt_netprofit_yoy": pl.Float64,
    "ocf_yoy": pl.Float64,
    "roe_yoy": pl.Float64,
    "bps_yoy": pl.Float64,
    "assets_yoy": pl.Float64,
    "eqt_yoy": pl.Float64,
    "tr_yoy": pl.Float64,
    "or_yoy": pl.Float64,
    "q_sales_yoy": pl.Float64,
    "q_op_qoq": pl.Float64,
    "equity_yoy": pl.Float64,
    "update_flag": pl.Date,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
    "source": pl.Utf8,
}

FORECAST_SCHEMA = {
    "symbol": pl.Utf8,
    "ann_date": pl.Date,
    "end_date": pl.Date,
    "type": pl.Utf8,
    "p_change_min": pl.Float64,
    "p_change_max": pl.Float64,
    "net_profit_min": pl.Float64,
    "net_profit_max": pl.Float64,
    "last_parent_net": pl.Float64,
    "first_ann_date": pl.Date,
    "summary": pl.Utf8,
    "change_reason": pl.Utf8,
    "update_flag": pl.Date,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
    "source": pl.Utf8,
}

EXPRESS_SCHEMA = {
    "symbol": pl.Utf8,
    "ann_date": pl.Date,
    "end_date": pl.Date,
    "revenue": pl.Float64,
    "operate_profit": pl.Float64,
    "total_profit": pl.Float64,
    "n_income": pl.Float64,
    "total_assets": pl.Float64,
    "total_hldr_eqy_exc_min_int": pl.Float64,
    "diluted_eps": pl.Float64,
    "diluted_roe": pl.Float64,
    "yoy_net_profit": pl.Float64,
    "bps": pl.Float64,
    "open_net_assets": pl.Float64,
    "open_bps": pl.Float64,
    "perf_summary": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
    "source": pl.Utf8,
}

REPORTRC_SCHEMA = {
    "symbol": pl.Utf8,
    "name": pl.Utf8,
    "report_date": pl.Date,
    "report_title": pl.Utf8,
    "report_type": pl.Utf8,
    "classify": pl.Utf8,
    "org_name": pl.Utf8,
    "author_name": pl.Utf8,
    "quarter": pl.Utf8,
    "op_rt": pl.Float64,
    "op_pr": pl.Float64,
    "tp": pl.Float64,
    "np": pl.Float64,
    "eps": pl.Float64,
    "pe": pl.Float64,
    "rd": pl.Float64,
    "roe": pl.Float64,
    "ev_ebitda": pl.Float64,
    "rating": pl.Utf8,
    "max_price": pl.Float64,
    "min_price": pl.Float64,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
    "source": pl.Utf8,
}

STKSURV_SCHEMA = {
    "symbol": pl.Utf8,
    "name": pl.Utf8,
    "surv_date": pl.Date,
    "fund_visitors": pl.Utf8,
    "rece_place": pl.Utf8,
    "rece_mode": pl.Utf8,
    "rece_org": pl.Utf8,
    "org_type": pl.Utf8,
    "comp_rece": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
    "source": pl.Utf8,
}

SHAREFLOATEXTERNAL_SCHEMA = {
    "symbol": pl.Utf8,
    "ann_date": pl.Date,
    "float_date": pl.Date,
    "float_share": pl.Float64,
    "float_ratio": pl.Float64,
    "holder_name": pl.Utf8,
    "share_type": pl.Utf8,
    "data_version": pl.Utf8,
    "fetched_at": FETCHED_AT_DTYPE,
    "source": pl.Utf8,
}

DATASET_SCHEMAS = {
    "namechange": NAMECHANGE_SCHEMA,
    "dividend": DIVIDEND_SCHEMA,
    "balancesheet": BALANCESHEET_SCHEMA,
    "income": INCOME_SCHEMA,
    "cashflow": CASHFLOW_SCHEMA,
    "fina_indicator": FINAINDICATOR_SCHEMA,
    "forecast": FORECAST_SCHEMA,
    "express": EXPRESS_SCHEMA,
    "report_rc": REPORTRC_SCHEMA,
    "stk_surv": STKSURV_SCHEMA,
    "share_float_external": SHAREFLOATEXTERNAL_SCHEMA,
    "instruments": INSTRUMENTS_SCHEMA,
    "trading_calendar": TRADING_CALENDAR_SCHEMA,
    "trading_status": TRADING_STATUS_SCHEMA,
    "daily_bars": DAILY_BARS_SCHEMA,
    "etf_bars": DAILY_BARS_SCHEMA,
    "fund_bars": DAILY_BARS_SCHEMA,
    "index_bars": {**DAILY_BARS_SCHEMA, "frequency": pl.Utf8},
    "minute_bars": MINUTE_BARS_SCHEMA,
    "minute_bars_5m": MINUTE_BARS_SCHEMA,
    "trade_ticks": TRADE_TICKS_SCHEMA,
    "commodity_bars": COMMODITY_BARS_SCHEMA,
    "corporate_actions": CORPORATE_ACTIONS_SCHEMA,
    "adj_factors": ADJ_FACTORS_SCHEMA,
    "financial_statement_items": FINANCIAL_STATEMENT_ITEMS_SCHEMA,
    "share_structure": SHARE_STRUCTURE_SCHEMA,
    "shareholder_counts": SHAREHOLDER_COUNTS_SCHEMA,
    "top_holders": TOP_HOLDERS_SCHEMA,
    "fund_flow": FUND_FLOW_SCHEMA,
    "margin_trading": MARGIN_TRADING_SCHEMA,
    "northbound_holdings": NORTHBOUND_HOLDINGS_SCHEMA,
    "northbound_flows": NORTHBOUND_FLOWS_SCHEMA,
    "valuation_metrics": VALUATION_METRICS_SCHEMA,
    "sector_members": SECTOR_MEMBERS_SCHEMA,
    "industry_index": INDUSTRY_INDEX_SCHEMA,
    "announcement_index": ANNOUNCEMENT_INDEX_SCHEMA,
    "earnings_disclosure_schedule": EARNINGS_DISCLOSURE_SCHEDULE_SCHEMA,
    "dragon_tiger": DRAGON_TIGER_SCHEMA,
    "block_trades": BLOCK_TRADES_SCHEMA,
    "index_constituents": INDEX_CONSTITUENTS_SCHEMA,
    "industry_members": INDUSTRY_MEMBERS_SCHEMA,
    "macro_indicators": MACRO_INDICATORS_SCHEMA,
    "market_breadth": MARKET_BREADTH_SCHEMA,
    "share_unlock_schedule": SHARE_UNLOCK_SCHEDULE_SCHEMA,
    "regulatory_events": REGULATORY_EVENTS_SCHEMA,
    "institutional_holdings": INSTITUTIONAL_HOLDINGS_SCHEMA,
    "analyst_consensus": ANALYST_CONSENSUS_SCHEMA,
    "sentiment_scores": SENTIMENT_SCORES_SCHEMA,
    "hot_rank": HOT_RANK_SCHEMA,
    "sector_bars": SECTOR_BARS_SCHEMA,
    "sector_fund_flow": SECTOR_FUND_FLOW_SCHEMA,
    "news_headlines": NEWS_HEADLINES_SCHEMA,
    "flash_news_wire": FLASH_NEWS_WIRE_SCHEMA,
    "sentiment_articles": SENTIMENT_ARTICLES_SCHEMA,
    "economic_calendar": ECONOMIC_CALENDAR_SCHEMA,
    "delisting_events": DELISTING_EVENTS_SCHEMA,
}

PRIMARY_KEYS = {
    "instruments": ["symbol"],
    "trading_calendar": ["trade_date"],
    "trading_status": ["symbol", "trade_date"],
    "daily_bars": ["symbol", "trade_date"],
    "etf_bars": ["symbol", "trade_date"],
    "fund_bars": ["symbol", "trade_date"],
    "index_bars": ["symbol", "trade_date", "frequency"],
    "minute_bars": ["symbol", "trade_date", "bar_time", "frequency"],
    "minute_bars_5m": ["symbol", "trade_date", "bar_time", "frequency"],
    # Not trade_time: it has no seconds, so a busy minute holds twenty records
    # sharing one. tick_seq is the only thing that separates them.
    "trade_ticks": ["symbol", "trade_date", "tick_seq"],
    "commodity_bars": ["symbol", "trade_date"],
    "corporate_actions": ["symbol", "ex_date", "action_type"],
    "adj_factors": ["symbol", "trade_date", "adjust_type"],
    # announce_date is part of the key, not an attribute of it: a restatement
    # republishes the same (period, item) with a new value on a new date, and
    # keying without the date lets the newer row destroy the original. That
    # silently rewrites history — a query as of a date before the restatement
    # would find the item missing entirely — and makes revision-based signals
    # impossible. With the date in the key, vintages accumulate and the reader
    # picks the latest one known as of the query date.
    "financial_statement_items": [
        "symbol",
        "report_period",
        "statement_type",
        "item_code",
        "announce_date",
    ],
    # announce_date is in the key for the same reason it is in FSI's: a
    # restatement republishes the same period, and overwriting in place would
    # make a query as of a date before the restatement find the old figure
    # gone rather than find the figure that was known then.
    "share_structure": ["symbol", "change_date", "announce_date"],
    "shareholder_counts": ["symbol", "count_date", "announce_date"],
    # holder_name is in the key because holder_rank is NOT unique: holders tied
    # on share count share a rank. 600010.SH 2025-06-30 rank 9 is both 博时基金
    # and 易方达基金 at 167,831,580 shares each. Keying without the name drops
    # one of them — 1,730 rows market-wide in that one period, and concentrated
    # in the parallel-vehicle holders (中证金融 et al.) worth noticing.
    "top_holders": [
        "symbol",
        "record_date",
        "holder_scope",
        "holder_rank",
        "holder_name",
        "announce_date",
    ],
    "fund_flow": ["symbol", "trade_date"],
    "margin_trading": ["symbol", "trade_date"],
    "northbound_holdings": ["symbol", "trade_date", "channel"],
    "northbound_flows": ["trade_date", "channel"],
    "valuation_metrics": ["symbol", "trade_date"],
    "sector_members": ["symbol", "sector_code", "as_of_date"],
    "industry_index": ["trade_date", "industry_code", "level", "weighting"],
    "announcement_index": ["announcement_id"],
    "earnings_disclosure_schedule": ["symbol", "report_period"],
    "dragon_tiger": ["symbol", "trade_date", "reason"],
    "block_trades": ["symbol", "trade_date", "price", "volume"],
    "index_constituents": ["index_symbol", "symbol", "as_of_date"],
    "industry_members": ["symbol", "classification_system", "as_of_date"],
    "macro_indicators": ["indicator_id", "obs_date"],
    "market_breadth": ["trade_date", "metric_id"],
    "share_unlock_schedule": ["symbol", "unlock_date"],
    "regulatory_events": ["event_id"],
    "institutional_holdings": ["symbol", "holder_type", "report_period"],
    "analyst_consensus": ["symbol", "forecast_date"],
    "sentiment_scores": ["symbol", "trade_date", "score_channel"],
    "hot_rank": ["symbol", "trade_date"],
    "sector_bars": ["sector_code", "trade_date"],
    "sector_fund_flow": ["sector_code", "trade_date"],
    "news_headlines": ["news_id"],
    "flash_news_wire": ["wire_id", "wire_source"],
    "sentiment_articles": ["article_id"],
    "economic_calendar": ["event_id"],
    "delisting_events": ["symbol"],
    # Tushare wide tables — not registered in DATASET_SCHEMAS (columns are
    # vendor-controlled and may change); validate_dataframe passes them through
    # untouched, so only PK matters for compact dedup.
    "balancesheet": ["symbol", "end_date", "report_type"],
    "income": ["symbol", "end_date", "report_type"],
    "cashflow": ["symbol", "end_date", "report_type"],
    "fina_indicator": ["symbol", "end_date"],
    "dividend": ["symbol", "end_date", "div_proc"],
    "share_float_external": ["symbol", "ann_date"],
    "namechange": ["symbol", "start_date"],
    "forecast": ["symbol", "ann_date", "end_date"],
    "express": ["symbol", "end_date"],
    "stk_surv": ["symbol", "surv_date"],
    "report_rc": ["symbol", "report_date", "org_name", "report_title"],
}


class SchemaValidationError(ValueError):
    """Raised when a DataFrame does not match the dataset contract."""


_CORE_BAR_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "daily_bars": ("open", "high", "low", "close", "volume"),
    "etf_bars": ("open", "high", "low", "close", "volume"),
    "fund_bars": ("open", "high", "low", "close", "volume"),
    "index_bars": ("open", "high", "low", "close", "volume"),
    "minute_bars": ("open", "high", "low", "close", "volume"),
    "minute_bars_5m": ("open", "high", "low", "close", "volume"),
    "commodity_bars": ("open", "high", "low", "close", "volume"),
    "sector_bars": ("open", "high", "low", "close", "volume"),
    "trade_ticks": ("price", "volume"),
}

# These fields are not part of a bar's numeric shape, but a null value is still
# semantically unusable.  In particular, Polars casts an invalid boolean to
# null; allowing that through would make calendar consumers treat an unknown
# session as non-trading and would make status history silently incomplete.
_CORE_SEMANTIC_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "trading_calendar": ("is_trading",),
    "trading_status": ("is_trading", "status"),
    "trade_ticks": ("trade_time", "direction"),
}


def required_columns_for_dataset(dataset: str, schema: dict[str, pl.DataType]) -> list[str]:
    """Return columns that must be present and non-null for a stored row.

    Most fundamental fields are legitimately nullable (for example PE for a
    loss-making company, or Sina's unavailable turnover). Primary keys and
    provenance are never nullable, while market-bar price/volume fields are
    required because a null there means the source row was malformed.
    """
    required = list(PRIMARY_KEYS.get(dataset, []))
    required.extend(col for col in PROVENANCE if col in schema and col not in required)
    required.extend(col for col in _CORE_BAR_REQUIRED_COLUMNS.get(dataset, ()) if col in schema)
    required.extend(
        col for col in _CORE_SEMANTIC_REQUIRED_COLUMNS.get(dataset, ()) if col in schema
    )
    return list(dict.fromkeys(required))


def _validate_bar_semantics(df: pl.DataFrame, dataset: str) -> None:
    """Reject impossible numeric market rows before they reach Parquet."""
    bar_datasets = {
        "daily_bars",
        "index_bars",
        "minute_bars",
        "minute_bars_5m",
        "commodity_bars",
        "sector_bars",
        "trade_ticks",
    }
    if dataset not in bar_datasets or df.is_empty():
        return

    checks: list[pl.Expr] = []
    price_cols = [col for col in ("open", "high", "low", "close", "price") if col in df.columns]
    checks.extend(pl.col(col) <= 0 for col in price_cols)
    if "volume" in df.columns:
        checks.append(pl.col("volume") < 0)
    if "amount" in df.columns:
        checks.append(pl.col("amount").is_not_null() & (pl.col("amount") < 0))

    if all(col in df.columns for col in ("open", "high", "low", "close")):
        # Intraday sources emit zero-volume carried-forward placeholders for
        # suspended names. Their OHLC fields can legitimately straddle the
        # stale close, so only enforce the candle envelope on rows that carry
        # an actual print. Positive-price and non-negative-volume checks above
        # still apply to every row.
        ohlc_row = (
            pl.col("volume").is_null() | (pl.col("volume") > 0)
            if "volume" in df.columns
            else pl.lit(True)
        )
        checks.extend(
            [
                ohlc_row & (pl.col("high") < pl.col("open")),
                ohlc_row & (pl.col("high") < pl.col("close")),
                ohlc_row & (pl.col("low") > pl.col("open")),
                ohlc_row & (pl.col("low") > pl.col("close")),
                ohlc_row & (pl.col("low") > pl.col("high")),
            ]
        )

    if not checks:
        return
    bad = df.filter(pl.any_horizontal(checks)).height
    if bad:
        raise SchemaValidationError(
            f"dataset '{dataset}': {bad} row(s) violate numeric market-data invariants"
        )


def _validate_finite_values(df: pl.DataFrame, dataset: str) -> None:
    """Reject NaN/Inf in any optional numeric field as well as required ones."""
    float_columns = [col for col, dtype in df.schema.items() if dtype in (pl.Float32, pl.Float64)]
    if not float_columns or df.is_empty():
        return
    bad_counts = df.select(
        [
            (pl.col(col).is_not_null() & ~pl.col(col).is_finite()).sum().alias(col)
            for col in float_columns
        ]
    ).row(0, named=True)
    invalid = {col: int(count) for col, count in bad_counts.items() if count}
    if invalid:
        detail = ", ".join(f"{col}={count}" for col, count in invalid.items())
        raise SchemaValidationError(
            f"dataset '{dataset}': non-finite numeric values are not allowed: {detail}"
        )


def validate_dataframe(
    df: pl.DataFrame,
    dataset: str,
    *,
    allow_missing_optional: bool = False,
) -> pl.DataFrame:
    """Cast and validate *df* against the curated schema for *dataset*.

    Writers use the strict default: every registered column must be present.
    Historical audits may set ``allow_missing_optional`` because old Parquet
    files can legitimately predate a nullable column that was added to the
    schema. Primary keys, provenance, and core bar fields remain mandatory in
    that mode; a file that lacks one of those is still invalid.
    """
    schema = DATASET_SCHEMAS.get(dataset)
    if schema is None:
        return df

    if df.is_empty():
        return pl.DataFrame(schema=schema)

    missing = [col for col in schema if col not in df.columns]
    if allow_missing_optional:
        required = required_columns_for_dataset(dataset, schema)
        missing_required = [col for col in required if col not in df.columns]
        if missing_required:
            raise SchemaValidationError(
                f"dataset '{dataset}': missing required columns {missing_required}"
            )
        columns = [col for col in schema if col in df.columns]
    else:
        columns = list(schema)
    if missing and not allow_missing_optional:
        raise SchemaValidationError(f"dataset '{dataset}': missing columns {missing}")

    casts = []
    for col in columns:
        dtype = schema[col]
        if isinstance(dtype, pl.Datetime) and df.schema[col] == pl.Utf8:
            casts.append(
                pl.col(col)
                .str.to_datetime(time_unit=dtype.time_unit, time_zone=dtype.time_zone, strict=False)
                .alias(col)
            )
        elif dtype == pl.Date and df.schema[col] == pl.Utf8:
            casts.append(pl.col(col).str.to_date(strict=False).alias(col))
        else:
            casts.append(pl.col(col).cast(dtype, strict=False))
    try:
        normalized = df.with_columns(casts).select(columns)
    except pl.exceptions.PolarsError as exc:
        # ``strict=False`` turns many bad scalar casts into nulls, but Polars
        # still raises for some Utf8 -> Boolean conversions. Keep all schema
        # failures on the domain error boundary so writers and audits report a
        # consistent contract violation instead of leaking an engine-specific
        # exception.
        raise SchemaValidationError(
            f"dataset '{dataset}': values cannot be cast to the registered schema: {exc}"
        ) from exc

    required = [col for col in required_columns_for_dataset(dataset, schema) if col in columns]
    if required:
        null_counts = normalized.select(
            [pl.col(col).null_count().alias(col) for col in required]
        ).row(0, named=True)
        missing_values = {col: int(count) for col, count in null_counts.items() if count}
        if missing_values:
            detail = ", ".join(f"{col}={count}" for col, count in missing_values.items())
            raise SchemaValidationError(
                f"dataset '{dataset}': required columns contain null or unparseable values: {detail}"
            )

        string_required = [col for col in required if normalized.schema[col] == pl.Utf8]
        if string_required:
            blank_counts = normalized.select(
                [pl.col(col).str.strip_chars().eq("").sum().alias(col) for col in string_required]
            ).row(0, named=True)
            blank_values = {col: int(count) for col, count in blank_counts.items() if count}
            if blank_values:
                detail = ", ".join(f"{col}={count}" for col, count in blank_values.items())
                raise SchemaValidationError(
                    f"dataset '{dataset}': required string columns cannot be blank: {detail}"
                )

    _validate_finite_values(normalized, dataset)
    _validate_bar_semantics(normalized, dataset)
    return normalized


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def with_provenance(df: pl.DataFrame, source: str, data_version: str) -> pl.DataFrame:
    # An adapter may pre-set `source` (e.g. MOCK_SOURCE) to flag row origin;
    # that marker must survive normalization.
    cols = [
        pl.lit(data_version).alias("data_version"),
        pl.lit(datetime.now(timezone.utc)).cast(FETCHED_AT_DTYPE).alias("fetched_at"),
    ]
    if "source" not in df.columns:
        cols.append(pl.lit(source).alias("source"))
    return df.with_columns(cols)
