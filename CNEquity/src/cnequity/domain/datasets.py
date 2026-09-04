"""Single source of truth for dataset metadata (DatasetSpec registry).

Every module that needs per-dataset knowledge — compact partitioning, watermark
policy, fetch semantics, query date columns, DuckDB views, audit — derives it
from ``DATASETS`` below. Schema and primary keys live in
``domain/schemas.py`` (polars dtypes); ``test_dataset_registry.py`` asserts the
two stay in sync.

Adding a dataset = one ``DatasetSpec`` entry here + schema/PK in schemas.py +
a registered step.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

from cnequity.domain.partitions import Granularity, partition_value

FetchSemantics = Literal["by_date", "snapshot", "rolling_window"]
HistoryMode = Literal["by_date", "snapshot_with_backfill", "snapshot_only"]
Layer = Literal["curated", "derived", "external"]

# Publication cadence — controls how often ``cne run daily`` actually fetches
# a dataset.  ``daily`` fetches every trading session; ``monthly`` and
# ``quarterly`` skip the fetch (but still advance the watermark) when the
# watermark is in the same period as ``trade_date``; ``skip`` never fetches
# (source retired or middleware limitation, historical data in curated).
Cadence = Literal["daily", "monthly", "quarterly", "skip"]

# Research-use classification, orthogonal to ``Layer`` (which is a storage
# location). L0 is the reference spine everything joins on and L8 the risk
# overlay; the ordering is roughly "how far from the price series".
Tier = Literal["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"]

TIERS: tuple[Tier, ...] = ("L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8")

TIER_LABELS: dict[Tier, str] = {
    "L0": "基础参考",
    "L1": "行情",
    "L2": "公司事件",
    "L3": "基本面",
    "L4": "资金面",
    "L5": "结构行业",
    "L6": "宏观",
    "L7": "舆情 / 轮动",
    "L8": "风险合规",
}


@dataclass(frozen=True)
class DatasetSpec:
    """Orchestration/query metadata for one dataset.

    tier:
        L0–L8 research classification (see ``Tier``). Mandatory and without a
        default: an unclassified dataset would silently fall into whichever
        bucket the default named, and this is what groups the catalog docs and
        the lake dashboard. ``layer`` is where the parquet lives; this is what
        the data is *for*, and the two are independent — ``adj_factors`` is
        derived storage but L1 research input.

    partition_col:
        Hive partition directory key under the lake (None = merge-style single
        file, e.g. instruments).
    partition_granularity:
        Period each partition directory covers — ``day``, ``month``, ``quarter``
        or ``year``. ``quarter`` is for the ``report_period`` datasets, whose
        directories are the reporting periods themselves (``2016Q1``) rather
        than a period chosen for file size.
        Pick it from rows per day, not from habit: a Parquet footer costs ~1KB
        whatever it holds, so a dataset with a handful of rows a day spends
        almost all its bytes and all its file opens on metadata. Rough bands
        used here: ≥1000 rows/day → ``day``, 50–1000 → ``month``, <50 → ``year``.
        Only ``day`` values can be hive-parsed (see domain/partitions.py).
    date_col:
        Column used for query date-range filters; defaults to ``partition_col``.
    fetch_semantics:
        ``by_date`` — source returns values for a requested day (gap catch-up
        allowed). ``snapshot`` — live page stamped with trade_date; historical
        replay would forge rows, so only the run day is ever fetched.
        ``rolling_window`` — a live page, but the vendor serves a *rolling
        history window*: requesting past dates within
        ``history_horizon_days`` returns genuine historical rows, so backfill
        may walk the reachable window (news_headlines: the 7×24 wire list
        pages backwards ~2 weeks). Outside the horizon it behaves exactly like
        ``snapshot`` — never forge rows the source did not serve.
    watermark:
        Maintain a date watermark under ``meta/state`` (False for datasets
        partitioned by non-date keys like report_period).
    pit:
        Point-in-time dataset — ``load()`` requires ``as_of`` and filters on
        ``announce_date``.
    backfill_source:
        Name of an external historical source that can replay this dataset even
        though daily ``fetch_semantics`` is ``snapshot`` (e.g. valuation_metrics:
        EastMoney live snapshot daily, baostock for history). ``cne backfill``
        is allowed for snapshot datasets only when this is set.
    required:
        When False, an empty curated root is a warning (not an error) and does
        not alone make ``lake_health`` UNHEALTHY. Use for registered datasets
        whose source is not yet wired or is temporarily unavailable.
    history_horizon_days:
        Trading days of history the *source* still serves, counted back from
        today. ``None`` (the default) means the source has no such limit and
        history is bounded only by what has been backfilled.

        This is a property of the vendor, not of this lake, and it is the
        difference between "not fetched yet" and "can never be fetched". Asking
        for 2016 intraday does not return less data, it returns none, and no
        backfill source extends it — ``by_date`` alone would promise a decade.

        The vendor caps a **bar count** per symbol, not a date: ~22,800 bars of
        1m and ~23,568 of 5m. This field is that count divided by a full
        session, so it holds for any instrument quoted every session — which is
        every A-share stock, and what these datasets are for. An instrument
        that only has bars on scattered days reaches proportionally further
        back (162107.SZ, a barely-traded LOF, holds 3,216 5m bars spread over
        67 days and so reaches 2012). Treat it as the guarantee for a normal
        stock, not as a hard ceiling for every symbol.

        Use ``history_floor_date`` instead when the source's edge is a *date*
        rather than a count — the two are different mechanisms and only one of
        them moves with today.
    history_floor_date:
        Earliest date the source serves, as a fixed calendar date. Wins over
        ``history_horizon_days`` when both are set.

        The distinction matters because a rolling count and a fixed floor
        diverge every day. TDX serves ``trade_ticks`` back to exactly
        2024-01-02 for *every* symbol, across both exchanges and every
        liquidity band (measured 2026-08-02; 2023-12-28 is empty). Expressing
        that as "624 trading days" would be true on the day it was measured and
        wrong the day after: ``earliest_available`` would walk the floor
        forward and refuse windows the source still happily serves.
    source_retired_date:
        Last date the source ever published, for a feed that has stopped. The
        upper bound to ``history_floor_date``'s lower one.

        Without it a retired feed is indistinguishable from a broken pipeline:
        its watermark freezes, ``is_stale`` calls it stale forever, and
        ``cne verify`` offers a backfill that runs the whole window, writes zero
        rows, and leaves the identical gap behind. Both are the same wrong
        answer — "you are missing data" — to a dataset that holds everything
        that exists.

        Set it to the last session with real values, not the first without.
        ``northbound_flows`` is 2024-08-16: the exchanges stopped publishing
        daily northbound net flow after that, and every row from 2024-08-19 on
        carries a null amount (see adapters/eastmoney/capital.py).
    """

    name: str
    tier: Tier
    layer: Layer = "curated"
    partition_col: str | None = None
    partition_granularity: Granularity = "day"
    date_col: str | None = None
    fetch_semantics: FetchSemantics = "by_date"
    watermark: bool = True
    pit: bool = False
    # Which upstream the daily path actually reads, and what it falls back to.
    # Kept in the registry rather than in prose because the prose drifted: the
    # published source table had `sector_bars` on EastMoney long after it moved
    # to 同花顺, and `fund_flow` in the wrong schedule group entirely. Anything
    # generated from here — the README data table, the docs — cannot say
    # something the code does not do, and a test asserts the pairing.
    primary_source: str = ""
    backup_source: str | None = None
    backfill_source: str | None = None
    # How many days the freshest data may lag the last trading day before it is
    # flagged STALE. 1 tolerates normal T+1 EOD publication; larger values mark
    # sources with a slower cadence (margin T+1, quarterly northbound holdings)
    # so their inherent lag is not mistaken for a stuck pipeline.
    max_staleness_days: int = 1
    # Publication cadence — see ``Cadence``.  Controls whether ``cne run daily``
    # fetches this dataset on every run or skips it until the next cadence
    # period boundary.  Step functions call ``should_fetch()`` to check.
    cadence: Cadence = "daily"
    required: bool = True
    history_horizon_days: int | None = None
    history_floor_date: date | None = None
    source_retired_date: date | None = None
    # Calendar days of history one backfill sub-run may cover. None = one run
    # for the whole window, which is what every daily-cadence dataset wants.
    #
    # Set it where a full window's staging would not fit in memory: compact
    # reads *every* staging file of a run into one frame. Prefer
    # ``backfill_chunk_symbols`` for tip-paged sources (see below) — date
    # slices re-walk the same tip→start pages on every chunk.
    backfill_chunk_days: int | None = None
    # Symbols per backfill sub-run for tip-paged sources (TDX intraday pages
    # backwards from today). None = do not symbol-chunk.
    #
    # Date-chunking a tip-paged source is catastrophically wasteful: each
    # slice still has to walk tip → slice_start before any in-window row
    # appears, so a 15-slice CSI300 1m seed paid ~8× the wire traffic of one
    # tip→horizon walk. Symbol chunks keep one walk per name and still bound
    # compact memory (200 × 240 × 95 ≈ 4.6M rows of 1m per sub-run).
    backfill_chunk_symbols: int | None = None
    # Bar frequency for intraday datasets ("1m", "5m"), None for everything
    # else. One dataset holds exactly one frequency, so this is also what marks
    # a dataset as intraday — steps, audit checks and the reader all derive the
    # set from here rather than each keeping its own list of names.
    #
    # It is one-per-dataset because ``history_horizon_days`` is one-per-dataset
    # and the two disagree: TDX keeps 95 trading days of 1m but 491 of 5m. So
    # is the watermark, and so is ``coverage_start``. A single dataset holding
    # both frequencies could not answer "how far back does this go" truthfully
    # for either of them.
    intraday_frequency: str | None = None
    # What one row covers, when that is finer than a trading day: "1m", "5m",
    # "tick". Purely descriptive — nothing fetches, checks or reads differently
    # because of it, which is exactly why `trade_ticks` can carry it while
    # deliberately leaving `intraday_frequency` unset.
    #
    # The two exist separately because they answer different questions.
    # `intraday_frequency` means "this dataset holds bars at this frequency",
    # and every consumer of it assumes a `bar_time` column and a bar count per
    # session. Transaction records have neither, so inheriting those code paths
    # would give them checks that pass on the wrong column. But a reader
    # scanning the catalog still needs to see that this is intraday data, and
    # without this field the dashboard showed it a dash — indistinguishable
    # from a daily dataset.
    #
    # Where both are set they must agree; `test_dataset_registry` enforces it.
    row_grain: str | None = None
    # Whether the dataset promises at least one row on every exchange session
    # in its covered span. `fetch_semantics="by_date"` only describes how the
    # source is queried; event and announcement feeds are still sparse.
    coverage_mode: Literal["sparse", "session_dense"] = "sparse"
    # Short Chinese description for the dashboard.  Empty string means the
    # dashboard will fall back to showing only the dataset name.
    description: str = ""
    # --- External compaction (plugin adapter write-back) -------------------
    # When ``layer="external"`` and ``compactable=True``, the daily run's
    # ``compact`` step merges staging into the adapter's own files rather than
    # into ``curated/``.  The adapter decides where files live and how they are
    # laid out (hive partitions vs. yearly files); CNE just drives staging →
    # compact and records the manifest/watermark.
    compactable: bool = False
    # Logical adapter name for auto-discovery (matches the ``[external_adapters.<name>]``
    # config key).  When set, the registry can discover write-capable adapters
    # without hardcoded imports.  Empty = read-only or curated/derived dataset.
    adapter_name: str = ""

    @property
    def query_date_col(self) -> str | None:
        return self.date_col or self.partition_col

    def earliest_available(self, today: date, *, trading_days_per_year: int = 242) -> date | None:
        """Rough calendar date before which the source serves nothing.

        Two mechanisms, and they must not be confused. A fixed
        ``history_floor_date`` is a date the vendor keeps back to and does not
        move as today does — it is returned as-is. ``history_horizon_days`` is
        a per-symbol retention *count*, expressed in trading days because that
        is how the vendor caps it, and converted with the usual ~242 sessions a
        year. Deliberately approximate and deliberately early: it guards a CLI
        window, and refusing a window the source would in fact have served is
        worse than fetching a few empty days.
        """
        if self.history_floor_date is not None:
            return self.history_floor_date
        if self.history_horizon_days is None:
            return None
        calendar_days = round(self.history_horizon_days * 365 / trading_days_per_year)
        return date.fromordinal(max(1, today.toordinal() - calendar_days))

    def partition_for(self, d: date) -> str:
        """Directory value of the partition holding *d* for this dataset."""
        return partition_value(d, self.partition_granularity)


_SPECS = [
    # L0 reference
    # Live sources (TDX/EM) only list what trades today; baostock's stock_basic
    # is what recovers delisted codes, so `cne backfill instruments` is the only
    # path to a survivorship-free universe.
    DatasetSpec(
        "instruments",
        primary_source="tdx_protocol",
        backup_source="baostock",
        tier="L0",
        partition_col=None,
        watermark=False,
        backfill_source="baostock",
        description="A股全量标的清单（含退市）",
    ),
    DatasetSpec(
        "trading_calendar",
        primary_source="tdx_protocol",
        backup_source="exchange",
        tier="L0",
        partition_col="trade_date",
        partition_granularity="year",
        description="交易日历",
    ),
    DatasetSpec(
        "trading_status",
        primary_source="tdx_protocol",
        backup_source="eastmoney",
        tier="L0",
        partition_col="trade_date",
        partition_granularity="month",
        # EastMoney's daily ST board is a live current-state snapshot; the
        # dedicated backfill path uses baostock history.  Marking the daily
        # contract as by_date would replay today's labels into missed past
        # sessions when the watermark falls behind.
        fetch_semantics="snapshot",
        backfill_source="baostock",
        description="停复牌/ST状态",
    ),
    # L1 bars
    DatasetSpec(
        "daily_bars",
        primary_source="tushare_wide",
        tier="L1",
        layer="external",
        partition_col="trade_date",
        coverage_mode="session_dense",
        compactable=True,
        adapter_name="tushare_wide",
        description="日K线（开高低收量额）",
    ),
    DatasetSpec(
        "index_bars",
        primary_source="tdx_protocol",
        backup_source="eastmoney",
        tier="L1",
        partition_col="trade_date",
        partition_granularity="year",
        coverage_mode="session_dense",
        description="指数日K线",
    ),
    # 1-minute bars. Day partitions: ~240 bars × the configured scope, which is
    # 1.3M rows a day at full market — the top of the ≥1000 rows/day band, and
    # ~30MB a partition. The schema draft once sketched
    # frequency/trade_date/symbol_bucket; a second directory level buys nothing
    # at that size and every partition-aware module here assumes exactly one.
    #
    # Opt-in (required=False): this is not on the default daily waves and a lake
    # that never enabled it must not be judged unhealthy for holding no rows.
    DatasetSpec(
        "minute_bars",
        primary_source="tdx_protocol",
        tier="L1",
        partition_col="trade_date",
        partition_granularity="day",
        fetch_semantics="by_date",
        required=False,
        coverage_mode="session_dense",
        # Measured 2026-08-01 against 120.76.1.198:7709 — 22,800 bars for every
        # symbol probed, across both exchanges and every liquidity band, so it
        # is a server retention window rather than a per-symbol artefact.
        history_horizon_days=95,
        # Tip-paged: chunk by symbol, not by date (see backfill_chunk_symbols).
        backfill_chunk_symbols=200,
        intraday_frequency="1m",
        row_grain="1m",
        description="1分钟K线",
    ),
    # 5-minute bars — a separate dataset, not a `frequency` value inside
    # minute_bars, because the horizon differs by 5× and a dataset carries one
    # watermark and one coverage_start (see DatasetSpec.intraday_frequency).
    #
    # This is the only intraday frequency with real history: two years against
    # 1m's four and a half months, at a fifth of the volume (~7MB a day at full
    # market). For most research it is the more useful of the two, which is why
    # it is registered rather than left as a resampling exercise.
    #
    # 15m/30m/60m are deliberately absent. TDX serves them over the same 491-day
    # window, but they aggregate exactly from 5m (48 bars divide by 3, 6 and 12
    # onto identical closing-minute boundaries), so storing them would be three
    # more datasets holding a `group_by_dynamic` away from data already here.
    DatasetSpec(
        "minute_bars_5m",
        primary_source="tdx_protocol",
        tier="L1",
        partition_col="trade_date",
        partition_granularity="day",
        fetch_semantics="by_date",
        required=False,
        coverage_mode="session_dense",
        # Measured 2026-08-01: 23,568 bars = 491 trading days, back to
        # 2024-07-23. 15m/30m/60m share exactly that window (7,856 / 3,928 /
        # 1,964 bars), which is what says it is a time-based retention policy
        # rather than a bar-count cap.
        history_horizon_days=491,
        # Same tip-paged contract as 1m; 200 symbols × 48 bars × 491 days ≈
        # 4.7M rows per sub-run — comparable compact memory to 1m's chunk.
        backfill_chunk_symbols=200,
        intraday_frequency="5m",
        row_grain="5m",
        description="5分钟K线",
    ),
    # Transaction records (分笔). Not tick data: A-share Level-1 is a 3-second
    # snapshot, so a record aggregates 6–33 real trades (measured) and a
    # session holds ~2,700 on average, ~4,800 at most.
    #
    # Deliberately *not* `intraday_frequency`. That field means "one bar
    # frequency", and every consumer of it — the audit's session-shape checks,
    # the reader's adjustment set, `cne backfill --symbols` — assumes a
    # `bar_time` column and a bar count per session. This dataset has neither,
    # and inheriting those code paths would give it checks that silently pass
    # on the wrong column. It carries its own step group and its own checks.
    DatasetSpec(
        "trade_ticks",
        primary_source="tdx_protocol",
        tier="L1",
        partition_col="trade_date",
        partition_granularity="day",
        fetch_semantics="by_date",
        required=False,
        coverage_mode="session_dense",
        # A *date*, not a rolling count — see DatasetSpec.history_floor_date.
        # Measured 2026-08-02: every symbol probed serves back to exactly
        # 2024-01-02 and no further, which is ~624 trading days and growing.
        # The edge landing on a calendar boundary suggests the retention may be
        # year-granular rather than a fixed date, so re-measure each January
        # with scripts/probe_trade_ticks.py.
        history_floor_date=date(2024, 1, 2),
        # By-date requests, so date chunks are the cheap axis — the exact
        # opposite of the minute bars above, where the wire always walks from
        # today's tip and a date slice re-fetches everything newer than it.
        # 5 days × 200 symbols × ~2,700 rows ≈ 2.7M rows per sub-run.
        backfill_chunk_days=5,
        # Intraday, but not a bar frequency — see DatasetSpec.row_grain for why
        # this is not `intraday_frequency`.
        row_grain="tick",
        description="分笔成交（3秒快照）",
    ),
    # Domestic commodity futures main-continuous (东财主连) + narrow offshore
    # gold (Sina COMEX ``GC0.CMX``); not A-share equity.
    DatasetSpec(
        "commodity_bars",
        primary_source="sina",
        backup_source="eastmoney",
        tier="L1",
        partition_col="trade_date",
        partition_granularity="year",
        fetch_semantics="by_date",
        backfill_source="eastmoney_kline+sina_global",
        required=False,
        max_staleness_days=2,
        description="商品期货主连（黄金等）",
    ),
    # L2 corporate events
    DatasetSpec(
        "corporate_actions",
        primary_source="tdx_protocol",
        backup_source="eastmoney",
        tier="L2",
        partition_col="ex_date",
        partition_granularity="year",
        description="除权除息/送转/配股",
    ),
    DatasetSpec(
        "announcement_index",
        primary_source="cninfo",
        tier="L2",
        partition_col="announce_date",
        pit=True,
        description="巨潮公告索引",
    ),
    # Current-state timetable (revisions overwrite scheduled_date; not PIT).
    DatasetSpec(
        "earnings_disclosure_schedule",
        primary_source="eastmoney",
        tier="L2",
        partition_col="report_period",
        partition_granularity="quarter",
        watermark=False,
        cadence="quarterly",
        description="财报预约披露时间表",
    ),
    # L3 fundamentals
    DatasetSpec(
        "financial_statement_items",
        primary_source="eastmoney",
        tier="L3",
        partition_col="report_period",
        partition_granularity="quarter",
        watermark=False,
        pit=True,
        cadence="quarterly",
        description="财报长表（三大表科目）",
    ),
    # Shareholder structure — the dimensions the long-format statement table
    # cannot hold. `top_holders` is a ranked repeating group of ten, which no
    # amount of `item_code` rows expresses; the other two are wide fixed
    # records that would only be item_codes by accident of shape.
    #
    # All three are PIT for the reason FSI is: a 半年报 shareholder list is
    # dated 06-30 and disclosed in late August, so keying it by period alone
    # would let a July backtest read August's filing.
    DatasetSpec(
        "share_structure",
        primary_source="eastmoney",
        tier="L3",
        partition_col="change_date",
        partition_granularity="year",
        date_col="change_date",
        # Measured 2026-08 against RPT_F10_EH_EQUITY: 1990 serves 19 rows, and
        # nothing before it. A fixed vendor floor, not a rolling budget.
        history_floor_date=date(1990, 1, 1),
        watermark=False,
        pit=True,
        cadence="quarterly",
        description="股本结构变动",
    ),
    DatasetSpec(
        "shareholder_counts",
        primary_source="eastmoney",
        tier="L3",
        partition_col="count_date",
        partition_granularity="year",
        date_col="count_date",
        # RPT_F10_EH_HOLDERNUM: 25 rows in 1992, none in 1990/1991.
        history_floor_date=date(1992, 1, 1),
        watermark=False,
        pit=True,
        cadence="quarterly",
        description="股东户数",
    ),
    DatasetSpec(
        "top_holders",
        primary_source="eastmoney",
        tier="L3",
        partition_col="record_date",
        partition_granularity="year",
        date_col="record_date",
        # 2003, and the binding constraint is PIT rather than availability.
        # RPT_F10_EH_HOLDERS reaches back to the 1990s, but it carries no
        # NOTICE_DATE and borrows its disclosure date from
        # RPT_F10_EH_FREEHOLDERS — which starts in 2003 (0 rows in 1999-2002,
        # 13,853 in 2003). Before that the total-scope rows have nothing to
        # borrow from and are dropped as undated, so a backfill reaching
        # further back fetches ~112k rows across four years and writes none.
        history_floor_date=date(2003, 1, 1),
        watermark=False,
        pit=True,
        cadence="quarterly",
        description="十大流通股东",
    ),
    DatasetSpec(
        "valuation_metrics",
        primary_source="eastmoney",
        tier="L3",
        partition_col="trade_date",
        fetch_semantics="snapshot",
        backfill_source="baostock",
        description="估值指标（PE/PB/PCF等）",
    ),
    DatasetSpec(
        "analyst_consensus",
        primary_source="eastmoney",
        tier="L3",
        partition_col="forecast_date",
        fetch_semantics="snapshot",
        description="分析师一致预期",
    ),
    # L4 capital flows
    DatasetSpec(
        "fund_flow",
        primary_source="eastmoney",
        tier="L4",
        partition_col="trade_date",
        fetch_semantics="snapshot",
        # Live clist snapshot cannot replay history, but the Tushare
        # moneyflow fallback/primary CAN — that is what unlocks `cne backfill`.
        backfill_source="tushare",
        description="个股资金流向",
    ),
    DatasetSpec(
        "margin_trading",
        primary_source="eastmoney",
        tier="L4",
        partition_col="trade_date",
        # Exchange publishes session T's balances on the T+1 morning, so the
        # freshest possible evening mark is the previous trading day — a
        # Monday-night run legitimately holds Friday. 4 days covers the
        # weekend plus that publication lag without masking real outages.
        max_staleness_days=4,
        description="融资融券",
    ),
    # Per-stock northbound holdings are quarterly since Aug 2024; tolerate the
    # gap to the next quarter-end before flagging stale.
    DatasetSpec(
        "northbound_holdings",
        primary_source="eastmoney",
        tier="L4",
        partition_col="trade_date",
        max_staleness_days=100,
        description="陆股通持股（季报）",
    ),
    DatasetSpec(
        "northbound_flows",
        primary_source="eastmoney",
        tier="L4",
        partition_col="trade_date",
        partition_granularity="year",
        max_staleness_days=2,
        required=False,
        # The northbound channel opened on 2014-11-17. Earlier dates are not
        # missing lake rows: the source had not published this feed yet.
        history_floor_date=date(2014, 11, 17),
        # The exchanges stopped publishing daily northbound net flow after this
        # session; every row from 2024-08-19 on carries a null amount, and those
        # are dropped rather than zero-filled. The lake holds everything that
        # exists, so this is not staleness and no backfill can change it.
        source_retired_date=date(2024, 8, 16),
        cadence="skip",
        description="北向资金每日净流入（已停产）",
    ),
    DatasetSpec(
        "dragon_tiger",
        primary_source="eastmoney",
        tier="L4",
        partition_col="trade_date",
        partition_granularity="month",
        description="龙虎榜",
    ),
    DatasetSpec(
        "block_trades",
        primary_source="eastmoney",
        tier="L4",
        partition_col="trade_date",
        partition_granularity="month",
        description="大宗交易",
    ),
    DatasetSpec(
        "institutional_holdings",
        primary_source="eastmoney",
        tier="L4",
        partition_col="report_period",
        partition_granularity="quarter",
        watermark=False,
        cadence="quarterly",
        description="机构持股汇总",
    ),
    # L5 structure
    DatasetSpec(
        "sector_members",
        primary_source="eastmoney",
        tier="L5",
        partition_col="as_of_date",
        fetch_semantics="snapshot",
        description="板块成分股",
    ),
    DatasetSpec(
        "index_constituents",
        primary_source="eastmoney",
        tier="L5",
        partition_col="as_of_date",
        partition_granularity="month",
        fetch_semantics="snapshot",
        # CNI adjustment history reconstructs 399001/399006 from ~2021-12;
        # CSI indices still accumulate via daily EM snapshots only.
        backfill_source="cni",
        cadence="monthly",
        # Monthly cadence skips same-month refetches, so the day-based
        # default tolerance (1) would flag it STALE all month long.
        max_staleness_days=35,
        description="指数成分股快照",
    ),
    DatasetSpec(
        "industry_members",
        primary_source="eastmoney",
        tier="L5",
        partition_col="as_of_date",
        fetch_semantics="snapshot",
        # Shenwan StockClassifyUse intervals → monthly as_of from 2020.
        backfill_source="sw",
        cadence="monthly",
        # Same monthly-vs-daily-tolerance mismatch as index_constituents.
        max_staleness_days=35,
        description="申万行业分类",
    ),
    # L6 macro
    DatasetSpec(
        "macro_indicators",
        primary_source="eastmoney",
        backup_source="pboc",
        tier="L6",
        partition_col="obs_date",
        partition_granularity="year",
        description="宏观经济指标",
    ),
    DatasetSpec(
        "market_breadth",
        primary_source="derived",
        tier="L6",
        partition_col="trade_date",
        partition_granularity="year",
        # Derived from daily_bars and emits one metric set for every session
        # with a current and prior bar; a missing interior session is a
        # derivation gap, not the natural sparsity of an event feed.
        coverage_mode="session_dense",
        description="市场宽度（涨跌家数等）",
    ),
    # L7 sentiment / rotation
    DatasetSpec(
        "sentiment_scores",
        primary_source="derived",
        backup_source="eastmoney",
        tier="L7",
        partition_col="trade_date",
        partition_granularity="month",
        description="情绪得分",
    ),
    DatasetSpec(
        "hot_rank",
        primary_source="eastmoney",
        tier="L7",
        partition_col="trade_date",
        partition_granularity="month",
        fetch_semantics="snapshot",
        description="东财人气榜",
    ),
    DatasetSpec(
        "sector_bars",
        primary_source="ths",
        tier="L7",
        partition_col="trade_date",
        partition_granularity="month",
        fetch_semantics="snapshot",
        # 同花顺 per-year board-kline files (adapters/ths/boards.sweep_board_bars),
        # not EastMoney: the source was migrated to a single 同花顺 base to end the
        # mixed-source basis breaks, and this label had not followed.
        backfill_source="ths",
        description="板块K线（同花顺）",
    ),
    DatasetSpec(
        "sector_fund_flow",
        primary_source="eastmoney",
        tier="L7",
        partition_col="trade_date",
        partition_granularity="month",
        fetch_semantics="snapshot",
        description="板块资金流向",
    ),
    DatasetSpec(
        "news_headlines",
        primary_source="eastmoney",
        tier="L7",
        partition_col="publish_date",
        partition_granularity="month",
        fetch_semantics="rolling_window",
        history_horizon_days=14,
        description="新闻提要",
    ),
    DatasetSpec(
        "flash_news_wire",
        primary_source="eastmoney",
        tier="L7",
        partition_col="publish_date",
        partition_granularity="month",
        fetch_semantics="rolling_window",
        history_horizon_days=14,
        description="东方财富7×24快讯",
    ),
    DatasetSpec(
        "sentiment_articles",
        primary_source="eastmoney",
        tier="L7",
        partition_col="publish_date",
        partition_granularity="month",
        fetch_semantics="snapshot",
        required=False,
        max_staleness_days=3,
        description="舆情文章（东财个股新闻+情感打分）",
    ),
    # EM datacenter report RPT_ECONOMICCALENDAR was retired (code 9501); keep the
    # schema/registry for a replacement source, but do not fail lake health.
    DatasetSpec(
        "economic_calendar",
        primary_source="eastmoney",
        tier="L7",
        partition_col="event_date",
        partition_granularity="year",
        fetch_semantics="snapshot",
        # The adapter returns a rolling [today-2, today+14] window. Its
        # partition/date column is the event date, so future scheduled events
        # must not advance a freshness watermark beyond the run day.
        watermark=False,
        required=False,
        cadence="skip",
        description="经济日历（端点已下线）",
    ),
    # L8 risk
    DatasetSpec(
        "share_unlock_schedule",
        primary_source="eastmoney",
        tier="L8",
        partition_col="unlock_date",
        partition_granularity="year",
        # The endpoint returns a rolling future window, not historical values
        # for the requested day. A raw max(unlock_date) would therefore push
        # the watermark into the future (the lake once reached 2027-02-04).
        fetch_semantics="snapshot",
        watermark=False,
        backfill_source="eastmoney",
        description="限售解禁计划",
    ),
    DatasetSpec(
        "regulatory_events",
        primary_source="cninfo",
        tier="L8",
        partition_col="event_date",
        partition_granularity="year",
        description="监管事件",
    ),
    # derived — ``layer`` is where the parquet lives, ``tier`` what the data is
    # for, so these carry the tier of the question they answer, not "derived".
    DatasetSpec(
        "adj_factors",
        primary_source="sina",
        tier="L1",
        layer="derived",
        partition_col="trade_date",
        coverage_mode="session_dense",
        description="后复权因子",
    ),
    # Industry returns computed from 申万 membership × hfq bars rather than
    # fetched, so index and constituents cannot disagree. Yearly partitions:
    # ~3 levels × 2 weightings × ~500 industries a day.
    # L5, not L1: the unit of observation is an industry (PK carries
    # industry_code/level/weighting, not symbol), so it belongs beside the
    # membership table that produces it rather than beside the bars.
    DatasetSpec(
        "industry_index",
        primary_source="derived",
        tier="L5",
        layer="derived",
        partition_col="trade_date",
        partition_granularity="year",
        # Computed from daily bars for every available session; a missing
        # session is a derivation hole rather than sparse event-feed behavior.
        coverage_mode="session_dense",
        description="行业指数收益（申万）",
    ),
    # How each recovered delisting's price series ends — see
    # DELISTING_EVENTS_SCHEMA. Merge-style: one row per symbol, a few hundred
    # rows total. date_col (not partition_col) so load(start=/end=) still filters.
    DatasetSpec(
        "delisting_events",
        primary_source="derived",
        tier="L1",
        layer="derived",
        partition_col=None,
        date_col="last_trade_date",
        watermark=False,
        description="退市事件",
    ),
    # ────────────────────────────────────────────────────────────────────
    # External read-only datasets — served by adapter modules, not native
    # CNE ingestion.  They are registered so query, dashboard and catalog treat
    # them as first-class CNE-managed names while preserving source contracts.
    # ────────────────────────────────────────────────────────────────────
    # L0 reference (pg_parquet exports)
    DatasetSpec(
        "instruments_external",
        primary_source="pg_parquet",
        tier="L0",
        layer="external",
        partition_col=None,
        date_col="list_date",
        watermark=False,
        required=False,
        description="标的清单（外部Tushare）",
    ),
    DatasetSpec(
        "trading_calendar_external",
        primary_source="pg_parquet",
        tier="L0",
        layer="external",
        partition_col="trade_date",
        partition_granularity="year",
        watermark=False,
        required=False,
        description="交易日历（外部Tushare）",
    ),
    DatasetSpec(
        "namechange",
        primary_source="tushare",
        tier="L0",
        layer="curated",
        partition_col="start_date",
        partition_granularity="year",
        watermark=False,
        description="曾用名变更",
    ),
    # L1 bars — ETF / fund daily bars from Tushare fund_daily.
    # Native curated datasets: the dedicated etf_bars/fund_bars steps fetch
    # via Tushare fund_daily and write to curated, independent of daily_bars.
    DatasetSpec(
        "etf_bars",
        primary_source="tushare",
        tier="L1",
        layer="curated",
        partition_col="trade_date",
        coverage_mode="session_dense",
        description="ETF日K线",
    ),
    DatasetSpec(
        "etf_list",
        primary_source="local_assets",
        tier="L0",
        layer="external",
        partition_col=None,
        watermark=False,
        required=False,
        description="ETF清单",
    ),
    DatasetSpec(
        "fund_bars",
        primary_source="tushare",
        tier="L1",
        layer="curated",
        partition_col="trade_date",
        coverage_mode="session_dense",
        description="基金/LOF日K线",
    ),
    DatasetSpec(
        "fund_nav",
        primary_source="akshare",
        tier="L1",
        layer="external",
        partition_col="date",
        # EM's open-fund daily table carries the last two trading sessions and
        # publishes through the evening, so a run can legitimately hold
        # yesterday while today is still filling in. PK-level upsert on
        # (code, date) lets the next snapshot overwrite partial cells.
        max_staleness_days=3,
        compactable=True,
        adapter_name="local_assets",
        required=False,
        description="基金净值",
    ),
    DatasetSpec(
        "fund_list",
        primary_source="local_assets",
        tier="L0",
        layer="external",
        partition_col=None,
        watermark=False,
        required=False,
        description="基金清单",
    ),
    # Fund fee reference table (purchase/management/custodian/sales-service
    # rates + redemption rules), one row per fund. Not a time series: the step
    # merges into the adapter's single parquet file directly and gates itself
    # to a weekly cadence via its own state key.
    DatasetSpec(
        "fund_fees",
        primary_source="akshare",
        tier="L0",
        layer="external",
        partition_col=None,
        watermark=False,
        required=False,
        description="基金费率",
    ),
    # AlphaAgent 因子面板（panel_1d.parquet, 1.17GB, 37 列）：日频特征矩阵，
    # 由 external/alphaagent.py 适配器只读映射。注册为 external 层数据集，
    # 使 catalog/dashboard/reader 能与其他数据集统一查询。
    DatasetSpec(
        "alpha_panel_1d",
        primary_source="alphaagent",
        tier="L1",
        layer="external",
        partition_col="date",
        watermark=False,
        required=False,
        description="AlphaAgent因子面板（日频）",
    ),
    # 框架本地指数 CSV（6 大指数：沪深300/中证500/中证1000/创业板指/科创50/上证指数）。
    # 由 external/index_local.py 适配器只读映射，与 native index_bars（TDX 来源）
    # 区分，框架回测基准用这个。
    DatasetSpec(
        "index_bars_external",
        primary_source="local_index",
        tier="L1",
        layer="external",
        partition_col="date",
        # Benchmark panel for 6 major indices, fetched from Tencent kline and
        # compacted into <root>/stock/index.parquet by the local_index adapter.
        max_staleness_days=3,
        compactable=True,
        adapter_name="local_index",
        required=False,
        description="指数日K线（本地基准面板）",
    ),
    # Full Tushare-wide source row used by derived research panels. Canonical
    # daily_bars stays normalized above; this bridge preserves adj_factor,
    # daily_basic, market-cap and ST fields without duplicating the archive.
    DatasetSpec(
        "stock_daily_wide",
        primary_source="tushare_wide",
        tier="L1",
        layer="external",
        partition_col="trade_date",
        partition_granularity="year",
        date_col="trade_date",
        watermark=True,
        required=False,
        description="Tushare日行宽表（含复权因子/市值/ST）",
    ),
    # L2 corporate events (Tushare curated — fetched by steps/tushare_wide.py)
    DatasetSpec(
        "dividend",
        primary_source="tushare",
        tier="L2",
        layer="curated",
        partition_col="end_date",
        partition_granularity="year",
        watermark=False,
        pit=True,
        description="分红送转",
    ),
    # L3 fundamentals (Tushare curated — fetched by steps/tushare_wide.py)
    DatasetSpec(
        "balancesheet",
        primary_source="tushare",
        tier="L3",
        layer="curated",
        partition_col="end_date",
        partition_granularity="year",
        watermark=False,
        pit=True,
        cadence="quarterly",
        description="资产负债表",
    ),
    DatasetSpec(
        "income",
        primary_source="tushare",
        tier="L3",
        layer="curated",
        partition_col="end_date",
        partition_granularity="year",
        watermark=False,
        pit=True,
        cadence="quarterly",
        description="利润表",
    ),
    DatasetSpec(
        "cashflow",
        primary_source="tushare",
        tier="L3",
        layer="curated",
        partition_col="end_date",
        partition_granularity="year",
        watermark=False,
        pit=True,
        cadence="quarterly",
        description="现金流量表",
    ),
    DatasetSpec(
        "fina_indicator",
        primary_source="tushare",
        tier="L3",
        layer="curated",
        partition_col="end_date",
        partition_granularity="year",
        watermark=False,
        pit=True,
        cadence="quarterly",
        description="财务指标",
    ),
    DatasetSpec(
        "forecast",
        primary_source="tushare",
        tier="L3",
        layer="curated",
        partition_col="ann_date",
        partition_granularity="year",
        watermark=False,
        cadence="skip",
        description="业绩预告",
    ),
    DatasetSpec(
        "express",
        primary_source="tushare",
        tier="L3",
        layer="curated",
        partition_col="end_date",
        partition_granularity="year",
        watermark=False,
        cadence="skip",
        description="业绩快报",
    ),
    DatasetSpec(
        "report_rc",
        primary_source="tushare",
        tier="L3",
        layer="curated",
        partition_col="report_date",
        partition_granularity="year",
        watermark=False,
        cadence="quarterly",
        description="业绩报告口径",
    ),
    # L7 sentiment (Tushare curated — fetched by steps/tushare_wide.py)
    DatasetSpec(
        "stk_surv",
        primary_source="tushare",
        tier="L7",
        layer="curated",
        partition_col="surv_date",
        partition_granularity="year",
        watermark=False,
        description="股东大会调查",
    ),
    # L8 risk (Tushare curated — fetched by steps/tushare_wide.py)
    DatasetSpec(
        "share_float_external",
        primary_source="tushare",
        tier="L8",
        layer="curated",
        partition_col="ann_date",
        partition_granularity="year",
        watermark=False,
        description="限售股解禁（Tushare）",
    ),
]

DATASETS: dict[str, DatasetSpec] = {spec.name: spec for spec in _SPECS}


def get_dataset(name: str) -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError:
        raise KeyError(f"unknown dataset {name!r}") from None


def curated_dataset_names() -> frozenset[str]:
    return frozenset(s.name for s in DATASETS.values() if s.layer == "curated")


def derived_dataset_names() -> frozenset[str]:
    return frozenset(s.name for s in DATASETS.values() if s.layer == "derived")


def external_dataset_names() -> frozenset[str]:
    """Names mounted by read adapters rather than produced by a fetch step."""
    return frozenset(s.name for s in DATASETS.values() if s.layer == "external")


def compactable_external_datasets() -> frozenset[str]:
    """External datasets whose staging → compact goes through an adapter."""
    return frozenset(
        s.name for s in DATASETS.values() if s.layer == "external" and s.compactable
    )


def datasets_by_tier() -> dict[Tier, list[str]]:
    """``{tier: [dataset, ...]}`` for every tier, in registry order.

    Every tier is present even when empty, so a consumer grouping by tier
    (catalog docs, the lake dashboard) renders a stable set of sections.
    """
    grouped: dict[Tier, list[str]] = {tier: [] for tier in TIERS}
    for spec in _SPECS:
        grouped[spec.tier].append(spec.name)
    return grouped


def pit_dataset_names() -> frozenset[str]:
    # External wide tables can carry ann_date, but CNE PIT semantics apply to
    # native facts; pass-through contracts are owned by their adapters.
    return frozenset(s.name for s in DATASETS.values() if s.pit and s.layer != "external")


def intraday_dataset_names() -> frozenset[str]:
    return frozenset(s.name for s in DATASETS.values() if s.intraday_frequency)


def intraday_datasets() -> dict[str, str]:
    """``{frequency: dataset}`` for every registered intraday dataset.

    The one place the mapping lives. Steps register from it, the config
    validates against it, and audit iterates it, so adding a frequency is a
    registry entry rather than an edit in four modules.
    """
    return {s.intraday_frequency: s.name for s in DATASETS.values() if s.intraday_frequency}


def fetch_semantics(dataset: str) -> FetchSemantics:
    spec = DATASETS.get(dataset)
    return spec.fetch_semantics if spec else "by_date"


def backfill_reachable_floor(dataset: str, start: date | None, today: date) -> date | None:
    """Earliest date a backfill may honestly request, or None when unbounded.

    ``None`` when the dataset either does not backfill or has no vendor
    horizon. For a ``rolling_window`` dataset the reachable floor is
    ``today - history_horizon_days``: requesting older dates would either
    return nothing or forge a stamp over a live page, and the caller should
    clip its window (and report the unreachable remainder) rather than run it.
    """
    spec = DATASETS.get(dataset)
    if spec is None:
        return None
    if spec.fetch_semantics != "rolling_window":
        return None
    horizon = spec.history_horizon_days or 0
    if horizon <= 0:
        return None
    floor = today - timedelta(days=horizon)
    if start is not None and start > floor:
        return start
    return floor


def history_mode_for(spec: DatasetSpec) -> HistoryMode:
    """Whether the dataset can expose an honest historical series.

    Derived only from registry fields (no parallel flags):
    - ``by_date`` — gap-fill / date-walk
    - ``snapshot_with_backfill`` — daily tip snapshot + dedicated history source
    - ``snapshot_only`` — tip-only; no honest historical replay
    """
    if spec.fetch_semantics == "by_date":
        return "by_date"
    if spec.backfill_source:
        return "snapshot_with_backfill"
    return "snapshot_only"


def history_mode(dataset: str) -> HistoryMode:
    spec = DATASETS.get(dataset)
    if spec is None:
        return "by_date"
    return history_mode_for(spec)


def granularity_for_dataset(dataset: str) -> Granularity:
    spec = DATASETS.get(dataset)
    return spec.partition_granularity if spec else "day"


def is_stale(dataset: str, mark, anchor) -> bool:
    """Whether *dataset*'s freshest date (*mark*) lags *anchor* beyond tolerance.

    *mark* and *anchor* are ``datetime.date`` (or None). A dataset with no mark
    is not judged here (callers treat empty separately).

    A retired source is never stale once the lake has caught up to its last
    published session: there is nothing further to fetch, and calling that
    "stale" forever is how a freshness signal stops being read.
    """
    if mark is None or anchor is None:
        return False
    spec = DATASETS.get(dataset)
    if spec is not None and spec.source_retired_date is not None:
        if mark >= spec.source_retired_date:
            return False
    tolerance = spec.max_staleness_days if spec else 1
    return (anchor - mark).days > tolerance


def should_fetch(dataset: str, watermark: date | None, trade_date: date) -> bool:
    """Whether *dataset* should be fetched on this daily run.

    Returns True when a fetch is needed, False when the cadence says skip.
    When skipping, the step should still advance the watermark so the engine
    records the run.

    Rules:
    - ``daily``: always fetch (every trading session).
    - ``monthly``: fetch only when the watermark is in a different year-month
      than ``trade_date`` (i.e. trade_date crossed into a new month).
    - ``quarterly``: fetch only when the watermark is in a different year-quarter
      than ``trade_date``.
    - ``skip``: never fetch — source is retired or middleware limitation;
      historical data is in the curated layer.
    """
    spec = DATASETS.get(dataset)
    if spec is None:
        return True
    cadence = spec.cadence
    if cadence == "daily":
        return True
    if cadence == "skip":
        return False
    if cadence == "monthly":
        if watermark is None:
            return True
        return (watermark.year, watermark.month) != (trade_date.year, trade_date.month)
    if cadence == "quarterly":
        if watermark is None:
            return True
        return (watermark.year, (watermark.month - 1) // 3) != (
            trade_date.year,
            (trade_date.month - 1) // 3,
        )
    return True


def is_dataset_enabled(dataset: str, config) -> bool:
    """Whether an optional capture is enabled in *config*.

    Optional datasets can outlive the configuration that produced them. A
    disabled tick or minute-bar capture may therefore still have historical
    Parquet on disk, but its old tip is not an ingestion failure and must not
    be reported as stale. Keep this mapping next to the registry so status,
    verify, and the dashboard share the same opt-in semantics.
    """
    if dataset == "trade_ticks":
        return bool(getattr(config, "trade_ticks_enabled", False))
    if dataset == "minute_bars":
        return bool(
            getattr(config, "minute_bars_enabled", False)
            and "1m" in getattr(config, "minute_bars_frequencies", ())
        )
    if dataset == "minute_bars_5m":
        return bool(
            getattr(config, "minute_bars_enabled", False)
            and "5m" in getattr(config, "minute_bars_frequencies", ())
        )
    return True


# ---------------------------------------------------------------------------
# Derived legacy tables (kept so existing imports stay valid; do not edit these
# directly — edit the DatasetSpec entries above).
# ---------------------------------------------------------------------------

# partition column per dataset with a partition key; None = merge-style.
# Includes both curated and compactable-external datasets — both go through
# the staging → compact path and need watermark management.
PARTITION_COLS: dict[str, str | None] = {
    s.name: s.partition_col
    for s in DATASETS.values()
    if (s.layer == "curated" or s.compactable) and s.partition_col is not None
}

FETCH_SEMANTICS: dict[str, FetchSemantics] = {
    s.name: s.fetch_semantics for s in DATASETS.values() if s.fetch_semantics != "by_date"
}

# Datasets partitioned by non-date keys — skip date-based watermarks.
WATERMARK_SKIP = frozenset(
    s.name
    for s in DATASETS.values()
    if (s.layer == "curated" or s.compactable)
    and s.partition_col is not None
    and not s.watermark
)

# Warn when a partition's row/symbol count falls below this fraction of the prior partition.
ROW_COUNT_MUTATION_MIN_RATIO = 0.5

# Ignore mutation checks when the baseline partition is smaller than this.
ROW_COUNT_MUTATION_MIN_BASELINE_ROWS = 50
