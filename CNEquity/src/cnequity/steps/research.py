"""L3/L4/L7 research steps: institutional holdings, analyst consensus, sentiment."""

from __future__ import annotations

import hashlib
import logging
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import polars as pl

from cnequity.adapters.eastmoney.consensus import fetch_analyst_consensus
from cnequity.adapters.eastmoney.institutional import fetch_institutional_holdings
from cnequity.config import Config
from cnequity.derive.sentiment_scores import compute_sentiment_scores
from cnequity.domain.symbols import format_symbol, infer_exchange_from_code
from cnequity.orchestrator.registry import register_step
from cnequity.steps.common import load_symbols
from cnequity.steps.http_common import empty_ok, run_incremental_fetched, write_fetched

logger = logging.getLogger(__name__)

_MIN_INSTITUTIONAL_HOLDING_ROWS_PER_PERIOD = 100

# How many EM stock news headlines to fetch per symbol per day.
_NEWS_PER_SYMBOL = 20
# Cap symbols per run to keep the step within a reasonable time budget.
# The full A-share universe is ~5500; fetching 20 headlines each is ~110k
# HTTP requests which takes hours. For daily sentiment we sample the top
# symbols by liquidity — the universe_csv from sentiment-mvp has ~230.
_MAX_SYMBOLS_PER_RUN = 300


def _quarter_labels(config: Config, trade_date: date) -> set[str]:
    from cnequity.adapters.eastmoney.institutional import _quarter_end_dates

    periods = _quarter_end_dates(
        trade_date,
        start=getattr(config, "_backfill_start", None),
        end=getattr(config, "_backfill_end", None),
    )
    return {f"{period[:4]}Q{(int(period[5:7]) - 1) // 3 + 1}" for period in periods}


def _validate_institutional_holdings_snapshot(df):
    """Reject a non-empty but obviously truncated quarterly holdings response."""
    if df.is_empty():
        return df
    required = {"symbol", "holder_type", "report_period"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(
            "institutional_holdings: response is missing required column(s): " + ", ".join(missing)
        )
    counts = (
        df.unique(subset=["symbol", "holder_type", "report_period"])
        .group_by("report_period")
        .agg(pl.len().alias("_holding_rows"))
        .filter(pl.col("_holding_rows") < _MIN_INSTITUTIONAL_HOLDING_ROWS_PER_PERIOD)
    )
    if not counts.is_empty():
        details = ", ".join(
            f"{row['report_period']}={row['_holding_rows']}" for row in counts.iter_rows(named=True)
        )
        raise RuntimeError(
            "institutional_holdings: incomplete quarterly snapshot; each observed "
            f"period needs at least {_MIN_INSTITUTIONAL_HOLDING_ROWS_PER_PERIOD} "
            f"unique holding row(s) ({details})"
        )
    return df


@register_step("institutional_holdings", group="research", depends_on=["instruments"])
def step_institutional_holdings(
    config: Config, trade_date: date, run_id: str, context: dict
) -> dict:
    if not config.sources.get("eastmoney", True):
        raise RuntimeError("institutional_holdings: eastmoney source disabled in config")
    # Quarterly by REPORT_DATE: daily refreshes the latest quarter, backfill
    # walks all quarters from 2016.
    backfill = getattr(config, "_backfill", False)
    df = _validate_institutional_holdings_snapshot(
        fetch_institutional_holdings(trade_date, backfill=backfill, config=config)
    )
    missing_periods: set[str] = set()
    if backfill:
        expected = _quarter_labels(config, trade_date)
        observed = (
            set(df.get_column("report_period").drop_nulls().to_list())
            if not df.is_empty() and "report_period" in df.columns
            else set()
        )
        missing_periods = expected - observed
    if backfill and not missing_periods and df.is_empty():
        return {"rows_read": 0, "rows_written": 0}
    if backfill and missing_periods:
        result: dict
        if df.is_empty():
            result = {"rows_read": 0, "rows_written": 0}
        else:
            result = write_fetched(config, run_id, "institutional_holdings", df, source="eastmoney")
        result["status"] = "warning"
        result["missing_periods"] = len(missing_periods)
        result["context_updates"] = {
            "audit_findings": [
                {
                    "dataset": "institutional_holdings",
                    "severity": "warning",
                    "check": "backfill_missing_quarters",
                    "message": (
                        f"institutional holdings missing {len(missing_periods)} requested "
                        f"quarter(s): {', '.join(sorted(missing_periods)[:8])}"
                    ),
                    "missing_periods": sorted(missing_periods),
                }
            ]
        }
        return result
    empty_ok(df, "institutional_holdings", trade_date)
    return write_fetched(config, run_id, "institutional_holdings", df, source="eastmoney")


@register_step("analyst_consensus", group="research", depends_on=["instruments"])
def step_analyst_consensus(config: Config, trade_date: date, run_id: str, context: dict) -> dict:
    if not config.sources.get("eastmoney", True):
        raise RuntimeError("analyst_consensus: eastmoney source disabled in config")
    # Live consensus snapshot stamped with trade_date (no dated EM report).
    # Use the common helper so snapshot backfill is rejected and missed daily
    # snapshots remain visible as coverage findings instead of looking complete.
    return run_incremental_fetched(
        config,
        trade_date,
        run_id,
        "analyst_consensus",
        lambda d: fetch_analyst_consensus(d, config=config),
        source="eastmoney",
        date_col="forecast_date",
    )


@register_step(
    "sentiment_scores",
    group="research",
    depends_on=["announcement_index", "news_headlines", "hot_rank"],
)
def step_sentiment_scores(config: Config, trade_date: date, run_id: str, context: dict) -> dict:
    return run_incremental_fetched(
        config,
        trade_date,
        run_id,
        "sentiment_scores",
        lambda d: compute_sentiment_scores(config, d),
        source="derived",
        allow_empty=True,
    )


def _article_id(code: str, publish_time: str, title: str) -> str:
    """Stable SHA1 hash matching sentiment-mvp store.key_of convention."""
    raw = f"{code}|{publish_time}|{title}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


# Path to sentiment-mvp project (sibling of CNEquity) — code only (lib/, config.yaml).
# Data (articles.db, universe.csv, etc.) has been moved to quant_ui/data/sentiment-mvp.
_MVP_ROOT = Path(__file__).resolve().parents[4] / "sentiment-mvp"
_MVP_DATA = Path(__file__).resolve().parents[4] / "data" / "sentiment-mvp"


def _fetch_sentiment_articles(trade_date: date, *, config: Config) -> pl.DataFrame:
    """Fetch EM per-symbol stock news for trade_date and score sentiment.

    Delegates to sentiment-mvp's lib/fetch_news.py (search API) and
    lib/sentiment.py (lexicon scoring) so the CNE step reuses the exact
    same collection and scoring pipeline as the MVP project.

    Returns a DataFrame matching SENTIMENT_ARTICLES_SCHEMA (minus provenance
    columns, which are added by write_fetched).
    """
    import sys

    if str(_MVP_ROOT) not in sys.path:
        sys.path.insert(0, str(_MVP_ROOT))

    import httpx  # noqa: PLC0415
    from lib import fetch_news as fn  # noqa: PLC0415
    from lib import sentiment as st  # noqa: PLC0415

    # Read the MVP universe CSV (code,name,amount_rank) — already curated.
    universe_csv = _MVP_DATA / "universe.csv"
    if universe_csv.exists():
        uni_df = pd.read_csv(universe_csv, dtype={"code": str})
        codes = uni_df["code"].astype(str).str.zfill(6).tolist()
    else:
        # Fallback: use CNE instruments
        symbols = load_symbols(config)
        codes = [
            s.split(".")[0] if "." in s else s for s in symbols
        ]

    if not codes:
        logger.warning("sentiment_articles: no symbols available; returning empty")
        return pl.DataFrame()

    # Limit the universe to keep the step within a reasonable time budget.
    if len(codes) > _MAX_SYMBOLS_PER_RUN:
        logger.info(
            "sentiment_articles: truncating universe from %d to %d codes",
            len(codes),
            _MAX_SYMBOLS_PER_RUN,
        )
        codes = codes[:_MAX_SYMBOLS_PER_RUN]

    target_str = trade_date.isoformat()
    all_arts: list[dict] = []
    errors = 0

    session = httpx.Client()
    try:
        for i, code in enumerate(codes):
            try:
                batch = fn.fetch_stock_news(
                    session, code, max_pages=_NEWS_PER_SYMBOL // 10 + 1, delay=0.05
                )
            except Exception as exc:
                errors += 1
                if errors <= 5:
                    logger.warning(
                        "sentiment_articles: fetch_stock_news(%s) failed: %s",
                        code,
                        exc,
                    )
                continue

            if not batch:
                continue

            # Filter to trade_date
            for a in batch:
                pub = (a.get("publish_time") or "")[:10]
                if pub != target_str:
                    continue
                all_arts.append(a)

            if (i + 1) % 100 == 0:
                logger.info(
                    "sentiment_articles: progress %d/%d codes, %d articles so far",
                    i + 1,
                    len(codes),
                    len(all_arts),
                )
    finally:
        session.close()

    if errors > 5:
        logger.warning(
            "sentiment_articles: %d/%d codes failed fetch", errors, len(codes)
        )

    if not all_arts:
        return pl.DataFrame()

    # Score with MVP's lexicon sentiment
    scored = st.score_articles(all_arts, method="lexicon")

    rows: list[dict] = []
    now_iso = datetime.now(timezone.utc).isoformat()
    for a in scored:
        title = a.get("title", "")
        if not title:
            continue
        pub_time = a.get("publish_time", "")
        code = a.get("code", "")
        rows.append(
            {
                "article_id": _article_id(code, pub_time, title),
                "symbol": format_symbol(code, infer_exchange_from_code(code)),
                "publish_date": trade_date,
                "publish_time": pub_time,
                "media": a.get("media", ""),
                "title": title,
                "summary": (a.get("content") or "")[:500],
                "url": a.get("url", ""),
                "source": a.get("source", "em"),
                "label": a.get("label", "neutral"),
                "score": a.get("score", 0.0),
                "pos_hits": a.get("pos_hits", 0),
                "neg_hits": a.get("neg_hits", 0),
                "fetched_at": now_iso,
            }
        )

    if not rows:
        return pl.DataFrame()

    df = pl.DataFrame(rows)
    df = df.unique(subset=["article_id"], keep="last")
    logger.info(
        "sentiment_articles: %d articles from %d codes for %s",
        df.height,
        len(codes),
        trade_date.isoformat(),
    )
    return df


@register_step(
    "sentiment_articles",
    group="research",
    depends_on=["instruments"],
    parallelizable=False,
    description="Fetch EM per-symbol stock news with sentiment scoring, write to sentiment_articles.",
)
def step_sentiment_articles(
    config: Config, trade_date: date, run_id: str, context: dict
) -> dict:
    if not config.sources.get("eastmoney", True):
        raise RuntimeError("sentiment_articles: eastmoney source disabled in config")

    # snapshot dataset: fetch only trade_date
    df = _fetch_sentiment_articles(trade_date, config=config)
    if df.is_empty():
        return {
            "rows_read": 0,
            "rows_written": 0,
            "status": "warning",
            "error": f"no articles returned for {trade_date.isoformat()}",
        }
    return write_fetched(config, run_id, "sentiment_articles", df, source="eastmoney")
