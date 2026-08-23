"""Primary daily-bar sources used by the existing CNE ``daily_bars`` step."""

from __future__ import annotations

import logging
from datetime import date

import polars as pl

from cnequity.config import Config
from cnequity.domain.schemas import data_version_for, with_provenance
from cnequity.external import tushare_fetch

logger = logging.getLogger(__name__)


def _empty() -> pl.DataFrame:
    return pl.DataFrame(schema={
        "symbol": pl.Utf8,
        "trade_date": pl.Date,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Int64,
        "amount": pl.Float64,
    })


def _from_tushare(frame: pl.DataFrame, symbols: set[str]) -> pl.DataFrame:
    if frame.is_empty():
        return _empty()
    frame = frame.filter(pl.col("ts_code").is_in(list(symbols)))
    if frame.is_empty():
        return _empty()
    return frame.select([
        pl.col("ts_code").alias("symbol"),
        pl.col("trade_date").cast(pl.Date),
        pl.col("open").cast(pl.Float64, strict=False),
        pl.col("high").cast(pl.Float64, strict=False),
        pl.col("low").cast(pl.Float64, strict=False),
        pl.col("close").cast(pl.Float64, strict=False),
        (pl.col("vol").cast(pl.Float64, strict=False).fill_null(0) * 100)
        .round().cast(pl.Int64).alias("volume"),
        (pl.col("amount").cast(pl.Float64, strict=False).fill_null(0) * 1000)
        .alias("amount"),
    ]).filter(
        (pl.col("open") > 0)
        & (pl.col("high") > 0)
        & (pl.col("low") > 0)
        & (pl.col("close") > 0)
    )


def _code(symbol: str) -> str:
    return str(symbol).split(".", 1)[0].zfill(6)


def _exchange_symbol(code: str) -> str:
    """Map a 6-digit code to the exchange-suffixed symbol used by the lake."""
    if code.startswith(("5", "6", "9")):
        return f"{code}.SH"
    if code.startswith(("4", "8")):
        return f"{code}.BJ"
    return f"{code}.SZ"


def _from_akshare(frame, symbol: str, start: date, end: date) -> pl.DataFrame:
    if frame is None or frame.empty:
        return _empty()
    frame = frame.rename(columns={
        "日期": "trade_date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
    })
    required = {"trade_date", "open", "high", "low", "close", "volume", "amount"}
    if not required.issubset(frame.columns):
        raise ValueError(f"AkShare daily response missing columns: {sorted(required - set(frame.columns))}")
    out = pl.from_pandas(frame[list(required)]).with_columns(
        pl.lit(_exchange_symbol(_code(symbol))).alias("symbol"),
        pl.col("trade_date").cast(pl.Date),
        pl.col("open").cast(pl.Float64, strict=False),
        pl.col("high").cast(pl.Float64, strict=False),
        pl.col("low").cast(pl.Float64, strict=False),
        pl.col("close").cast(pl.Float64, strict=False),
        pl.col("volume").cast(pl.Float64, strict=False).fill_null(0).round().cast(pl.Int64),
        pl.col("amount").cast(pl.Float64, strict=False).fill_null(0),
    ).select(["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount"])
    return out.filter(
        (pl.col("trade_date") >= start)
        & (pl.col("trade_date") <= end)
        & (pl.col("open") > 0)
        & (pl.col("high") > 0)
        & (pl.col("low") > 0)
        & (pl.col("close") > 0)
    )


def fetch_tushare(config: Config, symbols: list[str], start: date, end: date) -> pl.DataFrame:
    """Fetch the requested window from Tushare's date-level APIs."""
    wanted = {str(s) for s in symbols}
    frames: list[pl.DataFrame] = []
    current = start
    while current <= end:
        frame = tushare_fetch.fetch_one_trade_date(config, current)
        if not frame.is_empty():
            frames.append(_from_tushare(frame, wanted))
        current = date.fromordinal(current.toordinal() + 1)
    nonempty = [f for f in frames if not f.is_empty()]
    out = pl.concat(nonempty, how="vertical") if nonempty else _empty()
    if out.is_empty():
        raise RuntimeError(f"Tushare returned no daily bars for {start}..{end}")
    requested = set(symbols)
    observed = set(out.get_column("symbol").unique().to_list())
    missing = requested - observed
    if missing:
        raise RuntimeError(
            f"Tushare returned incomplete daily bars: {len(missing)} symbol(s) missing"
        )
    return with_provenance(
        out, source="tushare", data_version=data_version_for("daily_bars")
    )


def fetch_akshare(
    symbols: list[str],
    start: date,
    end: date,
    *,
    max_workers: int = 8,
) -> pl.DataFrame:
    """Fetch the requested window from AkShare.

    Uses a thread pool to parallelise per-symbol requests.  Individual symbol
    failures are tolerated — the stock may be suspended or delisted — and
    surfaced via logging.  An empty result for the *entire* batch raises so
    that ``fetch_primary`` can report the combined failure.
    """
    import akshare as ak
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _fetch_one(symbol: str) -> pl.DataFrame | None:
        try:
            frame = ak.stock_zh_a_hist(
                symbol=_code(symbol),
                period="daily",
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
                adjust="",
            )
            return _from_akshare(frame, symbol, start, end)
        except Exception as exc:  # noqa: BLE001 — one symbol must not fail the batch
            logger.warning("AkShare fetch failed for %s: %s", symbol, exc)
            return None

    frames: list[pl.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one, s): s for s in symbols}
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None and not result.is_empty():
                frames.append(result)

    if not frames:
        raise RuntimeError(f"AkShare returned no daily bars for {start}..{end}")
    return with_provenance(
        pl.concat(frames, how="vertical"),
        source="akshare",
        data_version=data_version_for("daily_bars"),
    )


def fetch_primary(
    config: Config,
    symbols: list[str],
    start: date,
    end: date,
) -> tuple[pl.DataFrame, dict]:
    """Tushare primary, AkShare fallback; raise when both are unavailable.

    Returns ``(df, source_status)`` where *source_status* is a dict suitable
    for ``StateStore.set_fields("daily_bars", source_status=...)``::

        {
            "provider": "tushare" | "akshare",
            "tushare_available": bool,
            "fallback_used": bool,
            "tushare_error": str | None,     # present when Tushare failed
            "akshare_error": str | None,     # present when both failed (in the exception)
            "rows": int,                      # rows delivered
            "symbols_requested": int,
            "symbols_delivered": int,
        }
    """
    n_requested = len(symbols)

    try:
        df = fetch_tushare(config, symbols, start, end)
        return df, {
            "provider": "tushare",
            "tushare_available": True,
            "fallback_used": False,
            "tushare_error": None,
            "rows": df.height,
            "symbols_requested": n_requested,
            "symbols_delivered": df["symbol"].n_unique(),
        }
    except Exception as exc:  # noqa: BLE001
        tushare_error = f"{exc.__class__.__name__}: {exc}"
        logger.warning("Tushare daily_bars unavailable: %s", tushare_error)
        try:
            df = fetch_akshare(symbols, start, end)
            return df, {
                "provider": "akshare",
                "tushare_available": False,
                "fallback_used": True,
                "tushare_error": tushare_error,
                "rows": df.height,
                "symbols_requested": n_requested,
                "symbols_delivered": df["symbol"].n_unique(),
            }
        except Exception as fallback_exc:  # noqa: BLE001
            raise RuntimeError(
                f"Tushare unavailable ({tushare_error}); "
                f"AkShare unavailable ({fallback_exc.__class__.__name__}: {fallback_exc})"
            ) from fallback_exc
