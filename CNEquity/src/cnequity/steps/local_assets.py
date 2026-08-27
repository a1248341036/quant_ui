"""Steps for datasets whose files live under the local-assets root.

``fund_nav`` and ``index_bars_external`` follow the staging → compact path:
each step stages a snapshot batch and ``compact`` merges it into the
adapter-owned file (fund_nav.parquet / index.parquet) through that adapter's
write protocol.

``fund_fees`` is a per-fund reference table, not a time series, so it does not
fit the yearly/hive compact layout; its step merges into the target file
atomically itself and gates its own cadence (weekly) via a state key.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import polars as pl
import requests

from cnequity.config import Config
from cnequity.external.index_local import ADAPTER as _index_local_adapter
from cnequity.external.local_assets import ADAPTER as _local_assets_adapter
from cnequity.orchestrator.manifest import Manifest
from cnequity.orchestrator.registry import register_step
from cnequity.storage.atomic import write_parquet_atomic
from cnequity.storage.state import StateStore

logger = logging.getLogger(__name__)

_FEE_WEEKLY_CADENCE_DAYS = 7
_FEE_DETAIL_WORKERS = 4
_UNIT_NAV_RE = re.compile(r"\d{4}-\d{2}-\d{2}-单位净值")
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_DAY_RE = re.compile(r"(\d+)")
_OPS_INDICATOR = "运作费用"
_REDEEM_INDICATOR = "赎回费率"

# Benchmark indices maintained here (mirrors quant_ui core.fetcher.INDEX_SYMBOLS).
INDEX_SYMBOLS = {
    "sh000300": "沪深300",
    "sh000905": "中证500",
    "sh000852": "中证1000",
    "sz399006": "创业板指",
    "sh000688": "科创50",
    "sh000001": "上证指数",
}
_TX_KLINE_URL = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
_TX_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def _read_fund_csv(path: Path) -> pl.DataFrame:
    """fund.csv is written by pandas on Windows and carries '\\r' in header
    cells (\\r\\r\\n line endings); normalize header whitespace before use."""
    df = pl.read_csv(path, schema_overrides={"code": pl.Utf8})
    return df.rename({c: c.strip() for c in df.columns})


def _pool_codes(config: Config) -> list[str] | None:
    """Equity OTC fund universe maintained by refresh_data (fund.csv)."""
    root = getattr(config, "external_local_assets_root", None)
    if root is None:
        return None
    path = Path(root) / "fund" / "fund.csv"
    if not path.exists():
        return None
    df = _read_fund_csv(path)
    return (
        df.get_column("code").cast(pl.Utf8).str.zfill(6).unique().sort().to_list()
    )


def _fetch_open_fund_daily() -> pd.DataFrame:
    import akshare as ak

    return ak.fund_open_fund_daily_em()


def _parse_unit_nav_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Wide EM table ('2026-08-25-单位净值' columns) → long date/code/nav."""
    code_col = next((c for c in raw.columns if "代码" in str(c)), None)
    frames: list[pd.DataFrame] = []
    if code_col is not None:
        for col in raw.columns:
            if not _UNIT_NAV_RE.fullmatch(str(col)):
                continue
            day = pd.Timestamp(str(col)[:10])
            part = raw[[code_col, col]].copy()
            part.columns = ["code", "nav"]
            part["code"] = part["code"].astype(str).str.zfill(6)
            part["nav"] = pd.to_numeric(part["nav"], errors="coerce")
            part["date"] = day
            frames.append(part.dropna(subset=["nav"])[["date", "code", "nav"]])
    if not frames:
        return pd.DataFrame(columns=["date", "code", "nav"])
    return pd.concat(frames, ignore_index=True)


@register_step("fund_nav", group="capital", description="EM open-fund NAV snapshot → staging → compact into fund_nav.parquet.")
def step_fund_nav(
    config: Config,
    trade_date: date,
    run_id: str,
    context: dict,
) -> dict:
    if not _local_assets_adapter.enabled(config, "fund_nav"):
        return {
            "status": "warning",
            "dataset": "fund_nav",
            "error": "external_local_assets bridge is disabled",
        }
    if getattr(config, "_backfill", False):
        # The EM daily table only serves the last two sessions; history must
        # come from the legacy per-fund full-history fetcher (quant_ui side).
        return {
            "status": "warning",
            "dataset": "fund_nav",
            "error": "fund_nav backfill unsupported: EM serves only the last two sessions",
        }

    try:
        raw = _fetch_open_fund_daily()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"fund_nav: EM open-fund daily snapshot failed: {exc}") from exc

    snap = pl.from_pandas(_parse_unit_nav_columns(raw))
    if snap.is_empty():
        raise RuntimeError("fund_nav: EM snapshot returned no unit-NAV columns")
    # Lock dtypes to the target file's contract (date/code/nav) so the
    # compact-time anti-join never sees cat-vs-str or ns-datetime drift.
    snap = snap.with_columns(
        pl.col("date").cast(pl.Date),
        pl.col("code").cast(pl.Utf8),
        pl.col("nav").cast(pl.Float64),
    ).select("date", "code", "nav")

    pool = _pool_codes(config)
    if pool:
        snap = snap.filter(pl.col("code").is_in(pool))
    else:
        logger.warning("fund_nav: fund.csv universe unavailable; ingesting all snapshot codes")

    watermark = StateStore(config.meta_root).get_date("fund_nav")
    if watermark is not None:
        snap = snap.filter(pl.col("date") > watermark)

    total = snap.height
    if total == 0:
        logger.info("fund_nav: snapshot adds nothing beyond watermark; nothing staged")
        return {"dataset": "fund_nav", "rows_read": 0, "rows_written": 0}

    manifest = Manifest(config.manifest_path)
    # polars returns datetime64 max values; batch ids become file names, so
    # normalize to plain dates before formatting (Windows forbids ':').
    def _as_date(value) -> date:
        return value.date() if hasattr(value, "date") else value

    min_day = _as_date(snap.get_column("date").min())
    max_day = _as_date(snap.get_column("date").max())
    batch_id = f"snap-{max_day.isoformat()}"
    manifest.start_batch(
        run_id, batch_id, task_id="fund_nav", dataset="fund_nav",
        window_start=min_day.isoformat(),
        window_end=max_day.isoformat(),
    )
    try:
        from cnequity.storage import StagingWriter

        StagingWriter(config.staging_root).write_batch(
            "fund_nav", run_id, batch_id, snap
        )
    except Exception as exc:  # noqa: BLE001
        manifest.finish_batch(run_id, batch_id, "failed", error_message=str(exc))
        raise
    manifest.finish_batch(
        run_id, batch_id, "success", rows_read=total, rows_written=total
    )
    logger.info(
        "fund_nav: staged %d row(s) for %s..%s",
        total,
        min_day,
        max_day,
    )
    return {"dataset": "fund_nav", "rows_read": total, "rows_written": total}


# ── fund_fees ────────────────────────────────────────────────────────────


def _rate(value: object) -> float | None:
    match = _PERCENT_RE.search(str(value or "").strip())
    if match:
        return float(match.group(1)) / 100
    return None


def _column(df: pd.DataFrame, fragment: str) -> str | None:
    return next((str(c) for c in df.columns if fragment in str(c)), None)


def _parse_operations(df: pd.DataFrame) -> dict[str, float | None]:
    out = {
        "management_fee_rate": None,
        "custodian_fee_rate": None,
        "sales_service_fee_rate": None,
    }
    if df is None or df.empty:
        return out
    values = [str(v) for v in df.iloc[0].tolist()]
    for i, value in enumerate(values[:-1]):
        for label, key in (
            ("管理费率", "management_fee_rate"),
            ("托管费率", "custodian_fee_rate"),
            ("销售服务费率", "sales_service_fee_rate"),
        ):
            if label in value:
                out[key] = _rate(values[i + 1])
    return out


def _parse_redemption(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return ""
    period_col = _column(df, "适用期限") or str(df.columns[0])
    rate_col = _column(df, "赎回费率") or str(df.columns[-1])
    rules: list[dict] = []
    for _, row in df.iterrows():
        period = str(row.get(period_col, "")).strip()
        rate = _rate(row.get(rate_col, ""))
        if not period or rate is None:
            continue
        numbers = [int(x) for x in _DAY_RE.findall(period)]
        rule: dict = {"period": period, "rate": rate}
        if "小于" in period and numbers:
            rule["max_days_exclusive"] = numbers[-1]
        if "大于等于" in period and numbers:
            rule["min_days"] = numbers[0]
        rules.append(rule)
    return json.dumps(rules, ensure_ascii=False, separators=(",", ":"))


def _bulk_purchase_rates(raw: pd.DataFrame) -> dict[str, float | None]:
    code_col = _column(raw, "基金代码")
    fee_col = _column(raw, "手续费")
    if not code_col or not fee_col:
        return {}
    out: dict[str, float | None] = {}
    for _, row in raw.iterrows():
        code = str(row[code_col]).zfill(6)
        text = str(row[fee_col]).strip()
        match = _PERCENT_RE.search(text)
        if match:
            out[code] = float(match.group(1)) / 100
        else:
            try:
                out[code] = float(text) / 100
            except (TypeError, ValueError):
                out[code] = None
    return out


_FEE_RATE_COLUMNS = (
    "purchase_fee_rate",
    "management_fee_rate",
    "custodian_fee_rate",
    "sales_service_fee_rate",
)


def _coerce_fee_rates(df: pl.DataFrame) -> pl.DataFrame:
    """Legacy CSV stores rates as text like '0.002'; force them numeric."""
    casts = [
        pl.col(c).cast(pl.Utf8).str.strip_chars().cast(pl.Float64, strict=False).alias(c)
        for c in _FEE_RATE_COLUMNS
        if c in df.columns
    ]
    return df.with_columns(casts) if casts else df


def _fee_target(config: Config) -> Path:
    root = getattr(config, "external_local_assets_root", None)
    if root is None:
        raise RuntimeError("external_local_assets_root not configured")
    return Path(root) / "fund" / "fund_fee.parquet"


def _existing_fees(target: Path) -> pl.DataFrame:
    if target.exists():
        return _coerce_fee_rates(pl.read_parquet(target))
    # One-time bootstrap: adopt the legacy quant_ui CSV so only funds that
    # never parsed cleanly need a detail-page fetch.
    legacy = target.with_suffix(".csv")
    if legacy.exists():
        df = _read_fund_csv(legacy)
        return _coerce_fee_rates(
            df.with_columns(pl.col("code").cast(pl.Utf8).str.zfill(6))
        )
    return pl.DataFrame()


@register_step("fund_fees", group="signals", description="Weekly fund fee reference refresh, merged atomically into fund_fee.parquet.")
def step_fund_fees(
    config: Config,
    trade_date: date,
    run_id: str,
    context: dict,
) -> dict:
    if not _local_assets_adapter.enabled(config, "fund_fees"):
        return {
            "status": "warning",
            "dataset": "fund_fees",
            "error": "external_local_assets bridge is disabled",
        }

    state = StateStore(config.meta_root)
    # Key must be a valid Windows file name (StateStore persists <key>.json).
    last_run = state.get_date("fund_fees_last_run")
    if last_run is not None and (trade_date - last_run).days < _FEE_WEEKLY_CADENCE_DAYS:
        logger.info(
            "fund_fees: cadence skip (last run %s, weekly cadence)", last_run
        )
        return {"dataset": "fund_fees", "rows_read": 0, "rows_written": 0}

    pool = _pool_codes(config)
    if not pool:
        return {
            "status": "warning",
            "dataset": "fund_fees",
            "error": "fund.csv universe unavailable",
        }
    pool_set = set(pool)

    existing = _existing_fees(_fee_target(config))
    ok_codes: set[str] = set()
    if not existing.is_empty() and "fee_status" in existing.columns:
        ok_codes = set(
            existing.filter(pl.col("fee_status") == "ok")
            .get_column("code")
            .cast(pl.Utf8)
            .to_list()
        )

    raw = _fetch_open_fund_daily()
    purchase = _bulk_purchase_rates(raw)

    todo = [c for c in pool if c not in ok_codes]
    logger.info(
        "fund_fees: %d fund(s), %d detail page(s) to fetch", len(pool), len(todo)
    )

    def fetch_one(code: str) -> dict:
        import akshare as ak

        row: dict = {"code": code, "fee_status": "error", "last_error": ""}
        errors: list[str] = []
        try:
            row.update(_parse_operations(ak.fund_fee_em(code, indicator=_OPS_INDICATOR)))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"operations: {exc}")
        try:
            row["redemption_fee_rule"] = _parse_redemption(
                ak.fund_fee_em(code, indicator=_REDEEM_INDICATOR)
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"redemption: {exc}")
        row["fee_status"] = (
            "ok"
            if any(
                row.get(k) is not None
                for k in (
                    "management_fee_rate",
                    "custodian_fee_rate",
                    "sales_service_fee_rate",
                    "redemption_fee_rule",
                )
            )
            else "error"
        )
        row["last_error"] = ("; ".join(errors) or "fee page returned no usable fields")[:300]
        return row

    # Accumulate plain dicts and build ONE frame at the end: per-row
    # DataFrames turn all-None columns into polars String, and a later
    # diagonal_relaxed concat promotes String over Float64, stringifying the
    # whole rate column.
    detail_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=_FEE_DETAIL_WORKERS) as pool_ex:
        futures = {pool_ex.submit(fetch_one, code): code for code in todo}
        done = 0
        for fut in as_completed(futures):
            detail_rows.append(fut.result())
            done += 1
            if done % 200 == 0 or done == len(todo):
                logger.info("fund_fees: %d/%d detail pages", done, len(todo))

    base = pd.DataFrame({"code": pool})
    uni_path = Path(getattr(config, "external_local_assets_root")) / "fund" / "fund.csv"
    if uni_path.exists():
        uni = _read_fund_csv(uni_path).select(
            pl.col("code").cast(pl.Utf8).str.zfill(6).alias("code"),
            "name",
            "type",
        )
        base = base.merge(uni.to_pandas(), on="code", how="left")
    base["purchase_fee_rate"] = base["code"].map(purchase)

    keep_cols = [
        c
        for c in existing.columns
        if c not in {"name", "type", "purchase_fee_rate"} and c != "code"
    ]
    if not existing.is_empty() and keep_cols:
        old = existing.to_pandas()
        old["code"] = old["code"].astype(str).str.zfill(6)
        old = old.drop_duplicates("code")
        base = base.merge(old[["code", *keep_cols]], on="code", how="left")

    if detail_rows:
        details = _coerce_fee_rates(
            pl.from_pandas(pd.DataFrame(detail_rows))
        ).to_pandas()
        base = base.merge(details, on="code", how="left", suffixes=("", "_new"))
        # Overlapping columns land as '<col>_new' (right side); fresh detail
        # values win over stale ones, NaN keeps whatever the base had.
        for col in (
            "management_fee_rate",
            "custodian_fee_rate",
            "sales_service_fee_rate",
            "redemption_fee_rule",
            "fee_status",
            "last_error",
        ):
            new_col = f"{col}_new"
            if new_col in base.columns:
                fallback = base[col] if col in base.columns else None
                base[col] = base[new_col].fillna(fallback)
                base = base.drop(columns=[new_col])
        if "fee_updated_at" not in base.columns:
            base["fee_updated_at"] = pd.NA
        now = pd.Timestamp.now().isoformat(timespec="seconds")
        base.loc[base["fee_status"] == "ok", "fee_updated_at"] = now

    fee_df = _coerce_fee_rates(pl.from_pandas(base))
    residual = [
        c
        for c in _FEE_RATE_COLUMNS
        if c in fee_df.columns and fee_df.get_column(c).dtype == pl.Utf8
    ]
    if residual:
        logger.error("fund_fees: rate columns still textual after coercion: %s", residual)
    target = _fee_target(config)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_parquet_atomic(target, fee_df, compression="zstd")

    state.set_date("fund_fees_last_run", trade_date)
    n_ok = int((fee_df.get_column("fee_status") == "ok").sum())
    logger.info(
        "fund_fees: wrote %s rows=%d ok=%d (%d detail pages fetched)",
        target, fee_df.height, n_ok, len(todo),
    )
    return {
        "dataset": "fund_fees",
        "rows_read": len(todo),
        "rows_written": fee_df.height,
        "detail_pages_fetched": len(todo),
        "context_updates": {},
    }


# ── index_bars_external ──────────────────────────────────────────────────


def _fetch_index_kline(symbol: str, name: str, start: str, end: str) -> pl.DataFrame:
    """Tencent daily kline for one benchmark index (qfq, OHLC not needed beyond open/close)."""
    try:
        resp = requests.get(
            _TX_KLINE_URL,
            params={
                "_var": "kline_dayqfq",
                "param": f"{symbol},day,{start},{end},640,qfq",
            },
            headers=_TX_HEADERS,
            timeout=15,
        )
        body = resp.text[resp.text.find("=") + 1 :]
        data = json.loads(body)
        d = data["data"].get(symbol, {})
        key = next((k for k in ("day", "qfqday", "hfqday") if k in d), None)
        if key is None:
            return pl.DataFrame()
        rows = [
            {
                "date": r[0],
                "open": r[1],
                "close": r[2],
                "code": symbol,
                "name": name,
            }
            for r in d[key]
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("index_bars_external: kline fetch failed for %s: %s", symbol, exc)
        return pl.DataFrame()
    if not rows:
        return pl.DataFrame()
    return (
        pl.from_pandas(pd.DataFrame(rows))
        .with_columns(
            pl.col("date").str.to_date().cast(pl.Date),
            pl.col("code").cast(pl.Utf8),
            pl.col("name").cast(pl.Utf8),
            pl.col("open").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
        )
        .drop_nulls(subset=["date", "close"])
        .filter((pl.col("date") >= pl.lit(start).str.to_date()) & (pl.col("date") <= pl.lit(end).str.to_date()))
    )


@register_step(
    "index_bars_external",
    group="capital",
    description="Tencent kline snapshot for the 6 benchmark indices → staging → compact into index.parquet.",
)
def step_index_bars_external(
    config: Config,
    trade_date: date,
    run_id: str,
    context: dict,
) -> dict:
    if not _index_local_adapter.enabled(config, "index_bars_external"):
        return {
            "status": "warning",
            "dataset": "index_bars_external",
            "error": "external_local_assets bridge is disabled",
        }
    if getattr(config, "_backfill", False):
        return {
            "status": "warning",
            "dataset": "index_bars_external",
            "error": "backfill unsupported here; use the legacy quant_ui refresh or extend the window via --start",
        }

    state = StateStore(config.meta_root)
    watermark = state.get_date("index_bars_external")
    start = (watermark + timedelta(days=1)) if watermark else trade_date - timedelta(days=30)
    end = trade_date
    if start > end:
        start = end

    frames = []
    failed = []
    for symbol, name in INDEX_SYMBOLS.items():
        df = _fetch_index_kline(symbol, name, start.isoformat(), end.isoformat())
        if df.is_empty():
            failed.append(symbol)
        else:
            frames.append(df)

    total = sum(f.height for f in frames)
    snap = (
        pl.concat(frames, how="vertical_relaxed").unique(subset=["code", "date"], keep="last")
        if frames
        else pl.DataFrame()
    )
    manifest = Manifest(config.manifest_path)
    batch_id = f"tx-{trade_date.isoformat()}"
    manifest.start_batch(
        run_id, batch_id, task_id="index_bars_external", dataset="index_bars_external",
        window_start=start.isoformat(), window_end=end.isoformat(),
    )
    if total == 0:
        # Every source empty on a trading day usually means the day session has
        # not closed yet — record a clean success so compact does not stall.
        manifest.finish_batch(run_id, batch_id, "success")
        logger.info("index_bars_external: no rows for %s..%s", start, end)
        return {"dataset": "index_bars_external", "rows_read": 0, "rows_written": 0}

    try:
        from cnequity.storage import StagingWriter

        StagingWriter(config.staging_root).write_batch(
            "index_bars_external", run_id, batch_id, snap.select("date", "code", "name", "open", "close")
        )
    except Exception as exc:  # noqa: BLE001
        manifest.finish_batch(run_id, batch_id, "failed", error_message=str(exc))
        raise
    manifest.finish_batch(
        run_id, batch_id, "success", rows_read=total, rows_written=snap.height
    )
    result = {
        "dataset": "index_bars_external",
        "rows_read": total,
        "rows_written": snap.height,
    }
    if failed:
        result["status"] = "warning"
        result["error"] = f"kline fetch failed for: {failed}"
    logger.info(
        "index_bars_external: staged %d row(s) (%d symbols) for %s..%s",
        snap.height, len(frames), start, end,
    )
    return result
