"""EastMoney index constituents and weights."""

from __future__ import annotations

import logging
from datetime import date

import polars as pl

from cnequity.adapters.eastmoney.common import exchange_from_datacenter, symbol_from_em
from cnequity.adapters.eastmoney.datacenter import fetch_datacenter
from cnequity.adapters.eastmoney.em_auth import EastMoneyClient
from cnequity.domain.symbols import format_symbol

DEFAULT_INDICES = [
    "000001.SH",
    "000300.SH",
    "000688.SH",
    "399001.SZ",
    "399006.SZ",
]

_INDEX_CODE_MAP = {
    "000001": "000001.SH",
    "000300": "000300.SH",
    "000688": "000688.SH",
    "399001": "399001.SZ",
    "399006": "399006.SZ",
}

_REPORT = "RPT_INDEX_CONSTITUENT"
_COLUMNS = "INDEX_CODE,SECURITY_CODE,TRADE_DATE"

logger = logging.getLogger(__name__)


def _index_symbol(index_code: str) -> str:
    code = str(index_code).zfill(6)
    return _INDEX_CODE_MAP.get(code, format_symbol(code, "SH" if code.startswith("0") else "SZ"))


def fetch_index_constituents(
    as_of_date: date,
    *,
    indices: list[str] | None = None,
    client: EastMoneyClient | None = None,
    config=None,
) -> pl.DataFrame:
    """Fetch the latest available index constituent snapshot.

    The ``RPT_INDEX_CONSTITUENT`` endpoint returns *all* historical rebalance
    records for an index, not a date-filtered snapshot.  The API ignores
    ``TRADE_DATE`` in the filter expression.  To get the *current* constituents
    we fetch all rows, find the most recent ``TRADE_DATE`` among them, and keep
    only rows matching that latest date.
    """
    owns = client is None
    if client is None:
        client = EastMoneyClient(config=config)

    # ``None`` means use the default universe; an explicit empty list is a
    # deliberate no-op and must not turn into a full-index snapshot.
    target_indices = DEFAULT_INDICES if indices is None else indices
    rows: list[dict] = []
    missing_indices: list[str] = []
    try:
        for index_sym in target_indices:
            index_code = index_sym.split(".")[0]
            raw = fetch_datacenter(
                client,
                _REPORT,
                _COLUMNS,
                filter_expr=f'(INDEX_CODE="{index_code}")',
                page_size=5000,
            )
            # Group by TRADE_DATE to find the latest snapshot
            by_date: dict[str, list[dict]] = {}
            for item in raw:
                returned_code = str(item.get("INDEX_CODE") or "").zfill(6)
                if returned_code != index_code.zfill(6):
                    continue
                returned_date = str(item.get("TRADE_DATE") or "")[:10]
                if not returned_date:
                    continue
                by_date.setdefault(returned_date, []).append(item)

            if not by_date:
                missing_indices.append(index_sym)
                continue

            # Use the most recent rebalance date available
            latest_date = max(by_date.keys())
            latest_batch = by_date[latest_date]
            if latest_date != as_of_date.isoformat():
                logger.info(
                    "EastMoney index constituents: using latest snapshot %s for %s "
                    "(requested %s)",
                    latest_date,
                    index_sym,
                    as_of_date.isoformat(),
                )

            matched = 0
            for item in latest_batch:
                code = str(item.get("SECURITY_CODE", "")).zfill(6)
                exch = exchange_from_datacenter(item)
                sym = symbol_from_em(code, 1 if exch == "SH" else (2 if exch == "BJ" else 0))
                if not sym:
                    continue
                matched += 1
                rows.append(
                    {
                        "index_symbol": _index_symbol(index_code),
                        "symbol": sym,
                        "as_of_date": as_of_date,
                        "weight": 0.0,
                    }
                )
            if matched == 0:
                missing_indices.append(index_sym)
            else:
                logger.info(
                    "EastMoney index constituents: %s → %d constituents (snapshot %s)",
                    index_sym,
                    matched,
                    latest_date,
                )
    finally:
        if owns:
            client.close()

    if missing_indices:
        raise RuntimeError(
            "EastMoney index constituents returned no matching rows for: "
            + ", ".join(missing_indices)
        )
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).unique(subset=["index_symbol", "symbol", "as_of_date"], keep="last")
