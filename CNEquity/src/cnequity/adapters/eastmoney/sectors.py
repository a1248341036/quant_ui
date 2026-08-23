"""EastMoney sector / concept board membership."""

from __future__ import annotations

from datetime import date

import polars as pl

from cnequity.adapters.eastmoney.common import exchange_from_datacenter, symbol_from_em
from cnequity.adapters.eastmoney.datacenter import fetch_datacenter
from cnequity.adapters.eastmoney.em_auth import EastMoneyClient

_BOARD_REPORT = "RPT_BOARD_CONSTITUENT"
_BOARD_COLUMNS = "SECURITY_CODE,BOARD_CODE,BOARD_NAME,BOARD_TYPE_NEW"
# BOARD_TYPE_NEW: 1=地域, 2=行业, 3=概念/主题, 4=指数成分与风格标签.
# Industry (2) is included because rotation joins board bars to their members by
# name, and the bars now come from 同花顺 whose universe is 90 行业 + 361 概念 —
# without type 2 every industry board would carry a signal but no tradable names.
_CONCEPT_BOARD_TYPES = {"2", "3", "4"}


def fetch_sector_members(
    as_of_date: date, *, client: EastMoneyClient | None = None, config=None
) -> pl.DataFrame:
    owns = client is None
    if client is None:
        client = EastMoneyClient(config=config)
    try:
        raw = fetch_datacenter(
            client,
            _BOARD_REPORT,
            _BOARD_COLUMNS,
            # Every board type, ~92k rows: clamped to 500 that is 185 pages, past
            # the pageNumber cap, and the step failed every capital run. This report
            # honors a full 5000-row page (measured 2026-08-11) — 19 pages, with
            # room for the board list to keep growing.
            page_size=5000,
            trust_page_size=True,
        )
        rows = []
        for item in raw:
            if str(item.get("BOARD_TYPE_NEW") or "") not in _CONCEPT_BOARD_TYPES:
                continue
            code = str(item.get("SECURITY_CODE", "")).zfill(6)
            exch = exchange_from_datacenter(item)
            sym = symbol_from_em(code, 1 if exch == "SH" else (2 if exch == "BJ" else 0))
            if not sym:
                continue
            sector_code = str(item.get("BOARD_CODE") or "").strip()
            sector_name = str(item.get("BOARD_NAME") or "").strip()
            if not sector_code or not sector_name:
                continue
            rows.append(
                {
                    "symbol": sym,
                    "sector_code": sector_code,
                    "sector_name": sector_name,
                    "as_of_date": as_of_date,
                }
            )
    finally:
        if owns:
            client.close()
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).unique(subset=["symbol", "sector_code", "as_of_date"], keep="last")
